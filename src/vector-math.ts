// =============================================================================
// vector-math.ts — pure, dependency-free vector + index serialization helpers.
//
// This module imports NOTHING from obsidian or @huggingface/transformers, so it
// can be exercised directly by a plain-`node` benchmark (see bench/at-rest.mjs)
// via Node's TypeScript type-stripping. KEEP IT STRIP-TYPES-SAFE: no enums, no
// namespaces, no parameter properties, no decorators — only types + plain
// runtime code, so `node --experimental-strip-types` (or Node >= 23.6 unflagged)
// can run it without a build step.
//
// It owns:
//   - cosineSimilarity / meanOf / dotRow  (the hot-loop math)
//   - quantizeChunksRaw / dequantizeChunksRaw (int8 <-> fp32, byte-for-byte the
//     same quantization the live path used at load time, so ranking is identical)
//   - base64FromInt8 / int8FromBase64 (ONLY used now by the bench's old-format
//     reproduction — no longer on the live persistence path)
//   - serializeIndex / deserializeIndex (the binary index format) + its types
// =============================================================================

// =============================================================================
// hot-loop vector math
// =============================================================================

// Cosine similarity for two already-L2-normalized vectors == their dot product.
// Returns 0 on a length mismatch (e.g. a stale vector from a different model).
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// Mean of a set of unit vectors, re-L2-normalized. Operates over a contiguous
// buffer of `count` rows of `dims`.
export function meanOf(
  buffer: Float32Array,
  count: number,
  dims: number,
): Float32Array {
  const mean = new Float32Array(dims);
  for (let c = 0; c < count; c++) {
    const off = c * dims;
    for (let i = 0; i < dims; i++) mean[i] += buffer[off + i];
  }
  let sumSq = 0;
  for (let i = 0; i < dims; i++) {
    mean[i] /= count;
    sumSq += mean[i] * mean[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < dims; i++) mean[i] *= inv;
  }
  return mean;
}

// Dot of chunk row `r` (offset r*dims) of a contiguous chunk buffer with a vector.
// Both operands are L2-normalized, so the result is their cosine similarity.
export function dotRow(
  buf: Float32Array,
  r: number,
  vec: Float32Array,
  dims: number,
): number {
  const off = r * dims;
  let dot = 0;
  for (let d = 0; d < dims; d++) dot += buf[off + d] * vec[d];
  return dot;
}

// =============================================================================
// Mean-centering (anisotropy correction)
// =============================================================================
// Sentence-embedding spaces are anisotropic: every note shares a common direction,
// so even unrelated notes score a high baseline cosine (measured ~0.49 "noise floor"
// on a real vault). Subtracting the corpus CENTROID (the shared direction) from each
// vector before comparing pushes unrelated notes toward 0/negative while keeping
// genuinely-related notes high — so similarity reflects TOPIC, not the shared
// baseline. (Measured live: Bio-vs-security 0.52 -> -0.06, the real match 0.86 -> 0.79.)
// Simple common-mean removal is the SAFE form of isotropy correction — on an already-
// isotropic corpus the centroid norm is tiny, so it's a near-no-op, never harmful.

// The corpus centroid: the L2-normalized mean of all note mean-vectors. Returns null
// for an empty corpus. O(n*dims), cheap to recompute as the vault changes.
export function computeCentroid(
  means: Float32Array[],
  dims: number,
): Float32Array | null {
  if (means.length === 0) return null;
  const c = new Float32Array(dims);
  let used = 0;
  for (const m of means) {
    if (m.length !== dims) continue;
    for (let i = 0; i < dims; i++) c[i] += m[i];
    used++;
  }
  if (used === 0) return null;
  let sumSq = 0;
  for (let i = 0; i < dims; i++) {
    c[i] /= used;
    sumSq += c[i] * c[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return null;
  const inv = 1 / norm;
  for (let i = 0; i < dims; i++) c[i] *= inv;
  return c;
}

// Remove the centroid direction from a unit vector and re-L2-normalize, so the dot of
// two centered vectors is the centered cosine. Returns a new array.
export function centerVector(
  v: Float32Array,
  centroid: Float32Array,
  dims: number,
): Float32Array {
  let proj = 0;
  for (let i = 0; i < dims; i++) proj += v[i] * centroid[i];
  const out = new Float32Array(dims);
  let sumSq = 0;
  for (let i = 0; i < dims; i++) {
    const x = v[i] - proj * centroid[i];
    out[i] = x;
    sumSq += x * x;
  }
  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < dims; i++) out[i] *= inv;
  }
  return out;
}

// In-place centering of every row of a contiguous chunk buffer (count rows of dims):
// each row gets the centroid direction removed and is re-L2-normalized. Mutates buf.
export function centerChunksInPlace(
  buf: Float32Array,
  count: number,
  dims: number,
  centroid: Float32Array,
): void {
  for (let r = 0; r < count; r++) {
    const off = r * dims;
    let proj = 0;
    for (let i = 0; i < dims; i++) proj += buf[off + i] * centroid[i];
    let sumSq = 0;
    for (let i = 0; i < dims; i++) {
      const x = buf[off + i] - proj * centroid[i];
      buf[off + i] = x;
      sumSq += x * x;
    }
    const norm = Math.sqrt(sumSq);
    if (norm > 0) {
      const inv = 1 / norm;
      for (let i = 0; i < dims; i++) buf[off + i] *= inv;
    }
  }
}

// =============================================================================
// int8 quantization (chunk block only — the mean stays fp32)
// =============================================================================

// Quantize a contiguous fp32 chunk buffer (chunkCount rows of `dims`) into a
// symmetric per-row int8 block: one Int8Array of length chunkCount*dims plus one
// fp32 scale per row (max|v|/127). This is the EXACT loop the old quantizeChunks
// used, minus the base64 step — the int8 buffer is returned raw so it can live in
// RAM and be written straight into the binary blob.
export function quantizeChunksRaw(
  chunks: Float32Array,
  chunkCount: number,
  dims: number,
): { q: Int8Array; scales: number[] } {
  const q = new Int8Array(chunkCount * dims);
  const scales = new Array<number>(chunkCount);
  for (let c = 0; c < chunkCount; c++) {
    const off = c * dims;
    let maxAbs = 0;
    for (let i = 0; i < dims; i++) {
      const a = Math.abs(chunks[off + i]);
      if (a > maxAbs) maxAbs = a;
    }
    const scale = maxAbs > 0 ? maxAbs / 127 : 1;
    scales[c] = scale;
    for (let i = 0; i < dims; i++) {
      let v = Math.round(chunks[off + i] / scale);
      if (v > 127) v = 127;
      else if (v < -127) v = -127;
      q[off + i] = v;
    }
  }
  return { q, scales };
}

// Dequantize an int8 chunk block back to a contiguous fp32 buffer and RE-NORMALIZE
// each row (quantization perturbs the L2 norm by ~1-4%; cosineSimilarity assumes
// unit vectors). This is byte-for-byte the body of the old dequantizeChunks
// (including the per-row L2 renorm after v = q*scale), so BiMax scores produced
// from a lazily-dequantized buffer are bit-identical to the old load-time-dequant
// path. Returns null when the payload length is wrong.
export function dequantizeChunksRaw(
  q: Int8Array,
  scales: number[],
  chunkCount: number,
  dims: number,
): Float32Array | null {
  if (q.length !== chunkCount * dims) return null;
  if (scales.length !== chunkCount) return null;
  const out = new Float32Array(chunkCount * dims);
  for (let c = 0; c < chunkCount; c++) {
    const off = c * dims;
    const scale = scales[c];
    let sumSq = 0;
    for (let i = 0; i < dims; i++) {
      const v = q[off + i] * scale;
      out[off + i] = v;
      sumSq += v * v;
    }
    const norm = Math.sqrt(sumSq);
    if (norm > 0) {
      const inv = 1 / norm;
      for (let i = 0; i < dims; i++) out[off + i] *= inv;
    }
  }
  return out;
}

// =============================================================================
// base64 (Int8) — only used now by the bench's old-format (v3) reproduction
// =============================================================================

export function base64FromInt8(arr: Int8Array): string {
  // Reinterpret the signed bytes as unsigned for btoa, chunked to avoid call-stack
  // limits on very large blocks.
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = "";
  const STEP = 0x8000;
  for (let i = 0; i < bytes.length; i += STEP) {
    const slice = bytes.subarray(i, Math.min(i + STEP, bytes.length));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

export function int8FromBase64(b64: string): Int8Array {
  const binary = atob(b64);
  const out = new Int8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    // charCodeAt is 0..255; reinterpret as signed int8.
    out[i] = (binary.charCodeAt(i) << 24) >> 24;
  }
  return out;
}

// =============================================================================
// binary index format
// =============================================================================

// One persisted entry's METADATA. The vectors themselves live in the binary blob
// at the recorded byte offsets; `scales` stay in the JSON (~chunkCount fp32 per
// note, tiny) so a chunk row can be dequantized lazily without a second blob read.
export interface StoredEntryMeta {
  path: string;
  mtime: number;
  chunkCount: number;
  scales: number[]; // one fp32 scale per chunk row
  summaryLabel?: string;
  chunkTexts?: string[];
  meanOffset: number; // byte offset of the fp32 mean vector (always %4 === 0)
  chunkOffset: number; // byte offset of the int8 chunk buffer
}

// The small JSON manifest written alongside the binary blob. `version`/`modelId`/
// `dims` gate a full rebuild; `quantized`/`hasChunkText` make a future change of
// quantization or text-persistence policy detectable and self-invalidating.
// `totalBytes` lets load() validate the blob length matches the manifest.
export interface StoredIndexHeader {
  version: number;
  modelId: string;
  dims: number;
  quantized: boolean;
  hasChunkText: boolean;
  totalBytes: number;
  entries: StoredEntryMeta[];
}

// The minimal shape serializeIndex needs from an in-memory entry: the fp32 mean
// vector plus the already-quantized int8 chunk buffer + per-row scales. The live
// IndexEntry is a structural superset of this, so it is accepted directly.
export interface SerializableEntry {
  path: string;
  mtime: number;
  dims: number;
  chunkCount: number;
  meanVector: Float32Array;
  chunkBytes: Int8Array;
  scales: number[];
  chunkTexts?: string[];
  summaryLabel?: string;
}

// A deserialized in-memory entry. The vectors are VIEWS into the one shared blob
// ArrayBuffer (no copy, no dequant) — the int8 chunks at their at-rest footprint,
// the mean as a 4-aligned Float32Array. The caller (IndexStore) dequantizes chunk
// rows lazily through its LRU cache.
export interface DeserializedEntry {
  path: string;
  mtime: number;
  dims: number;
  chunkCount: number;
  meanVector: Float32Array;
  chunkBytes: Int8Array;
  scales: number[];
  chunkTexts?: string[];
  summaryLabel?: string;
}

export interface SerializeMeta {
  modelId: string;
  dims: number;
  hasChunkText: boolean;
  version: number;
}

// Serialize entries into { json manifest, binary blob }. PURE: no IO. The blob
// holds every fp32 mean vector FIRST, then every int8 chunk buffer, concatenated;
// the manifest records each region's byte offset. Means-first guarantees every
// meanOffset is a multiple of 4 (fp32 alignment for the Float32Array view on
// deserialize); int8 has no alignment constraint, so the chunk region can follow
// the (4-aligned, dims*4-sized) means region freely.
export function serializeIndex(
  entries: SerializableEntry[],
  meta: SerializeMeta,
): { json: string; blob: ArrayBuffer } {
  const dims = meta.dims;

  // Size pass: means region first, then chunks region.
  let total = 0;
  for (const e of entries) total += e.dims * 4; // mean fp32
  for (const e of entries) total += e.chunkCount * e.dims; // int8 chunks

  const blob = new ArrayBuffer(total);
  const bytes = new Uint8Array(blob);
  let cursor = 0;

  const metaEntries = new Array<StoredEntryMeta>(entries.length);

  // Means region (every meanOffset lands on a multiple of 4: total so far is a
  // running sum of dims*4 terms, hence always %4 === 0).
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const meanOffset = cursor;
    const mv = e.meanVector;
    // CRITICAL: copy only this view's OWN bytes (byteOffset/byteLength), never the
    // whole — possibly pooled/larger — backing buffer, which would write garbage.
    bytes.set(new Uint8Array(mv.buffer, mv.byteOffset, mv.byteLength), cursor);
    cursor += e.dims * 4;
    metaEntries[i] = {
      path: e.path,
      mtime: e.mtime,
      chunkCount: e.chunkCount,
      scales: e.scales,
      meanOffset,
      chunkOffset: 0, // filled in the chunks pass below
      ...(meta.hasChunkText && e.chunkTexts ? { chunkTexts: e.chunkTexts } : {}),
      ...(meta.hasChunkText && e.summaryLabel
        ? { summaryLabel: e.summaryLabel }
        : {}),
    };
  }

  // Chunks region.
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    metaEntries[i].chunkOffset = cursor;
    const cb = e.chunkBytes;
    bytes.set(new Uint8Array(cb.buffer, cb.byteOffset, cb.byteLength), cursor);
    cursor += e.chunkCount * e.dims;
  }

  const header: StoredIndexHeader = {
    version: meta.version,
    modelId: meta.modelId,
    dims,
    quantized: true,
    hasChunkText: meta.hasChunkText,
    totalBytes: total,
    entries: metaEntries,
  };

  return { json: JSON.stringify(header), blob };
}

// Build ONLY the JSON manifest (header + per-entry metadata, with the SAME byte
// offsets + totalBytes serializeIndex would produce) WITHOUT allocating the blob.
// Used by the lazy-label drainer's manifest-only persist: the vectors did not move,
// so the on-disk blob is unchanged and only summaryLabel fields differ — this lets
// us rewrite just index.json without a multi-MB blob copy. The offset math mirrors
// serializeIndex exactly (means region first, then chunks region).
export function serializeManifest(
  entries: SerializableEntry[],
  meta: SerializeMeta,
): string {
  let total = 0;
  for (const e of entries) total += e.dims * 4;
  for (const e of entries) total += e.chunkCount * e.dims;

  const metaEntries = new Array<StoredEntryMeta>(entries.length);
  let cursor = 0;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const meanOffset = cursor;
    cursor += e.dims * 4;
    metaEntries[i] = {
      path: e.path,
      mtime: e.mtime,
      chunkCount: e.chunkCount,
      scales: e.scales,
      meanOffset,
      chunkOffset: 0,
      ...(meta.hasChunkText && e.chunkTexts ? { chunkTexts: e.chunkTexts } : {}),
      ...(meta.hasChunkText && e.summaryLabel
        ? { summaryLabel: e.summaryLabel }
        : {}),
    };
  }
  for (let i = 0; i < entries.length; i++) {
    metaEntries[i].chunkOffset = cursor;
    cursor += entries[i].chunkCount * entries[i].dims;
  }

  const header: StoredIndexHeader = {
    version: meta.version,
    modelId: meta.modelId,
    dims: meta.dims,
    quantized: true,
    hasChunkText: meta.hasChunkText,
    totalBytes: total,
    entries: metaEntries,
  };
  return JSON.stringify(header);
}

// Deserialize a (manifest JSON, blob ArrayBuffer) pair into in-memory entries.
// PURE: no dequant, no IO. Each entry's meanVector is a Float32Array VIEW into the
// blob (4-aligned by the means-first layout) and chunkBytes is an Int8Array VIEW —
// so the entries collectively hold the blob alive at its int8 at-rest footprint.
// Per-entry dims / mean-length guards drop a corrupt row rather than throwing.
// Returns the parsed header and the surviving entries.
export function deserializeIndex(
  json: string,
  blob: ArrayBuffer,
): { header: StoredIndexHeader; entries: DeserializedEntry[] } {
  const header = JSON.parse(json) as StoredIndexHeader;
  const dims = header.dims;
  const out: DeserializedEntry[] = [];
  for (const meta of header.entries) {
    const meanLen = dims;
    const chunkLen = meta.chunkCount * dims;
    // Bounds guards: a truncated/garbled blob must not let a TypedArray view run
    // off the end (the constructor throws on out-of-range offsets/lengths).
    if (meta.meanOffset + meanLen * 4 > blob.byteLength) continue;
    if (meta.chunkOffset + chunkLen > blob.byteLength) continue;
    if (meta.scales.length !== meta.chunkCount) continue;
    const meanVector = new Float32Array(blob, meta.meanOffset, meanLen);
    const chunkBytes = new Int8Array(blob, meta.chunkOffset, chunkLen);
    out.push({
      path: meta.path,
      mtime: meta.mtime,
      dims,
      chunkCount: meta.chunkCount,
      meanVector,
      chunkBytes,
      scales: meta.scales,
      chunkTexts: meta.chunkTexts,
      summaryLabel: meta.summaryLabel,
    });
  }
  return { header, entries: out };
}
