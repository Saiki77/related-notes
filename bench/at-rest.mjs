// =============================================================================
// bench/at-rest.mjs — at-rest + ranking micro-benchmark for the NEW perf changes.
//
// Synthesizes 1000 notes x ~17 chunks x 384 dims and exercises the NEW
// serialize/deserialize + quantize/dequantize + an LRU-dequant rank simulation,
// comparing them against the OLD base64-in-JSON (v3) format they replace. Prints:
//   1. index file size (new binary vs old base64-JSON)
//   2. cold-load time (lazy int8 deserialize vs eager base64-decode + full dequant)
//   3. peak RAM (int8-in-RAM entries vs eager fp32 entries)
//   4. Stage-1 + Stage-2 rank latency at 1000 notes, with a bit-identical assert
//      that the lazy-LRU Stage-2 score equals the eager-fp32 Stage-2 score.
//
// RUNNING: this imports ../src/vector-math.ts (a pure, dependency-free TS module
// with NO obsidian / NO transformers imports) via Node's TypeScript type-stripping.
//   node --experimental-strip-types bench/at-rest.mjs        (Node 23.x)
//   node bench/at-rest.mjs                                    (Node >= 23.6, unflagged)
//   node --expose-gc bench/at-rest.mjs                        (cleaner RAM peaks)
// .mjs is always ESM, so this works regardless of package.json "type". vector-math
// must stay strip-types-safe (no enums/namespaces/param-properties/decorators).
// =============================================================================

import { performance } from "node:perf_hooks";
import {
  meanOf,
  dotRow,
  cosineSimilarity,
  quantizeChunksRaw,
  dequantizeChunksRaw,
  serializeIndex,
  serializeManifest,
  deserializeIndex,
  base64FromInt8,
  int8FromBase64,
} from "../src/vector-math.ts";

// --- synthesis params --------------------------------------------------------
const N_NOTES = 1000;
const DIMS = 384;
const MEAN_CHUNKS = 17;
const TOPK = 12;
const SHORTLIST = 60;
const SHORTLIST_WIDTH = Math.max(TOPK * 4, SHORTLIST); // 60
const CACHE_CAP = 180; // 3x shortlist width
const TITLE_CHUNK_INDEX = 0;
const TITLE_WEIGHT = 2;
const N_SWITCHES = 20;
const MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

// --- seeded PRNG (mulberry32) ------------------------------------------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(0x9e3779b9);

// --- helpers -----------------------------------------------------------------
function l2normalizeRow(buf, off, dims) {
  let sumSq = 0;
  for (let i = 0; i < dims; i++) sumSq += buf[off + i] * buf[off + i];
  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < dims; i++) buf[off + i] *= inv;
  }
}

// chunkCount randomized 1..33 around ~MEAN_CHUNKS (chunk[0] is the title chunk).
function randomChunkCount() {
  // Triangular-ish around the mean, clamped to [1, 2*MEAN-1].
  const span = MEAN_CHUNKS - 1; // 16
  const c = 1 + Math.round((rand() + rand()) * span); // 1..33, peaked near 17
  return Math.max(1, Math.min(2 * MEAN_CHUNKS - 1, c));
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function fmtMs(ms) {
  return `${ms.toFixed(2)} ms`;
}

function maybeGc() {
  if (typeof global.gc === "function") global.gc();
}

// =============================================================================
// 1. Synthesize the int8 IndexEntry set (this IS the steady-state at-rest shape).
// =============================================================================
console.log("Synthesizing", N_NOTES, "notes x ~" + MEAN_CHUNKS, "chunks x", DIMS, "dims…\n");

// Each int8 entry: { path, mtime, dims, chunkCount, meanVector(fp32), chunkBytes(int8), scales }
const int8Entries = [];
for (let n = 0; n < N_NOTES; n++) {
  const chunkCount = randomChunkCount();
  const buffer = new Float32Array(chunkCount * DIMS);
  for (let c = 0; c < chunkCount; c++) {
    const off = c * DIMS;
    for (let i = 0; i < DIMS; i++) buffer[off + i] = rand() * 2 - 1;
    l2normalizeRow(buffer, off, DIMS);
  }
  const meanVector = meanOf(buffer, chunkCount, DIMS);
  const { q, scales } = quantizeChunksRaw(buffer, chunkCount, DIMS);
  int8Entries.push({
    path: `notes/note-${n}.md`,
    mtime: 1700000000000 + n,
    dims: DIMS,
    chunkCount,
    meanVector,
    chunkBytes: q,
    scales,
    chunkTexts: undefined, // not exercising text persistence here
    summaryLabel: undefined,
  });
}

// =============================================================================
// 2. INDEX FILE SIZE — new binary vs old base64-JSON (v3 reproduction).
// =============================================================================
const { json: newJson, blob: newBlob } = serializeIndex(int8Entries, {
  modelId: MODEL_ID,
  dims: DIMS,
  hasChunkText: false,
  version: 4,
});
const newJsonBytes = Buffer.byteLength(newJson, "utf8");
const newBlobBytes = newBlob.byteLength;
const newTotal = newJsonBytes + newBlobBytes;

// serializeManifest (the manifest-only path used by lazy-label persist) must yield
// a header byte-identical to serializeIndex's JSON when no labels differ.
const manifestOnly = serializeManifest(int8Entries, {
  modelId: MODEL_ID,
  dims: DIMS,
  hasChunkText: false,
  version: 4,
});
const manifestMatches = manifestOnly === newJson;

// OLD v3: per note { meanVector: number[], chunks: { scales, q: base64 }, ... } all
// in one JSON.stringify. This is the exact shape the binary format replaces.
const oldStored = {
  version: 3,
  modelId: MODEL_ID,
  dims: DIMS,
  quantized: true,
  hasChunkText: false,
  entries: int8Entries.map((e) => ({
    path: e.path,
    mtime: e.mtime,
    dims: e.dims,
    chunkCount: e.chunkCount,
    meanVector: Array.from(e.meanVector),
    chunks: { scales: e.scales, q: base64FromInt8(e.chunkBytes) },
  })),
};
const oldJson = JSON.stringify(oldStored);
const oldBytes = Buffer.byteLength(oldJson, "utf8");

// =============================================================================
// 3. COLD-LOAD TIME — lazy int8 deserialize vs eager v3 (base64 + full dequant).
// =============================================================================
function coldLoadNew() {
  // Parse manifest + build lazy int8 views; NO per-chunk dequant.
  const { entries } = deserializeIndex(newJson, newBlob);
  return entries.length;
}

function coldLoadOld() {
  const data = JSON.parse(oldJson);
  let kept = 0;
  for (const e of data.entries) {
    const q = int8FromBase64(e.chunks.q);
    const f = dequantizeChunksRaw(q, e.chunks.scales, e.chunkCount, e.dims);
    if (f) kept++; // hold nothing — just measure the eager dequant cost
  }
  return kept;
}

const loadNewSamples = [];
const loadOldSamples = [];
for (let i = 0; i < 5; i++) {
  let t = performance.now();
  coldLoadNew();
  loadNewSamples.push(performance.now() - t);
  t = performance.now();
  coldLoadOld();
  loadOldSamples.push(performance.now() - t);
}
const loadNew = median(loadNewSamples);
const loadOld = median(loadOldSamples);

// =============================================================================
// 4. PEAK RAM — int8-in-RAM (lazy) entries vs eager fp32 entries.
// =============================================================================
// Lazy: deserialize from the blob, hold the int8 entries (views into the blob).
maybeGc();
const ramBaseLazy = process.memoryUsage();
const lazyHeld = deserializeIndex(newJson, newBlob).entries;
maybeGc();
const ramLazy = process.memoryUsage();

// Eager: dequantize every note's chunks to fp32 and hold them (the OLD steady state).
const eagerHeld = [];
maybeGc();
const ramBaseEager = process.memoryUsage();
for (const e of int8Entries) {
  eagerHeld.push({
    path: e.path,
    meanVector: e.meanVector,
    chunks: dequantizeChunksRaw(e.chunkBytes, e.scales, e.chunkCount, e.dims),
  });
}
maybeGc();
const ramEager = process.memoryUsage();

const lazyRss = ramLazy.rss - ramBaseLazy.rss;
const eagerRss = ramEager.rss - ramBaseEager.rss;
const lazyHeap = ramLazy.heapUsed - ramBaseLazy.heapUsed;
const eagerHeap = ramEager.heapUsed - ramBaseEager.heapUsed;

// Without --expose-gc the rss/heapUsed deltas are dominated by GC timing (rss often
// doesn't move at all), so any ratio computed from them is noise. Gate the ratio on
// gc being exposed; the chunk-data theoretical row below is gc-independent and is the
// load-bearing RAM number either way.
const gcExposed = typeof global.gc === "function";
function ramRatio(eager, lazy) {
  if (!gcExposed) return "n/a (run with --expose-gc)";
  if (lazy <= 0 || eager <= 0) return "n/a";
  return `${(eager / lazy).toFixed(2)}x`;
}

// Theoretical chunk-data footprint (the dominant term): fp32 vs int8.
let totalChunkRows = 0;
for (const e of int8Entries) totalChunkRows += e.chunkCount;
const fp32ChunkBytes = totalChunkRows * DIMS * 4;
const int8ChunkBytes = totalChunkRows * DIMS * 1;

// Keep refs alive so GC can't reclaim before the print (and quiet "unused" lint).
void lazyHeld.length;
void eagerHeld.length;

// =============================================================================
// 5. RANK LATENCY at 1000 notes — Stage 1 + Stage 2 (LRU) + bit-identical assert.
// =============================================================================
// Port of the live DequantCache LRU.
class DequantCache {
  constructor(cap) {
    this.cache = new Map();
    this.cap = cap;
    this.hits = 0;
    this.misses = 0;
  }
  get(entry) {
    const hit = this.cache.get(entry.path);
    if (hit) {
      this.cache.delete(entry.path);
      this.cache.set(entry.path, hit);
      this.hits++;
      return hit;
    }
    this.misses++;
    const f = dequantizeChunksRaw(
      entry.chunkBytes,
      entry.scales,
      entry.chunkCount,
      entry.dims,
    );
    this.cache.set(entry.path, f);
    if (this.cache.size > this.cap) {
      const lru = this.cache.keys().next().value;
      if (lru !== undefined) this.cache.delete(lru);
    }
    return f;
  }
}

function directionalMax(x, xCount, y, yCount, dims) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < xCount; i++) {
    const xOff = i * dims;
    let best = -1;
    for (let j = 0; j < yCount; j++) {
      const yOff = j * dims;
      let dot = 0;
      for (let d = 0; d < dims; d++) dot += x[xOff + d] * y[yOff + d];
      if (dot > best) best = dot;
    }
    const clamped = best > 0 ? best : 0;
    const w = i === TITLE_CHUNK_INDEX ? TITLE_WEIGHT : 1;
    num += w * clamped;
    den += w;
  }
  return den > 0 ? num / den : 0;
}

// biMax over two notes given resolver fns yielding each note's fp32 chunk buffer.
function biMax(a, b, aChunks, bChunks) {
  const dims = a.dims;
  if (b.dims !== dims) return 0;
  if (a.chunkCount === 0 || b.chunkCount === 0) return 0;
  const aToB =
    a.chunkCount === 1
      ? Math.max(0, dotRow(aChunks, TITLE_CHUNK_INDEX, b.meanVector, dims))
      : directionalMax(aChunks, a.chunkCount, bChunks, b.chunkCount, dims);
  const bToA =
    b.chunkCount === 1
      ? Math.max(0, dotRow(bChunks, TITLE_CHUNK_INDEX, a.meanVector, dims))
      : directionalMax(bChunks, b.chunkCount, aChunks, a.chunkCount, dims);
  return (aToB + bToA) / 2;
}

// Eager fp32 buffers, used for the correctness reference (full dequant, no LRU).
const eagerChunks = new Map();
for (const e of int8Entries) {
  eagerChunks.set(
    e.path,
    dequantizeChunksRaw(e.chunkBytes, e.scales, e.chunkCount, e.dims),
  );
}

const stage1Samples = [];
const stage2Samples = [];
const totalSamples = [];
const cache = new DequantCache(CACHE_CAP);
let maxScoreDiff = 0;
let pairsChecked = 0;

for (let s = 0; s < N_SWITCHES; s++) {
  const active = int8Entries[Math.floor(rand() * int8Entries.length)];

  // --- Stage 1: coarse mean-vector cosine over all notes + sort + slice width. ---
  const t1 = performance.now();
  const coarse = [];
  for (const e of int8Entries) {
    if (e.path === active.path) continue;
    coarse.push({ entry: e, c: cosineSimilarity(active.meanVector, e.meanVector) });
  }
  coarse.sort((a, b) => b.c - a.c);
  const shortlist = coarse.slice(0, SHORTLIST_WIDTH);
  stage1Samples.push(performance.now() - t1);

  // --- Stage 2: BiMax over the shortlist via the LRU. ---
  const t2 = performance.now();
  const activeChunks = cache.get(active); // self dequant once, stays MRU
  const lazyScores = [];
  for (const { entry } of shortlist) {
    const bChunks = cache.get(entry);
    lazyScores.push(biMax(active, entry, activeChunks, bChunks));
  }
  stage2Samples.push(performance.now() - t2);
  totalSamples.push(stage1Samples[s] + stage2Samples[s]);

  // --- Correctness: same Stage 2 EAGERLY (full fp32, no LRU); diff must be 0. ---
  const activeEager = eagerChunks.get(active.path);
  for (let k = 0; k < shortlist.length; k++) {
    const entry = shortlist[k].entry;
    const eager = biMax(active, entry, activeEager, eagerChunks.get(entry.path));
    const diff = Math.abs(eager - lazyScores[k]);
    if (diff > maxScoreDiff) maxScoreDiff = diff;
    pairsChecked++;
  }
}

const stage1Med = median(stage1Samples);
const stage2Med = median(stage2Samples);
const totalMed = median(totalSamples);
const hitRate = cache.hits / (cache.hits + cache.misses);
const assertOk = maxScoreDiff === 0;

// =============================================================================
// PRINT
// =============================================================================
function row(metric, oldV, newV, delta) {
  console.log(
    metric.padEnd(26) + "| " + String(oldV).padEnd(16) + "| " +
      String(newV).padEnd(16) + "| " + delta,
  );
}

console.log("=".repeat(82));
console.log("AT-REST + RANK BENCHMARK  (" + N_NOTES + " notes, ~" + MEAN_CHUNKS +
  " chunks/note, " + DIMS + " dims)");
console.log("gc:", typeof global.gc === "function" ? "exposed" : "NOT exposed (run with --expose-gc for clean RAM)");
console.log("=".repeat(82));
console.log(
  "metric".padEnd(26) + "| " + "old (v3 base64)".padEnd(16) + "| " +
    "new (binary)".padEnd(16) + "| ratio/delta",
);
console.log("-".repeat(82));

row(
  "1. index file size",
  mb(oldBytes),
  `${mb(newBlobBytes)}+${(newJsonBytes / 1024).toFixed(0)}KB`,
  `${(newTotal / oldBytes).toFixed(2)}x  (new = ${mb(newTotal)})`,
);
row(
  "2. cold-load (median/5)",
  fmtMs(loadOld),
  fmtMs(loadNew),
  `${(loadNew / loadOld).toFixed(2)}x`,
);
row("3. RAM rss delta", mb(eagerRss), mb(lazyRss), ramRatio(eagerRss, lazyRss));
row(
  "   RAM heapUsed delta",
  mb(eagerHeap),
  mb(lazyHeap),
  ramRatio(eagerHeap, lazyHeap),
);
row(
  "   chunk data (theory)",
  mb(fp32ChunkBytes),
  mb(int8ChunkBytes),
  `${(fp32ChunkBytes / int8ChunkBytes).toFixed(2)}x`,
);
console.log("-".repeat(82));
console.log("4. RANK LATENCY (median of " + N_SWITCHES + " switches):");
console.log("     Stage 1 (coarse, all " + N_NOTES + "):  " + fmtMs(stage1Med));
console.log("     Stage 2 (BiMax, shortlist " + SHORTLIST_WIDTH + " via LRU): " + fmtMs(stage2Med));
console.log("     total per switch:            " + fmtMs(totalMed));
console.log("     LRU hit rate:                " + (hitRate * 100).toFixed(1) + "%  (cap " + CACHE_CAP + ")");
console.log("-".repeat(82));
console.log(
  "5. BIT-IDENTICAL ASSERT (lazy LRU == eager fp32 BiMax over " + pairsChecked + " pairs):",
);
console.log(
  "     max |score diff| = " + maxScoreDiff + "  ->  " +
    (assertOk ? "PASS (ranking identical)" : "FAIL (lazy dequant drifted!)"),
);
console.log(
  "   manifest-only header == full serialize header:  " +
    (manifestMatches ? "PASS" : "FAIL"),
);
console.log("=".repeat(82));

if (!assertOk || !manifestMatches) process.exitCode = 1;
