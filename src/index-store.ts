import { App, TFile, Notice, normalizePath } from "obsidian";
import {
  EmbeddingEngine,
  cosineSimilarity,
  type ProgressCallback,
} from "./embeddings";

// =============================================================================
// SCHEMA / version constant. Bumped 1 -> 2 for the multi-vector layout (mean vector
// PLUS a capped set of int8-quantized chunk vectors). Bumped 2 -> 3 for the keyphrase
// summary LABEL: the per-note summary is now a tight 3–7-word topic label computed at
// index time (KeyBERT-style: candidate phrases embedded and ranked by cosine to the
// note mean, diversified with MMR) and PERSISTED on the entry — not a render-time
// centroid-sentence pick. The label needs a model pass to compute, so it is produced
// once during build and stored; a v2 index has no label field, so it is detected as
// stale on load() and a full rebuild is triggered. Any index written for a different
// model/dims, or a different quantization/text-persistence policy, is likewise
// invalidated. No silent half-migration.
// =============================================================================
const INDEX_VERSION = 3 as const;

const STORE_FILE = "index.json";
const BATCH_SIZE = 8;

// Chunking knobs. MAX_CHUNKS is the body-chunk cap (the title chunk is extra, so a
// note holds up to MAX_CHUNKS + 1 vectors). TARGET_WORDS controls greedy sentence
// windowing (~200 tokens). Defaults are overridable via IndexStoreOptions.maxChunks.
const DEFAULT_MAX_CHUNKS = 16;
const TARGET_WORDS = 60;
const MIN_WINDOW_WORDS = 10;

// Title chunk lives at index 0 of every note's chunk buffer and is weighted ~2x in
// the per-direction BiMax means so a strong title match lifts the score.
const TITLE_CHUNK_INDEX = 0;
const TITLE_WEIGHT = 2;

// Stage-1 -> Stage-2 funnel: keep at least this many coarse candidates (and at
// least topK*4) for the fine re-rank. Overridable via options.shortlistSize.
const DEFAULT_SHORTLIST = 60;

// Stage-1 RECALL floor on the mean-vector cosine. This is deliberately LOW (and
// distinct from the user-facing minSimilarity, which is applied against the final
// BiMax score in Stage 2). Its only job is to drop obvious noise from the coarse
// shortlist while KEEPING related-but-low-mean-cosine notes — exactly the class
// the chunk-level redesign exists to rescue (a short on-topic note whose whole-note
// mean blurs toward a centroid, yet whose chunks match strongly). Filtering on the
// unreliable mean here would re-introduce the very failure we fixed.
const COARSE_FLOOR = 0.2;

// Hybrid structural boost weights and cap. boost is added to the semantic score
// AFTER it is scaled into [0, B_MAX]; B_MAX itself comes from options
// (structureInfluence) so the user can tune "structure influence". These weights
// are the RELATIVE contribution of each signal before that scale.
const W_DIRECT_LINK = 1.0;
const W_SHARED_TAGS = 0.6;
const W_BIBLIO = 0.5;
const W_FRONTMATTER = 0.3;
// The maximum raw weighted-signal sum (all signals firing fully). Used to scale the
// raw sum into [0, B_MAX]; a fixed denominator keeps the boost interpretable.
// (The same-folder signal was removed: in an atomic-note vault the whole library
// lives in ~3 folders, so it fired for nearly every candidate — constant noise that
// both inflated the boost indiscriminately and produced a near-useless pill. With it
// gone the denominator is re-derived WITHOUT W_SAME_FOLDER so the remaining signals
// keep their full [0,1] range.)
const SIGNAL_NORM =
  W_DIRECT_LINK + W_SHARED_TAGS + W_BIBLIO + W_FRONTMATTER;

// --- keyphrase summary LABEL knobs ------------------------------------------
// Defensive char cap on the joined label (a real 3–7-word label is well under this).
const SUMMARY_LABEL_CHARS = 64;
// Max candidate n-gram phrase length (words). 1–3 covers "Theory of mind",
// "Mitochondrium", "spanning tree" etc. without runaway phrases.
const MAX_PHRASE_WORDS = 3;
// Cap on candidates EMBEDDED per note. We keep the longest / most distinctive phrases
// first. Every note's candidates in a build batch are flattened into ONE shared
// embedBatch() pass (see computeSummaryLabels), so the aggregate label cost is one
// extra ONNX pass per BATCH, not per note.
const MAX_CANDIDATES = 36;
// How many MMR picks to assemble into the final label, and the target word budget.
const MAX_LABEL_PHRASES = 3;
const TARGET_LABEL_WORDS_MIN = 3;
const TARGET_LABEL_WORDS_MAX = 7;
// MMR trade-off: lambda 0.6 is slightly relevance-biased (KeyBERT diversity ≈ 0.4).
const MMR_LAMBDA = 0.6;

// On-disk shape of one chunk block: a symmetric per-vector int8 quantization. The
// `q` string is base64 of one Int8Array of length chunkCount*dims; `scales` holds
// one fp32 scale per chunk row (max|v|/127). Dequantized + re-normalized on load.
interface StoredChunkBlock {
  scales: number[];
  q: string; // base64(Int8Array)
}

// On-disk shape of one embedded note. The mean vector stays fp32 (it drives the
// Stage-1 coarse shortlist and is only ~1/N of the data; the user-facing
// minSimilarity floor is applied later against the Stage-2 BiMax score).
interface StoredEntry {
  path: string;
  mtime: number;
  dims: number;
  chunkCount: number;
  meanVector: number[]; // fp32
  chunks: StoredChunkBlock; // int8-quantized chunk vectors
  chunkTexts?: string[]; // only persisted when summary feature is on
  summaryLabel?: string; // KeyBERT-style topic label, computed at index time
}

// Header + body persisted to the plugin config dir. `version`/`modelId`/`dims`
// gate a full rebuild. `quantized`/`hasChunkText` make a future change of
// quantization or text-persistence policy detectable and self-invalidating.
interface StoredIndex {
  version: typeof INDEX_VERSION;
  modelId: string;
  dims: number;
  quantized: boolean;
  hasChunkText: boolean;
  entries: StoredEntry[];
}

// In-memory entry. The chunk vectors are kept as ONE contiguous Float32Array of
// length chunkCount*dims (cache-friendly for the Stage-2 dot loops); the mean
// drives Stage 1. chunkTexts are only present when summaries are enabled.
interface IndexEntry {
  path: string;
  mtime: number;
  dims: number;
  chunkCount: number;
  meanVector: Float32Array;
  chunks: Float32Array; // length == chunkCount * dims
  chunkTexts?: string[];
  // The note's topic LABEL (3–7 words), computed once at index time from the note's
  // own chunks (KeyBERT-style) and persisted. Present only when summaries are on.
  summaryLabel?: string;
}

// Why a note was surfaced — derived in priority order from the structural signals
// that actually fired, falling back to "semantic". `detail` names the top shared
// tag for the shared-tags kind.
export type WhyKind =
  | "linked"
  | "shared-tags"
  | "co-cited"
  | "semantic";

export interface WhyReason {
  kind: WhyKind;
  detail?: string;
}

export type ConnectionType = "linked" | "related";

// A single ranked result handed to the view. Back-compat: file/score/approximate
// are unchanged. The new fields are optional so keywordRank results (which lack
// them) render gracefully as plain cards.
export interface RankedNote {
  file: TFile;
  score: number; // final score (semantic + boost), or a keyword overlap score
  approximate: boolean; // true when produced by the keyword fallback
  semantic?: number; // the pre-boost BiMax similarity, when available
  reason?: WhyReason;
  connection?: ConnectionType;
}

// Index lifecycle state, surfaced to the view for its status line.
export type IndexStatus = "idle" | "loading" | "building" | "ready" | "error";

export interface IndexProgress {
  status: IndexStatus;
  done: number;
  total: number;
  message?: string;
}

export interface IndexStoreOptions {
  embedCharLimit: number;
  excludeFolders: string[];
  topK: number;
  minSimilarity: number;
  // New multi-vector / ranking knobs.
  chunking: boolean; // master toggle; off reverts to a single mean vector
  structureInfluence: number; // B_MAX for the hybrid boost (0..~0.3)
  maxChunks: number; // body-chunk cap (excludes the title chunk)
  shortlistSize: number; // Stage-1 -> Stage-2 funnel width
  showSummary: boolean; // persist chunkTexts so summaries survive a reload
}

type ProgressListener = (p: IndexProgress) => void;

// One paragraph/sentence-window chunk plus the structural metadata we track for it.
interface NoteChunk {
  text: string;
  isTitle: boolean;
  heading?: string;
}

// Active-note context for the hybrid structural boost, computed once per rank().
interface StructuralContext {
  path: string;
  tags: Set<string>;
  outlinks: Set<string>; // resolved out-target paths
  linkTargets: Set<string>; // raw link labels (basename/path, lower-cased)
  ambiguousBasenames: Set<string>; // basenames shared by 2+ files (link-ambiguous)
  frontmatter: Record<string, unknown> | undefined;
}

// =============================================================================
// Markdown cleaning
// =============================================================================

// Shared body cleaner used by both stripMarkdown (single-line) and
// stripMarkdownBlocks (structure-preserving). Removes frontmatter, code, links,
// headings markers, emphasis and HTML, WITHOUT touching newlines.
function cleanMarkdownInline(content: string): string {
  let text = content;
  // YAML frontmatter at the very start of the file.
  text = text.replace(/^---\n[\s\S]*?\n---\n?/, "\n");
  // Fenced + inline code (drop the code, keep surrounding prose).
  text = text.replace(/```[\s\S]*?```/g, "\n");
  text = text.replace(/`[^`]*`/g, " ");
  // Images ![alt](url) and ![[embed]] -> drop entirely.
  text = text.replace(/!\[\[[^\]]*\]\]/g, " ");
  text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, " ");
  // Wikilinks [[Target|Alias]] / [[Target]] -> keep the visible label.
  text = text.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2");
  text = text.replace(/\[\[([^\]]+)\]\]/g, "$1");
  // Markdown links [text](url) -> keep text.
  text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
  // Headings, blockquotes, list bullets at line start (keep the text after them).
  text = text.replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, "");
  // Emphasis / strikethrough markers.
  text = text.replace(/[*_~]/g, "");
  // HTML tags.
  text = text.replace(/<[^>]+>/g, " ");
  return text;
}

// Strip markdown/frontmatter to a single line of plain prose. Used by the snippet
// path and keywordRank (which want a flat blob). UNCHANGED behavior.
export function stripMarkdown(content: string): string {
  return cleanMarkdownInline(content).replace(/\s+/g, " ").trim();
}

// Structure-preserving variant: collapses whitespace WITHIN a line only and KEEPS
// blank-line paragraph boundaries, so chunkNote() can split on them. Headings are
// preserved as their own lines (their `#` markers are already stripped above).
export function stripMarkdownBlocks(content: string): string {
  const cleaned = cleanMarkdownInline(content);
  return cleaned
    .split("\n")
    .map((line) => line.replace(/[ \t\f\v]+/g, " ").trimEnd())
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// =============================================================================
// Sentence segmentation + chunking
// =============================================================================

// Intl.Segmenter ships in Electron/Chromium and is locale-agnostic + correct for
// German + English. Created once and reused. Falls back to a regex when absent so
// chunking can never throw.
let sentenceSegmenter: Intl.Segmenter | null = null;
let segmenterResolved = false;

function getSegmenter(): Intl.Segmenter | null {
  if (segmenterResolved) return sentenceSegmenter;
  segmenterResolved = true;
  try {
    if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
      sentenceSegmenter = new Intl.Segmenter(undefined, {
        granularity: "sentence",
      });
    }
  } catch {
    sentenceSegmenter = null;
  }
  return sentenceSegmenter;
}

function splitSentences(paragraph: string): string[] {
  const seg = getSegmenter();
  if (seg) {
    const out: string[] = [];
    for (const { segment } of seg.segment(paragraph)) {
      const s = segment.trim();
      if (s.length > 0) out.push(s);
    }
    return out;
  }
  // Fallback: split after sentence-final punctuation followed by whitespace.
  return paragraph
    .split(/(?<=[.!?…])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function countWords(text: string): number {
  const m = text.match(/\S+/g);
  return m ? m.length : 0;
}

// Greedily group consecutive sentences into ~TARGET_WORDS windows with one-sentence
// overlap. A trailing sub-MIN_WINDOW_WORDS fragment is appended to the previous
// window rather than emitted alone.
function windowSentences(sentences: string[]): string[] {
  if (sentences.length === 0) return [];
  const windows: string[] = [];
  let current: string[] = [];
  let currentWords = 0;

  for (let i = 0; i < sentences.length; i++) {
    const s = sentences[i];
    current.push(s);
    currentWords += countWords(s);
    if (currentWords >= TARGET_WORDS) {
      windows.push(current.join(" "));
      // Carry the last sentence forward as ~1-sentence overlap.
      const last = current[current.length - 1];
      current = [last];
      currentWords = countWords(last);
    }
  }

  // Flush the tail.
  const tail = current.join(" ").trim();
  if (tail.length > 0) {
    // If the tail is just the carried-over overlap sentence (already in the prior
    // window) skip it; if it's a genuine short remainder, append to the previous
    // window rather than emit a tiny fragment.
    if (windows.length > 0 && countWords(tail) < MIN_WINDOW_WORDS) {
      windows[windows.length - 1] = `${windows[windows.length - 1]} ${tail}`.trim();
    } else if (
      windows.length === 0 ||
      windows[windows.length - 1] !== tail
    ) {
      windows.push(tail);
    }
  }
  return windows;
}

// =============================================================================
// int8 quantization helpers (chunk block only — the mean stays fp32)
// =============================================================================

function base64FromInt8(arr: Int8Array): string {
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

function int8FromBase64(b64: string): Int8Array {
  const binary = atob(b64);
  const out = new Int8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    // charCodeAt is 0..255; reinterpret as signed int8.
    out[i] = (binary.charCodeAt(i) << 24) >> 24;
  }
  return out;
}

// Quantize a contiguous fp32 chunk buffer (chunkCount rows of `dims`) into a
// symmetric per-row int8 block. Returns the base64 payload + per-row scales.
function quantizeChunks(
  chunks: Float32Array,
  chunkCount: number,
  dims: number,
): StoredChunkBlock {
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
  return { scales, q: base64FromInt8(q) };
}

// Dequantize an int8 chunk block back to a contiguous fp32 buffer and RE-NORMALIZE
// each row (quantization perturbs the L2 norm by ~1-4%; cosineSimilarity assumes
// unit vectors). Returns null when the payload is the wrong length.
function dequantizeChunks(
  block: StoredChunkBlock,
  chunkCount: number,
  dims: number,
): Float32Array | null {
  const q = int8FromBase64(block.q);
  if (q.length !== chunkCount * dims) return null;
  if (block.scales.length !== chunkCount) return null;
  const out = new Float32Array(chunkCount * dims);
  for (let c = 0; c < chunkCount; c++) {
    const off = c * dims;
    const scale = block.scales[c];
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

// Mean of a set of unit vectors, re-L2-normalized. Operates over a contiguous
// buffer of `count` rows of `dims`.
function meanOf(buffer: Float32Array, count: number, dims: number): Float32Array {
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
function dotRow(
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
// IndexStore
// =============================================================================

// Owns the per-note vectors: builds them, persists them, keeps them in step with
// the vault, and answers similarity queries. The view never touches files
// directly — it asks the store to rank.
export class IndexStore {
  private readonly app: App;
  // Mutable so the plugin can swap the engine IN PLACE on a model/device change,
  // keeping this single store instance (and the view's progress subscription)
  // valid. Never replace the IndexStore itself.
  private engine: EmbeddingEngine;
  private readonly configDir: string;
  private options: IndexStoreOptions;

  private entries = new Map<string, IndexEntry>();
  private progress: IndexProgress = { status: "idle", done: 0, total: 0 };
  private listeners = new Set<ProgressListener>();

  // Guards against two concurrent builds (e.g. a manual rebuild during startup).
  private building = false;
  // Paths queued for incremental re-embedding while a build is in flight.
  private pending = new Set<string>();

  // Memoized significant-word sets for the keyword fallback, keyed by path and
  // invalidated by mtime. Avoids re-tokenizing every candidate on every render.
  private wordCache = new Map<string, { mtime: number; words: Set<string> }>();

  // Summary-label cache, keyed by mtime. The label itself is computed at INDEX time
  // (it needs a model pass) and persisted on the entry; this cache just memoizes the
  // synchronous lookup + truncation so the render hot path never recomputes.
  private summaryCache = new Map<string, { mtime: number; text: string }>();

  // Lower-cased basenames that occur on MORE THAN ONE file in the vault. A raw
  // (unresolved) wikilink label that hits one of these is ambiguous — it cannot
  // safely be attributed to a specific candidate — so the structural boost ignores
  // it and relies on the authoritative resolvedLinks instead. Rebuilt lazily and
  // invalidated whenever a file is added/removed/renamed.
  private ambiguousBasenames: Set<string> | null = null;

  constructor(
    app: App,
    engine: EmbeddingEngine,
    configDir: string,
    options: IndexStoreOptions,
  ) {
    this.app = app;
    this.engine = engine;
    this.configDir = configDir;
    this.options = options;
  }

  // --- status plumbing -------------------------------------------------------
  onProgress(fn: ProgressListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  getProgress(): IndexProgress {
    return this.progress;
  }

  private setProgress(p: Partial<IndexProgress>): void {
    this.progress = { ...this.progress, ...p };
    for (const fn of this.listeners) fn(this.progress);
  }

  get count(): number {
    return this.entries.size;
  }

  updateOptions(options: IndexStoreOptions): void {
    this.options = options;
  }

  // Swap the embedding engine in place and drop every vector (a different model or
  // device invalidates them all). Keeps the same listener Set, so a view already
  // subscribed to this store keeps receiving the rebuild's progress. The caller
  // follows this with build().
  setEngine(engine: EmbeddingEngine): void {
    this.engine = engine;
    this.entries = new Map();
    this.pending.clear();
    this.wordCache.clear();
    this.summaryCache.clear();
    this.ambiguousBasenames = null;
    this.setProgress({ status: "idle", done: 0, total: 0, message: undefined });
  }

  private get maxChunks(): number {
    return Math.max(1, this.options.maxChunks || DEFAULT_MAX_CHUNKS);
  }

  // --- persistence -----------------------------------------------------------
  private get storePath(): string {
    return normalizePath(`${this.configDir}/${STORE_FILE}`);
  }

  // Load a persisted index from the plugin config dir. Returns false (so the
  // caller triggers a build) when the file is missing, malformed, or was written
  // for a different model/dimension/version — including any old version-1 index,
  // which forces a full rebuild on upgrade.
  async load(): Promise<boolean> {
    this.setProgress({ status: "loading", done: 0, total: 0 });
    try {
      const adapter = this.app.vault.adapter;
      if (!(await adapter.exists(this.storePath))) {
        this.setProgress({ status: "idle" });
        return false;
      }
      const raw = await adapter.read(this.storePath);
      const data = JSON.parse(raw) as StoredIndex;
      if (
        data.version !== INDEX_VERSION ||
        data.modelId !== this.engine.modelId ||
        data.quantized !== true ||
        !Array.isArray(data.entries)
      ) {
        this.setProgress({ status: "idle" });
        return false;
      }
      // The summary feature needs persisted chunk text. If the user has it ON but
      // this index was written without it (hasChunkText false), the cards would show
      // empty summaries (snippet fallback) until some future rebuild. Treat it as
      // stale so the caller re-embeds once and summaries work from the first load,
      // rather than degrading silently. (The reverse — text persisted but feature
      // off — is harmless: the extra text is simply ignored.)
      if (this.options.showSummary && data.hasChunkText !== true) {
        this.setProgress({ status: "idle" });
        return false;
      }
      const loaded = new Map<string, IndexEntry>();
      for (const e of data.entries) {
        if (e.dims !== data.dims) continue;
        if (e.meanVector.length !== data.dims) continue;
        const chunks = dequantizeChunks(e.chunks, e.chunkCount, e.dims);
        if (!chunks) continue;
        loaded.set(e.path, {
          path: e.path,
          mtime: e.mtime,
          dims: e.dims,
          chunkCount: e.chunkCount,
          meanVector: Float32Array.from(e.meanVector),
          chunks,
          chunkTexts: e.chunkTexts,
          summaryLabel: e.summaryLabel,
        });
      }
      this.entries = loaded;
      this.setProgress({ status: "ready", done: loaded.size, total: loaded.size });
      return true;
    } catch (e) {
      console.warn("[related-notes] failed to load index, will rebuild", e);
      this.setProgress({ status: "idle" });
      return false;
    }
  }

  private async persist(): Promise<void> {
    const dims = this.firstDims();
    const keepText = this.options.showSummary;
    const data: StoredIndex = {
      version: INDEX_VERSION,
      modelId: this.engine.modelId,
      dims,
      quantized: true,
      hasChunkText: keepText,
      entries: Array.from(this.entries.values()).map((e) => ({
        path: e.path,
        mtime: e.mtime,
        dims: e.dims,
        chunkCount: e.chunkCount,
        meanVector: Array.from(e.meanVector),
        chunks: quantizeChunks(e.chunks, e.chunkCount, e.dims),
        ...(keepText && e.chunkTexts ? { chunkTexts: e.chunkTexts } : {}),
        ...(keepText && e.summaryLabel ? { summaryLabel: e.summaryLabel } : {}),
      })),
    };
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(this.configDir))) {
      await adapter.mkdir(this.configDir);
    }
    await adapter.write(this.storePath, JSON.stringify(data));
  }

  private firstDims(): number {
    for (const e of this.entries.values()) return e.dims;
    return 0;
  }

  // --- file selection --------------------------------------------------------
  private isExcluded(path: string): boolean {
    return this.options.excludeFolders.some((folder) => {
      const f = folder.replace(/\/+$/, "");
      if (f.length === 0) return false;
      return path === f || path.startsWith(`${f}/`);
    });
  }

  private indexableFiles(): TFile[] {
    return this.app.vault
      .getMarkdownFiles()
      .filter((f) => !this.isExcluded(f.path));
  }

  // --- chunking --------------------------------------------------------------
  // Split a note into a title chunk + capped sentence-window body chunks. The
  // title chunk is standalone (no glued body prefix) so a short directly-related
  // note stays specific. When chunking is disabled, a single body chunk is emitted
  // (== the old single-blob behavior, modulo the standalone title chunk).
  private chunkNote(file: TFile, body: string): NoteChunk[] {
    const chunks: NoteChunk[] = [
      { text: file.basename, isTitle: true },
    ];

    const charLimit = this.options.embedCharLimit;

    if (!this.options.chunking) {
      const flat = stripMarkdown(body);
      const limited = flat.length > charLimit ? flat.slice(0, charLimit) : flat;
      if (limited.length > 0) chunks.push({ text: limited, isTitle: false });
      return chunks;
    }

    const blocks = stripMarkdownBlocks(body);
    const limited =
      blocks.length > charLimit ? blocks.slice(0, charLimit) : blocks;

    // Split into paragraphs on blank lines, tracking the nearest preceding heading.
    // Heading lines are short lines that originally started with `#` — by now the
    // marker is gone, so we approximate a heading as a standalone short line that is
    // immediately followed by a blank line. We don't prepend it (avoids embedding
    // collapse); the hook is left for future light context.
    const paragraphs = limited
      .split(/\n{2,}/)
      .map((p) => p.replace(/\n/g, " ").trim())
      .filter((p) => p.length > 0);

    const bodyChunks: NoteChunk[] = [];
    for (const para of paragraphs) {
      const sentences = splitSentences(para);
      for (const w of windowSentences(sentences)) {
        bodyChunks.push({ text: w, isTitle: false });
      }
    }

    // Cap: keep the first N-k windows + k evenly-spaced later windows so a long
    // essay's tail is not dropped wholesale.
    const cap = this.maxChunks;
    if (bodyChunks.length <= cap) {
      chunks.push(...bodyChunks);
    } else {
      const head = Math.ceil(cap * 0.6);
      const tailCount = cap - head;
      for (let i = 0; i < head; i++) chunks.push(bodyChunks[i]);
      // Evenly space the remaining picks across the tail region.
      const start = head;
      const span = bodyChunks.length - start;
      for (let k = 0; k < tailCount; k++) {
        const idx = start + Math.floor(((k + 1) * span) / (tailCount + 1));
        chunks.push(bodyChunks[Math.min(idx, bodyChunks.length - 1)]);
      }
    }

    return chunks;
  }

  // --- (re)build -------------------------------------------------------------
  // Embed every indexable note in batches, yielding to the event loop between
  // batches so the UI never freezes. Reuses an existing entry when the file's
  // mtime is unchanged. Each batch FLATTENS all chunks across its files into one
  // embedBatch() call (a single ONNX pass), then regroups by offsets.
  //
  // If every candidate fails to embed the status flips to "error" with a Notice.
  // `force` re-embeds every note even if its mtime is unchanged.
  async build(onProgress?: ProgressCallback, force = false): Promise<void> {
    if (this.building) return;
    this.building = true;
    try {
      const files = this.indexableFiles();
      const total = files.length;
      this.setProgress({ status: "building", done: 0, total, message: undefined });

      const notice = new Notice("Related notes: indexing…", 0);
      const next = new Map<string, IndexEntry>();
      let done = 0;
      let attempted = 0; // files that needed a fresh embed (not mtime-reused)
      let embedded = 0; // of those, how many succeeded
      let firstError: unknown = null;

      try {
        for (let i = 0; i < files.length; i += BATCH_SIZE) {
          const batch = files.slice(i, i + BATCH_SIZE);

          // Reuse unchanged entries; collect the rest, flattening their chunks into
          // ONE texts[] with an offsets table so we regroup after one ONNX pass.
          const pendingFiles: { file: TFile; chunks: NoteChunk[] }[] = [];
          const allTexts: string[] = [];
          const offsets: number[] = [0];

          for (const file of batch) {
            const existing = this.entries.get(file.path);
            if (!force && existing && existing.mtime === file.stat.mtime) {
              next.set(file.path, existing);
              continue;
            }
            const chunks = await this.readChunks(file);
            if (chunks.length === 0) continue;
            pendingFiles.push({ file, chunks });
            for (const c of chunks) allTexts.push(c.text);
            offsets.push(allTexts.length);
          }

          if (pendingFiles.length > 0) {
            attempted += pendingFiles.length;
            try {
              const vectors = await this.engine.embedBatch(
                allTexts,
                // Symmetric "query:" prefix on BOTH sides for note-to-note
                // similarity (no-op for prefix-free paraphrase models).
                "query",
                onProgress,
              );
              const assembled: IndexEntry[] = [];
              for (let f = 0; f < pendingFiles.length; f++) {
                const { file, chunks } = pendingFiles[f];
                const start = offsets[f];
                const end = offsets[f + 1];
                const entry = this.assembleEntry(file, chunks, vectors, start, end);
                if (entry) {
                  next.set(file.path, entry);
                  this.summaryCache.delete(file.path);
                  assembled.push(entry);
                  embedded++;
                }
              }
              // One extra ONNX pass for the WHOLE batch's keyphrase labels (no-op when
              // summaries are off), not one per note — keeps the build cost bounded.
              await this.computeSummaryLabels(assembled);
            } catch (e) {
              if (!firstError) firstError = e;
              console.warn("[related-notes] batch embed failed", e);
            }
          }

          done = Math.min(files.length, i + batch.length);
          const pct = total > 0 ? Math.round((done / total) * 100) : 100;
          notice.setMessage(`Related notes: indexing… ${pct}% (${done}/${total})`);
          this.setProgress({ done, total });
          await sleep(0);
        }
      } finally {
        notice.hide();
      }

      // Total model failure: notes needed embedding but none succeeded.
      if (attempted > 0 && embedded === 0) {
        this.entries = next;
        const detail =
          firstError instanceof Error ? firstError.message : String(firstError);
        this.setProgress({
          status: "error",
          message: `could not load the embedding model (${detail})`,
        });
        new Notice(
          "Related notes: could not load the embedding model. Check your internet connection (first run downloads the model) and that your firewall/CSP allows it. See the console for details.",
          0,
        );
        return;
      }

      this.entries = next;
      this.setProgress({ status: "ready", done, total });
      await this.persist();
      await this.flushPending(onProgress);
    } catch (e) {
      console.error("[related-notes] index build failed", e);
      this.setProgress({
        status: "error",
        message: e instanceof Error ? e.message : String(e),
      });
      new Notice("Related notes: indexing failed. See the console for details.");
    } finally {
      this.building = false;
    }
  }

  // Regroup a slice [start, end) of the batch's flat vector list into one note's
  // contiguous chunk buffer + re-normalized mean. Returns null if the slice is
  // empty or dims can't be determined. The summaryLabel is filled separately by
  // computeSummaryLabels() (it needs its own model pass), so it is left undefined here.
  private assembleEntry(
    file: TFile,
    chunks: NoteChunk[],
    vectors: Float32Array[],
    start: number,
    end: number,
  ): IndexEntry | null {
    const count = end - start;
    if (count <= 0) return null;
    const first = vectors[start];
    if (!first || first.length === 0) return null;
    const dims = first.length;

    const buffer = new Float32Array(count * dims);
    for (let c = 0; c < count; c++) {
      const v = vectors[start + c];
      if (!v || v.length !== dims) return null;
      buffer.set(v, c * dims);
    }
    const meanVector = meanOf(buffer, count, dims);

    return {
      path: file.path,
      mtime: file.stat.mtime,
      dims,
      chunkCount: count,
      meanVector,
      chunks: buffer,
      chunkTexts: this.options.showSummary ? chunks.map((c) => c.text) : undefined,
    };
  }

  // Read + chunk a single note. Returns [] when the file is empty/unreadable.
  private async readChunks(file: TFile): Promise<NoteChunk[]> {
    try {
      const body = await this.app.vault.cachedRead(file);
      const chunks = this.chunkNote(file, body);
      // A title-only chunk set is still useful (very short note), so we keep it.
      return chunks.length > 0 ? chunks : [];
    } catch (e) {
      console.warn(`[related-notes] failed to read ${file.path}`, e);
      return [];
    }
  }

  // Embed a single file end-to-end (used by the incremental path). One ONNX pass
  // over its chunks; regroups into a full IndexEntry.
  private async embedFile(
    file: TFile,
    onProgress?: ProgressCallback,
  ): Promise<IndexEntry | null> {
    const chunks = await this.readChunks(file);
    if (chunks.length === 0) return null;
    try {
      const vectors = await this.engine.embedBatch(
        chunks.map((c) => c.text),
        "query",
        onProgress,
      );
      const entry = this.assembleEntry(file, chunks, vectors, 0, vectors.length);
      // Single-note path: the batched labeller still does exactly one extra embed pass
      // for this one note (or nothing when summaries are off).
      if (entry) await this.computeSummaryLabels([entry]);
      return entry;
    } catch (e) {
      console.warn(`[related-notes] failed to embed ${file.path}`, e);
      return null;
    }
  }

  // --- incremental updates ---------------------------------------------------
  async updateFile(file: TFile): Promise<void> {
    if (this.isExcluded(file.path) || file.extension !== "md") return;
    // A new file may introduce (or a re-embed of an existing one may not change) a
    // basename collision; invalidate the cache so the next rank recomputes it.
    this.ambiguousBasenames = null;
    if (this.building) {
      this.pending.add(file.path);
      return;
    }
    const entry = await this.embedFile(file);
    if (entry) {
      this.entries.set(file.path, entry);
      this.summaryCache.delete(file.path);
      this.setProgress({ done: this.entries.size, total: this.entries.size });
      await this.persist();
    }
  }

  removeFile(path: string): void {
    this.wordCache.delete(path);
    this.summaryCache.delete(path);
    this.ambiguousBasenames = null;
    if (this.entries.delete(path)) {
      this.setProgress({ done: this.entries.size, total: this.entries.size });
      void this.persist();
    }
  }

  renameFile(oldPath: string, file: TFile): void {
    this.wordCache.delete(oldPath);
    this.summaryCache.delete(oldPath);
    this.ambiguousBasenames = null;
    this.entries.delete(oldPath);
    void this.updateFile(file);
  }

  private async flushPending(onProgress?: ProgressCallback): Promise<void> {
    if (this.pending.size === 0) return;
    const paths = Array.from(this.pending);
    this.pending.clear();
    for (const path of paths) {
      const file = this.app.vault.getAbstractFileByPath(path);
      if (file instanceof TFile) {
        const entry = await this.embedFile(file, onProgress);
        if (entry) {
          this.entries.set(path, entry);
          this.summaryCache.delete(path);
        }
      }
    }
    await this.persist();
  }

  // --- ranking ---------------------------------------------------------------
  // Two-stage funnel, runs on every active-leaf-change (debounced by the view at
  // 300ms):
  //   Stage 1 (coarse, every note): one mean-vector dot per note + a LOW recall
  //   floor (COARSE_FLOOR), take the top `width` as the shortlist. The coarse mean
  //   cosine is the metric the diagnosis proved unreliable, so it is used ONLY to
  //   build the shortlist — never as the user-facing similarity floor.
  //   Stage 2 (fine, shortlist only): symmetric Bidirectional MaxSim over chunk
  //   sets, plus a bounded structural boost, then the user-facing minSimilarity
  //   floor is applied against the FINAL score (not the mean cosine).
  //
  // Cost: Stage 1 is O(notes * dims) — a handful of ms even for thousands of notes.
  // Stage 2 is O(width * Ca * Cb * dims) where Ca/Cb are the two notes' chunk
  // counts — quadratic in chunk count on BOTH notes. At defaults (width 60, ~17
  // vectors/note, 384 dims) that is single-digit ms; at the advanced caps
  // (shortlistSize 150, maxChunks 32 -> 33 vectors/note, mpnet 768 dims) it climbs
  // to tens of ms, still comfortably inside the 300ms view debounce. Raising
  // shortlistSize and maxChunks together multiplies the per-switch cost.
  //
  // Falls back to keywordRank when the active note has no embedding yet.
  rank(active: TFile): RankedNote[] {
    const self = this.entries.get(active.path);
    if (!self) return this.keywordRank(active);

    // --- Stage 1: coarse mean-vector shortlist (LOW recall floor) ------------
    const shortlist: { entry: IndexEntry; coarse: number }[] = [];
    for (const entry of this.entries.values()) {
      if (entry.path === active.path) continue;
      if (entry.dims !== self.dims) continue;
      const coarse = cosineSimilarity(self.meanVector, entry.meanVector);
      if (coarse < COARSE_FLOOR) continue;
      shortlist.push({ entry, coarse });
    }
    shortlist.sort((a, b) => b.coarse - a.coarse);

    const width = Math.max(
      this.options.topK * 4,
      this.options.shortlistSize || DEFAULT_SHORTLIST,
    );
    const candidates = shortlist.slice(0, width);

    // --- Stage 2: fine BiMax re-rank + structural boost ----------------------
    // The user-facing minSimilarity floor is applied HERE, against the final score
    // (BiMax + boost) — the metric actually shown on the card — so a related note
    // with a modest mean cosine but a high chunk-level match survives Stage 1 and is
    // judged on its real similarity.
    const minSim = this.options.minSimilarity;
    const bMax = Math.max(0, this.options.structureInfluence);
    const activeStruct = this.structuralContext(active);
    const results: RankedNote[] = [];

    for (const { entry } of candidates) {
      const file = this.app.vault.getAbstractFileByPath(entry.path);
      if (!(file instanceof TFile)) continue;

      const semantic = this.biMax(self, entry);

      const signals = this.structuralSignals(activeStruct, file, entry);
      // signals.raw is already clamped to [0,1], so raw * bMax <= bMax; no extra cap.
      const boost = bMax > 0 ? signals.raw * bMax : 0;
      const finalScore = Math.min(1, semantic + boost);

      if (finalScore < minSim) continue;

      results.push({
        file,
        score: finalScore,
        approximate: false,
        semantic,
        reason: signals.reason,
        connection: signals.directLink ? "linked" : "related",
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, this.options.topK);
  }

  // Symmetric Bidirectional MaxSim over two notes' chunk buffers, with the title
  // chunk (index 0) weighted TITLE_WEIGHT in each per-direction weighted mean.
  private biMax(a: IndexEntry, b: IndexEntry): number {
    const dims = a.dims;
    if (b.dims !== dims) return 0;
    // Guard the invariant explicitly: a zero-chunk entry (no title chunk) would
    // otherwise feed an empty inner loop. Today chunkCount >= 1 always (the title
    // chunk), but a future stricter chunkNote must not silently emit negatives.
    if (a.chunkCount === 0 || b.chunkCount === 0) return 0;
    // A title-only STUB (chunkCount === 1 — an empty note) is grounded in the other
    // note's OVERALL topic (its L2-normalized mean vector), not the single luckiest
    // chunk match. directionalMax takes a MAX, so a one-word title would otherwise
    // spuriously align with some passage and rank the empty note for an unrelated
    // topic (e.g. a stub "Ableiten" surfacing under a security essay). Its meaning
    // then comes mainly from the title-vs-overall similarity, as it should. Notes
    // with any body (chunkCount >= 2) use full bidirectional MaxSim. mean + chunk
    // rows are both normalized, so the dot IS a cosine; floor at 0 like the max.
    const aToB =
      a.chunkCount === 1
        ? Math.max(0, dotRow(a.chunks, TITLE_CHUNK_INDEX, b.meanVector, dims))
        : this.directionalMax(a.chunks, a.chunkCount, b.chunks, b.chunkCount, dims);
    const bToA =
      b.chunkCount === 1
        ? Math.max(0, dotRow(b.chunks, TITLE_CHUNK_INDEX, a.meanVector, dims))
        : this.directionalMax(b.chunks, b.chunkCount, a.chunks, a.chunkCount, dims);
    return (aToB + bToA) / 2;
  }

  // Weighted mean over X's chunks of max cosine to any Y chunk. The title chunk of
  // X (row 0) is counted TITLE_WEIGHT times in numerator + denominator.
  private directionalMax(
    x: Float32Array,
    xCount: number,
    y: Float32Array,
    yCount: number,
    dims: number,
  ): number {
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
      // Clamp away the best=-1 sentinel: an empty Y (yCount===0) would otherwise
      // contribute a negative max into the mean. Cosine of unit vectors can be
      // negative, but a "no match at all" should floor at 0, not pull the score down.
      const clamped = best > 0 ? best : 0;
      const w = i === TITLE_CHUNK_INDEX ? TITLE_WEIGHT : 1;
      num += w * clamped;
      den += w;
    }
    return den > 0 ? num / den : 0;
  }

  // --- hybrid structural boost ----------------------------------------------
  // Lower-cased basenames shared by 2+ markdown files, computed lazily and cached
  // until the next add/remove/rename. Used to recognise an AMBIGUOUS raw wikilink
  // label (one that could resolve to several files) so the direct-link boost does
  // not misattribute it to the wrong candidate.
  private getAmbiguousBasenames(): Set<string> {
    if (this.ambiguousBasenames) return this.ambiguousBasenames;
    const seen = new Set<string>();
    const dupes = new Set<string>();
    for (const f of this.app.vault.getMarkdownFiles()) {
      const b = f.basename.toLowerCase();
      if (seen.has(b)) dupes.add(b);
      else seen.add(b);
    }
    this.ambiguousBasenames = dupes;
    return dupes;
  }

  // Per-active context computed once per rank() call.
  private structuralContext(active: TFile): StructuralContext {
    const cache = this.app.metadataCache.getFileCache(active);
    const tags = new Set<string>(
      (cache?.tags ?? []).map((t) => normalizeTag(t.tag)),
    );
    if (cache?.frontmatter?.tags) addFrontmatterTags(tags, cache.frontmatter.tags);

    const resolved = this.app.metadataCache.resolvedLinks[active.path] ?? {};
    const outlinks = new Set<string>(Object.keys(resolved));
    const linkTargets = new Set<string>(
      (cache?.links ?? []).map((l) => l.link.toLowerCase()),
    );
    return {
      path: active.path,
      tags,
      outlinks,
      linkTargets,
      ambiguousBasenames: this.getAmbiguousBasenames(),
      frontmatter: cache?.frontmatter,
    };
  }

  // Compute the bounded structural signal for one candidate and pick its why-reason.
  private structuralSignals(
    ctx: StructuralContext,
    file: TFile,
    entry: IndexEntry,
  ): { raw: number; directLink: boolean; reason: WhyReason } {
    const cache = this.app.metadataCache.getFileCache(file);

    // Direct link, either direction. The AUTHORITATIVE signal is resolvedLinks
    // (Obsidian's own path resolution): active -> candidate is ctx.outlinks.has,
    // candidate -> active is a lookup in the candidate's resolved outlinks. The raw
    // wikilink label is only a fallback for the rare case the link is recorded but
    // not yet resolved, and is used ONLY when the basename is unambiguous in the
    // vault — a label like "Index" shared by several files must not be attributed
    // to this specific candidate (wrong "Linked" pill + a full W_DIRECT_LINK boost).
    const base = file.basename.toLowerCase();
    const rawLabelMatch =
      (!ctx.ambiguousBasenames.has(base) && ctx.linkTargets.has(base)) ||
      ctx.linkTargets.has(file.path.toLowerCase());
    const aLinksB = ctx.outlinks.has(entry.path) || rawLabelMatch;
    const otherResolved = this.app.metadataCache.resolvedLinks[file.path] ?? {};
    const bLinksA = Boolean(otherResolved[ctx.path]);
    const directLink = aLinksB || bLinksA;

    // Shared tags (Jaccard) + the top shared tag for the pill.
    const otherTags = new Set<string>(
      (cache?.tags ?? []).map((t) => normalizeTag(t.tag)),
    );
    if (cache?.frontmatter?.tags) addFrontmatterTags(otherTags, cache.frontmatter.tags);
    let shared = 0;
    let topTag: string | undefined;
    for (const t of otherTags) {
      if (ctx.tags.has(t)) {
        shared++;
        if (!topTag) topTag = t;
      }
    }
    const union = ctx.tags.size + otherTags.size - shared;
    const tagJaccard = union > 0 ? shared / union : 0;

    // Bibliographic coupling: shared outgoing resolved links / min(|outlinks|).
    const otherOut = Object.keys(otherResolved);
    let coShared = 0;
    for (const k of otherOut) if (ctx.outlinks.has(k)) coShared++;
    const minOut = Math.min(ctx.outlinks.size, otherOut.length);
    const biblio = minOut > 0 ? coShared / minOut : 0;

    // Shared frontmatter key/value pairs (excluding tags, handled above).
    const fmShared = sharedFrontmatter(ctx.frontmatter, cache?.frontmatter);

    const raw =
      (directLink ? W_DIRECT_LINK : 0) +
      W_SHARED_TAGS * tagJaccard +
      W_BIBLIO * biblio +
      W_FRONTMATTER * fmShared;
    const normRaw = SIGNAL_NORM > 0 ? Math.min(1, raw / SIGNAL_NORM) : 0;

    // Why-reason, priority order: linked > shared-tags > co-cited > semantic.
    // (Same-folder was removed — meaningless in an atomic-note vault with ~3 folders.)
    let reason: WhyReason;
    if (directLink) reason = { kind: "linked" };
    else if (shared > 0) reason = { kind: "shared-tags", detail: topTag };
    else if (biblio > 0) reason = { kind: "co-cited" };
    else reason = { kind: "semantic" };

    return { raw: normRaw, directLink, reason };
  }

  // Cheap fallback ranking when no embedding is available for the active note.
  keywordRank(active: TFile): RankedNote[] {
    const cache = this.app.metadataCache.getFileCache(active);
    const activeWords = this.significantWords(active);
    const activeTags = new Set(
      (cache?.tags ?? []).map((t) => t.tag.toLowerCase()),
    );
    const linked = new Set<string>(
      (cache?.links ?? []).map((l) => l.link.toLowerCase()),
    );

    const results: RankedNote[] = [];
    for (const file of this.indexableFiles()) {
      if (file.path === active.path) continue;
      const words = this.significantWords(file);
      let overlap = 0;
      for (const w of words) if (activeWords.has(w)) overlap++;
      const denom = Math.max(1, Math.min(activeWords.size, words.size));
      let score = overlap / denom;

      const otherCache = this.app.metadataCache.getFileCache(file);
      const otherTags = (otherCache?.tags ?? []).map((t) => t.tag.toLowerCase());
      for (const t of otherTags) if (activeTags.has(t)) score += 0.25;

      if (linked.has(file.basename.toLowerCase()) || linked.has(file.path.toLowerCase())) {
        score += 0.5;
      }

      if (score <= 0) continue;
      results.push({ file, score: Math.min(1, score), approximate: true });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, this.options.topK);
  }

  // Lower-cased title words longer than 2 chars, memoized per file by mtime.
  private significantWords(file: TFile): Set<string> {
    const cached = this.wordCache.get(file.path);
    if (cached && cached.mtime === file.stat.mtime) return cached.words;
    const words = new Set(
      file.basename
        .toLowerCase()
        .split(/[^\p{L}\p{N}]+/u)
        .filter((w) => w.length > 2),
    );
    this.wordCache.set(file.path, { mtime: file.stat.mtime, words });
    return words;
  }

  // --- keyphrase summary label ----------------------------------------------
  // Synchronous, render-hot-path-safe: just returns the PERSISTED topic label for the
  // note (computed once at index time by computeSummaryLabels below). Cached by mtime
  // only to memoize the defensive truncation. Returns "" when the note isn't indexed
  // or has no label yet (caller falls back to the snippet).
  getSummary(file: TFile): string {
    const cached = this.summaryCache.get(file.path);
    if (cached && cached.mtime === file.stat.mtime) return cached.text;

    const entry = this.entries.get(file.path);
    const label = entry?.summaryLabel;
    if (!label || label.length === 0) return "";

    // Suppress a label that just echoes the title — including basename labels stored
    // by an older index for stub notes — so we don't repeat the card name. Done at
    // display time too so the fix lands on a reload without forcing a re-embed.
    const text =
      label.toLowerCase() === file.basename.toLowerCase()
        ? ""
        : truncateAtWord(label, SUMMARY_LABEL_CHARS);
    this.summaryCache.set(file.path, { mtime: file.stat.mtime, text });
    return text;
  }

  // Compute tight 3–7-word TOPIC LABELS for a whole batch of freshly-assembled entries
  // in ONE embedBatch() pass, storing each label on its entry (so persist() writes it).
  // KeyBERT-style and fully offline: it reuses the SAME embedding engine — no second
  // model, no extra download. Pipeline, per entry:
  //   1) source text = the note's own persisted chunk texts (already markdown-cleaned)
  //   2) generate 1–3-gram candidate phrases (multilingual DE+EN stopword filtered)
  // …then ACROSS the batch:
  //   3) flatten every entry's candidate surfaces into one engine.embedBatch() pass
  //      (an offsets table regroups the result per entry — exactly like assembleEntry
  //      does for chunk vectors), so N notes cost ONE extra ONNX pass, not N
  //   4) per entry: relevance(c) = cosine(candidateVec, entry.meanVector)
  //   5) MMR (lambda 0.6) picks 2–3 diverse, relevant phrases
  //   6) assemble into a label landing in the 3–7-word budget; fall back to basename.
  // Runs ONLY at index/build time (never on rank() or render). On any failure or a
  // too-short note it leaves a basename fallback so a card never shows a blank line.
  private async computeSummaryLabels(entries: IndexEntry[]): Promise<void> {
    if (!this.options.showSummary || entries.length === 0) return;

    // Phase 1: per-entry candidate generation (CPU only, no model). Entries with no
    // usable text/candidates get their basename fallback now and are dropped from the
    // embed batch; the rest contribute a flattened slice of candidate surfaces.
    const pending: { entry: IndexEntry; candidates: Candidate[] }[] = [];
    const allSurfaces: string[] = [];
    const offsets: number[] = [0];
    for (const entry of entries) {
      const texts = entry.chunkTexts;
      // No persisted text (empty/stub note): leave the label EMPTY so the card shows
      // no line rather than echoing the title (already the card name). The stub's
      // meaning is carried by the title-grounded score in biMax(), not a label.
      if (!texts || texts.length === 0) {
        entry.summaryLabel = "";
        continue;
      }
      // Candidate phrases from the body chunks (skip the title chunk as a source —
      // including it would just echo the title).
      const source = texts.filter((_, i) => i !== TITLE_CHUNK_INDEX).join("\n");
      const candidates = generateCandidates(source);
      if (candidates.length === 0) {
        entry.summaryLabel = "";
        continue;
      }
      pending.push({ entry, candidates });
      for (const c of candidates) allSurfaces.push(c.surface);
      offsets.push(allSurfaces.length);
    }

    if (pending.length === 0) return;

    // Phase 2: ONE embed pass over every candidate in the batch (prefix-free for the
    // paraphrase model; "passage" is a no-op there but keeps intent explicit). If the
    // pass throws or its length is wrong, every pending entry falls back to its
    // basename — a label is never left undefined.
    let vecs: Float32Array[];
    try {
      vecs = await this.engine.embedBatch(allSurfaces, "passage");
    } catch (e) {
      console.warn("[related-notes] keyphrase label batch failed", e);
      for (const { entry } of pending) entry.summaryLabel = "";
      return;
    }
    if (vecs.length !== allSurfaces.length) {
      for (const { entry } of pending) entry.summaryLabel = "";
      return;
    }

    // Phase 3: regroup the flat vector list per entry and select each label (CPU only).
    for (let p = 0; p < pending.length; p++) {
      const { entry, candidates } = pending[p];
      const slice = vecs.slice(offsets[p], offsets[p + 1]);
      const label = selectLabel(candidates, slice, entry.meanVector);
      // Drop an empty label or one that just echoes the title — the view then shows
      // the snippet (or nothing) instead of repeating the card name.
      const echoesTitle =
        label.toLowerCase() === baseNameFromPath(entry.path).toLowerCase();
      entry.summaryLabel = label.length > 0 && !echoesTitle ? label : "";
    }
  }
}

// =============================================================================
// helpers (module-scope, no `this`)
// =============================================================================

// Normalize a tag for comparison: lower-cased, leading '#' stripped. Nested tags
// (#a/b) are kept whole — a/b and a are distinct, the conservative choice.
function normalizeTag(tag: string): string {
  return tag.replace(/^#/, "").toLowerCase();
}

// Merge frontmatter `tags` (string or string[]) into a tag set.
function addFrontmatterTags(into: Set<string>, raw: unknown): void {
  if (typeof raw === "string") {
    for (const t of raw.split(/[\s,]+/)) {
      const n = normalizeTag(t);
      if (n.length > 0) into.add(n);
    }
  } else if (Array.isArray(raw)) {
    for (const t of raw) {
      if (typeof t === "string") {
        const n = normalizeTag(t);
        if (n.length > 0) into.add(n);
      }
    }
  }
}

// Count of shared frontmatter key/value pairs (excluding `tags`), normalized into
// [0,1] by the smaller frontmatter's key count. Only primitive values are compared
// (objects/arrays are skipped to keep this cheap and interpretable).
function sharedFrontmatter(
  a: Record<string, unknown> | undefined,
  b: Record<string, unknown> | undefined,
): number {
  if (!a || !b) return 0;
  const aKeys = Object.keys(a).filter((k) => k !== "tags" && k !== "position");
  const bKeys = new Set(Object.keys(b).filter((k) => k !== "tags" && k !== "position"));
  if (aKeys.length === 0 || bKeys.size === 0) return 0;
  let shared = 0;
  for (const k of aKeys) {
    if (!bKeys.has(k)) continue;
    const av = a[k];
    const bv = b[k];
    if (isPrimitive(av) && isPrimitive(bv) && av === bv) shared++;
  }
  const denom = Math.min(aKeys.length, bKeys.size);
  return denom > 0 ? shared / denom : 0;
}

function isPrimitive(v: unknown): v is string | number | boolean {
  return (
    typeof v === "string" || typeof v === "number" || typeof v === "boolean"
  );
}

// Truncate to `max` chars on a word boundary, appending an ellipsis when cut.
function truncateAtWord(text: string, max: number): string {
  const t = text.trim();
  if (t.length <= max) return t;
  const slice = t.slice(0, max);
  const lastSpace = slice.lastIndexOf(" ");
  const cut = lastSpace > max * 0.6 ? slice.slice(0, lastSpace) : slice;
  return `${cut.trimEnd()}…`;
}

// Basename (no directory, no .md) from a vault path. Used as the always-available
// label fallback for too-short notes, mirroring the view's existing basename fallback.
function baseNameFromPath(path: string): string {
  const slash = path.lastIndexOf("/");
  const name = slash >= 0 ? path.slice(slash + 1) : path;
  return name.replace(/\.md$/i, "");
}

// =============================================================================
// KeyBERT-style keyphrase labelling (module-scope, no model state)
// =============================================================================

// A candidate phrase: its original-cased SURFACE form (shown to the user), a
// lower-cased KEY for dedupe, the WORDS count, and the index of its first occurrence
// in the source (for left-to-right ordering of the final label).
interface Candidate {
  surface: string;
  key: string;
  words: number;
  pos: number;
}

// Inlined multilingual (German + English) stopword set. A candidate is dropped when
// its FIRST or LAST token is a stopword — interior stopwords are kept ("theory of
// mind", "Gesetz der großen Zahlen"). No dependency: a hardcoded Set is enough.
const STOPWORDS = new Set<string>([
  // --- English function words ---
  "the", "a", "an", "and", "or", "but", "nor", "so", "yet", "of", "to", "in",
  "on", "at", "by", "for", "with", "about", "against", "between", "into",
  "through", "during", "before", "after", "above", "below", "from", "up",
  "down", "out", "off", "over", "under", "again", "further", "then", "once",
  "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
  "few", "more", "most", "other", "some", "such", "no", "not", "only", "own",
  "same", "than", "too", "very", "can", "will", "just", "should", "now", "is",
  "are", "was", "were", "be", "been", "being", "am", "has", "have", "had",
  "having", "do", "does", "did", "doing", "would", "could", "shall", "may",
  "might", "must", "this", "that", "these", "those", "it", "its", "he", "she",
  "they", "them", "his", "her", "their", "our", "your", "my", "we", "you", "i",
  "as", "if", "because", "while", "which", "who", "whom", "what", "also", "via",
  "etc", "e", "g", "ie", "vs", "per", "using", "use", "used", "one", "two",
  // --- German function words ---
  "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
  "einer", "eines", "und", "oder", "aber", "doch", "sondern", "denn", "weil",
  "dass", "ob", "wenn", "als", "wie", "wo", "warum", "weshalb", "ist", "sind",
  "war", "waren", "sein", "bin", "bist", "seid", "hat", "haben", "hatte",
  "hatten", "wird", "werden", "wurde", "wurden", "worden", "kann", "können",
  "muss", "müssen", "soll", "sollen", "darf", "dürfen", "mag", "möchte", "zu",
  "von", "mit", "bei", "nach", "aus", "auf", "für", "an", "am", "im", "in",
  "ins", "um", "über", "unter", "vor", "hinter", "neben", "zwischen", "durch",
  "gegen", "ohne", "bis", "seit", "während", "wegen", "trotz", "nicht", "kein",
  "keine", "keinen", "auch", "nur", "noch", "schon", "sehr", "mehr", "viel",
  "viele", "alle", "alles", "man", "es", "er", "sie", "wir", "ihr", "ich",
  "du", "mich", "dich", "sich", "uns", "euch", "ihm", "ihn", "ihnen", "mein",
  "dein", "sein", "unser", "euer", "dieser", "diese", "dieses", "diesem",
  "diesen", "jener", "jene", "welche", "welcher", "welches", "etwa", "also",
  "dann", "hier", "dort", "da", "so", "zum", "zur", "beim", "vom", "ja", "nein",
]);

// True for a token that should never anchor (start/end) a candidate phrase: a
// stopword, a pure-number, or a too-short fragment (<=2 chars, mirroring the existing
// significantWords filter). Comparison is on the lower-cased form.
function isWeakAnchor(tokenLower: string): boolean {
  if (tokenLower.length <= 2) return true;
  if (/^\p{N}+$/u.test(tokenLower)) return true;
  return STOPWORDS.has(tokenLower);
}

// One token of the source, carrying its surface form, lower-cased form, and whether
// it looks like a German content noun (mid-sentence Capitalized word — a cheap,
// tagger-free topic signal that we never stopword-filter and slightly prefer).
interface Token {
  surface: string;
  lower: string;
  contentNoun: boolean;
}

// Tokenize one sentence into Tokens, flagging mid-sentence Capitalized words (likely
// German nouns / proper nouns) — the first token of a sentence is excluded from that
// flag since sentence-initial capitalization is not a noun signal.
function tokenizeSentence(sentence: string): Token[] {
  const raw = sentence.split(/[^\p{L}\p{N}]+/u).filter((w) => w.length > 0);
  return raw.map((w, i) => {
    const first = w[0] ?? "";
    const capitalized = first !== first.toLowerCase() && i > 0;
    return { surface: w, lower: w.toLowerCase(), contentNoun: capitalized };
  });
}

// Generate deduped 1–3-gram candidate phrases from cleaned source text. Phrases never
// cross a sentence boundary; their first/last token must be a strong anchor (not a
// stopword / number / tiny fragment); interior stopwords are allowed. Candidates are
// capped to MAX_CANDIDATES, preferring longer phrases and ones containing a likely
// content noun (the German-noun signal) so the cheap embed pass spends its budget on
// the most topical spans. Surface casing of the FIRST occurrence is preserved.
function generateCandidates(source: string): Candidate[] {
  const byKey = new Map<string, Candidate & { nounScore: number }>();
  let globalPos = 0;

  const sentences = splitSentences(source);
  for (const sentence of sentences) {
    const tokens = tokenizeSentence(sentence);
    for (let i = 0; i < tokens.length; i++) {
      // Skip starting a phrase on a weak anchor.
      if (isWeakAnchor(tokens[i].lower)) {
        globalPos++;
        continue;
      }
      for (let n = 1; n <= MAX_PHRASE_WORDS && i + n <= tokens.length; n++) {
        const span = tokens.slice(i, i + n);
        const last = span[span.length - 1];
        // Last token must also be a strong anchor.
        if (isWeakAnchor(last.lower)) continue;
        const surface = span.map((t) => t.surface).join(" ");
        const key = span.map((t) => t.lower).join(" ");
        if (byKey.has(key)) continue;
        const nounScore = span.reduce((s, t) => s + (t.contentNoun ? 1 : 0), 0);
        byKey.set(key, {
          surface,
          key,
          words: n,
          pos: globalPos,
          nounScore,
        });
      }
      globalPos++;
    }
  }

  const all = Array.from(byKey.values());
  // Prefer phrases that look topical first: more content nouns, then longer, then
  // earlier — then cap so the embed pass stays one cheap call.
  all.sort(
    (a, b) =>
      b.nounScore - a.nounScore || b.words - a.words || a.pos - b.pos,
  );
  return all.slice(0, MAX_CANDIDATES).map((c) => ({
    surface: c.surface,
    key: c.key,
    words: c.words,
    pos: c.pos,
  }));
}

// Rank candidates by cosine to the note mean (the KeyBERT document anchor), pick
// 2–3 diverse phrases with MMR, then assemble them into a tight label. The final
// word count is held inside [TARGET_LABEL_WORDS_MIN, TARGET_LABEL_WORDS_MAX]: the
// assembly skips any phrase that would overflow the max (three trigrams cap at 6,
// not 9), and tops a collapsed-to-one-word pick back up toward the min from the
// most-relevant remaining phrases — though a genuinely single-topic note may still
// land on a tight 1-word label. `meanVector` and each candidate vector are already
// L2-normalized, so cosine is a dot product (reuse cosineSimilarity). Returns ""
// only when nothing survives.
function selectLabel(
  candidates: Candidate[],
  vecs: Float32Array[],
  meanVector: Float32Array,
): string {
  const n = candidates.length;
  if (n === 0) return "";

  const relevance = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    relevance[i] = cosineSimilarity(vecs[i], meanVector);
  }

  // MMR selection: first pick = argmax relevance; each next pick maximizes
  // lambda*relevance - (1-lambda)*max similarity to anything already picked.
  const selected: number[] = [];
  const remaining = new Set<number>();
  for (let i = 0; i < n; i++) remaining.add(i);

  while (selected.length < MAX_LABEL_PHRASES && remaining.size > 0) {
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (const i of remaining) {
      let redundancy = 0;
      for (const s of selected) {
        const sim = cosineSimilarity(vecs[i], vecs[s]);
        if (sim > redundancy) redundancy = sim;
      }
      const mmr =
        selected.length === 0
          ? relevance[i]
          : MMR_LAMBDA * relevance[i] - (1 - MMR_LAMBDA) * redundancy;
      if (mmr > bestScore) {
        bestScore = mmr;
        bestIdx = i;
      }
    }
    if (bestIdx < 0) break;
    selected.push(bestIdx);
    remaining.delete(bestIdx);

    // Stop early once the assembled label already covers the word budget so we don't
    // pad a clean multi-word phrase with extra unigrams.
    const wordsSoFar = selected.reduce((w, i) => w + candidates[i].words, 0);
    if (wordsSoFar >= TARGET_LABEL_WORDS_MIN && candidates[bestIdx].words >= 2) {
      if (wordsSoFar >= TARGET_LABEL_WORDS_MAX) break;
    }
  }

  if (selected.length === 0) return "";

  // Drop a pick whose words are wholly contained (as a contiguous run) in an already-
  // kept pick, keeping the longer phrase. Compared on whitespace-padded keys so the
  // match is on whole words, never a mid-word substring ("art" ⊄ "smart"). The
  // subsumption test is reused below when topping the label back up.
  const subsumedBy = (i: number, against: number[]): boolean => {
    const ki = ` ${candidates[i].key} `;
    return against.some((j) => {
      const kj = ` ${candidates[j].key} `;
      return kj.includes(ki) || ki.includes(kj);
    });
  };
  const kept: number[] = [];
  for (const i of selected) {
    if (!subsumedBy(i, kept)) kept.push(i);
  }

  // The collapse can leave a single short pick (all MMR picks folded into one unigram).
  // If the kept words fall below the floor, append the most-relevant remaining
  // non-subsumed candidate(s) so the label reaches TARGET_LABEL_WORDS_MIN where the
  // note has the material for it. (A genuinely one-topic note may still end at 1 word —
  // that is an acceptable tight label, not a bug.)
  let keptWords = kept.reduce((w, i) => w + candidates[i].words, 0);
  if (keptWords < TARGET_LABEL_WORDS_MIN) {
    const order = Array.from({ length: candidates.length }, (_, i) => i)
      .filter((i) => !kept.includes(i))
      .sort((a, b) => relevance[b] - relevance[a]);
    for (const i of order) {
      if (keptWords >= TARGET_LABEL_WORDS_MIN) break;
      if (keptWords + candidates[i].words > TARGET_LABEL_WORDS_MAX) continue;
      if (subsumedBy(i, kept)) continue;
      kept.push(i);
      keptWords += candidates[i].words;
    }
  }

  // Order left-to-right by original position, then assemble, never letting the running
  // word count exceed the budget: PEEK each phrase's length and skip one that would
  // push past TARGET_LABEL_WORDS_MAX (so e.g. three trigrams cap at 6 words, not 9).
  kept.sort((a, b) => candidates[a].pos - candidates[b].pos);
  const parts: string[] = [];
  let words = 0;
  for (const i of kept) {
    if (words >= TARGET_LABEL_WORDS_MAX) break;
    if (words + candidates[i].words > TARGET_LABEL_WORDS_MAX) continue;
    parts.push(candidates[i].surface);
    words += candidates[i].words;
  }
  // Guard: if every kept phrase was longer than the whole budget (a lone >7-word
  // phrase can't happen given MAX_PHRASE_WORDS=3, but be defensive), keep the first.
  if (parts.length === 0 && kept.length > 0) {
    parts.push(candidates[kept[0]].surface);
  }
  // Single multi-word phrase reads as a phrase; multiple disjoint picks read better
  // separated with a middot so they aren't misread as one run-on phrase.
  const label = parts.length <= 1 ? parts.join(" ") : parts.join(" · ");
  return label.trim();
}
