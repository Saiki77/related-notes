import { App, TFile, Notice, normalizePath, debounce, type Debouncer } from "obsidian";
import { EmbeddingEngine, type ProgressCallback } from "./embeddings";
import { yieldToUI } from "./async-yield";
import {
  cosineSimilarity,
  meanOf,
  dotRow,
  computeCentroid,
  centerVector,
  centerChunksInPlace,
  quantizeChunksRaw,
  dequantizeChunksRaw,
  serializeIndex,
  serializeManifest,
  deserializeIndex,
  type StoredIndexHeader,
  type SerializableEntry,
} from "./vector-math";

// =============================================================================
// SCHEMA / version constant. Bumped 1 -> 2 for the multi-vector layout (mean vector
// PLUS a capped set of int8-quantized chunk vectors). Bumped 2 -> 3 for the keyphrase
// summary LABEL (KeyBERT-style topic label persisted on the entry). Bumped 3 -> 4 for
// the BINARY index format: the int8 chunk buffers + fp32 means now live in a single
// binary blob (index.bin) referenced by byte offsets from a small JSON manifest
// (index.json), instead of megabytes of base64-in-JSON. The bump forces ONE clean
// rebuild on upgrade — an old v3 single-file index.json parses with version !== 4 (or
// has no companion index.bin) and is detected as stale, deleted, and rebuilt. Any
// index written for a different model/dims, or a different quantization/text-
// persistence policy, is likewise invalidated. No silent half-migration.
//
// NOTE: chunk vectors are now kept in RAM as int8 (the quantized buffer + per-row
// scales) and dequantized to fp32 LAZILY — only when a note enters the Stage-2
// shortlist — behind a small LRU cache. The dequant math (dequantizeChunksRaw) is
// byte-for-byte the old load-time dequant, so ranking results are identical; only
// memory + load time change.
// =============================================================================
const INDEX_VERSION = 4 as const;

// The small JSON manifest (header + per-entry metadata + scales + chunkTexts) and
// the binary blob (all fp32 means + all int8 chunk buffers). Written/renamed via
// matching .tmp files for crash-safety.
const STORE_FILE = "index.json";
const BIN_FILE = "index.bin";
const JSON_TMP_FILE = "index.json.tmp";
const BIN_TMP_FILE = "index.bin.tmp";
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

// CONTENT CONFIDENCE. Short/empty notes embed to a non-distinctive vector that sits
// near the corpus centroid (embedding-space anisotropy), so they score a moderate
// ~0.5 cosine against EVERY note — e.g. a math stub "Ableiten" surfacing at 52% under
// an unrelated security essay. We have low confidence in that similarity, so the
// SEMANTIC score is scaled by how much real BODY text the note has: full trust at
// CONFIDENT_BODY_CHARS, down to MIN_CONTENT_CONFIDENCE for a title-only note. Applied
// to the semantic part ONLY, so a directly-linked / shared-tag stub still surfaces via
// its structural boost. (When chunkTexts aren't persisted — summaries off — body size
// is approximated from the body chunk count.)
const CONFIDENT_BODY_CHARS = 220;
const MIN_CONTENT_CONFIDENCE = 0.4;
const APPROX_CHARS_PER_CHUNK = 110;

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
// When mean-centering is active the cosine distribution is shifted down (unrelated
// notes go to ~0/negative, related stay positive), so the coarse floor is lower and
// slightly negative to preserve recall — the real cut is the Stage-2 minSimilarity.
const COARSE_FLOOR_CENTERED = -0.1;
// Glow context gate: a content-rich target only glows when the active note's centered
// (topical) similarity to it clears this floor — so a common word like "analysis" only
// glows the math "Analysis" note when the note is actually on that topic.
const GLOW_CONTEXT_FLOOR = 0.15;

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

// --- lazy keyphrase-label drainer knobs -------------------------------------
// First getSummary() demand for a note schedules its label; the drainer coalesces
// pending paths and computes them in batches. DRAIN_BATCH labels per pass keeps the
// extra ONNX work bounded; the debounce mirrors main.ts's flushDirty/getSnippet
// coalescing so a flurry of first-paint cards collapses into a few drains.
const LABEL_DRAIN_DEBOUNCE_MS = 250;
const LABEL_DRAIN_BATCH = 8;

// --- scale-aware adaptive chunk cap -----------------------------------------
// The index/RAM/rank cost is roughly linear in chunkCount and the Stage-2 cost
// quadratic in it, so for very large vaults we clamp the effective per-note chunk
// cap DOWN from whatever the user configured. Normal vaults (<= 2000 notes) keep
// the configured cap exactly — byte-for-byte unchanged behavior. This only ever
// LOWERS the cap automatically; nothing to configure.
const ADAPTIVE_CHUNK_TIERS: { maxNotes: number; cap: number }[] = [
  { maxNotes: 5000, cap: 12 },
  { maxNotes: 10000, cap: 10 },
  { maxNotes: Infinity, cap: 8 },
];
const ADAPTIVE_CHUNK_FLOOR_NOTES = 2000;

// --- LRU dequant cache floor ------------------------------------------------
// The fp32 dequant cache is sized 3x the shortlist width (active note + current
// shortlist + the previous switch's shortlist stay warm), clamped to this floor so
// it never collapses when the user lowers topK/shortlistSize.
const DEQUANT_CACHE_FLOOR = 256;


// In-memory entry. The chunk vectors are kept as int8 in RAM (`chunkBytes`, length
// chunkCount*dims, plus one fp32 `scales` per row) — ~4x smaller than fp32 — and
// dequantized to a contiguous Float32Array LAZILY (only when the note enters the
// Stage-2 shortlist) behind the DequantCache LRU. The fp32 mean drives Stage 1 and
// stays in RAM. chunkTexts are only present when summaries are enabled.
//
// After deserialize, `meanVector` is a Float32Array VIEW into the shared on-disk
// blob and `chunkBytes` an Int8Array VIEW into it (the int8 at-rest footprint);
// freshly-built entries own standalone typed arrays. Both are structurally a
// SerializableEntry, so persist() can hand them straight to serializeIndex.
interface IndexEntry {
  path: string;
  mtime: number;
  dims: number;
  chunkCount: number;
  meanVector: Float32Array;
  chunkBytes: Int8Array; // length == chunkCount * dims, int8-quantized rows
  scales: number[]; // length == chunkCount, one fp32 scale per row
  chunkTexts?: string[];
  // The note's topic LABEL (3–7 words). With LAZY labels it is computed on FIRST
  // getSummary() demand (not at build), persisted, and re-rendered when ready.
  // Present only when summaries are on. Undefined until first computed.
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
// DequantCache — lazy int8 -> fp32 chunk dequant behind an LRU
// =============================================================================

// Dequantizing every note's chunks at load time costs ~4x the steady-state RAM
// (fp32 vs int8) and a load-time CPU pass. Instead chunks live in RAM as int8 and
// are dequantized to a contiguous fp32 buffer ON DEMAND — only when a note enters
// the Stage-2 shortlist — through this small LRU. The dequant math is byte-for-byte
// the old load-time dequant (dequantizeChunksRaw), so the fp32 buffers BiMax sees
// are bit-identical: ranking results are unchanged, only WHEN dequant runs moved.
//
// A JS Map preserves insertion order, which we use directly as LRU order: a get()
// hit re-inserts (moves to MRU); on overflow we evict the first (LRU) key.
class DequantCache {
  private cache = new Map<string, Float32Array>();
  private cap: number;
  // One-time guard so a genuine length/scale mismatch (the null path below) is
  // observable in the console rather than silently scoring the note 0 forever.
  private warnedNull = false;
  // Corpus centroid (anisotropy correction). When set, dequantized chunk rows are
  // mean-centered + re-normalized before caching, so BiMax consumes centered vectors.
  private centroid: Float32Array | null = null;

  constructor(cap: number) {
    this.cap = Math.max(DEQUANT_CACHE_FLOOR, cap);
  }

  // Set the centroid used to center chunks. A change invalidates the cache so no
  // stale raw/old-centroid buffers survive (cheap: chunks re-dequant on next demand).
  setCentroid(centroid: Float32Array | null): void {
    if (this.centroid === centroid) return;
    this.centroid = centroid;
    this.cache.clear();
  }

  // Return the fp32 chunk buffer for an entry, dequantizing + caching on a miss.
  // On a hit the key is moved to MRU so the active note + current shortlist stay warm.
  get(entry: IndexEntry): Float32Array {
    const hit = this.cache.get(entry.path);
    if (hit) {
      this.cache.delete(entry.path);
      this.cache.set(entry.path, hit);
      return hit;
    }
    const dequantized = dequantizeChunksRaw(
      entry.chunkBytes,
      entry.scales,
      entry.chunkCount,
      entry.dims,
    );
    if (dequantized === null && !this.warnedNull) {
      // Should be unreachable — entries are validated on load()/build() — so a hit
      // here means a real invariant break (int8 buffer / scales length mismatch).
      // Warn once; the note still scores 0 via the zero-buffer fallback (no crash).
      this.warnedNull = true;
      console.warn(
        "[related-notes] chunk dequant returned null (length/scale mismatch); the affected note will score 0. This indicates a corrupt index entry — a manual reindex should clear it.",
      );
    }
    const f = dequantized ?? new Float32Array(entry.chunkCount * entry.dims);
    // Mean-center the rows (anisotropy correction) so BiMax scores topical, not
    // baseline, similarity. No-op when no centroid is set (e.g. an empty corpus).
    if (this.centroid && dequantized && this.centroid.length === entry.dims) {
      centerChunksInPlace(f, entry.chunkCount, entry.dims, this.centroid);
    }
    this.cache.set(entry.path, f);
    if (this.cache.size > this.cap) {
      const lru = this.cache.keys().next().value;
      if (lru !== undefined) this.cache.delete(lru);
    }
    return f;
  }

  // Evict one path's fp32 buffer (on re-embed/remove/rename) so ranking never reads
  // an outdated vector.
  delete(path: string): void {
    this.cache.delete(path);
  }

  clear(): void {
    this.cache.clear();
  }

  // Resize on an options change. Lowering the cap evicts the oldest entries down to
  // the new (floored) cap; raising it just lets the cache grow.
  setCap(cap: number): void {
    this.cap = Math.max(DEQUANT_CACHE_FLOOR, cap);
    while (this.cache.size > this.cap) {
      const lru = this.cache.keys().next().value;
      if (lru === undefined) break;
      this.cache.delete(lru);
    }
  }
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

  // LAZY chunk dequant: chunks live in RAM as int8; this LRU holds the fp32 buffers
  // for notes currently being ranked (Stage-2 shortlist + active note). Sized 3x the
  // shortlist width in updateOptions(); biMax() resolves chunks through it.
  private dequant: DequantCache;

  // LAZY keyphrase labels. A note's label is computed on FIRST getSummary() demand,
  // not at build time. `labelQueue` holds paths pending a (debounced, batched)
  // compute; `labelDone` holds paths already attempted (so a note whose label
  // computes to "" — e.g. a stub — never re-queues forever). The drainer reuses
  // computeSummaryLabels() VERBATIM and persists ONLY the manifest (no blob rewrite).
  private labelQueue = new Set<string>();
  private labelDone = new Set<string>();
  private debouncedDrainLabels: Debouncer<[], void>;

  // The store has no view reference; the plugin installs this so a resolved label
  // can ask the view to re-render the affected card(s). requestRender is itself
  // debounced in the view, so N resolving labels collapse into one render pass.
  private renderHook?: () => void;

  // Set while a VECTOR-MUTATING op (incremental embed/remove/rename) is between its
  // mutation of `this.entries` and the matching full persist(). During that window
  // the live entries and the on-disk blob disagree, so the label drainer must NOT
  // run persistManifestOnly() — its manifest offsets/totalBytes would describe the
  // NEW entry set while the blob on disk is still the OLD one. (load() self-heals a
  // skew via the totalBytes check, but this avoids a needless rebuild after a crash
  // in that window.) When set, drainLabels() defers and lets the next full persist
  // carry the labels.
  private vectorWriteInFlight = false;

  // Serializes ALL persistence. persist(), persistManifestOnly(), and every
  // incremental save path funnel through persistSerial(), which chains onto this
  // promise so only ONE write+swap sequence runs at a time. Previously a label
  // drain's persistManifestOnly() could race a concurrent persist(): both wrote the
  // SAME index.json.tmp, the first swapInto renamed it away, and the second
  // swapInto's rename(tmp,dest) threw ENOENT. Chaining eliminates the shared-tmp race.
  private persistChain: Promise<void> = Promise.resolve();

  // Anisotropy correction: the corpus centroid + per-note CENTERED mean vectors,
  // recomputed whenever `entries` changes (see recomputeCentroid). The chunk-level
  // centering lives in DequantCache (fed the same centroid). null = centering off
  // (empty corpus), in which case the raw vectors are used unchanged.
  private centroid: Float32Array | null = null;
  private centeredMeans = new Map<string, Float32Array>();

  // Scale-aware effective chunk cap, set at the start of build() from the vault
  // size. Undefined until the first build, where the getter falls back to the base.
  private effectiveMaxChunks: number | undefined;

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
    this.dequant = new DequantCache(this.cacheCapFor(options));
    this.debouncedDrainLabels = debounce(
      () => void this.drainLabels(),
      LABEL_DRAIN_DEBOUNCE_MS,
      false,
    );
  }

  // The plugin installs a re-render callback after constructing the store, so a
  // lazily-computed label can refresh its card without the store holding a view ref.
  setRenderHook(fn: () => void): void {
    this.renderHook = fn;
  }

  // The shortlist Stage-1 -> Stage-2 funnel width (also the slice width in rank()).
  private shortlistWidth(o: IndexStoreOptions): number {
    return Math.max(o.topK * 4, o.shortlistSize || DEFAULT_SHORTLIST);
  }

  // Dequant LRU capacity: 3x the shortlist width keeps the active note + the current
  // shortlist + the previous switch's shortlist warm, floored so it never collapses.
  private cacheCapFor(o: IndexStoreOptions): number {
    return Math.max(DEQUANT_CACHE_FLOOR, this.shortlistWidth(o) * 3);
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
    // Re-size the dequant LRU when topK/shortlistSize change (clamped to the floor).
    this.dequant.setCap(this.cacheCapFor(options));
  }

  // --- suggester support (Feature B) -----------------------------------------
  // Embed an arbitrary context string with the SAME engine the index uses, so the
  // smart `[[` suggester can rank notes by semantic relevance to the cursor's
  // surrounding text without reaching into the engine privately. Returns a
  // normalized Float32Array (or throws if the model can't load — the caller falls
  // back to recency order).
  async embedQuery(text: string): Promise<Float32Array> {
    return this.engine.embed(text, "query");
  }

  // Rank every indexed note by cosine of its MEAN vector to a context vector,
  // reusing the Stage-1 coarse-shortlist loop. Returns the top `limit` as
  // { file, semantic } pairs. Cheap: O(notes * dims). The suggester blends this
  // `semantic` ([0,1]) with a fuzzy text score when the user has typed a query.
  //
  // `excludePath` skips the active note itself the way rank() does — a context
  // vector embedded from text INSIDE the active note is most similar to that
  // note's own mean vector, so without this the active note ranks #1 and the
  // suggester would offer a `[[Self]]` link.
  rankForContext(
    vec: Float32Array,
    limit: number,
    excludePath?: string,
  ): { file: TFile; semantic: number }[] {
    const scored: { entry: IndexEntry; semantic: number }[] = [];
    for (const entry of this.entries.values()) {
      if (excludePath !== undefined && entry.path === excludePath) continue;
      if (entry.meanVector.length !== vec.length) continue;
      scored.push({ entry, semantic: cosineSimilarity(vec, entry.meanVector) });
    }
    scored.sort((a, b) => b.semantic - a.semantic);
    const out: { file: TFile; semantic: number }[] = [];
    for (const { entry, semantic } of scored) {
      const file = this.app.vault.getAbstractFileByPath(entry.path);
      if (file instanceof TFile) out.push({ file, semantic });
      if (out.length >= limit) break;
    }
    return out;
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
    // A different model invalidates every dequantized buffer + every pending/done
    // label (they were computed against the old model's mean vectors).
    this.dequant.clear();
    this.labelQueue.clear();
    this.labelDone.clear();
    this.effectiveMaxChunks = undefined;
    this.setProgress({ status: "idle", done: 0, total: 0, message: undefined });
  }

  // Base chunk cap from options. The EFFECTIVE cap (what chunkNote actually uses) is
  // this clamped down for very large vaults — see effectiveMaxChunks / build().
  private get baseMaxChunks(): number {
    return Math.max(1, this.options.maxChunks || DEFAULT_MAX_CHUNKS);
  }

  // The cap chunkNote() reads. Before the first build (e.g. an incremental embedFile
  // on startup) effectiveMaxChunks is unset and we fall back to the base cap.
  private get maxChunks(): number {
    return this.effectiveMaxChunks ?? this.baseMaxChunks;
  }

  // SCALE-AWARE: lower the configured cap for very large vaults (cost is ~linear in
  // chunkCount, Stage-2 quadratic). Vaults at/under the floor keep the base cap
  // exactly. Never raises the cap.
  private adaptiveMaxChunks(noteCount: number): number {
    const base = this.baseMaxChunks;
    if (noteCount <= ADAPTIVE_CHUNK_FLOOR_NOTES) return base;
    for (const tier of ADAPTIVE_CHUNK_TIERS) {
      if (noteCount <= tier.maxNotes) return Math.min(base, tier.cap);
    }
    return base;
  }

  // --- persistence -----------------------------------------------------------
  private get jsonPath(): string {
    return normalizePath(`${this.configDir}/${STORE_FILE}`);
  }

  private get binPath(): string {
    return normalizePath(`${this.configDir}/${BIN_FILE}`);
  }

  private get jsonTmpPath(): string {
    return normalizePath(`${this.configDir}/${JSON_TMP_FILE}`);
  }

  private get binTmpPath(): string {
    return normalizePath(`${this.configDir}/${BIN_TMP_FILE}`);
  }

  // Load a persisted index (binary format). Returns false (so the caller triggers a
  // build) when either file is missing, malformed, or was written for a different
  // model/dimension/version — including ANY old single-file (v3 base64-JSON) index,
  // which has version !== 4 and/or no companion index.bin and so forces a clean
  // rebuild on upgrade. A detected-stale state also removes the old artifacts so no
  // orphaned multi-MB base64 file lingers.
  async load(): Promise<boolean> {
    this.setProgress({ status: "loading", done: 0, total: 0 });
    const adapter = this.app.vault.adapter;
    try {
      const hasJson = await adapter.exists(this.jsonPath);
      const hasBin = await adapter.exists(this.binPath);
      // Missing manifest entirely: nothing to load (and clean any stale .tmp).
      if (!hasJson) {
        await this.discardStaleIndex();
        this.setProgress({ status: "idle" });
        return false;
      }
      // Manifest present but no blob (old v3 single-file format, or a crash mid-
      // migration): stale. Delete both + tmp so the next persist writes a clean pair.
      if (!hasBin) {
        await this.discardStaleIndex();
        this.setProgress({ status: "idle" });
        return false;
      }

      const json = await adapter.read(this.jsonPath);
      const header = JSON.parse(json) as StoredIndexHeader;
      if (
        header.version !== INDEX_VERSION ||
        header.modelId !== this.engine.modelId ||
        header.quantized !== true ||
        !header.dims ||
        !Array.isArray(header.entries)
      ) {
        await this.discardStaleIndex();
        this.setProgress({ status: "idle" });
        return false;
      }
      // The summary feature needs persisted chunk text (lazy labels still read
      // chunkTexts on first demand). If the user has it ON but this index was written
      // without it, treat it as stale so the caller re-embeds once.
      if (this.options.showSummary && header.hasChunkText !== true) {
        await this.discardStaleIndex();
        this.setProgress({ status: "idle" });
        return false;
      }

      const blob = await adapter.readBinary(this.binPath);
      if (blob.byteLength !== header.totalBytes) {
        // Half-written blob or manifest/blob skew: rebuild from scratch.
        await this.discardStaleIndex();
        this.setProgress({ status: "idle" });
        return false;
      }

      // Build lazy int8 entries — NO dequant pass. Each entry's meanVector/chunkBytes
      // are views into `blob`, kept alive by the entries themselves.
      const { entries } = deserializeIndex(json, blob);
      const loaded = new Map<string, IndexEntry>();
      for (const e of entries) {
        if (e.dims !== header.dims) continue;
        if (e.meanVector.length !== header.dims) continue;
        loaded.set(e.path, {
          path: e.path,
          mtime: e.mtime,
          dims: e.dims,
          chunkCount: e.chunkCount,
          meanVector: e.meanVector,
          chunkBytes: e.chunkBytes,
          scales: e.scales,
          chunkTexts: e.chunkTexts,
          summaryLabel: e.summaryLabel,
        });
      }
      this.entries = loaded;
      this.dequant.clear();
      this.recomputeCentroid();
      this.labelQueue.clear();
      this.labelDone.clear();
      this.setProgress({ status: "ready", done: loaded.size, total: loaded.size });
      return true;
    } catch (e) {
      console.warn("[related-notes] failed to load index, will rebuild", e);
      // Best-effort cleanup so a corrupt pair doesn't wedge every future load.
      await this.discardStaleIndex().catch(() => undefined);
      this.setProgress({ status: "idle" });
      return false;
    }
  }

  // Remove the on-disk index artifacts (manifest, blob, and any leftover .tmp files,
  // including an old single-file v3 index.json that has no companion blob). Best
  // effort: each remove is guarded so a missing file is not an error.
  private async discardStaleIndex(): Promise<void> {
    const adapter = this.app.vault.adapter;
    for (const p of [
      this.jsonPath,
      this.binPath,
      this.jsonTmpPath,
      this.binTmpPath,
    ]) {
      try {
        if (await adapter.exists(p)) await adapter.remove(p);
      } catch {
        // ignore — a missing/locked stale file must not block a rebuild.
      }
    }
  }

  // Build SerializableEntry[] from the live entries. chunkBytes/scales are already
  // int8 in RAM — no re-quantize. meanVector is fp32 (a standalone array on fresh
  // entries, or a view into the previous blob on loaded ones — serializeIndex copies
  // only each view's own bytes, so a view is safe to pass).
  private serializableEntries(): SerializableEntry[] {
    return Array.from(this.entries.values());
  }

  // Run a persistence operation under the single-writer mutex. Every persist path
  // (full persist + manifest-only + any incremental save) chains here, so two writes
  // can never interleave their write/swap on the shared .tmp paths. Failures are
  // logged (never thrown) so a persist error is not an UNCAUGHT promise rejection;
  // the chain is kept alive (we swallow into .catch) so one failure doesn't poison
  // every later write.
  private persistSerial(op: () => Promise<void>): Promise<void> {
    const run = this.persistChain.then(op).catch((e: unknown) => {
      console.warn("[related-notes] persist failed", e);
    });
    // Advance the chain to this run's settlement (already caught above, so the chain
    // never holds a rejection).
    this.persistChain = run;
    return run;
  }

  // Full persist (mutex-guarded). The actual write+swap lives in persistFull().
  private persist(): Promise<void> {
    return this.persistSerial(() => this.persistFull());
  }

  // Full persist: manifest + blob, crash-safe. Writes both .tmp files, then renames
  // the BLOB into place BEFORE the manifest, so a crash never leaves a manifest
  // pointing at a missing/half-written blob (the worst case is a committed blob with
  // the old manifest — load() validates totalBytes and rebuilds if they skew).
  // ALWAYS call via persist() so it runs under the persistChain mutex.
  private async persistFull(): Promise<void> {
    // Empty vault (everything excluded, or a brand-new vault): there is nothing to
    // rank and no dims to record. Writing a dims=0 / 0-byte index would only make
    // load()'s `!header.dims` gate treat it as stale and rebuild every startup, so
    // skip the write and clear any prior artifacts instead. An empty build is instant.
    if (this.entries.size === 0) {
      await this.discardStaleIndex();
      return;
    }
    const dims = this.firstDims();
    const keepText = this.options.showSummary;
    const { json, blob } = serializeIndex(this.serializableEntries(), {
      modelId: this.engine.modelId,
      dims,
      hasChunkText: keepText,
      version: INDEX_VERSION,
    });
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(this.configDir))) {
      await adapter.mkdir(this.configDir);
    }
    // writeBinary requires a true ArrayBuffer; serializeIndex returns exactly that.
    await adapter.writeBinary(this.binTmpPath, blob);
    await adapter.write(this.jsonTmpPath, json);
    // Swap the BLOB into place before the manifest. The complete new data already
    // lives in the .tmp files; swapInto handles the fact that adapter.rename throws
    // over an existing destination.
    await this.swapInto(this.binTmpPath, this.binPath);
    await this.swapInto(this.jsonTmpPath, this.jsonPath);
  }

  // Atomically replace `dest` with `tmp`. Obsidian's adapter.rename THROWS when the
  // destination already exists ("Destination file already exists!"), so we remove
  // it first. The new bytes are already fully written to `tmp`, so the only failure
  // window is the brief remove→rename gap; load()'s header/totalBytes validation
  // detects and self-heals any resulting skew into a clean rebuild.
  //
  // The persistChain mutex now guarantees no two swaps race on the shared .tmp paths;
  // as defence-in-depth we also no-op gracefully if `tmp` is already gone (e.g. a
  // prior swap consumed it), so a stale call can never throw the old ENOENT.
  private async swapInto(tmp: string, dest: string): Promise<void> {
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(tmp))) return;
    if (await adapter.exists(dest)) await adapter.remove(dest);
    await adapter.rename(tmp, dest);
  }

  // MANIFEST-ONLY persist (mutex-guarded). The actual write+swap lives in
  // persistManifestOnlyInner().
  private persistManifestOnly(): Promise<void> {
    return this.persistSerial(() => this.persistManifestOnlyInner());
  }

  // MANIFEST-ONLY persist for a label-only change. The vectors did not move, so the
  // blob (and every byte offset / totalBytes) is unchanged — rewriting just the
  // manifest is consistent with the on-disk blob and avoids a multi-MB rewrite per
  // label batch. ONLY the label drainer may call this; any vector mutation must go
  // through full persist(). ALWAYS call via persistManifestOnly() so it runs under
  // the persistChain mutex (else it could race a full persist on index.json.tmp).
  private async persistManifestOnlyInner(): Promise<void> {
    if (this.entries.size === 0) return;
    const dims = this.firstDims();
    const keepText = this.options.showSummary;
    // Header-only: same offsets/totalBytes as the live blob, no multi-MB blob copy.
    const json = serializeManifest(this.serializableEntries(), {
      modelId: this.engine.modelId,
      dims,
      hasChunkText: keepText,
      version: INDEX_VERSION,
    });
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(this.configDir))) {
      await adapter.mkdir(this.configDir);
    }
    await adapter.write(this.jsonTmpPath, json);
    await this.swapInto(this.jsonTmpPath, this.jsonPath);
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
      // SCALE-AWARE: fix the effective per-note chunk cap for this build from the
      // vault size (only lowers it for very large vaults; normal vaults unchanged).
      this.effectiveMaxChunks = this.adaptiveMaxChunks(total);
      // A forced full re-embed replaces every vector; drop any stale fp32 dequant +
      // pending labels so ranking/labels never read outdated data mid-rebuild.
      if (force) {
        this.dequant.clear();
        this.labelQueue.clear();
        this.labelDone.clear();
      }
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
              for (let f = 0; f < pendingFiles.length; f++) {
                const { file, chunks } = pendingFiles[f];
                const start = offsets[f];
                const end = offsets[f + 1];
                const entry = this.assembleEntry(file, chunks, vectors, start, end);
                if (entry) {
                  next.set(file.path, entry);
                  this.summaryCache.delete(file.path);
                  // A re-embedded note's old fp32 dequant + label state is stale.
                  this.dequant.delete(file.path);
                  this.labelQueue.delete(file.path);
                  this.labelDone.delete(file.path);
                  embedded++;
                }
              }
              // LAZY LABELS: no keyphrase pass at build time. Labels are computed on
              // first getSummary() demand (halving indexing time); chunkTexts are
              // still persisted (when summaries are on) so the demand-time pass works.
            } catch (e) {
              if (!firstError) firstError = e;
              console.warn("[related-notes] batch embed failed", e);
            }
          }

          done = Math.min(files.length, i + batch.length);
          const pct = total > 0 ? Math.round((done / total) * 100) : 100;
          notice.setMessage(`Related notes: indexing… ${pct}% (${done}/${total})`);
          this.setProgress({ done, total });
          // YIELD a real macrotask so the renderer paints + processes input between
          // batches — a big vault never freezes Obsidian.
          await yieldToUI();
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
      this.recomputeCentroid();
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
  // re-normalized mean + an int8-quantized chunk buffer. Returns null if the slice
  // is empty or dims can't be determined. The fp32 chunk buffer is quantized to int8
  // immediately and DISCARDED, so RAM stays int8 even mid-build. The summaryLabel is
  // left undefined — it is now computed lazily on first getSummary() demand.
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
    // Quantize to int8 NOW and drop the fp32 buffer; it is re-expanded lazily by the
    // DequantCache only if/when this note enters a Stage-2 shortlist.
    const { q, scales } = quantizeChunksRaw(buffer, count, dims);

    return {
      path: file.path,
      mtime: file.stat.mtime,
      dims,
      chunkCount: count,
      meanVector,
      chunkBytes: q,
      scales,
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
      // LAZY LABELS: no keyphrase pass here either — the label is computed on first
      // getSummary() demand. The entry carries chunkTexts (when summaries are on) so
      // that demand-time pass has its source text.
      return this.assembleEntry(file, chunks, vectors, 0, vectors.length);
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
      // Mark the vector-mutation window so a concurrent label drain can't write a
      // manifest that disagrees with the not-yet-rewritten on-disk blob.
      this.vectorWriteInFlight = true;
      try {
        this.entries.set(file.path, entry);
        this.summaryCache.delete(file.path);
        // Evict the stale fp32 dequant + label state so ranking/labels recompute.
        this.dequant.delete(file.path);
        this.labelQueue.delete(file.path);
        this.labelDone.delete(file.path);
        // The corpus shifted by one note; refresh the centroid + centered means
        // (also resets the dequant cache so chunks re-center against the new centroid).
        this.recomputeCentroid();
        this.setProgress({ done: this.entries.size, total: this.entries.size });
        await this.persist();
      } finally {
        this.vectorWriteInFlight = false;
      }
    }
  }

  removeFile(path: string): void {
    this.wordCache.delete(path);
    this.summaryCache.delete(path);
    this.dequant.delete(path);
    this.labelQueue.delete(path);
    this.labelDone.delete(path);
    this.ambiguousBasenames = null;
    if (this.entries.delete(path)) {
      this.recomputeCentroid();
      this.setProgress({ done: this.entries.size, total: this.entries.size });
      // The entry set just changed but the on-disk blob hasn't yet; hold the
      // vector-write flag across the async persist so a label drain can't slip a
      // mismatched manifest in between (cleared whether persist resolves or throws).
      this.vectorWriteInFlight = true;
      void this.persist().finally(() => {
        this.vectorWriteInFlight = false;
      });
    }
  }

  renameFile(oldPath: string, file: TFile): void {
    this.wordCache.delete(oldPath);
    this.summaryCache.delete(oldPath);
    this.dequant.delete(oldPath);
    this.labelQueue.delete(oldPath);
    this.labelDone.delete(oldPath);
    this.ambiguousBasenames = null;
    this.entries.delete(oldPath);
    void this.updateFile(file);
  }

  private async flushPending(onProgress?: ProgressCallback): Promise<void> {
    if (this.pending.size === 0) return;
    const paths = Array.from(this.pending);
    this.pending.clear();
    this.vectorWriteInFlight = true;
    try {
      for (const path of paths) {
        const file = this.app.vault.getAbstractFileByPath(path);
        if (file instanceof TFile) {
          const entry = await this.embedFile(file, onProgress);
          if (entry) {
            this.entries.set(path, entry);
            this.summaryCache.delete(path);
            this.dequant.delete(path);
            this.labelQueue.delete(path);
            this.labelDone.delete(path);
          }
        }
      }
      await this.persist();
    } finally {
      this.vectorWriteInFlight = false;
    }
  }

  // Recompute the corpus centroid + per-note centered means and feed the centroid to
  // the dequant cache (which centers chunks). Call after ANY change to `entries`.
  // Cheap: O(n*dims). On an empty/already-isotropic corpus it is a near-no-op.
  private recomputeCentroid(): void {
    const entries = Array.from(this.entries.values());
    if (entries.length === 0) {
      this.centroid = null;
      this.centeredMeans.clear();
      this.dequant.setCentroid(null);
      return;
    }
    const dims = entries[0].dims;
    const centroid = computeCentroid(
      entries.filter((e) => e.dims === dims).map((e) => e.meanVector),
      dims,
    );
    this.centroid = centroid;
    this.centeredMeans.clear();
    if (centroid) {
      for (const e of entries) {
        if (e.dims === dims && e.meanVector.length === dims) {
          this.centeredMeans.set(
            e.path,
            centerVector(e.meanVector, centroid, dims),
          );
        }
      }
    }
    this.dequant.setCentroid(centroid);
  }

  // The centered mean for a note — falls back to its raw mean when centering is off or
  // not yet computed. Used by Stage 1 and the stub-grounding branch of biMax().
  private centeredMean(entry: IndexEntry): Float32Array {
    return this.centeredMeans.get(entry.path) ?? entry.meanVector;
  }

  // Does a glow for `targetPath` make sense in the active note's CONTEXT? (Used by the
  // inline glow + Link-all + auto-link so a title that is also a common word only
  // links where it fits.) Two paths because an EMPTY target carries no topic to judge:
  //   - Content-rich target: gate by centered (topical) similarity to the active note.
  //   - Stub target: glow only if the active note already LINKS to it — the title alone
  //     (often an ambiguous word) isn't enough to infer the sense.
  // Returns true (don't gate) when either note isn't indexed yet.
  glowAllowed(activePath: string, targetPath: string): boolean {
    const a = this.entries.get(activePath);
    const b = this.entries.get(targetPath);
    if (!a || !b) return true;
    if (b.chunkCount > 1 && b.dims === a.dims) {
      return (
        cosineSimilarity(this.centeredMean(a), this.centeredMean(b)) >=
        GLOW_CONTEXT_FLOOR
      );
    }
    // Stub target: no topic to judge, so glow only when there's an explicit structural
    // tie — a link in EITHER direction or a shared tag. (An isolated empty stub like a
    // bare "Analysis" note has none, so it won't glow off-topic.)
    const mc = this.app.metadataCache;
    const fwd = mc.resolvedLinks[activePath];
    if (fwd && fwd[targetPath]) return true;
    const back = mc.resolvedLinks[targetPath];
    if (back && back[activePath]) return true;
    const ac = mc.getCache(activePath);
    const bc = mc.getCache(targetPath);
    if (ac && bc) {
      const at = new Set<string>();
      for (const t of ac.tags ?? []) at.add(normalizeTag(t.tag));
      if (ac.frontmatter?.tags) addFrontmatterTags(at, ac.frontmatter.tags);
      if (at.size > 0) {
        const bt = new Set<string>();
        for (const t of bc.tags ?? []) bt.add(normalizeTag(t.tag));
        if (bc.frontmatter?.tags) addFrontmatterTags(bt, bc.frontmatter.tags);
        for (const t of bt) if (at.has(t)) return true;
      }
    }
    return false;
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
    // Centered means (anisotropy correction) when available, so the shortlist already
    // reflects topical — not baseline — similarity; the floor shifts down with them.
    const selfMean = this.centeredMean(self);
    const floor = this.centroid ? COARSE_FLOOR_CENTERED : COARSE_FLOOR;
    const shortlist: { entry: IndexEntry; coarse: number }[] = [];
    for (const entry of this.entries.values()) {
      if (entry.path === active.path) continue;
      if (entry.dims !== self.dims) continue;
      const coarse = cosineSimilarity(selfMean, this.centeredMean(entry));
      if (coarse < floor) continue;
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

      const signals = this.structuralSignals(activeStruct, file, entry);
      // Scale the semantic similarity by the candidate's content confidence so a
      // near-empty note can't ride the embedding-space baseline to a spuriously high
      // rank. A DIRECTLY LINKED note is relevant by the user's own connection, so it
      // keeps full semantic weight; only purely-discovered notes get the penalty.
      const confidence = signals.directLink ? 1 : this.contentConfidence(entry);
      const semantic = this.biMax(self, entry) * confidence;
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

  // Confidence in a note's semantic score, from how much BODY text it carries (see
  // CONFIDENT_BODY_CHARS). 1.0 for a note with a real paragraph, down to
  // MIN_CONTENT_CONFIDENCE for a title-only stub — which keeps near-empty notes from
  // riding the embedding-space baseline to a spurious high rank.
  private contentConfidence(entry: IndexEntry): number {
    let bodyChars: number;
    if (entry.chunkTexts && entry.chunkTexts.length > 0) {
      bodyChars = 0;
      for (let i = 0; i < entry.chunkTexts.length; i++) {
        if (i !== TITLE_CHUNK_INDEX) bodyChars += entry.chunkTexts[i].length;
      }
    } else {
      // Summaries off → no persisted text; approximate from the body chunk count.
      bodyChars = Math.max(0, entry.chunkCount - 1) * APPROX_CHARS_PER_CHUNK;
    }
    const t = Math.min(1, bodyChars / CONFIDENT_BODY_CHARS);
    return MIN_CONTENT_CONFIDENCE + (1 - MIN_CONTENT_CONFIDENCE) * t;
  }

  // Symmetric Bidirectional MaxSim over two notes' chunk buffers, with the title
  // chunk (index 0) weighted TITLE_WEIGHT in each per-direction weighted mean.
  //
  // LAZY DEQUANT: chunks live in RAM as int8; here they are expanded to fp32 through
  // the LRU cache. dequantizeChunksRaw is byte-for-byte the old load-time dequant
  // (same per-row L2 renorm), so directionalMax/dotRow consume IDENTICAL fp32 values
  // and the score is bit-identical to the eager-load path — only WHEN dequant runs
  // moved. `self` (the active note) is fetched once per rank() and stays MRU, so its
  // dequant happens once per switch, not per candidate.
  private biMax(a: IndexEntry, b: IndexEntry): number {
    const dims = a.dims;
    if (b.dims !== dims) return 0;
    // Guard the invariant explicitly: a zero-chunk entry (no title chunk) would
    // otherwise feed an empty inner loop. Today chunkCount >= 1 always (the title
    // chunk), but a future stricter chunkNote must not silently emit negatives.
    if (a.chunkCount === 0 || b.chunkCount === 0) return 0;

    const aChunks = this.dequant.get(a);
    const bChunks = this.dequant.get(b);

    // A title-only STUB (chunkCount === 1 — an empty note) is grounded in the other
    // note's OVERALL topic (its L2-normalized mean vector), not the single luckiest
    // chunk match. directionalMax takes a MAX, so a one-word title would otherwise
    // spuriously align with some passage and rank the empty note for an unrelated
    // topic (e.g. a stub "Ableiten" surfacing under a security essay). Its meaning
    // then comes mainly from the title-vs-overall similarity, as it should. Notes
    // with any body (chunkCount >= 2) use full bidirectional MaxSim. mean + chunk
    // rows are both normalized, so the dot IS a cosine; floor at 0 like the max.
    // Use CENTERED means here: aChunks/bChunks are centered by the dequant cache, so
    // the title-vs-mean dot must compare like with like (centered chunk · centered mean).
    const aToB =
      a.chunkCount === 1
        ? Math.max(0, dotRow(aChunks, TITLE_CHUNK_INDEX, this.centeredMean(b), dims))
        : this.directionalMax(aChunks, a.chunkCount, bChunks, b.chunkCount, dims);
    const bToA =
      b.chunkCount === 1
        ? Math.max(0, dotRow(bChunks, TITLE_CHUNK_INDEX, this.centeredMean(a), dims))
        : this.directionalMax(bChunks, b.chunkCount, aChunks, a.chunkCount, dims);
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

  // --- keyphrase summary label (LAZY) ---------------------------------------
  // SYNCHRONOUS + snippet-first: never blocks the render path and never embeds on
  // the sync path. The label is now computed on FIRST demand (not at build time):
  //   1. mtime cache hit -> cached text.
  //   2. label already set (this session or persisted) -> truncate + title-echo
  //      suppress as before, cache, return.
  //   3. label NOT set but the note has chunkTexts and summaries are on and it isn't
  //      already attempted -> schedule a debounced async compute and return ""
  //      immediately (the view falls back to the snippet). The drainer fills the
  //      label and re-renders the card when ready; the next render returns it.
  //   4. summaries off -> "" and schedule nothing.
  getSummary(file: TFile): string {
    const cached = this.summaryCache.get(file.path);
    if (cached && cached.mtime === file.stat.mtime) return cached.text;

    const entry = this.entries.get(file.path);
    const label = entry?.summaryLabel;
    if (label !== undefined && label.length > 0) {
      // Suppress a label that just echoes the title (incl. older basename labels) so
      // we don't repeat the card name. Done at display time so a reload picks it up.
      const text =
        label.toLowerCase() === file.basename.toLowerCase()
          ? ""
          : truncateAtWord(label, SUMMARY_LABEL_CHARS);
      this.summaryCache.set(file.path, { mtime: file.stat.mtime, text });
      return text;
    }

    // Not computed yet: kick off a debounced async compute (when eligible) and return
    // the snippet-fallback empty string immediately. We do NOT cache "" here — only a
    // resolved (possibly empty) label is cached by the drainer / branch above — so a
    // pending note re-checks on the next render and picks up its label once ready.
    if (
      label === undefined &&
      this.options.showSummary &&
      entry?.chunkTexts &&
      entry.chunkTexts.length > 0
    ) {
      this.scheduleLabel(file.path);
    }
    return "";
  }

  // Queue a note for a debounced, batched label computation. Deduped against both
  // pending (labelQueue) and already-attempted (labelDone) sets so a note whose label
  // resolves to "" never re-queues forever.
  private scheduleLabel(path: string): void {
    if (this.labelQueue.has(path) || this.labelDone.has(path)) return;
    // NEVER kick label work during bootstrap (load) or a (re)index. getSummary() is
    // called from renderCard during the initial render that load()/setProgress drives,
    // so without this gate the render -> getSummary -> drainLabels -> embedBatch ->
    // model-init chain fires mid-bootstrap and compounds the freeze. The card still
    // renders immediately with its snippet fallback; once status is ready/idle a later
    // render re-demands the label and it computes normally. We do NOT add to the queue
    // here (so it isn't silently lost-then-marked-done) — the note re-schedules on the
    // next render once labels are allowed.
    if (!this.labelsAllowed()) return;
    this.labelQueue.add(path);
    this.debouncedDrainLabels();
  }

  // Labels may compute only when the index is settled: not building, not loading, and
  // no build/vector-mutation in flight. During "building"/"loading" the model session
  // and embed passes must stay off the render path (see scheduleLabel/drainLabels).
  private labelsAllowed(): boolean {
    if (this.building || this.vectorWriteInFlight) return false;
    const status = this.progress.status;
    return status !== "building" && status !== "loading";
  }

  // Drain up to LABEL_DRAIN_BATCH queued paths: resolve them to live entries, run the
  // EXISTING computeSummaryLabels() over the slice (one extra ONNX pass for the
  // batch), move each from labelQueue to labelDone, persist ONLY the manifest (the
  // vectors did not move), and re-render the affected cards. Re-arms itself while the
  // queue is non-empty so the rest drains in further batches.
  private async drainLabels(): Promise<void> {
    if (this.labelQueue.size === 0) return;
    // DEFER while a build or an incremental vector mutation is in flight: a full
    // build() rewrites every vector (any label we computed against the old entries
    // is stale), and a vector mutation leaves the on-disk blob out of step with the
    // live entries — so a manifest-only write here would describe an entry set the
    // blob doesn't match. Re-arm and let the in-flight op's full persist() carry the
    // labels, or the next drain run them once the window closes. The queue is left
    // intact (we computed nothing), so nothing is lost.
    // Also DEFER while the index is loading (bootstrap) — the initial render that
    // drives load()/setProgress would otherwise trigger model-session creation +
    // embeds during startup. labelsAllowed() folds in building/vectorWriteInFlight too.
    if (!this.labelsAllowed()) {
      this.debouncedDrainLabels();
      return;
    }
    const paths = Array.from(this.labelQueue).slice(0, LABEL_DRAIN_BATCH);
    const slice: IndexEntry[] = [];
    for (const path of paths) {
      const entry = this.entries.get(path);
      // Always move out of the pending set; if the entry vanished (removed/renamed)
      // we still mark it done so it can't wedge the queue.
      this.labelQueue.delete(path);
      this.labelDone.add(path);
      if (entry) slice.push(entry);
    }

    if (slice.length > 0) {
      try {
        await this.computeSummaryLabels(slice);
        // A label-only change: rewrite just the manifest (offsets/blob unchanged).
        await this.persistManifestOnly();
      } catch (e) {
        console.warn("[related-notes] lazy label compute failed", e);
      }
      // Drop any memoized "" so getSummary re-reads the freshly-set label, then ask
      // the view to re-render (debounced there, so a batch collapses to one pass).
      for (const entry of slice) this.summaryCache.delete(entry.path);
      this.renderHook?.();
    }

    // More queued? re-arm the debounced drainer for the next batch.
    if (this.labelQueue.size > 0) this.debouncedDrainLabels();
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
