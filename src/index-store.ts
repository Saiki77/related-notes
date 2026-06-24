import { App, TFile, Notice, normalizePath } from "obsidian";
import {
  EmbeddingEngine,
  cosineSimilarity,
  type ProgressCallback,
} from "./embeddings";

// On-disk shape of one embedded note. The vector is stored as a plain number[]
// (JSON has no typed arrays); `dims` lets us reject a vector whose length no
// longer matches the active model without re-reading it.
interface StoredEntry {
  path: string;
  mtime: number;
  dims: number;
  vector: number[];
}

// Header + body persisted to the plugin config dir. `modelId`/`dims` gate a full
// rebuild: when either changes, every stored vector is stale and discarded.
interface StoredIndex {
  version: 1;
  modelId: string;
  dims: number;
  entries: StoredEntry[];
}

// In-memory entry. Keeps the Float32Array for fast cosine ranking.
interface IndexEntry {
  path: string;
  mtime: number;
  vector: Float32Array;
}

// A single ranked result handed to the view.
export interface RankedNote {
  file: TFile;
  score: number; // cosine similarity in [-1, 1], or a keyword overlap score
  approximate: boolean; // true when produced by the keyword fallback, not embeddings
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
}

type ProgressListener = (p: IndexProgress) => void;

const STORE_FILE = "index.json";
const BATCH_SIZE = 8;

// Strip markdown/frontmatter down to plain text suitable for embedding. This is a
// lightweight cleaner (not a full parser): it removes the YAML frontmatter block,
// code fences, link/image syntax, headings, and emphasis markers, then collapses
// whitespace. Good enough to feed the model the note's actual prose.
export function stripMarkdown(content: string): string {
  let text = content;
  // YAML frontmatter at the very start of the file.
  text = text.replace(/^---\n[\s\S]*?\n---\n?/, " ");
  // Fenced + inline code (drop the code, keep surrounding prose).
  text = text.replace(/```[\s\S]*?```/g, " ");
  text = text.replace(/`[^`]*`/g, " ");
  // Images ![alt](url) and ![[embed]] -> drop entirely.
  text = text.replace(/!\[\[[^\]]*\]\]/g, " ");
  text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, " ");
  // Wikilinks [[Target|Alias]] / [[Target]] -> keep the visible label.
  text = text.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2");
  text = text.replace(/\[\[([^\]]+)\]\]/g, "$1");
  // Markdown links [text](url) -> keep text.
  text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
  // Headings, blockquotes, list bullets at line start.
  text = text.replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, "");
  // Emphasis / strikethrough markers.
  text = text.replace(/[*_~]/g, "");
  // HTML tags.
  text = text.replace(/<[^>]+>/g, " ");
  // Collapse whitespace.
  return text.replace(/\s+/g, " ").trim();
}

// The text we embed for a note: its title plus the first N cleaned body chars.
function buildEmbedText(file: TFile, body: string, charLimit: number): string {
  const cleaned = stripMarkdown(body);
  const limited = cleaned.length > charLimit ? cleaned.slice(0, charLimit) : cleaned;
  return `${file.basename}. ${limited}`.trim();
}

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
    this.setProgress({ status: "idle", done: 0, total: 0, message: undefined });
  }

  // --- persistence -----------------------------------------------------------
  private get storePath(): string {
    return normalizePath(`${this.configDir}/${STORE_FILE}`);
  }

  // Load a persisted index from the plugin config dir. Returns false (so the
  // caller triggers a build) when the file is missing, malformed, or was written
  // for a different model/dimension.
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
        data.version !== 1 ||
        data.modelId !== this.engine.modelId ||
        !Array.isArray(data.entries)
      ) {
        this.setProgress({ status: "idle" });
        return false;
      }
      const loaded = new Map<string, IndexEntry>();
      for (const e of data.entries) {
        if (e.dims !== data.dims || e.vector.length !== data.dims) continue;
        loaded.set(e.path, {
          path: e.path,
          mtime: e.mtime,
          vector: Float32Array.from(e.vector),
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
    const data: StoredIndex = {
      version: 1,
      modelId: this.engine.modelId,
      dims,
      entries: Array.from(this.entries.values()).map((e) => ({
        path: e.path,
        mtime: e.mtime,
        dims: e.vector.length,
        vector: Array.from(e.vector),
      })),
    };
    const adapter = this.app.vault.adapter;
    if (!(await adapter.exists(this.configDir))) {
      await adapter.mkdir(this.configDir);
    }
    await adapter.write(this.storePath, JSON.stringify(data));
  }

  private firstDims(): number {
    for (const e of this.entries.values()) return e.vector.length;
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

  // --- (re)build -------------------------------------------------------------
  // Embed every indexable note in batches, yielding to the event loop between
  // batches so the UI never freezes. Reuses an existing vector when the file's
  // mtime is unchanged, so a "rebuild" after a small edit is cheap. Each batch is
  // embedded in a SINGLE pipeline call for real throughput.
  //
  // If every candidate fails to embed (e.g. the model never loaded — CDN blocked,
  // offline, wasm mismatch) the status flips to "error" with a Notice, instead of
  // silently finishing "ready" with zero vectors and degrading to keyword mode.
  // `force` re-embeds every note even if its mtime is unchanged — used by the
  // manual "Rebuild index" command so it is never a silent no-op (a model/setting
  // change should be reflected even though the files themselves didn't change).
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

          // Reuse unchanged vectors; collect the rest for a single batched embed.
          const toEmbed: { file: TFile; text: string }[] = [];
          for (const file of batch) {
            const existing = this.entries.get(file.path);
            if (!force && existing && existing.mtime === file.stat.mtime) {
              next.set(file.path, existing);
              continue;
            }
            const text = await this.readEmbedText(file);
            if (text) toEmbed.push({ file, text });
          }

          if (toEmbed.length > 0) {
            attempted += toEmbed.length;
            try {
              const vectors = await this.engine.embedBatch(
                toEmbed.map((t) => t.text),
                // Symmetric "query:" prefix on BOTH sides for note-to-note
                // similarity — e5's own guidance for non-retrieval tasks. (No-op
                // for prefix-free paraphrase models.)
                "query",
                onProgress,
              );
              for (let j = 0; j < toEmbed.length; j++) {
                const { file } = toEmbed[j];
                const vector = vectors[j];
                if (vector && vector.length > 0) {
                  next.set(file.path, {
                    path: file.path,
                    mtime: file.stat.mtime,
                    vector,
                  });
                  embedded++;
                }
              }
            } catch (e) {
              // A backend failure fails the whole batch. Remember the first error;
              // keep going so a transient single-batch glitch can't wipe progress,
              // but the all-failed check below will catch a total failure.
              if (!firstError) firstError = e;
              console.warn("[related-notes] batch embed failed", e);
            }
          }

          done = Math.min(files.length, i + batch.length);
          const pct = total > 0 ? Math.round((done / total) * 100) : 100;
          notice.setMessage(`Related notes: indexing… ${pct}% (${done}/${total})`);
          this.setProgress({ done, total });
          // Yield so Obsidian can paint and stay responsive.
          await sleep(0);
        }
      } finally {
        notice.hide();
      }

      // Total model failure: notes needed embedding but none succeeded.
      if (attempted > 0 && embedded === 0) {
        // Keep any previously-good vectors we reused; if there are none either,
        // the panel falls back to keyword mode but the status makes the cause clear.
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
      // Drain anything that changed while we were building.
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

  // Read + clean a single note into the string we embed, or null if empty.
  private async readEmbedText(file: TFile): Promise<string | null> {
    try {
      const body = await this.app.vault.cachedRead(file);
      const text = buildEmbedText(file, body, this.options.embedCharLimit);
      return text.length === 0 ? null : text;
    } catch (e) {
      console.warn(`[related-notes] failed to read ${file.path}`, e);
      return null;
    }
  }

  private async embedFile(
    file: TFile,
    onProgress?: ProgressCallback,
  ): Promise<Float32Array | null> {
    const text = await this.readEmbedText(file);
    if (!text) return null;
    try {
      return await this.engine.embed(text, "query", onProgress);
    } catch (e) {
      console.warn(`[related-notes] failed to embed ${file.path}`, e);
      return null;
    }
  }

  // --- incremental updates ---------------------------------------------------
  // Re-embed a single changed/created file. Queued (not run) while a full build
  // is in flight, then drained by flushPending() when the build finishes.
  async updateFile(file: TFile): Promise<void> {
    if (this.isExcluded(file.path) || file.extension !== "md") return;
    if (this.building) {
      this.pending.add(file.path);
      return;
    }
    const vector = await this.embedFile(file);
    if (vector) {
      this.entries.set(file.path, {
        path: file.path,
        mtime: file.stat.mtime,
        vector,
      });
      this.setProgress({ done: this.entries.size, total: this.entries.size });
      await this.persist();
    }
  }

  removeFile(path: string): void {
    this.wordCache.delete(path);
    if (this.entries.delete(path)) {
      this.setProgress({ done: this.entries.size, total: this.entries.size });
      void this.persist();
    }
  }

  renameFile(oldPath: string, file: TFile): void {
    this.wordCache.delete(oldPath);
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
        const vector = await this.embedFile(file, onProgress);
        if (vector) {
          this.entries.set(path, { path, mtime: file.stat.mtime, vector });
        }
      }
    }
    await this.persist();
  }

  // --- ranking ---------------------------------------------------------------
  // Rank all indexed notes by cosine similarity to the active note. Falls back to
  // a cheap keyword/tag/link-overlap score when the active note has no embedding
  // yet (still indexing, or just created), so the panel is never empty.
  rank(active: TFile): RankedNote[] {
    const self = this.entries.get(active.path);
    if (!self) return this.keywordRank(active);

    const results: RankedNote[] = [];
    for (const entry of this.entries.values()) {
      if (entry.path === active.path) continue;
      const file = this.app.vault.getAbstractFileByPath(entry.path);
      if (!(file instanceof TFile)) continue;
      const score = cosineSimilarity(self.vector, entry.vector);
      if (score < this.options.minSimilarity) continue;
      results.push({ file, score, approximate: false });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, this.options.topK);
  }

  // Cheap fallback ranking when no embedding is available for the active note.
  // Scores other notes by overlap of significant title words, shared tags, and
  // direct links — normalized into a rough [0, 1] so the view can show a "~" pill.
  // Significant-word sets are memoized per note (keyed by mtime) so switching
  // notes during the initial build does not re-tokenize the whole vault each time.
  // Never throws; always returns something to show.
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
}
