// MUST be first: installs our bundled onnxruntime-web under
// Symbol.for("onnxruntime") BEFORE @huggingface/transformers is pulled in (via
// ./embeddings below), so the renderer uses the web runtime instead of the
// externalized, undefined onnxruntime-node. See ort-shim.ts for the full why.
import "./ort-shim";
import {
  Plugin,
  PluginSettingTab,
  Setting,
  App,
  TFile,
  TAbstractFile,
  WorkspaceLeaf,
  Notice,
  normalizePath,
  debounce,
  type Debouncer,
} from "obsidian";
import {
  EmbeddingEngine,
  setWasmBaseUrl,
  type DevicePref,
} from "./embeddings";
import { IndexStore, stripMarkdown, type IndexStoreOptions } from "./index-store";
import { RelatedNotesView, VIEW_TYPE_RELATED } from "./view";

// --- settings ---------------------------------------------------------------

export interface RelatedNotesSettings {
  modelId: string;
  device: DevicePref;
  topK: number;
  minSimilarity: number;
  embedCharLimit: number;
  excludeFolders: string; // comma- or newline-separated folder paths
  showSnippet: boolean;
  // --- multi-vector / ranking ---
  chunking: boolean; // master toggle for the chunk-level path
  structureInfluence: number; // B_MAX for the hybrid structural boost (0..0.3)
  showSummary: boolean; // centroid-sentence line (supersedes snippet when on)
  showRecency: boolean; // muted "edited Nd ago" line
  maxChunks: number; // body-chunk cap (advanced)
  shortlistSize: number; // Stage-1 -> Stage-2 funnel width (advanced)
}

export const DEFAULT_SETTINGS: RelatedNotesSettings = {
  // A SYMMETRIC sentence-similarity model — the right tool for "which notes are
  // alike". (multilingual-e5-* are RETRIEVAL models, tuned for short-query →
  // document search, and rank note-to-note similarity poorly.)
  modelId: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
  device: "auto",
  topK: 12,
  // Chunk-level BiMax scores land in a higher, less-spread range than the old
  // whole-note centroid cosines, so the floor is re-centered up from 0.3.
  minSimilarity: 0.45,
  embedCharLimit: 1500,
  excludeFolders: "",
  showSnippet: true,
  chunking: true,
  structureInfluence: 0.15,
  showSummary: true,
  showRecency: false,
  maxChunks: 16,
  shortlistSize: 60,
};

// A few vetted model ids surfaced as a dropdown so users don't have to memorise
// HF repo paths. Any other id can still be typed in the text field below.
// Paraphrase (symmetric) models first — they judge note-to-note similarity far
// better than retrieval models for this use case.
const MODEL_OPTIONS: Record<string, string> = {
  "Xenova/paraphrase-multilingual-MiniLM-L12-v2":
    "MiniLM-L12 multilingual — best for related notes, fast (default)",
  "Xenova/paraphrase-multilingual-mpnet-base-v2":
    "mpnet-base multilingual — strongest matches, larger & slower",
  "Xenova/multilingual-e5-small":
    "e5-small — retrieval/search model, weaker for note similarity",
};

// One-click presets. "Balanced" is light and fast; "Best quality" uses a larger
// model and more context for the strongest matches. Each applies to the relevant
// settings; the index rebuilds automatically if the model changes.
type ProfileName = "balanced" | "best";
const PROFILES: Record<ProfileName, Partial<RelatedNotesSettings>> = {
  balanced: {
    modelId: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    device: "auto",
    topK: 8,
    minSimilarity: 0.45,
    embedCharLimit: 1200,
    showSnippet: true,
    chunking: true,
    showSummary: true,
    structureInfluence: 0.15,
    maxChunks: 16,
  },
  best: {
    modelId: "Xenova/paraphrase-multilingual-mpnet-base-v2",
    device: "auto",
    topK: 20,
    minSimilarity: 0.35,
    embedCharLimit: 3500,
    showSnippet: true,
    chunking: true,
    showSummary: true,
    structureInfluence: 0.2,
    maxChunks: 20,
  },
};

// Length of the muted snippet shown on each card.
const SNIPPET_CHARS = 160;

// Filename the plugin probes to derive the app:// base URL for its self-hosted
// onnxruntime-web wasm folder. Copied next to main.js by the build (gen-ort.mjs).
const ORT_PROBE_FILE = "ort/ort-wasm-simd-threaded.jsep.wasm";

// ---------------------------------------------------------------------------

export default class RelatedNotesPlugin extends Plugin {
  declare settings: RelatedNotesSettings;
  store!: IndexStore;
  private engine!: EmbeddingEngine;

  // The model id / device preference the current engine was built for. Compared
  // against settings on save to decide whether a re-embed is actually needed —
  // critically NOT against engine.device (the RESOLVED device), which would never
  // equal the "auto" preference and would rebuild on every unrelated save.
  private appliedModelId!: string;
  private appliedDevicePref!: DevicePref;
  // The embedding-shape settings the current index was built for. Changing any of
  // them alters WHAT is embedded per note, so they trigger a rebuild like a model
  // change. (showSummary also changes whether chunk text is persisted.)
  private appliedChunking!: boolean;
  private appliedMaxChunks!: number;
  private appliedShowSummary!: boolean;
  // Guards the engine-swap + rebuild path against re-entrancy.
  private swapping = false;

  // Coalesce vault-change bursts (bulk edits, sync) into batched re-embeds.
  private debouncedUpdate!: Debouncer<[TFile], void>;
  // Pending changed files, drained by the debounced updater above.
  private dirty = new Set<string>();

  async onload(): Promise<void> {
    const saved = (await this.loadData()) as Partial<RelatedNotesSettings> | null;
    this.settings = Object.assign({}, DEFAULT_SETTINGS, saved);

    // Point onnxruntime-web at the plugin's self-hosted wasm folder so the .wasm
    // matches the bundled glue exactly and the plugin works offline. If the folder
    // didn't ship (BRAT/manual install) we leave the engine on its pinned-CDN
    // fallback. Awaited so the base is set before the first embed.
    await this.configureWasmPaths();

    this.engine = new EmbeddingEngine(this.settings.modelId, this.settings.device);
    this.appliedModelId = this.settings.modelId;
    this.appliedDevicePref = this.settings.device;
    this.appliedChunking = this.settings.chunking;
    this.appliedMaxChunks = this.settings.maxChunks;
    this.appliedShowSummary = this.settings.showSummary;
    this.store = new IndexStore(
      this.app,
      this.engine,
      this.pluginDir(),
      this.storeOptions(),
    );

    this.registerView(VIEW_TYPE_RELATED, (leaf) => new RelatedNotesView(leaf, this));

    this.addRibbonIcon("sparkles", "Smart related notes", () => {
      void this.activateView();
    });

    this.addCommand({
      id: "open-view",
      name: "Open the panel",
      callback: () => {
        void this.activateView();
      },
    });

    this.addCommand({
      id: "rebuild-index",
      name: "Rebuild the index",
      callback: () => {
        void this.rebuildIndex();
      },
    });

    this.addSettingTab(new RelatedNotesSettingTab(this.app, this));

    // Re-rank the panel when the active note changes (the view debounces internally).
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => {
        this.getView()?.requestRender();
      }),
    );

    // --- incremental index maintenance ---------------------------------------
    // 20s idle before re-embedding a changed note: typing (and the short pauses
    // while typing) never kicks off embeddings — only a real edit pause does.
    this.debouncedUpdate = debounce(() => void this.flushDirty(), 20000, false);

    this.registerEvent(
      this.app.vault.on("modify", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.dirty.add(file.path);
          this.debouncedUpdate(file);
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("create", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.dirty.add(file.path);
          this.debouncedUpdate(file);
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("delete", (file: TAbstractFile) => {
        this.store.removeFile(file.path);
        this.snippetCache.delete(file.path);
        this.getView()?.requestRender();
      }),
    );
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        this.snippetCache.delete(oldPath);
        if (file instanceof TFile && file.extension === "md") {
          this.store.renameFile(oldPath, file);
        } else {
          this.store.removeFile(oldPath);
        }
        this.getView()?.requestRender();
      }),
    );

    // Load (or build) the index once the layout is ready, so the vault file list
    // and metadata cache are fully populated first.
    this.app.workspace.onLayoutReady(() => {
      void this.bootstrapIndex();
    });
  }

  onunload(): void {
    // The registered view is torn down by Obsidian; nothing else global to clean.
  }

  // --- wasm path resolution --------------------------------------------------
  // Derive an app:// URL for the plugin's ort/ folder from the resource path of a
  // known file inside it, then hand the directory base to the embedding engine.
  // getResourcePath returns something like "app://<hash>/<abs>/ort/<file>?<mtime>";
  // we strip the query and the trailing filename to get the directory base. On any
  // failure the engine keeps its pinned-CDN fallback, so this never blocks load.
  private async configureWasmPaths(): Promise<void> {
    try {
      const probe = normalizePath(`${this.pluginDir()}/${ORT_PROBE_FILE}`);
      // Only self-host if the wasm actually shipped (the full related-notes.zip).
      // BRAT / manual installs copy just main.js+manifest+styles, so there is no
      // ort/ folder — getResourcePath would still return an app:// URL, but it
      // 404s, wedging onnxruntime with ZERO providers ("Unsupported device: wasm.
      // Should be one of: ."). When the file is absent we leave the base unset so
      // the engine uses its version-pinned CDN fallback instead.
      if (!(await this.app.vault.adapter.exists(probe))) return;
      const resource = this.app.vault.adapter.getResourcePath(probe);
      if (!resource) return;
      const noQuery = resource.split("?")[0];
      const slash = noQuery.lastIndexOf("/");
      if (slash < 0) return;
      const base = noQuery.slice(0, slash + 1); // keep trailing slash
      setWasmBaseUrl(base);
    } catch (e) {
      console.warn(
        "[related-notes] could not resolve local wasm path; using CDN fallback",
        e,
      );
    }
  }

  private pluginDir(): string {
    return (
      this.manifest.dir ??
      `${this.app.vault.configDir}/plugins/related-notes`
    );
  }

  // --- index lifecycle -------------------------------------------------------
  private async bootstrapIndex(): Promise<void> {
    const loaded = await this.store.load();
    this.getView()?.requestRender();
    if (!loaded) {
      await this.store.build();
      this.getView()?.requestRender();
    }
  }

  // Manual "Rebuild index" command/button: FORCE a full re-embed of every note
  // (not the cheap mtime-reuse build), so it always reflects the current model and
  // settings instead of finishing instantly with the old vectors.
  async rebuildIndex(): Promise<void> {
    await this.store.build(undefined, true);
    this.getView()?.requestRender();
  }

  private async flushDirty(): Promise<void> {
    if (this.dirty.size === 0) return;
    const paths = Array.from(this.dirty);
    this.dirty.clear();
    for (const path of paths) {
      const file = this.app.vault.getAbstractFileByPath(path);
      if (file instanceof TFile) await this.store.updateFile(file);
    }
    this.getView()?.requestRender();
  }

  // --- view plumbing ---------------------------------------------------------
  getView(): RelatedNotesView | null {
    const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_RELATED);
    const view = leaves[0]?.view;
    return view instanceof RelatedNotesView ? view : null;
  }

  // Open the view in the LEFT sidebar (reuse an existing one if present), then
  // reveal it so it is the active tab.
  async activateView(): Promise<void> {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = workspace.getLeavesOfType(VIEW_TYPE_RELATED)[0] ?? null;
    if (!leaf) {
      leaf = workspace.getLeftLeaf(false);
      if (!leaf) return;
      await leaf.setViewState({ type: VIEW_TYPE_RELATED, active: true });
    }
    await workspace.revealLeaf(leaf);
  }

  // --- snippet ---------------------------------------------------------------
  // The 1–2 line muted preview shown on each card. A synchronous read of file
  // content isn't possible here, so we cache by mtime and kick off an async read
  // when missing, returning any stale value meanwhile so the card isn't blank.
  // The async resolution schedules ONE debounced re-render (via the view), so N
  // cards resolving on first paint collapse into a single extra render pass
  // instead of a render storm.
  getSnippet(file: TFile): string {
    const cached = this.snippetCache.get(file.path);
    if (cached && cached.mtime === file.stat.mtime) return cached.text;
    void this.app.vault.cachedRead(file).then((content) => {
      const text = stripMarkdown(content).slice(0, SNIPPET_CHARS);
      this.snippetCache.set(file.path, { mtime: file.stat.mtime, text });
      this.getView()?.requestRender();
    });
    return cached?.text ?? "";
  }

  private snippetCache = new Map<string, { mtime: number; text: string }>();

  // --- settings glue ---------------------------------------------------------
  private storeOptions(): IndexStoreOptions {
    return {
      embedCharLimit: this.settings.embedCharLimit,
      excludeFolders: this.parseExcludeFolders(),
      topK: this.settings.topK,
      minSimilarity: this.settings.minSimilarity,
      chunking: this.settings.chunking,
      structureInfluence: this.settings.structureInfluence,
      maxChunks: this.settings.maxChunks,
      shortlistSize: this.settings.shortlistSize,
      showSummary: this.settings.showSummary,
    };
  }

  private parseExcludeFolders(): string[] {
    return this.settings.excludeFolders
      .split(/[\n,]/)
      .map((s) => s.trim().replace(/\/+$/, ""))
      .filter((s) => s.length > 0);
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
    this.store.updateOptions(this.storeOptions());

    // Only a MODEL or DEVICE-PREFERENCE change invalidates the stored vectors.
    // Compare against the last-APPLIED preferences (not the engine's resolved
    // device, which is "webgpu"/"wasm"/null and would never equal an "auto"
    // preference — the bug that made every slider drag rebuild the vault).
    const modelChanged = this.appliedModelId !== this.settings.modelId;
    const deviceChanged = this.appliedDevicePref !== this.settings.device;
    // Embedding-SHAPE changes alter WHAT is embedded (chunking on/off, the chunk
    // cap) or whether chunk text is persisted (showSummary), so they too need a
    // full re-embed — but they keep the SAME engine.
    const shapeChanged =
      this.appliedChunking !== this.settings.chunking ||
      this.appliedMaxChunks !== this.settings.maxChunks ||
      this.appliedShowSummary !== this.settings.showSummary;

    if ((modelChanged || deviceChanged) && !this.swapping) {
      this.swapping = true;
      try {
        this.engine = new EmbeddingEngine(
          this.settings.modelId,
          this.settings.device,
        );
        this.appliedModelId = this.settings.modelId;
        this.appliedDevicePref = this.settings.device;
        this.appliedChunking = this.settings.chunking;
        this.appliedMaxChunks = this.settings.maxChunks;
        this.appliedShowSummary = this.settings.showSummary;
        // Swap the engine IN PLACE: the store (and the view's progress
        // subscription) stay valid, so the rebuild's status line stays live.
        this.store.setEngine(this.engine);
        new Notice("Related notes: model changed, rebuilding index…");
        await this.store.build();
      } finally {
        this.swapping = false;
      }
    } else if (shapeChanged && !this.swapping) {
      this.swapping = true;
      try {
        this.appliedChunking = this.settings.chunking;
        this.appliedMaxChunks = this.settings.maxChunks;
        this.appliedShowSummary = this.settings.showSummary;
        new Notice("Related notes: chunking settings changed, rebuilding index…");
        // Same engine, but force a full re-embed so every note's chunk set matches
        // the new shape.
        await this.store.build(undefined, true);
      } finally {
        this.swapping = false;
      }
    }
    this.getView()?.requestRender();
  }
}

export class RelatedNotesSettingTab extends PluginSettingTab {
  plugin: RelatedNotesPlugin;
  private readonly debouncedSave: Debouncer<[], void>;

  constructor(app: App, plugin: RelatedNotesPlugin) {
    super(app, plugin);
    this.plugin = plugin;
    this.debouncedSave = debounce(() => void this.plugin.saveSettings(), 500, false);
  }

  // Apply a one-click preset, persist (the index rebuilds if the model changed),
  // then re-render so every control reflects the new values.
  private async applyProfile(name: ProfileName): Promise<void> {
    Object.assign(this.plugin.settings, PROFILES[name]);
    await this.plugin.saveSettings();
    this.render();
    new Notice(
      `Related notes: applied the ${name === "best" ? "Best quality" : "Balanced"} profile.`,
    );
  }

  display(): void {
    this.render();
  }

  // The tab body, callable directly (e.g. after applying a profile) without the
  // deprecated display() entry point.
  private render(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("Performance profile")
      .setDesc(
        "Quick presets. Balanced is lighter and faster; Best quality uses a larger model and more context for the strongest matches.",
      )
      .addButton((b) =>
        b
          .setButtonText("Balanced")
          .onClick(() => void this.applyProfile("balanced")),
      )
      .addButton((b) =>
        b
          .setButtonText("Best quality")
          .onClick(() => void this.applyProfile("best")),
      );

    new Setting(containerEl)
      .setName("Model")
      .setDesc(
        "The embedding model. Paraphrase (symmetric) models judge note-to-note similarity best — MiniLM-L12 (default) is fast; mpnet-base is the strongest. e5 is a retrieval model and ranks similarity poorly here. Weights download once and are cached; changing the model rebuilds the index.",
      )
      .addDropdown((d) => {
        for (const [id, label] of Object.entries(MODEL_OPTIONS)) d.addOption(id, label);
        // Allow a custom id the dropdown doesn't list.
        if (!(this.plugin.settings.modelId in MODEL_OPTIONS)) {
          d.addOption(this.plugin.settings.modelId, `${this.plugin.settings.modelId} (custom)`);
        }
        d.setValue(this.plugin.settings.modelId).onChange(async (v) => {
          this.plugin.settings.modelId = v;
          await this.plugin.saveSettings();
        });
      });

    new Setting(containerEl)
      .setName("Compute device")
      .setDesc(
        "Auto uses your GPU (WebGPU) when available and falls back to WASM (CPU) otherwise. WASM always works and needs no GPU.",
      )
      .addDropdown((d) =>
        d
          .addOption("auto", "Auto (recommended)")
          .addOption("webgpu", "WebGPU (GPU)")
          .addOption("wasm", "WASM (CPU)")
          .setValue(this.plugin.settings.device)
          .onChange(async (v) => {
            this.plugin.settings.device = v as DevicePref;
            await this.plugin.saveSettings();
          }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("Number of results")
        .setDesc("How many related notes to show in the card stack.");
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.topK),
      });
      setting.addSlider((s) =>
        s
          .setLimits(4, 30, 1)
          .setValue(this.plugin.settings.topK)
          .onChange((v) => {
            this.plugin.settings.topK = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    {
      const setting = new Setting(containerEl)
        .setName("Minimum similarity")
        .setDesc(
          "Hide notes below this similarity (0–1). Chunk-level matching scores higher and tighter than before, so the default floor moved up — around 0.45 is a good starting point. Lower shows more, looser matches.",
        );
      const fmt = (v: number) => v.toFixed(2);
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: fmt(this.plugin.settings.minSimilarity),
      });
      setting.addSlider((s) =>
        s
          .setLimits(0, 0.9, 0.05)
          .setValue(this.plugin.settings.minSimilarity)
          .onChange((v) => {
            this.plugin.settings.minSimilarity = v;
            valueEl.setText(fmt(v));
            this.debouncedSave();
          }),
      );
    }

    {
      const setting = new Setting(containerEl)
        .setName("Embed character limit")
        .setDesc(
          "Total characters of each note's body considered for chunking. More context is more accurate but slower to index.",
        );
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.embedCharLimit),
      });
      setting.addSlider((s) =>
        s
          .setLimits(500, 4000, 100)
          .setValue(this.plugin.settings.embedCharLimit)
          .onChange((v) => {
            this.plugin.settings.embedCharLimit = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    new Setting(containerEl)
      .setName("Excluded folders")
      .setDesc(
        "Folders to leave out of the index, one per line (or comma-separated). Matches a folder and everything beneath it.",
      )
      .addTextArea((t) =>
        t
          .setPlaceholder("Templates\nArchive/2023")
          .setValue(this.plugin.settings.excludeFolders)
          .onChange((v) => {
            this.plugin.settings.excludeFolders = v;
            this.debouncedSave();
          }),
      );

    new Setting(containerEl)
      .setName("Show summary line")
      .setDesc(
        "Show the note's most representative sentence on each card (computed locally from its own chunks — no extra model). Falls back to a plain preview when off. Toggling this rebuilds the index so the summary text is available.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.showSummary).onChange(async (v) => {
          this.plugin.settings.showSummary = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Show snippet")
      .setDesc(
        "Show a one- to two-line text preview on each card. Used as the fallback when the summary line is off.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.showSnippet).onChange(async (v) => {
          this.plugin.settings.showSnippet = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Show last-edited time")
      .setDesc("Add a muted “edited Nd ago” line to each card.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.showRecency).onChange(async (v) => {
          this.plugin.settings.showRecency = v;
          await this.plugin.saveSettings();
        }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("Structure influence")
        .setDesc(
          "How much shared tags, links, and folders nudge the ranking. Bounded so it only re-orders near-ties and never promotes an unrelated note. 0 disables it.",
        );
      const fmt = (v: number) => v.toFixed(2);
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: fmt(this.plugin.settings.structureInfluence),
      });
      setting.addSlider((s) =>
        s
          .setLimits(0, 0.3, 0.01)
          .setValue(this.plugin.settings.structureInfluence)
          .onChange((v) => {
            this.plugin.settings.structureInfluence = v;
            valueEl.setText(fmt(v));
            this.debouncedSave();
          }),
      );
    }

    new Setting(containerEl)
      .setName("Chunk-level matching")
      .setDesc(
        "Embed each note as several sentence-level chunks instead of one whole-note vector — far more accurate for long notes. Turning this off reverts to single-vector matching. Changing it rebuilds the index.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.chunking).onChange(async (v) => {
          this.plugin.settings.chunking = v;
          await this.plugin.saveSettings();
        }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("Max chunks per note")
        .setDesc(
          "Advanced. Cap on sentence-window chunks embedded per note (the title is extra). Higher captures more of long notes but grows the index and slows ranking. Changing it rebuilds the index.",
        );
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.maxChunks),
      });
      setting.addSlider((s) =>
        s
          .setLimits(4, 32, 1)
          .setValue(this.plugin.settings.maxChunks)
          .onChange((v) => {
            this.plugin.settings.maxChunks = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    {
      const setting = new Setting(containerEl)
        .setName("Shortlist size")
        .setDesc(
          "Advanced. How many coarse candidates are re-ranked with the precise chunk comparison on each note switch. Higher is slightly more thorough but slower (kept at least 4× the result count).",
        );
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.shortlistSize),
      });
      setting.addSlider((s) =>
        s
          .setLimits(20, 150, 10)
          .setValue(this.plugin.settings.shortlistSize)
          .onChange((v) => {
            this.plugin.settings.shortlistSize = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    // --- index status + manual rebuild ---------------------------------------
    const progress = this.plugin.store.getProgress();
    new Setting(containerEl)
      .setName("Index")
      .setDesc(
        progress.status === "building"
          ? `Indexing… ${progress.done}/${progress.total}`
          : `${this.plugin.store.count} notes embedded.`,
      )
      .addButton((b) =>
        b
          .setButtonText("Rebuild index")
          .setCta()
          .onClick(() => {
            void this.plugin.rebuildIndex();
          }),
      );
  }

  hide(): void {
    // Persist any value typed/dragged right before the pane closed.
    this.debouncedSave.run();
  }
}
