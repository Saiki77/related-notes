import {
  ItemView,
  WorkspaceLeaf,
  TFile,
  setIcon,
  debounce,
  type Debouncer,
} from "obsidian";
import type RelatedNotesPlugin from "./main";
import type {
  IndexProgress,
  RankedNote,
  WhyReason,
} from "./index-store";

export const VIEW_TYPE_RELATED = "smart-related-notes";

// The left-sidebar card stack. Subscribes to the index store's progress so its
// status line tracks indexing live, and re-ranks (debounced) whenever the active
// note changes. Because the plugin keeps ONE stable IndexStore (swapping the
// engine in place on a model change rather than replacing the store), this
// subscription stays valid for the lifetime of the view.
export class RelatedNotesView extends ItemView {
  private readonly plugin: RelatedNotesPlugin;
  private listEl!: HTMLElement;
  private subtitleEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private unsubscribe: (() => void) | null = null;

  // Re-rendering is debounced so a flurry of active-leaf-change events (e.g. fast
  // tab switching) collapses into one ranking pass.
  private readonly scheduleRender: Debouncer<[], void>;

  // Search box state (toggled by the header search icon).
  private searchRowEl!: HTMLElement;
  private searchInputEl!: HTMLInputElement;
  private searchQuery = "";
  private searchSeq = 0; // guards against stale async query results
  private readonly scheduleSearch: Debouncer<[], void>;

  // "Linked notes" mode (toggled by the header link icon): instead of related notes,
  // show the notes the active note LINKS TO — i.e. a MOC's members — as cards.
  private linksMode = false;
  private linksToggleEl!: HTMLElement;

  constructor(leaf: WorkspaceLeaf, plugin: RelatedNotesPlugin) {
    super(leaf);
    this.plugin = plugin;
    this.scheduleRender = debounce(() => this.render(), 300, false);
    this.scheduleSearch = debounce(() => void this.runSearch(), 250, false);
  }

  getViewType(): string {
    return VIEW_TYPE_RELATED;
  }

  getDisplayText(): string {
    return "Smart related notes";
  }

  getIcon(): string {
    return "sparkles";
  }

  protected async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("related-notes-view");

    const header = root.createDiv({ cls: "rn-header" });
    const titleRow = header.createDiv({ cls: "rn-title-row" });
    titleRow.createDiv({ cls: "rn-heading", text: "Smart related notes" });
    const actions = titleRow.createDiv({ cls: "rn-actions" });
    const searchToggle = actions.createDiv({
      cls: "rn-search-toggle clickable-icon",
      attr: { "aria-label": "Search notes" },
    });
    setIcon(searchToggle, "search");
    searchToggle.addEventListener("click", () => this.toggleSearch());
    this.linksToggleEl = actions.createDiv({
      cls: "rn-links-toggle clickable-icon",
      attr: { "aria-label": "Show notes this note links to (MOC members)" },
    });
    setIcon(this.linksToggleEl, "link");
    this.linksToggleEl.addEventListener("click", () => this.toggleLinks());
    const refresh = actions.createDiv({
      cls: "rn-refresh clickable-icon",
      attr: { "aria-label": "Rebuild the index" },
    });
    setIcon(refresh, "refresh-cw");
    refresh.addEventListener("click", () => {
      void this.plugin.rebuildIndex();
    });

    // Hidden until the search icon is clicked. Typing runs a semantic search.
    this.searchRowEl = header.createDiv({ cls: "rn-search-row" });
    this.searchInputEl = this.searchRowEl.createEl("input", {
      cls: "rn-search-input",
      attr: {
        type: "text",
        placeholder: "Search notes by meaning…",
        "aria-label": "Search notes",
        spellcheck: "false",
      },
    });
    this.searchInputEl.addEventListener("input", () => {
      this.searchQuery = this.searchInputEl.value.trim();
      if (!this.searchQuery) {
        this.scheduleSearch.cancel();
        this.render();
      } else {
        this.scheduleSearch();
      }
    });
    this.searchInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        this.clearSearch();
      }
    });

    this.subtitleEl = header.createDiv({ cls: "rn-subtitle" });
    this.statusEl = header.createDiv({ cls: "rn-status" });

    this.listEl = root.createDiv({ cls: "rn-list" });

    // Live status line: track the index store's progress.
    this.unsubscribe = this.plugin.store.onProgress((p) => this.renderStatus(p));
    this.renderStatus(this.plugin.store.getProgress());
    this.render();
  }

  protected async onClose(): Promise<void> {
    this.unsubscribe?.();
    this.unsubscribe = null;
    // Cancel armed debouncers so a timer can't fire into the torn-down view.
    this.scheduleRender.cancel();
    this.scheduleSearch.cancel();
  }

  // Public so the plugin can poke it on active-leaf-change.
  requestRender(): void {
    this.scheduleRender();
  }

  private renderStatus(p: IndexProgress): void {
    if (!this.statusEl) return;
    this.statusEl.empty();
    if (p.status === "building") {
      const pct = p.total > 0 ? Math.round((p.done / p.total) * 100) : 0;
      this.statusEl.setText(`Indexing… ${pct}% (${p.done}/${p.total})`);
      this.statusEl.removeClass("rn-status-error");
    } else if (p.status === "loading") {
      this.statusEl.setText("Loading index…");
      this.statusEl.removeClass("rn-status-error");
    } else if (p.status === "error") {
      this.statusEl.setText(p.message ? `Indexing failed: ${p.message}` : "Indexing failed");
      this.statusEl.addClass("rn-status-error");
    } else if (p.status === "ready") {
      this.statusEl.setText(`${p.done} notes indexed`);
      this.statusEl.removeClass("rn-status-error");
    } else {
      this.statusEl.setText("");
    }
    // When the index just turned ready, the active note may now have a real vector;
    // refresh the cards so the fallback ranking is replaced by semantic results.
    if (p.status === "ready") this.scheduleRender();
  }

  render(): void {
    if (!this.listEl) return;
    // A search is active — its (async) results own the list; don't clobber them.
    if (this.searchQuery) return;
    const active = this.app.workspace.getActiveFile();
    this.listEl.empty();

    if (!active || active.extension !== "md") {
      if (this.linksMode) {
        this.subtitleEl.setText("Open a note to see what it links to");
        this.renderEmpty("No active note.");
        return;
      }
      this.renderRecent();
      return;
    }

    if (this.linksMode) {
      this.renderLinks(active);
      return;
    }

    this.subtitleEl.empty();
    this.subtitleEl.appendText("Based on ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: active.basename });

    const ranked = this.plugin.store.rank(active);
    if (ranked.length === 0) {
      const status = this.plugin.store.getProgress().status;
      this.renderEmpty(
        status === "building" || status === "loading"
          ? "Indexing… related notes will appear here."
          : "No related notes found.",
      );
      return;
    }

    for (const item of ranked) this.renderCard(item);
  }

  private renderEmpty(text: string): void {
    this.listEl.createDiv({ cls: "rn-empty", text });
  }

  private toggleSearch(): void {
    const willShow = !this.searchRowEl.hasClass("is-visible");
    this.searchRowEl.toggleClass("is-visible", willShow);
    if (willShow) {
      window.setTimeout(() => this.searchInputEl.focus(), 0);
    } else {
      this.scheduleSearch.cancel();
      this.searchInputEl.value = "";
      this.searchQuery = "";
      this.render();
    }
  }

  private clearSearch(): void {
    this.scheduleSearch.cancel();
    this.searchInputEl.value = "";
    this.searchQuery = "";
    this.searchRowEl.removeClass("is-visible");
    this.render();
  }

  // Toggle "linked notes" mode. Mutually exclusive with search (leaving search if open).
  private toggleLinks(): void {
    this.linksMode = !this.linksMode;
    this.linksToggleEl.toggleClass("is-active", this.linksMode);
    if (this.searchRowEl.hasClass("is-visible") || this.searchQuery) {
      this.scheduleSearch.cancel();
      this.searchInputEl.value = "";
      this.searchQuery = "";
      this.searchRowEl.removeClass("is-visible");
    }
    this.render();
  }

  // Show the notes the active note links to (resolved wikilinks) as cards. For a MOC
  // this is its member list — the structural counterpart to similarity ranking.
  private renderLinks(active: TFile): void {
    this.subtitleEl.empty();
    this.subtitleEl.appendText("Linked from ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: active.basename });

    const resolved = this.app.metadataCache.resolvedLinks[active.path] ?? {};
    const targets: TFile[] = [];
    const seen = new Set<string>();
    for (const path of Object.keys(resolved)) {
      if (seen.has(path)) continue;
      seen.add(path);
      const f = this.app.vault.getAbstractFileByPath(path);
      if (f instanceof TFile && f.extension === "md") targets.push(f);
    }
    if (targets.length === 0) {
      this.renderEmpty("This note doesn't link to any notes yet.");
      return;
    }
    for (const file of targets) this.renderLinkCard(file);
  }

  private renderLinkCard(file: TFile): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });
    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: file.basename });
    const pills = card.createDiv({ cls: "rn-pills" });
    pills.createSpan({ cls: "rn-conn rn-conn-linked", text: "Linked" });
    const parentPath = file.parent?.path ?? "";
    if (parentPath.length > 0 && parentPath !== "/") {
      card.createDiv({ cls: "rn-path", text: parentPath });
    }
    card.addEventListener("click", () => {
      void this.app.workspace.getLeaf(false).openFile(file);
    });
  }

  // Semantic search: rank notes by similarity to the typed query (keyword fallback
  // in the store when the engine isn't ready). Guarded against stale async results.
  private async runSearch(): Promise<void> {
    const query = this.searchQuery;
    if (!query) {
      this.render();
      return;
    }
    const seq = ++this.searchSeq;
    this.subtitleEl.empty();
    this.subtitleEl.appendText("Search: ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: query });
    let results: RankedNote[] = [];
    try {
      results = await this.plugin.store.rankByQuery(query);
    } catch {
      results = [];
    }
    if (seq !== this.searchSeq || this.searchQuery !== query) return; // superseded
    this.listEl.empty();
    if (results.length === 0) {
      const status = this.plugin.store.getProgress().status;
      this.renderEmpty(
        status === "building" || status === "loading"
          ? "Indexing… search results will improve as notes are added."
          : "No matches found.",
      );
      return;
    }
    for (const item of results) this.renderCard(item);
  }

  // With no active note, surface recent notes so the panel stays useful: recently
  // opened, falling back to recently modified for a vault with no open-history yet.
  private renderRecent(): void {
    const recent = this.recentNotes();
    if (recent.length === 0) {
      this.subtitleEl.setText("Open a note to see related notes");
      this.renderEmpty("No active note.");
      return;
    }
    this.subtitleEl.setText("Recent notes");
    for (const file of recent) this.renderRecentCard(file);
  }

  private recentNotes(): TFile[] {
    const out: TFile[] = [];
    const seen = new Set<string>();
    for (const path of this.app.workspace.getLastOpenFiles()) {
      if (seen.has(path)) continue;
      const f = this.app.vault.getAbstractFileByPath(path);
      if (f instanceof TFile && f.extension === "md") {
        seen.add(path);
        out.push(f);
        if (out.length >= this.plugin.settings.topK) break;
      }
    }
    if (out.length > 0) return out;
    // Fresh vault / no open-history: fall back to the most recently modified notes.
    return this.app.vault
      .getMarkdownFiles()
      .sort((a, b) => b.stat.mtime - a.stat.mtime)
      .slice(0, this.plugin.settings.topK);
  }

  private renderRecentCard(file: TFile): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });
    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: file.basename });
    const parentPath = file.parent?.path ?? "";
    if (parentPath.length > 0 && parentPath !== "/") {
      card.createDiv({ cls: "rn-path", text: parentPath });
    }
    const rel = relativeTime(file.stat.mtime);
    if (rel) card.createDiv({ cls: "rn-recency", text: `edited ${rel}` });
    card.addEventListener("click", () => {
      void this.app.workspace.getLeaf(false).openFile(file);
    });
  }

  private renderCard(item: RankedNote): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });

    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: item.file.basename });

    // Similarity pill: a "~" prefix flags the keyword fallback (approximate).
    const pct = Math.round(item.score * 100);
    const pill = top.createDiv({ cls: "rn-score" });
    pill.setText(`${item.approximate ? "~" : ""}${pct}%`);
    if (item.approximate) pill.addClass("rn-score-approx");

    // Why-related + connection pills, derived from the structural signals that
    // fired. keywordRank results carry no reason/connection — render no pill rather
    // than a wrong one.
    if (item.reason || item.connection) {
      const pills = card.createDiv({ cls: "rn-pills" });
      if (item.connection === "linked") {
        pills.createSpan({ cls: "rn-conn rn-conn-linked", text: "Linked" });
      } else if (item.connection === "related") {
        pills.createSpan({ cls: "rn-conn", text: "Related" });
      }
      if (item.reason) {
        const why = this.whyLabel(item.reason);
        // Skip a redundant "Linked" why when the connection pill already says it.
        if (why && !(item.reason.kind === "linked" && item.connection === "linked")) {
          pills.createSpan({ cls: "rn-why", text: why });
        }
      }
    }

    const parentPath = item.file.parent?.path ?? "";
    if (parentPath.length > 0 && parentPath !== "/") {
      card.createDiv({ cls: "rn-path", text: parentPath });
    }

    // Topic-label summary line (preferred) with the snippet as a graceful fallback.
    if (this.plugin.settings.showSummary) {
      const summary = this.plugin.store.getSummary(item.file);
      const line = summary.length > 0 ? summary : this.plugin.getSnippet(item.file);
      if (line.length > 0) {
        card.createDiv({ cls: "rn-snippet", text: line });
      }
    } else if (this.plugin.settings.showSnippet) {
      const snippet = this.plugin.getSnippet(item.file);
      if (snippet.length > 0) {
        card.createDiv({ cls: "rn-snippet", text: snippet });
      }
    }

    if (this.plugin.settings.showRecency) {
      const rel = relativeTime(item.file.stat.mtime);
      if (rel) card.createDiv({ cls: "rn-recency", text: `edited ${rel}` });
    }

    card.addEventListener("click", () => {
      void this.app.workspace.getLeaf(false).openFile(item.file);
    });
  }

  // Human label for a why-reason. Names the top shared tag for the shared-tags kind.
  private whyLabel(reason: WhyReason): string {
    switch (reason.kind) {
      case "linked":
        return "Linked";
      case "shared-tags":
        return reason.detail ? `#${reason.detail}` : "Shared tags";
      case "co-cited":
        return "Co-cited";
      case "semantic":
        return "Similar text";
      default:
        return "";
    }
  }
}

// Compact "edited 3d ago"-style relative time from an mtime (ms). Returns "" for a
// missing/invalid timestamp.
function relativeTime(mtime: number): string {
  if (!mtime || !Number.isFinite(mtime)) return "";
  const diff = Date.now() - mtime;
  if (diff < 0) return "";
  const min = 60 * 1000;
  const hour = 60 * min;
  const day = 24 * hour;
  if (diff < hour) return "just now";
  if (diff < day) return `${Math.floor(diff / hour)}h ago`;
  if (diff < 30 * day) return `${Math.floor(diff / day)}d ago`;
  if (diff < 365 * day) return `${Math.floor(diff / (30 * day))}mo ago`;
  return `${Math.floor(diff / (365 * day))}y ago`;
}
