import {
  App,
  MarkdownView,
  TFile,
  getFrontMatterInfo,
  editorLivePreviewField,
  type Editor,
} from "obsidian";
import {
  ViewPlugin,
  Decoration,
  EditorView,
  type DecorationSet,
  type ViewUpdate,
} from "@codemirror/view";
import { RangeSetBuilder } from "@codemirror/state";
import type { TitleIndex } from "./title-index";

// =============================================================================
// Feature A — INLINE GLOW + 1-CLICK LINK (CodeMirror 6).
//
// A module-scoped ViewPlugin (defined once at load, before any plugin instance
// exists, so it cannot close over `this`) reads the live state through a
// module-scoped mutable BRIDGE the plugin keeps in sync — exactly the pattern
// easy-links' prefixHidePlugin / activeSettings uses. @codemirror/* is
// externalized in esbuild.config.mjs, so this uses Obsidian's own CM6 singleton.
//
// We build the DecorationSet MANUALLY (RangeSetBuilder), NOT MatchDecorator:
// MatchDecorator is regex-only and static-until-text-changes, so it cannot
// enforce only-a-real-note-matches, only-FIRST-occurrence, or skip
// links/code/frontmatter — and it doesn't re-derive on selection moves.
// =============================================================================

// The glow mark: a STABLE module-scope singleton (keeps RangeSet diffing cheap).
// It only adds a class + attributes; it never replaces/collapses text, so it
// composes with easy-links' replace decorations (which only ever cover the
// `Folder/` path prefix INSIDE `[[...]]` and are spatially disjoint from our
// unlinked-prose glow).
const GLOW_CLASS = "srn-glow";
const GLOW_MARK = Decoration.mark({
  class: GLOW_CLASS,
  attributes: { "aria-label": "Click to link", "data-srn": "1" },
});

// What the plugin performs when a glow is clicked: insert the wikilink for the
// surface text in [from, to). Routed through Obsidian's Editor (not view.dispatch)
// so it joins Obsidian undo + link bookkeeping and re-fires metadataCache changed.
export type GlowInsert = (range: {
  from: number;
  to: number;
  surface: string;
}) => void;

// The live state the module-scoped ViewPlugin reads. The plugin reassigns the
// fields and calls app.workspace.updateOptions() to force a rebuild on change.
export interface GlowBridge {
  enabled: boolean;
  restrictToLivePreview: boolean;
  glowAmbiguous: boolean;
  // First-only by default; when true (the autoLinkSubsequent display preview)
  // every occurrence glows.
  glowAll: boolean;
  titleIndex: TitleIndex | null;
  // The app handle, set once on load so buildGlow can resolve WHICH file each
  // EditorView belongs to (split-pane-safe), instead of relying on one global
  // active path that would be wrong for non-focused panes.
  app: App | null;
  // Path of the active (focused) note, kept as a FALLBACK only — used when a view
  // can't be matched to a leaf (should not normally happen).
  activePath: string | null;
  insert: GlowInsert | null;
}

// The single shared bridge instance. Exported so main.ts mutates the same object
// the ViewPlugin closes over.
export const glowBridge: GlowBridge = {
  enabled: true,
  restrictToLivePreview: true,
  glowAmbiguous: false,
  glowAll: false,
  titleIndex: null,
  app: null,
  activePath: null,
  insert: null,
};

// Resolve the file PATH that owns a given EditorView by matching its CM6 instance
// against each markdown leaf's editor (`editor.cm === view`). This makes the glow
// split-pane correct: every pane scopes self-exclusion + skip ranges to its OWN
// note, and the per-path compiled-regex cache no longer thrashes between panes.
// Falls back to the global active path if no leaf matches.
function pathForView(view: EditorView): string | null {
  const app = glowBridge.app;
  if (!app) return glowBridge.activePath;
  let match: string | null = null;
  app.workspace.iterateAllLeaves((leaf) => {
    if (match) return;
    const v = leaf.view;
    if (v instanceof MarkdownView) {
      // The CM6 EditorView lives at editor.cm on Obsidian's editor wrapper.
      const cm = (v.editor as unknown as { cm?: EditorView }).cm;
      if (cm === view && v.file) match = v.file.path;
    }
  });
  return match ?? glowBridge.activePath;
}

// A half-open span [from, to) the glow must never overlap (frontmatter, existing
// links, code, the H1 title line).
interface Span {
  from: number;
  to: number;
}

// One detected, link-able mention in the document.
export interface Mention {
  from: number;
  to: number;
  surface: string;
}

// True when [from, to) overlaps any skip span. Spans are sorted ascending by
// `from`; we early-out once a span starts past `to`.
function overlapsAny(from: number, to: number, spans: Span[]): boolean {
  for (const s of spans) {
    if (s.from >= to) break;
    if (s.to > from) return true;
  }
  return false;
}

// Compute the skip ranges for a whole document ONCE: frontmatter, existing
// `[[wikilinks]]` and `[text](url)` links, fenced + inline code, and the H1
// title line. Any glow hit overlapping one of these is dropped. Returns spans
// sorted ascending by `from`.
//
// Fail-closed: anything we cannot parse cleanly (e.g. an unclosed fence) extends
// the skip region to end-of-document so we never glow inside ambiguous code.
export function computeSkipSpans(doc: string): Span[] {
  const spans: Span[] = [];

  // Frontmatter: getFrontMatterInfo gives the authoritative content range.
  const fm = getFrontMatterInfo(doc);
  if (fm.exists) {
    // contentStart is the offset just after the closing `---\n`; skip [0, that).
    spans.push({ from: 0, to: fm.contentStart });
  }

  // Fenced code blocks ``` ... ``` (and ~~~). Tracked line-aware so an unclosed
  // fence skips to EOF (fail-closed).
  const fenceRe = /^[ \t]*(`{3,}|~{3,})/;
  let inFence = false;
  let fenceStart = 0;
  let pos = 0;
  const lines = doc.split("\n");
  for (const line of lines) {
    const lineStart = pos;
    const lineEnd = pos + line.length;
    if (fenceRe.test(line)) {
      if (!inFence) {
        inFence = true;
        fenceStart = lineStart;
      } else {
        inFence = false;
        spans.push({ from: fenceStart, to: lineEnd });
      }
    }
    pos = lineEnd + 1; // +1 for the consumed "\n"
  }
  if (inFence) {
    // Unclosed fence: skip from its start to end-of-document.
    spans.push({ from: fenceStart, to: doc.length });
  }

  // Inline code `...`, existing wikilinks, markdown links, and the H1 line.
  addRegexSpans(spans, doc, /`[^`\n]*`/g);
  // Wikilinks (incl. embeds) — cover the whole `[[...]]` / `![[...]]`.
  addRegexSpans(spans, doc, /!?\[\[[^\]\n]*\]\]/g);
  // Markdown links `[text](url)` and images `![alt](url)`.
  addRegexSpans(spans, doc, /!?\[[^\]\n]*\]\([^)\n]*\)/g);
  // The H1 title line (a `#` heading) — never glow inside the note's own H1.
  addRegexSpans(spans, doc, /^#\s.*$/gm);

  spans.sort((a, b) => a.from - b.from || a.to - b.to);
  return spans;
}

function addRegexSpans(spans: Span[], doc: string, re: RegExp): void {
  re.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(doc)) !== null) {
    if (m[0].length === 0) {
      re.lastIndex++;
      continue;
    }
    spans.push({ from: m.index, to: m.index + m[0].length });
  }
}

// Detect every link-able mention in a document: run the active-path alternation,
// drop hits inside any skip span, and (unless `all`) keep only the FIRST surviving
// occurrence of each target note. Shared by the ViewPlugin, the "link all" command
// and the idle auto-link pass.
export function detectMentions(
  doc: string,
  titleIndex: TitleIndex,
  activePath: string,
  opts: { all: boolean; allowAmbiguous: boolean },
): Mention[] {
  const regex = titleIndex.compiledFor(activePath);
  if (!regex) return [];
  const skip = computeSkipSpans(doc);
  const out: Mention[] = [];
  const seenTargets = new Set<string>();

  regex.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(doc)) !== null) {
    const surface = m[0];
    if (surface.length === 0) {
      regex.lastIndex++;
      continue;
    }
    const from = m.index;
    const to = from + surface.length;
    if (overlapsAny(from, to, skip)) continue;

    const resolved = titleIndex.resolve(surface);
    // Only a surface that resolves to a real note glows; this is what makes ANY
    // hit a real matching note.
    if (!resolved) continue;
    if (resolved.ambiguous && !opts.allowAmbiguous) continue;
    // Never link a note to itself (the alternation already excludes own
    // surfaces, but a basename collision could still resolve here).
    if (resolved.file.path === activePath) continue;

    if (!opts.all) {
      if (seenTargets.has(resolved.file.path)) continue;
      seenTargets.add(resolved.file.path);
    }
    out.push({ from, to, surface });
  }
  return out;
}

// Build the wikilink text for a surface. When the surface is an ALIAS of the
// target (not its basename) we keep the user's surface as the display text:
// `[[Note|Surface]]`; otherwise a plain `[[Note]]`. The target uses the file's
// basename (Obsidian resolves it; this matches the vault's short-link style and
// coexists with easy-links' path hiding without fighting its `Folder/` collapse).
export function buildWikiLink(
  file: TFile,
  surface: string,
  titleIndex: TitleIndex,
): string {
  const target = file.basename;
  if (titleIndex.isAlias(surface, file)) {
    return `[[${target}|${surface}]]`;
  }
  return `[[${target}]]`;
}

// Apply a set of mentions as wikilinks to an editor in ONE undo step, from LAST
// offset to FIRST (descending) so earlier replacements don't shift later ranges.
// Each range is RE-VALIDATED against the current text right before applying (the
// surface must still match exactly) so an edit between detection and apply never
// clobbers the note. Returns the number of links inserted.
export function applyMentions(
  editor: Editor,
  mentions: Mention[],
  titleIndex: TitleIndex,
): number {
  if (mentions.length === 0) return 0;
  // Descending by start offset.
  const ordered = mentions.slice().sort((a, b) => b.from - a.from);
  let applied = 0;
  for (const mention of ordered) {
    const resolved = titleIndex.resolve(mention.surface);
    if (!resolved) continue;
    const fromPos = editor.offsetToPos(mention.from);
    const toPos = editor.offsetToPos(mention.to);
    // Re-validate: the live text in the range must still equal the surface.
    if (editor.getRange(fromPos, toPos) !== mention.surface) continue;
    const link = buildWikiLink(resolved.file, mention.surface, titleIndex);
    editor.replaceRange(link, fromPos, toPos);
    applied++;
  }
  return applied;
}

// --- the ViewPlugin ---------------------------------------------------------

// Build the glow DecorationSet for the current viewport.
function buildGlow(view: EditorView): DecorationSet {
  const b = glowBridge;
  if (!b.enabled || !b.titleIndex) return Decoration.none;
  // Live-preview gate (recommended default): the mark is class-only, so source
  // mode is harmless, but the reading flow lives in live preview.
  if (b.restrictToLivePreview && !view.state.field(editorLivePreviewField)) {
    return Decoration.none;
  }

  // Resolve the path PER VIEW (split-pane-safe) rather than from one global.
  const activePath = pathForView(view);
  if (!activePath) return Decoration.none;

  const titleIndex = b.titleIndex;
  const regex = titleIndex.compiledFor(activePath);
  if (!regex) return Decoration.none;

  const doc = view.state.doc;
  // Materialize the document string ONCE and reuse it for both the skip-span scan
  // and the alternation scan below — never re-serialize the CM6 rope inside the
  // match loop (that was O(docLen * matches) on every keystroke).
  const text = doc.toString();
  // Skip ranges are computed over the WHOLE document once (cheap regex spans),
  // so a match near a viewport edge is still correctly rejected.
  const skip = computeSkipSpans(text);

  const builder = new RangeSetBuilder<Decoration>();
  // First-occurrence-per-target is enforced across the WHOLE doc scan (not just
  // the viewport) so scrolling can't reveal a second glow for the same note.
  const seenTargets = new Set<string>();

  // Pre-scan the full document for the ordered set of glow ranges, then add only
  // those inside a visible range (RangeSetBuilder needs ascending order, which a
  // single left-to-right regex scan already gives us).
  regex.lastIndex = 0;
  let m: RegExpExecArray | null;
  const visible = view.visibleRanges;
  while ((m = regex.exec(text)) !== null) {
    const surface = m[0];
    if (surface.length === 0) {
      regex.lastIndex++;
      continue;
    }
    const from = m.index;
    const to = from + surface.length;
    if (overlapsAny(from, to, skip)) continue;

    const resolved = titleIndex.resolve(surface);
    if (!resolved) continue;
    if (resolved.ambiguous && !b.glowAmbiguous) continue;
    if (resolved.file.path === activePath) continue;

    if (!b.glowAll) {
      if (seenTargets.has(resolved.file.path)) continue;
      seenTargets.add(resolved.file.path);
    }

    // Only paint ranges that intersect a visible range (perf), but the
    // first-occurrence bookkeeping above already consumed earlier occurrences.
    const isVisible = visible.some((r) => from < r.to && to > r.from);
    if (!isVisible) continue;
    builder.add(from, to, GLOW_MARK);
  }

  return builder.finish();
}

// The ViewPlugin: rebuilds on doc/viewport change (NOT on plain selection moves —
// buildGlow has no cursor-aware logic, so a selection-only rebuild would produce
// an identical DecorationSet and waste CPU on every arrow key) and binds a single
// mousedown handler (eventHandlers spec, NOT per-DOM listeners) that converts a
// clicked glow into a link via the bridge.
export const glowPlugin = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;

    constructor(view: EditorView) {
      this.decorations = buildGlow(view);
    }

    update(u: ViewUpdate): void {
      if (u.docChanged || u.viewportChanged) {
        this.decorations = buildGlow(u.view);
      }
    }
  },
  {
    decorations: (v) => v.decorations,
    eventHandlers: {
      mousedown(this: { decorations: DecorationSet }, e: MouseEvent, view: EditorView) {
        if (e.button !== 0) return false; // left-click only
        const titleIndex = glowBridge.titleIndex;
        if (!titleIndex) return false;

        // Resolve the click to a document offset from POINTER COORDINATES, not from
        // e.target. In live preview the glow <span> is routinely wrapped by (or wraps)
        // other CM/Obsidian spans, so the clicked node is usually a CHILD that does NOT
        // carry the glow class — the old classList check on e.target failed there and
        // the click silently did nothing. posAtCoords is robust to that nesting.
        const pos = view.posAtCoords({ x: e.clientX, y: e.clientY });
        if (pos === null) return false;

        // Is that offset inside a glow range? (between() yields ranges overlapping the
        // point; require pos strictly inside [from, to) so a click just past the word
        // doesn't convert it.)
        let found: { from: number; to: number } | null = null;
        this.decorations.between(pos, pos, (from, to) => {
          if (from <= pos && pos < to) {
            found = { from, to };
            return false; // stop iterating
          }
          return undefined;
        });
        if (!found) return false;
        const range = found as { from: number; to: number };

        // Re-validate against the live doc (it may have changed since buildGlow) and
        // resolve the target, then apply the wikilink with the CM6 view DIRECTLY —
        // same offsets, no Obsidian-editor round-trip (offsetToPos translation + an
        // active-editor lookup were extra silent-failure points).
        const surface = view.state.doc.sliceString(range.from, range.to);
        const resolved = titleIndex.resolve(surface);
        if (!resolved) return false;
        const link = buildWikiLink(resolved.file, surface, titleIndex);

        e.preventDefault();
        view.dispatch({
          changes: { from: range.from, to: range.to, insert: link },
          selection: { anchor: range.from + link.length },
        });
        return true;
      },
    },
  },
);
