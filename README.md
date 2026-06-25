# Smart Related Notes

A left-sidebar panel that surfaces the notes most **semantically similar** to the
one you're reading — so you browse your vault by meaning, not by folders. Open any
note and the panel ranks the rest by how closely they relate, as a stack of cards
you can click to jump to.

It's powered by a small **multilingual embedding model that runs entirely on your
machine** — no cloud, no API key, no second app or server. Everything happens
locally inside the renderer, so your notes never leave your computer. After a
**one-time** model download (cached on first use), it works **fully offline**. The
model understands German, English, and 100+ other languages, so matches cross
languages naturally.

## How it works

Each Markdown note (its title plus the first ~1500 characters of cleaned body text)
is turned into a vector — a list of numbers that captures its meaning — by the
embedding model. For the note you're viewing, every other note is ranked by
**cosine similarity** to it, and the closest matches are shown as cards with a
similarity percentage.

The model runs through [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
on the local ONNX runtime. By default it uses your **GPU via WebGPU** when one is
available — a full reindex takes seconds — and falls back to **multi-threaded WASM**
on the CPU otherwise (still fast, no GPU needed). Both are memory-stable. The
runtime's `.wasm` is **shipped inside the plugin**, so the only network traffic ever
is the one-time download of the model weights from the Hugging Face Hub; afterwards
the weights are cached and nothing is fetched again.

Vectors persist as compact JSON in the plugin's config dir, so the index survives
restarts and only changed notes are re-embedded.

## Features

- **Semantic ranking** — for the active note, ranks every other note by cosine
  similarity and shows the top matches as cards: **title**, muted **folder path**,
  a short **snippet**, and a **similarity %** pill. Click a card to open that note.
- **Fully local & private** — embeddings run in-app on your GPU (WebGPU) or CPU
  (WASM); notes are never sent anywhere. Works offline after the one-time download.
- **Multilingual** — a multilingual model matches notes across German, English, and
  100+ other languages.
- **Persisted index** — vectors are saved to disk, so reopening the vault is instant
  and doesn't re-embed everything.
- **Incremental updates** — changed, created, and renamed notes are re-embedded on a
  20-second idle pause, so typing never kicks off work mid-edit.
- **Keyword fallback** — while the index is still building (or for a brand-new note
  with no vector yet), the panel falls back to a cheap keyword / tag / link-overlap
  ranking — shown with a `~` pill — so it's never empty.
- **Clear status** — a live status line shows indexing progress; if the model can't
  load (e.g. no connection on first run), it surfaces an error instead of silently
  showing nothing.

## Settings

- **Performance profile** — one-click presets. **Balanced** is lighter and faster;
  **Best quality** uses a larger model and more context for the strongest matches.
- **Model** — the embedding model (a dropdown of vetted choices, or paste a custom
  Hugging Face id). The default `paraphrase-multilingual-MiniLM-L12-v2` is a
  symmetric sentence-similarity model — the right tool for ranking how alike two
  notes are. `paraphrase-multilingual-mpnet-base-v2` (Best quality) is stronger but
  larger. e5 models are retrieval-oriented and rank note similarity less well.
- **Compute device** — **Auto** (recommended) uses the **GPU (WebGPU)** when
  available — fastest by far — and falls back to **multi-threaded WASM** on the CPU.
  Pin **WebGPU** or **WASM** to force one. Both are memory-stable; switching
  re-downloads the model for that backend.
- **Number of results** — how many cards to show.
- **Minimum similarity** — hide matches below this topical-similarity score (0–1).
  Scores are mean-centered (the embedding noise floor is removed), so unrelated notes
  sit near 0 and ~0.2 cleanly separates on-topic notes. Raise for a tighter list.
- **Embed character limit** — how much of each note's body to embed after the title.
- **Excluded folders** — folders to leave out of the index (and everything beneath
  them), one per line or comma-separated.
- **Show snippet** — toggle the per-card text preview.
- **Rebuild index** — force a full re-embed (also on the command palette and the
  panel's refresh icon).

Changing the model or compute device transparently rebuilds the index; unrelated
changes (sliders, toggles) never trigger a re-embed.

## Install

### From a release

Download `related-notes.zip` from the latest release and extract it into
`.obsidian/plugins/related-notes/`. The zip already includes the ONNX runtime
`.wasm` in its `ort/` folder, so nothing extra is needed — only the model weights
are fetched, once, on first use.

### With BRAT

Add this repository in [BRAT](https://github.com/TfTHacker/obsidian42-brat) and
enable **Smart Related Notes** from the community-plugins list.

On first launch the model weights download from the Hugging Face Hub with a progress
notice, then cache. This happens once; after that the plugin works offline. The
WebGPU (GPU) path uses fp32 weights (~470 MB for the default model); the WASM (CPU)
path uses smaller quantized weights (~110 MB). The model is downloaded once per
backend, then cached.

## Requirements

- Desktop only (the embedding runtime needs a desktop Electron environment).
- Obsidian 1.7.2 or newer.

## Development

A TypeScript project bundled with esbuild (entry `src/main.ts` → root `main.js`).

```bash
npm install          # install dev dependencies
npm run dev          # esbuild watch build (inline sourcemap, no minify)
npm run build        # gen-ort -> tsc --noEmit -> minified production bundle
npm run lint         # eslint (typescript-eslint + eslint-plugin-obsidianmd)
```

`gen-ort.mjs` runs before tsc/lint/esbuild: it writes `src/ort-version.ts` (the
pinned `onnxruntime-web` version + a CDN fallback URL) and copies the matching
`onnxruntime-web` `.wasm`/`.mjs` assets into `ort/`. Both are build artifacts and are
gitignored. The release workflow packages `main.js`, `manifest.json`, `styles.css`,
and the `ort/` folder into `related-notes.zip`.

The renderer reports itself as a Node environment, which would otherwise make
transformers.js pick the (externalized, unavailable) `onnxruntime-node` backend.
`src/ort-shim.ts` — imported first in `main.ts` — installs the bundled
`onnxruntime-web` under `Symbol.for("onnxruntime")` before transformers loads, so the
web runtime is used and WebGPU/WASM work.

## License

MIT © 2026 Saiki77
