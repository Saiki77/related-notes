// A/B eval: MiniLM vs BGE-M3 on the user's real vault, scored by link-recall.
// Ground truth = the user's own [[wikilinks]] (notes they judged related). For each
// note with >=1 outlink we rank all others by each config's PURE SEMANTIC similarity
// (no structural/link boost — that would be circular) and measure how many linked
// targets land in the top-K. Reads /tmp/rn_corpus.json (dumped from Obsidian).
//
// Run: node bench/ab-eval.mjs   (downloads BGE-M3 ~568MB to the HF cache on first run)
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync } from "node:fs";

env.allowLocalModels = false;

const MAX_CHUNK_CHARS = 480, TARGET_WORDS = 80, MIN_WINDOW_WORDS = 10, CAP = 40, TITLE_W = 2;
const corpus = JSON.parse(readFileSync("/tmp/rn_corpus.json", "utf8"));

// ---- chunking (compact copy of the plugin's pipeline) ----
function clean(c) {
  let t = c;
  t = t.replace(/^---\n[\s\S]*?\n---\n?/, "\n").replace(/```[\s\S]*?```/g, "\n").replace(/`[^`]*`/g, " ");
  t = t.replace(/!\[\[[^\]]*\]\]/g, " ").replace(/!\[[^\]]*\]\([^)]*\)/g, " ");
  t = t.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
  t = t.replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, "").replace(/[*_~]/g, "").replace(/<[^>]+>/g, " ");
  return t;
}
function strip(c) { return clean(c).split("\n").map((l) => l.replace(/[ \t\f\v]+/g, " ").trimEnd()).join("\n").replace(/\n{3,}/g, "\n\n").trim(); }
const seg = new Intl.Segmenter(undefined, { granularity: "sentence" });
function sents(p) { const o = []; for (const { segment } of seg.segment(p)) { const s = segment.trim(); if (s) o.push(s); } return o; }
function cw(t) { const m = t.match(/\S+/g); return m ? m.length : 0; }
function win(ss) { if (!ss.length) return []; const w = []; let cur = [], c = 0; for (const s of ss) { cur.push(s); c += cw(s); if (c >= TARGET_WORDS) { w.push(cur.join(" ")); const l = cur[cur.length - 1]; cur = [l]; c = cw(l); } } const t = cur.join(" ").trim(); if (t) { if (w.length && cw(t) < MIN_WINDOW_WORDS) w[w.length - 1] = `${w[w.length - 1]} ${t}`.trim(); else if (!w.length || w[w.length - 1] !== t) w.push(t); } return w; }
function budget(text) { if (text.length <= MAX_CHUNK_CHARS) return [text]; const o = []; let bf = ""; const pb = () => { const t = bf.trim(); if (t) o.push(t); bf = ""; }; for (const s of sents(text)) { if (bf.length && bf.length + 1 + s.length > MAX_CHUNK_CHARS) pb(); if (s.length > MAX_CHUNK_CHARS) { pb(); let r = s; while (r.length > MAX_CHUNK_CHARS) { let ct = r.lastIndexOf(" ", MAX_CHUNK_CHARS); if (ct < MAX_CHUNK_CHARS * 0.5) ct = MAX_CHUNK_CHARS; o.push(r.slice(0, ct).trim()); r = r.slice(ct).trim(); } bf = r; } else bf = bf.length ? `${bf} ${s}` : s; } pb(); return o.length ? o : [text]; }
function windowsOf(body) { const out = []; for (const para of strip(body).split(/\n{2,}/).map((p) => p.replace(/\n/g, " ").trim()).filter(Boolean)) for (const w of win(sents(para))) for (const pc of budget(w)) out.push(pc); return out; }

// Build per-note chunk texts (title + capped windows) and whole-body text.
const notes = corpus.map((n) => {
  let ws = windowsOf(n.body);
  if (ws.length > CAP) ws = ws.slice(0, CAP); // simple cap for the eval
  return { path: n.path, basename: n.basename, links: n.links, chunkTexts: [n.basename, ...ws], whole: (n.basename + ". " + strip(n.body)).slice(0, 6000) };
});
const pathIndex = new Map(notes.map((n, i) => [n.path, i]));

// ---- vector math ----
function l2(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function meanOf(vs) { const d = vs[0].length, m = new Array(d).fill(0); for (const v of vs) for (let i = 0; i < d; i++) m[i] += v[i]; for (let i = 0; i < d; i++) m[i] /= vs.length; return l2(m); }
function centroidOf(means) { const d = means[0].length, c = new Array(d).fill(0); for (const m of means) for (let i = 0; i < d; i++) c[i] += m[i]; for (let i = 0; i < d; i++) c[i] /= means.length; return l2(c); }
function center(v, c) { const p = dot(v, c); return l2(v.map((x, i) => x - p * c[i])); }

async function embedAll(extractor, texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 32) {
    const batch = texts.slice(i, i + 32);
    const t = await extractor(batch, { pooling: "mean", normalize: true });
    out.push(...t.tolist());
  }
  return out;
}

// biMax over centered chunk arrays (title row 0 weighted TITLE_W), symmetric.
function dirMax(x, y) { let num = 0, den = 0; for (let i = 0; i < x.length; i++) { let best = -1; for (const yj of y) { const d = dot(x[i], yj); if (d > best) best = d; } const w = i === 0 ? TITLE_W : 1; num += w * Math.max(0, best); den += w; } return den ? num / den : 0; }
function biMax(a, b) { return (dirMax(a, b) + dirMax(b, a)) / 2; }

function evalConfig(name, sim) {
  const Ks = [5, 10, 20];
  const rec = { 5: 0, 10: 0, 20: 0 };
  let mrr = 0, n = 0;
  for (let i = 0; i < notes.length; i++) {
    const targets = notes[i].links.map((p) => pathIndex.get(p)).filter((j) => j !== undefined && j !== i);
    if (!targets.length) continue;
    n++;
    const ranked = [];
    for (let j = 0; j < notes.length; j++) if (j !== i) ranked.push([j, sim(i, j)]);
    ranked.sort((a, b) => b[1] - a[1]);
    const order = ranked.map((r) => r[0]);
    const tset = new Set(targets);
    for (const K of Ks) { let hit = 0; for (let k = 0; k < Math.min(K, order.length); k++) if (tset.has(order[k])) hit++; rec[K] += hit / targets.length; }
    let fr = 0; for (let k = 0; k < order.length; k++) if (tset.has(order[k])) { fr = k + 1; break; }
    mrr += fr ? 1 / fr : 0;
  }
  return { name, notes: n, r5: rec[5] / n, r10: rec[10] / n, r20: rec[20] / n, mrr: mrr / n };
}

async function runWindows(modelId, dtype, label) {
  process.stdout.write(`\n[${label}] loading ${modelId} (${dtype})...\n`);
  const extractor = await pipeline("feature-extraction", modelId, { dtype });
  const flat = []; const offs = [0];
  for (const note of notes) { for (const t of note.chunkTexts) flat.push(t); offs.push(flat.length); }
  process.stdout.write(`[${label}] embedding ${flat.length} windows...\n`);
  const vecs = await embedAll(extractor, flat);
  const noteChunks = notes.map((_, i) => vecs.slice(offs[i], offs[i + 1]).map(l2));
  const means = noteChunks.map(meanOf);
  const c = centroidOf(means);
  const cChunks = noteChunks.map((ch) => ch.map((v) => center(v, c)));
  return evalConfig(label, (i, j) => biMax(cChunks[i], cChunks[j]));
}

async function runWhole(modelId, dtype, label) {
  process.stdout.write(`\n[${label}] loading ${modelId} (${dtype})...\n`);
  const extractor = await pipeline("feature-extraction", modelId, { dtype });
  process.stdout.write(`[${label}] embedding ${notes.length} whole notes...\n`);
  const vecs = (await embedAll(extractor, notes.map((n) => n.whole))).map(l2);
  const c = centroidOf(vecs);
  const cv = vecs.map((v) => center(v, c));
  return evalConfig(label, (i, j) => dot(cv[i], cv[j]));
}

const results = [];
results.push(await runWindows("Xenova/paraphrase-multilingual-MiniLM-L12-v2", "q8", "MiniLM / windows+biMax (current)"));
results.push(await runWindows("Xenova/bge-m3", "q8", "BGE-M3 / windows+biMax"));
results.push(await runWhole("Xenova/bge-m3", "q8", "BGE-M3 / whole-note single vec"));

console.log("\n\n===== LINK-RECALL A/B (higher = better; ground truth = your wikilinks) =====");
console.log("notes scored:", results[0].notes, "\n");
const pad = (s, n) => String(s).padEnd(n);
console.log(pad("config", 36), pad("R@5", 8), pad("R@10", 8), pad("R@20", 8), "MRR");
for (const r of results) console.log(pad(r.name, 36), pad(r.r5.toFixed(3), 8), pad(r.r10.toFixed(3), 8), pad(r.r20.toFixed(3), 8), r.mrr.toFixed(3));
