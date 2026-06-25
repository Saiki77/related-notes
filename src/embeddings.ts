import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import { ORT_WEB_CDN, ORT_GLUE_JSEP } from "./ort-version";
import { yieldToUI } from "./async-yield";
// cosineSimilarity now lives in the dependency-free vector-math module (so the
// node benchmark can import it without pulling in transformers). Re-exported here
// for back-compat with existing importers and the `cosineSimilarity` symbol.
export { cosineSimilarity } from "./vector-math";

// --- transformers.js environment ---------------------------------------------
// The onnxruntime-WEB backend is forced in ort-shim.ts (imported first in
// main.ts), which installs our bundled ort-web under Symbol.for("onnxruntime")
// before transformers loads. That makes InferenceSession defined and rebuilds the
// device list as [webgpu?, wasm] — the renderer no longer misdetects as Node and
// no longer dead-ends on the invalid "cpu" device. Here we only configure env
// (cache + wasm source) before the first pipeline() call.
//   - allowRemoteModels: pull weights from the HF Hub (no model files shipped).
//   - allowLocalModels:  off, so the Node FS/getModelFile path is never taken.
//   - useBrowserCache:   cache weights in the browser Cache API (persists, works
//                        in the renderer) so the model downloads exactly once.
//   - wasm.wasmPaths:    where the onnxruntime-web .wasm is fetched from. We point
//                        this at the plugin's SELF-HOSTED ort/ folder (passed in as
//                        an app:// resource URL) so the .wasm is the exact build
//                        that matches the JS glue transformers bundled, and so the
//                        plugin works fully offline. If no local URL is available
//                        we fall back to a version-PINNED CDN dir (ORT_WEB_CDN is
//                        generated at build time from the resolved package version,
//                        so it can never drift from the shipped glue).
//   - wasm.numThreads=1: the renderer is not cross-origin isolated, so threaded
//                        WASM (SharedArrayBuffer) is unavailable — run single-threaded.
let envConfigured = false;

// Set by the plugin (from adapter.getResourcePath) before the first init(). When
// present it overrides the CDN fallback. Must end in a trailing slash.
let wasmBaseUrl: string | null = null;

// Point onnxruntime-web at a locally-served directory of .wasm files. The plugin
// resolves this from its own folder via the vault adapter, guaranteeing the .wasm
// matches the bundled glue and that the plugin runs offline. Call before init().
export function setWasmBaseUrl(url: string): void {
  wasmBaseUrl = url.endsWith("/") ? url : `${url}/`;
}

// The onnxruntime-web wasm flags. ort reads these from ort.env.wasm in the web
// build, which transformers exposes as env.backends.onnx.env.wasm (NOT
// env.backends.onnx.wasm — that's the Node-build shape). wasmPaths may be a
// directory string OR a { mjs, wasm } map.
interface OrtWasmFlags {
  wasmPaths?: string | { mjs?: string; wasm?: string };
  numThreads?: number;
  proxy?: boolean;
}

// Blob URL of the web-patched glue, created once and reused for the session.
let glueBlobUrl: string | null = null;

function configureEnv(): void {
  if (envConfigured) return;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  // ort loads the JS GLUE from wasmPaths.mjs and the .wasm binary from
  // wasmPaths.wasm. The bundled ort-web fetches the glue EXTERNALLY (it is not
  // inlined), and the stock glue imports the Node-only 'worker_threads' on load
  // (two separate Node checks) — which kills init in Obsidian's renderer with
  // "no available backend found". So we serve our PATCHED glue (gen-ort forced
  // both Node checks off) from a Blob, and point .wasm at the self-hosted ort/
  // folder when present else the version-pinned CDN.
  const wasmDir = wasmBaseUrl ?? ORT_WEB_CDN;
  glueBlobUrl ??= URL.createObjectURL(
    new Blob([ORT_GLUE_JSEP], { type: "text/javascript" }),
  );
  const wasmPaths = {
    mjs: glueBlobUrl,
    wasm: `${wasmDir}ort-wasm-simd-threaded.jsep.wasm`,
  };
  // Set the flags on BOTH shapes so the object ort actually reads is always hit.
  const onnx = env.backends?.onnx as
    | { wasm?: OrtWasmFlags; env?: { wasm?: OrtWasmFlags } }
    | undefined;
  for (const wasm of [onnx?.wasm, onnx?.env?.wasm]) {
    if (!wasm) continue;
    wasm.wasmPaths = wasmPaths;
    // Single-threaded, no proxy worker — run the wasm inline on the main thread.
    wasm.numThreads = 1;
    wasm.proxy = false;
  }
  envConfigured = true;
}

// Device preference for the inference backend. The onnxruntime-web backend only
// supports "webgpu" (when navigator.gpu is present) and "wasm" — there is NO
// "cpu" provider in the web runtime ("wasm" IS the CPU path). "auto" probes
// WebGPU and falls back to WASM.
export type DevicePref = "auto" | "webgpu" | "wasm";

// Progress events surfaced from pipeline() during the one-time model download.
// transformers.js types this as `any`; narrow it to just the fields we read.
export interface ProgressInfo {
  // One of "initiate" | "download" | "progress" | "done" | "ready", but typed as
  // a plain string since transformers.js may add states and we only switch on a few.
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
  progress?: number;
}

export type ProgressCallback = (info: ProgressInfo) => void;

// Models that need no "query:"/"passage:" prefix (paraphrase-* family). Everything
// else (the e5 family) requires it. Keyed by a substring of the model id so a user
// can paste either the Xenova/* port or any compatible mirror.
function modelNeedsPrefix(modelId: string): boolean {
  return !/paraphrase-multilingual/i.test(modelId);
}

export type EmbedKind = "query" | "passage";

// embedBatch processes its inputs in SUB-BATCHES of this many texts, awaiting a
// macrotask yield between sub-batches so the renderer can paint/handle input
// instead of being blocked by one long synchronous WASM forward pass over a whole
// outer batch (~130+ chunk texts froze the app). A sub-batch of ~12 keeps each
// pipe() call short while staying large enough for ORT batch throughput. This only
// changes WHEN pipe() runs — never the order of the returned rows.
const EMBED_SUB_BATCH = 12;

// Yield to the renderer only after this much wall-clock has accrued since the last
// yield, instead of after every sub-batch. On a fast machine many sub-batches fit
// in one budget so we yield rarely (near full throughput); on a slow one we still
// yield often enough to stay responsive. ~20ms ≈ one frame of headroom.
const YIELD_BUDGET_MS = 20;

// One feature-extraction engine, bound to a single model id. The plugin holds one
// instance and rebuilds it when the model id or device preference changes.
export class EmbeddingEngine {
  readonly modelId: string;
  readonly devicePref: DevicePref;
  private pipePromise: Promise<FeatureExtractionPipeline> | null = null;
  private resolvedDevice: "webgpu" | "wasm" | null = null;

  constructor(modelId: string, devicePref: DevicePref) {
    this.modelId = modelId;
    this.devicePref = devicePref;
  }

  // The device the pipeline actually initialised on, once init() has resolved.
  get device(): "webgpu" | "wasm" | null {
    return this.resolvedDevice;
  }

  // Probe WebGPU the way the renderer exposes it. Returns false on any error so a
  // missing/disabled adapter transparently falls back to WASM.
  private static async webgpuAvailable(): Promise<boolean> {
    try {
      const nav = navigator as Navigator & {
        gpu?: { requestAdapter(): Promise<unknown> };
      };
      if (!nav.gpu) return false;
      const adapter = await nav.gpu.requestAdapter();
      return adapter !== null && adapter !== undefined;
    } catch {
      return false;
    }
  }

  // Lazily build the pipeline. Safe to call repeatedly: the first call wins and
  // every later caller awaits the same promise. WebGPU is tried first (only when a
  // real adapter is available), with a WASM fallback. If WASM itself also fails
  // (CDN blocked, offline first-run, wasm/glue mismatch) the rejected promise
  // propagates to the caller, which surfaces a Notice — the failure is never
  // swallowed into a silent keyword-only mode.
  init(onProgress?: ProgressCallback): Promise<FeatureExtractionPipeline> {
    if (this.pipePromise) return this.pipePromise;
    configureEnv();

    this.pipePromise = (async () => {
      const build = (device: "webgpu" | "wasm") => {
        this.resolvedDevice = device;
        return pipeline("feature-extraction", this.modelId, {
          device,
          // q8 keeps the WASM download small and fast; fp32 is the WebGPU default
          // (accuracy over a slightly larger download).
          dtype: device === "webgpu" ? "fp32" : "q8",
          progress_callback: onProgress,
        });
      };

      // The web runtime supports exactly "webgpu" (when an adapter exists) and
      // "wasm". Every order ends in "wasm" — the always-available CPU path — so
      // init can never dead-end on an invalid device.
      //
      // "auto" deliberately uses WASM, NOT WebGPU. onnxruntime-web's WebGPU backend
      // accumulates GPU buffers across the many embed() calls of a full (re)index;
      // on a unified-memory Mac that IS system RAM, and a reindex ballooned to ~70GB
      // and crashed Obsidian. WASM is memory-stable for batch embedding and plenty
      // fast for note-sized inputs. WebGPU remains available only as an EXPLICIT pin
      // for users who accept that cost (and mainly want single-query speed).
      let order: Array<"webgpu" | "wasm">;
      if (this.devicePref === "webgpu") {
        // Honour the explicit pin, but still fall back to wasm if the adapter or
        // the webgpu session fails to come up.
        order = ["webgpu", "wasm"];
      } else {
        // "auto" and "wasm" both resolve to the memory-stable WASM backend.
        order = ["wasm"];
      }

      let lastErr: unknown;
      for (const device of order) {
        try {
          return await build(device);
        } catch (e) {
          lastErr = e;
          console.warn(`[related-notes] ${device} init failed`, e);
        }
      }
      // Nothing worked: reset so a later retry can re-attempt, and surface it.
      this.pipePromise = null;
      this.resolvedDevice = null;
      if (lastErr instanceof Error) throw lastErr;
      throw new Error("No embedding backend available.");
    })();

    return this.pipePromise;
  }

  // Embed one string into a normalized 384-dim Float32Array. Vectors are L2-
  // normalized, so cosine similarity reduces to a dot product. `kind` selects the
  // e5 prefix; it is ignored for prefix-free models.
  async embed(
    text: string,
    kind: EmbedKind = "query",
    onProgress?: ProgressCallback,
  ): Promise<Float32Array> {
    const pipe = await this.init(onProgress);
    const input = modelNeedsPrefix(this.modelId) ? `${kind}: ${text}` : text;
    const out = await pipe(input, { pooling: "mean", normalize: true });
    // `out.data` is the flat tensor; copy into a plain Float32Array so callers can
    // store/persist it without holding onto the tensor backing store.
    return new Float32Array(out.data as Float32Array);
  }

  // Embed a batch in ONE pipeline call. transformers runs the whole batch through
  // a single ORT session pass (real throughput, unlike N awaited single calls that
  // just queue on the same backend). The output is a [N, dims] tensor; slice its
  // flat data into one normalized Float32Array per input. Returns [] for an empty
  // batch. Throws on backend failure (the caller decides how to surface it).
  async embedBatch(
    texts: string[],
    kind: EmbedKind = "passage",
    onProgress?: ProgressCallback,
  ): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const pipe = await this.init(onProgress);
    const prefix = modelNeedsPrefix(this.modelId);
    const inputs = prefix ? texts.map((t) => `${kind}: ${t}`) : texts;

    // SUB-BATCH the forward passes so the main thread is never blocked for long: one
    // pipe() call over a whole outer batch is a single uninterruptible synchronous
    // WASM pass (~seconds for 130+ chunks) that froze Obsidian. We run pipe() per
    // EMBED_SUB_BATCH slice and yield (a MessageChannel macrotask — NOT setTimeout,
    // which Chromium throttles to ~1/sec when unfocused and turned this into minutes
    // of idle waiting) only once YIELD_BUDGET_MS of work has accrued. Rows are
    // concatenated IN INPUT ORDER, so the returned Float32Array[] is byte-identical to
    // the old whole-batch result — build()'s offset regrouping and ranking output are
    // unchanged. A single input (or one sub-batch's worth) takes exactly one pass with
    // NO yield, so embed() and a 1-text batch behave exactly as before.
    const result: Float32Array[] = [];
    let budgetStart = performance.now();
    for (let start = 0; start < inputs.length; start += EMBED_SUB_BATCH) {
      const slice = inputs.slice(start, start + EMBED_SUB_BATCH);
      const out = await pipe(slice, { pooling: "mean", normalize: true });
      const data = out.data as Float32Array;
      const n = slice.length;
      // dims is [N, D]; derive D defensively from the flat length when shapes drift.
      const dims =
        Array.isArray(out.dims) && out.dims.length === 2
          ? out.dims[1]
          : Math.floor(data.length / n);
      for (let i = 0; i < n; i++) {
        result.push(new Float32Array(data.subarray(i * dims, (i + 1) * dims)));
      }
      // Yield between sub-batches, but only after enough work to be worth a macrotask
      // (and never after the final slice). Order/values above are already committed.
      if (
        start + EMBED_SUB_BATCH < inputs.length &&
        performance.now() - budgetStart > YIELD_BUDGET_MS
      ) {
        await yieldToUI();
        budgetStart = performance.now();
      }
    }
    return result;
  }
}
