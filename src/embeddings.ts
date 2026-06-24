import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import { ORT_WEB_CDN } from "./ort-version";

// --- transformers.js environment ---------------------------------------------
// Force the WEB/WASM resolution path inside the Electron renderer. transformers.js
// otherwise detects the renderer as Node (apis.IS_NODE_ENV === true) and takes the
// filesystem/return_path code path, which throws "Unable to return path for
// response" in Obsidian. Setting these BEFORE the first pipeline() call avoids it.
//   - allowRemoteModels: pull weights from the HF Hub (no model files shipped).
//   - allowLocalModels:  off, so the Node FS/getModelFile path is never taken.
//   - useBrowserCache:   cache weights in the browser Cache API (persists, works
//                        in the renderer) so the model downloads exactly once.
//   - useFSCache:        off — the Node disk cache is the source of the bug above.
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

function configureEnv(): void {
  if (envConfigured) return;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  env.useFSCache = false;
  // `env.backends.onnx.wasm` is typed optional; it is always present at runtime,
  // but guard the access so the build stays strict-null clean.
  const wasm = env.backends?.onnx?.wasm;
  if (wasm) {
    wasm.wasmPaths = wasmBaseUrl ?? ORT_WEB_CDN;
    // The renderer is not cross-origin isolated (no COOP/COEP), so threaded WASM
    // can't allocate SharedArrayBuffer — run single-threaded. `numThreads` is a
    // valid ORT option not in the published typings, so set it via a narrow cast.
    (wasm as { numThreads?: number }).numThreads = 1;
    wasm.proxy = false;
  }
  envConfigured = true;
}

// Device preference for the inference backend.
export type DevicePref = "auto" | "webgpu" | "wasm" | "cpu";

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

// One feature-extraction engine, bound to a single model id. The plugin holds one
// instance and rebuilds it when the model id or device preference changes.
export class EmbeddingEngine {
  readonly modelId: string;
  readonly devicePref: DevicePref;
  private pipePromise: Promise<FeatureExtractionPipeline> | null = null;
  private resolvedDevice: "webgpu" | "wasm" | "cpu" | null = null;

  constructor(modelId: string, devicePref: DevicePref) {
    this.modelId = modelId;
    this.devicePref = devicePref;
  }

  // The device the pipeline actually initialised on, once init() has resolved.
  get device(): "webgpu" | "wasm" | "cpu" | null {
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
  // every later caller awaits the same promise. WebGPU is tried first (unless the
  // user pinned a device), with a WASM fallback in a try/catch. If WASM itself
  // also fails (CDN blocked, offline first-run, wasm/glue mismatch) the rejected
  // promise propagates to the caller, which surfaces a Notice — the failure is
  // never swallowed into a silent keyword-only mode.
  init(onProgress?: ProgressCallback): Promise<FeatureExtractionPipeline> {
    if (this.pipePromise) return this.pipePromise;
    configureEnv();

    this.pipePromise = (async () => {
      const build = (device: "webgpu" | "wasm" | "cpu") => {
        this.resolvedDevice = device;
        return pipeline("feature-extraction", this.modelId, {
          device,
          // q8 keeps the download small and fast on CPU/WASM; fp32 is the WebGPU
          // default (accuracy over a slightly larger download).
          dtype: device === "webgpu" ? "fp32" : "q8",
          progress_callback: onProgress,
        });
      };

      // Desktop Obsidian is Electron, so transformers.js loads onnxruntime-NODE,
      // whose only execution provider is "cpu" — "webgpu"/"wasm" THROW there
      // (that's the "Unsupported device" error). Under Node we therefore go
      // straight to native CPU (fast); only a pure-web context (no Node) gets the
      // WebGPU→WASM path. Every order ends in "cpu" so init can't dead-end.
      const isNode =
        typeof process !== "undefined" && !!process.versions?.node;

      let order: Array<"webgpu" | "wasm" | "cpu">;
      if (this.devicePref === "cpu") {
        order = ["cpu"];
      } else if (this.devicePref === "wasm") {
        order = ["wasm", "cpu"];
      } else if (this.devicePref === "webgpu") {
        order = ["webgpu", "cpu"];
      } else if (isNode) {
        order = ["cpu"];
      } else {
        order = (await EmbeddingEngine.webgpuAvailable())
          ? ["webgpu", "wasm", "cpu"]
          : ["wasm", "cpu"];
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
    const out = await pipe(inputs, { pooling: "mean", normalize: true });
    const data = out.data as Float32Array;
    const n = texts.length;
    // dims is [N, D]; derive D defensively from the flat length when shapes drift.
    const dims =
      Array.isArray(out.dims) && out.dims.length === 2
        ? out.dims[1]
        : Math.floor(data.length / n);
    const result: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      result.push(new Float32Array(data.subarray(i * dims, (i + 1) * dims)));
    }
    return result;
  }
}

// Cosine similarity for two already-L2-normalized vectors == their dot product.
// Returns 0 on a length mismatch (e.g. a stale vector from a different model).
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}
