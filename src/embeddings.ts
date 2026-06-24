import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import { ORT_WEB_CDN, ORT_GLUE_JSEP } from "./ort-version";

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

// The onnxruntime-web wasm flags. ort reads these from ort.env.wasm. wasmPaths may
// be a directory string OR a { mjs, wasm } map pointing the glue and binary at
// distinct URLs.
interface OrtWasmFlags {
  wasmPaths?: string | { mjs?: string; wasm?: string };
  numThreads?: number;
  proxy?: boolean;
}

// Created lazily from the patched glue text, reused for the session.
let glueBlobUrl: string | null = null;

function configureEnv(): void {
  if (envConfigured) return;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  env.useBrowserCache = true;

  // The .wasm comes from the self-hosted ort/ folder when present, else the
  // version-pinned CDN. The GLUE (.mjs), however, is served from a Blob of our
  // PATCHED copy (gen-ort forces its Node check off) so onnxruntime-web takes the
  // pure-web path and never imports the Node-only 'worker_threads' — that import,
  // triggered because Obsidian's renderer has process.versions.node set and
  // process.type !== "renderer" (read-only, can't flip), was the "no available
  // backend found" error. ort accepts wasmPaths = { mjs, wasm }.
  const wasmDir = wasmBaseUrl ?? ORT_WEB_CDN;
  glueBlobUrl ??= URL.createObjectURL(
    new Blob([ORT_GLUE_JSEP], { type: "text/javascript" }),
  );
  const wasmPaths = {
    mjs: glueBlobUrl,
    wasm: `${wasmDir}ort-wasm-simd-threaded.jsep.wasm`,
  };

  // ort reads its flags from ort.env.wasm, which transformers exposes as
  // env.backends.onnx.env.wasm in the WEB build (NOT env.backends.onnx.wasm —
  // that's the Node-build shape). Set on BOTH so the right object is always hit.
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
      let order: Array<"webgpu" | "wasm">;
      if (this.devicePref === "wasm") {
        order = ["wasm"];
      } else if (this.devicePref === "webgpu") {
        // Honour the explicit pin, but still fall back to wasm if the adapter or
        // the webgpu session fails to come up.
        order = ["webgpu", "wasm"];
      } else {
        // "auto": only try WebGPU when a real adapter is present, else wasm.
        order = (await EmbeddingEngine.webgpuAvailable())
          ? ["webgpu", "wasm"]
          : ["wasm"];
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
