// MUST be the very first import in main.ts, before anything that pulls in
// @huggingface/transformers.
//
// Why this exists: in Obsidian's Electron renderer, `process.release.name` is
// "node", so transformers.js computes IS_NODE_ENV = true (once, at import time:
// `process?.release?.name === "node"`) and selects the onnxruntime-NODE backend.
// But onnxruntime-node is externalised — no native binding loads in the renderer —
// so its InferenceSession is undefined. The first pipeline() call then throws
// "Cannot read properties of undefined (reading 'create')", and because the Node
// branch is taken, the supported-device list is built for Node too, so "webgpu"
// and "wasm" are later rejected as unsupported ("Should be one of: .").
//
// We are really in a browser-like renderer (DOM + WebGPU + WASM are all present),
// so the WEB backend is the correct one: transformers' bundled onnxruntime-web,
// with a real device list of [webgpu?, wasm]. transformers decides node-vs-web
// exactly once, from `process.release.name`, at the moment its backend module is
// imported. We flip that single value to a non-"node" string just for that import,
// then restore it on the next microtask — which runs only after the whole
// synchronous module-eval (transformers included) has finished, so IS_NODE_ENV is
// already captured and nothing else ever observes the change.
//
// NOTE: an earlier approach installed onnxruntime-web under
// Symbol.for("onnxruntime"). transformers checks that symbol FIRST and, when
// present, uses it as the runtime but skips building the device list entirely —
// leaving supportedDevices empty, so every device threw "Should be one of: .".
// Flipping IS_NODE_ENV instead takes the real web branch, which both selects
// onnxruntime-web AND populates the device list.
const release =
  typeof process !== "undefined"
    ? (process.release as { name?: string } | undefined)
    : undefined;

if (release && release.name === "node") {
  const original = release.name;
  try {
    release.name = "obsidian-renderer";
  } catch {
    // If the property is read-only on this platform, fall back to defineProperty;
    // if that also fails the plugin surfaces its normal init error.
    try {
      Object.defineProperty(release, "name", {
        value: "obsidian-renderer",
        configurable: true,
        writable: true,
      });
    } catch {
      /* best effort */
    }
  }
  // Restore after transformers has read IS_NODE_ENV (next microtask = after the
  // current synchronous module-eval, where the static import chain runs).
  queueMicrotask(() => {
    try {
      release.name = original;
    } catch {
      /* best effort */
    }
  });
}
