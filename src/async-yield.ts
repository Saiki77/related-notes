// A "yield to the renderer" primitive: lets Obsidian paint and process input
// between chunks of CPU work (e.g. WASM embedding passes during a reindex).
//
// We deliberately do NOT use setTimeout(0). Chromium clamps nested setTimeout to
// >=4ms AND throttles timers hard (down to ~1 per second) when the window is
// unfocused or the renderer is busy. A reindex does many yields, so on a loaded
// Obsidian window (Dataview/Excalidraw/etc.) that throttling turned a few-second
// index into MINUTES of waiting with the CPU idle and the whole app laggy.
//
// A MessageChannel message is a macrotask that still yields to paint/input but is
// NOT subject to that clamping/throttling — the same technique React's scheduler
// uses for cooperative work. We keep one channel and a FIFO queue of resolvers so
// concurrent yielders each resume in order, one macrotask apart.
let sharedPort: MessagePort | null = null;
const pendingResolvers: Array<() => void> = [];

function channelPort(): MessagePort {
  if (sharedPort) return sharedPort;
  const channel = new MessageChannel();
  channel.port1.onmessage = () => {
    // Resolve exactly one waiter per message, in FIFO order.
    const resolve = pendingResolvers.shift();
    if (resolve) resolve();
  };
  channel.port1.start();
  sharedPort = channel.port2;
  return sharedPort;
}

// Resolve on the next macrotask (after a paint opportunity), without setTimeout
// throttling. Safe to await in a tight loop.
export function yieldToUI(): Promise<void> {
  const port = channelPort();
  return new Promise<void>((resolve) => {
    pendingResolvers.push(resolve);
    port.postMessage(0);
  });
}
