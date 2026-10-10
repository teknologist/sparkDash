/**
 * Load progress for a model whose container is up but whose API is not serving
 * yet. A big model takes minutes to come up and its API is dark the whole time,
 * so the board needs two other signals to show something better than "loading":
 *
 *  - the PHASE, read from the tail of the container log (vLLM and the GLM
 *    launcher print recognisable lines at each step), with vLLM's own shard
 *    percentage when it prints one;
 *  - an ETA, from how long this brick took to load the last few times
 *    (container start → API ready), measured by the composer itself.
 */

/**
 * Ordered most-advanced first: the tail can hold lines from several steps, and
 * the latest step wins. Each entry: [regex, phase label].
 */
const PHASES = [
  [/Application startup complete|Uvicorn running|Starting vLLM API server|Available routes are/i, "starting API"],
  [/warm ?-?up/i, "warming up"],
  [/Capturing CUDA graph|cudagraph(s)? captur|Graph capturing finished/i, "capturing CUDA graphs"],
  // Not a bare "KV cache": weight loading logs e.g. "Using fp8_ds_mla KV cache format".
  [/memory profiling|Determining available memory|Available KV cache memory|GPU KV cache size|Setting kv cache block size/i, "sizing KV cache"],
  [/Model loading took|Loading weights took|weights loaded/i, "weights loaded"],
  [/checkpoint shards|Loading weights|load_weights|Loading model|instanttensor|safetensors/i, "loading weights"],
  [/Initializing a V1 LLM engine|init engine|Starting to load model|world_size|nccl|torch\.distributed/i, "initializing engine"],
  [/patched|already present|overlay|placeholders aligned/i, "patching"],
];

/** Phases whose log lines carry a tqdm-style "NN%" worth surfacing. */
const PCT_PHASES = new Set(["loading weights", "capturing CUDA graphs"]);
const PCT_RE = /(\d{1,3})%/;
const PATCHING = "patching";

/**
 * @param {string[]} lines container log tail, oldest first
 * @returns {{ phase: string | null, phasePct: number | null }}
 */
export function parseLoadPhase(lines) {
  let patching = false;
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    for (const [re, phase] of PHASES) {
      if (!re.test(line)) continue;
      // Launcher patch lines interleave with vLLM's own (each worker process
      // re-applies them), so they only describe the load until vLLM speaks.
      if (phase === PATCHING) {
        patching = true;
        break;
      }
      const m = PCT_PHASES.has(phase) ? line.match(PCT_RE) : null;
      const phasePct = m ? Math.min(100, Number(m[1])) : null;
      return { phase, phasePct };
    }
  }
  return { phase: patching ? PATCHING : null, phasePct: null };
}

/** Median of the recorded load durations (seconds), or null with no history. */
export function expectedLoadSec(history) {
  const xs = (Array.isArray(history) ? history : []).filter((x) => Number.isFinite(x) && x > 0);
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * @param {{ startedAtMs: number, now: number, history?: number[], phase?: string|null, phasePct?: number|null }} p
 * @returns {{ phase: string|null, phasePct: number|null, elapsedSec: number, expectedSec: number|null, etaSec: number|null, pct: number|null }}
 */
export function loadProgress({ startedAtMs, now, history, phase = null, phasePct = null }) {
  const elapsedSec = Math.max(0, Math.round((now - startedAtMs) / 1000));
  const expectedSec = expectedLoadSec(history);
  if (expectedSec == null) {
    return { phase, phasePct, elapsedSec, expectedSec: null, etaSec: null, pct: null };
  }
  // Never claim done before the API answers: a slower-than-usual load sits at
  // 99% with no ETA rather than counting down past zero.
  const pct = Math.min(99, Math.round((elapsedSec / expectedSec) * 100));
  const etaSec = elapsedSec < expectedSec ? Math.round(expectedSec - elapsedSec) : null;
  return { phase, phasePct, elapsedSec, expectedSec: Math.round(expectedSec), etaSec, pct };
}

/** Keep the last few loads so a changed profile re-centres the ETA quickly. */
export const LOAD_HISTORY_MAX = 5;
