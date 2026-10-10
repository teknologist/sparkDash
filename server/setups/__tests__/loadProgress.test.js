import { test } from "node:test";
import assert from "node:assert/strict";
import { parseLoadPhase, loadProgress, expectedLoadSec } from "../loadProgress.js";

// Lines lifted from a real GLM-5.3 EXL3 head log (2026-10-10). The board relies
// on these to say what a multi-minute load is doing while its API is dark.
const GLM = {
  patch: "patched scheduler.py (# [glm53-mamba-align-chunking-v1])",
  engine: "(EngineCore pid=2767) INFO 10-10 10:22:39 [core.py:122] Initializing a V1 LLM engine (v0.1.dev20051) with config: model='/root/.cache/x'",
  weights40: "(Worker_TP0 pid=2964) Loading safetensors using InstantTensor loader:  40% Completed | 64.9G/164G [00:12<00:1",
  mlaKv: "(Worker_TP0 pid=2964) INFO 10-10 10:22:49 [mla_attention.py:475] Using fp8_ds_mla KV cache format for FLASHINFER",
  loaded: "(Worker_TP0 pid=2964) INFO 10-10 10:23:24 [default_loader.py:430] Loading weights took 30.54 seconds",
  kv: "(Worker_TP0 pid=2964) INFO 10-10 10:23:27 [interface.py:635] Setting kv cache block size to 64 for DEEPSEEK_V3",
  graphs: "(Worker_TP0 pid=2964) Capturing CUDA graphs (FULL):  25%|██▌       | 1/4 [00:00<00:00",
  api: "(APIServer pid=1) INFO:     Application startup complete.",
};

test("the phase follows the latest vLLM step, with its own percentage", () => {
  assert.deepEqual(parseLoadPhase([GLM.patch, GLM.engine, GLM.weights40]), { phase: "loading weights", phasePct: 40 });
  assert.deepEqual(parseLoadPhase([GLM.weights40, GLM.loaded, GLM.kv]), { phase: "sizing KV cache", phasePct: null });
  assert.deepEqual(parseLoadPhase([GLM.kv, GLM.graphs]), { phase: "capturing CUDA graphs", phasePct: 25 });
  assert.equal(parseLoadPhase([GLM.graphs, GLM.api]).phase, "starting API");
});

test("launcher patch lines interleaved with vLLM output do not drag the phase back", () => {
  assert.equal(parseLoadPhase([GLM.engine, GLM.weights40, GLM.patch]).phase, "loading weights");
  assert.equal(parseLoadPhase([GLM.patch, GLM.patch]).phase, "patching", "before vLLM speaks it IS patching");
});

test("a weights line that mentions a KV cache format is still weight loading", () => {
  assert.equal(parseLoadPhase([GLM.weights40, GLM.mlaKv]).phase, "loading weights");
});

test("ETA comes from the median of measured loads and never runs past zero", () => {
  assert.equal(expectedLoadSec([]), null);
  assert.equal(expectedLoadSec([240, 900, 250]), 250, "one outlier load does not skew it");
  const mid = loadProgress({ startedAtMs: 0, now: 60_000, history: [240] });
  assert.equal(mid.pct, 25);
  assert.equal(mid.etaSec, 180);
  const late = loadProgress({ startedAtMs: 0, now: 400_000, history: [240] });
  assert.equal(late.pct, 99, "a slow load is not shown as done before the API answers");
  assert.equal(late.etaSec, null);
});

test("without any measured load there is no ETA, only elapsed time", () => {
  const p = loadProgress({ startedAtMs: 0, now: 75_000, phase: "patching" });
  assert.deepEqual(p, { phase: "patching", phasePct: null, elapsedSec: 75, expectedSec: null, etaSec: null, pct: null });
});
