import { test } from "node:test";
import assert from "node:assert/strict";
import { SetupManager } from "../SetupManager.js";

function mgrIn(phase, detected, { endpoints = [], gatewayStarting = null } = {}) {
  const m = Object.create(SetupManager.prototype);
  Object.assign(m, {
    _transitioning: false,
    _machine: { phase, activeSetupId: null, error: phase === "failed" ? "glm start exited with code 1" : null },
    _activeModels: [],
    detectActiveId: async () => detected,
    // Endpoint liveness the active setup would report once refreshed.
    _refreshActiveModels: async () => {
      m._activeModels = m._machine.activeSetupId ? endpoints : [];
    },
    _gatewayLoadingSetupId: async () => gatewayStarting,
    _emit: () => {},
  });
  return m;
}

// The header pill read "No model running" in red over a serving GLM: one failed
// start (before an ssh fix) pinned `failed` even after a later start succeeded.
test("a detected running setup clears a stale failure", async () => {
  const m = mgrIn("failed", { activeSetupId: "glm", error: null });
  await m._reconcile();
  assert.equal(m._machine.phase, "running");
  assert.equal(m._machine.activeSetupId, "glm");
  assert.equal(m._machine.error, null);
});

test("a failure stays visible while nothing is serving", async () => {
  const m = mgrIn("failed", { activeSetupId: null, error: null });
  await m._reconcile();
  assert.equal(m._machine.phase, "failed");
  assert.match(m._machine.error, /exited with code 1/);
});

// A container exists the moment a start script runs, but TensorFold then loads
// ~100 GiB of weights for minutes. The pill said "No model running" (or a green
// name) over that window; the user asked for "Loading" until the model answers.
test("a detected setup whose endpoints do not answer yet is loading", async () => {
  const ep = { host: "127.0.0.1", port: 8000, servedModel: "DeepSeek-V4.1-Flash-1017", up: false, modelId: null };
  const m = mgrIn("idle", { activeSetupId: "deepseek-v4.1-1017", error: null }, { endpoints: [ep] });
  await m._reconcile();
  assert.equal(m._machine.phase, "loading");
  assert.equal(m._machine.activeSetupId, "deepseek-v4.1-1017");
});

test("a detected setup whose endpoints all answer is running", async () => {
  const ep = { host: "127.0.0.1", port: 8000, servedModel: "DeepSeek-V4.1-Flash-1017", up: true, modelId: "DeepSeek-V4.1-Flash-1017" };
  const m = mgrIn("loading", { activeSetupId: "deepseek-v4.1-1017", error: null }, { endpoints: [ep] });
  await m._reconcile();
  assert.equal(m._machine.phase, "running");
});

// The gateway runs memcheck / image pull / worker launch before the local
// container exists, so host detection alone reports idle for that stretch.
test("a setup the gateway is still starting is loading before its container exists", async () => {
  const m = mgrIn("idle", { activeSetupId: null, error: null }, { gatewayStarting: "glm-5.3-flash-exl3" });
  await m._reconcile();
  assert.equal(m._machine.phase, "loading");
  assert.equal(m._machine.activeSetupId, "glm-5.3-flash-exl3");
});

test("nothing detected and nothing starting is idle", async () => {
  const m = mgrIn("running", { activeSetupId: null, error: null });
  m._machine.activeSetupId = "glm";
  await m._reconcile();
  assert.equal(m._machine.phase, "idle");
  assert.equal(m._machine.activeSetupId, null);
});
