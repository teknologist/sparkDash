import { test } from "node:test";
import assert from "node:assert/strict";
import { SetupManager } from "../SetupManager.js";

function mgrIn(phase, detected) {
  const m = Object.create(SetupManager.prototype);
  Object.assign(m, {
    _transitioning: false,
    _machine: { phase, activeSetupId: null, error: phase === "failed" ? "glm start exited with code 1" : null },
    detectActiveId: async () => detected,
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
