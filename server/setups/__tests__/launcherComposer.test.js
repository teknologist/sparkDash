import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ComposerManager } from "../ComposerManager.js";
import { LauncherBridge, launcherBrickId, parseLauncherBrickId } from "../launcherBridge.js";
import { generateLlamaSwapConfig } from "../llamaSwapGen.js";
import { ModelStats } from "../ModelStats.js";
import { ModelCatalog } from "../ModelCatalog.js";

const tmpStats = (models = {}) => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "model-stats-")), "model-stats.json");
  if (Object.keys(models).length) fs.writeFileSync(file, JSON.stringify({ version: 1, models }));
  return { stats: new ModelStats(file), file };
};

// Upstream's launcher (start.sh/stop.sh per Spark) had no notion of memory: it
// would start a model into a node the composer knew was full. These tests pin
// down how launcher models share the composer's memory accounting.

function bridgeWith({ launchers = [], models = {} } = {}) {
  const { stats, file } = tmpStats(models);
  const jobs = [];
  const bridge = new LauncherBridge({
    store: { list: () => launchers, get: (_s, id) => launchers.find((l) => l.id === id) || null },
    manager: {
      startJob: (_spark, l, action) => {
        const job = { id: `j${jobs.length}`, status: action === "stop" ? "completed" : "running", action, launcherName: l.name };
        jobs.push(job);
        return { ok: true, job };
      },
      latestJob: () => jobs[jobs.length - 1] || null,
    },
    getSpark: (id) => ({ id, name: id, isLocal: id === "spark1", lanIp: "192.168.10.114" }),
    sparkIds: () => ["spark2"],
    statuses: async () => ({}),
    stats,
    dashUrl: "http://127.0.0.1:5555",
  });
  return { bridge, stats, file, jobs };
}

const QWEN = { id: "qwen-ab12cd", name: "Qwen 32B", dir: "~/models/qwen", startScript: "start.sh", port: 8010 };

test("brick ids round-trip and stay distinct from models.json ids", () => {
  const id = launcherBrickId("spark2", "qwen-ab12cd");
  assert.deepEqual(parseLauncherBrickId(id), { sparkId: "spark2", launcherId: "qwen-ab12cd" });
  assert.equal(parseLauncherBrickId("glm-5.3-flash-exl3"), null);
});

test("an unmeasured launcher model is a gateway-less brick that claims its node", () => {
  const { bridge } = bridgeWith({ launchers: [QWEN, { ...QWEN, id: "noport-1", port: null }] });
  const bricks = bridge.bricks();
  assert.equal(bricks.length, 1, "a launcher without a port cannot be tracked, so it is not a brick");
  const b = bricks[0];
  assert.equal(b.gateway, false, "no served name yet => nothing for llama-swap to route");
  assert.equal(b.launch.spark2.ready.host, "192.168.10.114");
  assert.equal(b.launch.spark2.ready.match, "any");
});

test("a learned served name puts the launcher model in the gateway", () => {
  const id = launcherBrickId("spark2", QWEN.id);
  const { bridge, file } = bridgeWith({ launchers: [QWEN] });
  bridge.learnServed(id, "Qwen/Qwen3-32B");
  assert.equal(JSON.parse(fs.readFileSync(file, "utf-8")).models[id].servedModel, "Qwen/Qwen3-32B");
  const b = bridge.bricks()[0];
  assert.equal(b.gateway, true);
  assert.equal(b.launch.spark2.ready.servedModel, "Qwen/Qwen3-32B");
});

// ─── Measured stats (persisted) ────────────────────────────
test("existing load times are imported once from the old load-times.json", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stats-migrate-"));
  fs.writeFileSync(path.join(dir, "load-times.json"), JSON.stringify({ "glm-5.3-flash-exl3": [241] }));
  const stats = new ModelStats(path.join(dir, "model-stats.json"), { legacyLoadTimesFile: path.join(dir, "load-times.json") });
  assert.deepEqual(stats.loads("glm-5.3-flash-exl3"), [241]);
  assert.deepEqual(new ModelStats(path.join(dir, "model-stats.json")).loads("glm-5.3-flash-exl3"), [241], "persisted");
});

test("peaks keep the last runs, a run that never registered evicts nothing, exact and approx are apart", () => {
  const { stats } = tmpStats();
  for (const gb of [60, 70, 65]) {
    stats.beginPeakRun("m");
    stats.updatePeak("m", gb);
  }
  stats.beginPeakRun("m"); // stopped before any reading
  stats.beginPeakRun("m", { approx: true });
  stats.updatePeak("m", 90);
  assert.deepEqual(stats.peak("m"), { exact: 70, approx: 90 }, "60 rolled off; 70 and 65 kept; the empty run dropped");
});

function catalogWith(models, statsModels) {
  const cat = new ModelCatalog();
  cat._models = models;
  cat._refreshIfChanged = () => {};
  cat.nodeCapacityGB = () => 121;
  cat.reserveGB = () => 8;
  cat.setStats(tmpStats(statsModels).stats);
  return cat;
}

test("an exact measured peak decides placement; an approximate one only raises the estimate", () => {
  const ds4 = { id: "ds4", placement: "single", nodes: ["spark1"], weightGB: 112, launch: {} };
  const exact = catalogWith([ds4], { ds4: { peaks: [{ gb: 86, at: 1 }] } }).getModel("ds4");
  assert.equal(exact.weightGB, 86, "measured 86 GB frees 26 GB the 112 GB estimate was blocking");
  const approxLow = catalogWith([ds4], { ds4: { peaks: [{ gb: 40, at: 1, approx: true }] } }).getModel("ds4");
  assert.equal(approxLow.weightGB, 112, "a late baseline reads low and must not shrink the estimate");
  const approxHigh = catalogWith([ds4], { ds4: { peaks: [{ gb: 118, at: 1, approx: true }] } }).getModel("ds4");
  assert.equal(approxHigh.weightGB, 118);
});

test("an elastic vLLM model keeps its configured share; the peak is shown, not used", () => {
  const laguna = { id: "laguna", placement: "single", nodes: ["spark2"], weightGB: 81, maxUtil: 0.8, launch: {} };
  const m = catalogWith([laguna], { laguna: { peaks: [{ gb: 95, at: 1 }] } }).getModel("laguna");
  assert.equal(m.weightGB, 81);
  assert.equal(m.measuredPeakGB, 95);
});

test("a launcher model claims its whole node until measured, then its peak", () => {
  const brick = { id: "launcher:spark2:q", placement: "single", nodes: ["spark2"], weightGB: null, launcher: {}, launch: {} };
  const cat = catalogWith([], {});
  cat.setExtraModels(() => [brick]);
  assert.equal(cat.getModel(brick.id).weightGB, 113);
  assert.equal(cat.getModel(brick.id).footprintKnown, false);
  cat.setStats(tmpStats({ [brick.id]: { peaks: [{ gb: 50, at: 1 }] } }).stats);
  assert.equal(cat.getModel(brick.id).weightGB, 50);
  assert.equal(cat.getModel(brick.id).footprintKnown, true);
});

test("peak memory is charged to the model that started, never to a later co-resident", () => {
  const mem = { spark1: 10, spark2: 10 };
  const a = { id: "a", placement: "single", nodes: ["spark2"] };
  const b = { id: "b", placement: "single", nodes: ["spark2"] };
  const glm = { id: "glm", placement: "dual", nodes: ["spark1", "spark2"] };
  const byId = { a, b, glm };
  const { stats } = tmpStats();
  const mgr = Object.create(ComposerManager.prototype);
  Object.assign(mgr, {
    catalog: { getModel: (id) => byId[id] },
    stats,
    _peakRuns: new Map(),
    nodeMemory: (n) => ({ usedGB: mem[n], availableGB: 121 - mem[n] }),
  });
  const running = (...ids) => ids.map((id) => ({ id, node: "spark2", state: "ready" }));

  mgr.beginPeak("a", "spark2");
  mem.spark2 = 50; // a loads 40
  mgr._trackPeaks(running("a"));
  mgr.beginPeak("b", "spark2"); // b starts: a's window closes
  mem.spark2 = 80; // b loads 30
  mgr._trackPeaks(running("a", "b"));
  assert.equal(stats.peak("a").exact, 40, "b's 30 GB is not charged to a");
  assert.equal(stats.peak("b").exact, 30);

  mgr.beginPeak("glm", "spark1"); // dual: measured on both nodes, the higher counts
  mem.spark1 = 105;
  mem.spark2 = 120;
  mgr._trackPeaks([{ id: "glm", node: "spark1", state: "loading" }]);
  assert.equal(stats.peak("glm").exact, 95);
});

// ─── Room planning ─────────────────────────────────────────
function plannerMgr({ running, bricks, availableGB }) {
  const byId = Object.fromEntries(bricks.map((b) => [b.id, b]));
  const mgr = Object.create(ComposerManager.prototype);
  Object.assign(mgr, {
    catalog: {
      models: bricks,
      getModel: (id) => byId[id] || null,
      nodeBudgetGB: () => 113,
      nodeCapacityGB: () => 121,
      reserveGB: () => 8,
      toPublic: () => ({ nodeCapacityGB: { spark1: 121, spark2: 121 } }),
    },
    _running: running,
    _phase: "idle",
    _log: [],
    _loadPresets: () => [],
    launchers: { nodeMemory: () => ({ usedGB: 121 - availableGB, availableGB }) },
  });
  return mgr;
}

const GLM = { id: "glm", displayName: "GLM-5.3", placement: "dual", nodes: ["spark1", "spark2"], weightGB: 104, launch: { spark1: {} } };
const ORNITH = { id: "ornith", displayName: "Ornith", placement: "single", nodes: ["spark2"], weightGB: 38, launch: { spark2: {} } };
const JEFF = { id: "jeff", displayName: "jeff", placement: "single", nodes: ["spark2"], weightGB: 3, launch: { spark2: {} } };
const target = (weightGB, footprintKnown = true) => ({
  id: "launcher:spark2:qwen", displayName: "Qwen", placement: "single", nodes: ["spark2"], weightGB, footprintKnown, launcher: {}, launch: { spark2: {} },
});

test("nothing is stopped when the node already has room", () => {
  const mgr = plannerMgr({ bricks: [ORNITH, target(40)], running: [{ id: "ornith", node: "spark2" }], availableGB: 80 });
  const plan = mgr.planRoom("spark2", "launcher:spark2:qwen");
  assert.deepEqual(plan.victims, []);
  assert.equal(plan.fits, true);
});

test("the fewest models are stopped, biggest first, and a dual one is listed with both nodes", () => {
  const mgr = plannerMgr({
    bricks: [GLM, JEFF, target(60)],
    running: [{ id: "glm", node: "spark1" }, { id: "jeff", node: "spark2" }],
    availableGB: 12,
  });
  const plan = mgr.planRoom("spark2", "launcher:spark2:qwen");
  assert.deepEqual(plan.victims.map((v) => v.id), ["glm"], "stopping GLM alone frees enough; jeff stays");
  assert.deepEqual(plan.victims[0].nodes, ["spark1", "spark2"]);
});

test("a first run (footprint unknown) clears everything on its node", () => {
  const mgr = plannerMgr({
    bricks: [ORNITH, JEFF, target(113, false)],
    running: [{ id: "ornith", node: "spark2" }, { id: "jeff", node: "spark2" }],
    availableGB: 70,
  });
  const plan = mgr.planRoom("spark2", "launcher:spark2:qwen");
  assert.equal(plan.firstRun, true);
  assert.deepEqual(plan.victims.map((v) => v.id).sort(), ["jeff", "ornith"]);
});

// ─── Gateway ───────────────────────────────────────────────
test("a learned launcher model is routed by the gateway and refuses to evict on its own", () => {
  const { bridge } = bridgeWith({
    launchers: [QWEN],
    models: { [launcherBrickId("spark2", QWEN.id)]: { servedModel: "Qwen/Qwen3-32B", peaks: [{ gb: 50, at: 1 }] } },
  });
  const [b] = bridge.bricks();
  const catalog = { models: [b], nodeBudgetGB: () => 113, nodeCapacityGB: () => 121, getModel: () => b };
  const yaml = generateLlamaSwapConfig(catalog, { spark2: [b.id] });
  const line = yaml.split("\n").find((l) => l.includes("cmd:") && l.includes("llm-launchers"));
  assert.match(yaml, /"Qwen\/Qwen3-32B":/);
  assert.match(line, /start\?gateway=1" >\/dev\/null \|\| exit 1/, "a 409 'no room' fails the load");
  assert.doesNotMatch(line, /memcheck/, "the dashboard's room check replaces the memory guard");
});

test("a launcher may not take a port a composer model already serves on that Spark", async () => {
  const { composerPortConflict } = await import("../launcherBridge.js");
  const models = [
    { id: "qwen3.6-extract", displayName: "Qwen 3.6 Extract", launch: { spark2: { ready: { port: 8001 } }, spark1: { ready: { port: 8001 } } } },
    { id: "launcher:spark2:other", launcher: {}, launch: { spark2: { ready: { port: 8010 } } } },
  ];
  assert.deepEqual(composerPortConflict(models, "spark2", 8001), { id: "qwen3.6-extract", name: "Qwen 3.6 Extract" });
  assert.equal(composerPortConflict(models, "spark2", 8010), null, "other launchers are upstream's business");
  assert.equal(composerPortConflict(models, "spark2", 9000), null);
  assert.equal(composerPortConflict(models, "spark2", null), null, "no port: not composer-managed, nothing to clash");
});

// Qwen3.8 on TensorFold measured 74 GB idle, but its cache grows into free
// memory while serving. Shrinking its claim to 74 would let another model in,
// and the growth would then exhaust the node (which freezes a Spark).
test("a wholeNode model's measured peak can raise its claim but never lower it", () => {
  const q = { id: "q38", placement: "single", nodes: ["spark1"], weightGB: 104, wholeNode: true, launch: {} };
  assert.equal(catalogWith([q], { q38: { peaks: [{ gb: 74, at: 1 }] } }).getModel("q38").weightGB, 104);
  assert.equal(catalogWith([q], { q38: { peaks: [{ gb: 110, at: 1 }] } }).getModel("q38").weightGB, 110);
  assert.equal(catalogWith([q], { q38: { peaks: [{ gb: 74, at: 1 }] } }).getModel("q38").measuredPeakGB, 74, "still shown");
});
