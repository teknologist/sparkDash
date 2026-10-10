import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import { validateAssignment, ComposerManager } from "../ComposerManager.js";
import { generateLlamaSwapConfig } from "../llamaSwapGen.js";
import { ModelCatalog } from "../ModelCatalog.js";

// Minimal in-test catalog so validation tests don't depend on models.json values.
function makeCatalog({ budget = 100, cap = 128 } = {}) {
  const models = {
    big1: { id: "big1", displayName: "Big 1", placement: "single", nodes: ["spark1"], ramGB: 90, port: 8000 },
    big2: { id: "big2", displayName: "Big 2", placement: "single", nodes: ["spark1"], ramGB: 90, port: 8000 },
    either: { id: "either", displayName: "Either", placement: "single", nodes: ["spark1", "spark2"], ramGB: 40, port: 8000 },
    a8000: { id: "a8000", displayName: "A", placement: "single", nodes: ["spark2"], ramGB: 30, port: 8000 },
    b8001: { id: "b8001", displayName: "B", placement: "single", nodes: ["spark2"], ramGB: 30, port: 8001 },
    dual: { id: "dual", displayName: "Dual", placement: "dual", nodes: ["spark1", "spark2"], ramGB: 80, port: 8000 },
  };
  return {
    getModel: (id) => models[id] || null,
    nodeBudgetGB: () => budget,
    nodeCapacityGB: () => cap,
  };
}

test("valid single-per-node assignment passes", () => {
  const r = validateAssignment(makeCatalog(), { spark1: ["big1"], spark2: ["a8000"] });
  assert.equal(r.ok, true);
  assert.equal(r.errors.length, 0);
  assert.equal(r.perNode.spark1.ramUsed, 90);
});

test("RAM over budget is rejected", () => {
  const r = validateAssignment(makeCatalog({ budget: 100 }), { spark1: ["big1", "big2"] });
  assert.equal(r.ok, false);
  assert.equal(r.perNode.spark1.over, true);
  // nodeAlloc reports the specific reason: two 90 GB fixed bricks can't be
  // water-filled into a 100 GB budget, so it fails on weights before util.
  assert.match(r.errors.join(" "), /over budget|weights don't fit/);
});

test("same-port collision on a node is rejected", () => {
  const r = validateAssignment(makeCatalog({ budget: 500 }), { spark1: ["big1", "big2"] });
  assert.equal(r.ok, false);
  assert.match(r.errors.join(" "), /Port 8000 conflict/);
});

test("different ports co-reside (OCR+Extract pattern)", () => {
  const r = validateAssignment(makeCatalog(), { spark2: ["a8000", "b8001"] });
  assert.equal(r.ok, true);
  assert.equal(r.perNode.spark2.ramUsed, 60);
});

test("node-affinity violation is rejected", () => {
  const r = validateAssignment(makeCatalog(), { spark1: ["a8000"] });
  assert.equal(r.ok, false);
  assert.match(r.errors.join(" "), /can't run on spark1/);
});

test("a brick eligible on either node is accepted on both", () => {
  assert.equal(validateAssignment(makeCatalog(), { spark1: ["either"] }).ok, true);
  assert.equal(validateAssignment(makeCatalog(), { spark2: ["either"] }).ok, true);
});

test("dual model must occupy both nodes exclusively", () => {
  // Valid: dual on both nodes, alone.
  assert.equal(validateAssignment(makeCatalog(), { spark1: ["dual"], spark2: ["dual"] }).ok, true);
  // Invalid: dual only on one node.
  assert.equal(validateAssignment(makeCatalog(), { spark1: ["dual"] }).ok, false);
  // Invalid: dual sharing a node with another model.
  const r = validateAssignment(makeCatalog({ budget: 500 }), { spark1: ["dual"], spark2: ["dual", "a8000"] });
  assert.equal(r.ok, false);
  assert.match(r.errors.join(" "), /to itself/);
});

test("unknown model id is rejected", () => {
  const r = validateAssignment(makeCatalog(), { spark1: ["nope"] });
  assert.equal(r.ok, false);
  assert.match(r.errors.join(" "), /Unknown model/);
});

// Brick ids, ports and served names change as deployments evolve. Pinning them
// here only produced failures that tracked catalog edits rather than real
// regressions, so the real-catalog tests assert the INVARIANTS and derive every
// expected value from the catalog itself.
const reEsc = (v) => String(v).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const gatewayBrickOn = (cat, node, skip) =>
  cat.models.find(
    (m) =>
      m !== skip &&
      m.placement !== "dual" &&
      m.gateway !== false &&
      m.launch?.[node]?.ready?.port
  );

test("real catalog loads the seeded bricks", () => {
  const cat = new ModelCatalog();
  assert.ok(cat.models.length > 0, "catalog should load at least one brick");
  for (const m of cat.models) {
    assert.ok(m.id, "every brick needs an id");
    assert.ok(m.launch && Object.keys(m.launch).length, `${m.id} needs a launch block`);
    for (const [node, spec] of Object.entries(m.launch)) {
      assert.ok(spec.start?.cmd, `${m.id}@${node} needs a start command`);
      assert.ok(spec.ready?.port, `${m.id}@${node} needs a ready probe`);
    }
  }
  assert.equal(cat.nodeBudgetGB("spark1"), cat.nodeCapacityGB("spark1") - cat.reserveGB());
});

test("generated llama-swap config routes bricks to their nodes", () => {
  const cat = new ModelCatalog();
  const m1 = gatewayBrickOn(cat, "spark1");
  const m2 = gatewayBrickOn(cat, "spark2", m1);
  assert.ok(m1 && m2, "catalog needs a gateway brick runnable on each node");

  const yaml = generateLlamaSwapConfig(cat, { spark1: [m1.id], spark2: [m2.id] });
  for (const [m, node] of [
    [m1, "spark1"],
    [m2, "spark2"],
  ]) {
    const ready = m.launch[node].ready;
    const served = ready.servedModel || m.servedModel || m.id;
    assert.match(yaml, new RegExp(`"${reEsc(served)}":`), `${m.id} should be advertised`);
    assert.match(
      yaml,
      new RegExp(`proxy: "http://${reEsc(ready.host)}:${ready.port}"`),
      `${m.id} should proxy to its ${node} endpoint`
    );
  }
  assert.match(yaml, /groups:/);
});

test("a gateway:false brick is kept out of the llama-swap config", () => {
  // jeff answers /v1/systemone only — llama-swap could neither health-check nor
  // proxy it, so advertising it would hand clients a model that 404s. The
  // composer starts those directly instead.
  const cat = new ModelCatalog();
  const direct = cat.models.find((m) => m.gateway === false);
  if (!direct) return; // no such brick in the catalog right now
  const node = Object.keys(direct.launch)[0];
  const yaml = generateLlamaSwapConfig(cat, { [node]: [direct.id] });
  const served = direct.servedModel || direct.id;
  assert.doesNotMatch(yaml, new RegExp(`"${reEsc(served)}":`));
});

// A dual brick is probed once per node it occupies, and every probe hits the
// head's endpoint. Counting each hit as its own instance made getState() report
// the brick twice on every node (200 GB used of a 121 GB cap for the 0731 TP=2
// stack), so the board showed the cluster over budget instead of "loaded".
test("a serving dual brick is detected once and counted once per node", async () => {
  const dual = {
    id: "dual", placement: "dual", nodes: ["spark1", "spark2"], ramGB: 80, port: 8000,
    servedModel: "dual",
    launch: { spark1: { target: "local", ready: { host: "127.0.0.1", port: 8000, servedModel: "dual" } } },
  };
  const base = makeCatalog();
  const mgr = Object.create(ComposerManager.prototype);
  Object.assign(mgr, {
    catalog: {
      ...base,
      models: [dual],
      getModel: (id) => (id === "dual" ? dual : base.getModel(id)),
      toPublic: () => ({ nodeCapacityGB: { spark1: 128, spark2: 128 } }),
    },
    _running: [], _applying: false, _disposed: false, _log: [], onChange: null,
    _loadPresets: () => [],
    _probeReady: async () => true,
    _gatewayStarting: async () => [],
  });
  await mgr._detect();
  assert.equal(mgr._running.length, 1);
  const { perNode } = mgr.getState();
  for (const node of ["spark1", "spark2"]) {
    assert.deepEqual(perNode[node].running.map((r) => r.id), ["dual"], node);
  }
});

// A dual brick holds its weightGB on BOTH nodes, and memcheck is what drops the
// page cache CUDA counts as used. Guarding only the head let a cache-full worker
// fail GLM's InstantTensor load (its buffer is sized from the MIN free memory
// across ranks), so the generated start must guard the peer node too.
test("a dual brick's start guards memory on both nodes", () => {
  const cat = new ModelCatalog();
  const dual = cat.models.find((m) => m.placement === "dual" && m.gateway !== false && m.weightGB > 0);
  if (!dual) return; // no dual brick in the catalog right now
  const yaml = generateLlamaSwapConfig(cat, { spark1: [dual.id], spark2: [dual.id] });
  const served = dual.launch[Object.keys(dual.launch)[0]].ready.servedModel || dual.servedModel;
  const cmd = yaml.split("\n").find((l, i, a) => a[i - 1]?.includes(`"${served}":`));
  const n = Math.round(dual.weightGB);
  assert.match(cmd, new RegExp(`\\{ /\\S+/memcheck\\.sh ${n} \\|\\| exit 1; \\$\\{ssh2\\} "/\\S+/memcheck\\.sh ${n}" \\|\\| exit 1;`));
});

// Auto-repair: a TP worker left running after its head is gone serves nothing
// but keeps its whole memory share, so every later start fails (a GLM worker
// held 103 GiB on spark2 for 3 days). The reaper must remove it — and must
// never touch a pair that is merely starting or a node it could not inspect.
function reaperMgr() {
  const glm = {
    id: "glm", placement: "dual", nodes: ["spark1", "spark2"], container: "glm-head",
    peerContainers: { spark2: ["glm-worker"] },
    launch: { spark1: { target: "local" } },
  };
  const removed = [];
  const mgr = Object.create(ComposerManager.prototype);
  Object.assign(mgr, {
    _orphanSince: new Map(), _log: [], _disposed: false, onChange: null,
    _removeContainer: async (node, name) => (removed.push(`${name}@${node}`), true),
  });
  const ps = (names, answered = true) => Object.assign(new Set(names), answered ? { answered } : {});
  return { mgr, removed, bricks: [{ model: glm }], ps };
}

test("an orphaned dual worker is removed once it outlives the grace period", async () => {
  const { mgr, removed, bricks, ps } = reaperMgr();
  const cont = { spark1: ps([]), spark2: ps(["glm-worker"]) };
  await mgr._reapOrphans(bricks, cont, 0);
  await mgr._reapOrphans(bricks, cont, 60_000);
  assert.deepEqual(removed, [], "a worker that just started (head coming up) must survive");
  await mgr._reapOrphans(bricks, cont, 181_000);
  assert.deepEqual(removed, ["glm-worker@spark2"]);
});

test("a worker whose head is up, or whose head node could not be inspected, is never reaped", async () => {
  const { mgr, removed, bricks, ps } = reaperMgr();
  for (const t of [0, 400_000]) {
    await mgr._reapOrphans(bricks, { spark1: ps(["glm-head"]), spark2: ps(["glm-worker"]) }, t);
  }
  for (const t of [500_000, 900_000]) {
    // docker ps failed on the head node: unknown is not "gone"
    await mgr._reapOrphans(bricks, { spark1: ps([], false), spark2: ps(["glm-worker"]) }, t);
  }
  assert.deepEqual(removed, []);
});

test("the orphan clock resets when the head comes back", async () => {
  const { mgr, removed, bricks, ps } = reaperMgr();
  await mgr._reapOrphans(bricks, { spark1: ps([]), spark2: ps(["glm-worker"]) }, 0);
  await mgr._reapOrphans(bricks, { spark1: ps(["glm-head"]), spark2: ps(["glm-worker"]) }, 100_000);
  await mgr._reapOrphans(bricks, { spark1: ps([]), spark2: ps(["glm-worker"]) }, 200_000);
  assert.deepEqual(removed, [], "the 3-minute window restarts after the head reappeared");
});

// An apply blocks in the gateway load for the whole cold start; detection used
// to be skipped while applying, so the board showed nothing exactly while the
// user was watching a model load. It must keep reporting the loading brick.
test("detection keeps reporting a loading brick while an apply is in flight", async () => {
  const glm = {
    id: "glm", placement: "dual", nodes: ["spark1", "spark2"], container: "glm-head", servedModel: "glm",
    launch: { spark1: { target: "local", ready: { host: "127.0.0.1", port: 8888, servedModel: "glm" } } },
  };
  const mgr = Object.create(ComposerManager.prototype);
  Object.assign(mgr, {
    catalog: { models: [glm], getModel: (id) => (id === "glm" ? glm : null) },
    _running: [], _applying: true, _disposed: false, _log: [], onChange: null,
    _loadTimes: { glm: [240] }, _loadStart: new Map(),
    _probeReady: async () => false,
    _gatewayStarting: async () => [],
    _nodeContainers: async () => Object.assign(new Set(["glm-head"]), { answered: true }),
    _containerLoadInfo: async () => ({ startedAtMs: Date.now() - 60_000, lines: ["Loading safetensors: 40% Completed"] }),
  });
  await mgr._detect();
  assert.equal(mgr._running.length, 1);
  const r = mgr._running[0];
  assert.equal(r.state, "loading");
  assert.equal(r.progress.phase, "loading weights");
  assert.ok(r.progress.etaSec > 150 && r.progress.etaSec <= 180, `eta ${r.progress.etaSec}`);
});

// Every model — not only container ones — must learn its load time on its first
// load and persist it, so the NEXT load has an ETA even after a restart/reboot.
function timingMgr({ model, ready, starting = [], containers = [] }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "composer-load-"));
  const mgr = Object.create(ComposerManager.prototype);
  Object.assign(mgr, {
    catalog: { models: [model], getModel: (id) => (id === model.id ? model : null) },
    _running: [], _applying: false, _disposed: false, _log: [], onChange: null,
    _loadTimes: {}, _loadStart: new Map(),
    _sidecarPath: (name) => path.join(dir, name),
    _probeReady: async () => ready.value,
    _gatewayStarting: async () => starting,
    _nodeContainers: async () => Object.assign(new Set(containers), { answered: true }),
  });
  return { mgr, file: path.join(dir, "load-times.json") };
}

test("a gateway-started model without a container is shown loading and its first load is persisted", async (t) => {
  const ds4 = {
    id: "ds4", placement: "single", nodes: ["spark1", "spark2"], servedModel: "ds4",
    launch: {
      spark1: { target: "local", ready: { host: "127.0.0.1", port: 8000, servedModel: "ds4" } },
      spark2: { target: "ssh", host: "Spark2", ready: { host: "10.100.72.1", port: 8000, servedModel: "ds4" } },
    },
  };
  const ready = { value: false };
  const starting = [{ model: "ds4", proxy: "http://10.100.72.1:8000" }];
  const { mgr, file } = timingMgr({ model: ds4, ready, starting });
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  await mgr._detect();
  assert.equal(mgr._running[0].state, "loading");
  assert.equal(mgr._running[0].node, "spark2", "placed on the node the gateway proxies to");
  assert.equal(mgr._running[0].progress.etaSec, null, "no ETA before the first measured load");

  starting.length = 0;
  ready.value = true;
  t.mock.timers.tick(95_000);
  await mgr._detect();
  assert.deepEqual(JSON.parse(fs.readFileSync(file, "utf-8")), { ds4: [95] });

  // A fresh manager (dashboard restart) reads it back and gives the next load an ETA.
  const again = timingMgr({ model: ds4, ready: { value: false }, starting: [{ model: "ds4", proxy: "http://127.0.0.1:8000" }] });
  again.mgr._sidecarPath = () => file;
  again.mgr._loadTimes = again.mgr._readJsonSidecar("load-times.json");
  await again.mgr._detect();
  t.mock.timers.tick(30_000);
  await again.mgr._detect();
  assert.equal(again.mgr._running[0].progress.etaSec, 65);
});

test("a serving model whose probe blips is not recorded as a days-long load", async (t) => {
  const glm = {
    id: "glm", placement: "dual", nodes: ["spark1", "spark2"], container: "glm-head", servedModel: "glm",
    launch: { spark1: { target: "local", ready: { host: "127.0.0.1", port: 8888, servedModel: "glm" } } },
  };
  const ready = { value: true };
  const { mgr, file } = timingMgr({ model: glm, ready, containers: ["glm-head"] });
  t.mock.timers.enable({ apis: ["Date"], now: 10 * 86_400_000 });
  mgr._containerLoadInfo = async () => ({ startedAtMs: 7 * 86_400_000, lines: [] }); // started 3 days ago
  await mgr._detect(); // serving
  ready.value = false;
  t.mock.timers.tick(5_000);
  await mgr._detect(); // one missed probe: looks "loading"
  ready.value = true;
  t.mock.timers.tick(5_000);
  await mgr._detect(); // back
  assert.equal(fs.existsSync(file), false);
});
