import { test } from "node:test";
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
