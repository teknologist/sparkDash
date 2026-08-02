import { test } from "node:test";
import assert from "node:assert/strict";
import { validateAssignment } from "../ComposerManager.js";
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
  assert.match(r.errors.join(" "), /over budget/);
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

test("real catalog loads the seeded bricks", () => {
  const cat = new ModelCatalog();
  const ids = cat.models.map((m) => m.id);
  assert.ok(ids.includes("deepseek-v4-flash"));
  assert.ok(ids.includes("laguna-s"));
  assert.equal(cat.nodeBudgetGB("spark1"), cat.nodeCapacityGB("spark1") - cat.reserveGB());
});

test("generated llama-swap config routes bricks to their nodes", () => {
  const cat = new ModelCatalog();
  const yaml = generateLlamaSwapConfig(cat, { spark1: ["deepseek-v4-flash"], spark2: ["laguna-s"] });
  assert.match(yaml, /"deepseek-v4-flash":/);
  assert.match(yaml, /proxy: "http:\/\/127\.0\.0\.1:8000"/);
  assert.match(yaml, /"laguna-s":/);
  assert.match(yaml, /proxy: "http:\/\/10\.100\.72\.1:8000"/);
  assert.match(yaml, /groups:/);
});
