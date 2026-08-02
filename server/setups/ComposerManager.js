import fs from "fs";
import { spawn } from "child_process";
import { ModelCatalog } from "./ModelCatalog.js";
import { generateLlamaSwapConfig } from "./llamaSwapGen.js";
import { COMPOSITIONS_PATH, LLAMA_SWAP_CONFIG_PATH } from "../config.js";

const DETECT_INTERVAL_MS = 5000;
const READY_TIMEOUT_MS = 180000;
const READY_POLL_MS = 5000;
const LOG_MAX = 400;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Pure validation of a per-node assignment against the catalog.
 * assignment = { <nodeId>: [modelId, ...], ... }. Dual models are listed on
 * every node they occupy. Exported standalone so it can be unit-tested and run
 * client-side-equivalently on the server.
 *
 * @returns {{ok:boolean, perNode:Record<string,{models:string[],ramUsed:number,ramCap:number,budget:number,over:boolean}>, errors:string[], warnings:string[]}}
 */
export function validateAssignment(catalog, assignment) {
  const errors = [];
  const warnings = [];
  const perNode = {};
  const a = assignment && typeof assignment === "object" ? assignment : {};
  const nodes = Object.keys(a);

  // Resolve + basic per-node checks.
  const resolved = {}; // node -> [{id, model}]
  for (const node of nodes) {
    const ids = Array.isArray(a[node]) ? a[node] : [];
    resolved[node] = [];
    const seen = new Set();
    for (const id of ids) {
      if (seen.has(id)) {
        warnings.push(`${id} listed twice on ${node}`);
        continue;
      }
      seen.add(id);
      const model = catalog.getModel(id);
      if (!model) {
        errors.push(`Unknown model "${id}" on ${node}`);
        continue;
      }
      const eligible = Array.isArray(model.nodes) ? model.nodes : Object.keys(model.launch || {});
      if (!eligible.includes(node)) {
        errors.push(`${model.displayName || id} can't run on ${node} (only ${eligible.join(", ")})`);
      }
      resolved[node].push({ id, model });
    }
  }

  // Dual-model exclusivity: a dual brick must occupy ALL its nodes and be the
  // sole occupant of each (it claims both GPUs, TP=2).
  const dualIds = new Set();
  for (const node of nodes) {
    for (const { model } of resolved[node]) {
      if (model.placement === "dual") dualIds.add(model.id);
    }
  }
  for (const id of dualIds) {
    const model = catalog.getModel(id);
    const need = Array.isArray(model.nodes) ? model.nodes : Object.keys(model.launch || {});
    for (const node of need) {
      const onNode = (resolved[node] || []).map((r) => r.id);
      if (!onNode.includes(id)) {
        errors.push(`${model.displayName || id} is a dual model — it must occupy ${need.join(" + ")}`);
      }
      const others = onNode.filter((x) => x !== id);
      if (others.length) {
        errors.push(`${model.displayName || id} (dual) needs ${node} to itself; remove ${others.join(", ")}`);
      }
    }
  }

  // Per-node RAM budget + port collisions.
  for (const node of nodes) {
    const list = resolved[node];
    const ramUsed = list.reduce((sum, r) => sum + (Number(r.model.ramGB) || 0), 0);
    const budget = catalog.nodeBudgetGB(node);
    const ramCap = catalog.nodeCapacityGB(node);
    const over = ramUsed > budget;
    if (over) {
      errors.push(`${node} over budget: ${ramUsed} GB used > ${budget} GB available (of ${ramCap} GB)`);
    }
    const byPort = new Map();
    for (const { id, model } of list) {
      const port = model.port;
      if (port == null) continue;
      if (byPort.has(port)) {
        errors.push(`Port ${port} conflict on ${node}: ${byPort.get(port)} and ${id}`);
      } else {
        byPort.set(port, id);
      }
    }
    perNode[node] = { models: list.map((r) => r.id), ramUsed, ramCap, budget, over };
  }

  return { ok: errors.length === 0, perNode, errors, warnings };
}

/**
 * ComposerManager — runs arbitrary per-node model combinations.
 *
 * Promotes the single-global "active setup" (SetupManager) into a per-node
 * desired-state engine: verify a composition, then apply it by diffing the
 * desired assignment against what is currently serving and starting/stopping
 * bricks accordingly, and regenerating the llama-swap gateway config.
 */
export class ComposerManager {
  constructor() {
    this.catalog = new ModelCatalog();
    /** @type {Set<import('child_process').ChildProcess>} */
    this._procs = new Set();
    this._applying = false;
    this._cancelled = false;
    this._disposed = false;
    this._phase = "idle"; // idle | applying | failed
    this._error = null;
    this._log = [];
    /** Last-detected running set: [{id, node, port, servedModel, up}] */
    this._running = [];
    /** Last applied assignment (best-effort). */
    this._currentAssignment = null;
    /** @type {null | (() => void)} */
    this.onChange = null;

    this._detectTimer = setInterval(() => {
      this._detect().catch(() => {});
    }, DETECT_INTERVAL_MS);
    if (typeof this._detectTimer.unref === "function") this._detectTimer.unref();
    this._detect().catch(() => {});
  }

  dispose() {
    this._disposed = true;
    if (this._detectTimer) clearInterval(this._detectTimer);
    this._detectTimer = null;
    this.onChange = null;
    for (const p of this._procs) this._killChild(p);
  }

  // ─── State / presets ─────────────────────────────────────
  isBusy() {
    return this._applying;
  }

  getState() {
    const cat = this.catalog.toPublic();
    const perNode = {};
    for (const node of Object.keys(cat.nodeCapacityGB)) {
      const running = this._running.filter((r) => r.node === node);
      const ramUsed = running.reduce(
        (s, r) => s + (Number(this.catalog.getModel(r.id)?.ramGB) || 0),
        0
      );
      perNode[node] = {
        running,
        ramUsed,
        ramCap: this.catalog.nodeCapacityGB(node),
        budget: this.catalog.nodeBudgetGB(node),
      };
    }
    return {
      catalog: cat,
      perNode,
      presets: this._loadPresets(),
      phase: this._phase,
      error: this._error,
      applying: this._applying,
      currentAssignment: this._currentAssignment,
      log: this._applying || this._phase === "failed" ? this._log.slice(-120) : [],
    };
  }

  _loadPresets() {
    try {
      const data = JSON.parse(fs.readFileSync(COMPOSITIONS_PATH, "utf-8"));
      return Array.isArray(data?.presets) ? data.presets : [];
    } catch {
      return [];
    }
  }

  savePreset(preset) {
    if (!preset || typeof preset.name !== "string" || !preset.assignment) {
      throw new Error("preset requires { name, assignment }");
    }
    const id = preset.id || preset.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    const presets = this._loadPresets().filter((p) => p.id !== id);
    presets.push({ id, name: preset.name, assignment: preset.assignment });
    fs.writeFileSync(COMPOSITIONS_PATH, JSON.stringify({ version: 1, presets }, null, 2));
    this._emit();
    return { id };
  }

  deletePreset(id) {
    const presets = this._loadPresets().filter((p) => p.id !== id);
    fs.writeFileSync(COMPOSITIONS_PATH, JSON.stringify({ version: 1, presets }, null, 2));
    this._emit();
    return { deleted: true };
  }

  // ─── Verify ──────────────────────────────────────────────
  verify(assignment) {
    return validateAssignment(this.catalog, assignment);
  }

  // ─── Detection (per-node running set) ────────────────────
  /** Force a fresh detection and return the current state (used by GET /state). */
  async detectNow() {
    await this._detect();
    return this.getState();
  }

  async _detect() {
    if (this._disposed || this._applying) return;
    // Probe every eligible endpoint in PARALLEL — sequential probing stalls on
    // each down endpoint's timeout and made the board lag reality.
    const checks = [];
    for (const model of this.catalog.models) {
      const eligible = Array.isArray(model.nodes) ? model.nodes : Object.keys(model.launch || {});
      for (const node of eligible) {
        const spec = model.launch?.[node] || model.launch?.[eligible[0]];
        const ready = spec?.ready;
        if (!ready) continue;
        checks.push(
          this._probeReady(ready).then((up) =>
            up
              ? {
                  id: model.id,
                  node,
                  port: ready.port,
                  servedModel: ready.servedModel || model.servedModel,
                  up: true,
                }
              : null
          )
        );
      }
    }
    const found = (await Promise.all(checks)).filter(Boolean);
    // One entry per model (a model eligible on multiple nodes runs on one).
    const seen = new Set();
    const running = [];
    for (const r of found) {
      if (seen.has(r.id)) continue;
      seen.add(r.id);
      running.push(r);
    }
    const changed = JSON.stringify(running) !== JSON.stringify(this._running);
    this._running = running;
    if (changed) this._emit();
  }

  async _probeReady(ready) {
    const host = ready.host || "127.0.0.1";
    try {
      const res = await fetch(`http://${host}:${ready.port}/v1/models`, {
        signal: AbortSignal.timeout(3500),
      });
      if (!res.ok) return false;
      const data = await res.json().catch(() => null);
      const ids = (data?.data || []).map((m) => String(m.id).toLowerCase());
      const want = String(ready.servedModel || "").toLowerCase();
      return want ? ids.some((x) => x.includes(want)) : ids.length > 0;
    } catch {
      return false;
    }
  }

  // ─── Apply ───────────────────────────────────────────────
  /** Realize `assignment`: stop removed bricks, start new ones, regen router. */
  async apply(assignment) {
    if (this._applying) throw new Error("An apply is already in progress");
    const check = this.verify(assignment);
    if (!check.ok) throw new Error(`Invalid config: ${check.errors.join("; ")}`);

    this._applying = true;
    this._cancelled = false;
    this._error = null;
    this._log = [];
    this._setPhase("applying");
    try {
      await this._detect();
      const desired = this._runUnits(assignment); // [{id, node, model, spec}]
      const running = [...this._running];
      // Diff by id+node so moving a movable brick (e.g. laguna spark2→spark1)
      // is a stop-on-old + start-on-new, not a mistaken no-op.
      const desiredKey = new Set(desired.map((u) => `${u.id}@${u.node}`));
      const runningKey = new Set(running.map((r) => `${r.id}@${r.node}`));

      // Keep (don't touch) models already serving where the new config wants
      // them — incremental apply, not a blanket teardown. Logged for visibility.
      const kept = desired.filter((u) => runningKey.has(`${u.id}@${u.node}`));
      for (const u of kept) this._appendLog(`Keeping ${u.id} on ${u.node} (already serving)`);

      // Stop anything running that isn't desired on that same node.
      const toStop = running.filter((r) => !desiredKey.has(`${r.id}@${r.node}`));
      for (const r of toStop) {
        const model = this.catalog.getModel(r.id);
        const spec = model?.launch?.[r.node] || Object.values(model?.launch || {})[0];
        if (spec) {
          this._appendLog(`Stopping ${r.id} on ${r.node}…`);
          await this._runAction(r.id, r.node, spec, "stop").catch((e) =>
            this._appendLog(`[warn] stop ${r.id}: ${e.message}`)
          );
        }
      }

      // Start desired bricks not already serving on their target node.
      const toStart = desired.filter((u) => !runningKey.has(`${u.id}@${u.node}`));
      await Promise.all(
        toStart.map((u) => {
          this._appendLog(`Starting ${u.id} on ${u.node}…`);
          return this._runAction(u.id, u.node, u.spec, "start");
        })
      );

      // Wait for the started bricks to answer a smoke chat.
      await this._waitReady(toStart);

      // Regenerate + write the llama-swap gateway (hot-reloads via -watch-config).
      try {
        const yaml = generateLlamaSwapConfig(this.catalog, assignment);
        fs.writeFileSync(LLAMA_SWAP_CONFIG_PATH, yaml);
        this._appendLog(`Regenerated llama-swap config → ${LLAMA_SWAP_CONFIG_PATH}`);
      } catch (e) {
        this._appendLog(`[warn] llama-swap regen failed: ${e.message}`);
      }

      this._currentAssignment = assignment;
      await this._detect();
      this._setPhase("idle");
      this._appendLog("Apply complete.");
    } catch (err) {
      this._error = err.message;
      this._appendLog(`FAILED: ${err.message}`);
      this._setPhase("failed");
      throw err;
    } finally {
      this._applying = false;
      this._emit(true);
    }
  }

  cancel() {
    if (!this._applying) return { cancelled: false };
    this._cancelled = true;
    this._appendLog("Cancel requested — terminating…");
    for (const p of this._procs) this._killChild(p);
    return { cancelled: true };
  }

  /** Desired run-units: single→its node; dual→its head launch node (started once). */
  _runUnits(assignment) {
    const units = [];
    const added = new Set();
    for (const node of Object.keys(assignment)) {
      for (const id of assignment[node] || []) {
        if (added.has(id)) continue;
        const model = this.catalog.getModel(id);
        if (!model) continue;
        const launchNode =
          model.placement === "dual" ? Object.keys(model.launch || {})[0] : node;
        const spec = model.launch?.[launchNode];
        if (!spec) continue;
        added.add(id);
        units.push({ id, node: launchNode, model, spec });
      }
    }
    return units;
  }

  // ─── Runner (local + ssh), modeled on SetupManager._runComponent ─────────
  _runAction(id, node, spec, action) {
    if (this._disposed) return Promise.reject(new Error("disposed"));
    const actionSpec = spec[action];
    if (!actionSpec || !actionSpec.cmd) return Promise.resolve();
    const extraEnv = spec.env && typeof spec.env === "object" ? spec.env : {};
    const opts = { env: { ...process.env, ...extraEnv }, detached: true };
    let file;
    let args;
    if (spec.target === "ssh") {
      if (!spec.host) return Promise.reject(new Error(`${id}: ssh launch missing host`));
      const envPrefix = Object.entries(extraEnv).map(([k, v]) => `${k}=${String(v)}`);
      const remote = [...envPrefix, actionSpec.cmd, ...(actionSpec.args || [])].join(" ");
      file = "ssh";
      args = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8", "--", spec.host, remote];
    } else {
      file = actionSpec.cmd;
      args = actionSpec.args || [];
      if (spec.cwd) opts.cwd = spec.cwd;
    }
    const timeoutMs = Number(spec.startTimeoutMs) || 900000;
    const label = `${id}@${node}`;

    return new Promise((resolve, reject) => {
      let child;
      try {
        child = spawn(file, args, opts);
      } catch (err) {
        reject(new Error(`${label}: failed to spawn (${err.message})`));
        return;
      }
      this._procs.add(child);
      let settled = false;
      const finish = (fn, arg) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this._procs.delete(child);
        fn(arg);
      };
      const timer = setTimeout(() => {
        this._appendLog(`[${label}] TIMEOUT after ${Math.round(timeoutMs / 1000)}s — killing`);
        this._killChild(child);
        finish(reject, new Error(`${label} ${action} timed out`));
      }, timeoutMs);
      const onData = (buf) => {
        if (settled) return;
        for (const line of String(buf).split("\n")) {
          const t = line.trimEnd();
          if (t) this._appendLog(`[${label}] ${t}`);
        }
      };
      child.stdout?.on("data", onData);
      child.stderr?.on("data", onData);
      child.on("error", (err) => finish(reject, new Error(`${label}: ${err.message}`)));
      child.on("close", (code, signal) => {
        if (code === 0) finish(resolve, undefined);
        else if (signal) finish(reject, new Error(`${label} ${action} terminated (${signal})`));
        else finish(reject, new Error(`${label} ${action} exited with code ${code}`));
      });
    });
  }

  _killChild(child) {
    try {
      if (child.pid) process.kill(-child.pid, "SIGTERM");
    } catch {
      try {
        child.kill("SIGTERM");
      } catch {
        /* already gone */
      }
    }
  }

  // ─── Readiness ───────────────────────────────────────────
  async _waitReady(units) {
    const endpoints = units.map((u) => u.spec.ready).filter(Boolean);
    if (endpoints.length === 0) return;
    const deadline = Date.now() + READY_TIMEOUT_MS;
    const pending = new Set(endpoints);
    while (Date.now() < deadline) {
      if (this._cancelled) throw new Error("cancelled during readiness wait");
      if (this._disposed) throw new Error("disposed");
      for (const ep of [...pending]) {
        if (await this._probeReady(ep)) {
          pending.delete(ep);
          this._appendLog(`[ready] ${ep.servedModel || `${ep.host}:${ep.port}`} OK`);
        }
      }
      if (pending.size === 0) return;
      await sleep(READY_POLL_MS);
    }
    const stuck = [...pending].map((e) => e.servedModel || `${e.host}:${e.port}`).join(", ");
    throw new Error(`readiness timeout waiting for: ${stuck}`);
  }

  // ─── Log + emit ──────────────────────────────────────────
  _appendLog(line) {
    this._log.push(line);
    if (this._log.length > LOG_MAX) this._log = this._log.slice(-LOG_MAX);
    this._emit();
  }

  _setPhase(phase) {
    this._phase = phase;
    this._emit(true);
  }

  _emit() {
    if (this._disposed || typeof this.onChange !== "function") return;
    try {
      this.onChange();
    } catch {
      /* ignore */
    }
  }
}
