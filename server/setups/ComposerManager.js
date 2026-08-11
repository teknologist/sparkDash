import fs from "fs";
import { spawn, execFile } from "child_process";
import { ModelCatalog } from "./ModelCatalog.js";
import { generateLlamaSwapConfig } from "./llamaSwapGen.js";
import { allocateNode } from "./nodeAlloc.js";
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

  // Per-node memory (DYNAMIC util allocation) + port collisions.
  for (const node of nodes) {
    const list = resolved[node];
    const alloc = allocateNode(catalog, node, list.map((r) => r.id));
    const budget = catalog.nodeBudgetGB(node);
    const ramCap = catalog.nodeCapacityGB(node);
    const over = !alloc.ok;
    if (over) {
      errors.push(
        alloc.error
          ? `${node}: ${alloc.error}`
          : `${node} over budget: ${alloc.ramUsed} GB > ${budget} GB available (of ${ramCap} GB)`
      );
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
    perNode[node] = {
      models: list.map((r) => r.id),
      ramUsed: alloc.ramUsed,
      ramCap,
      budget,
      over,
      perModel: alloc.perModel, // { id: {footprintGB, util} } — util drives GPU_UTIL at launch
    };
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
    /**
     * gpu-util each unit was LAUNCHED with, keyed `id@node`. A vLLM reservation
     * is fixed at startup, so this is the only way to know whether a serving
     * model still matches the plan or is squatting a stale (larger) share.
     */
    this._launchedUtil = this._loadLaunchedUtil();
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

  /** Live per-node running set from the last detection probe: [{id,node,port,servedModel,up}]. */
  get running() {
    return this._running;
  }

  getState() {
    const cat = this.catalog.toPublic();
    const perNode = {};
    const isDual = (id) => this.catalog.getModel(id)?.placement === "dual";
    const dualRunning = this._running.filter((r) => isDual(r.id));
    for (const node of Object.keys(cat.nodeCapacityGB)) {
      // Single-node models on this node, plus any dual/TP model whose brick
      // spans this node (a dual serves from one endpoint but occupies all nodes).
      const running = this._running.filter((r) => r.node === node && !isDual(r.id));
      for (const r of dualRunning) {
        const nodesOf = this.catalog.getModel(r.id)?.nodes;
        if (Array.isArray(nodesOf) ? nodesOf.includes(node) : r.node === node) {
          running.push({ ...r, node });
        }
      }
      const alloc = allocateNode(this.catalog, node, running.map((r) => r.id));
      perNode[node] = {
        running,
        ramUsed: alloc.ramUsed,
        perModel: alloc.perModel,
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

  /** Path of the launched-util sidecar (next to compositions.json). */
  _launchedUtilPath() {
    return COMPOSITIONS_PATH.replace(/[^/]+$/, "launched-util.json");
  }

  /**
   * Launched utils survive a dashboard restart. Without this the map is empty on
   * boot, every serving elastic model reads as "unknown", and the next apply
   * needlessly re-launches healthy models (a multi-minute reload each).
   */
  _loadLaunchedUtil() {
    try {
      const d = JSON.parse(fs.readFileSync(this._launchedUtilPath(), "utf-8"));
      return d && typeof d === "object" && !Array.isArray(d) ? d : {};
    } catch {
      return {};
    }
  }

  _saveLaunchedUtil() {
    try {
      fs.writeFileSync(this._launchedUtilPath(), JSON.stringify(this._launchedUtil, null, 2));
    } catch (e) {
      this._appendLog(`[warn] could not persist launched utils: ${e.message}`);
    }
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
    // (1) Probe every eligible endpoint in PARALLEL — sequential probing stalls
    // on each down endpoint's timeout and made the board lag reality. A model
    // that answers /v1/models with its servedModel is READY (serving).
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
                  state: "ready",
                }
              : null
          )
        );
      }
    }
    const readyFound = (await Promise.all(checks)).filter(Boolean);
    const readyIds = new Set(readyFound.map((r) => r.id));

    // (2) LOADING detection: a brick whose dedicated `container` is up on a node
    // but whose API isn't serving yet is loading (weights + compile, minutes for
    // big models). The API is down during load, so the only real-time signal is
    // the node's container state — gather it with ONE `docker ps` per node, and
    // only for nodes that actually have a not-ready container-brick (idle-cheap).
    const containerBricks = [];
    const nodesNeeded = new Set();
    for (const model of this.catalog.models) {
      if (!model.container || readyIds.has(model.id)) continue;
      const eligible = Array.isArray(model.nodes) ? model.nodes : Object.keys(model.launch || {});
      containerBricks.push({ model, eligible });
      for (const node of eligible) nodesNeeded.add(node);
    }
    const contByNode = {};
    await Promise.all(
      [...nodesNeeded].map(async (node) => {
        contByNode[node] = await this._nodeContainers(node);
      })
    );
    const loadingFound = [];
    for (const { model, eligible } of containerBricks) {
      for (const node of eligible) {
        if (contByNode[node]?.has(model.container)) {
          const spec = model.launch?.[node] || model.launch?.[eligible[0]];
          loadingFound.push({
            id: model.id,
            node,
            port: spec?.ready?.port ?? model.port,
            servedModel: model.servedModel || model.id,
            up: false,
            state: "loading",
          });
          break; // a model loads on one node
        }
      }
    }

    // (3) Merge — one entry per model; a serving endpoint wins over a loading one.
    const seen = new Set();
    const running = [];
    for (const r of [...readyFound, ...loadingFound]) {
      if (seen.has(r.id)) continue;
      seen.add(r.id);
      running.push(r);
    }
    const changed = JSON.stringify(running) !== JSON.stringify(this._running);
    this._running = running;
    if (changed) this._emit();
  }

  /** node → exec target (local on spark1, ssh host elsewhere), from any launch spec. */
  _nodeExec(node) {
    for (const m of this.catalog.models) {
      const spec = m.launch?.[node];
      if (spec?.target === "local") return { target: "local" };
      if (spec?.target === "ssh" && spec.host) return { target: "ssh", host: spec.host };
    }
    return null;
  }

  /** Set of running container names on a node (one `docker ps`); empty on any error. */
  _nodeContainers(node) {
    const info = this._nodeExec(node);
    if (!info) return Promise.resolve(new Set());
    const file = info.target === "ssh" ? "ssh" : "docker";
    const args =
      info.target === "ssh"
        ? ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=6", "--", info.host, "docker ps --format '{{.Names}}'"]
        : ["ps", "--format", "{{.Names}}"];
    return new Promise((resolve) => {
      execFile(file, args, { timeout: 6000 }, (err, stdout) => {
        if (err) return resolve(new Set());
        resolve(new Set(String(stdout).split("\n").map((s) => s.trim()).filter(Boolean)));
      });
    });
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
      // EXACT match — substring matching false-positives across bricks that
      // share a prefix (e.g. "deepseek-v4-flash" ⊂ "deepseek-v4-flash-0731"),
      // which double-attributes the endpoint and breaks RAM/port accounting.
      return want ? ids.includes(want) : ids.length > 0;
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

      // A vLLM model reserves util×capacity at STARTUP and never shrinks. So a
      // model that is already serving with a bigger util than this assignment
      // grants it would squat that memory and starve the models we're about to
      // start — the co-residence then OOMs even though verify() approved the
      // plan. Such models must be RE-LAUNCHED at their new util, not "kept".
      const alreadyKept = desired.filter((u) => runningKey.has(`${u.id}@${u.node}`));
      const nodesGainingModels = new Set(
        desired.filter((u) => !runningKey.has(`${u.id}@${u.node}`)).map((u) => u.node)
      );
      const needsRelaunch = (u) => {
        if (u.util == null) return false; // fixed footprint (ds4/dual): no util knob
        const prev = this._launchedUtil[`${u.id}@${u.node}`];
        if (prev != null) return Math.abs(prev - u.util) > 0.02;
        // Unknown launch util (started outside the composer, or before a server
        // restart): only assume it is stale when this node's mix is changing.
        return nodesGainingModels.has(u.node);
      };
      const toRelaunch = alreadyKept.filter(needsRelaunch);
      const relaunchKey = new Set(toRelaunch.map((u) => `${u.id}@${u.node}`));
      for (const u of alreadyKept) {
        if (!relaunchKey.has(`${u.id}@${u.node}`)) {
          this._appendLog(`Keeping ${u.id} on ${u.node} (already serving)`);
        }
      }

      // Stop anything running that isn't desired on that same node, PLUS the
      // stale-reservation models above — freeing their memory before any start.
      const toStop = running.filter((r) => !desiredKey.has(`${r.id}@${r.node}`));
      for (const r of toStop) {
        const model = this.catalog.getModel(r.id);
        const spec = model?.launch?.[r.node] || Object.values(model?.launch || {})[0];
        if (spec) {
          this._appendLog(`Stopping ${r.id} on ${r.node}…`);
          await this._runAction(r.id, r.node, spec, "stop").catch((e) =>
            this._appendLog(`[warn] stop ${r.id}: ${e.message}`)
          );
          delete this._launchedUtil[`${r.id}@${r.node}`];
          this._saveLaunchedUtil();
        }
      }
      for (const u of toRelaunch) {
        const prev = this._launchedUtil[`${u.id}@${u.node}`];
        this._appendLog(
          `Re-launching ${u.id} on ${u.node}: gpu-util ${
            prev != null ? prev.toFixed(2) : "unknown"
          } → ${u.util.toFixed(2)} (must shrink to fit its new co-residents)…`
        );
        await this._runAction(u.id, u.node, u.spec, "stop").catch((e) =>
          this._appendLog(`[warn] stop ${u.id}: ${e.message}`)
        );
        delete this._launchedUtil[`${u.id}@${u.node}`];
        this._saveLaunchedUtil();
      }

      // Start desired bricks not already serving on their target node, plus the
      // ones we just stopped for resizing.
      const toStart = desired.filter(
        (u) => !runningKey.has(`${u.id}@${u.node}`) || relaunchKey.has(`${u.id}@${u.node}`)
      );
      // Start SEQUENTIALLY within a node, in parallel ACROSS nodes.
      //
      // vLLM sizes its KV cache from a one-shot memory profile at init. Two
      // engines profiling the same GPU at once each see the other's allocation
      // mid-flight, so the later one can conclude there is nothing left and die
      // with "No available memory for the cache blocks" — even when the plan
      // fits. Waiting for each model to finish coming up before starting its
      // node-mate makes every profile see a settled device.
      const startsByNode = new Map();
      for (const u of toStart) {
        if (!startsByNode.has(u.node)) startsByNode.set(u.node, []);
        startsByNode.get(u.node).push(u);
      }
      const startOne = async (u) => {
        const dynEnv = u.util != null ? { GPU_UTIL: u.util.toFixed(3) } : {};
        this._appendLog(
          `Starting ${u.id} on ${u.node}${u.util != null ? ` (gpu-util ${u.util.toFixed(2)})` : ""}…`
        );
        // Remember what we launched with so a later apply can tell whether the
        // live reservation still matches the plan (see needsRelaunch above).
        this._launchedUtil[`${u.id}@${u.node}`] = u.util != null ? u.util : null;
        this._saveLaunchedUtil();
        await this._runAction(u.id, u.node, u.spec, "start", dynEnv);
        await this._waitReady([u]); // settle before this node's next model
      };
      await Promise.all(
        [...startsByNode.values()].map(async (units) => {
          for (const u of units) {
            if (this._cancelled) break;
            await startOne(u);
          }
        })
      );

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
    // Per-node dynamic allocation gives each elastic (vLLM) model the GPU
    // utilization it should launch with, given its co-residents on that node.
    const utilByModel = {};
    for (const node of Object.keys(assignment)) {
      const alloc = allocateNode(this.catalog, node, assignment[node] || []);
      for (const [id, info] of Object.entries(alloc.perModel || {})) {
        if (info.util != null) utilByModel[id] = info.util;
      }
    }
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
        units.push({ id, node: launchNode, model, spec, util: utilByModel[id] });
      }
    }
    return units;
  }

  // ─── Runner (local + ssh), modeled on SetupManager._runComponent ─────────
  _runAction(id, node, spec, action, dynEnv = {}) {
    if (this._disposed) return Promise.reject(new Error("disposed"));
    const actionSpec = spec[action];
    if (!actionSpec || !actionSpec.cmd) return Promise.resolve();
    const extraEnv = {
      ...(spec.env && typeof spec.env === "object" ? spec.env : {}),
      ...dynEnv, // dynamic GPU_UTIL from the per-node allocation
    };
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
    // Honour each brick's declared startTimeoutMs: a cold 35B/NVFP4 recreate is
    // weights + torch.compile + cudagraph capture, which routinely exceeds the
    // 3-minute default. (This mattered once starts became sequential per node —
    // each model now gets its own window instead of sharing one.)
    const declared = units.map((u) => Number(u.spec.startTimeoutMs) || 0);
    const budget = Math.max(READY_TIMEOUT_MS, ...declared);
    const deadline = Date.now() + budget;
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
