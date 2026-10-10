import fs from "fs";
import { spawn, execFile } from "child_process";
import { ModelCatalog } from "./ModelCatalog.js";
import { generateLlamaSwapConfig } from "./llamaSwapGen.js";
import { allocateNode } from "./nodeAlloc.js";
import { parseLoadPhase, loadProgress } from "./loadProgress.js";
import { ModelStats } from "./ModelStats.js";
import { COMPOSITIONS_PATH, LLAMA_SWAP_CONFIG_PATH, LLAMA_SWAP_URL } from "../config.js";
import { childEnv } from "./childEnv.js";

const DETECT_INTERVAL_MS = 5000;
const READY_TIMEOUT_MS = 180000;
const READY_POLL_MS = 5000;
const LOG_MAX = 400;
/**
 * How long a dual brick's peer container may run without its head before it is
 * reaped. A normal start launches the worker and then the head seconds apart, so
 * this only ever catches a half left behind by a stop that could not reach the
 * peer (observed: a GLM worker held 103 GiB on spark2 for 3 days, so every later
 * GLM start failed its memory check).
 */
const ORPHAN_GRACE_MS = 3 * 60 * 1000;
/** Free-memory wait (also drops the page cache CUDA counts as used). Same path on both nodes. */
const MEMCHECK = "/home/eric/sparkDash/deploy/memcheck.sh";
const CONTAINER_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
/** A "load" longer than this is a detection glitch, never recorded as a load time. */
const MAX_RECORDED_LOAD_SEC = 3 * 60 * 60;
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
    /** First time each `container@node` was seen orphaned (see _reapOrphans). */
    this._orphanSince = new Map();
    /** Measured load times + peak memory per brick (drives ETA and placement). */
    this.stats = new ModelStats(this._sidecarPath("model-stats.json"), {
      legacyLoadTimesFile: this._sidecarPath("load-times.json"),
    });
    this.catalog.setStats(this.stats);
    /** Open peak-memory measurements: id → { nodes, baseline, frozen, approx, startedAt } */
    this._peakRuns = new Map();
    /** (node) => { usedGB, availableGB } | null — set by the server (live metrics). */
    this.nodeMemory = null;
    /** Container start (ms) of each brick seen loading, until it serves. */
    this._loadStart = new Map();
    /** @type {null | (() => void)} */
    this.onChange = null;

    this._detectTimer = setInterval(() => {
      this._detect().catch(() => {});
    }, DETECT_INTERVAL_MS);
    if (typeof this._detectTimer.unref === "function") this._detectTimer.unref();
    this._detect().catch(() => {});
  }

  /**
   * What has to stop on `node` before `targetId` can start there.
   *
   * Uses the node's REAL available memory (not the composer's estimate), so
   * anything else holding memory is accounted for. A target whose footprint was
   * never measured claims the whole node: everything on it is listed. Otherwise
   * the biggest models go first, so as few as possible are stopped. A dual
   * model is listed with both nodes — stopping it frees both.
   */
  planRoom(node, targetId) {
    const target = this.catalog.getModel(targetId);
    if (!target) return null;
    const firstRun = target.footprintKnown === false;
    const budget = this.catalog.nodeBudgetGB(node);
    const needGB = firstRun ? budget : Number(target.weightGB) || 0;
    const state = this.getState();
    const pn = state.perNode[node] || { running: [], perModel: {}, ramUsed: 0 };
    const seen = new Set();
    const running = pn.running.filter((r) => r.id !== targetId && !seen.has(r.id) && seen.add(r.id));
    const footprint = (id) =>
      pn.perModel?.[id]?.footprintGB ?? (Number(this.catalog.getModel(id)?.weightGB) || 0);
    const mem = this.nodeMemory?.(node);
    const freeGB = mem
      ? Math.max(0, mem.availableGB - this.catalog.reserveGB())
      : Math.max(0, budget - (pn.ramUsed || 0));

    let victims = [];
    if (firstRun) {
      victims = running;
    } else if (freeGB < needGB) {
      let freed = 0;
      for (const r of [...running].sort((a, b) => footprint(b.id) - footprint(a.id))) {
        if (freeGB + freed >= needGB) break;
        victims.push(r);
        freed += footprint(r.id);
      }
    }
    const freedGB = victims.reduce((s, r) => s + footprint(r.id), 0);
    return {
      node,
      targetId,
      targetName: target.displayName || target.id,
      firstRun,
      needGB: Math.round(needGB),
      freeGB: Math.round(freeGB),
      fits: firstRun || freeGB + freedGB >= needGB,
      victims: victims.map((r) => {
        const m = this.catalog.getModel(r.id);
        return {
          id: r.id,
          name: m?.displayName || r.id,
          footprintGB: Math.round(footprint(r.id)),
          nodes: m?.placement === "dual" ? m.nodes : [node],
        };
      }),
    };
  }

  /**
   * Stop the plan's victims, wait until the memory is really free, then start
   * the target. Runs as an apply (blocks other applies, streams to the log).
   */
  async makeRoomAndStart(plan) {
    if (this._applying) throw new Error("An apply is already in progress");
    this._applying = true;
    this._cancelled = false;
    this._error = null;
    this._log = [];
    this._setPhase("applying");
    try {
      if (plan.victims.length) {
        this._appendLog(
          `Making room on ${plan.node} for ${plan.targetName}: stopping ${plan.victims.map((v) => v.name).join(", ")}`
        );
      }
      for (const v of plan.victims) {
        if (this._cancelled) throw new Error("cancelled");
        const m = this.catalog.getModel(v.id);
        const node = m?.placement === "dual" ? Object.keys(m.launch || {})[0] : plan.node;
        const spec = m?.launch?.[node] || Object.values(m?.launch || {})[0];
        if (!spec) continue;
        this._appendLog(`Stopping ${v.name}${v.nodes.length > 1 ? ` (${v.nodes.join(" + ")})` : ""}…`);
        await this._runAction(v.id, node, spec, "stop").catch((e) =>
          this._appendLog(`[warn] stop ${v.name}: ${e.message}`)
        );
      }
      if (plan.victims.length || plan.needGB > 0) {
        const need = plan.firstRun ? 0 : plan.needGB;
        this._appendLog(`Waiting for ${need ? `${need} GB` : "memory"} to be free on ${plan.node}…`);
        if (!(await this._waitFree(plan.node, need))) {
          throw new Error(`${plan.node} still lacks ${need} GB after stopping — not starting ${plan.targetName}`);
        }
      }
      const target = this.catalog.getModel(plan.targetId);
      this._appendLog(`Starting ${plan.targetName} on ${plan.node}…`);
      await this._runAction(plan.targetId, plan.node, target.launch[plan.node], "start");
      this._appendLog(`[ready] ${plan.targetName} is loading`);
      this._applying = false;
      await this._detect();
      this._setPhase("idle");
    } catch (e) {
      this._error = e.message;
      this._appendLog(`[error] ${e.message}`);
      this._setPhase("failed");
    } finally {
      this._applying = false;
      this._emit();
    }
  }

  /** memcheck.sh on a node: waits (up to 4 min) for `needGB` free, dropping the page cache. */
  _waitFree(node, needGB) {
    const info = this._nodeExec(node);
    if (!info) return Promise.resolve(true);
    const n = String(Math.max(0, Math.round(needGB)));
    const [file, args] =
      info.target === "ssh"
        ? ["ssh", ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--", info.host, `${MEMCHECK} ${n}`]]
        : [MEMCHECK, [n]];
    return new Promise((resolve) => {
      execFile(file, args, { timeout: 300000 }, (err, _out, stderr) => {
        if (stderr) for (const l of String(stderr).split("\n").filter(Boolean)) this._appendLog(`[${node}] ${l}`);
        resolve(!err);
      });
    });
  }

  /**
   * Treat user-registered launcher models as bricks: they join the catalog,
   * count against node memory, can be stopped to make room, and are routed
   * through the gateway once their served name is known.
   * @param {import('./launcherBridge.js').LauncherBridge} bridge
   */
  attachLauncherBridge(bridge) {
    this.launchers = bridge;
    this.catalog.setExtraModels(() => bridge.bricks());
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
  /** A state file next to compositions.json (config/, survives restarts and reboots). */
  _sidecarPath(name) {
    return COMPOSITIONS_PATH.replace(/[^/]+$/, name);
  }

  _readJsonSidecar(name) {
    try {
      const d = JSON.parse(fs.readFileSync(this._sidecarPath(name), "utf-8"));
      return d && typeof d === "object" && !Array.isArray(d) ? d : {};
    } catch {
      return {};
    }
  }

  /** A brick that was seen loading now serves: remember how long it took. */
  _recordLoad(id, startedAtMs, now = Date.now()) {
    const sec = Math.round((now - startedAtMs) / 1000);
    if (!(sec > 0) || sec > MAX_RECORDED_LOAD_SEC) return;
    this.stats?.recordLoad(id, sec);
    this._appendLog(`[ready] ${id} came up in ${sec}s`);
  }

  /**
   * Start measuring a brick's peak memory: baseline = node memory in use now,
   * right before it loads (exact), or when first seen already loading (approx).
   * A new start on a node ends the attribution window of models already on it,
   * so a co-resident's memory is never charged to them.
   */
  beginPeak(id, node, { approx = false } = {}) {
    if (!this.nodeMemory || !this.stats) return;
    const m = this.catalog.getModel(id);
    if (!m) return;
    const nodes = m.placement === "dual" ? m.nodes || [node] : [node];
    const baseline = {};
    for (const n of nodes) {
      const mem = this.nodeMemory(n);
      if (!mem) return;
      baseline[n] = mem.usedGB;
    }
    for (const [other, run] of this._peakRuns) {
      if (other !== id && run.nodes.some((n) => nodes.includes(n))) run.frozen = true;
    }
    this.stats.beginPeakRun(id, { approx });
    this._peakRuns.set(id, { nodes, baseline, frozen: false, approx, startedAt: Date.now() });
  }

  /** Raise each open run's peak from live node memory; close runs whose model left. */
  _trackPeaks(running, now = Date.now()) {
    if (!this.nodeMemory || !this.stats) return;
    this._peakRuns ??= new Map();
    const live = new Set(running.map((r) => r.id));
    for (const r of running) {
      if (r.state === "loading" && !this._peakRuns.has(r.id)) this.beginPeak(r.id, r.node, { approx: true });
    }
    for (const [id, run] of [...this._peakRuns]) {
      // A just-started model may not be detected yet: give it a moment.
      if (!live.has(id)) {
        if (now - run.startedAt > 120_000) this._peakRuns.delete(id);
        continue;
      }
      if (run.frozen) continue;
      let delta = 0;
      for (const n of run.nodes) {
        const mem = this.nodeMemory(n);
        if (mem) delta = Math.max(delta, mem.usedGB - run.baseline[n]);
      }
      this.stats.updatePeak(id, delta);
    }
  }

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
    // NOT gated on _applying: an apply blocks in the gateway load for the whole
    // cold start (minutes), and skipping detection then froze the board on
    // "nothing running" exactly while the user was waiting to see the load.
    if (this._disposed) return;
    this._loadStart ??= new Map();
    this._lastReadyAt ??= new Map();
    this._gatewaySeenAt ??= new Map();
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
      for (const node of Object.keys(model.peerContainers || {})) nodesNeeded.add(node);
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
            container: model.container,
          });
          break; // a model loads on one node
        }
      }
    }
    const now = Date.now();
    // Launcher models: upstream's status check says the process is alive, but
    // the API does not answer yet => loading.
    const launcherRunning = this.launchers ? await this.launchers.runningIds() : new Set();
    for (const model of this.catalog.models) {
      if (!model.launcher || readyIds.has(model.id) || !launcherRunning.has(model.id)) continue;
      const node = model.nodes[0];
      if (!this._gatewaySeenAt.has(model.id)) this._gatewaySeenAt.set(model.id, now);
      loadingFound.push({
        id: model.id,
        node,
        port: model.port,
        servedModel: model.servedModel,
        up: false,
        state: "loading",
        startedAtMs: this._gatewaySeenAt.get(model.id),
      });
    }
    // Learn a launcher model's served name the first time it answers.
    await Promise.all(
      readyFound
        .filter((r) => this.catalog.getModel(r.id)?.launcher && !this.stats?.servedModel(r.id))
        .map(async (r) => {
          const ready = this.catalog.getModel(r.id).launch[r.node]?.ready;
          const served = ready && (await this._firstModelId(ready));
          if (served) {
            this.launchers.learnServed(r.id, served);
            r.servedModel = served;
            this._appendLog(`[learn] ${r.id} serves "${served}" — added to the gateway on the next Apply`);
          }
        })
    );
    // Bricks with no container (ds4, MiMo, script-launched…) are only visible
    // as loading through the gateway, which reports what it is starting.
    const starting = await this._gatewayStarting();
    const startingNames = new Set(starting.map((g) => g.model));
    for (const model of this.catalog.models) {
      if (model.gateway === false || model.launcher || readyIds.has(model.id)) continue;
      if (loadingFound.some((r) => r.id === model.id)) continue;
      const nodes = Object.keys(model.launch || {});
      const served = (n) => model.launch[n]?.ready?.servedModel || model.servedModel || model.id;
      const hit = starting.find((g) => nodes.some((n) => served(n) === g.model));
      if (!hit) continue;
      // Which node: the one whose ready endpoint the gateway proxies to.
      const node =
        nodes.find((n) => {
          const r = model.launch[n]?.ready;
          return r && served(n) === hit.model && hit.proxy === `http://${r.host || "127.0.0.1"}:${r.port}`;
        }) ?? nodes[0];
      if (!this._gatewaySeenAt.has(model.id)) this._gatewaySeenAt.set(model.id, now);
      loadingFound.push({
        id: model.id,
        node,
        port: model.launch[node]?.ready?.port ?? model.port,
        servedModel: served(node),
        up: false,
        state: "loading",
        startedAtMs: this._gatewaySeenAt.get(model.id),
      });
    }
    for (const id of [...this._gatewaySeenAt.keys()]) {
      if (!loadingFound.some((r) => r.id === id && r.startedAtMs != null)) this._gatewaySeenAt.delete(id);
    }

    // Progress for each loading brick: phase from its container's log tail
    // (when it has one), ETA from this brick's past loads.
    await Promise.all(
      loadingFound.map(async (r) => {
        let lines = [];
        if (r.container) {
          const info = await this._containerLoadInfo(r.node, r.container);
          delete r.container;
          if (!info) return;
          r.startedAtMs = info.startedAtMs;
          lines = info.lines;
        }
        if (r.startedAtMs == null) return;
        // Serving AFTER this start means the brick never left: a probe blip,
        // not a load. Timing it would record days (the container's age).
        if ((this._lastReadyAt.get(r.id) ?? 0) <= r.startedAtMs) this._loadStart.set(r.id, r.startedAtMs);
        r.progress = loadProgress({
          startedAtMs: r.startedAtMs,
          now,
          history: this.stats?.loads(r.id),
          ...parseLoadPhase(lines),
        });
      })
    );
    for (const r of loadingFound) delete r.startedAtMs;
    // A brick seen loading that now serves: record its load time. One that
    // vanished without serving (failed/stopped) is dropped unrecorded.
    const loadingIds = new Set(loadingFound.map((r) => r.id));
    for (const [id, startedAtMs] of [...this._loadStart]) {
      if (readyIds.has(id)) this._recordLoad(id, startedAtMs, now);
      if (readyIds.has(id) || !loadingIds.has(id)) this._loadStart.delete(id);
    }
    for (const id of readyIds) this._lastReadyAt.set(id, now);

    // (3) Merge — one entry per model PER NODE; a serving endpoint wins over a
    // loading one. Keyed by id@node rather than id alone: a small brick can
    // legitimately serve on both nodes at once (jeff is 3 GB, so it is run on
    // each node rather than swapped between them), and collapsing those to one
    // entry hid the second instance and under-reported that node's RAM. Dual
    // bricks must still yield a single entry: step (1) probes them once per
    // node they occupy (all hitting the head's endpoint), so they dedupe by id,
    // and getState() expands them across the nodes they occupy.
    const seen = new Set();
    const running = [];
    for (const r of [...readyFound, ...loadingFound]) {
      const key =
        this.catalog.getModel(r.id)?.placement === "dual" ? r.id : `${r.id}@${r.node}`;
      if (seen.has(key)) continue;
      seen.add(key);
      running.push(r);
    }
    const changed = JSON.stringify(running) !== JSON.stringify(this._running);
    this._running = running;
    this._trackPeaks(running);
    if (changed) this._emit();

    // Never repair mid-apply: the apply itself may be stopping/starting pairs.
    if (!this._applying) await this._reapOrphans(containerBricks, contByNode);
  }

  /**
   * Container start time and log tail on a node, in ONE command. Start time
   * comes from docker (not first-seen) so elapsed/ETA survive a dashboard restart.
   * @returns {Promise<null | { startedAtMs: number, lines: string[] }>}
   */
  _containerLoadInfo(node, name) {
    const info = this._nodeExec(node);
    if (!info || !CONTAINER_NAME_RE.test(name)) return Promise.resolve(null);
    const cmd = `docker inspect -f '{{.State.StartedAt}}' ${name} && docker logs --tail 80 ${name} 2>&1`;
    const [file, args] =
      info.target === "ssh"
        ? ["ssh", ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=6", "--", info.host, cmd]]
        : ["bash", ["-c", cmd]];
    return new Promise((resolve) => {
      execFile(file, args, { timeout: 8000, maxBuffer: 4 * 1024 * 1024 }, (err, stdout) => {
        if (err) return resolve(null);
        const [first, ...lines] = String(stdout).split("\n");
        const startedAtMs = Date.parse(first.trim());
        if (!Number.isFinite(startedAtMs)) return resolve(null);
        resolve({ startedAtMs, lines: lines.filter(Boolean) });
      });
    });
  }

  /**
   * Auto-repair: remove a dual brick's peer container (e.g. the TP worker) that
   * outlived its head. Without its head it serves nothing yet keeps its full
   * memory share, which makes every later start of that brick — or anything
   * else on that node — fail. Only bricks that declare `peerContainers` are
   * touched, only when not serving and the head container is gone, and only
   * after ORPHAN_GRACE_MS of continuous orphanhood.
   */
  async _reapOrphans(containerBricks, contByNode, now = Date.now()) {
    this._orphanSince ??= new Map();
    const seen = new Set();
    for (const { model } of containerBricks) {
      if (model.placement !== "dual" || !model.peerContainers) continue;
      const headNode = Object.keys(model.launch || {})[0];
      // Unknown head state (docker ps failed → empty set) must not read as "gone":
      // only judge when the head node actually answered.
      const headSet = contByNode[headNode];
      if (!headSet || !headSet.answered || headSet.has(model.container)) continue;
      for (const [node, names] of Object.entries(model.peerContainers)) {
        for (const name of names || []) {
          if (!CONTAINER_NAME_RE.test(name) || !contByNode[node]?.has(name)) continue;
          const key = `${name}@${node}`;
          seen.add(key);
          const since = this._orphanSince.get(key) ?? now;
          this._orphanSince.set(key, since);
          if (now - since < ORPHAN_GRACE_MS) continue;
          this._orphanSince.delete(key);
          this._appendLog(
            `[repair] ${model.id}: ${name} on ${node} has run without its head for ${Math.round((now - since) / 1000)}s — removing it`
          );
          const ok = await this._removeContainer(node, name);
          this._appendLog(ok ? `[repair] removed ${name} on ${node}` : `[warn] could not remove ${name} on ${node}`);
          this._emit();
        }
      }
    }
    for (const key of [...this._orphanSince.keys()]) {
      if (!seen.has(key)) this._orphanSince.delete(key);
    }
  }

  /** `docker rm -f <name>` on a node (local or over the composer's ssh host). */
  _removeContainer(node, name) {
    const info = this._nodeExec(node);
    if (!info || !CONTAINER_NAME_RE.test(name)) return Promise.resolve(false);
    const file = info.target === "ssh" ? "ssh" : "docker";
    const args =
      info.target === "ssh"
        ? ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8", "--", info.host, `docker rm -f ${name}`]
        : ["rm", "-f", name];
    return new Promise((resolve) => {
      execFile(file, args, { timeout: 60000 }, (err) => resolve(!err));
    });
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
        const names = new Set(String(stdout).split("\n").map((s) => s.trim()).filter(Boolean));
        names.answered = true; // distinguishes "no containers" from "docker ps failed"
        resolve(names);
      });
    });
  }

  /** First model id an endpoint reports (original case), or null. */
  async _firstModelId(ready) {
    try {
      const res = await fetch(`http://${ready.host || "127.0.0.1"}:${ready.port}/v1/models`, {
        signal: AbortSignal.timeout(3500),
      });
      if (!res.ok) return null;
      const d = await res.json().catch(() => null);
      const id = d?.data?.[0]?.id;
      return typeof id === "string" && id ? id : null;
    } catch {
      return null;
    }
  }

  async _probeReady(ready) {
    const host = ready.host || "127.0.0.1";
    try {
      // Most bricks are OpenAI-shaped: GET /v1/models -> {data:[{id}]}. A brick
      // whose /v1/models is auth-gated or differently shaped (jeff: bearer-only,
      // {models:[{name}]}) declares `endpoint` + `match` to be probed elsewhere —
      // `match: "model"` reads the single id from a health payload's `model`.
      const endpoint = ready.endpoint || "/v1/models";
      const res = await fetch(`http://${host}:${ready.port}${endpoint}`, {
        signal: AbortSignal.timeout(3500),
      });
      if (!res.ok) return false;
      const data = await res.json().catch(() => null);
      const ids =
        ready.match === "model"
          ? (data?.model ? [String(data.model).toLowerCase()] : [])
          : (data?.data || []).map((m) => String(m.id).toLowerCase());
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

      // ─── Converge via llama-swap ────────────────────────────────────────
      //
      // The gateway OWNS model lifecycle; the composer plans. llama-swap only
      // tracks models it started itself, so starting them here directly left it
      // believing nothing was loaded (observed: both models reported "unloaded"
      // while 101 GiB was held) and its eviction could never fire. Apply now
      // writes the config, then asks the gateway to load the desired set.
      // A brick with `gateway:false` is not an OpenAI chat backend (jeff answers
      // /v1/systemone only), so llama-swap can neither load nor proxy it. Those
      // are started directly here and kept out of the gateway's desired set.
      // Launcher models are started through upstream's launcher (it owns their
      // scripts); llama-swap adopts them on first request via its ready guard.
      const isDirect = (u) => u.model.gateway === false || Boolean(u.model.launcher);
      const directUnits = desired.filter(isDirect);
      const desiredNames = desired
        .filter((u) => !isDirect(u))
        .map((u) => u.spec.ready?.servedModel || u.model.servedModel || u.id);

      for (const u of directUnits) {
        if (this._cancelled) break;
        // Already serving: the alreadyKept loop above has logged "Keeping …"
        // for it, so stay silent here rather than logging the same line twice.
        if (await this._probeReady(u.spec.ready)) continue;
        this._appendLog(`Starting ${u.id} on ${u.node} directly (not a gateway model)…`);
        try {
          await this._runAction(u.id, u.node, u.spec, "start");
          this._appendLog(`[ready] ${u.id} started on ${u.node}`);
        } catch (e) {
          this._appendLog(`[warn] start ${u.id} on ${u.node}: ${e.message}`);
        }
      }

      // 1. Config first: it defines placement, the per-model GPU_UTIL and the
      //    exclusivity groups llama-swap needs before it can load anything.
      try {
        const yaml = generateLlamaSwapConfig(this.catalog, assignment);
        fs.writeFileSync(LLAMA_SWAP_CONFIG_PATH, yaml);
        this._appendLog(`Regenerated llama-swap config → ${LLAMA_SWAP_CONFIG_PATH}`);
        await sleep(3000); // let -watch-config pick it up
      } catch (e) {
        this._appendLog(`[warn] llama-swap regen failed: ${e.message}`);
      }

      // 2. Evict anything loaded that the new assignment does not want. There is
      //    no per-model unload endpoint, so this is all-or-nothing; skip it
      //    entirely when the loaded set is already a subset of the desired one.
      const loaded = await this._gatewayRunning();
      const stale = loaded.filter((m) => !desiredNames.includes(m));
      if (stale.length) {
        this._appendLog(`Unloading via gateway (not in assignment): ${stale.join(", ")}`);
        await this._gatewayUnloadAll();
      }

      // 3. Load the desired set, one at a time. Concurrent loads would race
      //    vLLM's one-shot memory profile, and deploy/memcheck.sh already
      //    serialises on free memory.
      for (const name of desiredNames) {
        if (this._cancelled) break;
        const already = await this._gatewayRunning();
        if (already.includes(name)) {
          this._appendLog(`Keeping ${name} (already loaded in the gateway)`);
          continue;
        }
        this._appendLog(`Asking gateway to load ${name}…`);
        const unit = desired.find((d) => !isDirect(d) && (d.spec.ready?.servedModel || d.model.servedModel || d.id) === name);
        if (unit) this.beginPeak(unit.id, unit.node);
        const ok = await this._gatewayLoad(name);
        this._appendLog(ok ? `[ready] ${name} loaded` : `[warn] gateway could not load ${name}`);
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
    // Keyed per NODE: the same brick co-residing on both nodes has different
    // co-residents on each, so it gets a different util on each.
    const utilByModel = {};
    for (const node of Object.keys(assignment)) {
      const alloc = allocateNode(this.catalog, node, assignment[node] || []);
      for (const [id, info] of Object.entries(alloc.perModel || {})) {
        if (info.util != null) utilByModel[`${id}@${node}`] = info.util;
      }
    }
    for (const node of Object.keys(assignment)) {
      for (const id of assignment[node] || []) {
        const model = this.catalog.getModel(id);
        if (!model) continue;
        const launchNode =
          model.placement === "dual" ? Object.keys(model.launch || {})[0] : node;
        // A DUAL brick is listed on every node it occupies but launches ONCE
        // from its head node, so it dedupes by id. A single-placement brick is
        // per-node: a small one (jeff, 3 GB) can legitimately serve on both
        // nodes at once, and collapsing those to one unit made apply() stop the
        // second instance as "running but not desired".
        const key = model.placement === "dual" ? id : `${id}@${launchNode}`;
        if (added.has(key)) continue;
        const spec = model.launch?.[launchNode];
        if (!spec) continue;
        added.add(key);
        units.push({ id, node: launchNode, model, spec, util: utilByModel[`${id}@${launchNode}`] });
      }
    }
    return units;
  }


  // ─── llama-swap gateway (it owns model lifecycle) ─────────
  /** Served-model names llama-swap currently has loaded (or starting). */
  async _gatewayRunning() {
    try {
      const res = await fetch(`${LLAMA_SWAP_URL}/running`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) return [];
      const d = await res.json().catch(() => null);
      return (d?.running || []).map((r) => r.model).filter(Boolean);
    } catch {
      return [];
    }
  }

  /** Models the gateway is currently starting: [{ model, proxy }]. Empty on any error. */
  async _gatewayStarting() {
    try {
      const res = await fetch(`${LLAMA_SWAP_URL}/running`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) return [];
      const d = await res.json().catch(() => null);
      return (d?.running || [])
        .filter((r) => r.model && r.state === "starting")
        .map((r) => ({ model: r.model, proxy: r.proxy || null }));
    } catch {
      return [];
    }
  }

  /** Unload everything. llama-swap exposes no per-model unload, only GET /unload. */
  async _gatewayUnloadAll() {
    try {
      await fetch(`${LLAMA_SWAP_URL}/unload`, { signal: AbortSignal.timeout(120000) });
      // Give the backends time to actually release; deploy/memcheck.sh waits too,
      // but returning early here would race the next load's profile.
      await sleep(5000);
      return true;
    } catch (e) {
      this._appendLog(`[warn] gateway unload failed: ${e.message}`);
      return false;
    }
  }

  /**
   * Load a model by asking the gateway for one token. This is the only way to
   * make llama-swap start a model AND record it as loaded — which is what makes
   * its eviction work later. Cold starts are minutes, hence the long timeout.
   */
  async _gatewayLoad(servedModel, timeoutMs = 1500000) {
    try {
      const res = await fetch(`${LLAMA_SWAP_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: servedModel,
          messages: [{ role: "user", content: "ok" }],
          max_tokens: 1,
          temperature: 0,
        }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!res.ok) {
        const t = await res.text().catch(() => "");
        this._appendLog(`[warn] ${servedModel}: gateway returned ${res.status} ${t.slice(0, 160)}`);
        return false;
      }
      await res.json().catch(() => null);
      return true;
    } catch (e) {
      this._appendLog(`[warn] ${servedModel}: ${e.message}`);
      return false;
    }
  }

  // ─── Runner (local + ssh), modeled on SetupManager._runComponent ─────────
  _runAction(id, node, spec, action, dynEnv = {}) {
    if (this._disposed) return Promise.reject(new Error("disposed"));
    if (action === "start") this.beginPeak(id, node);
    if (spec.target === "launcher") {
      if (!this.launchers) return Promise.reject(new Error(`${id}: launcher bridge not attached`));
      if (action === "status") return Promise.resolve();
      return this.launchers.run(spec.launcher, action).then(() => undefined);
    }
    const actionSpec = spec[action];
    if (!actionSpec || !actionSpec.cmd) return Promise.resolve();
    const extraEnv = {
      ...(spec.env && typeof spec.env === "object" ? spec.env : {}),
      ...dynEnv, // dynamic GPU_UTIL from the per-node allocation
    };
    const opts = { env: childEnv(extraEnv), detached: true };
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
