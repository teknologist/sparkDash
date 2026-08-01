import fs from "fs";
import path from "path";
import { execFile, spawn } from "child_process";
import { MODEL_SETUPS_PATH, SETUP_LOGS_DIR } from "../config.js";

/**
 * SetupManager — owns the declarative model-setup registry, reports which
 * setup is running, and drives switching (stop the active setup, start the
 * selected one) as a single-flight background job.
 *
 * Setups are mutually exclusive: each occupies both Sparks' GPUs, so at most
 * one is "active" at a time. A switch runs asynchronously; progress (phase +
 * log tail) is exposed via snapshot() and pushed to WS clients through the
 * onChange callback.
 *
 * Requires a HOST-run backend: the launch scripts drive host `docker` /
 * `docker compose` + SSH with the host user's keys, so detection likewise
 * shells out to the host `docker`. When the backend runs inside the monitoring
 * container (no docker CLI / socket), detection reports a distinct `unknown`
 * state rather than a false "idle".
 */

const DETECT_CACHE_MS = 2000;
const DETECT_LOOP_MS = 4000;
const READY_TIMEOUT_MS = 180000;
const READY_POLL_MS = 5000;
const LOG_CAP = 500;
const LOG_TAIL = 120;
const EMIT_THROTTLE_MS = 250;
const KILL_GRACE_MS = 8000;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class SetupManager {
  constructor() {
    /** @type {any[]} raw setup definitions from disk */
    this._setups = [];
    this._load();

    // Fleet state machine. `phase` transitions during a switch; when idle the
    // detection loop keeps activeSetupId/phase in sync with reality.
    this._machine = {
      phase: "idle", // idle | starting | stopping | running | failed | unknown
      activeSetupId: null,
      targetSetupId: null,
      error: null,
    };
    /** @type {string[]} recent job log lines (ring buffer) */
    this._log = [];

    this._transitioning = false;
    this._cancelled = false;
    this._disposed = false;
    /** @type {Set<import('child_process').ChildProcess>} live job children */
    this._procs = new Set();

    this._detectCache = { at: 0, activeSetupId: null, error: null };
    /** @type {Array<{host:string,port:number,servedModel:string,up:boolean,modelId:string|null}>} */
    this._activeModels = [];
    /** Set by the server to force an immediate WS broadcast on state change. */
    this.onChange = null;
    this._lastEmit = 0;

    // Prime detection once, then keep it fresh on a slow loop.
    void this._reconcile();
    void this._refreshActiveModels();
    this._detectTimer = setInterval(() => {
      void this._reconcile();
      void this._refreshActiveModels();
    }, DETECT_LOOP_MS);
  }

  /** Clear timers and kill any in-flight job children (graceful shutdown). */
  dispose() {
    // Set first: a killed child's async `close` unwinds into the failure
    // handler, which must NOT spawn fresh cleanup/rollback processes now.
    this._disposed = true;
    if (this._detectTimer) clearInterval(this._detectTimer);
    this._detectTimer = null;
    // Silence broadcasts first: a killed child's async `close` would otherwise
    // drive activate()'s catch → _emit → onChange after we're shutting down.
    this.onChange = null;
    this._transitioning = false;
    for (const p of this._procs) this._killChild(p);
    this._procs.clear();
  }

  /**
   * SIGTERM a child's whole process group, escalating to SIGKILL after a grace
   * period. Group signalling (children are spawned `detached`) reaps grandchild
   * processes too — e.g. the `sleep`/`docker` a launch script forks — which a
   * signal to the direct child would orphan.
   */
  _killChild(child) {
    const pid = child.pid;
    const sig = (signal) => {
      try {
        if (pid) process.kill(-pid, signal); // negative pid → process group
        else child.kill(signal);
      } catch {
        try {
          child.kill(signal);
        } catch {
          /* already gone */
        }
      }
    };
    sig("SIGTERM");
    const t = setTimeout(() => sig("SIGKILL"), KILL_GRACE_MS);
    if (typeof t.unref === "function") t.unref();
  }

  // ─── Config load ─────────────────────────────────────────
  _load() {
    try {
      const raw = fs.readFileSync(MODEL_SETUPS_PATH, "utf-8");
      const data = JSON.parse(raw);
      const setups = Array.isArray(data?.setups) ? data.setups : [];
      this._setups = setups.filter((s) => s && typeof s.id === "string" && s.id.length > 0);
      if (this._setups.length !== setups.length) {
        console.warn("[SetupManager] dropped malformed setup entries in model-setups.json");
      }
    } catch (err) {
      if (err.code === "ENOENT") {
        console.warn("[SetupManager] model-setups.json not found — model switching disabled");
      } else {
        console.error("[SetupManager] failed to load model-setups.json:", err.message);
      }
      this._setups = [];
    }
  }

  get setups() {
    return this._setups;
  }

  getSetup(id) {
    return this._setups.find((s) => s.id === id) || null;
  }

  _name(id) {
    return this.getSetup(id)?.name || id;
  }

  /** Public, UI-facing view of a setup (no command/host internals). */
  toPublic(setup) {
    return {
      id: setup.id,
      name: setup.name || setup.id,
      description: setup.description || "",
      sparks: Array.isArray(setup.sparks) ? setup.sparks : [],
    };
  }

  listPublic() {
    return this._setups.map((s) => this.toPublic(s));
  }

  /** True while a switch (stop/start) is running — reject concurrent requests. */
  isBusy() {
    return this._transitioning;
  }

  // ─── Detection ───────────────────────────────────────────
  /**
   * Detect the running setup by matching declared container names against the
   * containers running on this host.
   * @returns {Promise<{activeSetupId: string|null, error: string|null}>}
   */
  async detectActiveId() {
    if (Date.now() - this._detectCache.at < DETECT_CACHE_MS) {
      return { activeSetupId: this._detectCache.activeSetupId, error: this._detectCache.error };
    }
    const probe = await this._runningContainers();
    if (!probe.ok) {
      // Docker CLI unavailable: container-based setups can't be detected, but a
      // native (non-container) setup that declares detect.endpoint still can.
      const viaEndpoint = await this._detectByEndpoint();
      if (viaEndpoint) {
        this._detectCache = { at: Date.now(), activeSetupId: viaEndpoint, error: null };
        return { activeSetupId: viaEndpoint, error: null };
      }
      this._detectCache = { at: Date.now(), activeSetupId: null, error: probe.error };
      return { activeSetupId: null, error: probe.error };
    }
    const running = probe.names;
    let best = null;
    let bestHits = 0;
    for (const setup of this._setups) {
      const names = setup?.detect?.localContainers;
      if (!Array.isArray(names) || names.length === 0) continue;
      // Match by exact name or compose "<project>-<service>-<n>" prefix — never
      // a loose substring, so a stray container can't false-trigger a setup.
      const hits = running.filter((r) => names.some((n) => r === n || r.startsWith(`${n}-`))).length;
      if (hits > bestHits) {
        best = setup.id;
        bestHits = hits;
      }
    }
    // No container matched — fall back to endpoint probing for native setups
    // (e.g. ds4-server, which runs as a bare process with no container).
    if (!best) best = await this._detectByEndpoint();
    this._detectCache = { at: Date.now(), activeSetupId: best, error: null };
    return { activeSetupId: best, error: null };
  }

  /**
   * Detect a running native setup by querying its declared serving endpoint and
   * matching the served model id — for setups that have no container to match
   * on. Returns the first setup id whose endpoint answers with the expected
   * model, or null. Only setups declaring detect.endpoint participate; multiple
   * such setups may share a port (distinguished by matchModel).
   * @returns {Promise<string|null>}
   */
  async _detectByEndpoint() {
    for (const setup of this._setups) {
      const ep = setup?.detect?.endpoint;
      if (!ep || !ep.port) continue;
      const host = ep.host || "127.0.0.1";
      try {
        const r = await fetch(`http://${host}:${ep.port}/v1/models`, {
          signal: AbortSignal.timeout(2500),
        });
        if (!r.ok) continue;
        const d = await r.json().catch(() => null);
        const id = String(d?.data?.[0]?.id || "");
        const needle = String(ep.matchModel || "").toLowerCase();
        if (!needle || id.toLowerCase().includes(needle)) return setup.id;
      } catch {
        /* endpoint not up */
      }
    }
    return null;
  }

  /** Detect, bypassing the short cache (used at the start of a switch). */
  _detectFresh() {
    this._detectCache.at = 0;
    return this.detectActiveId();
  }

  /**
   * @returns {Promise<{ok: true, names: string[]} | {ok: false, error: string}>}
   */
  _runningContainers() {
    return new Promise((resolve) => {
      execFile("docker", ["ps", "--format", "{{.Names}}"], { timeout: 4000 }, (err, stdout) => {
        if (err) {
          resolve({ ok: false, error: err.code === "ENOENT" ? "docker CLI not available" : err.message });
          return;
        }
        resolve({
          ok: true,
          names: String(stdout)
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
        });
      });
    });
  }

  /** Keep the machine's active/phase in sync with reality while idle. */
  async _reconcile() {
    // A switch owns phase during transitions; a `failed` result is sticky until
    // the user starts another action, so the failure + reason stay visible.
    if (this._transitioning || this._machine.phase === "failed") return;
    const { activeSetupId, error } = await this.detectActiveId();
    // A switch may have begun during the (up to 4s) docker probe above — bail
    // before overwriting the transition's phase/active state with stale data.
    if (this._transitioning) return;
    const phase = error ? "unknown" : activeSetupId ? "running" : "idle";
    const changed =
      this._machine.activeSetupId !== activeSetupId ||
      this._machine.phase !== phase ||
      this._machine.error !== (error || null);
    this._machine.activeSetupId = activeSetupId;
    this._machine.phase = phase;
    this._machine.error = error || null;
    if (changed) this._emit(true);
  }

  /**
   * Poll the active (or in-flight target) setup's serving endpoints so the UI
   * can show which models are actually up on each node — on their real ports,
   * not the fixed per-Spark llm probe port. Cheap GET /v1/models per endpoint.
   */
  async _refreshActiveModels() {
    const id = this._machine.targetSetupId || this._machine.activeSetupId;
    const setup = id ? this.getSetup(id) : null;
    if (!setup) {
      if (this._activeModels.length) {
        this._activeModels = [];
        this._emit(true);
      }
      return;
    }
    const eps = (setup.components || []).flatMap((c) => c.ready || []);
    const results = await Promise.all(
      eps.map(async (ep) => {
        const host = ep.host || "127.0.0.1";
        let up = false;
        let modelId = null;
        try {
          const r = await fetch(`http://${host}:${ep.port}/v1/models`, {
            signal: AbortSignal.timeout(3000),
          });
          if (r.ok) {
            up = true;
            const d = await r.json().catch(() => null);
            modelId = d?.data?.[0]?.id || null;
          }
        } catch {
          /* not up yet */
        }
        return { host, port: ep.port, servedModel: ep.servedModel || null, up, modelId };
      })
    );
    if (JSON.stringify(results) !== JSON.stringify(this._activeModels)) {
      this._activeModels = results;
      this._emit(true);
    }
  }

  /** Active setup's serving endpoints with liveness (for per-node model display). */
  getActiveModels() {
    return this._activeModels;
  }

  // ─── State snapshot (served by API / pushed over WS) ─────
  snapshot() {
    const transitioning = this._machine.phase === "starting" || this._machine.phase === "stopping";
    return {
      setups: this.listPublic(),
      activeSetupId: this._machine.activeSetupId,
      targetSetupId: this._machine.targetSetupId,
      phase: this._machine.phase,
      error: this._machine.error,
      // Only ship the log while a switch is in flight (the dialog only renders
      // it then) — otherwise every idle poll-tick broadcast would carry a stale
      // 120-line tail to all clients. Full logs persist to SETUP_LOGS_DIR.
      log: transitioning ? this._log.slice(-LOG_TAIL) : [],
    };
  }

  /** Alias kept for the REST GET handler. */
  getState() {
    return this.snapshot();
  }

  // ─── Switching (single-flight background job) ────────────
  /**
   * Stop the active setup (if any) and start `setupId`, then wait for readiness.
   * Runs to completion in the background; callers should not await it — poll
   * snapshot() / listen on WS for progress. Guarded by isBusy().
   */
  async activate(setupId) {
    const setup = this.getSetup(setupId);
    if (!setup) throw new Error(`Unknown setup: ${setupId}`);
    if (this._transitioning) throw new Error("A model switch is already in progress");

    this._transitioning = true;
    this._cancelled = false;
    this._resetLog();
    this._machine.targetSetupId = setupId;
    this._machine.error = null;
    let previous = null;
    try {
      const { activeSetupId: current } = await this._detectFresh();
      previous = current;
      if (current && current !== setupId) {
        this._setPhase("stopping");
        this._appendLog(`Stopping ${this._name(current)}…`);
        await this._runSetup(this.getSetup(current), "stop");
      }
      this._setPhase("starting");
      this._appendLog(`Starting ${this._name(setupId)}…`);
      await this._runSetup(setup, "start");
      this._appendLog("Waiting for API + smoke chat…");
      await this._waitReady(setup);
      this._machine.activeSetupId = setupId;
      this._machine.error = null;
      this._setPhase("running");
      this._appendLog(`${this._name(setupId)} is running.`);
    } catch (err) {
      await this._handleActivateFailure(setup, previous, err);
    } finally {
      this._machine.targetSetupId = null;
      this._transitioning = false;
      this._cancelled = false;
      this._emit(true);
      this._persistLog(`activate-${setupId}`);
    }
  }

  /**
   * A start/switch failed (or was cancelled). Clean up the half-started target,
   * then perform a single bounded rollback to the previously-running setup.
   * If rollback also fails, settle on `failed` with a combined error.
   */
  async _handleActivateFailure(setup, previous, err) {
    // Shutting down — don't spawn cleanup/rollback children after dispose().
    if (this._disposed) return;
    this._machine.error = err.message;
    this._appendLog(`ERROR: ${err.message}`);

    if (this._cancelled) {
      this._appendLog("Cancelling — stopping partially-started setup…");
      await this._runSetup(setup, "stop").catch(() => {});
      // If cancel landed while stopping the OLD setup, it was interrupted
      // mid-teardown — stop it too so the fleet ends cleanly, not half-up.
      if (previous && previous !== setup.id) {
        await this._runSetup(this.getSetup(previous), "stop").catch(() => {});
      }
      this._machine.activeSetupId = null;
      this._machine.error = "Switch cancelled";
      this._setPhase("failed");
      return;
    }

    // Stop the target so a partial start can't linger half-up.
    this._appendLog(`Cleaning up ${this._name(setup.id)}…`);
    await this._runSetup(setup, "stop").catch(() => {});

    if (!previous || previous === setup.id) {
      this._machine.activeSetupId = null;
      this._setPhase("failed");
      return;
    }

    // Single bounded auto-rollback to the previously-running setup.
    this._machine.targetSetupId = previous;
    this._setPhase("starting");
    this._appendLog(`Rolling back to ${this._name(previous)}…`);
    try {
      await this._runSetup(this.getSetup(previous), "start");
      await this._waitReady(this.getSetup(previous));
      this._machine.activeSetupId = previous;
      this._machine.error = null;
      this._setPhase("running");
      this._appendLog(`Rolled back to ${this._name(previous)}.`);
    } catch (rbErr) {
      this._appendLog(`ROLLBACK FAILED: ${rbErr.message}`);
      this._machine.activeSetupId = null;
      this._machine.error = `switch failed (${err.message}); rollback failed (${rbErr.message})`;
      this._setPhase("failed");
    }
  }

  /** Stop whatever is running and go idle. Guarded by isBusy(). */
  async stop() {
    if (this._transitioning) throw new Error("A model switch is already in progress");
    this._transitioning = true;
    this._cancelled = false;
    this._resetLog();
    this._machine.error = null;
    try {
      const { activeSetupId: current } = await this._detectFresh();
      if (current) {
        this._setPhase("stopping");
        this._appendLog(`Stopping ${this._name(current)}…`);
        await this._runSetup(this.getSetup(current), "stop");
      }
      this._machine.activeSetupId = null;
      this._setPhase("idle");
      this._appendLog("Fleet stopped.");
    } catch (err) {
      this._machine.error = err.message;
      this._appendLog(`ERROR: ${err.message}`);
      this._setPhase("failed");
    } finally {
      this._transitioning = false;
      this._cancelled = false;
      this._emit(true);
      this._persistLog("stop");
    }
  }

  /**
   * Request cancellation of the in-flight switch: kill the current child(ren);
   * the rejection routes through _handleActivateFailure, which cleans up to a
   * stopped/failed state. No-op when nothing is running.
   */
  cancel() {
    if (!this._transitioning) return { cancelled: false };
    this._cancelled = true;
    this._appendLog("Cancel requested — terminating current step…");
    for (const p of this._procs) this._killChild(p);
    return { cancelled: true };
  }

  /** Run one action across a setup's components (parallel start when flagged). */
  async _runSetup(setup, action) {
    const components = Array.isArray(setup.components) ? setup.components : [];
    if (components.length === 0) throw new Error(`Setup ${setup.id} has no components`);
    const runOne = (c) => this._runComponent(setup, c, action);
    // Start may fan out to independent machines; stop always runs all.
    if (action === "start" && setup.startParallel) {
      // allSettled so every child exits before we surface a failure — the
      // caller's cleanup `stop` then won't race a still-launching sibling.
      const results = await Promise.allSettled(components.map(runOne));
      const failed = results.filter((r) => r.status === "rejected");
      if (failed.length) {
        throw new Error(failed.map((f) => f.reason?.message || String(f.reason)).join("; "));
      }
    } else {
      for (const c of components) await runOne(c);
    }
  }

  /** Spawn one component's command, stream output to the log, resolve on exit 0. */
  _runComponent(setup, component, action) {
    if (this._disposed) return Promise.reject(new Error("SetupManager disposed"));
    const spec = component[action];
    if (!spec) return Promise.resolve(); // component doesn't define this action
    let file;
    let args;
    // Full host env on purpose (unlike collectors/ssh.js): the launch scripts
    // need the host user's PATH/HOME and SSH_AUTH_SOCK to reach the worker Spark
    // and drive docker; and the ssh client below needs the agent to auth.
    // Per-component `env` (config) is layered on top — e.g. ENABLE_SPEC=1.
    const extraEnv = component.env && typeof component.env === "object" ? component.env : {};
    const opts = { env: { ...process.env, ...extraEnv } };
    if (component.target === "ssh") {
      if (!component.host || !component.cmd) {
        return Promise.reject(new Error(`ssh component ${component.id} missing host/cmd`));
      }
      // Forward per-component env to the remote command (local env doesn't cross ssh).
      const envPrefix = Object.entries(extraEnv).map(([k, v]) => `${k}=${String(v)}`);
      const remoteCmd = [...envPrefix, component.cmd, ...(spec.args || [])].join(" ");
      file = "ssh";
      // accept-new + `--` mirror collectors/ssh.js so a first-seen host key
      // doesn't hang/fail under BatchMode on the LAN.
      args = [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8",
        "--", component.host, remoteCmd,
      ];
    } else {
      if (!spec.cmd) return Promise.reject(new Error(`component ${component.id} missing cmd`));
      file = spec.cmd;
      args = spec.args || [];
      if (component.cwd) opts.cwd = component.cwd;
    }

    const timeoutMs = Number(component.startTimeoutMs) || 900000;
    const label = component.id || setup.id;

    return new Promise((resolve, reject) => {
      let child;
      try {
        // detached → child leads its own process group so _killChild can reap
        // the whole tree (bash → sleep/docker/ssh) on cancel/timeout/dispose.
        child = spawn(file, args, { ...opts, detached: true });
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
        this._killChild(child); // SIGTERM now, SIGKILL after grace if it survives
        finish(reject, new Error(`${label} ${action} timed out`));
      }, timeoutMs);

      const onData = (buf) => {
        if (settled) return; // never bleed a dead job's late output into the next
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

  // ─── Readiness (smoke chat per serving endpoint) ─────────
  async _waitReady(setup) {
    const endpoints = (setup.components || []).flatMap((c) => c.ready || []);
    if (endpoints.length === 0) return;
    const deadline = Date.now() + READY_TIMEOUT_MS;
    const pending = new Set(endpoints);
    while (Date.now() < deadline) {
      if (this._cancelled) throw new Error("cancelled during readiness wait");
      if (this._disposed) throw new Error("SetupManager disposed");
      for (const ep of [...pending]) {
        if (await this._smokeChat(ep)) {
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

  /** One OpenAI-compatible smoke chat; true iff the endpoint answers a completion. */
  async _smokeChat(ep) {
    const host = ep.host || "127.0.0.1";
    const base = `http://${host}:${ep.port}`;
    try {
      const models = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(4000) });
      if (!models.ok) return false;
      const md = await models.json().catch(() => null);
      const actualId = md?.data?.[0]?.id || null;
      // Chat with the server's ACTUAL model id — a configured `servedModel` may
      // be a nickname/prefix the server rejects as the `model` param.
      const model = actualId || ep.servedModel;
      const chat = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "ping" }],
          max_tokens: 1,
          temperature: 0,
        }),
        signal: AbortSignal.timeout(8000),
      });
      if (!chat.ok) return false;
      const cd = await chat.json().catch(() => null);
      if (!(Array.isArray(cd?.choices) && cd.choices.length > 0)) return false;
      // When a servedModel is expected, confirm the RIGHT model is up (guards
      // against a leftover/other model answering on the same port).
      if (ep.servedModel && actualId && !actualId.toLowerCase().includes(ep.servedModel.toLowerCase())) {
        return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  // ─── Log + emit helpers ──────────────────────────────────
  _resetLog() {
    this._log = [];
  }

  _appendLog(line) {
    this._log.push(line);
    if (this._log.length > LOG_CAP) this._log.splice(0, this._log.length - LOG_CAP);
    this._emit(false);
  }

  _setPhase(phase) {
    this._machine.phase = phase;
    this._emit(true);
  }

  /** Write the completed job's log to SETUP_LOGS_DIR and prune old files. */
  _persistLog(name) {
    try {
      fs.mkdirSync(SETUP_LOGS_DIR, { recursive: true });
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      fs.writeFileSync(path.join(SETUP_LOGS_DIR, `${ts}-${name}.log`), this._log.join("\n") + "\n");
      this._pruneLogs(50);
    } catch (err) {
      console.error("[SetupManager] failed to persist switch log:", err.message);
    }
  }

  _pruneLogs(keep) {
    try {
      const files = fs
        .readdirSync(SETUP_LOGS_DIR)
        .filter((f) => f.endsWith(".log"))
        .sort();
      for (const f of files.slice(0, Math.max(0, files.length - keep))) {
        fs.unlinkSync(path.join(SETUP_LOGS_DIR, f));
      }
    } catch {
      /* best-effort */
    }
  }

  /** Notify the server to broadcast. Throttled unless `force`. */
  _emit(force) {
    if (typeof this.onChange !== "function") return;
    const now = Date.now();
    if (!force && now - this._lastEmit < EMIT_THROTTLE_MS) return;
    this._lastEmit = now;
    try {
      this.onChange();
    } catch {
      /* ignore broadcast errors */
    }
  }
}
