/**
 * LauncherBridge — makes user-registered launcher models (upstream's
 * start.sh / stop.sh "llm-launchers") first-class composer bricks.
 *
 * The launcher stays the ONLY thing that runs those scripts (its logs, live
 * output and stop semantics are kept); the composer just sees each launcher as
 * a single-node brick so it can count its memory, stop it to make room, and
 * route it through the gateway.
 *
 * A launcher knows neither its memory nor its served model name. The composer
 * measures peak memory for every model (ModelStats); the served name is learned
 * the first time it answers /v1/models. Both persist across restarts/reboots.
 * Until its peak is measured a launcher model claims its whole node, so a first
 * start clears the node and later starts only stop what is actually in the way.
 */

export const LAUNCHER_PREFIX = "launcher:";

const STOP_WAIT_MS = 15 * 60 * 1000;
const JOB_POLL_MS = 1000;

export function launcherBrickId(sparkId, launcherId) {
  return `${LAUNCHER_PREFIX}${sparkId}:${launcherId}`;
}

export function parseLauncherBrickId(id) {
  if (typeof id !== "string" || !id.startsWith(LAUNCHER_PREFIX)) return null;
  const rest = id.slice(LAUNCHER_PREFIX.length);
  const i = rest.indexOf(":");
  if (i <= 0) return null;
  return { sparkId: rest.slice(0, i), launcherId: rest.slice(i + 1) };
}

export class LauncherBridge {
  /**
   * @param {object} deps
   * @param {{ list(sparkId:string): any[], get(sparkId:string, id:string): any }} deps.store
   * @param {{ startJob(spark:any, launcher:any, action:string): any, latestJob(sparkId:string): any }} deps.manager
   * @param {(sparkId:string) => any} deps.getSpark       registry lookup
   * @param {(sparkId:string) => (null | { usedGB:number, availableGB:number })} deps.nodeMemory
   * @param {(spark:any, launchers:any[]) => Promise<Record<string,string>>} deps.statuses  cached RUNNING/STOPPED per launcher id
   * @param {string[] | (() => string[])} deps.sparkIds   nodes (read live when a function)
   * @param {{ servedModel(id:string): string|null, setServedModel(id:string, name:string): void }} deps.stats
   * @param {string} deps.dashUrl      this dashboard, as reachable from the gateway host
   */
  constructor({ store, manager, getSpark, statuses, sparkIds, stats, dashUrl }) {
    Object.assign(this, { store, manager, getSpark, statuses, sparkIds, stats, dashUrl });
  }

  _ids() {
    return typeof this.sparkIds === "function" ? this.sparkIds() : this.sparkIds;
  }

  /**
   * Catalog bricks for every launcher that declares a port (the port is how the
   * composer can tell it is serving). Cheap and synchronous: called on every
   * catalog read.
   */
  bricks() {
    const out = [];
    for (const sparkId of this._ids()) {
      const spark = this.getSpark(sparkId);
      if (!spark) continue;
      for (const l of this.store.list(sparkId) || []) {
        if (!Number.isInteger(l.port)) continue;
        const id = launcherBrickId(sparkId, l.id);
        const served = this.stats.servedModel(id);
        const host = spark.isLocal ? "127.0.0.1" : spark.lanIp || spark.host;
        const ref = { sparkId, launcherId: l.id };
        out.push({
          id,
          displayName: l.name,
          backend: "launcher",
          placement: "single",
          nodes: [sparkId],
          // Resolved by ModelCatalog from the measured peak (whole node until measured).
          weightGB: null,
          port: l.port,
          servedModel: served || l.name,
          notes: l.notes || `Launcher model: ${l.dir}/${l.startScript}`,
          launcher: ref,
          // In the gateway only once its served name is known: llama-swap routes
          // by that name, and before the first run there is nothing to route.
          gateway: Boolean(served),
          launch: {
            [sparkId]: {
              target: "launcher",
              launcher: ref,
              start: { cmd: "launcher" },
              stop: { cmd: "launcher" },
              ready: { host, port: l.port, ...(served ? { servedModel: served } : { match: "any" }) },
              gatewayStart: `${this.dashUrl}/api/sparks/${encodeURIComponent(sparkId)}/llm-launchers/${encodeURIComponent(l.id)}/start?gateway=1`,
              gatewayStop: `${this.dashUrl}/api/sparks/${encodeURIComponent(sparkId)}/llm-launchers/${encodeURIComponent(l.id)}/stop`,
            },
          },
        });
      }
    }
    return out;
  }

  /** Brick ids whose launcher process is alive (upstream's own status check). */
  async runningIds() {
    const ids = new Set();
    await Promise.all(
      this._ids().map(async (sparkId) => {
        const spark = this.getSpark(sparkId);
        const launchers = (this.store.list(sparkId) || []).filter((l) => Number.isInteger(l.port));
        if (!spark || !launchers.length) return;
        try {
          const st = await this.statuses(spark, launchers);
          for (const l of launchers) if (st?.[l.id] === "running") ids.add(launcherBrickId(sparkId, l.id));
        } catch {
          /* unreachable Spark: report nothing rather than guess */
        }
      })
    );
    return ids;
  }

  learnServed(id, servedModel) {
    this.stats.setServedModel(id, servedModel);
  }

  /**
   * Run a launcher action through upstream's LauncherManager.
   * start: resolves once the job is spawned (the model then loads on its own).
   * stop:  resolves once stop.sh has finished, so memory is actually released.
   */
  async run(ref, action) {
    const spark = this.getSpark(ref.sparkId);
    const launcher = this.store.get(ref.sparkId, ref.launcherId);
    if (!spark || !launcher) throw new Error(`launcher ${ref.sparkId}/${ref.launcherId} not found`);
    const result = this.manager.startJob(spark, launcher, action);
    if (!result?.ok) {
      throw new Error(`launcher busy on ${spark.name || ref.sparkId} (${result?.active?.action} ${result?.active?.launcherName})`);
    }
    if (action !== "stop") return result.job;
    const deadline = Date.now() + STOP_WAIT_MS;
    while (Date.now() < deadline) {
      const job = this.manager.latestJob(ref.sparkId);
      if (!job || job.id !== result.job.id || job.status !== "running") {
        if (job?.id === result.job.id && job.status === "failed") throw new Error(`${launcher.name}: stop.sh failed`);
        return result.job;
      }
      await new Promise((r) => setTimeout(r, JOB_POLL_MS));
    }
    throw new Error(`${launcher.name}: stop did not finish in time`);
  }
}

/**
 * The composer model already watching `port` on `sparkId`, if any. A launcher
 * model on the same port would be detected twice (the 2026-10-10 Qwen test
 * counted 58 GB twice on spark2), so registering one there is refused.
 * @param {any[]} models catalog bricks
 * @returns {null | { id: string, name: string }}
 */
export function composerPortConflict(models, sparkId, port) {
  if (!Number.isInteger(port)) return null;
  for (const m of models || []) {
    if (m.launcher) continue;
    if (m.launch?.[sparkId]?.ready?.port === port) return { id: m.id, name: m.displayName || m.id };
  }
  return null;
}
