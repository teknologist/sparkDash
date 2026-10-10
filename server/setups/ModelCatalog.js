import fs from "fs";
import { MODELS_PATH, NODE_RAM_RESERVE_GB } from "../config.js";

/**
 * ModelCatalog — the "bricks" for the RAM-aware composer.
 *
 * Loads config/models.json and hot-reloads it on mtime change (no server
 * restart needed to add/tune a model). Each model declares its placement
 * (single|dual), eligible nodes, a RAM-cost estimate, port, servedModel, and a
 * per-node `launch` block (start/stop/status/ready) so the same brick can be
 * started locally or over ssh depending on which node it lands on.
 */
export class ModelCatalog {
  constructor() {
    /** @type {any[]} */
    this._models = [];
    /** @type {Record<string, number>} */
    this._nodeCapacityGB = {};
    this._mtime = 0;
    /** @type {null | (() => any[])} extra bricks (launcher models), read live */
    this._extra = null;
    this._load();
  }

  /**
   * Add bricks that do not live in models.json (user-registered launcher
   * models). A brick with no measured footprint (weightGB null) claims its
   * node's whole budget, so placing it never under-counts memory.
   */
  setExtraModels(fn) {
    this._extra = typeof fn === "function" ? fn : null;
  }

  /**
   * Measured stats (ModelStats) override configured estimates for placement:
   * a fixed-footprint model's `weightGB` becomes its exact measured peak (an
   * approximate one only ever raises it). Elastic vLLM models keep their
   * configured share — their footprint is set by gpu-util, not by history —
   * and just expose the peak for reference.
   */
  setStats(stats) {
    this._stats = stats || null;
  }

  _withMeasured(m) {
    const peak = this._stats?.peak(m.id) || { exact: null, approx: null };
    const budget = () => Math.min(...(m.nodes || Object.keys(m.launch || {})).map((n) => this.nodeBudgetGB(n)));
    const measuredPeakGB = peak.exact ?? peak.approx;
    if (m.launcher) {
      // Never measured: claim the whole node, so a first start clears it.
      const gb = peak.exact ?? (peak.approx != null ? Math.max(peak.approx, budget()) : null);
      return { ...m, weightGB: gb ?? budget(), footprintKnown: gb != null, measuredPeakGB };
    }
    if (measuredPeakGB == null) return m;
    if (Number(m.maxUtil) > 0) return { ...m, measuredPeakGB };
    const estimate = Number(m.weightGB != null ? m.weightGB : m.ramGB) || 0;
    const gb = peak.exact ?? Math.max(estimate, peak.approx);
    return { ...m, weightGB: gb, estimateGB: estimate, measuredPeakGB };
  }

  _load() {
    try {
      const stat = fs.statSync(MODELS_PATH);
      this._mtime = stat.mtimeMs;
      const data = JSON.parse(fs.readFileSync(MODELS_PATH, "utf-8"));
      const models = Array.isArray(data?.models) ? data.models : [];
      this._models = models.filter((m) => m && typeof m.id === "string" && m.id.length > 0);
      this._nodeCapacityGB =
        data?.nodeCapacityGB && typeof data.nodeCapacityGB === "object" ? data.nodeCapacityGB : {};
    } catch (err) {
      if (err.code === "ENOENT") {
        console.warn("[ModelCatalog] models.json not found — composer disabled");
      } else {
        console.error("[ModelCatalog] failed to load models.json:", err.message);
      }
      this._models = [];
      this._nodeCapacityGB = {};
    }
  }

  /** Re-read the file if it changed on disk since the last load. */
  _refreshIfChanged() {
    try {
      const m = fs.statSync(MODELS_PATH).mtimeMs;
      if (m !== this._mtime) this._load();
    } catch {
      /* ENOENT / transient — keep the last good catalog */
    }
  }

  get models() {
    this._refreshIfChanged();
    let extra = [];
    try {
      extra = this._extra ? this._extra() || [] : [];
    } catch {
      /* a broken provider must not take the file catalog down with it */
    }
    const all = [...this._models, ...extra];
    return this._stats ? all.map((m) => this._withMeasured(m)) : all;
  }

  getModel(id) {
    return this.models.find((m) => m.id === id) || null;
  }

  /** Total unified RAM for a node (GB). Defaults to a 128 GB GB10. */
  nodeCapacityGB(nodeId) {
    this._refreshIfChanged();
    const v = this._nodeCapacityGB?.[nodeId];
    return Number.isFinite(v) ? v : 128;
  }

  /** Headroom (GB) kept free of the model budget on every node. */
  reserveGB() {
    return NODE_RAM_RESERVE_GB;
  }

  /** Budget available to models on a node = capacity − reserve. */
  nodeBudgetGB(nodeId) {
    return Math.max(0, this.nodeCapacityGB(nodeId) - this.reserveGB());
  }

  /** UI-facing catalog: bricks without launch internals + per-node capacity. */
  toPublic() {
    return {
      models: this.models.map((m) => ({
        id: m.id,
        displayName: m.displayName || m.id,
        backend: m.backend || null,
        placement: m.placement === "dual" ? "dual" : "single",
        nodes: Array.isArray(m.nodes) ? m.nodes : Object.keys(m.launch || {}),
        weightGB: Number(m.weightGB != null ? m.weightGB : m.ramGB) || 0,
        maxUtil: Number(m.maxUtil) || null,
        ramGB: Number(m.weightGB != null ? m.weightGB : m.ramGB) || 0, // floor (compat)
        port: Number(m.port) || null,
        servedModel: m.servedModel || m.id,
        notes: m.notes || "",
        ...(m.launcher ? { launcher: true, footprintKnown: Boolean(m.footprintKnown) } : {}),
        ...(m.measuredPeakGB != null ? { measuredPeakGB: m.measuredPeakGB } : {}),
        ...(m.estimateGB != null ? { estimateGB: m.estimateGB } : {}),
      })),
      nodeCapacityGB: this.models.length
        ? Object.fromEntries(
            [...new Set(this.models.flatMap((m) => m.nodes || []))].map((n) => [
              n,
              this.nodeCapacityGB(n),
            ])
          )
        : {},
      reserveGB: this.reserveGB(),
    };
  }
}
