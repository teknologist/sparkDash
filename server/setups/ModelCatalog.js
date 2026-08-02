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
    this._load();
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
    return this._models;
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
        ramGB: Number(m.ramGB) || 0,
        port: Number(m.port) || null,
        servedModel: m.servedModel || m.id,
        notes: m.notes || "",
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
