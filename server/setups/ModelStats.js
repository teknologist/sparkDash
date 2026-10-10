import fs from "fs";

/**
 * ModelStats — what the composer has MEASURED about each model, persisted in
 * config/model-stats.json so it survives dashboard restarts and Spark reboots:
 *
 *  - loads:       recent load durations (s), container/process start → API ready.
 *                 Drives the ETA shown while a model loads.
 *  - peaks:       peak memory (GB) above the node's baseline, one entry per
 *                 recent run. Drives placement: a measured model is allowed or
 *                 refused on what it really used, not on a configured estimate.
 *  - servedModel: the id a launcher model's /v1/models reports (learned on its
 *                 first run; the launcher itself does not know it).
 *
 * Several runs are kept so one odd run (a late baseline, a profile change)
 * neither hides a real peak nor pins a stale one forever.
 */

export const LOAD_HISTORY_MAX = 5;
export const PEAK_RUNS_MAX = 3;

export class ModelStats {
  /**
   * @param {string} file
   * @param {{ legacyLoadTimesFile?: string }} [opts] imported once when `file` does not exist yet
   */
  constructor(file, { legacyLoadTimesFile } = {}) {
    this.file = file;
    /** @type {Record<string, { loads?: number[], peaks?: { gb: number, at: number, approx?: boolean }[], servedModel?: string }>} */
    this.models = {};
    this._load(legacyLoadTimesFile);
  }

  _load(legacy) {
    try {
      const d = JSON.parse(fs.readFileSync(this.file, "utf-8"));
      if (d && typeof d.models === "object" && !Array.isArray(d.models)) this.models = d.models;
      return;
    } catch {
      /* missing or unreadable: start empty, maybe migrate below */
    }
    if (!legacy) return;
    try {
      const old = JSON.parse(fs.readFileSync(legacy, "utf-8"));
      for (const [id, loads] of Object.entries(old || {})) {
        if (Array.isArray(loads)) this.models[id] = { loads: loads.filter((x) => Number.isFinite(x)) };
      }
      if (Object.keys(this.models).length) this._save();
    } catch {
      /* no legacy file */
    }
  }

  _save() {
    try {
      fs.writeFileSync(this.file, JSON.stringify({ version: 1, models: this.models }, null, 2));
    } catch (e) {
      console.warn(`[ModelStats] could not persist ${this.file}: ${e.message}`);
    }
  }

  _entry(id) {
    return (this.models[id] ??= {});
  }

  loads(id) {
    return this.models[id]?.loads || [];
  }

  recordLoad(id, sec) {
    const e = this._entry(id);
    e.loads = [...(e.loads || []), sec].slice(-LOAD_HISTORY_MAX);
    this._save();
  }

  /**
   * Highest peak over the recent runs (GB), split by how it was measured:
   * `exact` (baseline taken by the composer right before it started the model)
   * and `approx` (model first seen already loading — the baseline may include
   * part of it, so it can read LOW). null where none.
   */
  peak(id) {
    const max = (xs) => (xs.length ? Math.max(...xs) : null);
    const ok = (this.models[id]?.peaks || []).filter((p) => Number.isFinite(p.gb) && p.gb > 0);
    return {
      exact: max(ok.filter((p) => !p.approx).map((p) => p.gb)),
      approx: max(ok.filter((p) => p.approx).map((p) => p.gb)),
    };
  }

  /** Open a new run's peak slot; returns its index for updatePeak. */
  beginPeakRun(id, { approx = false, at = Date.now() } = {}) {
    const e = this._entry(id);
    // A run that never registered (stopped at once) must not push a real one out.
    const kept = (e.peaks || []).filter((p) => p.gb > 0);
    e.peaks = [...kept, { gb: 0, at, ...(approx ? { approx: true } : {}) }].slice(-PEAK_RUNS_MAX);
    return e.peaks.length - 1;
  }

  /** Raise the current run's peak; persisted only when it grows (bounded writes). */
  updatePeak(id, gb) {
    const peaks = this.models[id]?.peaks;
    if (!peaks?.length) return false;
    const run = peaks[peaks.length - 1];
    const v = Math.round(gb);
    if (!(v > run.gb)) return false;
    run.gb = v;
    this._save();
    return true;
  }

  servedModel(id) {
    return this.models[id]?.servedModel || null;
  }

  setServedModel(id, name) {
    if (!name || this.models[id]?.servedModel === name) return;
    this._entry(id).servedModel = name;
    this._save();
  }
}
