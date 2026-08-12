/**
 * Dynamic per-node memory allocation for the composer.
 *
 * A vLLM model's footprint is ELASTIC: `--gpu-memory-utilization` reserves
 * `util * capacity` GB (weights + KV cache). So co-resident vLLM models can
 * share a node by shrinking their util — as long as their WEIGHTS fit. Models
 * without a `maxUtil` (ds4 in-RAM mmap, dual TP=2) have a FIXED footprint.
 *
 * allocateNode() gives fixed models their `weightGB`, then water-fills the
 * remaining budget across the elastic models — floor = weightGB (weights must
 * fit), cap = `maxUtil * capacity`, surplus distributed proportional to weight.
 * It returns each elastic model's computed util (to pass the launcher as
 * GPU_UTIL) and every model's resulting footprint.
 *
 * Brick fields consumed: `weightGB` (fixed floor, must-fit), `maxUtil` (natural
 * GPU fraction when alone; presence marks a model as elastic). Falls back to
 * `ramGB` when `weightGB` is absent so un-migrated bricks still work.
 */
export function allocateNode(catalog, nodeId, ids) {
  const cap = catalog.nodeCapacityGB(nodeId);
  const budget = catalog.nodeBudgetGB(nodeId);
  const models = (ids || []).map((id) => catalog.getModel(id)).filter(Boolean);

  const weightOf = (m) => Number(m.weightGB != null ? m.weightGB : m.ramGB) || 0;
  const isElastic = (m) => Number(m.maxUtil) > 0;

  const fixed = models.filter((m) => !isElastic(m));
  const elastic = models.filter((m) => isElastic(m));
  const fixedGB = fixed.reduce((s, m) => s + weightOf(m), 0);
  const weightSum = elastic.reduce((s, m) => s + weightOf(m), 0);

  const perModel = {};
  for (const m of fixed) perModel[m.id] = { footprintGB: Math.round(weightOf(m)), util: null };

  // Weights must fit regardless of how small we make the KV cache.
  if (fixedGB + weightSum > budget) {
    for (const m of elastic) perModel[m.id] = { footprintGB: Math.round(weightOf(m)), util: null };
    return {
      ok: false,
      ramUsed: Math.round(fixedGB + weightSum),
      budget,
      cap,
      perModel,
      error: `weights don't fit: ${Math.round(fixedGB + weightSum)} GB of weights > ${budget} GB budget`,
    };
  }

  // Water-fill the space left after fixed models across the elastic ones.
  const avail = budget - fixedGB;
  const foot = {};
  const capGB = {};
  for (const m of elastic) {
    foot[m.id] = weightOf(m); // floor: its weights
    capGB[m.id] = Math.min(Number(m.maxUtil) * cap, avail); // its natural ceiling
  }
  let remaining = avail - weightSum;
  for (let iter = 0; iter < 200 && remaining > 0.5; iter++) {
    const uncapped = elastic.filter((m) => foot[m.id] < capGB[m.id] - 1e-6);
    if (!uncapped.length) break;
    const totalW = uncapped.reduce((s, m) => s + weightOf(m), 0) || uncapped.length;
    let distributed = 0;
    for (const m of uncapped) {
      const add = Math.min(remaining * (weightOf(m) / totalW), capGB[m.id] - foot[m.id]);
      foot[m.id] += add;
      distributed += add;
    }
    remaining -= distributed;
    if (distributed < 0.5) break;
  }
  for (const m of elastic) {
    perModel[m.id] = {
      footprintGB: Math.round(foot[m.id]),
      util: Math.max(0.05, Math.min(0.95, foot[m.id] / cap)),
    };
  }

  const ramUsed = Object.values(perModel).reduce((s, x) => s + x.footprintGB, 0);
  // No slop: the old `budget + 1` let 117/116 verify as OK, and a ~1 GiB
  // shortfall is exactly how a co-residence OOMs at engine init.
  return { ok: ramUsed <= budget, ramUsed, budget, cap, perModel };
}
