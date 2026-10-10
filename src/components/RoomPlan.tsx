import type { RoomPlan } from "../api/types";

/** "GLM-5.3 (spark1 + spark2), Ornith" */
export function roomVictimsText(plan: RoomPlan): string {
  return plan.victims.map((v) => (v.nodes.length > 1 ? `${v.name} (${v.nodes.join(" + ")})` : v.name)).join(", ");
}

/** The question asked before anything is stopped to make room. */
export function roomPlanQuestion(plan: RoomPlan): string {
  const why = plan.firstRun
    ? "its memory use is not measured yet, so the first start clears the node"
    : `it needs ${plan.needGB} GB and ${plan.freeGB} GB is free`;
  return `Starting ${plan.targetName} will stop ${roomVictimsText(plan)} — ${why}.`;
}
