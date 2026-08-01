import type { ModelSetupsState } from "../api/types";
import { BotIcon } from "./ui/icons";

interface ModelSetupPillProps {
  state: ModelSetupsState | null;
  onClick?: () => void;
}

/** Dot colour per phase — green running, amber transitioning, red failed/unknown, grey idle. */
function phaseDotClass(phase: ModelSetupsState["phase"] | undefined): string {
  switch (phase) {
    case "running":
      return "bg-success";
    case "starting":
    case "stopping":
      return "bg-warning animate-pulse";
    case "failed":
    case "unknown":
      return "bg-danger";
    default:
      return "bg-muted/50";
  }
}

/** Short phase word shown while a switch is in flight. */
function transitionLabel(state: ModelSetupsState): string | null {
  const target = state.setups.find((s) => s.id === state.targetSetupId);
  if (state.phase === "stopping") return "Stopping…";
  if (state.phase === "starting") return target ? `Starting ${target.name}…` : "Starting…";
  return null;
}

/**
 * ModelSetupPill — header button showing the active model setup + status.
 * Clicking opens the switch dialog.
 */
export function ModelSetupPill({ state, onClick }: ModelSetupPillProps) {
  const active = state?.setups.find((s) => s.id === state.activeSetupId) || null;
  const unknown = state?.phase === "unknown";
  const transition = state ? transitionLabel(state) : null;
  const label = transition
    ? transition
    : unknown
      ? "Detection unavailable"
      : active
        ? active.name
        : "No model running";
  const title = unknown
    ? state?.error || "Cannot detect running model (docker unreachable)"
    : active?.description || "Choose a model setup to run";

  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-elevated px-3 py-1.5 text-xs text-muted transition-colors hover:bg-surface-hover"
      title={title}
      aria-label="Model setup"
    >
      <BotIcon className="h-3.5 w-3.5 text-accent" />
      <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${phaseDotClass(state?.phase)}`} />
      <span className="max-w-[14rem] truncate text-text">{label}</span>
    </button>
  );
}
