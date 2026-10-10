import type { LoadProgress as LoadProgressData } from "../api/types";

/** "1m 05s" / "45s" */
export function fmtDuration(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

/** One-line summary: phase · % · ETA (or elapsed while no load has been measured yet). */
export function loadProgressText(p: LoadProgressData | undefined): string {
  if (!p) return "loading";
  const parts = [p.phase ?? "loading"];
  if (p.phasePct != null) parts[0] += ` ${p.phasePct}%`;
  if (p.pct != null) parts.push(`${p.pct}%`);
  if (p.etaSec != null) parts.push(`~${fmtDuration(p.etaSec)} left`);
  else if (p.expectedSec != null) parts.push(`${fmtDuration(p.elapsedSec)} · slower than usual`);
  else parts.push(`${fmtDuration(p.elapsedSec)} elapsed`);
  return parts.join(" · ");
}

export function loadProgressTitle(p: LoadProgressData | undefined): string {
  if (!p) return "Container up, API not answering yet";
  const typical = p.expectedSec != null ? `typical load ${fmtDuration(p.expectedSec)}` : "no load measured yet — ETA appears after the first one";
  return `Loading for ${fmtDuration(p.elapsedSec)} · ${typical}`;
}

/**
 * Thin bar for a loading model. Determinate once a load time has been measured,
 * otherwise an indeterminate pulse (we know it is loading, not how far along).
 */
export function LoadProgressBar({ progress }: { progress?: LoadProgressData }) {
  const pct = progress?.pct ?? null;
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-warning/15" title={loadProgressTitle(progress)}>
      <div
        className={`h-full rounded-full bg-warning ${pct == null ? "w-1/3 animate-pulse" : "transition-[width] duration-1000"}`}
        style={pct == null ? undefined : { width: `${pct}%` }}
      />
    </div>
  );
}
