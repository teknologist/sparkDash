import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useModalPresence } from "../../hooks/useModalPresence";
import { MetricBar, bandColor } from "../ui/MetricBar";
import { BotIcon, MemoryIcon, PlusIcon } from "../ui/icons";
import {
  applyComposition,
  cancelComposer,
  saveComposerPreset,
  verifyComposition,
} from "../../api/client";
import type { Assignment, ComposerBrick, ComposerCatalog, ComposerState } from "../../api/types";

interface Props {
  open: boolean;
  onClose: () => void;
  state: ComposerState | null;
}

// ─── Client-side validation (mirrors server validateAssignment) ────────────
interface NodeVerdict {
  ramUsed: number;
  budget: number;
  over: boolean;
}
function validateClient(catalog: ComposerCatalog, assignment: Assignment) {
  const byId = new Map(catalog.models.map((m) => [m.id, m]));
  const nodes = Object.keys(catalog.nodeCapacityGB);
  const errors: string[] = [];
  const perNode: Record<string, NodeVerdict> = {};

  const dualIds = new Set<string>();
  for (const n of nodes)
    for (const id of assignment[n] || []) if (byId.get(id)?.placement === "dual") dualIds.add(id);
  for (const id of dualIds) {
    const m = byId.get(id)!;
    for (const n of m.nodes) {
      const on = assignment[n] || [];
      if (!on.includes(id)) errors.push(`${m.displayName} (dual) must occupy ${m.nodes.join(" + ")}`);
      if (on.filter((x) => x !== id).length) errors.push(`${m.displayName} (dual) needs ${n} to itself`);
    }
  }

  for (const n of nodes) {
    const ids = assignment[n] || [];
    let ram = 0;
    const ports = new Map<number, string>();
    for (const id of ids) {
      const m = byId.get(id);
      if (!m) continue;
      if (!m.nodes.includes(n)) errors.push(`${m.displayName} can't run on ${n}`);
      ram += m.ramGB || 0;
      if (m.port != null) {
        if (ports.has(m.port)) errors.push(`Port ${m.port} conflict on ${n}`);
        else ports.set(m.port, id);
      }
    }
    const budget = (catalog.nodeCapacityGB[n] ?? 128) - catalog.reserveGB;
    const over = ram > budget;
    if (over) errors.push(`${n} over budget: ${ram} > ${budget} GB`);
    perNode[n] = { ramUsed: ram, budget, over };
  }
  return { ok: errors.length === 0, perNode, errors };
}

const backendBadge: Record<string, string> = {
  ds4: "bg-accent-soft/60 text-accent",
  vllm: "bg-success/20 text-success",
  sglang: "bg-warning/20 text-warning",
};

function BrickTile({
  brick,
  onDragStart,
  compact,
  onRemove,
}: {
  brick: ComposerBrick;
  onDragStart?: () => void;
  compact?: boolean;
  onRemove?: () => void;
}) {
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData("text/plain", brick.id);
        e.dataTransfer.effectAllowed = "move";
        onDragStart?.();
      }}
      className={`group flex cursor-grab items-center gap-2 rounded-md border border-border bg-surface-elevated px-2.5 py-1.5 text-xs transition-colors hover:border-accent/60 active:cursor-grabbing ${
        compact ? "" : "w-full"
      }`}
      title={brick.notes || undefined}
    >
      <BotIcon className="h-3.5 w-3.5 shrink-0 text-accent" />
      <span className="truncate font-medium text-text">{brick.displayName}</span>
      {brick.backend && (
        <span
          className={`shrink-0 rounded px-1 py-0.5 text-[9px] font-semibold uppercase ${
            backendBadge[brick.backend] || "bg-border text-muted"
          }`}
        >
          {brick.backend}
        </span>
      )}
      {brick.placement === "dual" && (
        <span className="shrink-0 rounded bg-warning/20 px-1 py-0.5 text-[9px] font-semibold uppercase text-warning">
          dual
        </span>
      )}
      <span className="ml-auto shrink-0 font-tabular text-[11px] text-muted">{brick.ramGB} GB</span>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          className="shrink-0 rounded px-1 text-muted opacity-60 hover:text-danger hover:opacity-100"
          aria-label={`Remove ${brick.displayName}`}
        >
          ×
        </button>
      )}
    </div>
  );
}

export function ModelComposerDialog({ open, onClose, state }: Props) {
  const { mounted, visible } = useModalPresence(open);
  const [assignment, setAssignment] = useState<Assignment>({});
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [serverErrors, setServerErrors] = useState<string[] | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmApply, setConfirmApply] = useState(false);
  const [presetName, setPresetName] = useState("");
  const logRef = useRef<HTMLPreElement>(null);

  const catalog = state?.catalog ?? null;
  const nodes = catalog ? Object.keys(catalog.nodeCapacityGB) : [];
  const applying = state?.applying ?? false;
  const phase = state?.phase ?? "idle";
  const log = state?.log ?? [];

  // Seed the board from the live running layout when opened.
  useEffect(() => {
    if (!open || !state) return;
    const seed: Assignment = {};
    for (const [node, pn] of Object.entries(state.perNode)) seed[node] = pn.running.map((r) => r.id);
    setAssignment(seed);
    setServerErrors(null);
    setActionError(null);
    setConfirmApply(false);
  }, [open, state?.currentAssignment]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!mounted) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [mounted]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log.length]);

  const byId = useMemo(
    () => new Map((catalog?.models ?? []).map((m) => [m.id, m])),
    [catalog]
  );
  const placed = useMemo(() => new Set(Object.values(assignment).flat()), [assignment]);
  const verdict = useMemo(
    () => (catalog ? validateClient(catalog, assignment) : { ok: false, perNode: {}, errors: [] }),
    [catalog, assignment]
  );

  if (!mounted || !catalog) return null;

  const placeBrick = (id: string, node: string) => {
    const m = byId.get(id);
    if (!m) return;
    setServerErrors(null);
    setConfirmApply(false);
    setAssignment((prev) => {
      if (m.placement === "dual") {
        // Dual claims all its nodes exclusively.
        const next: Assignment = {};
        for (const n of nodes) next[n] = m.nodes.includes(n) ? [id] : [];
        return next;
      }
      const next: Assignment = {};
      for (const n of nodes) next[n] = (prev[n] || []).filter((x) => x !== id);
      // Dropping a single brick clears any dual occupant on this node.
      next[node] = next[node].filter((x) => byId.get(x)?.placement !== "dual");
      next[node] = [...next[node], id];
      return next;
    });
  };

  const removeBrick = (id: string) => {
    setServerErrors(null);
    setConfirmApply(false);
    setAssignment((prev) => {
      const next: Assignment = {};
      for (const n of nodes) next[n] = (prev[n] || []).filter((x) => x !== id);
      return next;
    });
  };

  const loadPreset = (a: Assignment) => {
    const next: Assignment = {};
    for (const n of nodes) next[n] = [...(a[n] || [])];
    setAssignment(next);
    setServerErrors(null);
    setConfirmApply(false);
  };

  const doVerify = async () => {
    setActionError(null);
    try {
      const r = await verifyComposition(assignment);
      setServerErrors(r.ok ? [] : r.errors);
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const doApply = async () => {
    setActionError(null);
    try {
      await applyComposition(assignment);
      setConfirmApply(false);
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const doSavePreset = async () => {
    if (!presetName.trim()) return;
    setActionError(null);
    try {
      await saveComposerPreset(presetName.trim(), assignment);
      setPresetName("");
    } catch (e) {
      setActionError((e as Error).message);
    }
  };

  const palette = catalog.models.filter((m) => !placed.has(m.id));

  return createPortal(
    <div
      className={`modal-overlay${visible ? " is-open" : ""}`}
      onClick={(e) => {
        if (applying) return;
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="modal-sheet w-full"
        style={{ maxWidth: "56rem", maxHeight: "min(92vh, 52rem)" }}
        role="dialog"
        aria-modal="true"
      >
        <div className="modal-sheet__header flex items-center gap-2">
          <MemoryIcon className="h-4 w-4 shrink-0 text-accent" />
          <span>Compose Models</span>
          <span className="ml-2 text-[11px] font-normal text-muted">
            Drag bricks onto a node · RAM budget updates live
          </span>
        </div>

        <div className="modal-sheet__body space-y-4">
          {/* Apply progress / failure banner */}
          {(applying || phase === "failed") && (
            <div className="rounded border border-border bg-surface-elevated p-3">
              <div className="mb-2 flex items-center gap-2 text-xs">
                {applying ? (
                  <>
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warning" />
                    <span className="text-text">Applying configuration…</span>
                    <button
                      type="button"
                      onClick={() => void cancelComposer()}
                      className="ml-auto rounded border border-border px-2 py-0.5 text-[11px] text-muted hover:bg-surface-hover"
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <span className="text-danger">Apply failed{state?.error ? `: ${state.error}` : ""}</span>
                )}
              </div>
              {log.length > 0 && (
                <pre
                  ref={logRef}
                  className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded bg-base/60 p-2 text-[10px] leading-relaxed text-muted"
                >
                  {log.join("\n")}
                </pre>
              )}
            </div>
          )}

          {/* Presets */}
          {(state?.presets?.length ?? 0) > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] uppercase tracking-wide text-muted">Presets</span>
              {state!.presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => loadPreset(p.assignment)}
                  className="rounded-full border border-border bg-surface-elevated px-2.5 py-0.5 text-[11px] text-text hover:border-accent/60"
                >
                  {p.name}
                </button>
              ))}
            </div>
          )}

          {/* Node columns */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {nodes.map((node) => {
              const v = verdict.perNode[node] ?? { ramUsed: 0, budget: 0, over: false };
              const ids = assignment[node] || [];
              return (
                <div
                  key={node}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(node);
                  }}
                  onDragLeave={() => setDragOver((d) => (d === node ? null : d))}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(null);
                    const id = e.dataTransfer.getData("text/plain");
                    if (id) placeBrick(id, node);
                  }}
                  className={`rounded-lg border p-3 transition-colors ${
                    dragOver === node ? "border-accent bg-accent-soft/20" : "border-border bg-base/30"
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-text-strong">{node}</span>
                    <span className="font-tabular text-[11px] text-muted">
                      {v.ramUsed} / {v.budget} GB
                    </span>
                  </div>
                  <MetricBar
                    label="RAM"
                    value={v.ramUsed}
                    max={v.budget}
                    color={v.over ? "bg-danger" : bandColor((v.ramUsed / (v.budget || 1)) * 100)}
                    caption={`${Math.round((v.ramUsed / (v.budget || 1)) * 100)}%`}
                  />
                  <div className="mt-3 min-h-[64px] space-y-1.5">
                    {ids.length === 0 && (
                      <div className="flex h-16 items-center justify-center rounded border border-dashed border-border text-[11px] text-muted">
                        drop a model here
                      </div>
                    )}
                    {ids.map((id) => {
                      const m = byId.get(id);
                      if (!m) return null;
                      return (
                        <BrickTile key={id} brick={m} onRemove={() => removeBrick(id)} />
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Palette */}
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted">
              <PlusIcon className="h-3 w-3" /> Available models
            </div>
            {palette.length === 0 ? (
              <div className="text-[11px] text-muted">All models placed.</div>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {palette.map((m) => (
                  <div key={m.id} className="w-[calc(50%-0.375rem)]">
                    <BrickTile brick={m} />
                    <div className="mt-0.5 pl-1 text-[9px] text-muted">{m.nodes.join(" / ")}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Validation errors */}
          {(!verdict.ok || serverErrors) && (
            <div className="space-y-1">
              {[...new Set([...(verdict.errors || []), ...(serverErrors || [])])].map((err, i) => (
                <div
                  key={i}
                  className="rounded border border-danger/30 bg-danger/10 px-2.5 py-1 text-[11px] text-danger"
                >
                  {err}
                </div>
              ))}
              {serverErrors && serverErrors.length === 0 && (
                <div className="rounded border border-success/30 bg-success/10 px-2.5 py-1 text-[11px] text-success">
                  Verified — this configuration is runnable.
                </div>
              )}
            </div>
          )}
          {actionError && (
            <div className="rounded border border-danger/40 bg-danger/20 px-2.5 py-1 text-[11px] text-danger">
              {actionError}
            </div>
          )}
        </div>

        <div className="modal-sheet__footer">
          <div className="flex w-full flex-wrap items-center gap-2">
            <input
              type="text"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              placeholder="preset name"
              className="w-28 rounded border border-border bg-surface-elevated px-2 py-1 text-[11px] text-text outline-none focus:border-accent"
            />
            <button
              type="button"
              onClick={() => void doSavePreset()}
              disabled={!presetName.trim() || !verdict.ok}
              className="rounded border border-border bg-surface-elevated px-2.5 py-1.5 text-[11px] text-muted hover:bg-surface-hover disabled:opacity-40"
            >
              Save preset
            </button>
            <div className="ml-auto flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="rounded border border-border bg-surface-elevated px-3 py-1.5 text-xs text-muted hover:bg-surface-hover"
              >
                Close
              </button>
              <button
                type="button"
                onClick={() => void doVerify()}
                disabled={applying}
                className="rounded border border-border bg-surface-elevated px-3 py-1.5 text-xs text-text hover:bg-surface-hover disabled:opacity-40"
              >
                Verify
              </button>
              {confirmApply ? (
                <button
                  type="button"
                  onClick={() => void doApply()}
                  disabled={applying || !verdict.ok}
                  className="rounded border border-accent/50 bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40"
                >
                  Confirm apply
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirmApply(true)}
                  disabled={applying || !verdict.ok}
                  className="rounded border border-accent/50 bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40"
                >
                  Apply
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
