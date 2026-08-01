import { useEffect, useRef, useState } from "react";
import { activateModelSetup, cancelModelSetup, stopModelSetup } from "../api/client";
import type { ModelSetup, ModelSetupsState } from "../api/types";

interface ModelSwitchDialogProps {
  open: boolean;
  onClose: () => void;
  /** Live state pushed over the WebSocket (phase, active/target, log). */
  state: ModelSetupsState | null;
}

function useEscape(onClose: () => void, active: boolean) {
  useEffect(() => {
    if (!active) return; // only listen while the dialog is open
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose, active]);
}

const isBusy = (phase?: ModelSetupsState["phase"]) => phase === "starting" || phase === "stopping";

export function ModelSwitchDialog({ open, onClose, state }: ModelSwitchDialogProps) {
  useEscape(onClose, open);
  // Setup the user picked to switch to — held until they confirm.
  const [pending, setPending] = useState<ModelSetup | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const busy = isBusy(state?.phase);
  const unknown = state?.phase === "unknown";
  const log = state?.log ?? [];

  // Drop any stale confirmation / error when the dialog closes.
  useEffect(() => {
    if (!open) {
      setPending(null);
      setActionError(null);
    }
  }, [open]);

  // Auto-scroll the live log to the newest line.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  // Clear a pending confirmation once a transition actually begins.
  useEffect(() => {
    if (busy) setPending(null);
  }, [busy]);

  if (!open) return null;

  const setups = state?.setups ?? [];
  const activeId = state?.activeSetupId ?? null;

  const doActivate = async (id: string) => {
    setActionError(null);
    try {
      await activateModelSetup(id);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
    setPending(null);
  };

  const doStop = async () => {
    setActionError(null);
    try {
      await stopModelSetup();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const doCancel = async () => {
    setActionError(null);
    try {
      await cancelModelSetup();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="panel w-full max-w-lg p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-strong">Model setup</h2>
          {activeId && !busy && (
            <button
              type="button"
              onClick={doStop}
              className="rounded border border-border bg-surface-elevated px-2.5 py-1 text-[11px] text-muted hover:bg-surface-hover"
            >
              Stop all
            </button>
          )}
        </div>

        {/* Transition banner + live log */}
        {busy && (
          <div className="mb-4 rounded border border-border bg-surface-elevated p-3">
            <div className="mb-2 flex items-center justify-between gap-2 text-xs text-text">
              <span className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warning" />
                {state?.phase === "stopping" ? "Stopping current setup…" : "Starting setup…"}
              </span>
              <button
                type="button"
                onClick={doCancel}
                className="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-muted hover:bg-surface-hover"
              >
                Cancel
              </button>
            </div>
            <pre
              ref={logRef}
              className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded bg-base/60 p-2 text-[10px] leading-relaxed text-muted"
            >
              {log.length ? log.join("\n") : "…"}
            </pre>
          </div>
        )}

        {/* Failure banner */}
        {state?.phase === "failed" && state?.error && (
          <div className="mb-4 rounded bg-danger/20 px-3 py-2 text-xs text-danger">
            Switch failed: {state.error}
          </div>
        )}
        {state?.phase === "unknown" && (
          <div className="mb-4 rounded bg-danger/15 px-3 py-2 text-xs text-danger">
            Cannot detect running model{state.error ? ` (${state.error})` : ""}. Switching is disabled.
          </div>
        )}
        {actionError && (
          <div className="mb-4 rounded bg-danger/20 px-3 py-2 text-xs text-danger">{actionError}</div>
        )}

        {/* Setup cards */}
        <div className="space-y-2">
          {setups.map((s) => {
            const isActive = s.id === activeId;
            const isTarget = s.id === state?.targetSetupId;
            const confirming = pending?.id === s.id;
            return (
              <div
                key={s.id}
                className={`rounded border p-3 ${
                  isActive ? "border-accent bg-accent-soft/40" : "border-border bg-surface-elevated"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-xs font-semibold text-text-strong">{s.name}</span>
                      {isActive && state?.phase === "running" && (
                        <span className="rounded-full bg-success/20 px-1.5 py-0.5 text-[9px] font-medium text-success">
                          RUNNING
                        </span>
                      )}
                      {isTarget && busy && (
                        <span className="rounded-full bg-warning/20 px-1.5 py-0.5 text-[9px] font-medium text-warning">
                          STARTING
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 truncate text-[10px] text-muted">{s.description}</p>
                  </div>
                  {!isActive && (
                    <button
                      type="button"
                      disabled={busy || unknown}
                      onClick={() => setPending(s)}
                      className="shrink-0 rounded bg-accent px-2.5 py-1 text-[11px] font-medium text-white hover:bg-accent-hover disabled:opacity-40"
                    >
                      Switch
                    </button>
                  )}
                </div>

                {/* Inline confirmation */}
                {confirming && (
                  <div className="mt-3 rounded border border-border bg-base/40 p-2.5">
                    <p className="text-[11px] text-muted">
                      {activeId ? (
                        <>
                          Stop <span className="text-text">{state?.setups.find((x) => x.id === activeId)?.name}</span> and
                          start <span className="text-text">{s.name}</span>? First boot can take several minutes.
                        </>
                      ) : (
                        <>
                          Start <span className="text-text">{s.name}</span>? First boot can take several minutes.
                        </>
                      )}
                    </p>
                    <div className="mt-2 flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => setPending(null)}
                        className="rounded border border-border bg-surface-elevated px-2.5 py-1 text-[11px] text-muted hover:bg-surface-hover"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => doActivate(s.id)}
                        className="rounded bg-accent px-2.5 py-1 text-[11px] font-medium text-white hover:bg-accent-hover"
                      >
                        Confirm switch
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          {setups.length === 0 && (
            <p className="text-xs text-muted">No model setups configured.</p>
          )}
        </div>

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-border bg-surface-elevated px-3 py-1.5 text-xs text-muted hover:bg-surface-hover"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
