#!/usr/bin/env bash
#
# jeff.sh — control the jeff System One API (GLiFormer-large-v1) on either node.
#
#   uv-run FastAPI process (no container) serving TypeSafe's jev System One wire
#   format -> http://<host>:8006/v1/systemone  (model: gliformer-large-v1)
#
# jeff is NOT a chat model: it answers classification questions (choice / score /
# noul) and has no /v1/chat/completions. It is therefore a direct-access brick —
# `"gateway": false` in models.json keeps it out of the llama-swap config, and
# clients point TYPESAFE_BASE_URL straight at this port. Its /v1/models needs a
# bearer key, so readiness is probed on /healthz (open) instead.
#
# Tiny and fixed-footprint (~2.7 GiB measured, bf16 on GB10), so it co-resides
# with anything that leaves 3 GB free.
#
# Usage: jeff.sh {start|stop|restart|status|logs}
set -euo pipefail

JEFF_DIR="${JEFF_DIR:-$HOME/jeff}"
UV="${UV:-$HOME/.local/bin/uv}"
PORT="${JEFF_PORT:-8006}"          # unique backend port (composer passes JEFF_PORT)
HOST="${JEFF_HOST:-0.0.0.0}"
CHECK_HOST="${JEFF_CHECK_HOST:-127.0.0.1}"
DEVICE="${JEFF_DEVICE:-cuda}"
LOG="${JEFF_LOG:-$HOME/jeff-server.log}"
PIDFILE="${JEFF_PIDFILE:-$HOME/.jeff.pid}"
READY_TIMEOUT="${JEFF_READY_TIMEOUT:-180}"

# Bearer keys live outside the repo so they never land in git or models.json.
KEYFILE="${JEFF_KEYFILE:-$HOME/.jeff-api-keys}"
if [ -z "${JEFF_API_KEYS:-}" ] && [ -r "$KEYFILE" ]; then
  JEFF_API_KEYS="$(tr -d '[:space:]' < "$KEYFILE")"
fi
export JEFF_API_KEYS="${JEFF_API_KEYS:-}"

_pid()     { [ -f "$PIDFILE" ] && cat "$PIDFILE" 2>/dev/null || true; }
_running() { local p; p="$(_pid)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

start() {
  [ -x "$UV" ] || { echo "jeff: uv missing: $UV" >&2; return 1; }
  [ -d "$JEFF_DIR/models/gliformer-large-v1" ] || {
    echo "jeff: weights missing: $JEFF_DIR/models/gliformer-large-v1" >&2; return 1; }
  [ -n "$JEFF_API_KEYS" ] || {
    echo "jeff: no API keys — set JEFF_API_KEYS or write $KEYFILE (refusing to serve open)" >&2; return 1; }
  if _running; then
    echo "jeff already running (pid $(_pid))"
  else
    echo "Launching jeff (device=$DEVICE) -> http://$HOST:$PORT/v1/systemone"
    # JEFF_WARMUP=1 pays a ~1.5 s startup cost so the first real request does not
    # eat cuBLAS autotuning; the whole load is only ~7 s on GB10.
    #
    # The pid is written by the child itself ($$ before exec), NOT by `echo $!`:
    # in a non-interactive shell the backgrounded `setsid` is not a process-group
    # leader, so it FORKS and the pid `$!` reports exits immediately — the
    # pidfile would point at a corpse while the server ran on. Writing $$ then
    # exec'ing also makes that pid the new session/group leader, and `uv run`
    # spawns python as its child, so stop can signal the whole group with -PGID
    # and leave nothing behind.
    # JEFF_API_KEYS is INHERITED (exported above), never passed as an argument —
    # argv is world-readable through /proc, so a key on the command line would be
    # visible in `ps` to every user on the box.
    setsid bash -c '
      echo $$ > "$3"
      cd "$4" || exit 1
      exec env JEFF_DEVICE="$5" JEFF_HOST="$6" JEFF_PORT="$7" JEFF_WARMUP=1 \
               "$1" run jeff >"$2" 2>&1
    ' _ "$UV" "$LOG" "$PIDFILE" "$JEFF_DIR" "$DEVICE" "$HOST" "$PORT" \
      < /dev/null > /dev/null 2>&1 &
    # settle: the child writes the pidfile before exec
    for _ in $(seq 1 20); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
  fi
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  until curl -fsS --max-time 4 "http://$CHECK_HOST:$PORT/healthz" >/dev/null 2>&1; do
    if ! _running; then echo "jeff exited during startup — last log:" >&2; tail -n 30 "$LOG" >&2 || true; return 1; fi
    [ "$(date +%s)" -ge "$deadline" ] && { echo "jeff not ready within ${READY_TIMEOUT}s (see $LOG)" >&2; return 1; }
    sleep 2
  done
  echo "jeff API up: http://$HOST:$PORT/v1/systemone  (gliformer-large-v1)"
}

stop() {
  local p; p="$(_pid)"
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
    # Negative pid = the whole process group (uv + the python server it spawns).
    echo "Stopping jeff (pgid $p)…"; kill -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$p" 2>/dev/null && { echo "  SIGKILL"; kill -9 -- "-$p" 2>/dev/null || true; }
  else
    pkill -f '\.venv/bin/jeff' 2>/dev/null && echo "Killed stray jeff" || echo "jeff not running"
  fi
  rm -f "$PIDFILE"
}

status() {
  echo "== $(hostname): jeff System One API (GLiFormer-large-v1) =="
  _running && echo "  process: UP (pid $(_pid))" || echo "  process: down"
  if curl -fsS --max-time 4 "http://$CHECK_HOST:$PORT/healthz" >/dev/null 2>&1; then
    echo "  API: UP ($HOST:$PORT)  served: gliformer-large-v1"
  else
    echo "  API: down"
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; echo; start ;;
  status)  status ;;
  logs)    tail -f -n 100 "$LOG" ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 1 ;;
esac
