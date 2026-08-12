#!/usr/bin/env bash
#
# ds4-entrpi.sh — control the Entrpi single-node ds4 stack on spark1.
#
#   spark1: ds4-server (Entrpi/ds4 fork, CUDA sm_121) serving the
#   DeepSeek-V4-Flash-0731 IQ2XXS/Q2K GGUF (~81 GiB, in-RAM) + DSpark drafter
#   -> http://<host>:8000/v1  (served: deepseek-v4-flash)
#
# Uses the `ds4-serve` launcher installed by ds4-on-spark (Entrpi/ds4 v0.5.x
# built at ~/code/ds4). Own pidfile/log so it never collides with ds4-solo.sh
# (the antirez build) or the vLLM 0731 dual stack. Single-node, in-RAM.
#
# Usage: ds4-entrpi.sh {start|stop|restart|status|logs}
set -euo pipefail

DS4_SERVE="${DS4_SERVE:-$HOME/.local/bin/ds4-serve}"
export DS4_SRC_DIR="${DS4_SRC_DIR:-$HOME/code/ds4}"     # Entrpi fork build
export DS4_GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"
CTX="${DS4_CTX:-200000}"   # 200k: empirical single-node max (~8 GiB headroom on 128 GB)
HOST="${DS4_HOST:-0.0.0.0}"
CHECK_HOST="${DS4_CHECK_HOST:-127.0.0.1}"
PORT="${DS4_PORT:-8000}"
LOG="${DS4_LOG:-$HOME/ds4-entrpi.log}"
PIDFILE="${DS4_PIDFILE:-$HOME/.ds4-entrpi.pid}"
READY_TIMEOUT="${DS4_READY_TIMEOUT:-600}"

_pid()     { [ -f "$PIDFILE" ] && cat "$PIDFILE" 2>/dev/null || true; }
_running() { local p; p="$(_pid)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

start() {
  [ -x "$DS4_SERVE" ] || { echo "ds4-entrpi: launcher missing: $DS4_SERVE (run ds4-on-spark install.sh)" >&2; return 1; }
  [ -x "$DS4_SRC_DIR/ds4-server" ] || { echo "ds4-entrpi: $DS4_SRC_DIR/ds4-server not built" >&2; return 1; }
  if _running; then
    echo "ds4-entrpi already running (pid $(_pid))"
  else
    echo "Launching Entrpi ds4-serve (ctx=$CTX) -> http://$HOST:$PORT/v1"
    # Thinking is controlled per-request (ds4-server: send think:false), like the
    # vLLM side's default-chat-template-kwargs {thinking:false}. Server default
    # is thinking-on; agent/coding clients pass think:false for fast tool loops.
    ( setsid "$DS4_SERVE" -c "$CTX" --host "$HOST" --port "$PORT" \
        >"$LOG" 2>&1 < /dev/null & echo $! > "$PIDFILE" )
  fi
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  until curl -fsS --max-time 4 "http://$CHECK_HOST:$PORT/v1/models" >/dev/null 2>&1; do
    if ! _running; then echo "ds4-entrpi exited during startup — last log:" >&2; tail -n 30 "$LOG" >&2 || true; return 1; fi
    [ "$(date +%s)" -ge "$deadline" ] && { echo "ds4-entrpi not ready within ${READY_TIMEOUT}s (see $LOG)" >&2; return 1; }
    sleep 3
  done
  echo "ds4-entrpi API up: http://$HOST:$PORT/v1  (deepseek-v4-flash)"
}

stop() {
  local p; p="$(_pid)"
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
    echo "Stopping ds4-entrpi (pid $p)…"; kill "$p" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$p" 2>/dev/null && { echo "  SIGKILL"; kill -9 "$p" 2>/dev/null || true; }
  else
    pkill -f 'ds4-server --cuda' 2>/dev/null && echo "Killed stray ds4-server" || echo "ds4-entrpi not running"
  fi
  rm -f "$PIDFILE"
}

status() {
  echo "== spark1: DeepSeek V4 Flash 0731 (Entrpi ds4-server) =="
  _running && echo "  process: UP (pid $(_pid))" || echo "  process: down"
  if curl -fsS --max-time 4 "http://$CHECK_HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  API: UP ($HOST:$PORT)  served: deepseek-v4-flash"
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
