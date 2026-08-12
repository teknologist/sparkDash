#!/usr/bin/env bash
# laguna-ctl.sh — start/stop/status wrapper for Laguna-S on THIS node (spark2).
# Lets sparkDash's ssh model-setup component drive laguna with a single cmd +
# per-action arg. start delegates to serve.sh (honors ENABLE_SPEC/UTIL/etc env).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CID="${NAME:-laguna-s}"
PORT="${LAGUNA_PORT:-8000}"
case "${1:-}" in
  start)  exec "$DIR/serve.sh" ;;
  stop)   docker stop "$CID" >/dev/null 2>&1 && echo "stopped $CID" || echo "$CID not running" ;;
  status)
    if docker ps --format '{{.Names}}' | grep -qx "$CID"; then echo "  container: UP ($CID)"; else echo "  container: down"; fi
    if curl -fsS --max-time 4 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "  API: UP  (http://127.0.0.1:${PORT}/v1)  served: laguna-s"
    else echo "  API: down"; fi ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 1 ;;
esac
