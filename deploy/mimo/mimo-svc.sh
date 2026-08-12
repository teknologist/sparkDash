#!/usr/bin/env bash
# mimo-svc.sh — manage the MiMo-V2.5 Omni dual-DGX-Spark deployment.
#
# Thin wrapper over the recipe's start.sh / stop.sh that first sources
# cluster.env, so our node IPs, SSH user, repo paths and RoCE (f1) devices are
# always applied and the recipe's zurih/10.0.0.x/f0 defaults never leak in.
# Run from spark1 (the head).
#
#   ./mimo-svc.sh check      # dry-run: plan + weight completeness, no changes
#   ./mimo-svc.sh start      # full two-node bring-up + chat verify (long)
#   ./mimo-svc.sh stop       # stop vLLM + Ray + containers (both nodes)
#   ./mimo-svc.sh teardown   # heavier stop (start.sh --teardown)
#   ./mimo-svc.sh status     # container state (both nodes) + API health
#   ./mimo-svc.sh logs       # tail the head vLLM log
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$DIR/cluster.env" ] || { echo "missing $DIR/cluster.env" >&2; exit 1; }
set -a; # shellcheck disable=SC1091
source "$DIR/cluster.env"; set +a

API="http://127.0.0.1:${VLLM_PORT:-8000}/v1"
CN="${CONTAINER_NAME:-mimo-nvfp4}"

case "${1:-}" in
  start)    exec bash "$DIR/start.sh" ;;
  check)    exec bash "$DIR/start.sh" --check ;;
  stop)     exec bash "$DIR/stop.sh" ;;
  teardown) exec bash "$DIR/start.sh" --teardown ;;
  status)
    echo "== head (spark1) container =="
    docker ps -a --filter "name=${CN}" --format '{{.Names}}\t{{.Status}}\t{{.Image}}' || true
    echo "== worker (spark2) container =="
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER}@${WORKER_IP}" \
      "docker ps -a --filter name=${CN} --format '{{.Names}}\t{{.Status}}\t{{.Image}}'" 2>/dev/null || echo "(worker unreachable)"
    echo "== API ${API} =="
    if curl -fsS --max-time 5 "${API}/models" 2>/dev/null; then echo; else echo "(API down)"; fi
    ;;
  logs)
    tail -n "${LINES:-80}" "${VLLM_LOG:-$DIR/vllm.log}" 2>/dev/null || echo "(no vllm.log yet)"
    ;;
  *) echo "usage: $0 {check|start|stop|teardown|status|logs}"; exit 1 ;;
esac
