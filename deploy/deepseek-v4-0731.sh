#!/usr/bin/env bash
#
# deepseek-v4-0731.sh — control the DeepSeek-V4-Flash-*0731* DSpark 2-node vLLM
# service (TP=2 over RoCE across spark1 head + spark2 worker).
#
# 0731 sibling of deepseek-v4.sh: same ~/deploy/dspark deployment scripts and
# Anemll GX10 image, but pinned to the 0731 model via .env.dspark.0731 and a
# distinct docker compose project so it never collides with the prior-version
# stack. Serves on :8000 as `deepseek-v4-flash-0731` for the sparkDash composer
# (llama-swap proxies 8000 -> this backend). Run from spark1 (the head node).
#
# Usage: ./deepseek-v4-0731.sh {start|stop|restart|status|logs}
#
#   start    worker-first, then head; waits for the API + a smoke chat
#            (first boot loads NVFP4 weights + cudagraph capture; several min)
#   stop     tear down head and worker (docker compose down on both nodes)
#   restart  stop then start
#   status   container/API status on both nodes
#   logs     follow head + worker vLLM logs
#
# NOTE: the 0731 model must be cached on BOTH nodes first (offline serve):
#   ENV_FILE=~/deploy/dspark/.env.dspark.0731 ~/deploy/dspark/prepare-dspark-model-cache.sh
set -euo pipefail

DEPLOY_DIR="${DSPARK_DEPLOY_DIR:-$HOME/deploy/dspark}"

# Select the 0731 profile + an isolated docker project for the deploy scripts.
export ENV_FILE="${ENV_FILE:-$DEPLOY_DIR/.env.dspark.0731}"
export PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash-0731}"
export API_URL="${API_URL:-http://127.0.0.1:8000/v1/models}"
export CHAT_URL="${CHAT_URL:-http://127.0.0.1:8000/v1/chat/completions}"

API_SHOW="http://10.100.72.2:8000/v1   (model: deepseek-v4-flash-0731)"

if [ ! -d "$DEPLOY_DIR" ]; then
  echo "Deploy dir not found: $DEPLOY_DIR (set DSPARK_DEPLOY_DIR to override)" >&2
  exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "Missing 0731 env file: $ENV_FILE" >&2
  exit 1
fi

run() {
  local script="$DEPLOY_DIR/$1"
  if [ ! -x "$script" ]; then
    echo "Missing or non-executable: $script" >&2
    exit 1
  fi
  shift
  "$script" "$@"
}

case "${1:-}" in
  start)
    run start-deepseek-v4-flash-dspark.sh
    echo "Endpoint: ${API_SHOW}" ;;
  stop)
    run stop-deepseek-v4-flash-dspark.sh ;;
  restart)
    run stop-deepseek-v4-flash-dspark.sh
    run start-deepseek-v4-flash-dspark.sh
    echo "Endpoint: ${API_SHOW}" ;;
  status)
    run status-deepseek-v4-flash-dspark.sh ;;
  logs)
    run logs-deepseek-v4-flash-dspark.sh ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 1 ;;
esac
