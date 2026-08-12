#!/usr/bin/env bash
#
# ornith.sh — control the Ornith-1.0-35B-FP8 vLLM service (single-node, spark1)
#
# Captured from the live `docker run` config of container `ornith-ornith-35b`.
# Usage: ./ornith.sh {start|stop|restart|status|logs}
#
set -euo pipefail

# ---- Configuration (edit here if the deployment changes) --------------------
NAME="ornith-ornith-35b"
IMAGE="vllm-node:latest"
PORT="${ORNITH_PORT:-8000}"   # unique backend port (composer passes ORNITH_PORT); NOT $PORT (that's sparkDash's 5555)
HF_CACHE="/home/eric/.cache/huggingface"

# vLLM server arguments (passed after the image; entrypoint is `python3`)
VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model deepreinforce-ai/Ornith-1.0-35B-FP8
  --served-model-name ornith-35b
  --host 0.0.0.0
  --port "${PORT}"
  --gpu-memory-utilization "${GPU_UTIL:-0.42}"
  --max-model-len 262144
  --trust-remote-code
)
# -----------------------------------------------------------------------------

container_exists() { docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; }
container_running() { docker ps --format '{{.Names}}' | grep -qx "${NAME}"; }

create() {
  echo "Creating and starting ${NAME}..."
  docker run -d \
    --name "${NAME}" \
    --restart unless-stopped \
    --network host \
    --ipc host \
    --gpus all \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_HOME=/cache/huggingface \
    --entrypoint python3 \
    "${IMAGE}" \
    "${VLLM_ARGS[@]}"
}

# Numeric compare: the composer passes util as toFixed(3) ("0.420") while the
# container records "0.42" — a string compare would force a needless recreate.
same_util() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0==b+0)}'; }

# gpu-memory-utilization baked into the existing container's argv (empty if none).
container_util() {
  docker inspect "${NAME}" --format '{{join .Args " "}}' 2>/dev/null \
    | grep -oE 'gpu-memory-utilization[= ][0-9.]+' | grep -oE '[0-9.]+' | tail -1 || true
}

start() {
  local want have
  want="${GPU_UTIL:-0.42}"
  if container_exists; then
    have="$(container_util)"
    if container_running && [ -n "${have}" ] && same_util "${have}" "${want}"; then
      echo "${NAME} is already running (gpu-util ${have})."
      echo "Endpoint: http://localhost:${PORT}/v1  (model: ornith-35b)"
      return 0
    fi
    # RECREATE rather than `docker start`. Two reasons this matters:
    #  1) `docker start` replays the original argv, silently ignoring a NEW
    #     GPU_UTIL — so the composer's computed util would be discarded and the
    #     model would squat its old (larger) reservation.
    #  2) A container created before a driver reload/reboot carries stale
    #     /dev/nvidia* mappings; restarting it fails with "No CUDA GPUs are
    #     available" even though the host is healthy.
    echo "Recreating ${NAME} (gpu-util ${have:-unknown} -> ${want})..."
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
  fi
  create
  echo "Endpoint: http://localhost:${PORT}/v1  (model: ornith-35b)"
}

stop() {
  if container_running; then
    echo "Stopping ${NAME}..."
    docker stop "${NAME}"
  else
    echo "${NAME} is not running."
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  docker ps -a --filter "name=^/${NAME}$" \
             --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' ;;
  logs)    docker logs -f --tail 100 "${NAME}" ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 1 ;;
esac
