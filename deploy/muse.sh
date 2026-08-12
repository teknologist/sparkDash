#!/usr/bin/env bash
#
# muse.sh — control the Muse-Glimmer-30B (NVFP4) vLLM service (single-node)
#
# Multimodal (image-text-to-text), dense 30B, arch `muse_glimmer`. Requires a
# vLLM >= 0.27 image with muse_glimmer support (PR vllm-project/vllm#51655):
# the GB10 build lives in `vllm-muse:latest` (retagged vllm/vllm-openai:cu130-nightly).
# NVFP4 checkpoint (Blackwell-native FP4, sm_121) is auto-detected from config.json.
#
# Usage: ./muse.sh {start|stop|restart|status|logs}
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
NAME="muse-glimmer"
IMAGE="vllm-muse:latest"
PORT="${MUSE_PORT:-8005}"      # unique backend port (composer passes MUSE_PORT); NOT $PORT (sparkDash's 5555)
MODEL="Preyazz/Muse-Glimmer-30B-NVFP4"
HF_CACHE="/home/eric/.cache/huggingface"

# vLLM server arguments (entrypoint is `python3`; recipe: recipes.vllm.ai/meta-models/Muse-Glimmer-30B)
VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name muse-glimmer
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size 1
  --gpu-memory-utilization "${GPU_UTIL:-0.34}"
  --max-model-len 131072
  --enable-auto-tool-choice --tool-call-parser muse_glimmer
  --reasoning-parser muse_glimmer
  --generation-config auto
)

# DFlash speculative decoding (Meta's drafter, ~3.1x claimed). The dense 30B is
# memory-bandwidth bound on GB10 (~273 GB/s / 21.8 GB weights = ~12.5 tok/s
# roofline), so drafting is the only way past it. Set DFLASH=0 to disable.
if [ "${DFLASH:-1}" = "1" ]; then
  VLLM_ARGS+=(
    --speculative-config "{\"method\": \"dflash\", \"model\": \"meta-models/Muse-Glimmer-30B-assistant\", \"num_speculative_tokens\": ${SPEC_TOKENS:-7}}"
  )
fi
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
  want="${GPU_UTIL:-0.34}"
  if container_exists; then
    have="$(container_util)"
    if container_running && [ -n "${have}" ] && same_util "${have}" "${want}"; then
      echo "${NAME} is already running (gpu-util ${have})."
      echo "Endpoint: http://localhost:${PORT}/v1  (model: muse-glimmer)"
      return 0
    fi
    # RECREATE, never `docker start`: it would replay the old argv (discarding a
    # new GPU_UTIL from the composer) and, for a container created before a
    # driver reload/reboot, carry stale /dev/nvidia* mappings that fail CUDA init.
    echo "Recreating ${NAME} (gpu-util ${have:-unknown} -> ${want})..."
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
  fi
  create
  echo "Endpoint: http://localhost:${PORT}/v1  (model: muse-glimmer)"
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
