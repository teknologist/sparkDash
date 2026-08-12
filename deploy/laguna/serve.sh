#!/usr/bin/env bash
# serve.sh — launch Laguna-S-2.1-NVFP4 on spark1 (single node) and wait for the API.
#
# Laguna needs vLLM >= 0.25 + the poolside_v1 parsers + the LagunaForCausalLM
# arch, none of which exist in the Ornith `vllm-node` image (0.23.1). It is
# served with the anemll dspark image (vLLM 0.25.2), which ships laguna.py,
# laguna_dflash.py and the poolside_v1 reasoning/tool parsers.
#
# Config via env (all optional):
#   NAME              served-model-name / container name   (default laguna-s)
#   LAGUNA_PORT       OpenAI API port                       (default 8000)
#                     (NOT "PORT" — that collides with sparkDash's PORT=5555 env)
#                     Convention: single model on a spark node -> 8000.
#   MML               --max-model-len                       (default 262144 = 256K native)
#   UTIL              --gpu-memory-utilization              (default 0.80)
#                     NOTE: GB10 is unified memory (128 GB shared CPU+GPU). A
#                     high util starves host RAM needed to stage the 67 GB
#                     checkpoint at load time -> kernel OOM (exit 137). 0.80
#                     leaves headroom; only raise it on an otherwise-idle node.
#   MAX_NUM_SEQS      --max-num-seqs                        (default 32)
#   ENABLE_SPEC       1=DFlash speculative decode, 0=off    (default 1)
#   SPEC_TOKENS       num_speculative_tokens when ENABLE_SPEC=1 (default 15)
#   IMAGE             serving image  (default laguna-vllm:fi-nightly, built
#                     from this dir's Dockerfile = dspark image + pinned
#                     FlashInfer nightly 0.6.15.dev20260712)
#   MOE_BACKEND       --moe-backend value                   (default auto)
#   ATTN_BACKEND      --attention-backend value             (default auto)
#                     Both 'auto' use the intended FlashInfer path, which works
#                     once FlashInfer is version-matched (the fi-nightly image).
#                     On the STOCK dspark image FlashInfer is mismatched and its
#                     init()/plan() throw TVM-FFI "Mismatched number of
#                     arguments"; work around with MOE_BACKEND=cutlass and
#                     ATTN_BACKEND=TRITON_ATTN. moe valid: auto, cutlass,
#                     flashinfer_cutlass, flashinfer_trtllm, marlin, triton, ...
#                     attn valid: auto, FLASHINFER, TRITON_ATTN, ...
set -euo pipefail

IMAGE="${IMAGE:-laguna-vllm:fi-nightly}"   # dspark image + pinned FlashInfer nightly (see Dockerfile)
REPO="poolside/Laguna-S-2.1-NVFP4"
DRAFT_REPO="poolside/Laguna-S-2.1-DFlash-NVFP4"

NAME="${NAME:-laguna-s}"
LAGUNA_PORT="${LAGUNA_PORT:-8000}"
MML="${MML:-262144}"
UTIL="${GPU_UTIL:-${UTIL:-0.80}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ENABLE_SPEC="${ENABLE_SPEC:-1}"
SPEC_TOKENS="${SPEC_TOKENS:-15}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
# With the pinned FlashInfer nightly (laguna-vllm:fi-nightly) the intended
# FlashInfer MoE + attention path works, so both backends are 'auto'. If you
# ever run this against the STOCK dspark image (mismatched FlashInfer), its
# init()/plan() throw TVM-FFI "Mismatched number of arguments" — work around it
# with MOE_BACKEND=cutlass ATTN_BACKEND=TRITON_ATTN.
ATTN_BACKEND="${ATTN_BACKEND:-auto}"
CID="$NAME"

# vLLM serve arguments (mirrors poolside's recommended command, adapted for
# offline docker serving).
SERVE_ARGS=(
  serve "$REPO"
  --served-model-name "$NAME"
  --host 0.0.0.0 --port "$LAGUNA_PORT"
  --gpu-memory-utilization "$UTIL"
  --max-model-len "$MML"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-auto-tool-choice
  --tool-call-parser poolside_v1
  --reasoning-parser poolside_v1
  --override-generation-config '{"temperature":0.7,"top_p":0.95}'
  --trust-remote-code
)
if [ "$MOE_BACKEND" != "auto" ]; then
  SERVE_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [ "$ATTN_BACKEND" != "auto" ]; then
  SERVE_ARGS+=(--attention-backend "$ATTN_BACKEND")
fi
if [ "$ENABLE_SPEC" = "1" ]; then
  SERVE_ARGS+=(--speculative-config "{\"model\":\"${DRAFT_REPO}\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
fi

docker rm -f "$CID" >/dev/null 2>&1 || true
docker run -d --name "$CID" \
  --gpus all --ipc host --network host --shm-size 64g \
  -v /home/eric/.cache/huggingface:/cache/huggingface \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DSL_ARCH=sm_121a -e MAX_JOBS=4 \
  --entrypoint vllm "$IMAGE" \
  "${SERVE_ARGS[@]}" >/dev/null

echo "Launched $CID (repo=$REPO port=$LAGUNA_PORT mml=$MML util=$UTIL spec=$ENABLE_SPEC). Waiting for API..."
for i in $(seq 1 90); do
  if curl -fsS --max-time 4 "http://127.0.0.1:${LAGUNA_PORT}/v1/models" >/dev/null 2>&1; then
    echo "READY: http://127.0.0.1:${LAGUNA_PORT}/v1 (after ${i} polls)"; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CID"; then
    echo "CONTAINER DIED. Last logs:" >&2; docker logs --tail 60 "$CID" 2>&1 | tail -60; exit 1
  fi
  sleep 10
done
echo "TIMEOUT waiting for API. Recent logs:" >&2; docker logs --tail 60 "$CID" 2>&1 | tail -60; exit 1
