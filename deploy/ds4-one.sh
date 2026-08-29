#!/usr/bin/env bash
#
# ds4-one.sh — control the single-node DeepSeek V4 Flash 0731 SparkInfer stack.
#
#   spark1: MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark — vLLM (NVIDIA 26.02 base)
#   + SparkInfer kernels serving the EXL3 3.0bpw REAP-K216 checkpoint (~99.5 GiB)
#   with DSpark K5 speculative decoding, 384k context, native 432 B NVFP4 KV.
#   -> http://<host>:8000/v1  (served: deepseek-v4-flash-0731-solo)
#
# Thin wrapper over that repo's own start.sh (a docker compose stack it
# generates), pinning the two settings sparkDash depends on so a hand-run and a
# dashboard-run are identical:
#
#   SERVING_PORT=8000        the ds4 slot behind the llama-swap gateway
#   SERVED_MODEL_NAME=…-solo distinct from the dual TP=2 brick, which also
#                            serves on :8000 as `deepseek-v4-flash-0731`
#   MAX_MODEL_LEN=340000     see the context note below
#   GPU_MEMORY_UTILIZATION=0.93
#
# Everything else stays at the upstream deep-context profile (MAX_NUM_SEQS=1,
# MODE=dspark, KV_RECORD=stock432).
#
# CONTEXT NOTE — why not upstream's 384000 @ util 0.94:
#   That profile claims 114.39 of the 121.69 GiB unified pool, leaving 7.30 GiB
#   for the host. This node's baseline is ~6.0 GiB (GNOME session + sparkDash +
#   docker), so vLLM's multi-GB startup transient overruns the margin and the
#   kernel OOM-killer takes VLLM::EngineCore right after KV sizing — observed
#   twice on 2026-08-26 (18:35:59 and 18:40:14), both times with graph capture
#   already finished. 384k is not reachable at a lower util either: util 0.935
#   yields only 380,194 tokens of KV, short of 384,000. Measured fixed cost is
#   106.41 GiB (weights 95.39 + activation 5.81 + non-torch 5.21), so:
#     util 0.930 -> KV 6.76 GiB = ~348,800 tokens, host keeps 8.52 GiB  <= chosen
#     util 0.940 -> KV 7.98 GiB = ~411,600 tokens, host keeps 7.30 GiB  (OOMs)
#   340000 sits inside the 0.93 pool at ~1.03x concurrency. Upstream validated
#   0.94 on a headless box; raise these two only if this host loses its desktop.
#
# This stack still takes the whole node — stop every other local model first.
# Weights must be staged once before the first start (~107 GB, no GPU needed):
#   /home/eric/deploy/ds4-one/download.sh
#
# Usage: ds4-one.sh {start|stop|restart|status|logs}
set -euo pipefail

DS4ONE_DIR="${DS4ONE_DIR:-/home/eric/deploy/ds4-one}"
export SERVING_PORT="${SERVING_PORT:-8000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-solo}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-340000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
CHECK_HOST="${DS4ONE_CHECK_HOST:-127.0.0.1}"
API="http://$CHECK_HOST:$SERVING_PORT"

[ -x "$DS4ONE_DIR/start.sh" ] || {
  echo "ds4-one: launcher missing: $DS4ONE_DIR/start.sh (clone MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark there)" >&2
  exit 1
}
cd "$DS4ONE_DIR"

_api_up() { curl -fsS --max-time 4 "$API/v1/models" >/dev/null 2>&1; }

case "${1:-}" in
  start)
    if _api_up; then
      echo "ds4-one already serving: $API/v1  ($SERVED_MODEL_NAME)"
      exit 0
    fi
    [ -f data/tp1/rank-sliced-tp1-manifest.json ] || {
      echo "ds4-one: TP1 checkpoint not staged — run $DS4ONE_DIR/download.sh first" >&2
      exit 1
    }
    ./start.sh
    echo "ds4-one API up: $API/v1  ($SERVED_MODEL_NAME)"
    ;;
  stop)    ./start.sh stop ;;
  restart) ./start.sh stop; ./start.sh ;;
  status)
    echo "== spark1: DeepSeek V4 Flash 0731 (SparkInfer 1-node, EXL3 3.0bpw) =="
    ./start.sh ps 2>/dev/null | tail -n +1
    if _api_up; then
      echo "  API: UP ($API)  served: $SERVED_MODEL_NAME"
    else
      echo "  API: down"
    fi
    ;;
  logs)    ./start.sh logs ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 1 ;;
esac
