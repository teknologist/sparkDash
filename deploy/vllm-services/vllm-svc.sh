#!/usr/bin/env bash
# vllm-svc.sh — manage the per-node vLLM services (OCR + Extract).
#
# Both services live inside persistent `sleep infinity` containers built from
# the `vllm-node` image; the actual `vllm serve` process is launched *inside*
# each container with `docker exec`. `docker start` alone does NOT relaunch the
# server, so this script automates the (re)launch / kill of that inner process.
#
# Usage:
#   ./vllm-svc.sh start   [ocr|extract|all]   # (re)launch server(s), wait for API
#   ./vllm-svc.sh stop    [ocr|extract|all]   # kill inner server, keep container up
#   ./vllm-svc.sh restart [ocr|extract|all]
#   ./vllm-svc.sh status  [ocr|extract|all]   # container state + API health
#   ./vllm-svc.sh logs     ocr|extract        # tail the inner server log
#   ./vllm-svc.sh down    [ocr|extract|all]   # also `docker stop` the container
# Omitting the target defaults to `all` (except `logs`, which needs one).
set -euo pipefail

ALL_SERVICES=(ocr extract)
BOOT_POLLS="${BOOT_POLLS:-48}"     # start: API poll attempts
BOOT_INTERVAL="${BOOT_INTERVAL:-10}"  # seconds between polls

# Container host config, used to (re)create a missing service container so the
# same brick can run on ANY node instead of only where one was hand-made.
SVC_IMAGE="${SVC_IMAGE:-vllm-node:latest}"
HF_CACHE="${HF_CACHE:-/home/eric/.cache/huggingface}"

# --- per-service definitions ------------------------------------------------
svc_container() { case "$1" in ocr) echo vllm_ocr ;; extract) echo vllm_extract ;; esac; }
svc_port()      { case "$1" in ocr) echo 8004     ;; extract) echo 8001         ;; esac; }
svc_served()    { case "$1" in ocr) echo qwen3-vl-ocr ;; extract) echo Qwen3.6-27B ;; esac; }
svc_cmd() {
  case "$1" in
    ocr) echo "vllm serve RedHatAI/Qwen3-VL-32B-Instruct-NVFP4 \
--served-model-name qwen3-vl-ocr --host 0.0.0.0 --port 8004 --trust-remote-code \
--max-model-len 65536 --gpu-memory-utilization ${GPU_UTIL:-0.30} \
--kv-cache-dtype fp8 --enable-prefix-caching \
--limit-mm-per-prompt '{\"image\":8}' --enable-chunked-prefill" ;;
    extract) echo "vllm serve nvidia/Qwen3.6-27B-NVFP4 \
--served-model-name Qwen3.6-27B --host 0.0.0.0 --port 8001 --tensor-parallel-size 1 \
--trust-remote-code --gpu-memory-utilization ${GPU_UTIL:-0.45} \
--kv-cache-dtype fp8 --max-model-len 262144 \
--max-num-seqs 8 --reasoning-parser qwen3 --enable-prefix-caching" ;;
  esac
}

# --- helpers ----------------------------------------------------------------
die() { echo "error: $*" >&2; exit 1; }

resolve_targets() {  # expand args (or default 'all') into a validated service list
  local args=("$@") out=()
  [ ${#args[@]} -eq 0 ] && args=(all)
  for a in "${args[@]}"; do
    case "$a" in
      all) out=("${ALL_SERVICES[@]}") ;;
      ocr|extract) out+=("$a") ;;
      *) die "unknown service '$a' (want: ocr | extract | all)" ;;
    esac
  done
  printf '%s\n' "${out[@]}"
}

container_state() {  # prints exactly one of: running | exited | missing
  # `docker inspect -f` on a MISSING container emits an empty line on stdout and
  # the error on stderr, so a naive `|| echo missing` yields "\nmissing" — which
  # matches no `case` branch and silently falls through to a doomed docker exec.
  local c s; c=$(svc_container "$1")
  s=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null | tr -d '\r' | grep -v '^[[:space:]]*$' | tail -n1)
  [ -n "$s" ] && printf '%s\n' "$s" || echo missing
}

api_up() { curl -fsS --max-time 4 "http://127.0.0.1:$(svc_port "$1")/v1/models" >/dev/null 2>&1; }

server_running() {  # is a `vllm serve` process alive inside the container?
  local c; c=$(svc_container "$1")
  docker exec "$c" bash -c "ps -eo args 2>/dev/null | grep -q '[v]llm serve'" 2>/dev/null
}

create_container() {  # idle shell the inner `vllm serve` is exec'd into
  local c="$1"
  echo "  creating container $c from ${SVC_IMAGE} ..."
  docker run -d --name "$c" \
    --network host --ipc host --gpus all \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint sleep \
    "${SVC_IMAGE}" infinity >/dev/null
}

ensure_container_running() {
  local s="$1" c; c=$(svc_container "$s")
  case "$(container_state "$s")" in
    running) : ;;
    exited)  echo "  starting container $c ..."; docker start "$c" >/dev/null ;;
    # Self-provision instead of dying: the composer may place this brick on a
    # node that has never run it, and every node has the image + HF cache.
    missing) create_container "$c" ;;
  esac
  # Guard the exec below: a create/start that silently failed used to surface as
  # "No such container" from `docker exec` and then a bogus "launched" message.
  [ "$(container_state "$s")" = running ] || die "container $c is not running (state: $(container_state "$s"))"
}

# --- actions ----------------------------------------------------------------
start_one() {
  local s="$1" c port; c=$(svc_container "$s"); port=$(svc_port "$s")
  echo "[$s] launching in $c (port $port, served '$(svc_served "$s")')"
  ensure_container_running "$s"
  if api_up "$s"; then echo "  already serving — API healthy on :$port"; return 0; fi
  if server_running "$s"; then echo "  a vllm serve process is already running (still loading?) — not launching a second"; else
    docker exec -d "$c" bash -c "$(svc_cmd "$s") > /tmp/vllm-${s}.log 2>&1" \
      || die "failed to launch server inside $c"
    echo "  launched; waiting for API (up to $((BOOT_POLLS*BOOT_INTERVAL))s)..."
  fi
  for i in $(seq 1 "$BOOT_POLLS"); do
    if api_up "$s"; then echo "  READY: http://127.0.0.1:${port}/v1 (after ${i} polls)"; return 0; fi
    if [ "$(container_state "$s")" != running ]; then
      echo "  CONTAINER DIED. Last log:" >&2; docker exec "$c" tail -n 40 "/tmp/vllm-${s}.log" 2>/dev/null || docker logs --tail 40 "$c" 2>&1 | tail -40; return 1
    fi
    if ! server_running "$s"; then
      echo "  SERVER PROCESS EXITED. Last log:" >&2; docker exec "$c" tail -n 40 "/tmp/vllm-${s}.log" 2>/dev/null; return 1
    fi
    sleep "$BOOT_INTERVAL"
  done
  echo "  TIMEOUT waiting for API. Recent log:" >&2; docker exec "$c" tail -n 40 "/tmp/vllm-${s}.log" 2>/dev/null; return 1
}

stop_one() {
  local s="$1" c; c=$(svc_container "$s")
  echo "[$s] stopping server in $c"
  if [ "$(container_state "$s")" != running ]; then echo "  container not running — nothing to do"; return 0; fi
  if ! server_running "$s"; then echo "  no vllm serve process running"; return 0; fi
  docker exec "$c" bash -c "ps -eo pid,args | awk '/[v]llm serve/{print \$1}' | xargs -r kill" 2>/dev/null || true
  for i in $(seq 1 15); do server_running "$s" || { echo "  stopped."; return 0; }; sleep 2; done
  echo "  graceful stop timed out — killing remaining vllm processes"
  docker exec "$c" bash -c "ps -eo pid,args | awk '/[v]llm/{print \$1}' | xargs -r kill -9" 2>/dev/null || true
  echo "  stopped (forced)."
}

down_one() {
  local s="$1" c; c=$(svc_container "$s")
  stop_one "$s"
  if [ "$(container_state "$s")" = running ]; then echo "[$s] docker stop $c"; docker stop "$c" >/dev/null; fi
}

status_one() {
  local s="$1" c port state; c=$(svc_container "$s"); port=$(svc_port "$s"); state=$(container_state "$s")
  local api="down" proc="no"
  api_up "$s" && api="UP"
  [ "$state" = running ] && server_running "$s" && proc="yes"
  printf '  %-8s container=%-8s(%s)  proc=%-3s  API=%-4s  http://127.0.0.1:%s/v1  [%s]\n' \
    "$s" "$c" "$state" "$proc" "$api" "$port" "$(svc_served "$s")"
}

logs_one() {
  local s="$1" c; c=$(svc_container "$s")
  [ "$(container_state "$s")" = missing ] && die "container $(svc_container "$s") missing"
  docker exec "$c" tail -n "${LOG_LINES:-60}" "/tmp/vllm-${s}.log" 2>/dev/null \
    || { echo "(no /tmp/vllm-${s}.log yet — falling back to docker logs)"; docker logs --tail "${LOG_LINES:-60}" "$c" 2>&1; }
}

# --- dispatch ---------------------------------------------------------------
[ $# -ge 1 ] || die "usage: $0 {start|stop|restart|status|logs|down} [ocr|extract|all]"
action="$1"; shift || true

case "$action" in
  logs)
    [ $# -eq 1 ] || die "logs needs exactly one target: ocr | extract"
    logs_one "$1" ;;
  start|stop|restart|down|status)
    mapfile -t targets < <(resolve_targets "$@")
    rc=0
    for s in "${targets[@]}"; do
      case "$action" in
        start)   start_one "$s"   || rc=1 ;;
        stop)    stop_one "$s"    || rc=1 ;;
        down)    down_one "$s"    || rc=1 ;;
        restart) stop_one "$s"; start_one "$s" || rc=1 ;;
        status)  : ;;
      esac
    done
    if [ "$action" = status ]; then
      echo "$(hostname) vLLM services:"
      for s in "${targets[@]}"; do status_one "$s"; done
    fi
    exit "$rc" ;;
  *) die "unknown action '$action' (want: start|stop|restart|status|logs|down)" ;;
esac
