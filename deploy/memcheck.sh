#!/usr/bin/env bash
#
# memcheck.sh <needGB> [waitSecs] — succeed once the node has <needGB> GiB free.
#
# Used by the generated llama-swap config before an on-demand model start.
# llama-swap models exclusivity, not capacity: without this it starts a model the
# node cannot hold, vLLM dies at engine init and the container crash-loops.
#
# It WAITS rather than failing instantly, because on a swap llama-swap tears the
# previous model down and starts the next one immediately, while the GPU frees
# ~100 GiB asynchronously. Observed 2026-08-29: GLM's health check passed at
# 18:53:43 and DeepSeek was refused 21s later purely because the memory had not
# been returned yet. Polling turns that race into a short wait.
#
# Lives in a script rather than inline in config.yaml because the generated cmd
# is wrapped in `bash -c '...'` and an inline awk program's quotes would end it.
#
# PAGE CACHE (added 2026-09-19): `available` counts reclaimable page cache, but
# CUDA's cuMemGetInfo — which is what vLLM actually gates on — reports only the
# kernel's `free`. After a big weight read the cache can hold >100 GiB, so this
# guard passed at 114 GiB available while vLLM refused with
#   "Free memory on device cuda:0 (113.09/121.69 GiB) ... less than desired
#    GPU memory utilization (0.93, 113.17 GiB)"
# and the container crash-looped. So for a brick that needs more than the
# currently-FREE memory, drop the cache first (best effort: needs root, and is
# harmless when refused — the kernel would reclaim under pressure anyway, just
# not in time for vLLM's one-shot check).
set -uo pipefail
need="${1:-0}"
wait_secs="${2:-240}"
interval=5
elapsed=0

_free_gb() { free -g | awk 'NR==2{print $4}'; }
drop_caches_if_needed() {
  local free_gb; free_gb="$(_free_gb)"; free_gb="${free_gb:-0}"
  [ "$free_gb" -ge "$need" ] && return 0
  sync
  if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    sleep 2
    echo "memcheck: dropped page cache ($(hostname)): free ${free_gb}G -> $(_free_gb)G" >&2
  else
    echo "memcheck: WARNING free is ${free_gb}G (< ${need}G) and page cache cannot be dropped" >&2
    echo "memcheck: CUDA sees 'free', not 'available' — a whole-node model may refuse to start." >&2
    echo "memcheck: grant NOPASSWD for /proc/sys/vm/drop_caches, or run: sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'" >&2
  fi
}
drop_caches_if_needed

while :; do
  avail="$(free -g | awk 'NR==2{print $7}')"
  avail="${avail:-0}"
  [ "$avail" -ge "$need" ] && exit 0
  [ "$elapsed" -ge "$wait_secs" ] && {
    echo "memcheck: need ${need}G on $(hostname), only ${avail}G after ${elapsed}s" >&2
    exit 1
  }
  [ "$elapsed" = 0 ] && echo "memcheck: waiting for ${need}G on $(hostname) (have ${avail}G)..." >&2
  sleep "$interval"; elapsed=$((elapsed + interval))
done
