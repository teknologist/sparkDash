#!/usr/bin/env bash
#
# build.sh — build `vllm-muse:latest`, the GB10 (aarch64 / sm_121a / CUDA 13)
# vLLM image that can serve Muse Glimmer (arch muse_glimmer, vLLM >= 0.28).
#
# Why a custom image: no prebuilt image serves muse_glimmer on GB10. Upstream's
# vllm/vllm-openai:muse-glimmer is x86_64, and the arm64 cu130 nightly lags far
# behind (0.19.x, no muse). Support comes from PR vllm-project/vllm#51655, so we
# compile that branch against the node's validated torch 2.11+cu130 stack.
#
# The Dockerfile COPYs a vLLM checkout, which is far too large to vendor here —
# this script fetches it first, making the build reproducible from the repo alone.
#
# Prerequisite: the base image must exist locally (a working GB10 vLLM build):
#   BASE_IMAGE (default laguna-vllm:fi-nightly)
#
# Usage: ./deploy/vllm-muse/build.sh          # ~35-90 min of CUDA compilation
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MUSE_VLLM_REPO:-https://github.com/xianbaoqian/vllm}"
BRANCH="${MUSE_VLLM_BRANCH:-tiezhen/new-model-support}"
TAG="${MUSE_IMAGE:-vllm-muse:latest}"
BASE="${BASE_IMAGE:-laguna-vllm:fi-nightly}"

docker image inspect "$BASE" >/dev/null 2>&1 || {
  echo "error: base image '$BASE' not found locally." >&2
  echo "       It must be a working GB10 vLLM image (torch 2.11+cu130, sm_121a)." >&2
  exit 1
}

if [ ! -d "$HERE/vllm-src" ]; then
  echo "==> cloning $BRANCH from $REPO"
  git clone --depth 1 -b "$BRANCH" "$REPO" "$HERE/vllm-src"
  rm -rf "$HERE/vllm-src/.git"   # keeps the docker build context small
else
  echo "==> reusing existing $HERE/vllm-src"
fi

echo "==> building $TAG from $BASE (this takes a while: CUDA kernels for sm_121a)"
docker build --build-arg BASE_IMAGE="$BASE" -t "$TAG" "$HERE"

echo "==> done. Verify muse_glimmer is registered:"
echo "docker run --rm --entrypoint python3 $TAG -c \\"
echo "  \"from vllm.model_executor.models.registry import ModelRegistry as R; \\"
echo "   print([x for x in R.get_supported_archs() if 'muse' in x.lower()])\""
