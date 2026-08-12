# vllm-muse:latest — vLLM with muse_glimmer support (PR vllm-project/vllm#51655)
# built for GB10 (aarch64, Blackwell sm_121a, CUDA 13) on the laguna GB10 base
# (vLLM 0.25.2, torch 2.11.0+cu130, DFlash-capable). NB: the muse branch pins
# torch 2.13; we force the base's 2.11 via --no-build-isolation + override.
ARG BASE_IMAGE=laguna-vllm:fi-nightly
FROM ${BASE_IMAGE}

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=16 \
    UV_SYSTEM_PYTHON=1 \
    VLLM_TARGET_DEVICE=cuda

# git: CMake FetchContent clones cutlass/etc. during the build
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# NVRTC dev symlink: the base ships libnvrtc.so.13 but not the unversioned
# libnvrtc.so that CMake FindCUDAToolkit resolves (CUDA_nvrtc_LIBRARY).
RUN d=/usr/local/cuda-13.0/targets/sbsa-linux/lib && \
    ln -sf "$d/libnvrtc.so.13" "$d/libnvrtc.so" && \
    ln -sf "$d/libnvrtc-builtins.so.13.0" "$d/libnvrtc-builtins.so" && \
    ln -sf "$d/libnvrtc.so.13" /usr/local/cuda/lib64/libnvrtc.so 2>/dev/null || true

# The base is a runtime-ish CUDA tree: math-lib dev headers (cusparse.h, ...)
# ship only in torch's nvidia/cu13 pip wheels. Link every header/lib the wheels
# provide into the toolkit tree so nvcc/g++ resolve them (fatal: cusparse.h).
RUN W=/usr/local/lib/python3.12/dist-packages/nvidia/cu13 && \
    INC=/usr/local/cuda-13.0/targets/sbsa-linux/include && \
    LIB=/usr/local/cuda-13.0/targets/sbsa-linux/lib && \
    for f in "$W"/include/*; do [ -e "$INC/$(basename "$f")" ] || ln -sf "$f" "$INC/"; done && \
    for f in "$W"/lib/*.so*; do [ -e "$LIB/$(basename "$f")" ] || ln -sf "$f" "$LIB/"; done && \
    for s in cusparse cublas cublasLt cufft curand cusolver; do \
      for cand in "$LIB/lib$s.so".*; do \
        [ -e "$cand" ] && [ ! -e "$LIB/lib$s.so" ] && ln -sf "$cand" "$LIB/lib$s.so"; \
      done; done; \
    echo "cusparse.h: $(ls -l $INC/cusparse.h 2>/dev/null || echo MISSING)"

COPY vllm-src /workspace/vllm-muse
WORKDIR /workspace/vllm-muse

# Pin to the base image's torch (2.11+cu130) so we build the muse-branch vLLM
# against the validated GB10 stack instead of pulling torch 2.13.
RUN PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    printf 'torch==%s\n' "${PINNED_TORCH}" > /tmp/torch-override.txt && \
    cat /tmp/torch-override.txt && \
    uv pip uninstall vllm || true && \
    uv pip install --no-build-isolation --override /tmp/torch-override.txt \
        setuptools wheel setuptools-scm setuptools-rust ninja cmake && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0 \
    uv pip install --no-build-isolation --override /tmp/torch-override.txt -e .

# The muse branch pins torchvision 0.28 (built for torch 2.13); installing it
# breaks the C++ ABI against the base's torch 2.11 ("operator torchvision::nms
# does not exist"). Restore the GB10-validated torch-family versions. Kept as a
# separate step so the long CUDA compile layer above stays cached.
RUN uv pip install --no-deps --reinstall \
        torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 \
        --index-url https://download.pytorch.org/whl/cu130 \
    || uv pip install --no-deps --reinstall torchvision==0.26.0+cu130 \
        --index-url https://download.pytorch.org/whl/cu130
RUN python3 -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__); import torchvision.ops as o; print('nms op OK')"

# sanity: arch must be registered
RUN python3 -c "from vllm.model_executor.models.registry import ModelRegistry as R; a=R.get_supported_archs(); assert any('muse' in x.lower() or 'glimmer' in x.lower() for x in a), 'muse_glimmer NOT registered'; import vllm; print('OK vllm', vllm.__version__)"
