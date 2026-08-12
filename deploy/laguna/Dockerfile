# Laguna-serving image: dspark vLLM (0.25.2.dev) + the FlashInfer nightly that
# poolside pins for Laguna-S-2.1-NVFP4 (flashinfer-python/cubin/jit-cache
# 0.6.15.dev20260712). The stock image ships a MISMATCHED FlashInfer
# (python 0.6.15 vs cubin/jit-cache 0.6.13), whose compiled init()/plan()
# signatures disagree with the Python API -> TVM-FFI "Mismatched number of
# arguments" crashes in MoE + attention. Pinning all three to the same nightly
# fixes it and enables the intended FlashInfer path.
FROM ghcr.io/anemll/dspark-vllm-gx10:0.1.1

RUN pip install --no-deps \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
      --extra-index-url https://flashinfer.ai/whl/nightly/cu130/ \
      flashinfer-python==0.6.15.dev20260712 \
      flashinfer-cubin==0.6.15.dev20260712 \
      flashinfer-jit-cache==0.6.15.dev20260712+cu130 \
 && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
