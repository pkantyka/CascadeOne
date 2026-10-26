ARG CUDA_BASE
FROM ${CUDA_BASE}

ARG TARGETARCH
RUN if [ "$TARGETARCH" != "arm64" ]; then \
      echo "DGX image build requires native ARM64/DGX Spark" >&2; \
      exit 2; \
    fi

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    USE_TF=0 \
    HF_HOME=/cache/huggingface

# CUDA/PyTorch comes from the DGX-selected base image. Keep it visible but isolate
# project packages so pip does not replace the NVIDIA-provided torch installation.
RUN python -m venv --system-site-packages /opt/cascadeone
ENV PATH=/opt/cascadeone/bin:$PATH

WORKDIR /workspace
COPY requirements-dgx.in /tmp/requirements-dgx.in
RUN python -c "import torch; print('base torch', torch.__version__, 'cuda', torch.version.cuda)" && \
    PIP_CONSTRAINT= python -m pip install --no-cache-dir -r /tmp/requirements-dgx.in && \
    python -c "import torch; print('final torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY . .
CMD ["python", "cascades/laya-clef/serve_backend.py", "--which", "fast"]
