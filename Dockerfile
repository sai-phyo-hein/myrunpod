# syntax=docker/dockerfile:1.7
#
# Slim RunPod Serverless image for VoxCPM2 with two switchable engines:
#   ENGINE=nano     nanovllm-voxcpm + flash-attn (default)
#   ENGINE=default  reference VoxCPM
#
# Needs no GPU to build: flash-attn comes in as a prebuilt wheel in ./wheels/
# (the build-image workflow downloads it from the build-wheels release).
# Model weights are NOT in the image; select the model as a RunPod cached model.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_VERSION=0.13.4

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
ARG NANOVLLM_REPO=https://github.com/a710128/nanovllm-voxcpm.git
ARG NANOVLLM_REF=b839543f6
ARG INSTALL_VOXCPM=1
ARG VOXCPM_SPEC=voxcpm

COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 1. nanovllm-voxcpm and its locked dependencies (torch, triton, ...), no dev tools
WORKDIR /src
RUN git clone "$NANOVLLM_REPO" nanovllm-voxcpm \
 && cd nanovllm-voxcpm && git checkout "$NANOVLLM_REF"
WORKDIR /src/nanovllm-voxcpm
RUN uv sync --frozen --no-dev --no-editable --no-install-package flash-attn

# 2. prebuilt flash-attn wheel for this image's GPU architecture
COPY wheels/ /tmp/wheels/
RUN set -eux; ls -l /tmp/wheels/*.whl; \
    uv pip install --no-config --python /opt/venv/bin/python --no-deps /tmp/wheels/*.whl

# 3. RunPod SDK and (optionally) the reference VoxCPM engine, in the SAME environment.
#    Constraints keep torch/torchaudio/triton exactly as locked, so no second torch is
#    pulled in. If VoxCPM can't live with this torch, the build fails here, loudly.
RUN set -eux; \
    uv pip freeze --python /opt/venv/bin/python \
      | grep -iE '^(torch|torchaudio|triton)==' > /tmp/constraints.txt; \
    cat /tmp/constraints.txt; \
    extra=""; [ "$INSTALL_VOXCPM" = "1" ] && extra="$VOXCPM_SPEC"; \
    uv pip install --no-config --python /opt/venv/bin/python \
      --constraint /tmp/constraints.txt runpod $extra

# Import-free sanity check (importing flash-attn needs a GPU driver, absent here)
RUN /opt/venv/bin/python - <<'PY'
import importlib.util as u, os
mods = ["nanovllm_voxcpm", "flash_attn", "runpod", "soundfile"]
if os.environ.get("INSTALL_VOXCPM", "1") == "1":
    mods.append("voxcpm")
missing = [m for m in mods if u.find_spec(m) is None]
assert not missing, f"missing modules: {missing}"
print("modules present:", mods)
PY

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE}
ARG FA_ARCH=""

# ffmpeg: torchaudio/torchcodec use it to read reference audio (mp3, flac, ...)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY handler.py /app/handler.py

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ENGINE=nano \
    MODEL_ID=openbmb/VoxCPM2 \
    FLASH_ATTN_ARCH=${FA_ARCH} \
    HF_HOME=/tmp/hf \
    TRITON_CACHE_DIR=/tmp/triton

CMD ["python", "-u", "/app/handler.py"]
