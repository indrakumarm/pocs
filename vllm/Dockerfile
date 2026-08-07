# =============================================================================
# Dockerfile — vLLM OpenAI-compatible server
#
# Base image: vllm/vllm-openai  (official image; already contains vLLM,
#             PyTorch, CUDA runtime, and the OpenAI-compatible API server)
#
# The model is NOT baked into this image. Choose one of the two options below.
#
# ── Build ────────────────────────────────────────────────────────────────────
#   docker build -t my-vllm .
#
# ── Option A: pull model from HuggingFace at container startup ───────────────
#   docker run --gpus all -p 8000:8000 \
#     -e VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
#     -e HF_TOKEN=hf_xxxx \
#     my-vllm
#
#   To persist the downloaded model across restarts (avoid re-downloading):
#   docker run --gpus all -p 8000:8000 \
#     -e VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
#     -e HF_TOKEN=hf_xxxx \
#     -v /host/hf-cache:/root/.cache/huggingface \
#     my-vllm
#
# ── Option B: mount a pre-downloaded model directory ─────────────────────────
#   # Mount entire HF hub cache (recommended — works with HF snapshot layout):
#   docker run --gpus all -p 8000:8000 \
#     -e VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
#     -v /host/hf-cache:/root/.cache/huggingface \
#     my-vllm
#
#   # Or mount a flat model folder directly:
#   docker run --gpus all -p 8000:8000 \
#     -e VLLM_MODEL=/model \
#     -v /host/models/Qwen2.5-7B-Instruct:/model:ro \
#     my-vllm
#
# ── Override config file ─────────────────────────────────────────────────────
#   docker run --gpus all -p 8000:8000 \
#     -v /host/path/vllm.yaml:/app/config/vllm.yaml:ro \
#     my-vllm
# =============================================================================

ARG VLLM_VERSION=latest
FROM vllm/vllm-openai:${VLLM_VERSION}

# Install yq for lightweight YAML introspection in the init script (optional)
# Using the static binary to avoid Python dependency noise.
# Remove this block if you prefer pure-bash config handling.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
        -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── App layout ───────────────────────────────────────────────────────────────
WORKDIR /app

# Copy vLLM config (editable without rebuilding via volume mount)
COPY config/vllm.yaml /app/config/vllm.yaml

# Copy and prepare entrypoint + verify script
COPY scripts/init-script.sh /app/init-script.sh
COPY scripts/verify.sh      /app/scripts/verify.sh
RUN chmod +x /app/init-script.sh /app/scripts/verify.sh

# ── HuggingFace cache directory ──────────────────────────────────────────────
# This is where HF downloads models to and where vLLM looks for cached models.
# Mount a host directory here to persist downloads or supply pre-downloaded models:
#   -v /host/hf-cache:/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# ── Runtime defaults (overridable via -e at docker run) ──────────────────────
ENV VLLM_CONFIG_FILE=/app/config/vllm.yaml

# Expose the default vLLM API port
EXPOSE 8000

# ── Entrypoint ───────────────────────────────────────────────────────────────
ENTRYPOINT ["/app/init-script.sh"]
