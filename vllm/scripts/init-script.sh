#!/usr/bin/env bash
# =============================================================================
# init-script.sh — vLLM container entrypoint
#
# Starts `vllm serve` using /app/config/vllm.yaml as the base config.
# Any parameter can be overridden at `docker run` time via environment vars:
#
#   ENV VAR                  OVERRIDES yaml key
#   ─────────────────────── ──────────────────────────────
#   VLLM_MODEL               model
#   VLLM_HOST                host
#   VLLM_PORT                port
#   VLLM_API_KEY             api-key
#   VLLM_DTYPE               dtype
#   VLLM_MAX_MODEL_LEN       max-model-len
#   VLLM_GPU_MEM_UTIL        gpu-memory-utilization
#   VLLM_TENSOR_PARALLEL     tensor-parallel-size
#   VLLM_TRUST_REMOTE_CODE   trust-remote-code  (set to "true" to enable)
#   VLLM_EXTRA_ARGS          any additional raw CLI flags appended verbatim
#                            e.g. "--disable-log-requests --uvicorn-log-level debug"
#
# HuggingFace token (for gated models):
#   HF_TOKEN                 written to $HF_HOME/token automatically
#
# Model delivery — choose one:
#   Pull at runtime:   set VLLM_MODEL=<HF repo id>  and optionally HF_TOKEN
#   Volume-mounted HF cache:
#                      -v /host/hf-cache:/root/.cache/huggingface
#                      -e VLLM_MODEL=<HF repo id>   (vLLM resolves from cache)
#   Volume-mounted flat dir:
#                      -v /host/models/mymodel:/model:ro
#                      -e VLLM_MODEL=/model
# =============================================================================

set -euo pipefail

CONFIG_FILE="${VLLM_CONFIG_FILE:-/app/config/vllm.yaml}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

# ── HuggingFace auth ─────────────────────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    mkdir -p "${HF_HOME}"
    echo "${HF_TOKEN}" > "${HF_HOME}/token"
    echo "[init] HF_TOKEN written to ${HF_HOME}/token"
fi

# ── Validate config file ─────────────────────────────────────────────────────
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[init] ERROR: config file not found at ${CONFIG_FILE}" >&2
    exit 1
fi
echo "[init] Using config: ${CONFIG_FILE}"
echo "[init] HF_HOME: ${HF_HOME}"

# ── Validate that a model is specified ───────────────────────────────────────
# Model must come from either VLLM_MODEL env var or the 'model:' key in vllm.yaml.
# We do a simple grep check on the config as a sanity guard.
if [[ -z "${VLLM_MODEL:-}" ]]; then
    if ! grep -qE '^\s*model\s*:' "${CONFIG_FILE}"; then
        echo "[init] ERROR: No model specified." >&2
        echo "[init]   Set -e VLLM_MODEL=<HF repo id or /local/path>" >&2
        echo "[init]   or uncomment 'model:' in ${CONFIG_FILE}" >&2
        exit 1
    fi
    echo "[init] Model: (from config file)"
else
    echo "[init] Model: ${VLLM_MODEL}  (from env override)"
fi

# ── Build override args from env vars ────────────────────────────────────────
OVERRIDE_ARGS=()

[[ -n "${VLLM_MODEL:-}"            ]] && OVERRIDE_ARGS+=("--model"                    "${VLLM_MODEL}")
[[ -n "${VLLM_HOST:-}"             ]] && OVERRIDE_ARGS+=("--host"                     "${VLLM_HOST}")
[[ -n "${VLLM_PORT:-}"             ]] && OVERRIDE_ARGS+=("--port"                     "${VLLM_PORT}")
[[ -n "${VLLM_API_KEY:-}"          ]] && OVERRIDE_ARGS+=("--api-key"                  "${VLLM_API_KEY}")
[[ -n "${VLLM_DTYPE:-}"            ]] && OVERRIDE_ARGS+=("--dtype"                    "${VLLM_DTYPE}")
[[ -n "${VLLM_MAX_MODEL_LEN:-}"    ]] && OVERRIDE_ARGS+=("--max-model-len"            "${VLLM_MAX_MODEL_LEN}")
[[ -n "${VLLM_GPU_MEM_UTIL:-}"     ]] && OVERRIDE_ARGS+=("--gpu-memory-utilization"   "${VLLM_GPU_MEM_UTIL}")
[[ -n "${VLLM_TENSOR_PARALLEL:-}"  ]] && OVERRIDE_ARGS+=("--tensor-parallel-size"     "${VLLM_TENSOR_PARALLEL}")
[[ "${VLLM_TRUST_REMOTE_CODE:-}" == "true" ]] && OVERRIDE_ARGS+=("--trust-remote-code")

# Split VLLM_EXTRA_ARGS into an array safely (word-split is intentional here)
# shellcheck disable=SC2206
EXTRA=( ${VLLM_EXTRA_ARGS:-} )

# ── Launch vLLM in the background ────────────────────────────────────────────
echo "[init] Starting vllm serve ..."
echo "[init] Override args: ${OVERRIDE_ARGS[*]:-<none>}"
echo "[init] Extra args:    ${EXTRA[*]:-<none>}"

vllm serve \
    --config "${CONFIG_FILE}" \
    "${OVERRIDE_ARGS[@]}" \
    "${EXTRA[@]}" &

VLLM_PID=$!
echo "[init] vllm serve started (PID ${VLLM_PID})"

# ── Verify vLLM is ready ──────────────────────────────────────────────────────
VERIFY_SCRIPT="${VLLM_VERIFY_SCRIPT:-/app/scripts/verify.sh}"

if [[ -x "${VERIFY_SCRIPT}" ]]; then
    # Pass VLLM_PORT through so verify.sh probes the right port
    export VLLM_PORT="${VLLM_PORT:-8000}"
    if ! "${VERIFY_SCRIPT}"; then
        echo "[init] ERROR: vLLM failed readiness check — shutting down." >&2
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
else
    echo "[init] WARNING: verify script not found at ${VERIFY_SCRIPT} — skipping readiness check."
fi

# ── Keep the container alive by waiting on vLLM ──────────────────────────────
# Propagate SIGTERM/SIGINT to vLLM so the container stops cleanly.
trap 'echo "[init] Caught signal — stopping vllm (PID ${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null' SIGTERM SIGINT

wait "${VLLM_PID}"
