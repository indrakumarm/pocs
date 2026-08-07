#!/usr/bin/env bash
# =============================================================================
# verify.sh — wait until the vLLM OpenAI-compatible server is ready
#
# Called automatically by init-script.sh after vLLM starts.
# Can also be run standalone from outside the container for smoke-testing.
#
# Configurable via env vars:
#   VLLM_PORT            port vLLM listens on              (default: 8000)
#   VLLM_VERIFY_TIMEOUT  max seconds to wait for readiness (default: 300)
#   VLLM_VERIFY_INTERVAL poll interval in seconds          (default: 5)
#   VLLM_VERIFY_HOST     host to probe                     (default: localhost)
#
# Exit codes:
#   0  — vLLM is up and healthy
#   1  — timed out or server returned an error
# =============================================================================

set -euo pipefail

PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_VERIFY_HOST:-localhost}"
TIMEOUT="${VLLM_VERIFY_TIMEOUT:-300}"
INTERVAL="${VLLM_VERIFY_INTERVAL:-5}"
HEALTH_URL="http://${HOST}:${PORT}/health"

echo "[verify] Waiting for vLLM to be ready at ${HEALTH_URL}"
echo "[verify] Timeout: ${TIMEOUT}s  |  Interval: ${INTERVAL}s"

elapsed=0

while true; do
    # Use curl: -sf = silent + fail on HTTP error, -o /dev/null = discard body
    if curl -sf -o /dev/null --max-time 5 "${HEALTH_URL}" 2>/dev/null; then
        echo "[verify] vLLM is UP after ${elapsed}s — ${HEALTH_URL} returned 200"
        exit 0
    fi

    if (( elapsed >= TIMEOUT )); then
        echo "[verify] TIMEOUT after ${elapsed}s — vLLM did not become ready at ${HEALTH_URL}" >&2
        echo "[verify] Check container logs for startup errors." >&2
        exit 1
    fi

    echo "[verify] Not ready yet (${elapsed}s elapsed, timeout ${TIMEOUT}s) — retrying in ${INTERVAL}s ..."
    sleep "${INTERVAL}"
    (( elapsed += INTERVAL ))
done
