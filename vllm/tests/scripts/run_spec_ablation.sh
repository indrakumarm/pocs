#!/usr/bin/env bash
# Spec-decoding ablation runner.
#
# Runs the concurrent-streams test (T3) at fixed concurrencies, once for a
# baseline (spec off) and once for spec-on. You are responsible for restarting
# vLLM with the appropriate flags between the two runs.
#
# Usage:
#   # 1. Start vLLM WITHOUT --speculative-model
#   ./run_spec_ablation.sh spec_off
#
#   # 2. Restart vLLM WITH:
#   #    --speculative-model <draft-model> --num-speculative-tokens 5
#   ./run_spec_ablation.sh spec_on
#
#   # 3. Compare
#   python3 compare_spec.py ~/spec_ablation

set -euo pipefail

LABEL="${1:?usage: $0 <label>  e.g. spec_off | spec_on}"

TARGET="${TARGET:-http://0.0.0.0:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
STREAMS_JSON="${STREAMS_JSON:-[1,2,4,6,8]}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
DURATION="${DURATION:-90}"
OUT_DIR="${OUT_DIR:-$HOME/spec_ablation}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_DIR/${LABEL}_${TS}"
mkdir -p "$RUN_DIR"

echo "Label:    $LABEL"
echo "Target:   $TARGET"
echo "Model:    $MODEL"
echo "Streams:  $STREAMS_JSON"
echo "Workload: ${PROMPT_TOKENS} in / ${OUTPUT_TOKENS} out, ${DURATION}s per stage"
echo "Output:   $RUN_DIR"
echo

# Snapshot vLLM /metrics BEFORE — used to compute spec acceptance rate delta
if command -v curl >/dev/null 2>&1; then
  curl -s "${TARGET%/}/metrics" > "$RUN_DIR/vllm_metrics_before.txt" || true
fi

# Start GPU logging
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucmt -o DT -d 1 -f "$RUN_DIR/gpu_dmon.log" &
  DMON_PID=$!
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,nounits -l 1 > "$RUN_DIR/gpu_mem.log" &
  MEM_PID=$!
  trap 'kill $DMON_PID $MEM_PID 2>/dev/null || true' EXIT
fi

# Run T3
guidellm run \
  --backend "kind=openai_http,target=${TARGET},model=${MODEL}" \
  --profile "{\"kind\":\"concurrent\",\"streams\":${STREAMS_JSON}}" \
  --constraint "kind=max_duration,seconds=${DURATION}" \
  --data "kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
  --output-path "$RUN_DIR/report.html" \
  2>&1 | tee "$RUN_DIR/guidellm.log"

# Stop GPU logging
if command -v nvidia-smi >/dev/null 2>&1; then
  kill $DMON_PID $MEM_PID 2>/dev/null || true
  trap - EXIT
fi

# Snapshot vLLM /metrics AFTER
if command -v curl >/dev/null 2>&1; then
  curl -s "${TARGET%/}/metrics" > "$RUN_DIR/vllm_metrics_after.txt" || true
fi

# Move guidellm side-artifacts
[[ -f benchmarks.json ]] && mv benchmarks.json "$RUN_DIR/benchmarks.json"
[[ -f benchmarks.csv ]]  && mv benchmarks.csv  "$RUN_DIR/benchmarks.csv"

# Meta
cat > "$RUN_DIR/meta.json" <<EOF
{
  "label": "$LABEL",
  "target": "$TARGET",
  "model": "$MODEL",
  "streams": ${STREAMS_JSON},
  "prompt_tokens": ${PROMPT_TOKENS},
  "output_tokens": ${OUTPUT_TOKENS},
  "duration_sec": ${DURATION},
  "timestamp": "$TS"
}
EOF

echo
echo "Done. Results in $RUN_DIR"
echo "Next: compare with"
echo "  python3 compare_spec.py $OUT_DIR"
