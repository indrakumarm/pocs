#!/usr/bin/env bash
# vLLM benchmark runner — runs T1 (sync), T2 (throughput), T3 (sweep)
# for each (model × workload) combination against a running vLLM server.
#
# Usage:
#   ./run_benchmarks.sh
#
# Prereqs:
#   - guidellm installed (pip install guidellm)
#   - vLLM server already running at $TARGET (script does NOT start/stop vLLM)
#   - Change MODELS / WORKLOADS / TARGET below to match your setup

set -euo pipefail

# -------- CONFIG --------
TARGET="${TARGET:-http://0.0.0.0:8000}"
OUT_DIR="${OUT_DIR:-$HOME/vllm_benchmarks}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Models to benchmark — must already be loaded/servable by the vLLM instance.
# For multi-model sweeps, restart vLLM between models and re-run this script.
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct-AWQ"
  # "meta-llama/Llama-3.1-8B-Instruct"
)

# Workload profiles: "label:prompt_tokens:output_tokens"
WORKLOADS=(
  "chat-short:128:128"
  "chat-long:512:256"
  "rag:2048:256"
  # "summarization:4096:512"
  # "gen-heavy:128:2048"
)

# Durations (seconds) per test
DUR_SYNC=60
DUR_THROUGHPUT=120
DUR_SWEEP=60
SWEEP_SIZE=10
# ------------------------

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_${RUN_TAG}.log"
echo "Run tag:    $RUN_TAG" | tee -a "$LOG"
echo "Target:     $TARGET" | tee -a "$LOG"
echo "Output dir: $OUT_DIR" | tee -a "$LOG"
echo | tee -a "$LOG"

start_gpu_logging() {
  local dir="$1"
  # dmon: sm%, mem-bw%, power, clocks, temp — 1s samples with timestamps
  nvidia-smi dmon -s pucmt -o DT -d 1 -f "$dir/gpu_dmon.log" &
  echo $! > "$dir/.dmon.pid"
  # VRAM used, GPU util — for peak VRAM tracking
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,nounits -l 1 > "$dir/gpu_mem.log" &
  echo $! > "$dir/.mem.pid"
}

stop_gpu_logging() {
  local dir="$1"
  for f in .dmon.pid .mem.pid; do
    if [[ -f "$dir/$f" ]]; then
      local pid; pid="$(cat "$dir/$f")"
      kill "$pid" 2>/dev/null || true
      rm -f "$dir/$f"
    fi
  done
}

run_one() {
  local test_id="$1"    # T1 / T2 / T3
  local model="$2"
  local wl_label="$3"
  local ptok="$4"
  local otok="$5"
  local profile_args="$6"
  local duration="$7"

  local safe_model
  safe_model="$(echo "$model" | tr '/' '_' | tr ':' '_')"
  local run_id="${RUN_TAG}_${safe_model}_${wl_label}_${test_id}"
  local dir="$OUT_DIR/$run_id"
  mkdir -p "$dir"

  echo "[$(date +%H:%M:%S)] === $run_id ===" | tee -a "$LOG"

  if command -v nvidia-smi >/dev/null 2>&1; then
    start_gpu_logging "$dir"
  else
    echo "  (nvidia-smi not found — skipping GPU logging)" | tee -a "$LOG"
  fi

  set +e
  guidellm run \
    --backend "kind=openai_http,target=${TARGET},model=${model}" \
    --profile "$profile_args" \
    --constraint "kind=max_duration,seconds=${duration}" \
    --data "kind=synthetic_text,prompt_tokens=${ptok},output_tokens=${otok}" \
    --output-path "$dir/report.html" \
    2>&1 | tee -a "$LOG"
  local rc=$?
  set -e

  stop_gpu_logging "$dir"

  # guidellm also writes benchmarks.json/csv to CWD by default; move them into dir
  [[ -f benchmarks.json ]] && mv benchmarks.json "$dir/benchmarks.json"
  [[ -f benchmarks.csv ]]  && mv benchmarks.csv  "$dir/benchmarks.csv"

  # Write a small metadata file for the analyzer
  cat > "$dir/meta.json" <<EOF
{
  "run_id": "$run_id",
  "run_tag": "$RUN_TAG",
  "test_id": "$test_id",
  "model": "$model",
  "workload": "$wl_label",
  "prompt_tokens": $ptok,
  "output_tokens": $otok,
  "target": "$TARGET",
  "duration_sec": $duration,
  "profile": "$profile_args",
  "exit_code": $rc
}
EOF

  echo "[$(date +%H:%M:%S)] --> exit=$rc, dir=$dir" | tee -a "$LOG"
  echo | tee -a "$LOG"
}

for model in "${MODELS[@]}"; do
  for wl in "${WORKLOADS[@]}"; do
    IFS=':' read -r label ptok otok <<< "$wl"

    # T1 — sync baseline
    run_one "T1" "$model" "$label" "$ptok" "$otok" \
      "kind=synchronous" "$DUR_SYNC"

    # T2 — max throughput
    run_one "T2" "$model" "$label" "$ptok" "$otok" \
      "kind=throughput" "$DUR_THROUGHPUT"

    # T3 — sweep (throughput vs latency curve)
    run_one "T3" "$model" "$label" "$ptok" "$otok" \
      "kind=sweep,sweep_size=${SWEEP_SIZE}" "$DUR_SWEEP"
  done
done

echo "All runs complete. Results under: $OUT_DIR" | tee -a "$LOG"
echo "Next: python3 analyze_benchmarks.py $OUT_DIR" | tee -a "$LOG"
