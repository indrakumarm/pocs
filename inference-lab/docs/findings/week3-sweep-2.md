# week3 sweep 2 (fixed prompts, constant KV memory)

**Tool:** sweep.py · **CSV:** `results/week3.csv` · **Logs:** `results/logs/*.log` · **Plot:** `make plot` → `results/week3.png`

## Setup
- GPU: A10G (22 GiB visible, 600 GB/s, 125 TFLOPS bf16). Model: Qwen/Qwen3-8B bf16, vLLM latest.
- Server: `--max-model-len 8192 --gpu-memory-utilization 0.90` in every experiment (except `gpu-util-0.85`).
- Bench: `--input-tokens 1024 --output-tokens 256 --concurrency 1 4 16 32 64`, 32 requests per level (64/128 at c=32/64).
- Measured prompt length: **1,058–1,073 tokens** (target 1,024 + chat template + nonce). Shared-prefix tests: **961 tokens**.
- First tooling-validated sweep: real token lengths, unique prompts, equal KV memory per experiment.

## Results
Usable = TTFT p99 ≤ 2 s **and** ITL p99 ≤ 100 ms. Knee = last level before throughput grows < 10%.

| Experiment | Knee | Peak tok/s | vs baseline | Usable (c, tok/s) | TTFT p99 @c16 | ITL p99 @c16 |
|---|---|---|---|---|---|---|
| **baseline** | 16 | 262 | – | 4, 97 | 4.1 s | 490 ms |
| max-seqs-256 | 16 | 262 | 0% | 4, 97 | 4.1 s | 492 ms |
| batched-tokens-2048 | 16 | 262 | 0% | 4, 97 | 4.1 s | 491 ms |
| max-seqs-32 | 16 | 280 | +7% | 4, 97 | 4.1 s | 492 ms |
| batched-tokens-16384 | 32 | 245 | −7% | 4, 97 | **8.0 s** | **68 ms** |
| no-chunked-prefill | 16 | 247 | −6% | 4, 97 | 4.0 s | **47 ms** |
| gpu-util-0.85 | 16 | 179 | **−32%** | 4, 97 | 15.2 s | 486 ms |
| **fp8-kv** | >64 | **365+** | **+39%** | 4, 98 | 4.1 s | 487 ms |
| prefix-cache-off-shared | 16 | 281 | (961-tok prompt) | 4, 97 | 3.8 s | 488 ms |
| **prefix-cache-on-shared** | >64 | **1,183** | **4.2× off** | **64, 1,183** | 0.15 s | 44 ms |

## Correlations

### 1. Estimate vs measured: the model holds (baseline)
| | Estimate (`make estimate IN=1058 OUT=256`) | Measured |
|---|---|---|
| TTFT c=1 | 0.28 s | 0.286 s |
| ITL p50 c=1 | 35 ms | 34.5 ms |
| Prefill speed | ~3,800 tok/s | 1,058 / 0.286 = 3,700 tok/s |
| agg tok/s c=4 | ~95 | 96.5 |
| Knee | ~15 | 16 |
| Ceiling | ~270 | 262 |
| ITL p99 past knee | ~310 ms | **490 ms** (model underestimates, see §4) |

Everything except the ITL p99 stall size is within ~5%.

### 2. Plateau throughput follows KV cache size
| Experiment | KV cache (approx.) | Requests that fit (÷ 1,316 tok) | Plateau tok/s |
|---|---|---|---|
| gpu-util-0.85 | ~10k tokens | ~8 | 179 |
| baseline | ~20k | ~15 | 262 |
| max-seqs-32 | ~25k (smaller CUDA-graph/activation reservation) | ~19 | 280 |
| fp8-kv | ~40k (1 byte/value) | ~30 | 365+ (not reached) |

More KV → more requests in flight → more throughput, almost linearly until ~15 running, with diminishing returns above (per-request KV reads add to every step). Confirm exact KV sizes with `make loginfo LOG=results/logs/<tag>.log`.

### 3. Batching is nearly free: per-user speed falls slowly
Baseline, below the knee:

| c | ITL p50 | per-user tok/s | agg tok/s |
|---|---|---|---|
| 1 | 34.5 ms | 28.2 | 28 |
| 4 | 37.5 ms | 24.1 | 97 (3.4×) |
| 16 | 43.0 ms | 16.9 | 240 (8.5×) |

16× the users cost each user ~25% ITL (each step also reads 16 requests' KV, ~2.8 GB extra). Prefix-cache-on shows the same trend further out: ITL p50 53 ms at 64 running.

### 4. ITL p99 = prefill stalls, controlled by how prefill is scheduled
- Below ~4 running, stalls are < 1% of gaps → ITL p99 ≈ p50 (~39 ms).
- At c ≥ 16, p99 jumps to **~490 ms**: one 2,048-token prefill chunk (≈ 2 prompts × 1,058 tokens ÷ 3,800 tok/s ≈ 0.55 s) pauses every running request. This is why the estimate (one prompt, ~0.3 s) undershoots.
- **batched-tokens-16384**: ITL p99 68 ms, but TTFT p50 doubles (1.75 → 3.8 s at c=16) and throughput −7%. Many prompts are prefilled in one big step: decode is interrupted *less often* (stalls fall below p99), but every new request waits for the whole big step.
- **no-chunked-prefill**: ITL p99 47 ms at c=16, 170 ms at c≥32, throughput −6%.
- `batched-tokens-2048` = baseline exactly, so **2048 is the effective default** here.

### 5. TTFT p99 at c=16 is mostly a test artefact
Each level starts all `c` requests at once. At c=16 the 16th request waits for 15 prefills: 16 × 0.28 s ≈ **4.5 s** vs measured **4.1 s**. With only 32 requests per level, half the samples come from this opening burst, and p99 of 32 samples is the max. TTFT p50 (1.75 s) is closer to steady state. So "usable = c=4" is pessimistic; steady-state capacity at the 2 s TTFT target is likely between 4 and 16.

### 6. Past the knee, extra load only becomes queue time
Baseline: c=16 → 32 → 64 gives 240 → 262 → 262 tok/s, while TTFT p50 goes 1.75 → 17 → 48 s. E2E ≈ TTFT + 255 × 43 ms ≈ TTFT + 11 s, so below the knee decode dominates request time; above it, queueing does.

### 7. Prefix caching is the largest single lever (when prompts share a prefix)
Identical 961-token prompts: TTFT 44 ms (cache hit) vs 271 ms at c=1; at c=64 TTFT p99 0.33 s and 1,183 tok/s vs 264 tok/s off. Prefill is skipped *and* the shared prefix's KV is stored once, so the KV limit effectively disappears. Real gains depend on how much of real prompts is shared (system prompt, RAG context).

## Conclusions for this workload (1k in / 256 out, Qwen3-8B, one A10G)
- **Capacity:** ~16 concurrent requests, ~260 tok/s, limited by **KV cache capacity**, not compute (no preemptions; prediction matched).
- **Best throughput knob:** `--kv-cache-dtype fp8` (+39%, knee not yet found). Output quality with fp8 KV was **not** checked.
- **No-op knobs here:** `--max-num-seqs 256`, `--max-num-batched-tokens 2048` (already defaults).
- **Latency (ITL p99) knob:** prefill scheduling (`batched-tokens`, chunking), trading TTFT for smooth streaming.
- **Don't lower `gpu-memory-utilization`** without a reason: −32% throughput.

## Missed / data quality
1. **Old invalid rows still in the CSV**: the first 25 rows (`baseline`, `max-seqs-32` ×2, `max-seqs-256`, `batched-tokens-2048` with `input_tokens` 1024 = old word-count prompts). `plot.py` keeps the latest row per tag/level so the plot is correct, but remove them before sharing the CSV.
2. **Prefix tests used 961-token prompts** (no nonce): compare `prefix-cache-on` only to `prefix-cache-off-shared`, not to baseline.
3. **fp8-kv knee not found**: levels stop at 64.
4. **Small samples**: 32 requests per level, so p99 = max and the start-of-level burst dominates TTFT p99 (§5).
5. **No E2E percentiles in the CSV**: plot uses `output_tokens / per_req_tok_s` as an approximation.
6. **No smaller prefill chunks tested** (512 / 1024), the most likely way to meet ITL p99 ≤ 100 ms at c=16 without the TTFT cost of 16384.
7. **No output-quality check** for fp8 KV.
8. **GPU-side evidence not captured** in this write-up (DCGM: bandwidth vs compute in use per level). Add from the Grafana dashboard on the next run.

## Next tests
1. `fp8-kv` with `--concurrency 16 32 64 96 128` to find its knee.
2. Baseline + fp8-kv with `--requests 128` (steady-state TTFT p99).
3. `--max-num-batched-tokens 512` and `1024` (ITL p99 vs TTFT trade-off).
4. Combine winners: `fp8-kv` + `max-seqs-32` + best chunk size.
5. Quick quality check of fp8 KV vs bf16 on a small eval set.
