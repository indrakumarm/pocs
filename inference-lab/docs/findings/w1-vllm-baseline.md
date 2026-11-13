# w1-vllm-baseline

**Tool:** bench.py · **CSV:** `results/week3.csv`

## Setup
- Model / engine: Qwen/Qwen3-8B on `vllm/vllm-openai:latest` via `make up` ([docker-compose.yml](../../docker-compose.yml))
- Server flags: `--max-model-len 8192 --gpu-memory-utilization 0.90 --enable-prefix-caching`
- Bench: `make bench TAG=w1-vllm-baseline` → `--concurrency 1 2 4 8 16 32`, defaults `--input-tokens 512 --output-tokens 256 --requests 32`
- GPU / instance: _fill in_

## Results
| conc | requests | agg tok/s | per-req tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 27 | 27.2 | 0.55 s | 0.55 s | 34.8 ms | 35.5 ms |
| 2 | 32 | 49 | 24.3 | 0.99 s | 1.08 s | 37.3 ms | 38.0 ms |
| 4 | 32 | 87 | 21.7 | 1.53 s | 1.99 s | 38.6 ms | 39.8 ms |
| 8 | 32 | **141** | 17.8 | 1.57 s | 3.97 s | 41.6 ms | **504 ms** |
| 16 | 32 | 137 | 10.8 | **16.1 s** | 19.9 s | 44.0 ms | 506 ms |
| 32 | 64 | 137 | 6.6 | **35.3 s** | 49.7 s | 44.2 ms | 506 ms |

## Interpretation
- **Knee between c=4 and c=8.** Aggregate throughput peaks at ~141 tok/s at c=8, then plateaus (~137). Past saturation throughput levels off rather than dropping; the extra load turns into queueing.
- **Server-side queueing above c=8.** TTFT is measured after the client semaphore, so 16–35 s TTFT is time spent waiting **inside vLLM**. The server appears to run only ~8 sequences at once.
- **Decode stalls from c=8.** ITL p50 barely moves (35 → 44 ms) but p99 jumps to ~500 ms: periodic long pauses, consistent with long prefills blocking decode steps or with preemption/recompute.
- **Single-stream decode** ~35 ms/token (~28 tok/s), memory-bandwidth bound. Compare with roofline: bandwidth / model bytes (~16 GB bf16).
- **Usable capacity** (TTFT p99 < 2 s, ITL p99 < 100 ms): **c=4**.

### Cause of the ~8-sequence ceiling (confirmed by [week3-sweep-1](week3-sweep-1.md))
The KV cache is full. The week 3 logs show ~3.7 tokens per prompt word and a KV cache of 20,224 tokens (`gpu-memory-utilization 0.90`). So 512 words ≈ 1,900 prompt tokens, about 2,150 tokens per request with output, and ~9 requests fit.

### Caveats on these numbers
- Same prompts reused across levels with prefix caching on, so TTFT at c≥2 may be optimistic ([bench.md history](../bench.md#history)).
- 32 requests per level is small; p99 ≈ max.

## Next tests
1. Confirm the ceiling: check the startup log for KV cache size / max concurrency; during a c=16 run, watch `Running`/`Waiting` and KV usage % (`make logs`, `make metrics`); grep for preemptions.
2. Measure the real prompt length with `/tokenize`.
3. Pin down the knee: `--concurrency 4 6 8 10 12 16 --requests 64`.
4. `make sweep3`, in particular `fp8-kv` and `gpu-util-0.6` (KV capacity → should move the plateau) and `batched-tokens-2048` / `no-chunked-prefill` (ITL p99 spikes).
5. Vary request shape: prefill-heavy (`--input-tokens 2048 --output-tokens 128`) vs decode-heavy (`--input-tokens 128 --output-tokens 1024`).
