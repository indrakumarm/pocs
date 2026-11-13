# week3 sweep (first run)

**Tool:** sweep.py · **CSV:** `results/week3.csv` · **Logs:** `results/logs/*.log`

## Setup
- GPU: 22.06 GiB visible (A10G-class). Qwen3-8B bf16 weights: 15.3 GiB.
- Bench: `--concurrency 1 4 16 32 64 --input-tokens 1024 --output-tokens 256`
- Ran: baseline, max-seqs-32 (twice; duplicate rows in CSV), max-seqs-256, batched-tokens-2048, gpu-util-0.6 (crashed).
- Not run: batched-tokens-16384, no-chunked-prefill, prefix-cache-on/off-shared (no log, no rows), fp8-kv (sweep aborted after the crash).

## Results (agg tok/s, TTFT p50, ITL p99)
| tag | KV cache (tokens) | Running at c≥16 | c=1 | c=4 | c=16 | c=64 | ITL p99 c≥4 |
|---|---|---|---|---|---|---|---|
| baseline (util 0.90) | 20,224 | 4 | 25 / 1.0 s | 71 / 2.1 s | 71 / 43 s | 71 / 216 s | ~505 ms |
| max-seqs-256 (util 0.92*) | 23,440 | 5 | 25 | 71 | 77 / 34 s | 80 / 180 s | ~545 ms |
| batched-tokens-2048 (util 0.92*) | 23,440 | 5 | 25 | 71 | 77 / 34 s | 80 / 180 s | ~545 ms |
| max-seqs-32 (util 0.92*) | 26,704 | 6 | 25 | 71 | 84 / 23 s | 88 / 159 s | ~548 ms |
| gpu-util-0.6 | -3.84 GiB | crashed | | | | | |

\* `args` in an experiment replaced `base.args`, dropping `--gpu-memory-utilization 0.90`, so vLLM used its default (0.92).

No preemptions in any log.

## What the logs show
1. **Real prompt ≈ 3,800 tokens, not 1,024.** At c=1 the log shows ~400 prompt tok/s with one request every ~10 s, and KV usage ~20% of 20,224 ≈ 4,050 tokens per request (prompt + 256 output). That's ~3.7 tokens per `wNNN` word. Confirms the old word-count bug ([bench.md history](../bench.md#history)).
2. **Plateau = KV cache full.** 20,224 / ~4,300 tokens per request ≈ 4.7, so only 4 requests run at once and the rest wait (`Running: 4, Waiting: 12…28`, KV ~80–90%). That's why throughput is flat at ~71 tok/s from c=4 onward and TTFT grows with queue length.
3. **The knob results are confounded by KV size.** max-seqs-32 "wins" (88 tok/s) only because it had the most KV cache: lower max-num-seqs → smaller activation and CUDA-graph profiling (1.32 vs 1.63 GiB, 0.07 vs 0.23 GiB) → 26.7k tokens → 6 running. Ranking follows KV size exactly: 4 → 5 → 6 running = 71 → 80 → 88 tok/s.
4. **max-seqs-256 and batched-tokens-2048 are identical**, which suggests 2048 is already the effective default for `--max-num-batched-tokens` here.
5. **ITL p99 ~500 ms = one prefill chunk.** Prefill runs at ~3,800 tok/s (TTFT 1.0 s for ~3.8k tokens), so a 2,048-token chunk takes ~0.5 s and stalls all decodes for that step. To shrink it, *lower* `--max-num-batched-tokens` (512/1024) rather than raising it.
6. **gpu-util-0.6 can't work on this GPU:** 0.6 × 22 GiB = 13.2 GiB < 15.3 GiB of weights → negative KV → `ValueError: No available memory for the cache blocks`. Expected, not a bug.

## Changes made after this run
- [scripts/sweep.py](../../scripts/sweep.py): a failed experiment is logged and skipped instead of aborting the sweep.
- [weeks/week3/config.yaml](../../weeks/week3/config.yaml): every experiment pins `--gpu-memory-utilization 0.90` so KV size stays constant; `gpu-util-0.6` → `gpu-util-0.85`.

## Next tests
1. Fix the prompt length (count real tokens) or halve `--input-tokens`; otherwise every test is KV-bound at 4–5 requests.
2. Delete or rename the old rows (or use a new `out:` file), then re-run `make sweep3`.
3. Add `--max-num-batched-tokens 512` / `1024` to test the ITL p99 explanation.
4. `fp8-kv` should roughly double KV tokens → ~9 running → expect ~2× plateau throughput.
