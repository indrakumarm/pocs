# Week 5: two vLLM replicas and routing

**Date:** 2026-11-12 · **Tool:** `scripts/replicas.py` · **Results:** [results/week5](../../results/week5/)

## Setup and evidence

- Two independently hosted vLLM replicas serving `Qwen/Qwen3-8B`, one complete
  model per GPU, behind a router on the benchmark machine. This is data
  replication, not tensor parallelism. The captured
  [replica A log](../../results/aws/replica-a/logs/week5-vllm-replica-a.log)
  reports `--max-model-len 8192`, `--gpu-memory-utilization 0.9`, prefix caching
  enabled, and 20,224 KV-cache tokens. Replica B's corresponding startup log,
  exact image digest, GPU identification, and model revision were not found in
  the collected artifacts; verify them before claiming a strictly matched setup.
- **Uniform:** target 1,024 input / 256 output tokens per request. **Mixed:** 75%
  target 256/128 and 25% 4,096/512, shuffled deterministically. Prompts have
  unique leading nonces; this is not a shared-prefix-cache test.
- Fixed-count, closed-loop concurrency is **total client concurrency across the
  router**, not per replica. Most levels used 128 requests; c=128 used 256.
  Throughput is successful output tokens divided by full level wall time, including
  the final drain. TTFT and streaming chunk-gap p99 describe successful requests
  only. p99 from 128 samples is sensitive to individual outliers and burst timing.
- Runs: [single A](../../results/week5/20261112T054345Z-b55a7f/),
  [uniform two-replica](../../results/week5/20261112T064727Z-fb4dd8/),
  [mixed seed 17, initial](../../results/week5/20261112T080845Z-28e2fc/),
  [mixed seed 17, repeat](../../results/week5/20261112T095907Z-2ea580/), and
  [mixed seed 18](../../results/week5/20261112T133755Z-042912/). Config files
  contain the full experiment list even when the CLI selects only some experiments;
  actual CSV rows show what ran. The initial single-A run was interrupted before
  a single-B baseline or the full matrix completed.

## Uniform workload: measured scaling

All values below are aggregate output tok/s; TTFT p99 is seconds. The single-A
baseline and two-replica policies were measured in **separate runs**, so the ratios
are indicative, not a fully controlled paired scaling claim.

| Total c | Single A tok/s | RR tok/s / TTFT p99 | Least-in-flight tok/s / TTFT p99 |
|---:|---:|---:|---:|
| 1 | 27.4 | 27.4 / 0.64 | 27.4 / 0.65 |
| 4 | 94.8 | 100.9 / 0.87 | 100.8 / 0.90 |
| 16 | 251.3 | 335.1 / 2.33 | 335.2 / 2.36 |
| 32 | 261.7 | 486.2 / 4.41 | 486.5 / 4.41 |
| 64 | 262.1 | 525.8 / 20.35 | 525.7 / 20.39 |
| 128 | 261.5 | 522.6 / 51.43 (1 failure) | 523.8 / 51.33 |

At c=64, RR's 525.8 tok/s is approximately **2.01 times** single A's
262.1 tok/s at the same total concurrency. This is an observed operating-point
ratio, not a guarantee of linear scaling or a maximum sustainable rate.
Both policies routed exactly 64 requests to each replica at c=64 and produced
nearly identical throughput. At c=128, RR had one 502, so that point is not a
clean comparison; least-in-flight completed all 256 requests.

For the guide's illustrative **TTFT p99 <= 2 s and chunk-gap p99 <= 100 ms**
SLO, the highest tested clean two-replica uniform point is c=4 for both policies:
RR ~100.9 tok/s and least-in-flight ~100.8 tok/s. At c=16 TTFT p99 is already
above 2 s; do not use c=64's peak as the SLO-compliant capacity. The gap metric
is between streamed content chunks, not necessarily one-token decode latency.

## Mixed workload: routing comparison

All rows below completed 128/128 requests. The two clean repeats used different
seeds **and** different experiment order, so neither effect is independently
isolated. The seed 17 repeat ran RR first; seed 18 ran least-in-flight first.

| Seed | Total c | RR tok/s | Least tok/s | Least vs RR | RR / least TTFT p99 |
|---:|---:|---:|---:|---:|---:|
| 17 | 16 | 184.2 | 230.7 | +25.2% | 26.70 / 26.79 s |
| 17 | 32 | 184.3 | 221.9 | +20.4% | 48.61 / 41.84 s |
| 18 | 16 | 220.1 | 247.2 | +12.3% | 19.03 / 12.34 s |
| 18 | 32 | 237.4 | 251.2 | +5.8% | 30.11 / 31.09 s |

Seed 17 at c=64 measured 184.1 / 213.4 tok/s (RR / least) with TTFT p99
81.15 / 67.04 s. At c=4, seed 17 measured 94.6 / 94.7 tok/s and TTFT p99
1.19 / 2.11 s. For the illustrative SLO above, c=4 is the highest tested clean
point for RR (94.6 tok/s); least-in-flight's c=4 misses TTFT p99 and its c=1
passes (27.3 tok/s). This is a finite-sample observation, not a precise capacity
boundary; intermediate levels and repeated runs were not tested.

Least-in-flight was higher in throughput at c=16 and c=32 in both clean repeats,
but the advantage varied substantially and did not always improve TTFT p99.
For seed 18 at c=16, RR sent 64/64 requests to A/B; least sent 75/53. The
before/after vLLM counters show RR processed approximately 93,702 / 70,696
prompt tokens on A/B versus 85,800 / 78,598 with least-in-flight. These are
**processed token counters**, not exact prompt-shape assignments, and may be
affected by cache accounting and other traffic. They indicate uneven work but do
not by themselves prove the reason for the speed difference.

The same seed 18 snapshots recorded preemption-counter deltas of **3 / 0**
on A/B for RR c=16 and **0 / 0** for least-in-flight c=16. At c=32 the deltas
were RR **3 / 4** and least **0 / 12**. These counters are supporting signals,
not proof that routing caused the measured throughput change. They are
server-wide and assume no unrelated concurrent traffic or counter resets.
Before/after snapshots do not capture peak queue depth, KV usage, GPU
utilization, or when preemptions occurred during a level.

## Reliability and cross-week caveats

- The initial seed 17 mixed run had one 502 each at c=4, 16, and 32 under RR.
  The two clean repeats are preferable for policy comparison.
- In the uniform run, RR c=128 had one 502. The interrupted single-A open-loop
  run at 2 requests/s recorded two 502s and six 503s after the backend was
  marked unhealthy. Periodic health checks also reported `Server disconnected`
  on both replicas in some uniform levels that otherwise completed successfully.
  A user check reported no container restart or OOM; this does not establish
  the connection failure's cause. Snapshots retain the last error, not a
  timestamped history. No retries were enabled.
- Week 3/4 used fixed ~1,024-in/256-out requests; mixed Week 5 uses different
  prompt/output lengths and cannot be compared directly by output tok/s.
  Week 5's uniform single-A result is roughly consistent with Week 3 throughput
  (~262 tok/s at c=64), but the runs have different clients/network paths,
  request counts, potentially versions/settings, and prompt token counts.
  Week 4 BF16 used a different GPU memory utilization and KV pool. A direct
  laptop-versus-localhost test is needed to assess Week 5's higher observed TTFT.
- `vs_best_single` is empty in the two-replica-only summaries because no clean
  single baseline was included **in those runs**. There is no matched single-B
  or mixed single-replica baseline. The user-facing router counts only its own
  outstanding requests; it does not route using vLLM queue depth or token cost.

## Conclusion and follow-ups

The Week 5 objectives—running two copies, measuring uniform scaling, comparing
two routing policies on mixed workloads, and inspecting per-replica evidence—are
complete. **The production-capacity and reliability questions remain open.**

1. Investigate connection disconnects with timestamped router/server logs and a
   controlled direct-to-vLLM versus router/connection-reuse experiment.
2. Run matched single-A, single-B, RR, and least-in-flight baselines together for
   both uniform and mixed workloads. Pin both GPU types, image digests, model
   revisions, server flags and KV pools; keep client location fixed.
3. Repeat several seeds and both policy orders independently. Record request shape
   per chosen replica, and scrape queue/KV/GPU metrics continuously to explain
   latency, preemption, and load distribution.
4. Compare a production load-aware router such as llm-d against this harness;
   use a separate shared-prefix workload to test cache-aware routing. No
   production-router or cache-affinity results were measured in Week 5.
