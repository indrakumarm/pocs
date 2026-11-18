# Docs

How the test tooling works and what each run showed.

| Doc | What it covers |
|---|---|
| [concepts.md](concepts.md) | **Read first.** Mental model: GPU memory budget, KV cache, prefill vs decode, why batching helps, the knee, glossary |
| [capacity.md](capacity.md) | **Then this.** 6-step estimate of knee/throughput/latency for any model + GPU, and which log line confirms each step |
| [bench.md](bench.md) | `scripts/bench.py`: load model, prompt generation, metrics, caveats |
| [sweep.md](sweep.md) | `scripts/sweep.py`: one-knob-per-experiment runner around `bench.py` |
| [findings/](findings/) | One file per test run: setup, results, interpretation, next steps |
| [Week 5](../weeks/week5/README.md) | Two-host replica scaling, streaming routing, mixed/fixed-rate load tests, and per-replica evidence |
| [Week 6](../weeks/week6/README.md) | Same-model TP=1 versus TP=2 versus two replicas on one host; matched workloads, GPU topology and server evidence |

## Adding a new test set
1. Run it (`make bench TAG=...` or `make sweepN`).
2. Copy [findings/_template.md](findings/_template.md) to `findings/<tag>.md` and fill it in.
3. Add a row to the findings index below.

## Findings index
| Tag | Tool | Summary |
|---|---|---|
| [w1-vllm-baseline](findings/w1-vllm-baseline.md) | bench.py | Qwen3-8B, vLLM defaults. Throughput plateaus ~140 tok/s at c=8; TTFT explodes beyond that (server-side queueing). |
| [week3-sweep-1](findings/week3-sweep-1.md) | sweep.py | KV cache fits only 4–6 requests (prompts are ~3.8k tokens, not 1,024). Knob comparisons confounded by unequal KV size; gpu-util-0.6 crashed (weights don't fit). |
| [week3-sweep-2](findings/week3-sweep-2.md) | sweep.py | First valid sweep. Knee c=16 / ~262 tok/s, matching the estimate within 5%; KV-bound. fp8-kv +39%, prefix caching 4.2× on shared prompts; ITL p99 ~490 ms from prefill stalls; usable at SLO c=4. |
| [week4](findings/week4.md) | sweep.py | Weight quantization improves per-user decode speed and enlarges KV pool; AWQ reaches 566 tok/s at c=64. SGLang FP8 smooths ITL p99; BF16 SGLang startup failed. Combined FP8-weight/KV tests added. |
| [week5](findings/week5.md) | replicas.py | Two replicas reached ~526 uniform tok/s at c=64; least-in-flight outperformed round-robin on mixed throughput at c=16/32 in two clean repeats. Tail latency and intermittent disconnects remain concerns. |
| [week6](findings/week6.md) | topologies.py | A10G topology comparison: uniform favors replicas at c=32; first mixed seed17/reverse run favors TP=2 at c=16/32. Includes methodology, plot walkthrough, KV/preemption evidence, decision guide and remaining repeats. |
