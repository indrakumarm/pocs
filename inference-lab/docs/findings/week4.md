# Week 4: weight quantization, speculation, and engine comparison

**Tool:** sweep.py · **CSV:** `results/week4.csv` · **Logs:** `results/logs/vllm-*.log`, `results/logs/sglang-*.log`

## Setup
- GPU: A10G, 22.06 GiB visible. Model family: Qwen3-8B. vLLM and SGLang runs use the same benchmark client.
- Load: target 1,024 input / 256 output tokens; concurrency 1, 4, 16, 32, 64. CSV measured input: 1,057–1,073 tokens (template and nonce included).
- 32 requests per level (64 at c=32; 128 at c=64). Fixed-count, closed-loop load; p99 at c=16 is the max of 32 request samples and includes the initial burst.
- vLLM's default `--gpu-memory-utilization` was 0.92 in the logs. The SGLang BF16 run failed before benchmarking; it is not in the CSV.

## Results
Throughput is aggregate generated tokens/s. "At c=64" is the highest measured result, not necessarily a true saturation ceiling.

| Engine / format | c=1 ITL p50 | c=1 TTFT p50 | Aggregate tok/s c=16 / 32 / 64 | c=64 ITL p50 / p99 | KV pool tokens |
|---|---:|---:|---:|---:|---:|
| vLLM BF16 | 34.4 ms | 286 ms | 272 / 257 / 279 | 45 / 499 ms | 23,440 |
| vLLM FP8 dynamic | 19.1 ms | 307 ms | 385 / 483 / 457 | 44 / 589 ms | 70,176 |
| vLLM FP8 prequant | 18.8 ms | 308 ms | 382 / 480 / 459 | 44 / 595 ms | 69,344 |
| vLLM AWQ int4 | 12.2 ms | 273 ms | 474 / 559 / 566 | 46 / 560 ms | 92,640 |
| vLLM n-gram speculation | 60.7 ms | 286 ms | 212 / 215 / 223 | 60 / 500 ms | 22,496 |
| SGLang FP8 | 18.7 ms | 337 ms | 384 / 419 / 474 | 32 / 46 ms | 41,317 (SGLang pool) |

SGLang BF16 has no result: startup failed because its default `--mem-fraction-static=0.684` left no KV cache; the log says the minimum viable value was 0.703.

## What the experiment says

### Weight quantization improves both single-user speed and capacity
At c=1, ITL p50 tracks weight size: 34.4 ms BF16, ~19 ms FP8 and 12.2 ms AWQ. That is consistent with decode spending much of its time moving weights. Relative to BF16, per-request generation speed rises from 28 tok/s to 49–50 tok/s with FP8 and 76 tok/s with AWQ.

The quantized models also leave more GPU memory for KV cache. The startup logs report 23.4k KV tokens for BF16, ~70k for FP8, and 92.6k for AWQ. So their higher aggregate throughput combines **faster weight reads and a larger KV pool**; this sweep does not isolate those effects. At c=64, AWQ reaches 566 tok/s, about 2.0× the BF16 run's 279 tok/s, but is still slightly increasing (559 at c=32 to 566 at c=64), so its exact knee was not measured.

### Once the batch is large, KV reads offset smaller weights
At c=64, ITL p50 converges to 44–46 ms for BF16, FP8, and AWQ. The requests' KV data grows with active batch size and is read at each decode step. This makes the benefit of smaller weights less visible at high concurrency. The configuration has a finite request count, so these points are a measured operating point, not proof of the maximum sustainable rate.

### FP8 dynamic and prequantized look equivalent for speed here
Their throughput, TTFT and ITL are within a few percent at all tested levels. This says the two approaches performed similarly on this setup; it does not establish that their output quality is equivalent. No quality evaluation was run.

### N-gram speculative decoding was a poor match for the synthetic prompts
It is slower than BF16: 26.4 vs 28.2 tok/s per request at c=1 and 223 vs 279 aggregate tok/s at c=64. The filler prompts do not represent a workload with strong prompt-to-output copying, where prompt-lookup speculation could help. Also, the benchmark's ITL is the gap between streamed content chunks; speculation can emit multiple tokens per chunk, so its ITL is not directly comparable to ordinary one-token decode.

### SGLang FP8 prioritizes smooth token delivery
SGLang FP8 has almost the same aggregate throughput as vLLM dynamic FP8 at c=16 (384 vs 385 tok/s), but its measured ITL p99 is 29 ms instead of 526 ms. At c=64, ITL p99 is 46 ms versus 589 ms. In exchange, SGLang TTFT p50 is higher: 2.46 s vs 1.85 s at c=16, and 20.9 s vs 5.28 s at c=64. This is a latency trade-off: smoother output gaps, longer waits for the first token. The results are specific to this workload and these engine settings.

## Week 3 comparison: important confound
The c=1 and c=4 BF16 results match across weeks almost exactly, but the startup logs show different KV pool sizes:

| Run | GPU utilization setting | KV pool |
|---|---:|---:|
| Week 3 baseline | 0.90 | 20,224 tokens |
| Week 4 vLLM BF16 | default 0.92 | 23,440 tokens |

That is a 16% larger KV pool in week 4. The BF16 c=16/32/64 results therefore are not a strict same-memory comparison with week 3. For cleaner cross-week comparisons, pin `--gpu-memory-utilization` explicitly in the next config and confirm each run's startup KV token count.

Week 3's FP8 **KV cache** result is a different lever from week 4's FP8 **weights**. Week 3 KV FP8 kept BF16 weights and doubled KV tokens; week 4 FP8 weights reduced weight memory and decode traffic, also allowing a larger KV pool. The planned combined tests will show whether the two gains compose.

## Follow-up experiments added
The week 4 config now includes:
- `vllm-fp8-dynamic-fp8kv`: dynamically quantized FP8 weights + FP8 KV cache.
- `vllm-awq-fp8kv`: AWQ int4 weights + FP8 KV cache.
- `sglang-bf16`: retry with `--mem-fraction-static 0.90` to provide a KV allocation; compare its log/startup and metrics with SGLang FP8.

These are configured but have **not** been run yet. Check logs for KV token counts and the CSV for completed rows before concluding that quantization gains multiply.

## Remaining caveats and next actions
1. Run the three new week 4 experiments; compare KV pool sizes, throughput, TTFT, ITL p50 and p99.
2. Add a matched BF16 run with explicit `--gpu-memory-utilization 0.92` if a strict comparison to week 4's quantized runs is needed. Week 3 used 0.90.
3. Increase `--requests` (e.g. 128) for a more stable TTFT p99; the present 32-request levels are short and front-loaded.
4. Test FP8/AWQ output quality on a representative evaluation set; throughput alone does not establish quality.
5. Test speculation on prompts where generated output repeats prompt text, or compare with a draft-model method.
6. Keep workload lengths and engine configuration fixed when comparing engines. The CSV's input count includes the chat template and per-request nonce.
