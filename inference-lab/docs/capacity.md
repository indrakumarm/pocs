# Capacity in 6 steps: estimate, then confirm from logs

Why these formulas work: [concepts.md](concepts.md).

For any model + GPU, six numbers decide throughput and latency. Compute them **before** running anything (`make estimate`), then confirm them from the vLLM log (`make loginfo`) and the benchmark.

```bash
make estimate GPU=A10G IN=1024 OUT=256            # paper estimate (MODEL=... to change model)
make loginfo                                        # read the running server's log
make loginfo LOG=results/logs/baseline.log          # or a saved sweep log
```

## The six steps

| # | Question | Formula | Where it shows up in the vLLM log |
|---|---|---|---|
| 1 | How big are the weights? | params × bytes per param (bf16 = 2) | `Model loading took 15.27 GiB` |
| 2 | How much memory does 1 token of context cost? | 2 × layers × kv_heads × head_dim × bytes | not printed; = step 3 GiB ÷ step 3 tokens |
| 3 | How much memory is left for context (KV cache)? | gpu_util × VRAM − weights − ~1.8 GiB | `Available KV cache memory: 2.78 GiB` and `GPU KV cache size: 20,224 tokens` |
| 4 | **How many requests fit at once? (the knee)** | KV tokens ÷ (input + output tokens) | under load: `Running: 4 reqs, Waiting: 12 reqs, GPU KV cache usage: 81%` |
| 5 | How fast is one token of output (ITL)? | weight GB ÷ memory bandwidth GB/s ÷ 0.75 | bench `itl_p50_ms` at c=1 |
| 6 | How fast is the prompt read (TTFT)? | prompt tokens ÷ (TFLOPS × 0.5 ÷ (2 × params)) | bench `ttft_p50_ms` at c=1 |

Then:
- **Throughput ceiling** ≈ `knee × output ÷ (knee × prefill time + output × ITL)`. Past the knee it stays flat.
- **TTFT past the knee** grows with the queue: each extra "knee-sized batch" of waiting requests adds one full request duration.
- **ITL p99** jumps to ≈ one prefill chunk (`min(input, 2048) ÷ prefill speed`, ~0.5 s on A10G) once several requests run together, because a new request's prefill pauses everyone's decode.

### Where model/GPU numbers come from
- Model: `config.json` on Hugging Face → `num_hidden_layers`, `num_key_value_heads`, `head_dim` (or hidden_size ÷ num_attention_heads). `estimate.py` reads it for you.
- GPU: datasheet memory, bandwidth, bf16 TFLOPS. Table in [scripts/estimate.py](../scripts/estimate.py); override with `--vram-gib --bw-gbs --tflops`.

### The one log caveat
vLLM's `Maximum concurrency for 8,192 tokens per request: 2.47x` assumes every request uses the full `--max-model-len`. Ignore it: use step 4 with **your** input + output length.

## Worked example: Qwen3-8B on A10G (22 GiB, 600 GB/s)

| Step | Estimate | Log / measured |
|---|---|---|
| 1 weights | 15.3 GiB | 15.27 GiB |
| 2 KV per token | 144 KiB | 2.78 GiB ÷ 20,224 = 144 KiB |
| 3 KV budget | 2.74 GiB ≈ 20.0k tokens | 2.78 GiB = 20,224 tokens |
| 4 knee (3,800 in + 256 out) | 20.0k ÷ 4,056 = **4** | `Running: 4`, `Waiting: 12…60`, KV ~80–99% |
| 5 ITL c=1 | 35 ms | 35.8 ms |
| 6 TTFT c=1 | 1.0 s | 1.05 s |
| ceiling | 74 tok/s | 71 tok/s |
| TTFT at c=64 | 211 s | 216 s |

Same GPU with 1,900-token prompts: knee 9, ceiling ~150 tok/s (measured 141 at c=8).

## What moves which number
| To change… | Lever | Step |
|---|---|---|
| Knee (more requests at once) | fp8 KV (`--kv-cache-dtype fp8`), higher `--gpu-memory-utilization`, quantized weights, bigger GPU, shorter requests | 1, 3, 4 |
| ITL (per-token speed) | quantized weights (fewer bytes to read), higher-bandwidth GPU | 5 |
| TTFT | faster GPU (TFLOPS), prefix caching for shared prompts, staying below the knee | 6 |
| ITL p99 stalls | smaller `--max-num-batched-tokens` (512/1024) | 6 |
| Nothing, while KV-bound | `--max-num-seqs`, larger `--max-num-batched-tokens` | – |

## Workflow for "model X on GPU Y"
1. `make estimate MODEL=X GPU=Y IN=… OUT=…`. If it says it doesn't fit, pick a bigger GPU or quantize before renting anything.
2. Start the server, run `make bench TAG=…` with the same IN/OUT.
3. `make loginfo`: check steps 1–4 match, and peak `Running` = predicted knee.
4. If estimate and measurement disagree by >20%, something in the setup is off (the wrong prompt length, a flag dropped, etc.). Find it before tuning.
5. Tune only the lever for the limit you hit (table above).
