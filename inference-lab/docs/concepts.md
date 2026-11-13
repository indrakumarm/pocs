# Concepts: how LLM serving uses the GPU

The mental model behind [capacity.md](capacity.md). Example numbers: Qwen3-8B (bf16) on an A10G (24 GB card, 600 GB/s, 125 TFLOPS).

**In one line:** weights decide whether the model fits. What's left is the KV cache; its size ÷ tokens per request sets how many requests run at once (the knee). Memory bandwidth sets ITL, TFLOPS set TTFT, and batching turns idle compute into throughput until the KV cache runs out.

**Contents**
1. [Where the GPU memory goes](#1-where-the-gpu-memory-goes)
2. [KV cache: memory to remember each token](#2-kv-cache-memory-to-remember-each-token)
3. [Two phases: prefill and decode](#3-two-phases-prefill-and-decode)
4. [Why batching raises throughput](#4-why-batching-raises-throughput)
5. [The knee](#5-the-knee)
6. [Glossary](#6-glossary)

---

## 1. Where the GPU memory goes

| Item | Amount |
|---|---|
| Card label 24 GB (decimal) | ≈ **22.06 GiB** visible |
| vLLM may use (`--gpu-memory-utilization 0.9`) | 0.9 × 22.06 ≈ **19.85 GiB** |
| Weights: 8.19B params × 2 bytes = 16.4 GB | **−15.27 GiB** (fixed) |
| Working memory: forward-pass scratch, CUDA graphs, runtime | **−~1.8 GiB** (fixed) |
| **KV cache pool** | **≈ 2.8 GiB**, shared by all running requests |

```
22 GiB GPU
├── 15.3 GiB  weights
├──  1.8 GiB  working memory
├──  2.8 GiB  KV cache pool  ← [req1 570 MB][req2][req3][req4]  full
└──  2.2 GiB  headroom left out by util=0.9
```

- GB = 10⁹ bytes, GiB = 2³⁰ bytes (~7% bigger). vLLM logs use GiB.
- Bytes per parameter: bf16 = 2, fp8/int8 = 1, int4 = 0.5.
- Log lines: `Model loading took 15.27 GiB`, `Available KV cache memory: 2.78 GiB`.

## 2. KV cache: memory to remember each token

To produce the next token, the model "looks back" at every earlier token (attention). Recomputing all of them at every step would be far too slow, so for each token, at every layer, the model stores a **K**ey and a **V**alue vector. That stored memory is the KV cache.

- It is per token **in the conversation** (prompt + generated so far), not memory spent generating a token.
- It grows by one token per output step and is freed only when the request finishes.

$$
\text{KV per token} = 2_{(K,V)} \times 36_{\text{layers}} \times 8_{\text{kv heads}} \times 128_{\text{head dim}} \times 2_{\text{bytes}} = 147{,}456\text{ B} = 144\text{ KiB}
$$

Layers, KV heads and head dim come from the model's `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim`).

| Quantity | Value |
|---|---|
| One request: 3,800 prompt + 256 output = 4,056 tokens | × 144 KiB ≈ **570 MiB** |
| Pool capacity: 2.78 GiB ÷ 144 KiB | ≈ **20,224 tokens** |
| Requests that fit at once: 20,224 ÷ 4,056 | ≈ **4** |

**Checking it from the log:** vLLM prints the pool size and its token capacity, not the per-token size:
```
Available KV cache memory: 2.78 GiB
GPU KV cache size: 20,224 tokens
```
2.78 GiB ÷ 20,224 = 144 KiB, which confirms the formula.

## 3. Two phases: prefill and decode

| | Prefill (read the prompt) | Decode (write the answer) |
|---|---|---|
| Tokens per step | all prompt tokens at once | 1 per request |
| Why | the whole prompt is already known | token *n+1* needs token *n* first |
| Limited by | **compute** (TFLOPS) | **memory bandwidth** (GB/s) |
| Shows up as | TTFT | ITL |

```
Prefill (1 pass):   [t1 t2 t3 ... t3800] ──matmul──> KV for all 3800 + first output token
Decode (per step):  [t3801] ──> t3802 ──> t3803 ──> ...   (each waits for the previous)
```

Analogy: decode is writing a story, where each word depends on the one before. Prefill is reading pages you already have; many readers can work on different pages at once.

### Prefill: why the whole prompt goes through in one pass
"One by one" is only required when the next input depends on the previous output. Every prompt token is known up front, so nothing has to wait.

1. The prompt becomes a matrix: N tokens × 4,096 numbers, one row per token.
2. Each layer multiplies the **whole matrix** by its weights in one matrix multiplication, so all N rows are computed in parallel.
3. Attention uses a **causal mask**: token 5 sees only tokens 1–5. The result is the same as reading in order, but every position is computed at once.
4. Outputs: K and V for all N prompt tokens go into the KV cache, and the prediction at the last position is the **first output token**. That moment is TTFT.

**Why it's compute-bound:** the weights are read once but used for N tokens.

$$
\text{TTFT} \approx \frac{N \times 2 \times \text{params}}{\text{TFLOPS} \times 0.5} = \frac{3{,}800 \times 2 \times 8.2\text{B}}{125\text{T} \times 0.5} \approx 1.0\text{ s} \quad (\text{measured } 1.05)
$$

×0.5 because real prefill reaches about half the datasheet TFLOPS.

**Chunked prefill:** long prompts are split into chunks of `--max-num-batched-tokens` (2048 here). Each chunk takes ~0.5 s, and every running request's decode waits for it. So a new request arriving gives everyone else one ~500 ms gap. That is the **ITL p99 spike**. Smaller chunks mean smaller stalls but slower TTFT.

### Decode: why ITL ≈ weight bytes ÷ bandwidth
Each output step runs the whole model for one token, so the GPU must stream **all weights** from memory to the compute units. The math for one token is tiny; the time is the data movement.

$$
\text{ITL} \approx \frac{\text{bytes read per step}}{\text{bandwidth} \times 0.75} = \frac{15.1\text{ GB}}{600\text{ GB/s} \times 0.75} \approx 34\text{ ms} \quad (\text{measured } 35)
$$

- 15.1 GB, not 16.4: only one row of the input-embedding table is read per token.
- ×0.75 because real GPUs reach ~75% of datasheet bandwidth.

## 4. Why batching raises throughput

Every decode step has two costs:

| Work in one decode step | Time on A10G |
|---|---|
| **Move** 15.1 GB of weights from memory to compute units | **~25 ms** (600 GB/s) |
| **Math** for 1 token: 2 FLOPs × 8.2B params = 16 GFLOP | **~0.13 ms** (125 TFLOPS) |
| Math for 8 tokens | ~1 ms, still hidden inside the 25 ms move |

Moving the weights takes ~200× longer than the math, so at batch 1 the compute units are idle ~99% of the time. Once a weight arrives, it can be multiplied with request 1's token, request 2's token, request 3's token… before the next weight is loaded. The expensive load happens **once per step, shared by every running request**.

Like a bus: the trip costs about the same with 1 passenger or 30.

| Requests running | ITL (each user waits) | Throughput (all users) |
|---|---|---|
| 1 | 35 ms | 1 ÷ 0.035 ≈ **28 tok/s** |
| 4 | 39 ms | 4 ÷ 0.039 ≈ **100 tok/s** |
| 8 | 42 ms | 8 ÷ 0.042 ≈ **190 tok/s** |

Decode-only numbers; measured throughput is lower because prefills interrupt decode.

**Why ITL still creeps up:** weights are shared, but each request's **own KV cache** must also be read every step. 4 requests × ~4,000 tokens × 144 KiB ≈ 2.3 GB extra → (15.1 + 2.3) GB ÷ 450 GB/s ≈ 39 ms.

**When batching stops helping:**
1. **KV cache full** (our case): only 4–9 requests fit, so the batch can't grow and throughput goes flat.
2. **Math catches up with data movement:** the A10G does ~125 TFLOPS ÷ 600 GB/s ≈ 200 FLOPs per byte read. Each request adds ~1 FLOP per weight byte, so decode becomes compute-bound only around a batch of ~200.

## 5. The knee

The knee is the concurrency where throughput stops growing.

- **Below the knee:** every request gets a KV slot. Throughput rises with concurrency; latency barely changes.
- **At the knee:** the KV cache is full (`Running: 4`, KV usage ~80–100%).
- **Past the knee:** extra requests wait in a queue (`Waiting: 12…60`). Throughput stays **flat** (it does not drop) and TTFT grows with the length of the queue.

$$
\text{knee} \approx \frac{\text{KV pool tokens}}{\text{input} + \text{output tokens}}
$$

| To move it | Lever |
|---|---|
| More KV memory | `--kv-cache-dtype fp8`, higher `--gpu-memory-utilization`, quantized weights, bigger GPU |
| Smaller requests | shorter prompts or outputs |
| Does nothing while KV-bound | `--max-num-seqs`, larger `--max-num-batched-tokens` |

## 6. Glossary

| Term | Meaning |
|---|---|
| TTFT | Time to first token: queue wait + prefill |
| ITL | Inter-token latency: gap between output tokens (one decode step) |
| p50 / p99 | Median / 99th-percentile value |
| Throughput (agg tok/s) | Output tokens per second across all users |
| Concurrency | Requests in flight from the client |
| Running / Waiting | Requests vLLM is serving / has queued (log stats line) |
| KV cache | Stored Key/Value vectors for every token in every running request |
| Prefill | Processing the prompt in one parallel pass |
| Decode | Generating output one token per step |
| Chunked prefill | Splitting long prompts into `--max-num-batched-tokens` chunks |
| Memory-bound / compute-bound | Limited by data movement (GB/s) / by arithmetic (TFLOPS) |
| Preemption | vLLM evicting a running request when the KV cache overflows |
