# vLLM Performance Benchmarking — Test Plan & Design Matrix

**Purpose:** Systematic guide to benchmark vLLM across models, GPU configs, and workloads using `guidellm`. Use this as a matrix to plan runs, capture results, and draw conclusions.

---

## 1. Goals

1. Characterize **max sustainable throughput** (req/s, tokens/s) per (model × GPU × config).
2. Identify **latency SLO** operating points (TTFT, ITL, e2e latency at p50/p95/p99).
3. Find the **knee of the throughput‑vs‑latency curve** (optimal concurrency).
4. Compare **cost/performance** across GPU SKUs and tensor‑parallel configs.
5. Detect **regressions** across vLLM versions, quantization schemes, and kernels.

---

## 2. Key Metrics (what to record)

| Metric | Definition | Why it matters |
|---|---|---|
| **TTFT** | Time to first token (ms) | Interactive UX / chat |
| **ITL** | Inter-token latency (ms) | Streaming smoothness |
| **TPOT** | Time per output token (ms) | Same as ITL, normalized |
| **E2E Latency** | Full request time (s) | Batch / RAG use cases |
| **Throughput — req/s** | Completed requests/sec | Capacity planning |
| **Throughput — output tok/s** | Generated tokens/sec | Cost per token |
| **Throughput — total tok/s** | Prompt + gen tokens/sec | HW utilization |
| **Concurrency** | In-flight requests | Saturation indicator |
| **GPU util %** | `nvidia-smi dmon` | HW efficiency |
| **GPU mem** | Peak VRAM (GB) | Config feasibility |
| **KV cache hit rate** | vLLM `/metrics` | Prefix caching effectiveness |
| **Error rate** | Failed / total | Stability |

Record **p50, p95, p99** for all latency metrics.

---

## 3. Test Dimensions (the Matrix)

### 3.1 Model axis
| Model | Params | Quantization | Notes |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | FP16 / AWQ / GPTQ | Small baseline |
| Llama-3.1-8B-Instruct | 8B | FP16 / AWQ / FP8 | Common workload |
| Qwen2.5-32B-Instruct | 32B | AWQ / GPTQ | Mid-size |
| Llama-3.1-70B-Instruct | 70B | AWQ / FP8 | Large, needs TP |
| Mixtral-8x7B | 47B MoE | FP16 / AWQ | MoE behavior |

### 3.2 Hardware axis
| GPU | VRAM | Notes |
|---|---|---|
| A10G | 24 GB | Small models only |
| L4 | 24 GB | Cost-efficient |
| A100 40G / 80G | 40 / 80 GB | Standard |
| H100 80G | 80 GB | FP8 support |
| H200 141G | 141 GB | Large context |

### 3.3 Parallelism axis
- **TP (tensor parallel):** 1, 2, 4, 8
- **PP (pipeline parallel):** typically 1 (single node)
- **DP (data parallel):** replicas behind LB

### 3.4 Workload axis (prompt / output token shapes)
| Profile | Prompt | Output | Real-world analog |
|---|---|---|---|
| **Chat-short** | 128 | 128 | Interactive chat |
| **Chat-long** | 512 | 256 | Assistant Q&A |
| **RAG** | 2048 | 256 | Retrieval-augmented |
| **Summarization** | 4096 | 512 | Doc summary |
| **Long-context** | 16384 | 512 | Codebase / long docs |
| **Generation-heavy** | 128 | 2048 | Content generation |

### 3.5 Load axis
- **Sync** (concurrency=1) — best-case latency
- **Constant rate** — steady-state SLO check
- **Sweep** — full throughput/latency curve
- **Concurrent streams** — [1, 2, 4, 8, 16, 32, 64, 128]
- **Burst** — spike testing

---

## 4. Test Plan — Standard Suite

Run **per (model, GPU config)** combination:

### T1 — Baseline latency (sync)
```bash
guidellm run \
  --profile kind=synchronous \
  --constraint kind=max_duration,seconds=60 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256
```
**Captures:** min TTFT, min ITL, best-case latency.

### T2 — Max throughput
```bash
guidellm run \
  --profile kind=throughput \
  --constraint kind=max_duration,seconds=120
```
**Captures:** ceiling req/s and tokens/s.

### T3 — Sweep (curve)
```bash
guidellm run \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=60
```
**Captures:** throughput‑vs‑latency curve → find the knee.

### T4 — Concurrency scaling
```bash
guidellm run \
  --profile kind=concurrent,streams=[1,2,4,8,16,32,64]
```
**Captures:** how latency degrades with load.

### T5 — SLO validation (constant rate)
Pick target rate below the knee; verify p95 latency meets SLO for 5–10 min.
```bash
guidellm run \
  --profile kind=constant,rate=<X> \
  --constraint kind=max_duration,seconds=600
```

### T6 — Workload variants
Repeat T3 for each **workload profile** in §3.4.

### T7 — Long-context stress (optional)
Prompt_tokens=16k+, low concurrency, check TTFT & VRAM.

### T8 — Stability soak (optional)
Constant rate at ~70% of max for 1 hour → check error rate, memory leak, throughput drift.

---

## 5. Results Matrix (fill in per run)

### 5.1 Config sheet
| Run ID | Model | Quant | GPU | TP | vLLM ver | Max-model-len | KV dtype | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | Qwen2.5-7B | AWQ | 1×A10G | 1 | 0.6.3 | 8192 | fp16 | baseline |
| R002 | | | | | | | | |

### 5.2 Performance matrix
| Run ID | Test | Workload | Rate/Conc | Req/s | Out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p95 | GPU util | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R001 | T1 | 512/256 | 1 | | | | | | | | | |
| R001 | T2 | 512/256 | max | | | | | | | | | |
| R001 | T3 | 512/256 | sweep | | | | | | | | | |

### 5.3 Comparison summary
| Config | Max req/s | Knee req/s | TTFT@knee p95 | ITL@knee p95 | $/1M tok | Winner? |
|---|---|---|---|---|---|---|
| Qwen-7B AWQ / A10G / TP1 | | | | | | |
| Qwen-7B AWQ / L4 / TP1 | | | | | | |
| Qwen-7B FP16 / A100 / TP1 | | | | | | |

---

## 6. vLLM Server Config — Levers to Vary

| Flag | Effect | Suggested sweep |
|---|---|---|
| `--tensor-parallel-size` | Splits model across GPUs | 1, 2, 4, 8 |
| `--max-model-len` | Max context | Match workload |
| `--max-num-seqs` | Max concurrent seqs | 64, 128, 256, 512 |
| `--max-num-batched-tokens` | Batch token cap | 2048, 4096, 8192 |
| `--gpu-memory-utilization` | VRAM headroom | 0.85, 0.90, 0.95 |
| `--kv-cache-dtype` | fp16 / fp8 | Both |
| `--quantization` | awq / gptq / fp8 | Per model |
| `--enable-prefix-caching` | Reuse prefix KV | on / off |
| `--enable-chunked-prefill` | Prefill batching | on / off |
| `--speculative-model` | Spec decoding | on / off |

Run **one lever at a time** to isolate effects.

---

## 7. Execution Checklist

- [ ] Pin **vLLM version**, **CUDA**, **driver**, **transformers** — record them.
- [ ] **Warm up** server (10–20 requests) before measuring.
- [ ] Isolate host: no other GPU jobs, CPU governor = performance.
- [ ] Run **each test ≥60 s** for statistical stability; 120 s for throughput.
- [ ] Collect vLLM `/metrics` (Prometheus) + `nvidia-smi dmon -s pucm` in parallel.
- [ ] Save `benchmarks.json` + HTML report per run with a **Run ID**.
- [ ] Repeat critical runs **3×** and report median.
- [ ] Record ambient: node type, region, network to client.

---

## 8. Analysis Framework

For each run, answer:

1. **Where is the knee?** Rate at which p95 latency crosses 2× the sync baseline.
2. **What is the bottleneck?**
   - GPU util <70% + high queue → CPU / scheduler / networking bound
   - GPU util >95% + rising latency → compute bound (expected)
   - VRAM near max + OOM risk → reduce `max-num-seqs` or context
3. **How does TP scale?** Ideal is linear; sub-linear beyond TP=4 is common.
4. **Quantization tradeoff:** tok/s gain vs quality regression (run eval separately).
5. **Prefix caching win:** compare on/off for RAG-like workloads.
6. **Cost:** `$/1M output tokens = (GPU_hourly_cost / 3600) / (out_tok_per_sec) * 1e6`

---

## 9. Reporting Template (per config)

```
### Run R001 — Qwen2.5-7B-AWQ on 1×A10G (TP=1, vLLM 0.6.3)

Workload: 512 in / 256 out

| Metric | Sync | Knee (rate=X) | Max throughput |
|---|---|---|---|
| Req/s | 0.9 | 3.2 | 4.1 |
| Output tok/s | 230 | 820 | 1050 |
| TTFT p95 (ms) | 95 | 180 | 640 |
| ITL p95 (ms) | 42 | 55 | 130 |
| E2E p95 (s) | 11 | 15 | 42 |
| GPU util % | 55 | 92 | 98 |
| VRAM (GB) | 18 | 22 | 22 |

Findings:
- Knee ≈ 3.2 req/s; latency doubles beyond this.
- KV cache saturates at ~48 concurrent seqs.
- FP8 KV cache gave +18% throughput at negligible quality delta.

Recommendation:
- Deploy at rate ≤ 3 req/s per replica for chat SLO (TTFT<200ms p95).
- Scale horizontally beyond that.
```

---

## 10. Common Pitfalls

- Running only **constant rate** — misses the curve. Use **sweep**.
- Prompt token count mismatch (tokenizer differences) — verify with `--data`.
- Measuring cold start — always warm up.
- Ignoring **client-side bottleneck** — benchmark from a beefy client.
- Comparing across different **max-model-len** — not apples to apples.
- Streaming disabled — TTFT/ITL become meaningless.
- Single run — variance can be 10–20%; median of 3.

---

## 11. Deliverables

1. Filled **Config sheet** (§5.1)
2. Filled **Performance matrix** (§5.2)
3. Filled **Comparison summary** (§5.3)
4. HTML reports per run
5. Executive summary: best config per (workload, SLO, budget)
