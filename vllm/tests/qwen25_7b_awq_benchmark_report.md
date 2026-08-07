# vLLM Benchmark Report — Qwen2.5-7B-Instruct-AWQ

**Date:** 2026-08-04
**Model:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
**Backend:** vLLM, OpenAI HTTP API at `http://0.0.0.0:8000`
**Workload:** synthetic text, **prompt = 512 tok, output = 256 tok**
**Tool:** guidellm (streaming enabled)
**Duration per test:** 90 s
**Artifacts:** `benchmarks.html` (sweep), `benchmark_concurrent.html` (concurrency)

> **Note on GPU:** Add GPU model, TP size, vLLM version, and driver here before publishing this report — those affect every number below.

---

## 1. Executive Summary

- **Single-user decode speed:** ~45 tok/s, ITL 22 ms, TTFT 42 ms (best case).
- **Per-replica knee:** between **N=4 and N=6 concurrent users**.
- **Practical capacity for interactive chat (TTFT p95 < 1 s):** **~2 users per replica**.
- **Practical capacity with relaxed SLO (TTFT p95 < 2 s):** **~4 users per replica**.
- **Aggregate throughput at N=8:** ~195 gen tok/s (still climbing → true max not reached).
- **Warning discovered:** naive `--profile kind=sweep` with default `max_concurrency=512` broke the sweep on this hardware — see §4.
- **Warning discovered:** naive `--profile kind=constant,rate=2` overloaded the server (0.5 req/s actual vs 2 requested, 131 incomplete) — see §5 and **Appendix A** for how this reconciles with T3.

---

## 2. Test Plan Executed

| # | Test | Purpose | Command flavor |
|---|---|---|---|
| T1 | Synchronous | Best-case latency, decode ceiling | `--profile kind=synchronous` (via sweep stage 1) |
| T2 | Constant rate | Steady-state at fixed req/s | `--profile kind=constant,rate=2` |
| T3 | Concurrent streams | Capacity vs users | `--profile '{"kind":"concurrent","streams":[1,2,4,6,8]}'` |

All tests used the same workload (512 in / 256 out) so results are directly comparable.

---

## 3. Test T1 — Synchronous baseline

**Command (as first stage of a sweep run):**
```bash
guidellm run \
  --backend kind=openai_http,target=http://0.0.0.0:8000,model='Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --profile kind=sweep,sweep_size=10 \
  --constraint kind=max_duration,seconds=90 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256
```

**Result (synchronous stage only):**

| Metric | Value |
|---|---|
| Completed requests | 16 |
| Req/s | 0.176 |
| Aggregate gen tok/s | 45.1 |
| TTFT p50 / p95 | 42 ms / 69 ms |
| ITL p50 / p95 | 22.1 ms / 22.3 ms |
| E2E p50 / p95 | 5.68 s / 5.68 s |
| Concurrency | 1 |

**Interpretation:**
- **Decode speed = 1000/22 ≈ 45 tok/s per stream.**
- **Prefill (541 tokens) took ~42 ms** — very fast.
- E2E = TTFT + 255 × ITL = 42 + 255×22 ≈ 5.65 s ✓ arithmetic sanity check passes.
- This is the **best-case UX** for one user with no contention.

---

## 4. Test T1b — Sweep attempt (illustrative failure)

Same command as above with `sweep_size=10`. **Result: unusable.**

- `synchronous` stage → 0.176 req/s ✓
- `throughput` stage → **0 completed, 512 incomplete**. guidellm's default `max_concurrency=512` overwhelmed the server; nothing finished in 90 s.
- Because the upper anchor was 0, guidellm interpolated 8 constant-rate stages **below** the sync rate (0.15, 0.12, 0.10 … 0.02 req/s) — all under-loaded, all showing ~6 s latency with concurrency ≤ 1 → **no useful information about the knee**.

**Lesson learned:** on modest GPUs, always cap concurrency in sweeps:
```bash
--profile kind=sweep,sweep_size=10,max_concurrency=32
```
Or use the `concurrent` profile directly (see T3).

---

## 5. Test T2 — Constant rate = 2 req/s (overload demo)

**Command:**
```bash
guidellm run \
  --backend kind=openai_http,target=http://0.0.0.0:8000,model='Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --profile kind=constant,rate=2 \
  --constraint kind=max_duration,seconds=90 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256
```

**Result:**

| Metric | Value | Interpretation |
|---|---|---|
| Target rate | 2.0 req/s | requested |
| Actual completed rate | 0.5 req/s | **server saturated** |
| Completed | 49 | |
| Incomplete | 131 | requests still in-flight at end |
| Errors | 0 | vLLM queued, did not crash |
| Concurrency mean | 58.7 | queue growing throughout |
| TTFT p95 | 165.9 ms | small because streaming starts once scheduled |
| ITL p95 | 210.9 ms | ~10× degraded due to large in-flight batch |
| E2E p95 | 53.9 s | mostly queue wait |

**Interpretation:**
- **Server cannot sustain 2 req/s** at 512 in / 256 out on this GPU.
- Requests accumulate: 2 in − 0.5 out = 1.5 req/s piled up × 90 s ≈ 135 in queue at end (matches the 131 "incomplete").
- **ITL degradation (22 → 211 ms)** reflects large-batch decoding under vLLM's continuous batching.
- **No errors, no crash** — confirms vLLM's queue behavior: it just gets slower.

**Lesson learned:** the sync test already gave the answer (0.176 req/s = max sustainable single-stream rate). Anything above that requires concurrency, and 2 req/s is ~11× the sync rate — far beyond one replica's capacity for this workload.

---

## 6. Test T3 — Concurrent streams (the useful test)

**Command:**
```bash
guidellm run \
  --backend kind=openai_http,target=http://0.0.0.0:8000,model='Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --profile '{"kind":"concurrent","streams":[1,2,4,6,8]}' \
  --constraint kind=max_duration,seconds=90 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --output kind=html
```

**Result:**

| Users (N) | Completed | Aggregate gen tok/s | Per-user tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p95 | Scaling |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 42.7 | 42.7 | 390 ms | 393 ms | 22.1 ms | 22.3 ms | 6.0 s | 100% |
| 2 | 27 | 78.5 | 39.3 | 408 ms | 776 ms | 23.8 ms | 25.2 ms | 6.8 s | **92%** |
| 4 | 45 | 132.0 | 33.0 | 443 ms | 1 529 ms | 26.1 ms | 30.5 ms | 8.2 s | **77%** |
| 6 | 55 | 167.3 | 27.8 | 1 537 ms | 2 270 ms | 30.4 ms | 36.0 ms | 9.6 s | **65%** |
| 8 | 65 | 194.8 | 24.4 | 1 860 ms | 3 024 ms | 34.0 ms | 39.6 ms | 10.6 s | **57%** |

*Scaling = aggregate(N) / (N × aggregate(1)) — 100% means perfect parallel speed-up.*

### Stage-by-stage interpretation

| N | State | Verdict |
|---|---|---|
| 1 | Baseline | Single user gets full 43 tok/s |
| 2 | Free doubling | Aggregate 1.84× — near-perfect scaling; per-user speed drops only 8% |
| 4 | Sweet spot | Aggregate 3.1×; TTFT p50 barely moved but p95 crosses 1.5 s |
| 6 | Knee begins | TTFT p50 jumps 443→1537 ms — **prefill queue kicks in**; only +27% aggregate for 50% more users |
| 8 | Past knee | TTFT p95 = 3 s; per-user 24 tok/s; +17% aggregate for +33% users → strong diminishing returns |

### Capacity by SLO

| Product SLO | Max users per replica | Aggregate tok/s |
|---|---:|---:|
| **Snappy chat** (TTFT p95 < 500 ms) | 1 | 43 |
| **Normal chat** (TTFT p95 < 1 s) | 2 | 79 |
| **Tolerant chat** (TTFT p95 < 2 s) | 4 | 132 |
| **Batch / async** (per-user tok/s ≥ 20) | 8+ | 195+ |

---

## 7. Server-side observations (vLLM Prometheus / Grafana)

Grafana dashboard captured during the T3 concurrent test window confirmed and refined the client-side view:

| Server-side signal | Observation | Implication |
|---|---|---|
| `num_requests_running` | Stepped 1 → 2 → 4 → 6 → 8 | Perfect match with client concurrency |
| `num_requests_waiting` | **0 throughout** | **No queueing — server accepted every request immediately** |
| Queue time | ~0 (single 20 ms spike) | Confirms no queue backlog |
| GPU KV cache usage | **peaked ~5%** | Massive KV headroom; KV is NOT the bottleneck |
| Prefill vs decode time | Decode dominates (~6 s); prefill small | Decode is the bulk of E2E — expected for 256-token output |
| Finish reason | 100% `length` | All requests hit the 256-token cap as intended |
| E2E / TTFT histograms | Match guidellm numbers | Client and server measurements are consistent |

### Revised knee diagnosis

Earlier we hypothesized "prefill queue kicks in at N=6". **Grafana disproves that** — waiting queue was zero. The real cause of the TTFT jump 443 → 1537 ms at N=6 is:

> **In-batch prefill contention.** vLLM's continuous batching runs new-request prefill in the same forward step as ongoing decodes. As the batch fills, per-step compute time grows → new arrivals wait for the next step to complete → TTFT rises.

So the bottleneck at N=4–8 on this GPU is **prefill compute**, not queueing and not KV memory. Enabling `--enable-chunked-prefill` (if not already on) may push the knee out; adding a faster GPU definitely will.

## 8. Cross-test observations

1. **TTFT differs between profiles at the same concurrency=1:**
   - `synchronous` T1 → 42 ms
   - `concurrent@1` T3 → 390 ms
   Both use one in-flight request. The delta is a **scheduler-tick / connection-setup artifact** of the async-arrival driver used by `concurrent` and `constant`. **Do not compare TTFT across profiles**; compare within the same profile.

2. **ITL is remarkably stable across N=1..8** (22 → 40 ms) — vLLM's continuous batching is doing its job. It's TTFT that degrades first.

3. **Prefill is the choke point.** Prompt = 512 tokens dominates TTFT. Halving prompt length would roughly halve TTFT and shift the knee to higher N.

4. **vLLM never errored** across any test — it queued and slowed down instead. Client-side timeouts would eventually fail, but the server itself is stable.

---

## 9. Conclusions

1. **This model / GPU / workload combination is prefill-compute-bound, not memory-bound.** KV cache peaked at ~5%; there is ~20× KV headroom unused. What breaks first under load is **prefill compute inside the batched forward step** — visible client-side as TTFT growth.
2. **Effective per-replica capacity is 2–4 concurrent chat users** depending on SLO strictness.
3. **Peak per-replica aggregate throughput** is ≥ 195 gen tok/s (upper bound not yet found — see §9).
4. **Do not use `--profile kind=constant,rate=N` blindly** — it can overload the server and give misleading numbers. Anchor rate choices to the sync test result.
5. **Do not use default `sweep` on modest GPUs** — cap `max_concurrency` or use `concurrent` streams.
6. **Preferred test for capacity planning:** `--profile '{"kind":"concurrent","streams":[…]}'` — directly answers "how many users can I serve?" and is robust to server-side saturation.

---

## 10. Recommended next tests

| Goal | Command | Expected insight |
|---|---|---|
| Find aggregate throughput ceiling | `--profile '{"kind":"concurrent","streams":[8,12,16,24,32]}'` | Where aggregate gen tok/s flattens → max per-replica throughput |
| Measure prompt-size sensitivity | Re-run T3 with `prompt_tokens=128,output_tokens=128` | TTFT should drop ~3–4× → confirms prefill dominance |
| Long-context stress | `prompt_tokens=8192,output_tokens=256` | Peak VRAM, TTFT under long prefill |
| Server-config tuning | Restart vLLM with `--max-num-seqs 128`, re-run T3 | Check if bumping batch size shifts the knee |
| Alternate quantization | Swap AWQ → FP8 (H100), re-run T1+T3 | Latency / throughput / quality tradeoff |
| GPU-side metrics | Run `nvidia-smi dmon -s pucmt -o DT -d 1` during T3 | Confirm compute vs memory-bandwidth bottleneck |
| Chunked prefill | Restart vLLM with `--enable-chunked-prefill`, re-run T3 | Should raise the knee (shorter TTFT under load) |
| Prefix caching win | Re-run T3 with `--enable-prefix-caching` and a fixed prompt prefix | Check `vllm:prefix_cache_hit_rate` — big TTFT win for RAG workloads |

---

## 11. Reproducibility checklist (fill in before publishing)

- [ ] GPU model & count
- [ ] vLLM version, CUDA, driver
- [ ] vLLM server flags (`--tensor-parallel-size`, `--max-num-seqs`, `--max-model-len`, `--kv-cache-dtype`, `--enable-prefix-caching`, etc.)
- [ ] Host type, region, client-to-server network hop
- [ ] guidellm version
- [ ] HF tokenizer commit hash
- [ ] Warm-up performed? (Y/N)
- [ ] Number of repetitions per test (recommend ≥3, report median)

---

*Report generated from three guidellm runs on 2026-08-04. HTML artifacts: `benchmarks.html` (T1+T2 sweep), `benchmark_concurrent.html` (T3).*

---

## Appendix A — Reconciling T2 and T3 (rate vs concurrency)

At first glance T2 and T3 look contradictory:
- **T2** demanded `rate=2 req/s` → server saturated, only 0.5 req/s completed.
- **T3** ran `streams=4` concurrent → completed 0.5 req/s cleanly with no queue.

They are actually **the same result**, viewed through two different lenses.

### Two benchmark modes

| | T2 (`constant, rate=2`) | T3 (`concurrent, streams=N`) |
|---|---|---|
| Client submits… | New request every 0.5 s regardless of server | Only after previous one finishes |
| In-flight count | Grows unbounded if server is slow | Fixed at N |
| Called | **Open-loop** / arrival-driven | **Closed-loop** / concurrency-driven |

### Little's Law ties them together

```
Concurrency = Arrival_rate × Average_latency
      L     =        λ      ×        W
```

Verifying with T3 measurements:

| N (streams) | Latency W (s) | Predicted λ = N/W | Measured req/s | ✓ |
|---:|---:|---:|---:|---:|
| 4 | 8.2 | 0.49 | 0.5 | ✓ |
| 8 | 10.6 | 0.75 | 0.7 | ✓ |

### Applying it to T2

- T2 asked for **λ = 2 req/s**.
- Server ceiling (from T3 extrapolation) is ~**0.8 req/s**.
- Since demand > capacity, the extra 1.2 req/s accumulate → `L` grows without bound.
- Measured mean L = 58.7 during T2, still climbing at test end → matches the diagnosis.
- Actual completed rate landed at ~0.5 req/s — exactly the server's ceiling in the T3 measurement.

### Rule of thumb

> **Max sustainable arrival rate ≈ max aggregate gen tok/s ÷ output tokens per request**
>
> For this instance: ~200 tok/s ÷ 256 tok/req ≈ **0.78 req/s ceiling**.
> Any `rate` requested above this reproduces the T2 overload pattern.

### When to use each profile

| Question | Use |
|---|---|
| "How many concurrent users can I serve at an SLO?" | **concurrent** |
| "What if traffic bursts to X req/s (X below ceiling)?" | **constant** |
| "Where is the server's ceiling?" | **concurrent** at high N, or **throughput** with capped concurrency |
| "What breaks first under overload?" | **constant** above ceiling (as T2 demonstrated) |

Both T2 and T3 are valid — they answer different questions. The report treats T3 as the primary capacity measurement and T2 as an overload-behavior demonstration.

