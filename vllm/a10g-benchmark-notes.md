# A10G vLLM Benchmark Notes — Qwen2.5-7B-Instruct-AWQ

**Date:** 2026-08-07
**Hardware:** AWS `g5.xlarge` — NVIDIA A10G, 24 GB VRAM, ~600 GB/s HBM
**Model:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
**Serving stack:** vLLM (OpenAI-compatible) behind nginx reverse proxy
**Benchmark tool:** `guidellm` (concurrent profile)
**Workload:** synthetic — 512 prompt tokens (541 with template), 256 output tokens

---

## Executive Summary

- Peak sustained throughput on a single A10G for Qwen2.5-7B-AWQ: **~1000–1400 gen tok/s aggregate**.
- Bottleneck confirmed: **HBM memory bandwidth** (classic LLM decode).
- Single-user experience: **~90 gen tok/s @ 25 ms TTFT** — very good for interactive chat.
- Practical interactive concurrency (ITL < 20 ms, TTFT < 1 s): **~16 users**.
- Batch/throughput concurrency: **~32–48 users**.

---

## Test 1 — Initial run (misconfigured)

### Command
```bash
guidellm run \
  --backend kind=openai_http,target=http://0.0.0.0,model='Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --profile '{"kind":"concurrent","streams":[1,2,4,6,8]}' \
  --constraint kind=max_duration,seconds=90 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --output kind=html
```

### Results

| Streams | Req/s | Gen tok/s | Tot tok/s | TTFT | ITL   | Latency | Errors |
| ------- | ----- | --------- | --------- | ---- | ----- | ------- | ------ |
| 1       | 0.3   | 90.3      | 281.0     | 142 ms | 10.6 ms | 2.85 s | 0      |
| 2       | 0.4   | 108.9     | 339.1     | 160 ms | 10.7 ms | 2.90 s | 12,261 |
| 4       | 0.3   | 88.7      | 276.0     | 168 ms | 10.7 ms | 2.89 s | 16,863 |
| 6       | 0.3   | 86.9      | 270.5     | 139 ms | 10.6 ms | 2.85 s | 16,511 |
| 8       | 0.3   | 88.3      | 274.8     | 139 ms | 10.6 ms | 2.85 s | 17,008 |

### Observations
- Throughput did NOT scale with concurrency — flatlined at ~90 gen tok/s.
- Massive error counts starting at concurrency ≥ 2.
- vLLM logs were clean (no server-side errors).
- GPU utilization showed 100% but only ~1 stream was actually executing.
- Reported "Conc" stayed at ~1.0 despite requesting up to 8.

### Root cause: **Nginx rate limiting**

Nginx access/error log:
```
limiting requests, excess: 10.756 by zone "api_per_ip", client: 172.18.0.1
POST /v1/chat/completions HTTP/1.1 503 197
```

Offending config:
```nginx
location /v1/ {
    limit_req zone=api_per_ip burst=10 nodelay;
    ...
}
```
Because the load generator ran from a single IP, `limit_req_zone` throttled all concurrent traffic. Requests never reached vLLM.

### Fix applied
Bypassed the rate-limit for the load generator's Docker subnet (`172.18.0.0/16`) using a `geo` + `map` trick, so `$binary_remote_addr` is only used as the key for external clients.

---

## Test 2 — After nginx fix

### Command
```bash
guidellm run \
  --backend kind=openai_http,target=http://0.0.0.0,model='Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --profile '{"kind":"concurrent","streams":[1,2,4,6,8]}' \
  --constraint kind=max_duration,seconds=90 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --output kind=html
```

### Results

| Streams | Req/s | Gen tok/s | Tot tok/s | TTFT (p50) | ITL   | Latency | Errors |
| ------- | ----- | --------- | --------- | ---------- | ----- | ------- | ------ |
| 1       | 0.4   | 93.7      | 291.9     | 23 ms      | 10.6 ms | 2.73 s | 0      |
| 2       | 0.7   | 172.2     | 536.1     | 139 ms     | 10.7 ms | 2.92 s | 0      |
| 4       | 1.2   | 305.9     | 952.4     | 474 ms     | 11.0 ms | 3.26 s | 0      |
| 6       | 1.7   | 419.0     | 1304.4    | 480 ms     | 12.1 ms | 3.55 s | 0      |
| 8       | 2.0   | 513.2     | 1597.6    | 590 ms     | 12.7 ms | 3.85 s | 0      |

### Scaling efficiency (baseline c=1)
| c   | Gen tok/s | Speedup | Efficiency |
| --- | --------- | ------- | ---------- |
| 1   | 93.7      | 1.00x   | 100%       |
| 2   | 172.2     | 1.84x   | 92%        |
| 4   | 305.9     | 3.27x   | 82%        |
| 6   | 419.0     | 4.47x   | 75%        |
| 8   | 513.2     | 5.48x   | 68%        |

### Observations
- Classic batching curve — sub-linear but healthy.
- ITL barely grew (10.6 → 12.7 ms) → decode not yet saturated.
- TTFT climbed sharply (23 → 590 ms) → prefill queueing under load.
- Zero errors — clean concurrency.
- KV cache from vLLM metrics: **~3%** at c=8 → huge headroom.

### Decision: push to higher concurrency

---

## Test 3 — Higher concurrency sweep

### Command
```bash
guidellm run \
  --backend kind=openai_http,target=http://0.0.0.0:8000,model='Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --profile '{"kind":"concurrent","streams":[8,16,24,32]}' \
  --constraint kind=max_duration,seconds=60 \
  --data kind=synthetic_text,prompt_tokens=512,output_tokens=256 \
  --output kind=html
```

### Results

| Streams | Req/s | Gen tok/s | Tot tok/s | TTFT     | ITL   | Latency | Errors |
| ------- | ----- | --------- | --------- | -------- | ----- | ------- | ------ |
| 8       | 2.0   | 503.7     | 1568.1    | 622 ms   | 12.7 ms | 3.85 s | 0      |
| 16      | 2.9   | 748.4     | 2330.0    | 900 ms   | 16.4 ms | 5.08 s | 0      |
| 24      | 3.6   | 937.7     | 2919.2    | 1095 ms  | 20.9 ms | 6.43 s | 0      |
| 32      | 4.0   | **1034.4** | **3220.5** | 1350 ms | 24.3 ms | 7.56 s | 0      |

### Scaling efficiency (baseline c=8)
| c    | Gen tok/s | Speedup | Efficiency |
| ---- | --------- | ------- | ---------- |
| 8    | 503.7     | 1.00x   | 100%       |
| 16   | 748.4     | 1.49x   | 74%        |
| 24   | 937.7     | 1.86x   | 62%        |
| 32   | 1034.4    | 2.05x   | 51%        |

### vLLM internal metrics at c=32
- **Queue depth (`num_requests_waiting`)**: ~14
- **Queue time**: ~0.8 s
- Queue accounts for ~60% of TTFT — `max_num_seqs` limit is reached.

---

## GPU Metrics (DCGM Profiling)

Metrics enabled via `dcgm-exporter` using `dcp-metrics-included.csv` and `--cap-add SYS_ADMIN`. Panels added to Grafana dashboard `grafana-dgexporter.json`:

| Panel ID | Title | Query |
| -------- | ----- | ----- |
| 20 | GPU SM Active | `DCGM_FI_PROF_SM_ACTIVE * 100` |
| 22 | GPU SM Occupancy | `DCGM_FI_PROF_SM_OCCUPANCY * 100` |
| 24 | GPU DRAM Active (Memory BW) | `DCGM_FI_PROF_DRAM_ACTIVE * 100` |

### Peak values observed at c=32

| Metric | Value | Interpretation |
| ------ | ----- | -------------- |
| **SM Active** | ~100% | SMs scheduled every cycle |
| **Tensor Core Active** | ~40% | Compute pipes idle >half the time |
| **DRAM Active** | ~80% | HBM near saturation (~480 GB/s of 600 GB/s theoretical) |
| **GPU Util (legacy)** | 100% | Not meaningful — any single kernel keeps it at 100% |
| **GPU FB Used** | ~90% | vLLM pre-allocation (`gpu_memory_utilization=0.9`), not real pressure |
| **KV cache usage** | ~3% at c=8 | Huge slack — `max_num_seqs`, not memory, was the cap |
| **Power draw** | ~140 W | Near A10G TDP of 150 W |
| **Temperature** | 42–55 °C | Well within limits |

### Interpretation
Classic **memory-bandwidth-bound LLM decode**:
- SM Active 100% with Tensor 40% → SMs are stalled waiting for weights/KV from HBM, not doing FMA.
- Decode = "stream all 7B weights + KV per token" — bandwidth workload, not compute workload.
- Explains why A100 / H100 outperform A10G at inference despite similar peak FLOPs on paper — they have 3–4x HBM bandwidth.

---

## Key Debugging Wins

### 1. Nginx rate limiting hid real vLLM performance
- Symptom: throughput flat, thousands of 503s, vLLM logs clean.
- Fix: bypass `limit_req` for internal load-generator subnet.

### 2. DCGM `GPU_UTIL` is misleading
- "100% GPU Util" ≠ "GPU saturated". It just means at least one SM had work in the sample window.
- For real answers, always use `DCGM_FI_PROF_*`:
  - `SM_ACTIVE` — % of time SMs scheduled
  - `SM_OCCUPANCY` — warp fill
  - `PIPE_TENSOR_ACTIVE` — real compute saturation (**most useful for LLM training/prefill**)
  - `DRAM_ACTIVE` — memory bandwidth (**most useful for LLM decode**)
- Requires `dcgm-exporter -f /etc/dcgm-exporter/dcp-metrics-included.csv` with `--cap-add SYS_ADMIN`.

### 3. GPU memory at 100% ≠ memory pressure
- vLLM pre-allocates `gpu_memory_utilization * VRAM` at startup and pools it for KV cache.
- Real pressure signal: `vllm:gpu_cache_usage_perc` and `num_requests_waiting`.

### 4. Queue depth + queue_time diagnose `max_num_seqs`
- Queue=14, queue_time=0.8 s at c=32 → running-batch cap was reached.
- Fix: raise `--max-num-seqs`.

---

## Recommended Config for Max Throughput on A10G

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin \
  --kv-cache-dtype fp8 \
  --max-model-len 2048 \
  --max-num-seqs 64 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92
```

| Flag | Why |
| ---- | --- |
| `awq_marlin` | Faster AWQ kernel than default |
| `kv-cache-dtype fp8` | Halves KV memory traffic — biggest lever for a bandwidth-bound workload |
| `max-model-len 2048` | Frees KV pool space; test workload is 797 tokens |
| `max-num-seqs 64` | Removes queue seen at c=32 |
| `enable-chunked-prefill` | Interleaves prefill with decode → uses idle Tensor cores → cuts TTFT |
| `max-num-batched-tokens 4096` | Bigger fused batches |
| `gpu-memory-utilization 0.92` | Slightly more KV room; leave 8% for activations |

**Expected uplift vs. Test 3:**
- Peak gen tok/s: 1000 → **1300–1500**
- Tensor Core Active: 40% → **55–70%**
- DRAM Active: 80% → **85–95%**
- ITL: 24 → **18–22 ms**
- TTFT at c=32: 1350 → **600–800 ms**

---

## Capacity Planning — Single A10G, Qwen2.5-7B-AWQ

| SLA Target | Config | Peak concurrency | Aggregate gen tok/s | Req/s (256-tok out) |
| ---------- | ------ | ---------------- | -------------------- | ------------------- |
| Interactive chat (ITL < 15 ms, TTFT < 250 ms) | Default | ~4–6 | ~300–420 | ~1.2–1.7 |
| Balanced (ITL < 20 ms, TTFT < 1 s) | Default | ~16 | ~750 | ~2.9 |
| Batch / throughput (latency tolerant) | Default | 32–48 | ~1000–1200 | ~4 |
| Batch, tuned (`fp8` KV + chunked prefill) | Recommended | 48–64 | **~1300–1500** | ~5–6 |

### Scaling beyond one A10G
- A10G has no NVLink → **no tensor parallelism** across cards on `g5.*`.
- Scale by **horizontal replication** (multiple A10Gs behind a router / load balancer).
- Or upgrade GPU class:
  - **L40S** (48 GB, ~864 GB/s) — ~1.5x A10G decode throughput
  - **A100 40/80 GB** (~1.5–2 TB/s) — ~3x A10G
  - **H100** (~3.35 TB/s) — ~5x A10G

---

## Grafana Dashboard Changes

File: `grafana-dgexporter.json` (backup: `grafana-dgexporter.json.bak`)

Added three panels below existing "GPU Framebuffer Mem Used":

| ID | Title | Position (grid) |
| -- | ----- | --------------- |
| 20 | GPU SM Active | (0, 48) 12x8 |
| 22 | GPU SM Occupancy | (12, 48) 12x8 |
| 24 | GPU DRAM Active (Memory BW) | (0, 56) 12x8 |

All cloned from the existing "GPU Utilization" panel — percent unit, 0–100 range, green→red @ 80 threshold, using `${instance}` and `${gpu}` template variables.

To load into Grafana: Dashboard Settings → JSON Model → paste and Save, or re-import with Overwrite.

---

## Reference: vLLM Prometheus Metrics to Add

Also scrape vLLM's `/metrics` endpoint — these are more informative than DCGM for LLM serving:

```promql
vllm:gpu_cache_usage_perc * 100                           # KV cache utilization
vllm:num_requests_running                                  # Currently in batch
vllm:num_requests_waiting                                  # Queued (bottleneck signal)
rate(vllm:generation_tokens_total[1m])                     # Aggregate gen tok/s
rate(vllm:prompt_tokens_total[1m])                         # Aggregate prompt tok/s
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[1m]))
histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[1m]))
histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[1m]))
```

---

## Timeline

| Time | Event |
| ---- | ----- |
| 06:27 | Test 1 start — concurrency 1–8, throughput flat, thousands of errors |
| ~06:35 | Root cause identified: nginx `limit_req` |
| 07:00 | Test 2 start after nginx bypass — clean scaling, KV cache 3% |
| 07:08 | Test 2 done — confirmed vLLM healthy, decision to push higher |
| ~07:15 | DCGM profiling metrics enabled (`dcp-metrics-included.csv`) |
| ~07:30 | Grafana panels added (SM Active, SM Occupancy, DRAM Active) |
| 08:15 | Test 3 start — concurrency 8–32, peaked at 1034 gen tok/s |
| 08:19 | Test 3 done — DRAM 80%, Tensor 40% → memory-bandwidth-bound |
