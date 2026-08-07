# LLM Serving Platform — Architecture & Design

> **Inspiration**: Netflix in-house LLM serving (https://netflixtechblog.com/in-house-llm-serving-at-netflix-a5a8e799ea2c)
>
> **Starting point**: This workspace's existing vLLM single-GPU experiments.
>
> **Philosophy**: Build the simplest thing that works, in layers. Each layer is independently useful. OpenRouter provider journey is a *consumer integration module* on top of this platform — not the platform itself.

---

## 1. The Problem We Are Solving

Running LLM inference is fundamentally different from running a web service:

| Dimension | Web Service | LLM Serving |
|---|---|---|
| Compute | CPU-bound, scales horizontally trivially | GPU-bound, expensive, scarce |
| State | Stateless (mostly) | Stateful KV cache per active request |
| Latency budget | ms per request | Seconds for TTFT + streaming tokens |
| Resource contention | I/O and memory | GPU VRAM is the critical resource |
| Failure modes | Crash/OOM | OOM mid-generation, model load failure, driver issues |
| Cost structure | Pay for uptime | Pay for GPU-hours; idle GPU = burn |

A good LLM serving platform must handle all of this transparently so that **operators** (who deploy models) and **consumers** (who call the API) don't have to think about it.

---

## 2. Goals

### Must Have (v1)
- **Deploy any open-weight model** on any SSH-accessible GPU instance in a single command.
- **Serve OpenAI-compatible API** — any client that works with OpenAI works with this platform.
- **Config-driven knobs** — quantization, context length, memory utilization — without touching code.
- **Built-in observability** — GPU, system, and vLLM metrics available out-of-the-box.
- **Provider-ready** — the endpoint meets quality gates for external providers (OpenRouter, etc.).

### Nice to Have (v2+)
- Multi-model routing — one public endpoint, multiple backends, routed by model ID.
- Multi-instance load balancing — N GPU boxes behind one gateway.
- Model registry and lifecycle management — catalog, version, hot-swap.
- Cost tracking — tokens per dollar, per model, per consumer.
- Auto-scaling — scale out/in based on queue depth and GPU utilization.

### Explicitly Out of Scope for v1
- Training or fine-tuning.
- Proprietary model support (closed-weight APIs).
- Multi-cloud orchestration / Kubernetes.
- SLA enforcement with billing.

---

## 3. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         CONSUMER INTEGRATIONS                            ║
║   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐      ║
║   │   OpenRouter    │   │   Direct API    │   │  Internal Apps  │      ║
║   │   (Phase 6+)    │   │    Clients      │   │   / Products    │      ║
║   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘      ║
╚════════════╪════════════════════╪════════════════════╪═════════════════╝
             │                   │                    │
╔════════════╪═══════════════════╪════════════════════╪═════════════════╗
║            └───────────────────┴────────────────────┘                  ║
║                         API GATEWAY LAYER                               ║
║   ┌────────────────────────────────────────────────────────────────┐   ║
║   │  Nginx  (TLS termination, rate-limiting, upstream health)      │   ║
║   │       :443 → forwards to FastAPI / vLLM :8000                  │   ║
║   └──────────────────────────────┬─────────────────────────────────┘   ║
║                                  │                                      ║
║   ┌──────────────────────────────▼─────────────────────────────────┐   ║
║   │  OpenAI-compatible API surface                                  │   ║
║   │  GET  /v1/models          POST /v1/chat/completions             │   ║
║   │  POST /v1/completions     GET  /health    GET /metrics          │   ║
║   └──────────────────────────────┬─────────────────────────────────┘   ║
╚═════════════════════════════════╪════════════════════════════════════╝
                                  │
╔═════════════════════════════════╪════════════════════════════════════╗
║                    INFERENCE ENGINE LAYER                              ║
║                                  │                                     ║
║   ┌───────────────────────────────▼────────────────────────────────┐  ║
║   │                    vLLM Engine (Docker)                        │  ║
║   │   PagedAttention │ Continuous Batching │ Prefix Caching        │  ║
║   │   Config: vllm.yaml  (model, dtype, max_model_len, etc.)       │  ║
║   └───────────────────────────────┬────────────────────────────────┘  ║
║                                   │                                    ║
║   ┌───────────────────────────────▼────────────────────────────────┐  ║
║   │                 MODEL MANAGEMENT                               │  ║
║   │  HuggingFace Cache (NVMe)  │  Model Registry  │  Config Store  │  ║
║   └────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════╗  ╔═══════════════════════════════════╗
║        CONTROL PLANE             ║  ║       OBSERVABILITY STACK         ║
║                                  ║  ║                                   ║
║  Provisioner (SSH-based)         ║  ║  Prometheus                       ║
║  ├─ GPU validation               ║  ║  ├─ GPU metrics (DCGM Exporter)   ║
║  ├─ NVIDIA runtime setup         ║  ║  ├─ System metrics (Node Exporter)║
║  ├─ Docker image deploy          ║  ║  └─ vLLM metrics (/metrics)       ║
║  └─ Stack health checks          ║  ║                                   ║
║                                  ║  ║  Grafana (dashboards)             ║
║  Config Manager                  ║  ║  Alertmanager (SLO alerts)        ║
║  ├─ vllm.yaml per deployment     ║  ║                                   ║
║  └─ Runtime overrides via env    ║  ╚═══════════════════════════════════╝
╚══════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                      INFRASTRUCTURE LAYER                              ║
║   AWS EC2 (g4dn, p3)  │  RunPod  │  CoreWeave  │  Lambda Labs  │ BM   ║
║   — accessed via SSH + IP address —                                    ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 4. Component Deep Dives

### 4.1 Control Plane — Provisioner

**Responsibility**: Given a GPU instance IP and SSH key, bring the entire stack to life.

**Inputs**:
```
IP: 54.x.x.x
SSH_KEY: ~/.ssh/my-gpu-key.pem
MODEL: Qwen/Qwen2.5-32B-Instruct
HF_TOKEN: hf_xxxx              # optional; only for gated models
CONFIG_OVERRIDES: max-model-len=16384, gpu-memory-utilization=0.92
```

**Steps executed by provisioner**:
```
Step 1: Validate instance
  → ssh -i $SSH_KEY ec2-user@$IP "nvidia-smi"
  → Confirm: driver version, VRAM size, CUDA version

Step 2: Ensure NVIDIA Container Toolkit
  → Check: docker info | grep nvidia
  → If missing: install nvidia-container-toolkit, configure, restart Docker

Step 3: Pull vLLM image
  → docker pull vllm/vllm-openai:latest (or pinned version)

Step 4: Pull/verify model
  → Check HF cache: ls ~/.cache/huggingface/hub/models--{model_slug}
  → If missing: docker run --rm ... huggingface-cli download {model} (pre-cache)
  → OR: download at vLLM startup (simpler, slower first start)

Step 5: Write config
  → Upload vllm.yaml with resolved overrides

Step 6: Start vLLM container
  → docker run -d --gpus all -p 8000:8000 vllm/vllm-openai ...
  → Poll GET /health until 200 (timeout 10m for large model loads)

Step 7: Start monitoring exporters
  → DCGM Exporter (GPU metrics, port 9400)
  → Node Exporter (system metrics, port 9100)
  → (vLLM /metrics is already on port 8000)

Step 8: Start Nginx + TLS
  → Upload nginx.conf
  → docker run nginx:alpine ...
  → Verify: curl https://$DOMAIN/health

Step 9: Register endpoint (optional)
  → Update provider registry (OpenRouter or internal catalog)
```

**Implementation target**: Ansible playbook (preferred) or shell scripts called via SSH. The existing `scripts/` directory and `Makefile` are the starting point.

---

### 4.2 Inference Engine Layer — vLLM

This is the core of the platform. vLLM handles everything related to actually running the model.

**Container strategy** (already exists in `Dockerfile` and `Makefile`):
```
Base image: vllm/vllm-openai:latest   (CUDA, cuDNN, vLLM pre-installed)
Config mount: -v ./config/vllm.yaml:/app/vllm.yaml
Model cache: -v ~/.cache/huggingface:/root/.cache/huggingface
```

**Key configuration knobs** (from `config/vllm.yaml`):

| Knob | What it controls | When to tune |
|---|---|---|
| `model` | Which model to serve | Per deployment |
| `dtype` | Precision (bf16/fp16/auto) | fp16 for Turing (T4), bf16 for Ampere+ |
| `quantization` | AWQ / GPTQ / bitsandbytes | When model is too large for VRAM |
| `max-model-len` | Max context window | Reduce to free up KV cache space |
| `gpu-memory-utilization` | Fraction of VRAM for KV cache | 0.90 default; lower if OOM |
| `max-num-seqs` | Max parallel requests | Tune for throughput vs latency tradeoff |
| `tensor-parallel-size` | Multi-GPU sharding | Only for multi-GPU boxes |
| `enable-prefix-caching` | Reuse KV cache for shared prefixes | Always on for RAG / system prompts |

**Model source options**:
- **Option A — HuggingFace at startup**: Set `VLLM_MODEL=org/model-name` + `HF_TOKEN`. Downloads on first run, cached for subsequent runs.
- **Option B — Pre-cached mount**: Download weights to NVMe ahead of time, mount as volume. Faster restarts, no network dependency.

---

### 4.3 API Gateway Layer

Two layers:

**Inner layer — vLLM's built-in OpenAI server**:
vLLM already ships an OpenAI-compatible HTTP server. This is sufficient for direct access.

Endpoints:
```
GET  /health                     → liveness check
GET  /v1/models                  → list loaded model(s)
POST /v1/chat/completions        → streaming + non-streaming
POST /v1/completions             → legacy completions
GET  /metrics                    → Prometheus metrics
```

**Outer layer — Nginx**:
- TLS termination (certificates via Let's Encrypt / certbot or pre-supplied)
- Rate limiting per client IP / API key
- Upstream health checks (if multiple vLLM backends)
- Access logging for audit trail

```nginx
# Minimal nginx.conf structure
upstream vllm_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl;
    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    location / {
        proxy_pass         http://vllm_backend;
        proxy_set_header   Host $host;
        proxy_read_timeout 300s;       # long timeout for streaming
        proxy_buffering    off;        # required for SSE streaming
    }

    location /metrics {
        allow 10.0.0.0/8;              # restrict to internal monitoring
        deny  all;
        proxy_pass http://vllm_backend/metrics;
    }
}
```

**Optional FastAPI wrapper** (for OpenRouter journey and beyond):
When you need custom logic (pricing headers, model aliasing, auth, request logging), a thin FastAPI proxy sits between Nginx and vLLM:
```
Nginx :443 → FastAPI wrapper :8001 → vLLM :8000
```
This is where the OpenRouter-specific endpoint logic lives.

---

### 4.4 Observability Stack

Three metric sources, one scraper, one dashboard tool.

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ DCGM Exporter│    │  Node Exporter   │    │  vLLM        │
│ :9400/metrics│    │  :9100/metrics   │    │  :8000/metrics│
└──────┬───────┘    └────────┬─────────┘    └──────┬───────┘
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                    ┌────────▼──────────┐
                    │   Prometheus      │
                    │   :9090           │
                    └────────┬──────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼──────────┐       ┌──────────▼────────┐
     │     Grafana        │       │   Alertmanager     │
     │     :3000          │       │   :9093            │
     └───────────────────┘       └────────────────────┘
```

**GPU Metrics (DCGM Exporter)** — what to watch:
```
DCGM_FI_DEV_GPU_UTIL          → GPU compute utilization %
DCGM_FI_DEV_MEM_COPY_UTIL     → memory bandwidth utilization %
DCGM_FI_DEV_FB_USED           → VRAM used (MiB)
DCGM_FI_DEV_FB_FREE           → VRAM free (MiB)
DCGM_FI_DEV_GPU_TEMP          → temperature (°C)
DCGM_FI_DEV_POWER_USAGE       → power draw (W)
```

**System Metrics (Node Exporter)** — what to watch:
```
node_cpu_seconds_total         → CPU utilization
node_memory_MemAvailable_bytes → free RAM
node_disk_io_time_seconds_total → disk I/O (model cache reads)
node_network_receive_bytes_total → inbound tokens (prompts)
node_network_transmit_bytes_total → outbound tokens (completions)
```

**vLLM Metrics** (built-in `/metrics`) — what to watch:
```
vllm:time_to_first_token_seconds      → TTFT histogram (p50, p95, p99)
vllm:time_per_output_token_seconds    → TPOT / inter-token latency
vllm:e2e_request_latency_seconds      → end-to-end latency
vllm:request_success_total            → successful completions
vllm:num_requests_running             → current active sequences
vllm:num_requests_waiting             → queue depth
vllm:gpu_cache_usage_perc             → KV cache utilization %
vllm:prompt_tokens_total              → input token count
vllm:generation_tokens_total          → output token count
```

**Prometheus scrape config** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ['localhost:8000']

  - job_name: gpu
    static_configs:
      - targets: ['localhost:9400']

  - job_name: node
    static_configs:
      - targets: ['localhost:9100']
```

**Key Grafana dashboards to build**:
1. **GPU health** — utilization, VRAM, temp, power
2. **vLLM engine** — TTFT p95, queue depth, KV cache %, tokens/sec
3. **Business / SLO** — requests/min, error rate, TTFT p95 vs SLO threshold
4. **Cost** — tokens generated per GPU-hour (proxy for revenue efficiency)

**SLO Alerts** (Alertmanager rules):
```yaml
- alert: HighTTFT
  expr: histogram_quantile(0.95, vllm:time_to_first_token_seconds_bucket) > 0.5
  for: 2m
  annotations:
    summary: "TTFT p95 > 500ms — below OpenRouter SLO"

- alert: KVCacheNearFull
  expr: vllm:gpu_cache_usage_perc > 90
  for: 1m
  annotations:
    summary: "KV cache > 90% — risk of request preemption"

- alert: GPUTempHigh
  expr: DCGM_FI_DEV_GPU_TEMP > 83
  for: 2m
  annotations:
    summary: "GPU temperature > 83°C — thermal throttle imminent"
```

---

### 4.5 Model Management

**HuggingFace Cache** (immediate / v1):
- Mount `-v ~/.cache/huggingface:/root/.cache/huggingface` on every vLLM container.
- Pre-download large models before deployment to avoid startup delays.
- Use `huggingface-cli download` to populate cache on the host NVMe.

**Model Registry** (v2):
A lightweight catalog (YAML file or SQLite) tracking:
```yaml
models:
  - id: qwen2.5-32b-instruct
    hf_name: Qwen/Qwen2.5-32B-Instruct
    params: 32B
    dtype: bfloat16
    vram_required_gb: 64
    recommended_max_model_len: 32768
    quantization: null
    cached_on:
      - host: gpu-box-1 (54.x.x.x)
        path: /home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct
        
  - id: llama-3.1-8b-instruct
    hf_name: meta-llama/Llama-3.1-8B-Instruct
    params: 8B
    dtype: bfloat16
    vram_required_gb: 16
    recommended_max_model_len: 131072
    quantization: null
```

---

## 5. Deployment Workflow (Step by Step)

```
OPERATOR INPUT
──────────────
  instance_ip:    54.x.x.x
  ssh_key:        ~/.ssh/gpu-key.pem
  model:          Qwen/Qwen2.5-32B-Instruct
  deployment_id:  qwen32b-prod-01
  config_overrides:
    max-model-len: 16384
    gpu-memory-utilization: 0.92

PROVISIONING PHASE
──────────────────
  1. SSH connect + validate
     ├─ nvidia-smi  → check GPU type, VRAM, driver version
     └─ Fail fast if GPU insufficient for model

  2. Runtime setup (idempotent)
     ├─ Install nvidia-container-toolkit (if not present)
     ├─ Configure Docker NVIDIA runtime
     └─ Pull vllm/vllm-openai Docker image

  3. Model cache preparation
     ├─ Check HF cache for model weights
     ├─ If missing: huggingface-cli download <model>
     └─ Estimate startup time (100GB model ≈ 5-10 min from cache)

  4. Write config
     └─ Render vllm.yaml with base config + overrides → upload to instance

LAUNCH PHASE
────────────
  5. Start vLLM container
     └─ docker run ... --name vllm-<deployment_id>

  6. Wait for readiness
     └─ Poll GET /health every 10s (timeout 15min)

  7. Start monitoring exporters
     ├─ docker run dcgm-exporter  → :9400
     ├─ docker run node-exporter  → :9100
     └─ vLLM /metrics already on  :8000

  8. Start Nginx
     └─ docker run nginx with TLS config → :443

VALIDATION PHASE
────────────────
  9. Smoke test
     ├─ GET /health → 200
     ├─ GET /v1/models → lists model
     └─ POST /v1/chat/completions (1 request) → receives tokens

  10. Metrics check
      ├─ curl :9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
      ├─ curl :9100/metrics | grep node_memory
      └─ curl :8000/metrics | grep vllm:

REGISTRATION PHASE (optional)
──────────────────────────────
  11. Register endpoint
      └─ Update platform catalog / OpenRouter application
```

---

## 6. OpenRouter as a Subset

The OpenRouter provider journey (documented in `openrouter-provider-journey.md`) maps exactly to modules of this platform:

| OpenRouter Journey Phase | Platform Component |
|---|---|
| Phase 6: Benchmark | Observability → guidellm against full stack |
| Phase 7: Model selection | Model Registry → VRAM / throughput catalog |
| Phase 8: OpenAI-compat API | API Gateway → vLLM built-in server |
| Phase 9: Nginx + TLS + systemd | API Gateway → Nginx layer |
| Phase 9-alt: Container-first | Inference Engine → Docker deployment |
| Phase 10: Observability | Observability Stack |
| Phase 11: Dynamic pricing | FastAPI wrapper (custom logic) |
| Phase 12: Multi-box gateway | v2+ → Platform Gateway router |
| Phase 13: Go-live | Registration Phase of deploy workflow |

**Takeaway**: Complete the OpenRouter journey as planned. It is the first working end-to-end deployment of this platform. Every component you build for OpenRouter is reusable.

---

## 7. Phased Build Plan

### Phase 1 — Foundation (current state)
**Status**: In progress  
**What exists**: Docker image, Makefile, vllm.yaml, init script, verify script, EC2 with T4 GPU

Tasks remaining:
- [ ] Fix NVIDIA Container Toolkit on EC2 (resolve `could not select device driver ""` error)
- [ ] Confirm: `make run` + `make verify` works end-to-end
- [ ] Complete benchmark baseline (Phase 6 of OpenRouter journey)

---

### Phase 2 — Config-Driven Provisioning
**Goal**: Deploy a model on any SSH-accessible GPU in one command

Deliverables:
- `scripts/provision.sh <ip> <ssh_key> <model> [overrides]`
- Idempotent — safe to run twice
- Validates GPU before attempting model load
- Polls `/health` until ready

Stack shape:
```
[GPU Instance]
  └── vLLM container (model serving)
  └── DCGM Exporter container
  └── Node Exporter container
  └── Nginx container (TLS)
```

---

### Phase 3 — Observability
**Goal**: Every deployment auto-wires to a monitoring stack

Deliverables:
- `docker-compose.monitoring.yml` — Prometheus + Grafana + Alertmanager
- Grafana dashboards (JSON export): GPU health, vLLM engine, SLO view
- Alert rules for TTFT, KV cache, GPU temp, error rate
- README: how to access dashboards from laptop (SSH tunnel or public port)

---

### Phase 4 — Multi-Model Support
**Goal**: Multiple models available on the platform (possibly same box, possibly different boxes)

Deliverables:
- `models.yaml` — model registry (catalog with VRAM requirements, cache locations)
- Scripts to manage model lifecycle (load, unload, check cache)
- Support for quantized variants (AWQ, GPTQ) for fitting bigger models on smaller GPUs
- Decision guide: which model for which GPU (T4 vs A100 vs H100)

---

### Phase 5 — Platform Gateway
**Goal**: Single public endpoint, routes by model ID to correct backend

```
https://api.myplatform.com/v1/chat/completions
  model: qwen2.5-32b    → routes to gpu-box-1:8000
  model: llama-3.1-8b   → routes to gpu-box-2:8000
```

Deliverables:
- FastAPI router service with model → upstream mapping
- Health-aware routing (removes unhealthy backends)
- Unified `/v1/models` response aggregating all backends

---

### Phase 6 — Provider Integrations
**Goal**: Platform is visible to external traffic sources

Deliverables:
- OpenRouter registration (completing the existing journey doc)
- Probe traffic handling and reliability monitoring
- Dynamic pricing endpoint (load-aware)

---

### Phase 7 — Cost & Efficiency Layer
**Goal**: Understand revenue per GPU-hour, act on it

Deliverables:
- Tokens-per-dollar tracking (Grafana dashboard)
- Idle detection: alert when GPU utilization < 5% for > 30min
- Graceful idle scale-down (if using spot/on-demand rentals)

---

## 8. Technology Choices

| Layer | Choice | Rationale |
|---|---|---|
| Inference engine | **vLLM** | Best OSS performance, OpenAI-compat, active project, matches our existing work |
| Container runtime | **Docker** | Already used, simple, portable |
| API compatibility | **OpenAI API** | Universal client support, OpenRouter requires it |
| Reverse proxy | **Nginx** | Proven, easy TLS, low overhead |
| Metrics | **Prometheus** | Industry standard, vLLM ships /metrics natively |
| Dashboards | **Grafana** | Works natively with Prometheus |
| Provisioning | **Shell + SSH** (v1), **Ansible** (v2) | Minimal dependencies to start |
| GPU metrics | **DCGM Exporter** | Official NVIDIA Prometheus exporter |
| OS | **Ubuntu 22.04 LTS** or **Amazon Linux 2023** | Both supported; Ubuntu has better DCGM packages |

---

## 9. GPU Provider Matrix

The platform is provider-agnostic — it only needs SSH access to a machine with an NVIDIA GPU.

| Provider | GPU Options | Hourly Cost (est.) | Notes |
|---|---|---|---|
| AWS EC2 | g4dn.xlarge (T4 16GB) | ~$0.53/hr | Current setup; easy to provision |
| AWS EC2 | p3.2xlarge (V100 16GB) | ~$3.06/hr | Older Volta; FP16 only |
| RunPod | RTX 4090 (24GB) | ~$0.44/hr | Good price/perf for 7B-32B |
| RunPod | A100 80GB | ~$1.99/hr | Large model support |
| CoreWeave | H100 80GB | ~$2.21/hr | Fastest; needed for 70B+ |
| Lambda Labs | A10 24GB | ~$0.60/hr | Stable, good for mid-range models |
| Bare metal | Any NVIDIA GPU | Fixed cost | Best $/token at scale |

Provisioner supports all of these — just provide IP + SSH key.

---

## 10. Security Considerations

- **API Key**: vLLM supports `--api-key` (set via `VLLM_API_KEY` env var). Always set in production.
- **Nginx TLS**: Never expose vLLM port 8000 directly to the internet. Always terminate TLS at Nginx.
- **Metrics endpoints**: Restrict `/metrics` to internal IPs only (monitoring network or VPN). Never expose raw Prometheus metrics publicly.
- **SSH key management**: Use dedicated deploy keys per instance. Rotate after provisioning is done if using ephemeral keys.
- **HuggingFace tokens**: Pass via env var (`HF_TOKEN`), never bake into Docker images or commit to git.
- **Model security**: Only load models from trusted HuggingFace repos. Enable `--trust-remote-code` only for known, audited models.
- **Rate limiting**: Configure Nginx rate limiting to prevent abuse and protect GPU resources.

---

## 11. Current State → Next Actions

```
Today's stack (Phase 1):
  EC2 T4 GPU
  └── Dockerfile + Makefile (manual docker run)
  └── vllm.yaml (config)
  └── init-script.sh (entrypoint)
  └── verify.sh (smoke test)
  └── scripts/vllm_test_client.py
  └── tests/benchmark_*.sh

Immediate blockers to resolve:
  1. Fix NVIDIA Container Toolkit → docker run --gpus all must work
  2. Validate full stack: make build && make run && make verify
  3. Run Phase 6 benchmarks (guidellm) → establish baseline SLO numbers

Next build target (Phase 2):
  1. scripts/provision.sh — SSH-based single-command deploy
  2. docker-compose.yml — combine vLLM + Nginx into one file
  3. deploy monitoring stack alongside vLLM

OpenRouter journey continues in parallel:
  → Phase 8: confirm OpenAI-compat API surface
  → Phase 9-alt: docker-compose stack (containerize nginx too)
  → Phase 10: wire up Prometheus + Grafana (this is Phase 3 of platform)
```

---

## 12. Open Questions / Discussion Points

These are areas where the design intentionally defers a decision:

1. **Provisioning tooling**: Start with raw shell scripts over SSH or go directly to Ansible? Ansible is cleaner and idempotent but adds a dependency.

2. **Monitoring co-location**: Run Prometheus/Grafana on the same GPU box or on a separate cheap CPU instance? Co-location is simpler; separate is cleaner and survives GPU instance restarts.

3. **Multi-model on one GPU**: Two small models on one 96GB GPU (e.g., 8B + 32B) vs one large model? Affects KV cache allocation and scheduling complexity.

4. **FastAPI wrapper vs vLLM direct**: For OpenRouter, do we need a FastAPI wrapper for custom logic (pricing, auth, logging) or is Nginx + vLLM directly sufficient?

5. **Model warm-up strategy**: Accept slow first-request latency after startup or pre-warm with synthetic requests? Matters for OpenRouter reliability SLOs.

6. **Spot/preemptible instances**: Lower cost but restart risk. How to handle model reload gracefully? Checkpoint state? Or just accept cold starts?

---

*Document version: 0.1 — initial design draft*
*Related docs: `openrouter-provider-journey.md`, `vllm-learning-notes.md`, `vllm_setup_and_model_notes.md`*
