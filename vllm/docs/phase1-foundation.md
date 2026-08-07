# Phase 1 — Foundation: Step-by-Step Execution Guide

> **Goal**: From a bare GPU instance (SSH access only) to a fully running, observable LLM serving stack.
>
> **What you'll have at the end**:
> - vLLM serving a model via OpenAI-compatible API
> - Nginx in front (HTTP initially, TLS when ready)
> - GPU, system, and inference metrics flowing into Prometheus
> - Grafana dashboards live
> - Smoke-tested and benchmark-ready
>
> **Time estimate**: 45–90 minutes first time (most of that is model download). Subsequent deploys: ~15 min.

---

## Prerequisites

On your **local machine** (where you run make/scripts):
- SSH key for the GPU instance
- This repository cloned locally
- `rsync` or `scp` available (for uploading files to the instance)

On the **GPU instance** (we'll verify with preflight):
- NVIDIA GPU with ≥15GB VRAM
- Fresh Ubuntu 22.04 or Amazon Linux 2023
- Docker installed (or we install it)
- Internet access (for HuggingFace downloads and Docker image pulls)

---

## Architecture Reminder

```
[Internet / OpenRouter probe / your curl]
           │
           ▼ :443 (TLS) / :80 (dev)
    ┌─────────────────┐
    │      Nginx      │  ← docker-compose.yml (support stack)
    └────────┬────────┘
             │ http://vllm:8000  (Docker internal network: vllm_platform)
             ▼
    ┌─────────────────┐
    │     vLLM        │  ← make run  (separate, long-running, --gpus all)
    └────────┬────────┘
             │ /metrics
             ▼
    ┌─────────────────┐    ┌────────────────┐    ┌──────────────────┐
    │  DCGM Exporter  │    │  Node Exporter │    │   Prometheus     │
    │  :9400 (GPU)    │    │  :9100 (host)  │    │   :9090          │
    └─────────────────┘    └────────────────┘    └────────┬─────────┘
                                                          │
                                                 ┌────────▼─────────┐
                                                 │     Grafana       │
                                                 │     :3000         │
                                                 └──────────────────┘
```

**Two separate lifecycles**:
- `make run` / `make stop` → manages vLLM only (GPU container)
- `docker compose up/down` → manages everything else (nginx, monitoring)

This means you can restart nginx or update prometheus config without touching the vLLM engine (and its 10+ minute startup time).

---

## Step 1 — SSH into the Instance and Prepare the Workspace

```bash
# From your local machine
ssh -i ~/.ssh/your-gpu-key.pem ec2-user@<INSTANCE_IP>
# Ubuntu: ssh -i ~/.ssh/your-gpu-key.pem ubuntu@<INSTANCE_IP>
```

Run the ephemeral NVMe setup (AWS only — puts HF cache on the fast local SSD):
```bash
# Amazon Linux 2023
sudo bash scripts/amazon_linux.sh

# Ubuntu
sudo bash scripts/ubuntu_startup.sh
```

Verify the cache drive is mounted:
```bash
df -h /mnt/instance_store
# Expect: ~200–400GB free on g4dn.xlarge
```

Set `HF_HOME` to point at the NVMe cache (already done by the startup scripts above, but confirm):
```bash
echo $HF_HOME
# Should print: /mnt/instance_store/hf_cache
# If empty: export HF_HOME=/mnt/instance_store/hf_cache
```

Upload the repository to the instance (from your local machine):
```bash
rsync -avz --exclude '.git' --exclude '.DS_Store' \
    /path/to/local/vllm/ \
    ec2-user@<INSTANCE_IP>:~/vllm_ws/
```

---

## Step 2 — Install Docker (if not already installed)

Check if Docker is installed:
```bash
docker version
```

If not installed — **Ubuntu 22.04**:
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

If not installed — **Amazon Linux 2023**:
```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker run --rm hello-world
```

---

## Step 3 — Run the Preflight Check

The preflight script validates all hard requirements before you attempt any deployment.

```bash
cd ~/vllm_ws
bash scripts/preflight.sh
```

**Read the output carefully.** The preflight checks:

| Check | What it validates |
|---|---|
| `[1] NVIDIA DRIVER & GPU` | `nvidia-smi` works, driver ≥525, VRAM reported |
| `[2] DOCKER` | daemon running, version ≥24 |
| `[3] NVIDIA CONTAINER RUNTIME` | `--gpus all` passthrough works |
| `[4] DOCKER COMPOSE V2` | `docker compose` plugin available |
| `[5] DISK SPACE` | ≥50GB free at `HF_HOME` |
| `[6] NETWORK` | HuggingFace, Docker Hub reachable |
| `[7] PORT AVAILABILITY` | 8000, 9400, 9100, 9090, 3000, 80, 443 all free |
| `[8] UTILITY TOOLS` | curl, git, huggingface-cli |
| `[9] RUNNING CONTAINERS` | no conflicting containers already running |

---

## Step 3a — Fix: NVIDIA Container Runtime (most common issue)

If preflight fails on check `[3]` with:
```
[FAIL] NVIDIA Container Runtime NOT registered with Docker
```

You have two options:

**Option A — Auto-fix**:
```bash
bash scripts/preflight.sh --fix
```

This will detect your OS (Ubuntu vs Amazon Linux) and run the correct install commands automatically, then restart Docker.

**Option B — Manual fix**:

*Amazon Linux 2023*:
```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

*Ubuntu 22.04*:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the fix:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# You should see nvidia-smi output from inside a container
```

---

## Step 4 — Pre-Download the Model (Optional but Recommended)

For large models (>10B), pre-downloading avoids a very long first-startup wait and removes internet dependency from the actual serve command.

Install huggingface-cli if needed:
```bash
pip install -q huggingface_hub
```

Download the model to the NVMe cache:
```bash
# For a public model (no token needed)
HF_HOME=/mnt/instance_store/hf_cache \
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False

# For a gated model (Llama etc. — requires HF token with model access)
HF_HOME=/mnt/instance_store/hf_cache \
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --token hf_xxxx \
    --local-dir-use-symlinks False
```

**Approximate download times on EC2 (10 Gbps network)**:

| Model | Size | Download time |
|---|---|---|
| Qwen2.5-7B-Instruct (BF16) | ~15GB | ~2 min |
| Qwen2.5-32B-Instruct (BF16) | ~65GB | ~7 min |
| Llama-3.1-8B-Instruct (BF16) | ~16GB | ~2 min |
| Llama-3.1-70B-Instruct (FP8) | ~70GB | ~8 min |

Verify the download:
```bash
ls -lh $HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/
```

---

## Step 5 — Build the vLLM Docker Image

```bash
cd ~/vllm_ws
make build
```

This builds a thin wrapper on top of `vllm/vllm-openai:latest` (adds `yq` for config handling). The model is NOT baked in.

First build takes 5–10 minutes (large base image). Subsequent builds are fast (cached layers).

Verify:
```bash
docker images | grep my-vllm
# Expect: my-vllm   latest   <id>   <size>
```

---

## Step 6 — Start the Support Stack

The support stack (nginx, exporters, Prometheus, Grafana) starts independently of vLLM. Start it first so monitoring is ready as soon as vLLM comes up.

```bash
cd ~/vllm_ws

# Optional: set a Grafana admin password (do this before first start)
export GRAFANA_ADMIN_PASSWORD="your-secure-password"

make support-up
```

This runs `docker compose up -d` and starts:
- `platform-nginx` on ports 80, 443
- `platform-dcgm-exporter` on port 9400
- `platform-node-exporter` on port 9100
- `platform-prometheus` on port 9090
- `platform-grafana` on port 3000

Wait for all containers to become healthy:
```bash
docker compose ps
```

Expected output:
```
NAME                      STATUS                    PORTS
platform-dcgm-exporter    Up (healthy)              0.0.0.0:9400->9400/tcp
platform-grafana          Up (healthy)              0.0.0.0:3000->3000/tcp
platform-nginx            Up (healthy)              0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
platform-node-exporter    Up (healthy)              0.0.0.0:9100->9100/tcp
platform-prometheus       Up (healthy)              0.0.0.0:9090->9090/tcp
```

Verify Prometheus is scraping (GPU and host metrics should already be flowing):
```bash
# From another terminal on your laptop (open SSH tunnel first)
# ssh -L 9090:localhost:9090 ec2-user@<INSTANCE_IP>
# Then browse http://localhost:9090/targets
# You should see node and gpu jobs as "UP"
```

---

## Step 7 — Start the vLLM Engine

```bash
cd ~/vllm_ws

# Replace with your actual model and token
make run \
    VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    HF_TOKEN=hf_xxxx             # omit if model is pre-cached and public
```

This runs:
```bash
docker run -d \
    --name vllm-server \
    --gpus all \
    --restart unless-stopped \
    --network vllm_platform \      # ← connects to shared network with nginx
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    -e HF_TOKEN=hf_xxxx \
    my-vllm
```

**Wait for vLLM to be ready** — this is the slow step (model loading):

```bash
make verify
# OR watch logs directly:
make logs
```

Expected final log lines when ready:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Typical model load times** (from NVMe cache):

| Model | GPU | Load time |
|---|---|---|
| 7B BF16 | T4 (16GB) | ~3 min |
| 32B BF16 | A100 (80GB) | ~8 min |
| 70B FP8 | H100 (80GB) | ~12 min |

> If `make verify` times out (default 5 min), the model may be larger than the GPU can hold.
> Check `make logs` for OOM errors. Try reducing `max-model-len` or enabling quantization.

---

## Step 8 — Validate the Full Stack

### 8.1 Direct vLLM test (bypass nginx)

```bash
# Quick health check
curl http://localhost:8000/health

# List loaded model
curl http://localhost:8000/v1/models | python3 -m json.tool

# Test chat completion (streaming)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one line."}],
        "max_tokens": 30,
        "stream": false
    }' | python3 -m json.tool
```

Expected response contains `"choices"` with a generated message.

### 8.2 Via Nginx (full stack path)

```bash
# Health check through nginx
curl http://localhost:80/health

# Chat completion through nginx
curl http://localhost:80/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Name one planet. One word only."}],
        "max_tokens": 10
    }'
```

### 8.3 Test with the built-in test client

```bash
python3 scripts/vllm_test_client.py \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct
```

### 8.4 Metrics sanity check

```bash
# vLLM metrics — should show queue/cache gauges and request counters
curl -s http://localhost:8000/metrics | grep -E "^vllm:" | head -20

# GPU metrics — should show utilization after you sent a request
curl -s http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL

# Node metrics — should show CPU/RAM
curl -s http://localhost:9100/metrics | grep "^node_memory_MemAvailable"
```

---

## Step 9 — Access Grafana Dashboards

Open an SSH tunnel from your laptop:
```bash
ssh -N -L 3000:localhost:3000 \
       -L 9090:localhost:9090 \
    -i ~/.ssh/your-gpu-key.pem ec2-user@<INSTANCE_IP>
```

Then in your browser:
- **Grafana**: http://localhost:3000 — login `admin` / `<GRAFANA_ADMIN_PASSWORD>`
- **Prometheus**: http://localhost:9090 — targets should all be green

In Grafana, the Prometheus datasource is auto-provisioned. Import these community dashboards to get started immediately:

| Dashboard | Grafana ID | What it shows |
|---|---|---|
| NVIDIA DCGM Exporter Dashboard | `12239` | GPU util, VRAM, temp, power |
| Node Exporter Full | `1860` | CPU, RAM, disk, network |
| vLLM Monitoring | search "vllm" | TTFT, queue depth, KV cache |

Import a dashboard:
1. Grafana → **Dashboards → Import**
2. Enter the ID → **Load**
3. Select **Prometheus** as the datasource → **Import**

---

## Step 10 — Run the Baseline Benchmark

This is Phase 6 of the OpenRouter journey: establish your SLO numbers before any optimization.

Install guidellm (from the GPU instance):
```bash
pip install guidellm
```

**Lab 1 — Quick smoke benchmark (30 seconds)**:
```bash
guidellm benchmark \
    --target http://localhost:8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data "prompt_tokens=512,output_tokens=256" \
    --rate-type constant \
    --rate 2 \
    --max-seconds 30
```

**Lab 2 — SLO sweep (finds the knee point)**:
```bash
guidellm benchmark \
    --target http://localhost:8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data "prompt_tokens=1024,output_tokens=256" \
    --rate-type sweep \
    --max-seconds 120 \
    --output-path ./tests/phase1-baseline-sweep.json
```

**Lab 3 — Bench through nginx** (closer to how OpenRouter hits you):
```bash
guidellm benchmark \
    --target http://localhost:80 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data "prompt_tokens=512,output_tokens=256" \
    --rate-type constant \
    --rate 2 \
    --max-seconds 60
```

**Record your three baseline numbers**:

| Metric | Value | Notes |
|---|---|---|
| TTFT p95 at 1 req/s | ? ms | |
| TTFT p95 at peak sustainable rate | ? ms | |
| Max sustainable QPS (TTFT p95 < 500ms) | ? req/s | OpenRouter gate |
| Output tokens/sec at knee | ? tok/s | Revenue rate proxy |

> These numbers gate all future decisions: model selection, optimization choices, and OpenRouter application timing.

---

## Step 11 — Make status Check

At any time, check the full platform state with:

```bash
make status
```

Expected output when everything is healthy:
```
=== vLLM Container ===
NAMES          STATUS          PORTS
vllm-server    Up 47 minutes   0.0.0.0:8000->8000/tcp

=== Support Stack ===
NAME                      STATUS       PORTS
platform-dcgm-exporter    Up           0.0.0.0:9400->9400/tcp
platform-grafana          Up           0.0.0.0:3000->3000/tcp
platform-nginx            Up           0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
platform-node-exporter    Up           0.0.0.0:9100->9100/tcp
platform-prometheus       Up           0.0.0.0:9090->9090/tcp
```

---

## Phase 1 Checklist

| # | Task | Command | Done? |
|---|---|---|---|
| 1 | SSH in, mount NVMe | `sudo bash scripts/amazon_linux.sh` | ☐ |
| 2 | Install Docker | `curl -fsSL https://get.docker.com \| sudo sh` | ☐ |
| 3 | Run preflight check | `bash scripts/preflight.sh` | ☐ |
| 3a | Fix NVIDIA runtime (if needed) | `bash scripts/preflight.sh --fix` | ☐ |
| 4 | Pre-download model | `huggingface-cli download <model>` | ☐ |
| 5 | Build vLLM image | `make build` | ☐ |
| 6 | Start support stack | `make support-up` | ☐ |
| 7 | Start vLLM engine | `make run VLLM_MODEL=... HF_TOKEN=...` | ☐ |
| 8a | Test direct vLLM | `curl localhost:8000/v1/chat/completions` | ☐ |
| 8b | Test through nginx | `curl localhost:80/v1/chat/completions` | ☐ |
| 8c | Verify metrics flowing | `curl localhost:9400/metrics` | ☐ |
| 9 | Access Grafana | SSH tunnel → localhost:3000 | ☐ |
| 10 | Run baseline benchmark | `guidellm benchmark --rate-type sweep` | ☐ |
| 11 | Record baseline SLO numbers | See benchmark table above | ☐ |

---

## Troubleshooting

### vLLM container exits immediately

```bash
make logs
```

Common causes:
- **OOM**: `CUDA out of memory` → reduce `max-model-len` or `gpu-memory-utilization`
  ```bash
  make run VLLM_MODEL=... VLLM_MAX_MODEL_LEN=4096 VLLM_GPU_MEM_UTIL=0.85
  ```
- **Model not found**: `Repository not found` → check `VLLM_MODEL` spelling, set `HF_TOKEN` for gated models
- **No GPU**: `RuntimeError: No GPU available` → run `bash scripts/preflight.sh` to diagnose

### `docker run --gpus all` fails

```
Error: could not select device driver "" with capabilities: [[gpu]]
```

NVIDIA Container Runtime is not configured. Run:
```bash
bash scripts/preflight.sh --fix
```

### Nginx returns 502 Bad Gateway

vLLM is not yet up or failed to start.
```bash
# Check vLLM status
make logs

# Confirm vLLM joined the platform network
docker inspect vllm-server | grep -A 5 '"Networks"'
# Should show: "vllm_platform": { ... }
```

If the container isn't on the `vllm_platform` network (e.g., it was started before `make support-up`):
```bash
docker network connect vllm_platform vllm-server
```

### Prometheus shows target as DOWN

Check that the target containers are on the `vllm_platform` network and their ports are published:
```bash
docker compose ps
# All services should show "Up (healthy)"

# Test Prometheus can reach vLLM metrics
docker exec platform-prometheus \
    wget -qO- http://vllm:8000/metrics | head -5
```

### DCGM Exporter fails to start

```
Error: could not select device driver "" with capabilities: [[gpu]]
```

Same root cause as GPU passthrough failure — fix NVIDIA Container Runtime first.

Alternatively, if you don't need GPU metrics yet, comment out the `dcgm-exporter` service and its references in `config/prometheus/prometheus.yml`.

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/preflight.sh` | Host readiness validation (run first) |
| `scripts/amazon_linux.sh` | NVMe + env setup for Amazon Linux |
| `scripts/ubuntu_startup.sh` | NVMe + env setup for Ubuntu |
| `scripts/cuda_installation.sh` | Native NVIDIA driver install for Amazon Linux |
| `scripts/verify.sh` | Polls vLLM `/health` until ready |
| `scripts/vllm_test_client.py` | Manual test client for chat completions |
| `Dockerfile` | vLLM container image (thin wrapper on vllm-openai) |
| `Makefile` | `make run/stop/logs/verify/preflight/support-up/status` |
| `config/vllm.yaml` | vLLM serve configuration (model, memory, context) |
| `config/nginx/nginx.conf` | Nginx reverse proxy config |
| `config/prometheus/prometheus.yml` | Prometheus scrape targets |
| `config/prometheus/rules/slo_alerts.yml` | Alert rules (TTFT, KV cache, GPU temp) |
| `config/grafana/provisioning/datasources/prometheus.yml` | Auto-wires Prometheus into Grafana |
| `docker-compose.yml` | Support stack (nginx + exporters + prometheus + grafana) |

---

## What's Next (Phase 2+)

Once Phase 1 is stable and you have baseline benchmark numbers:

| Phase | Focus | Key deliverable |
|---|---|---|
| Phase 2 | Config-driven provisioning | `scripts/provision.sh` — one command deploy |
| Phase 3 | Observability hardening | Custom Grafana dashboards, alertmanager wired up |
| Phase 4 | Multi-model support | `models.yaml` registry, quantized variants |
| Phase 5 | Platform gateway | Single endpoint routing by model ID |
| Phase 6 | OpenRouter go-live | Provider registration, probe traffic passing |

---

*Related docs: `llm-serving-platform-design.md`, `openrouter-provider-journey.md`*
