# OpenRouter Provider Journey — Labs & Learning

> Companion to `vllm-learning-notes.md`.
> Goal: go from "I understand vLLM" → "I run a production OpenRouter inference provider on 1 rented GPU".
> Style: **lab-first, understand-by-doing**. Each step has a *concept*, a *lab*, and a *check*.

---

## The North Star

Become an OpenRouter inference provider:
- Rent 1 GPU (target: RTX Pro 6000, 96GB, ~$500/mo).
- Serve 1–2 open-weight models via vLLM.
- Wrap with a FastAPI gateway that speaks OpenAI-compatible `/v1/models` + `/v1/chat/completions`.
- Nginx + TLS in front. systemd underneath.
- Apply to OpenRouter, pass their probe traffic, stay above their reliability threshold.
- Break even at ~0.2% GPU utilization (per PDF math). Everything above that is margin.

---

## Roadmap (this doc)

| Phase | Focus | Deliverable |
|---|---|---|
| 6  | Benchmark discipline: `vllm bench` vs **guidellm** | Reproducible SLO report on my current setup |
| 6-alt | Bench from outside the box (container-to-container / remote VPS) | Public-endpoint SLO numbers matching Phase 6 |
| 7  | Model selection for a single 96GB GPU | Shortlist + measured throughput per model |
| 8  | Provider-readiness: OpenAI-compat surface | Local FastAPI gateway serving `/v1/models` + streaming completions |
| 9  | Reliability layer: Nginx + TLS + systemd | 24/7 auto-recover stack, tested by killing processes |
| 9-alt | **Container-first deployment (recommended)** | `docker-compose.yml` stack, portable across rentals |
| 10 | Observability: metrics, load, cost tracking | Grafana or plain Prom + tokens/$ dashboard |
| 11 | Dynamic pricing gateway | Load-aware `/v1/models` pricing that respects OpenRouter's stability rules |
| 12 | Multi-box gateway pattern | One public endpoint routing to N GPU boxes by model id |
| 13 | Go-live checklist + OpenRouter application | Submitted application, passing probe traffic |

Each phase = concept + lab + check. Do not skip the check.

---

# Phase 6: Benchmark discipline — `vllm bench` vs `guidellm`

## 6.1 Why re-benchmark?

Earlier notes used `vllm bench` for engine tuning. For a **provider**, the question changes from:
- "Is my engine fast?" → **"Under realistic OpenRouter-shaped traffic, do I hit SLOs at price X?"**

OpenRouter's router cares about **TTFT p95, inter-token latency (ITL), and reliability**. Not peak throughput on a synthetic burst.

## 6.2 Tool comparison

| Aspect | `vllm bench serve` | `guidellm` |
|---|---|---|
| Origin | Built into vLLM | vllm-project/guidellm — separate tool, provider/SLO focus |
| Traffic model | Fixed concurrency or QPS ramp | **Rate sweeps** (constant / Poisson / concurrent) with SLO targets |
| Metrics | TTFT, TPOT, throughput | TTFT, ITL, TPOT, E2E, tokens/req — with **p50/p95/p99 distributions** |
| Datasets | ShareGPT, random, custom | Emulated (configurable in/out lengths), HuggingFace datasets, files |
| Output | stdout / JSON | JSON + **HTML report** + console tables |
| Best for | Engine-level A/B ("does prefix cache help?") | **Provider readiness** ("can I sustain 20 req/s at TTFT p95 < 500ms?") |
| Backend | vLLM only | Any **OpenAI-compatible endpoint** (so it also tests your gateway/Nginx!) |

**Verdict for our journey**: guidellm is primary. It benches your *whole stack* — vLLM + gateway + Nginx + TLS — exactly like OpenRouter's probe will hit it. Keep `vllm bench` for isolating engine changes.

## 6.3 Lab 6a — install and baseline with guidellm

```bash
# On your GPU box, in the same venv as vllm
pip install guidellm

# vLLM already running on :8000 (from earlier notes)
# Quick smoke: 30s constant rate, 2 req/s
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen/Qwen2.5-32B-Instruct \
  --data "prompt_tokens=512,output_tokens=256" \
  --rate-type constant \
  --rate 2 \
  --max-seconds 30
```

**What to read in the output:**
- `TTFT p95` — this is what OpenRouter's TTFT metric watches.
- `ITL p95` (inter-token latency) — smoothness of streaming.
- `Requests/sec successful` — did any drop?
- `Output tokens/sec` — your revenue rate.

## 6.4 Lab 6b — the SLO sweep (the real test)

Find the **max sustainable rate** where TTFT p95 stays under a target (say 500ms):

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen/Qwen2.5-32B-Instruct \
  --data "prompt_tokens=1024,output_tokens=256" \
  --rate-type sweep \
  --max-seconds 60 \
  --output-path ./guidellm-sweep.json
```

`sweep` mode ramps from low to overload automatically. Look for the "knee" where TTFT p95 breaks your budget.

## 6.5 Check ✅

You can answer:
1. What is my TTFT p95 at 1 req/s? At 5? At 20?
2. At what QPS does TTFT p95 cross 500ms?
3. What's my sustained output-tokens/sec at the knee?

Write these three numbers down. They gate every decision in later phases.

## 6.6 Optional: keep `vllm bench` for A/B

When you change ONE knob (e.g. `--enable-prefix-caching`), use `vllm bench serve` — fewer moving parts, faster iteration:

```bash
vllm bench serve --model Qwen/Qwen2.5-32B-Instruct \
  --dataset-name sharegpt --num-prompts 500 --request-rate 5
```

Compare before/after. Then re-run the guidellm sweep to confirm the end-to-end win.

---

# Phase 6-alt: Bench from *outside* the box (container-to-container)

## Why this matters
OpenRouter's probe hits you from the *public internet* — through Nginx, TLS, and any host-level networking. Benching from the same host as vLLM (Phase 6) misses:
- TLS handshake cost on TTFT
- Nginx buffering bugs
- Docker bridge / host networking overhead
- Host firewall / rate-limit surprises

Running guidellm in a **separate container** on the same host — or better, from a different machine — is a much closer proxy to OpenRouter's actual probe.

## Lab 6-alt — guidellm in a container against vLLM in a container

Assume vLLM is running in a container publishing port 8000 on the host.

```bash
# Simplest: run guidellm ad-hoc via a Python image
docker run --rm --network host \
  -v $PWD/reports:/reports \
  python:3.11-slim bash -c "
    pip install -q guidellm && \
    guidellm benchmark \
      --target http://localhost:8000 \
      --model Qwen/Qwen2.5-32B-Instruct \
      --data 'prompt_tokens=1024,output_tokens=256' \
      --rate-type sweep \
      --max-seconds 60 \
      --output-path /reports/sweep.json
  "
```

**Better** — hit your public HTTPS endpoint (once Phase 9-alt is up):
```bash
guidellm benchmark \
  --target https://api.yourdomain.com \
  --model qwen/qwen2.5-32b-instruct \
  --rate-type poisson --rate 5 --max-seconds 300
```

**Best** — run guidellm from a *different* cheap VPS in a *different* region. This measures TTFT the way a real client (or OpenRouter's probe) sees it.

## Check ✅

Compare three numbers for the same rate:
| Vantage point | TTFT p95 |
|---|---|
| Same host (Phase 6) | X ms |
| Container on same host, via public URL | Y ms |
| Remote VPS, via public URL | Z ms |

Gap between X and Z = your **networking + TLS + Nginx tax**. If it's >50ms, tune before applying to OpenRouter.

---

# Phase 7: Model selection for one 96GB GPU

## 7.1 The constraint

96GB VRAM = model weights + KV cache + activations. Rule of thumb:
- BF16: ~2 GB per 1B params.
- FP8/INT8: ~1 GB per 1B params.
- Leave **at least 30–40%** for KV cache if you want long context + concurrency.

## 7.2 Shortlist to bench

| Model | Params | Precision | Est. weights | Fits with headroom? | Why it's interesting |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 8B | BF16 | ~16GB | ✅ tons of KV room | Cheap, high TPS, RAG-friendly |
| Qwen2.5-32B-Instruct | 32B | BF16 | ~64GB | ✅ moderate KV | Strong general model, 32k ctx |
| DeepSeek-R1-Distill-Qwen-32B | 32B | BF16 | ~64GB | ✅ | Reasoning niche, high demand |
| Llama-3.1-70B-Instruct | 70B | FP8 | ~70GB | ⚠️ tight — reduce ctx | Prestige model, differentiator |
| Qwen2.5-Coder-32B | 32B | BF16 | ~64GB | ✅ | Coding niche, sticky users |

## 7.3 Lab 7 — bench each candidate

For each model:
1. `vllm serve <model> --gpu-memory-utilization 0.95 --max-model-len <ctx>`
2. Run the Phase 6 guidellm sweep.
3. Record: TTFT p95 at 5 req/s, max sustained QPS, tokens/sec at knee.

Fill this table (do it yourself — don't trust the PDF numbers):

| Model | Max ctx served | TTFT p95 @ 5 req/s | Knee QPS | Sustained out tok/s |
|---|---|---|---|---|
| Llama-3.1-8B | | | | |
| Qwen2.5-32B | | | | |
| Llama-3.1-70B FP8 | | | | |

## 7.4 Check ✅

Pick your **launch model** using: (tokens/sec × market price) − ($500/mo cost). The winner is the one with highest projected margin *and* competitive TTFT — OpenRouter routes on both.

---

# Phase 8: Provider-readiness — the OpenAI-compat surface

## 8.1 What OpenRouter actually probes

Per the PDF and OpenRouter docs, they hit:
- `GET /v1/models` — expects JSON with `id`, `pricing.prompt`, `pricing.completion`, `context_length`.
- `POST /v1/chat/completions` — must stream via SSE, must return `usage` counts even for streams.

vLLM already speaks `/v1/chat/completions` correctly. The `/v1/models` response is **not** OpenRouter-shaped by default (no pricing). So we need a gateway.

## 8.2 Lab 8a — minimal FastAPI gateway (local)

Create `gateway.py`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx, os

app = FastAPI()
VLLM = os.getenv("VLLM_URL", "http://127.0.0.1:8001")

PRICING = {
    "qwen/qwen2.5-32b-instruct": {
        "vllm_model": "Qwen/Qwen2.5-32B-Instruct",
        "prompt": "0.00000008",      # $0.08 / M input
        "completion": "0.00000010",  # $0.10 / M output
        "context_length": 32768,
    }
}

@app.get("/v1/models")
async def models():
    data = [{
        "id": pid,
        "object": "model",
        "context_length": p["context_length"],
        "pricing": {"prompt": p["prompt"], "completion": p["completion"]},
        "supported_features": ["tools", "json_mode"],
    } for pid, p in PRICING.items()]
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    pid = body.get("model")
    if pid not in PRICING:
        return JSONResponse({"error": f"model {pid} not served"}, status_code=404)
    body["model"] = PRICING[pid]["vllm_model"]  # rewrite to vLLM's internal name

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request("POST", f"{VLLM}/v1/chat/completions", json=body)
    r = await client.send(req, stream=True)
    return StreamingResponse(r.aiter_raw(), status_code=r.status_code,
                             media_type=r.headers.get("content-type"))
```

Run:
```bash
# vLLM on 8001 (change your existing serve to --port 8001)
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

## 8.3 Lab 8b — verify streaming + usage

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen2.5-32b-instruct","stream":true,
       "messages":[{"role":"user","content":"count to 5"}]}'
```

You should see SSE `data:` chunks and a final chunk with `usage` populated. **If `usage` is missing on stream, OpenRouter rejects you.** Add `"stream_options": {"include_usage": true}` when needed — verify vLLM version supports it.

## 8.4 Lab 8c — re-run guidellm through the gateway

Point guidellm at port 8000 (gateway) instead of 8001 (vLLM). Compare numbers to Phase 6.

**Expected**: TTFT rises ~1–5ms per hop. If it rises more, your gateway is buffering — check `StreamingResponse` chunk flushing.

## 8.5 Check ✅

- `GET /v1/models` returns pricing in the exact fractional-decimal format.
- Streaming works end-to-end.
- Gateway adds < 10ms to TTFT p95 vs raw vLLM.

---

# Phase 9: Reliability — Nginx + TLS + systemd

## 9.1 Concept

OpenRouter deprioritizes providers with reliability dips. You need:
- **TLS termination** (they will not route to plain HTTP).
- **Auto-restart** on crash (systemd).
- **No output buffering** at the proxy (breaks streaming).

## 9.2 Lab 9a — systemd units

Two unit files, `vllm.service` and `gateway.service` — see PDF for exact syntax. Key points to *understand*, not just copy:

- `After=network.target nvidia-persistenced.service` — order matters; GPU driver must be ready.
- `Restart=always`, `RestartSec=5` — must be present. Without this, one OOM = OpenRouter drops you for the day.
- `Environment=CUDA_VISIBLE_DEVICES=0` — pins to a GPU; important once you add a second card.

**Chaos check**: `sudo systemctl kill vllm` — service should come back within 10s. Re-run a curl. It should succeed.

## 9.3 Lab 9b — Nginx streaming config

Critical directives (paste-then-understand):

```nginx
proxy_http_version 1.1;
proxy_set_header Connection "";
proxy_buffering off;         # <-- kills SSE if omitted
proxy_cache off;
chunked_transfer_encoding on;
proxy_read_timeout 600s;     # long completions
```

Test with TLS via certbot (`certbot --nginx -d yourdomain.com`).

## 9.4 Lab 9c — the OpenRouter probe simulation

Simulate 24h of load:
```bash
guidellm benchmark --target https://yourdomain.com \
  --model qwen/qwen2.5-32b-instruct \
  --data "prompt_tokens=1024,output_tokens=256" \
  --rate-type poisson --rate 3 --max-seconds 3600
```

Watch `journalctl -u vllm -u gateway -f` in parallel. Any restart during the hour = fail. Investigate before going further.

## 9.5 Check ✅

- Kill vllm → recovers in < 10s.
- 1-hour Poisson load → zero failed requests, no restarts.
- TLS grade A on ssllabs.com.

---

# Phase 9-alt: Container-first deployment (recommended path)

> Replaces Phase 9's bare-metal systemd approach. Choose one; don't run both.
> Rationale: portability across GPU rentals (Vast → RunPod → GPU-mart) + pinned reproducibility + clean rollback.

## 9-alt.1 Concept

```
┌─────────────── Host (GPU box, Ubuntu 22.04) ───────────────┐
│  NVIDIA driver + nvidia-container-toolkit + docker         │
│                                                             │
│  ┌──────────┐  ┌─────────┐  ┌───────┐                       │
│  │  nginx   │◀─│ gateway │◀─│ vllm  │                       │
│  │  :443    │  │ :8000   │  │ :8001 │                       │
│  └──────────┘  └─────────┘  └───────┘                       │
│      ▲              (internal docker network)               │
└──────┼──────────────────────────────────────────────────────┘
       │
    OpenRouter probe / real traffic
```

- vLLM: **official image**, pinned tag, weights mounted from host volume.
- Gateway: **custom slim image** (~100MB), your `gateway.py`.
- Nginx: official image or on host — either works.

## 9-alt.2 Prereqs on the rented box

Verify before signing a monthly lease:
```bash
nvidia-smi                                     # driver works
docker --version                                # >= 24
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Last command must show GPUs inside the container. If it fails, the host lacks `nvidia-container-toolkit` — either install it or pick a different provider.

## 9-alt.3 Lab 9-alt-a — `docker-compose.yml` for the whole stack

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.6.3         # PIN this — do not use :latest
    restart: unless-stopped
    runtime: nvidia
    ipc: host                               # required for tensor parallelism
    shm_size: "16gb"                        # default 64MB will crash TP
    environment:
      HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN}
    volumes:
      - ./models:/root/.cache/huggingface   # persist weights across restarts
    command: >
      --model Qwen/Qwen2.5-32B-Instruct
      --port 8001
      --host 0.0.0.0
      --gpu-memory-utilization 0.95
      --max-model-len 32768
      --api-key ${VLLM_INTERNAL_KEY}
      --enable-prefix-caching
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 15s
      timeout: 5s
      start_period: 180s                    # weights download can be slow
      retries: 3

  gateway:
    build: ./gateway                        # your custom Dockerfile
    restart: unless-stopped
    environment:
      VLLM_URL: http://vllm:8001
      VLLM_INTERNAL_KEY: ${VLLM_INTERNAL_KEY}
    depends_on:
      vllm:
        condition: service_healthy
    ports:
      - "127.0.0.1:8000:8000"               # bind to loopback; nginx exposes publicly

  nginx:
    image: nginx:1.27
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - gateway
```

`.env`:
```
HF_TOKEN=hf_xxx
VLLM_INTERNAL_KEY=some-long-random-secret
```

## 9-alt.4 Lab 9-alt-b — custom gateway image

`gateway/Dockerfile`:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway.py .
RUN useradd -u 1001 -m app && chown -R app /app
USER app
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]
```

`gateway/requirements.txt`:
```
fastapi==0.115.*
uvicorn[standard]==0.32.*
httpx==0.27.*
```

Add a `/health` route to `gateway.py`:
```python
@app.get("/health")
async def health(): return {"ok": True}
```

## 9-alt.5 Lab 9-alt-c — Nginx as a container

`nginx/conf.d/api.conf`:
```nginx
server {
  listen 80;
  server_name api.yourdomain.com;
  return 301 https://$host$request_uri;
}
server {
  listen 443 ssl;
  server_name api.yourdomain.com;
  ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

  location / {
    proxy_pass http://gateway:8000;         # docker service DNS
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_buffering off;                     # critical for SSE streaming
    proxy_cache off;
    chunked_transfer_encoding on;
    proxy_read_timeout 600s;
  }
}
```

TLS: run `certbot` on the host once (before starting nginx container), then mount `/etc/letsencrypt`. Or use the `certbot/certbot` container in DNS-01 mode.

## 9-alt.6 Chaos check — the real reliability test

```bash
# Bring stack up
docker compose up -d

# Wait for vLLM to become healthy
watch -n 2 'docker compose ps'

# Baseline: run guidellm at 3 req/s for 10 min
# In another terminal, kill vLLM mid-load:
docker kill $(docker compose ps -q vllm)

# Watch: gateway health should stay green, vllm restarts within ~10s
# guidellm will report a small failure window — but recovery should be automatic
docker compose logs -f vllm gateway
```

Repeat for `gateway`, then `nginx`. Any container that doesn't come back cleanly = fix before Phase 13.

## 9-alt.7 Cold-start optimization (optional)

Model weights are 30–150GB. First `docker run` on a new box will HF-download for many minutes → OpenRouter's probe times out.

Options:
- **Pre-pull weights** into `./models` volume before starting vLLM (`huggingface-cli download <model> --local-dir ./models/...`).
- **Bake weights into a private image** and push to a nearby registry. Trades image size (~70GB) for zero-download boot.
- **`start_period: 300s`** in healthcheck so orchestrator waits during first-run download.

## 9-alt.8 Check ✅

- `docker compose up -d` on a fresh box → serving traffic in < 5 min (with pre-pulled weights).
- Killing any single container → auto-recovery in < 15s.
- guidellm run through https://api.yourdomain.com matches Phase 6 numbers within ~5% (Docker overhead is minimal with `ipc: host`).
- Same `docker-compose.yml` deploys identically to a second GPU rental → **your business is now portable**.

## 9-alt.9 When NOT to containerize

- Provider blocks the NVIDIA container runtime (rare — check before renting).
- You need to pass exotic driver-level flags (MIG partitioning, custom NCCL). Bare-metal simpler.
- Team of 1, one box forever, allergic to Docker. Fine — use Phase 9 systemd path.

---

# Phase 10: Observability

## 10.1 What to watch

- **From vLLM** (`/metrics`): running/waiting requests, KV cache usage %, prefix cache hit rate, token throughput.
- **From gateway**: request count, streaming duration, error rate per model.
- **Business**: tokens billed × price − hourly GPU cost, updated live.

## 10.2 Lab 10 — minimal Prometheus stack

- Add prometheus scrape for `http://localhost:8001/metrics` (vLLM).
- Add `prometheus_fastapi_instrumentator` to gateway.
- Grafana dashboard with 4 panels: TTFT p95, active requests, KV cache %, tokens/hr.

Skip Grafana if impatient — even a `watch -n 5 'curl -s :8001/metrics | grep vllm:num_requests'` teaches a lot.

## 10.3 Check ✅

You can answer at any moment: "how many tokens/min am I generating, and what's my $/hour right now?"

---

# Phase 11: Dynamic pricing (carefully)

## 11.1 The trap

PDF suggests millisecond-level surge pricing. **Don't.** OpenRouter fingerprints price stability; whipsaw pricing gets you flagged.

## 11.2 Safe pattern

- Recompute price every **N minutes** (e.g. 15 min) via a background task.
- Base on **rolling avg concurrency**, not instantaneous.
- Cap variation to ±20% off baseline.
- Log every price change.

## 11.3 Lab 11 — background repricer

Add to `gateway.py`:
```python
import asyncio, time
from collections import deque

_load_window = deque(maxlen=180)   # 15 min of 5s samples
_current_multiplier = 1.0

async def price_loop():
    global _current_multiplier
    while True:
        await asyncio.sleep(5)
        _load_window.append(_active_requests)
        avg = sum(_load_window)/len(_load_window) if _load_window else 0
        if avg < 1:   _current_multiplier = 0.90
        elif avg > 15: _current_multiplier = 1.15
        else:         _current_multiplier = 1.00
```

Then use `_current_multiplier` in the `/v1/models` handler. Instrument every change in Grafana.

## 11.4 Check ✅

Over a 24h synthetic load: fewer than ~6 price changes, all within ±20% of baseline.

---

# Phase 12: Multi-box gateway

When one 96GB card fills up, add a second box. Architecture:

```
[OpenRouter] --> https://api.yours.com --> Nginx --> FastAPI gateway
                                                        |-- Box 1 vLLM (Qwen-32B) @ localhost:8001
                                                        |-- Box 2 vLLM (Llama-70B) @ 10.0.0.2:8000
```

Gateway routes by `model` field. Lab: extend `PRICING` dict with a `backend_url` per model, forward accordingly. Health-check each backend every 10s; drop it from `/v1/models` if unhealthy.

---

# Phase 13: Go-live checklist

Before applying to OpenRouter:

- [ ] `/v1/models` returns pricing in correct format
- [ ] Streaming completions include `usage` counts
- [ ] TLS grade A, valid cert, auto-renew tested
- [ ] systemd auto-recovery verified via kill test (or `docker compose` chaos check if using 9-alt)
- [ ] 1h Poisson load: zero failures, no restarts
- [ ] guidellm sweep: TTFT p95 documented at 3 rate levels
- [ ] TTFT gap between local-host and remote-VPS benches is < 50ms (Phase 6-alt)
- [ ] Public domain + static IP
- [ ] Privacy/logging policy page live
- [ ] Business email + payment details ready
- [ ] Rolling backup of gateway.py + nginx conf + systemd units in a git repo

Then submit at https://openrouter.ai/docs/community/provider (Become a Provider).

---

# Appendix A: guidellm cheat sheet

```bash
# Constant rate
guidellm benchmark --target URL --model M --rate-type constant --rate 5

# Poisson (realistic)
guidellm benchmark --target URL --model M --rate-type poisson --rate 5

# Sweep (find the knee)
guidellm benchmark --target URL --model M --rate-type sweep --max-seconds 60

# Concurrency mode
guidellm benchmark --target URL --model M --rate-type concurrent --rate 10

# Use a HF dataset
guidellm benchmark --target URL --model M --data "hf://ShareGPT..." --data-args '{"split":"train"}'

# HTML report
guidellm benchmark ... --output-path report.json && guidellm report report.json
```

# Appendix B: Cost math template

```
Monthly cost:            $500
Daily breakeven tokens:  cost / (net_price_per_M / 1e6)
                         = $16.66 / $0.10-per-M
                         = 166.6 M tokens/day

At <sustained_out_tok_s> from Phase 6:
  Max daily tokens = <sustained_out_tok_s> * 86400
  Utilization to break even = 166.6M / max_daily

Fill in with YOUR measured numbers, not the PDF's optimistic 965 tok/s.
```

# Appendix C: Open questions to resolve as you go

1. Do I own a domain suitable for a public API? If not — buy before Phase 9.
2. Payment: does the provider accept my region? Check before renting.
3. Model licensing: Llama-3.1 has an acceptable-use policy; confirm commercial-serving is compliant.
4. Data retention: OpenRouter requires an explicit policy. Draft one *before* application.

---

_Last updated: 2026-08-03. Update this file phase-by-phase — treat each Check ✅ as your gate to advance._
