# Week 5: two vLLM replicas, routing, and throughput scaling

**Week 5 results:** see the [findings and follow-ups](../../docs/findings/week5.md)
for the measured scaling, policy comparison, latency and reliability caveats.

## Goal and topology

Run **the same complete model on two independent single-GPU instances**.
This is replica scaling, not tensor parallelism: each request runs entirely on
one replica. Nothing here provisions, reconfigures, or stops the GPU hosts.

```text
                  +--> host A: vLLM + full model + one GPU
client --> router |
                  +--> host B: vLLM + full model + one GPU
```

The [config](config.yaml) compares:

| Experiment | Endpoints | Routing |
|---|---|---|
| single-a | A only | Single-replica baseline through the same router |
| single-b | B only | Check host-to-host variation |
| two-rr | A + B | Round-robin across healthy replicas |
| two-least | A + B | Fewest router-tracked in-flight requests; round-robin ties |

Use the **same GPU type, model revision, weight/KV dtype, vLLM image,
memory utilization, context length, and scheduling flags** on both hosts.
The model must fit on each GPU. An older single-GPU result is useful context,
but rerun these baselines to avoid version, workload, and network confounds.

## 1. Start the model on both GPU hosts

On **each** host, from the lab directory:

```bash
# Pin the SAME tested image tag/digest on both hosts, rather than mutable latest.
export VLLM_IMAGE='vllm/vllm-openai:<your-tested-tag>'
export MODEL=Qwen/Qwen3-8B
export MAX_LEN=8192 GPU_UTIL=0.90
make up
curl --fail localhost:8000/health
curl --fail localhost:8000/v1/models
docker compose images
```

The existing Compose setup enables prefix caching. The default benchmark uses
unique leading nonces so caching cannot accidentally favor a routing policy.
If you use a quantized model instead, use that same model and configuration on
both hosts and update `model` in the Week 5 config to its served model ID.
Pin the model revision too, for example by adding the same `--revision` to both
servers' commands, and record the flags and image digest with your findings.
Save each host's startup log and GPU details:

```bash
mkdir -p results/logs
docker compose logs --no-color vllm > results/logs/week5-startup.log
nvidia-smi > results/logs/week5-gpu.txt
```

Keep ports private: allow 8000 only from your load-generator/router host (ideally
on the same private network). Do not expose this unauthenticated lab router to
the public internet. If vLLM is configured with an API key, set the same `API_KEY`
on the benchmark host; inference, calibration, and metrics requests forward it.

**AWS:** use the existing commands with a different name for each GPU host:

```bash
make aws-up INSTANCE=replica-a
make aws-up INSTANCE=replica-b
make aws-ssh INSTANCE=replica-a    # start vLLM using the same settings above
make aws-ssh INSTANCE=replica-b    # start vLLM here too
make aws-status INSTANCE=replica-a
make aws-status INSTANCE=replica-b
```

Repeating `aws-up` with the same name resumes setup or restarts its stopped
instance; a different name creates another host if none exists.
The single `aws/.state` file tracks names separately. Every subsequent lifecycle,
SSH, sync, tunnel, and pull command must select the intended `INSTANCE`; omitting
it uses `NAME` from `aws/lab.env`. `aws-pull INSTANCE=replica-a` downloads to
`results/aws/replica-a/`, keeping logs separate from B. See the
[AWS setup details](../../README.md#0-one-time-setup-on-the-gpu-machine) for
state migration and per-replica tunnel ports.
The Week 5 benchmark itself still only connects to running model endpoints.
Disable or account for idle auto-shutdown during long tests, and stop **both**
GPU instances individually when finished:

```bash
make aws-stop INSTANCE=replica-a
make aws-stop INSTANCE=replica-b
```

## 2. Configure and check the endpoints

On a CPU host able to reach both GPU instances:

```bash
make setup
export REPLICA_A_URL=http://10.0.1.10:8000
export REPLICA_B_URL=http://10.0.1.11:8000

# Server ROOT URLs: do not append /v1. Each must reach a different vLLM process.
make sweep5 WEEK5_ARGS=--plan   # config validation, no network requests
make check5                   # both /health and served model ID must match
```

For laptop-only experimentation, two SSH tunnels can expose A and B on distinct
local ports, but WAN/tunnel bandwidth and latency can distort scaling results.
Prefer a same-region CPU host. The router and load generator share that host
and process in the automated sweep; monitor its CPU/network to rule out a
client/router bottleneck.

## 3. Run the experiment matrix

Start with the uniform closed-loop comparison:

```bash
make sweep5 WEEK5_ARGS='--workloads uniform --modes closed'
# Add mixed request lengths, keeping the same request sequence across policies:
make sweep5 WEEK5_ARGS='--workloads mixed --modes closed'
# Or run the full configured matrix, including fixed-rate arrivals:
make sweep5
```

To select experiments:

```bash
make sweep5 WEEK5_ARGS='--experiments single-a single-b two-rr two-least --workloads uniform --modes open'
```

Edit a copy of the config for a shorter smoke test and pass
`WEEK5_CONFIG=path/to/config.yaml`. The default matrix is deliberately substantial
and can take hours; do not start it as an unattended cost-free check.

### Workloads and load modes

- **Uniform:** target 1,024 input / 256 output tokens.
- **Mixed:** 75% short (256/128), 25% long (4,096/512), deterministically shuffled.
  The mix is exact for counts divisible by four; other counts use a truncated
  weighted cycle. Each policy gets the same shape order at a given level.
- **Closed loop:** total concurrency 1, 4, 16, 32, 64, 128, shared across the
  replicas, not per replica. Requests per level = `max(requests, 2 * concurrency)`.
- **Open loop:** constant inter-arrival times at the configured requests/s,
  independent of completions. Exactly `requests` arrivals are scheduled, followed
  by a drain. There is no semaphore/client queue; arrivals above `max_inflight`
  are explicitly counted as `client_dropped` rather than silently delayed.
- Token counts are calibrated through vLLM `/tokenize`; calibration errors fail
  the experiment rather than falling back to words. Measured prompt lengths
  include nonce/template overhead. Output lengths are forced and verified from
  streamed usage; missing usage or truncated streams count as failed requests.
- Each host gets a small direct warmup, excluded from routing counts. Levels
  are fixed-count, not steady-state tests; increase requests for stable tails
  and repeat runs with different experiment orders to check noise.

Choose rates around the observed single-replica capacity. For uniform output
length, a rough starting point is peak output tokens/s divided by output tokens
per request. Do not compare raw tokens/s across different workload mixes.

## 4. Results and routing evidence

Every invocation creates a separate `results/week5/<UTC-time>-<id>/` directory:

| File | Contents |
|---|---|
| `config.yaml` | Input configuration (endpoint environment variable names, no API key) |
| `<workload>-closed.csv`, `<workload>-open.csv` | Throughput, TTFT/ITL, successes/failures/drops and load |
| `routing.jsonl` | Before/after per-replica counters and request errors for each level |
| `metrics/*.prom` | Raw per-replica Prometheus snapshots before and after each level |
| `metrics/*.error` | Explicit diagnostic if a metrics endpoint could not be collected |
| `summary.csv` | Peak **measured**, error-free closed-loop throughput per tag/workload |

```bash
make plot5 RUN=results/week5/<run-id>
```

This plots tokens/s, TTFT p99, chunk-gap p99, and successful request percentage
against total concurrency or offered request rate. Plotting does not mix
different workloads or treat rate as concurrency.

`agg_tok_s` includes **only successfully completed requests' output tokens**,
divided by the whole level's wall time including drain; `req_s` likewise counts
successful requests. Failed/dropped requests are separate, never success-shaped
samples. Latency percentiles describe successful requests only; inspect the
success percentage alongside them. Open-loop TTFT includes scheduler lateness
from intended arrival; `schedule_lag_p99_ms` exposes client overload.
SDK retries are disabled. The sweep exits nonzero after saving its results if
any measured requests failed or were dropped.

For each `routing.jsonl` entry, subtract `before` counters from `after` counters
to obtain that level's per-replica distribution. `peak_inflight` is a cumulative
high-water mark for that experiment, not a per-level delta. `completed` means a
successful HTTP transfer, whereas CSV success additionally verifies token usage.
The router does not inspect generated content or token usage.

Questions to answer:

1. Are A and B independently comparable before combining them?
2. How much does measured aggregate throughput increase with two replicas?
   `vs_best_single` uses the best clean single-host point **in this run**, even if
   its concurrency differs. It is not proof of saturation or a 2x guarantee.
3. What throughput still meets your TTFT/ITL SLO? Peak throughput alone is not
   usable capacity. An illustrative SLO is TTFT p99 <= 2 s and ITL p99 <= 100 ms.
4. Does round-robin split request counts evenly but leave one host busier on the
   mixed workload? Does least-in-flight improve tail latency?
5. Are `failed`, `client_dropped`, or scheduling lag hiding overload?

Prometheus snapshots retain counters and endpoint gauges; they do **not** capture
transient GPU utilization or continuous queue history. During a representative
loaded run, also collect `nvidia-smi dmon -s mu -d 1` on each GPU host and scrape
vLLM metrics periodically if you need that evidence.

## 5. Inspect routing and test failures manually

### Supported policies

Only these two values are accepted by `--policy` (or `POLICY` in `make router5`):

| Value | Selection rule | Example with two healthy replicas |
|---|---|---|
| `round-robin` (default) | Advance through healthy backends, regardless of current load | A, B, A, B |
| `least-inflight` | Choose the backend with the fewest unfinished inference requests; rotate ties | A has 5 active requests, B has 2: choose B |

Both policies skip unhealthy replicas and return 503 if none are healthy.
In-flight counts include requests waiting at vLLM and requests still streaming
back through this router. A long request counts as one, just like a short request;
least-inflight does not estimate token cost. Round-robin balances request counts,
not execution time. With uneven request lengths, least-inflight can favor the
replica that finishes requests faster.

Random, weighted round-robin, GPU/KV-aware, and prefix-cache-aware routing are
**not implemented**; passing an unsupported policy is rejected.
Automated sweeps use each experiment's `policy` in [config.yaml](config.yaml),
not the Makefile's standalone-router `POLICY` variable.

For interactive testing, run a standalone router (foreground; Ctrl-C to stop):

```bash
make router5 POLICY=least-inflight   # or round-robin; listens on 127.0.0.1:8000
# In a second terminal:
curl --fail localhost:8000/health
curl --fail localhost:8000/router/stats
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Explain replica scaling"}],"max_tokens":128,"stream":true}'
```

`X-Lab-Replica` identifies the selected backend. Least-in-flight holds a
reservation until the **entire streamed response** completes or is disconnected,
not merely until response headers arrive. It only sees requests sent through
this router: it is not token-cost-aware, KV-aware, or aware of other clients.

To test availability deliberately:

1. On host B, run `make down` while the standalone router is receiving traffic.
2. Watch `/router/stats`: periodic health checks remove B from new routing.
   Checks run approximately every two seconds plus probe time; failures before
   detection remain possible. A connection error also marks a replica unhealthy.
3. Existing failed requests surface as errors/disconnected streams; they are
   **not replayed**, since replaying a partially streamed generation is unsafe.
4. New requests go to A. If both hosts are unhealthy, the router returns 503.
5. Restart B with the same environment using `make up`; after readiness and a
   successful probe it rejoins routing.

For a reproducible slow-replica case, the automated mock tests hold A's stream
open and verify that least-in-flight continues choosing B. The mixed GPU workload
tests uneven request duration without injecting artificial production delays.

## Validation and findings

```bash
make test5
```

Tests use local fake vLLM servers, not GPUs: streaming, reservation lifetime,
tie-breaking, health/failure/recovery, cancellation, no retries, mixed/open-loop
loads, strict calibration, CSV/metrics/routing output, and plotting.
They establish harness behavior, not real GPU throughput.

After running on the two GPU hosts, write the measured setup, results, scaling
ratio, SLO capacity, and routing trade-offs using the
[findings template](../../docs/findings/_template.md). Do not record projected
numbers as observed results.
