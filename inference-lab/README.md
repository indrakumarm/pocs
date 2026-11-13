# Inference Engineering Lab: Weeks 1–3

How the benchmark tools work and what each run showed: [docs/](docs/README.md).

Run on a Linux box with an NVIDIA GPU (rented: RunPod/Lambda/Vast, 1× 4090/L4/A100). Use a CUDA + Docker template.
Automation runs from day 1: every manual step below has a scripted twin.

## 0. One-time setup (on the GPU machine)

**Option A: AWS (from your laptop).** Needs AWS CLI v2, an existing key pair, subnet, and a security group allowing SSH from your IP. Request the "Running On-Demand/Spot G and VT instances" vCPU quota first.
```bash
cp aws/lab.env.example aws/lab.env      # fill in profile, region, key, subnet, SG, instance type
export HF_TOKEN=hf_xxx                  # copied to the instance's HF cache on sync
make aws-up                             # Deep Learning Base GPU AMI, verifies GPU in Docker, rsyncs repo, make setup
make aws-ssh                            # then: make up / make sweep3
make aws-tunnel                         # (other terminal) localhost:8000 -> instance
make aws-sync                           # push local code changes;  make aws-pull  fetches results/
make aws-stop                           # keeps disk + model cache; make aws-start to resume
make aws-down                           # terminate + cancel spot request
```

To manage multiple GPU hosts, select a name on **every** command:

```bash
make aws-up INSTANCE=replica-a
make aws-up INSTANCE=replica-b
make aws-status INSTANCE=replica-a
make aws-ssh INSTANCE=replica-b           # then make up to start vLLM
make aws-stop INSTANCE=replica-a
make aws-start INSTANCE=replica-a
make aws-pull INSTANCE=replica-b          # results/aws/replica-b/
make aws-down INSTANCE=replica-b         # confirms termination of B only
```

Without `INSTANCE`, commands use `NAME` from `aws/lab.env`, preserving the default
single-host workflow. `aws-up` searches for an existing non-terminated lab
instance with that exact Name tag in the configured AWS profile/region:
running/pending instances resume setup; stopped/stopping instances restart;
otherwise it creates a new instance. Multiple matching instances are rejected.
All names share the same launch settings from `aws/lab.env`; a name is a selector,
not a separate configuration.

`aws/.state` remains a **single file**, now storing tab-separated
`profile / region / name / instance-ID` records. Existing single-ID state migrates
automatically using the instance's actual AWS Name tag in the configured
profile/region (use the original profile/region for the first migration).
Deleting one instance removes only its record. SSH known-hosts files and
downloaded results are isolated by name; `aws-pull` now writes to
`results/aws/<name>/` rather than merging into `results/`.
The old SSH known-hosts file is retained as `aws/.known_hosts.legacy`.

Lifecycle/state operations are serialized locally by `aws/.state.lock` and a
concurrent command reports an error rather than modifying the same file.
SSH/tunnels do not hold this lock while connected. If a process is forcibly
killed, remove a stale lock with `rmdir aws/.state.lock` **only after verifying no
AWS helper is still running**. This protects local commands, not separate laptops
launching the same name simultaneously.

For two simultaneous SSH tunnels, use different local ports in separate terminals:

```bash
make aws-tunnel INSTANCE=replica-a VLLM_PORT=8001 GRAFANA_PORT=3001 PROMETHEUS_PORT=9091
make aws-tunnel INSTANCE=replica-b VLLM_PORT=8002 GRAFANA_PORT=3002 PROMETHEUS_PORT=9092
```

`aws-up` prepares the host and `aws-start` resumes EC2; neither starts or verifies
the model server. Start vLLM with `make up` on each host separately. Public IPs
may change after restarting; use `aws-status` to refresh endpoint URLs.
Validate the lifecycle helper without creating AWS resources using `make test-aws`.

The instance stops itself after `IDLE_MINUTES` with idle GPU and no SSH sessions. Skip the manual steps below on AWS.

**Option B: any rented GPU box (manual).**
```bash
# verify GPU + Docker GPU passthrough
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# if docker/nvidia-container-toolkit are missing (Ubuntu)
curl -fsSL https://get.docker.com | sh
sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

export HF_TOKEN=hf_xxx                 # for gated models
make setup                             # venv + python deps
```

---

## Week 1: Fundamentals

**Concepts to read/note:** prefill (compute-bound) vs decode (memory-bandwidth-bound); KV cache size = 2 × layers × kv_heads × head_dim × bytes × tokens; TTFT, ITL, throughput.

- [ ] Compute KV cache per token for Qwen3-8B by hand, then verify against vLLM startup log ("GPU KV cache size: N tokens")
- [ ] Run via HF Transformers (baseline, no batching)
- [ ] Run via llama.cpp (GGUF Q4/Q8)
- [ ] Run via vLLM (Docker) and compare tokens/s
- [ ] Roofline estimate: decode tok/s ≈ memory_bandwidth / model_bytes. Compare with measured.

```bash
# --- vLLM (automated path) ---
make up MODEL=Qwen/Qwen3-8B
make logs                              # wait for "Application startup complete"

# --- smoke test ---
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

# --- llama.cpp (manual comparison) ---
docker run --rm --gpus all -p 8081:8080 -v ~/models:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 8192 --host 0.0.0.0
# download GGUF:  pip install -U huggingface_hub && hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models

# --- first benchmark ---
make bench TAG=w1-vllm-baseline
```

**Manual debugging toolkit**
```bash
nvidia-smi dmon -s mu -d 1             # live util/memory/bandwidth
watch -n1 nvidia-smi                   # VRAM usage
docker logs --tail 100 <container>
docker exec -it <container> bash
make metrics                           # Prometheus counters: running/waiting reqs, KV usage
python -c "import torch;print(torch.cuda.mem_get_info())"
```

---

## Week 2: Benchmarking

- [ ] Run concurrency sweep (1→64), plot throughput vs latency (`make plot`)
- [ ] Find the "knee" where ITL starts to climb: that's saturation
- [ ] Vary input length (128 / 1024 / 4096) with output fixed; then vice versa. Note prefill-heavy vs decode-heavy behaviour
- [ ] Compare against official tool: `vllm bench serve` and GuideLLM

```bash
# our client
.venv/bin/python scripts/bench.py --model Qwen/Qwen3-8B --tag in128  --input-tokens 128  --output-tokens 256 --concurrency 1 4 16 64 --out results/w2.csv
.venv/bin/python scripts/bench.py --model Qwen/Qwen3-8B --tag in4096 --input-tokens 4096 --output-tokens 256 --concurrency 1 4 16 64 --out results/w2.csv
.venv/bin/python scripts/plot.py results/w2.csv

# official vLLM benchmark (inside container)
docker exec -it $(docker ps -qf ancestor=vllm/vllm-openai:latest) \
  vllm bench serve --model Qwen/Qwen3-8B --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --num-prompts 200 --max-concurrency 16

# GuideLLM
pip install guidellm
guidellm benchmark --target http://localhost:8000 --rate-type sweep --max-seconds 30 \
  --data "prompt_tokens=512,output_tokens=256"
```

**Debugging:** if throughput plateaus early, check `make metrics` for `num_requests_waiting > 0` (queueing) and KV cache usage near 100% (preemption). If client CPU hits 100%, the bottleneck is the load generator.

---

## Week 3: Tuning knobs (fully automated)

One knob per experiment; results append to `results/week3.csv`.

```bash
make sweep3        # launches a container per experiment, benchmarks, saves logs in results/logs/, tears down
make plot
```

| Knob | Hypothesis to test | Look at |
|---|---|---|
| `--max-num-seqs` 32 vs 256 | Higher raises throughput but ITL grows | agg tok/s, ITL p99 |
| `--max-num-batched-tokens` 2048 vs 16384 | Larger helps TTFT/throughput; hurts ITL for decodes | TTFT, ITL |
| chunked prefill on/off | Off causes decode stalls when long prompts arrive | ITL p99 |
| prefix caching + shared prefix | Big TTFT drop when prefix is shared | TTFT p50 |
| `--gpu-memory-utilization` 0.6 vs 0.9 | Less KV cache, earlier preemption at high concurrency | log KV tokens, throughput |
| `--kv-cache-dtype fp8` | ~2× KV capacity, slight quality risk | KV tokens, throughput |
| `--max-model-len` | Smaller frees memory for batching | log KV tokens |

**Lab notebook template** (copy per experiment into `notes.md`):
```
Config: <flags>   Hypothesis: <..>   Result: <numbers>   Conclusion: <..>   Surprises: <..>
```

**Debugging**
```bash
grep -E "KV cache|Maximum concurrency|preempt" results/logs/<tag>.log
docker ps -a | grep lab-                         # leftover containers
docker rm -f $(docker ps -aq --filter name=lab-) # cleanup
curl -s localhost:8000/metrics | grep -E "preemptions|prefix_cache"
```
Common failures: OOM at startup → lower `--gpu-memory-utilization` / `--max-model-len`; hang at startup → check `HF_TOKEN` and the model download in logs; `NCCL`/shm errors → keep `ipc_mode=host`.

---

## Next (week 4+)
Each later week lives in `weeks/weekN/` (`README.md` + `config.yaml`). Week 4 (quantization, speculative decoding, SGLang) is in `weeks/week4/`; run it with `make sweep4`. The week 3 config is `weeks/week3/config.yaml`. To split a single model across GPUs (tensor parallelism), add `--tensor-parallel-size N` and `gpus: [0,1,...]` in a sweep config; this is different from independent replicas.

[Week 5](weeks/week5/README.md) runs the **same complete model on two separate
single-GPU hosts**, comparing single-replica baselines with round-robin and
least-in-flight routing. This is replica scaling, **not** tensor parallelism.
Set `REPLICA_A_URL` and `REPLICA_B_URL`, then run `make check5` and `make sweep5`.
The runner connects to existing servers; it does not provision GPU instances.

## Cost hygiene
Stop instances when idle. Cache models on a persistent volume. Script everything so each rental starts with `git clone && make setup && make sweep3` (on AWS: `make aws-up`).
