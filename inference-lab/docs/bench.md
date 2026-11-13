# bench.py: how the load test works

Source: [scripts/bench.py](../scripts/bench.py). Make target: `make bench TAG=<tag>`.

Talks to any OpenAI-compatible server already running (default `http://localhost:8000/v1`). It does **not** start or configure the server.

## Load model: closed loop, fixed request count

```
warmup (1 tiny request)
for c in --concurrency:                  # e.g. 1 2 4 8 16 32
    n_req = max(--requests, 2 * c)       # default --requests 32
    send n_req requests, at most c in flight at once
    wait for all to finish -> compute metrics -> one CSV row
append all rows to --out
```

- **Closed loop:** a new request starts only when an in-flight one finishes (asyncio semaphore of size `c`). The server never sees more than `c` concurrent requests from this client.
- **Not time-based:** each level runs until its `n_req` requests complete. Duration depends on speed (c=1 with 32 × ~9 s requests ≈ 5 min; c=8 ≈ 1 min).
- Levels run back-to-back on the same server; nothing is reset between them.
- For time-based / fixed arrival-rate (open loop) load, use `vllm bench serve --request-rate` or GuideLLM instead.
- [Week 5](../weeks/week5/README.md) adds a separate multi-replica harness with
  fixed-rate arrivals and mixed lengths, reusing this client's streaming request
  and prompt helpers. The existing benchmark CLI and CSV format are unchanged.

## Prompts

Built by `make_prompt(n_words, seed, nonce)`:

```python
words = " ".join(f"w{(seed * 7 + i) % 997}" for i in range(n_words))
prompt = f"{nonce}Summarize the following text.\n{words}"
```

| request | prompt |
|---|---|
| 0 | `[a1b2c3 c4 r0] Summarize the following text.\nw0 w1 w2 ...` |
| 1 | `[a1b2c3 c4 r1] Summarize the following text.\nw7 w8 w9 ...` |
| 2 | `[a1b2c3 c4 r2] Summarize the following text.\nw14 w15 ...` |

- **Length is in real tokens.** At startup bench.py asks the server's `/tokenize` how many tokens the filler words produce (~3.7 per `wNNN` for Qwen) and picks the word count that hits `--input-tokens`. It prints e.g. `--input-tokens 1024 -> 270 filler words`. If `/tokenize` isn't available (non-vLLM servers) it falls back to one word per token and warns.
- **No accidental cache hits.** Each prompt starts with a nonce `[run level request]`, so no two prompts share a prefix within a level, across levels, or across runs.
- `--shared-prefix`: every request uses the same prompt (no nonce), to test prefix caching.
- Output is fixed: `max_tokens = min_tokens = --output-tokens`, `temperature=0`. Every request generates exactly that many tokens.
- The CSV `input_tokens` column is the **measured** mean prompt length from the server's `usage.prompt_tokens` (includes the chat template, so it's a little above the target).

### History
Before 2026-11-04, `--input-tokens` counted words (real prompts were ~3.7× longer) and the same prompts were reused across levels. Rows from `w1-vllm-baseline` and `week3-sweep-1` have that problem; see [findings](findings/).

## Metrics (one CSV row per level)

| Column | Meaning |
|---|---|
| `ttft_p50_ms`, `ttft_p99_ms` | Time to first token, measured from when the request **leaves the client semaphore**. Client-side waiting is excluded, so a high TTFT means waiting **on the server** (queue + prefill). |
| `itl_p50_ms`, `itl_p99_ms` | Gap between consecutive streamed chunks, pooled across all requests. p99 spikes = decode stalls (prefill interference or preemption). |
| `per_req_tok_s` | Mean over requests of `output_tokens / request_wall_time`. What one user experiences. |
| `agg_tok_s` | Total output tokens / level wall time. Server throughput. |
| `req_s` | Requests completed / level wall time. |

Percentiles are nearest-rank on the sorted list. With 32 requests, TTFT p99 is effectively the max.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--model` | required | Must match the served model name |
| `--base-url` | `http://localhost:8000/v1` | |
| `--tag` | `run` | Label for the CSV rows |
| `--concurrency` | `1 4 16` | Space-separated levels |
| `--requests` | `32` | Minimum per level (actual = `max(requests, 2c)`) |
| `--input-tokens` | `512` | Words in the filler text (see caveat 2) |
| `--output-tokens` | `256` | Forced exact output length |
| `--shared-prefix` | off | Identical prompt for every request |
| `--out` | `results/bench.csv` | Appended to; header written only if new |

## Reading the results
- **Knee:** the concurrency where `agg_tok_s` stops growing. Past it, throughput **plateaus** (it doesn't drop) and the extra load turns into TTFT (queueing).
- **Usable capacity:** the highest concurrency that still meets a latency target (e.g. TTFT p99 < 2 s, ITL p99 < 100 ms).
- **Diagnose the plateau** while a level runs (`make logs` / `make metrics`): `Running` vs `Waiting` request counts, KV cache usage %, preemption count.
