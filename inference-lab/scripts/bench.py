"""Async load generator for any OpenAI-compatible server.

Measures TTFT, inter-token latency (ITL), per-request and aggregate tokens/s
across a concurrency sweep. Appends rows to a CSV tagged with --tag.

  python scripts/bench.py --model Qwen/Qwen3-8B --tag baseline \
      --concurrency 1 2 4 8 16 32 --input-tokens 512 --output-tokens 256
"""
import argparse, asyncio, csv, os, statistics, time, uuid
import httpx
from openai import AsyncOpenAI


def make_prompt(n_words: int, seed: int, nonce: str = "") -> str:
    words = " ".join(f"w{(seed * 7 + i) % 997}" for i in range(n_words))
    return f"{nonce}Summarize the following text.\n{words}"


def words_for_tokens(base_url: str, model: str, target: int, strict: bool = False) -> int:
    """Word count whose prompt is ~target tokens, measured with the server's /tokenize."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
    def count(n):
        response = httpx.post(
            url, json={"model": model, "prompt": make_prompt(n, 0)},
            headers={"Authorization": f"Bearer {os.getenv('API_KEY', 'x')}"}, timeout=30)
        response.raise_for_status()
        return response.json()["count"]

    try:
        hdr, full = count(0), count(target)
        if full <= hdr:
            raise ValueError("Token count did not increase with prompt length")
    except (httpx.HTTPError, KeyError, ValueError):
        if strict:
            raise
        print("warning: /tokenize unavailable; --input-tokens is used as a word count")
        return target
    return max(1, round((target - hdr) * target / (full - hdr)))


async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    first = None
    last = t0
    gaps = []
    n = p = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"min_tokens": max_tokens},
    )
    async for chunk in stream:
        if chunk.usage:
            n, p = chunk.usage.completion_tokens, chunk.usage.prompt_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            if first is None:
                first = now - t0
            else:
                gaps.append(now - last)
            last = now
    total = time.perf_counter() - t0
    return {"ttft": first or total, "itl": gaps, "tokens": n, "prompt_tokens": p, "total": total}


async def run_level(client, args, conc):
    sem = asyncio.Semaphore(conc)
    n_req = max(args.requests, conc * 2)
    shared = make_prompt(args.n_words, 0)

    async def task(i):
        async with sem:
            # unique leading nonce per run/level/request so earlier levels can't warm the prefix cache
            p = shared if args.shared_prefix else make_prompt(
                args.n_words, i, f"[{args.run_id} c{conc} r{i}] ")
            return await one_request(client, args.model, p, args.output_tokens)

    t0 = time.perf_counter()
    res = await asyncio.gather(*(task(i) for i in range(n_req)))
    wall = time.perf_counter() - t0
    ttfts = sorted(r["ttft"] for r in res)
    itls = sorted(g for r in res for g in r["itl"])
    pct = lambda a, p: a[min(len(a) - 1, int(len(a) * p))] if a else 0
    toks = sum(r["tokens"] for r in res)
    return {
        "tag": args.tag, "concurrency": conc, "requests": n_req,
        # measured from usage, includes chat template
        "input_tokens": round(statistics.mean(r["prompt_tokens"] for r in res)),
        "output_tokens": args.output_tokens,
        "ttft_p50_ms": 1000 * pct(ttfts, .5), "ttft_p99_ms": 1000 * pct(ttfts, .99),
        "itl_p50_ms": 1000 * pct(itls, .5), "itl_p99_ms": 1000 * pct(itls, .99),
        "per_req_tok_s": statistics.mean(r["tokens"] / r["total"] for r in res),
        "agg_tok_s": toks / wall, "req_s": n_req / wall,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--input-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--shared-prefix", action="store_true")
    ap.add_argument("--out", default="results/bench.csv")
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=os.getenv("API_KEY", "x"), timeout=600)
    await one_request(client, args.model, "warmup", 16)  # warmup
    args.run_id = uuid.uuid4().hex[:6]
    args.n_words = words_for_tokens(args.base_url, args.model, args.input_tokens)
    print(f"--input-tokens {args.input_tokens} -> {args.n_words} filler words per prompt")
    rows = []
    for c in args.concurrency:
        row = await run_level(client, args, c)
        rows.append(row)
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()})

    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    asyncio.run(main())
