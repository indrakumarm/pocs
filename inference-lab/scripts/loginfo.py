"""Pull the capacity numbers out of a vLLM server log.

  python scripts/loginfo.py results/logs/baseline.log
  docker compose logs vllm | python scripts/loginfo.py -
"""
import re, sys

STAT = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                  r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")


def grab(pattern, text):
    m = re.search(pattern, text)
    return m.groups() if m else None


def report(name, text):
    print(f"== {name}")
    if args := grab(r"non-default args: (\{.*\})", text):
        print(f"  server args          {args[0]}")
    if w := grab(r"Model loading took ([\d.]+) GiB", text):
        print(f"  1 weights            {w[0]} GiB")
    kv_gib, kv_tok = grab(r"Available KV cache memory: (-?[\d.]+) GiB", text), \
        grab(r"GPU KV cache size: ([\d,]+) tokens", text)
    if kv_gib:
        line = f"  3 KV budget          {kv_gib[0]} GiB"
        if kv_tok:
            n = int(kv_tok[0].replace(",", ""))
            line += f" = {n:,} tokens  ({float(kv_gib[0]) * 2**30 / n / 1024:.0f} KiB/token = step 2)"
        print(line)
    if mc := grab(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text):
        print(f"  4 vLLM max conc.     {mc[1]}x requests of {mc[0]} tokens "
              f"(divide KV tokens by YOUR input+output instead)")

    stats = [tuple(map(float, m)) for m in STAT.findall(text)]
    if stats:
        run, wait, kv = (max(s[k] for s in stats) for k in (2, 3, 4))
        print(f"  peak running         {run:.0f} reqs   <- measured knee")
        print(f"  peak waiting         {wait:.0f} reqs")
        print(f"  peak KV usage        {kv:.1f}%")
        print(f"  peak gen throughput  {max(s[1] for s in stats):.0f} tok/s (10 s average)")
        if wait and kv >= 80:
            print(f"  -> KV-bound: requests queue once ~{run:.0f} are running; more KV or shorter requests move the knee")
        elif wait:
            print("  -> queueing with KV to spare: limited by --max-num-seqs / --max-num-batched-tokens")
        else:
            print("  -> never saturated: push concurrency higher")
    print(f"  preemptions          {len(re.findall(r'preempt', text, re.I))}")
    if err := grab(r"((?:ValueError|RuntimeError|torch\.OutOfMemoryError|CUDA out of memory): [^\n]*)", text):
        print(f"  !! error             {err[0][:200]}")


if __name__ == "__main__":
    for path in sys.argv[1:] or ["-"]:
        report(path, sys.stdin.read() if path == "-" else open(path, errors="replace").read())
