"""Week 5: benchmark existing remote replicas through a temporary local router."""
import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import random
import uuid

from aiohttp import web
import httpx
from openai import AsyncOpenAI, OpenAIError
import yaml

from bench import make_prompt, one_request, words_for_tokens
from router import POLICIES, Replica, Router


LOG = logging.getLogger(__name__)


def positive(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def load_config(path):
    with open(path) as source:
        cfg = yaml.safe_load(source)
    positive(cfg["requests"], "requests", integer=True)
    positive(cfg["max_inflight"], "max_inflight", integer=True)
    for level in cfg["concurrency"]:
        positive(level, "concurrency", integer=True)
    for rate in cfg["request_rates"]:
        positive(rate, "request_rates")
    if not cfg["concurrency"] or not cfg["request_rates"]:
        raise ValueError("Provide concurrency and request_rates levels")
    replicas = [Replica(r["name"], os.environ[r["url_env"]]) for r in cfg["replicas"]]
    Router(replicas)  # Validate names and distinct root URLs before any network calls.
    if len(replicas) != 2:
        raise ValueError("Week 5 requires exactly two independent replicas")
    names = {r.name for r in replicas}
    for section in ("workloads", "experiments"):
        key = "name" if section == "workloads" else "tag"
        identifiers = [item[key] for item in cfg[section]]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{section} must have unique identifiers")
        if any(not identifier or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                                     for c in identifier) for identifier in identifiers):
            raise ValueError(f"{section} identifiers must use lowercase letters, digits, - or _")
    for exp in cfg["experiments"]:
        if (not exp["replicas"] or len(set(exp["replicas"])) != len(exp["replicas"])
                or not set(exp["replicas"]) <= names or exp["policy"] not in POLICIES):
            raise ValueError(f"Invalid experiment: {exp['tag']}")
    for workload in cfg["workloads"]:
        if not workload["shapes"]:
            raise ValueError("Workloads need at least one request shape")
        for shape in workload["shapes"]:
            for key in ("input_tokens", "output_tokens", "weight"):
                positive(shape[key], key, integer=True)
    return cfg, replicas


def request_shapes(workload, count, seed):
    cycle = [shape for shape in workload["shapes"] for _ in range(shape["weight"])]
    shapes = (cycle * math.ceil(count / len(cycle)))[:count]
    random.Random(seed).shuffle(shapes)
    return shapes


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))] if ordered else None


async def run_level(client, cfg, workload, words, tag, mode, load, run_id):
    """Fixed-rate mode has no semaphore; overflow is counted, never queued in the client."""
    count = max(cfg["requests"], 2 * load) if mode == "closed" else cfg["requests"]
    shapes = request_shapes(workload, count, cfg.get("seed", 17))
    semaphore = asyncio.Semaphore(load) if mode == "closed" else None
    loop = asyncio.get_running_loop()
    start = loop.time()
    results, errors, lags = [], [], []
    pending = set()
    tasks = []
    dropped = 0
    peak = 0
    active = 0

    async def send(index, shape, due):
        nonlocal active, peak
        prompt = make_prompt(words[shape["input_tokens"]], index,
                             f"[{run_id} {tag} {mode}{load} r{index}] ")
        lag = max(0, loop.time() - due) if due is not None else 0
        lags.append(lag)
        active += 1
        peak = max(peak, active)
        try:
            result = await one_request(client, cfg["model"], prompt, shape["output_tokens"])
            if result["tokens"] != shape["output_tokens"] or result["prompt_tokens"] <= 0:
                raise ValueError("Missing usage, truncated stream, or unexpected output token count")
            result["ttft"] += lag
            result["total"] += lag
            results.append(result)
        except (OpenAIError, httpx.HTTPError, ValueError) as error:
            message = str(error) or type(error).__name__
            LOG.error("%s %s=%s request %s failed: %s", tag, mode, load, index, message)
            errors.append({"request": index, "error": message})
        finally:
            active -= 1

    async def closed_send(index, shape):
        async with semaphore:
            await send(index, shape, None)

    if mode == "closed":
        await asyncio.gather(*(closed_send(i, shape) for i, shape in enumerate(shapes)))
    else:
        for i, shape in enumerate(shapes):
            due = start + i / load
            await asyncio.sleep(max(0, due - loop.time()))
            if len(pending) >= cfg["max_inflight"]:
                dropped += 1
                continue
            task = asyncio.create_task(send(i, shape, due))
            tasks.append(task)
            pending.add(task)
            task.add_done_callback(pending.discard)
        await asyncio.gather(*tasks)
    wall = loop.time() - start
    ttfts = [r["ttft"] * 1000 for r in results]
    itls = [gap * 1000 for r in results for gap in r["itl"]]
    row = {
        "tag": tag, "workload": workload["name"], "mode": mode,
        "concurrency": load if mode == "closed" else 0,
        "request_rate": load if mode == "open" else 0,
        "requests": count, "succeeded": len(results), "failed": len(errors),
        "client_dropped": dropped, "wall_s": wall,
        "input_tokens": sum(r["prompt_tokens"] for r in results) / len(results) if results else None,
        "output_tokens": sum(r["tokens"] for r in results) / len(results) if results else None,
        "ttft_p50_ms": percentile(ttfts, .5), "ttft_p99_ms": percentile(ttfts, .99),
        "itl_p50_ms": percentile(itls, .5), "itl_p99_ms": percentile(itls, .99),
        "agg_tok_s": sum(r["tokens"] for r in results) / wall,
        "req_s": len(results) / wall,
        "schedule_lag_p99_ms": percentile([lag * 1000 for lag in lags], .99),
        "peak_client_inflight": peak,
    }
    if dropped:
        LOG.error("%s dropped %s scheduled arrivals at max_inflight=%s", tag, dropped,
                  cfg["max_inflight"])
    return row, errors


async def preflight(replicas, model):
    headers = {"Authorization": f"Bearer {os.getenv('API_KEY', 'x')}"}
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        for replica in replicas:
            response = await client.get(replica.url + "/health")
            response.raise_for_status()
            response = await client.get(replica.url + "/v1/models")
            response.raise_for_status()
            models = [entry["id"] for entry in response.json()["data"]]
            if model not in models:
                raise ValueError(f"{replica.name} does not serve {model}; available: {models}")
            print(f"Ready: {replica.name} serves {model}", flush=True)


async def capture_metrics(replicas, directory, label):
    async with httpx.AsyncClient(timeout=10) as client:
        for replica in replicas:
            try:
                response = await client.get(
                    replica.url + "/metrics",
                    headers={"Authorization": f"Bearer {os.getenv('API_KEY', 'x')}"},
                )
                response.raise_for_status()
                (directory / f"{label}-{replica.name}.prom").write_text(response.text)
            except httpx.HTTPError as error:
                LOG.warning("%s /metrics unavailable: %s", replica.name, error)
                (directory / f"{label}-{replica.name}.error").write_text(str(error))


def append_csv(path, row):
    new = not path.exists()
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=row.keys())
        if new:
            writer.writeheader()
        writer.writerow(row)


async def experiment(cfg, all_replicas, exp, workloads, modes, output):
    replicas = [Replica(r.name, r.url) for r in all_replicas if r.name in exp["replicas"]]
    router = Router(replicas, exp["policy"])
    runner = web.AppRunner(router.application(), access_log=None)
    await runner.setup()
    try:
        # A temporary loopback port avoids colliding with an existing local vLLM server.
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        if not all(r.healthy for r in replicas):
            raise RuntimeError(f"{exp['tag']}: a replica failed its initial health check")
        async with AsyncOpenAI(
            base_url=f"http://127.0.0.1:{port}/v1", api_key=os.getenv("API_KEY", "x"),
            timeout=600, max_retries=0,
            http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=None,
                                                            max_keepalive_connections=256)),
        ) as client:
            for workload in workloads:
                # Calibrate once per prompt length; /tokenize is proxied, not counted as inference.
                words = {}
                for shape in workload["shapes"]:
                    length = shape["input_tokens"]
                    words[length] = await asyncio.to_thread(
                        words_for_tokens, f"http://127.0.0.1:{port}/v1", cfg["model"], length,
                        strict=True)
                for mode in modes:
                    levels = cfg["concurrency"] if mode == "closed" else cfg["request_rates"]
                    for load in levels:
                        label = f"{exp['tag']}-{workload['name']}-{mode}-{load}"
                        print(f"=== {label} ===", flush=True)
                        await capture_metrics(replicas, output / "metrics", label + "-before")
                        before = router.snapshot()
                        row, errors = await run_level(client, cfg, workload, words, exp["tag"],
                                                      mode, load, output.name)
                        # Allow stream handlers to finish their accounting before taking a snapshot.
                        for _ in range(100):
                            if not any(r.inflight for r in replicas):
                                break
                            await asyncio.sleep(.01)
                        after = router.snapshot()
                        append_csv(output / f"{workload['name']}-{mode}.csv", row)
                        with (output / "routing.jsonl").open("a") as log:
                            log.write(json.dumps({"label": label, "before": before, "after": after,
                                                  "errors": errors}) + "\n")
                        await capture_metrics(replicas, output / "metrics", label + "-after")
                        print(json.dumps(row), flush=True)
    finally:
        await runner.cleanup()


def summarize(output):
    summary = []
    for path in sorted(output.glob("*-closed.csv")):
        with path.open() as source:
            rows = list(csv.DictReader(source))
        clean = [r for r in rows if int(r["failed"]) == 0 and int(r["client_dropped"]) == 0]
        baseline = max((float(r["agg_tok_s"]) for r in clean
                        if r["tag"] in ("single-a", "single-b")), default=None)
        for tag in dict.fromkeys(r["tag"] for r in rows):
            candidates = [r for r in clean if r["tag"] == tag]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: float(r["agg_tok_s"]))
            value = float(best["agg_tok_s"])
            item = {"workload": best["workload"], "tag": tag,
                    "peak_measured_tok_s": value, "concurrency": int(best["concurrency"]),
                    "vs_best_single": value / baseline if baseline else None}
            append_csv(output / "summary.csv", item)
            summary.append(item)
    return summary


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="weeks/week5/config.yaml")
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("closed", "open"), default=["closed", "open"])
    parser.add_argument("--plan", action="store_true", help="Validate config without network calls")
    parser.add_argument("--check", action="store_true", help="Check both endpoints, then exit")
    args = parser.parse_args()
    cfg, replicas = load_config(args.config)
    experiments = cfg["experiments"]
    workloads = cfg["workloads"]
    for selected, items, key in ((args.experiments, experiments, "tag"),
                                 (args.workloads, workloads, "name")):
        if selected and not set(selected) <= {item[key] for item in items}:
            parser.error(f"Unknown {key} selection: {selected}")
    experiments = [e for e in experiments if not args.experiments or e["tag"] in args.experiments]
    workloads = [w for w in workloads if not args.workloads or w["name"] in args.workloads]
    if args.plan:
        print(yaml.safe_dump({"model": cfg["model"], "replicas": [r.name for r in replicas],
                              "experiments": experiments, "workloads": workloads,
                              "modes": args.modes, "concurrency": cfg["concurrency"],
                              "request_rates": cfg["request_rates"], "requests": cfg["requests"]}))
        return
    await preflight(replicas, cfg["model"])
    if args.check:
        return
    output = Path(cfg["out"]) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                               + "-" + uuid.uuid4().hex[:6])
    (output / "metrics").mkdir(parents=True)
    with (output / "config.yaml").open("w") as file:
        yaml.safe_dump(cfg, file)
    print(f"Results: {output}", flush=True)
    # Warm each replica directly so both start loaded; disable SDK retries everywhere.
    for replica in replicas:
        async with AsyncOpenAI(base_url=replica.url + "/v1", api_key=os.getenv("API_KEY", "x"),
                               timeout=600, max_retries=0) as client:
            await one_request(client, cfg["model"], "warmup", 16)
    for exp in experiments:
        await experiment(cfg, replicas, exp, workloads, args.modes, output)
    print(json.dumps(summarize(output), indent=2))
    failed = False
    for path in output.glob("*.csv"):
        if path.name == "summary.csv":
            continue
        with path.open() as source:
            failed |= any(int(row["failed"]) or int(row["client_dropped"])
                          for row in csv.DictReader(source))
    if failed:
        raise RuntimeError(f"Some requests failed or were dropped; inspect {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
