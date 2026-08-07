#!/usr/bin/env python3
"""
Analyze guidellm benchmarks + nvidia-smi logs produced by run_benchmarks.sh.

Curated (minimal) metric set — no clutter:

  Client (guidellm):
    req_per_sec, output_tok_per_sec, ttft_p95_ms, itl_p95_ms, e2e_p95_s

  GPU (nvidia-smi dmon + query-gpu):
    sm_util_p95, mem_bw_util_p95, vram_peak_gb, avg_power_w

  Derived:
    tokens_per_joule    = output_tok_per_sec / avg_power_w
    bottleneck          = compute | memory-bw | external | balanced

Outputs (into <results_dir>):
    summary.csv
    summary.md
    curve_<model>_<workload>.png   (if matplotlib installed)

Usage:
    python3 analyze_benchmarks.py ~/vllm_benchmarks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas tabulate", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------- helpers ----------

def _get(d: dict, path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _first(d: dict, *paths, default=None):
    for p in paths:
        v = _get(d, p)
        if v is not None:
            return v
    return default


def _p95(series) -> float | None:
    s = series.dropna()
    if len(s) == 0:
        return None
    return float(s.quantile(0.95))


# ---------- guidellm extraction ----------

def extract_client(bench: dict, meta: dict) -> dict:
    """Only the 5 client-side metrics that matter."""
    strategy = _first(bench, "args.strategy.kind", "strategy.kind", default="?")
    rate = _first(bench, "args.strategy.rate", "strategy.rate")
    if isinstance(rate, list):
        rate = rate[0] if rate else None

    return {
        "run_id":         meta.get("run_id"),
        "test_id":        meta.get("test_id"),
        "model":          meta.get("model"),
        "workload":       meta.get("workload"),
        "prompt_tokens":  meta.get("prompt_tokens"),
        "output_tokens":  meta.get("output_tokens"),
        "strategy":       strategy,
        "target_rate":    rate,

        "req_per_sec":       _first(bench,
            "metrics.requests_per_second.successful.mean",
            "metrics.requests_per_second.mean"),
        "output_tok_per_sec": _first(bench,
            "metrics.output_tokens_per_second.successful.mean",
            "metrics.output_tokens_per_second.mean"),
        "ttft_p95_ms":       _first(bench,
            "metrics.time_to_first_token_ms.successful.percentiles.p95",
            "metrics.time_to_first_token_ms.percentiles.p95"),
        "itl_p95_ms":        _first(bench,
            "metrics.inter_token_latency_ms.successful.percentiles.p95",
            "metrics.inter_token_latency_ms.percentiles.p95"),
        "e2e_p95_s":         _first(bench,
            "metrics.request_latency.successful.percentiles.p95",
            "metrics.request_latency.percentiles.p95"),
    }


# ---------- GPU log parsing ----------

def parse_dmon(path: Path) -> dict:
    """Parse `nvidia-smi dmon -s pucmt -o DT` output.

    Columns after `-o DT` prefix: date, time, gpu, pwr, gtemp, mtemp, sm, mem, ...
    Returns p95 of sm%, mem-bw%, and mean power. All None if file missing/empty.
    """
    result = {"sm_util_p95": None, "mem_bw_util_p95": None, "avg_power_w": None}
    if not path.exists():
        return result

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # With -o DT: date time gpu pwr gtemp mtemp sm mem enc dec ...
            # Some driver versions omit certain columns; be defensive.
            if len(parts) < 8:
                continue
            try:
                # parts[2]=gpu_idx, parts[3]=pwr, parts[6]=sm, parts[7]=mem
                pwr = float(parts[3])
                sm  = float(parts[6])
                mbw = float(parts[7])
                rows.append((pwr, sm, mbw))
            except ValueError:
                continue

    if not rows:
        return result
    df = pd.DataFrame(rows, columns=["pwr", "sm", "mem_bw"])
    # Only consider active samples (sm>0) so idle padding doesn't skew averages
    active = df[df["sm"] > 0]
    if active.empty:
        active = df
    result["sm_util_p95"]     = _p95(active["sm"])
    result["mem_bw_util_p95"] = _p95(active["mem_bw"])
    result["avg_power_w"]     = float(active["pwr"].mean())
    return result


def parse_mem(path: Path) -> dict:
    """Parse `nvidia-smi --query-gpu=... --format=csv,nounits -l 1` output.

    Header: timestamp, index, memory.used, memory.total, utilization.gpu
    Returns peak VRAM used across the run.
    """
    result = {"vram_peak_gb": None, "vram_total_gb": None}
    if not path.exists():
        return result
    try:
        df = pd.read_csv(path, skipinitialspace=True)
    except Exception:
        return result
    used_col = next((c for c in df.columns if "memory.used" in c), None)
    tot_col  = next((c for c in df.columns if "memory.total" in c), None)
    if used_col is None:
        return result
    # values are MiB when nounits
    result["vram_peak_gb"]  = round(float(df[used_col].max()) / 1024.0, 2)
    if tot_col is not None:
        result["vram_total_gb"] = round(float(df[tot_col].max()) / 1024.0, 2)
    return result


# ---------- derived metrics ----------

def classify_bottleneck(sm: float | None, mem_bw: float | None,
                        req_per_sec: float | None) -> str | None:
    """Simple, opinionated classifier — 4 buckets only."""
    if sm is None or mem_bw is None:
        return None
    if sm >= 90 and mem_bw >= 90:
        return "balanced"          # both maxed — server is fully utilized
    if sm >= 90 and mem_bw < 70:
        return "compute"           # compute-bound (typical prefill)
    if mem_bw >= 90 and sm < 80:
        return "memory-bw"         # HBM-bound (typical large-batch decode)
    if sm < 70 and mem_bw < 70 and (req_per_sec or 0) > 0:
        return "external"          # GPU idle-ish but requests flowing → scheduler/CPU/net
    return "balanced"


def enrich(row: dict) -> dict:
    out = row.get("output_tok_per_sec")
    pwr = row.get("avg_power_w")
    row["tokens_per_joule"] = (
        round(out / pwr, 2) if (out and pwr and pwr > 0) else None
    )
    row["bottleneck"] = classify_bottleneck(
        row.get("sm_util_p95"), row.get("mem_bw_util_p95"), row.get("req_per_sec")
    )
    return row


# ---------- walker ----------

def walk_results(root: Path) -> pd.DataFrame:
    rows = []
    for meta_file in root.rglob("meta.json"):
        run_dir = meta_file.parent
        bench_file = run_dir / "benchmarks.json"
        if not bench_file.exists():
            print(f"skip (no benchmarks.json): {run_dir}", file=sys.stderr)
            continue

        try:
            meta = json.loads(meta_file.read_text())
            data = json.loads(bench_file.read_text())
        except Exception as e:
            print(f"skip (parse error) {run_dir}: {e}", file=sys.stderr)
            continue

        gpu = {}
        gpu.update(parse_dmon(run_dir / "gpu_dmon.log"))
        gpu.update(parse_mem(run_dir / "gpu_mem.log"))

        benches = data.get("benchmarks") or data.get("results") or []
        if isinstance(benches, dict):
            benches = [benches]

        for b in benches:
            row = extract_client(b, meta)
            row.update(gpu)
            rows.append(enrich(row))

    cols = [
        "run_id", "test_id", "model", "workload",
        "prompt_tokens", "output_tokens", "strategy", "target_rate",
        # client
        "req_per_sec", "output_tok_per_sec",
        "ttft_p95_ms", "itl_p95_ms", "e2e_p95_s",
        # gpu
        "sm_util_p95", "mem_bw_util_p95", "vram_peak_gb", "avg_power_w",
        # derived
        "tokens_per_joule", "bottleneck",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


# ---------- reporting ----------

def write_markdown(df: pd.DataFrame, out_path: Path):
    lines = ["# vLLM Benchmark Summary", ""]
    if df.empty:
        lines.append("_No benchmark data found._")
        out_path.write_text("\n".join(lines))
        return

    lines += [
        f"Sub-benchmarks: **{len(df)}**",
        f"Models: {sorted(df['model'].dropna().unique().tolist())}",
        f"Workloads: {sorted(df['workload'].dropna().unique().tolist())}",
        "",
    ]

    # 1) Headline: peak output tok/s per (model, workload)
    lines += ["## Peak output throughput per (model × workload)", ""]
    peak = (df.sort_values("output_tok_per_sec", ascending=False)
              .groupby(["model", "workload"], as_index=False).first())
    lines.append(peak[[
        "model", "workload",
        "req_per_sec", "output_tok_per_sec",
        "ttft_p95_ms", "itl_p95_ms", "e2e_p95_s",
        "sm_util_p95", "mem_bw_util_p95", "vram_peak_gb",
        "tokens_per_joule", "bottleneck",
    ]].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # 2) Sync baseline
    lines += ["## Sync baseline — best-case latency", ""]
    sync = df[df["test_id"] == "T1"][[
        "model", "workload",
        "output_tok_per_sec", "ttft_p95_ms", "itl_p95_ms", "e2e_p95_s",
        "sm_util_p95", "mem_bw_util_p95",
    ]]
    lines.append(sync.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    # 3) Sweep curve (T3)
    lines += ["## Sweep — throughput vs p95 latency", ""]
    sweep = df[df["test_id"] == "T3"].sort_values(
        ["model", "workload", "req_per_sec"])[[
        "model", "workload",
        "req_per_sec", "output_tok_per_sec",
        "ttft_p95_ms", "itl_p95_ms", "e2e_p95_s",
        "sm_util_p95", "mem_bw_util_p95", "bottleneck",
    ]]
    lines.append(sweep.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append("---")
    lines.append("_Generated by analyze_benchmarks.py_")
    out_path.write_text("\n".join(lines))


def plot_curves(df: pd.DataFrame, out_dir: Path):
    if not HAS_MPL:
        return
    sweep = df[df["test_id"] == "T3"].dropna(subset=["req_per_sec", "e2e_p95_s"])
    for (model, wl), g in sweep.groupby(["model", "workload"]):
        g = g.sort_values("req_per_sec")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(g["req_per_sec"], g["e2e_p95_s"], "o-", label="E2E p95 (s)")
        ax1.set_xlabel("Requests / sec")
        ax1.set_ylabel("E2E latency p95 (s)")
        ax2 = ax1.twinx()
        ax2.plot(g["req_per_sec"], g["ttft_p95_ms"], "s--",
                 color="tab:orange", label="TTFT p95 (ms)")
        ax2.set_ylabel("TTFT p95 (ms)")
        fig.suptitle(f"{model} — {wl}")
        fig.tight_layout()
        safe = f"{model}_{wl}".replace("/", "_")
        fig.savefig(out_dir / f"curve_{safe}.png", dpi=120)
        plt.close(fig)


# ---------- main ----------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        sys.exit(1)

    df = walk_results(root)
    if df.empty:
        print("No benchmarks found.", file=sys.stderr)
        sys.exit(2)

    df.to_csv(root / "summary.csv", index=False)
    write_markdown(df, root / "summary.md")
    plot_curves(df, root)

    print(f"Summary CSV: {root/'summary.csv'}")
    print(f"Summary MD:  {root/'summary.md'}")
    print(f"Rows:        {len(df)}")


if __name__ == "__main__":
    main()
