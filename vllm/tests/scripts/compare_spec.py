#!/usr/bin/env python3
"""
Compare spec-decoding on vs off ablation runs produced by run_spec_ablation.sh.

Walks <root>/<label>_<timestamp>/ folders, uses the LATEST run per label
(so re-runs are safe), and prints:

  1. Side-by-side per-stream table (aggregate tok/s, TTFT p95, ITL p95).
  2. Delta table (% change of spec_on vs spec_off).
  3. Estimated inversion concurrency (where spec_on stops beating spec_off).
  4. Spec-decoding acceptance rate (from vLLM /metrics before/after).
  5. Peak SM% / peak VRAM per label (from nvidia-smi logs).

Usage:
    python3 compare_spec.py ~/spec_ablation
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas tabulate", file=sys.stderr)
    sys.exit(1)


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


# ---------- per-stage extraction ----------

def extract_stages(bench_json: dict) -> pd.DataFrame:
    benches = bench_json.get("benchmarks") or bench_json.get("results") or []
    if isinstance(benches, dict):
        benches = [benches]

    rows = []
    for b in benches:
        streams = _first(b, "args.strategy.streams", "strategy.streams")
        if isinstance(streams, list):
            streams = streams[0] if streams else None
        rows.append({
            "streams": streams,
            "req_per_sec": _first(b,
                "metrics.requests_per_second.successful.mean",
                "metrics.requests_per_second.mean"),
            "gen_tok_per_sec": _first(b,
                "metrics.output_tokens_per_second.successful.mean",
                "metrics.output_tokens_per_second.mean"),
            "ttft_p95_ms": _first(b,
                "metrics.time_to_first_token_ms.successful.percentiles.p95",
                "metrics.time_to_first_token_ms.percentiles.p95"),
            "itl_p95_ms": _first(b,
                "metrics.inter_token_latency_ms.successful.percentiles.p95",
                "metrics.inter_token_latency_ms.percentiles.p95"),
            "e2e_p95_s": _first(b,
                "metrics.request_latency.successful.percentiles.p95",
                "metrics.request_latency.percentiles.p95"),
        })
    df = pd.DataFrame(rows).dropna(subset=["streams"])
    if not df.empty:
        df["streams"] = df["streams"].astype(int)
        df["per_user_tok_per_sec"] = df["gen_tok_per_sec"] / df["streams"]
    return df.sort_values("streams").reset_index(drop=True)


# ---------- vLLM /metrics parsing (spec acceptance rate) ----------

def _parse_metric_line(line: str) -> tuple[str, float] | None:
    if line.startswith("#") or not line.strip():
        return None
    m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+\-]+)$", line.strip())
    if not m:
        return None
    return m.group(1), float(m.group(3))


def _sum_metric(path: Path, name: str) -> float | None:
    if not path.exists():
        return None
    total = 0.0
    found = False
    for line in path.read_text().splitlines():
        parsed = _parse_metric_line(line)
        if parsed and parsed[0] == name:
            total += parsed[1]
            found = True
    return total if found else None


def spec_acceptance(before: Path, after: Path) -> dict:
    """Compute spec-decoding acceptance rate over the run window."""
    names_accepted = [
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens",
    ]
    names_draft = [
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_draft_tokens",
    ]

    def diff(names: list[str]) -> float | None:
        for n in names:
            a = _sum_metric(after, n)
            b = _sum_metric(before, n)
            if a is not None and b is not None:
                return max(0.0, a - b)
        return None

    accepted = diff(names_accepted)
    drafted = diff(names_draft)
    if accepted is None or drafted is None or drafted == 0:
        return {"accepted": accepted, "drafted": drafted, "rate": None}
    return {"accepted": accepted, "drafted": drafted, "rate": accepted / drafted}


# ---------- GPU log summary ----------

def gpu_summary(run_dir: Path) -> dict:
    result = {"sm_p95": None, "power_mean_w": None, "vram_peak_gb": None}
    dmon = run_dir / "gpu_dmon.log"
    mem  = run_dir / "gpu_mem.log"

    if dmon.exists():
        rows = []
        for line in dmon.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rows.append((float(parts[3]), float(parts[6])))  # pwr, sm
            except ValueError:
                continue
        if rows:
            df = pd.DataFrame(rows, columns=["pwr", "sm"])
            active = df[df["sm"] > 0]
            if active.empty:
                active = df
            result["sm_p95"] = float(active["sm"].quantile(0.95))
            result["power_mean_w"] = float(active["pwr"].mean())

    if mem.exists():
        try:
            mdf = pd.read_csv(mem, skipinitialspace=True)
            used_col = next((c for c in mdf.columns if "memory.used" in c), None)
            if used_col:
                result["vram_peak_gb"] = round(float(mdf[used_col].max()) / 1024.0, 2)
        except Exception:
            pass
    return result


# ---------- walker ----------

def latest_run_per_label(root: Path) -> dict[str, Path]:
    """Given <root>/<label>_<timestamp>/, return {label: latest_dir}."""
    by_label: dict[str, list[Path]] = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^(.+)_(\d{8}_\d{6})$", d.name)
        if not m:
            continue
        label = m.group(1)
        by_label.setdefault(label, []).append(d)
    return {lbl: sorted(dirs)[-1] for lbl, dirs in by_label.items()}


# ---------- reporting ----------

def render(off: pd.DataFrame, on: pd.DataFrame) -> str:
    if off.empty or on.empty:
        return "One of the labels has no benchmark data — cannot compare."

    merged = pd.merge(
        off.add_suffix("_off"), on.add_suffix("_on"),
        left_on="streams_off", right_on="streams_on",
        how="inner",
    ).rename(columns={"streams_off": "streams"}).drop(columns=["streams_on"])

    for col in ("gen_tok_per_sec", "per_user_tok_per_sec", "ttft_p95_ms", "itl_p95_ms", "e2e_p95_s"):
        c_off = f"{col}_off"
        c_on = f"{col}_on"
        if c_off in merged and c_on in merged:
            merged[f"{col}_delta_pct"] = ((merged[c_on] - merged[c_off]) / merged[c_off]) * 100.0

    lines = ["## Side-by-side per-concurrency", ""]
    side = merged[[
        "streams",
        "gen_tok_per_sec_off", "gen_tok_per_sec_on",
        "per_user_tok_per_sec_off", "per_user_tok_per_sec_on",
        "ttft_p95_ms_off", "ttft_p95_ms_on",
        "itl_p95_ms_off", "itl_p95_ms_on",
    ]]
    lines.append(side.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines += ["## Delta (spec_on vs spec_off, %)", ""]
    delta = merged[[
        "streams",
        "gen_tok_per_sec_delta_pct",
        "per_user_tok_per_sec_delta_pct",
        "ttft_p95_ms_delta_pct",
        "itl_p95_ms_delta_pct",
    ]]
    lines.append(delta.to_markdown(index=False, floatfmt="+.1f"))
    lines.append("")

    # Inversion point: first N where per-user tok/s stops improving
    inversion = None
    for _, row in merged.iterrows():
        d = row.get("per_user_tok_per_sec_delta_pct")
        if d is not None and d <= 0:
            inversion = int(row["streams"])
            break

    lines += ["## Verdict", ""]
    if inversion is None:
        lines.append("- Spec decoding **helps at every measured concurrency**. Consider testing higher N to find the inversion.")
    else:
        best = merged[merged["per_user_tok_per_sec_delta_pct"] > 0]
        if best.empty:
            lines.append("- Spec decoding **hurts at every measured concurrency**. Turn it off.")
        else:
            best_n = int(best.iloc[0]["streams"])
            worst_n = int(best.iloc[-1]["streams"])
            lines.append(f"- Spec decoding **helps up to N={worst_n}**, inverts at **N={inversion}**.")
            lines.append(f"- Best single-user speedup at **N={best_n}** = {best.iloc[0]['per_user_tok_per_sec_delta_pct']:+.1f}%.")
    lines.append("")
    return "\n".join(lines)


# ---------- main ----------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        sys.exit(1)

    runs = latest_run_per_label(root)
    if "spec_off" not in runs or "spec_on" not in runs:
        print("Need both 'spec_off' and 'spec_on' runs under", root, file=sys.stderr)
        print("Found:", list(runs), file=sys.stderr)
        sys.exit(2)

    dfs = {}
    for label, run_dir in runs.items():
        bench_file = run_dir / "benchmarks.json"
        if not bench_file.exists():
            print(f"skip {label}: no benchmarks.json", file=sys.stderr)
            continue
        data = json.loads(bench_file.read_text())
        dfs[label] = extract_stages(data)

    print("# Spec-decoding ablation report")
    print()
    print(f"- spec_off run: `{runs['spec_off'].name}`")
    print(f"- spec_on  run: `{runs['spec_on'].name}`")
    print()

    print(render(dfs["spec_off"], dfs["spec_on"]))

    # Acceptance rate
    print("## Spec acceptance rate (from vLLM /metrics)")
    print()
    acc = spec_acceptance(
        runs["spec_on"] / "vllm_metrics_before.txt",
        runs["spec_on"] / "vllm_metrics_after.txt",
    )
    if acc["rate"] is not None:
        print(f"- Draft tokens generated: {acc['drafted']:.0f}")
        print(f"- Accepted tokens: {acc['accepted']:.0f}")
        print(f"- **Acceptance rate: {acc['rate']*100:.1f}%**  "
              f"({'good' if acc['rate']>=0.6 else 'low — spec likely hurting'})")
    else:
        print("- Acceptance metrics not found (metric name mismatch or spec off)")
    print()

    # GPU summary
    print("## GPU summary")
    print()
    for label in ("spec_off", "spec_on"):
        g = gpu_summary(runs[label])
        print(f"- **{label}**: sm p95 = "
              f"{g['sm_p95']:.0f}%  " if g['sm_p95'] is not None else f"- **{label}**: sm p95 = n/a  ",
              end="")
        print(f"avg power = {g['power_mean_w']:.1f} W  " if g['power_mean_w'] is not None else "avg power = n/a  ", end="")
        print(f"peak VRAM = {g['vram_peak_gb']:.2f} GB" if g['vram_peak_gb'] is not None else "peak VRAM = n/a")


if __name__ == "__main__":
    main()
