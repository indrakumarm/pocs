"""Throughput-vs-latency curves, knee and SLO check per tag, from a bench CSV.

  python scripts/plot.py results/week3.csv [--tags baseline fp8-kv] [--slo-ttft 2 --slo-itl 0.1]

Writes results/<csv-name>.png and prints a per-tag table (gain from the previous level, SLO, knee).
"""
import argparse, os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KNEE_GAIN = 1.10  # knee = last level before throughput grows < 10% from one level to the next


def summarize(g, slo_ttft, slo_itl):
    g = g.sort_values("concurrency").copy()
    # mean request time ≈ output tokens / per-request tok/s (bench.py doesn't store E2E percentiles)
    g["e2e_s"] = g.output_tokens / g.per_req_tok_s
    g["gain"] = g.agg_tok_s.pct_change()
    g["slo_ok"] = (g.ttft_p99_ms <= slo_ttft * 1000) & (g.itl_p99_ms <= slo_itl * 1000)
    flat = g[g.agg_tok_s.shift(-1) / g.agg_tok_s < KNEE_GAIN]
    knee = flat.concurrency.iloc[0] if len(flat) else None
    usable = g[g.slo_ok].concurrency.max() if g.slo_ok.any() else None
    return g, knee, usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tags", nargs="+")
    ap.add_argument("--slo-ttft", type=float, default=2.0, help="seconds, p99")
    ap.add_argument("--slo-itl", type=float, default=0.1, help="seconds, p99")
    ap.add_argument("--out")
    a = ap.parse_args()

    df = pd.read_csv(a.csv).drop_duplicates(["tag", "concurrency"], keep="last")
    if a.tags:
        df = df[df.tag.isin(a.tags)]
    levels = sorted(df.concurrency.unique())
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    (tp_ttft, tp_itl), (c_tp, c_e2e) = ax

    for tag, g in df.groupby("tag", sort=False):
        g, knee, usable = summarize(g, a.slo_ttft, a.slo_itl)
        color = tp_ttft.plot(g.agg_tok_s, g.ttft_p99_ms / 1000, marker="o", label=tag)[0].get_color()
        tp_itl.plot(g.agg_tok_s, g.itl_p99_ms, marker="o", color=color, label=tag)
        c_tp.plot(g.concurrency, g.agg_tok_s, marker="o", color=color, label=tag)
        c_e2e.plot(g.concurrency, g.e2e_s, marker="o", color=color, label=f"{tag} E2E")
        c_e2e.plot(g.concurrency, g.ttft_p50_ms / 1000, ls="--", color=color, label=f"{tag} TTFT p50")
        for _, r in g.iterrows():
            tp_ttft.annotate(f"c={r.concurrency}", (r.agg_tok_s, r.ttft_p99_ms / 1000),
                             textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)
        if knee is not None:
            k = g[g.concurrency == knee].iloc[0]
            tp_ttft.plot(k.agg_tok_s, k.ttft_p99_ms / 1000, "*", ms=16, color=color)
            c_tp.plot(knee, k.agg_tok_s, "*", ms=16, color=color)

        print(f"\n== {tag}   knee: c={knee}   usable at SLO (TTFT p99<={a.slo_ttft}s, "
              f"ITL p99<={a.slo_itl*1000:.0f}ms): c={usable}")
        print(f"  {'conc':>4} {'in':>5} {'agg tok/s':>9} {'gain':>6} {'TTFT p99':>9} {'ITL p99':>8} "
              f"{'E2E~':>7}  SLO")
        for _, r in g.iterrows():
            gain = "" if pd.isna(r.gain) else f"{r.gain:+.0%}"
            mark = "  <- knee" if r.concurrency == knee else ""
            print(f"  {r.concurrency:>4} {r.input_tokens:>5} {r.agg_tok_s:>9.0f} {gain:>6} "
                  f"{r.ttft_p99_ms/1000:>8.2f}s {r.itl_p99_ms:>6.0f}ms {r.e2e_s:>6.1f}s  "
                  f"{'ok' if r.slo_ok else '-'}{mark}")

    tp_ttft.axhline(a.slo_ttft, ls=":", color="red", label=f"SLO {a.slo_ttft}s")
    tp_itl.axhline(a.slo_itl * 1000, ls=":", color="red", label=f"SLO {a.slo_itl*1000:.0f}ms")
    tp_ttft.set(xlabel="aggregate output tok/s", ylabel="TTFT p99 (s, log)", yscale="log",
                title="Throughput vs TTFT p99  (★ = knee: curve turns vertical)")
    tp_itl.set(xlabel="aggregate output tok/s", ylabel="ITL p99 (ms)", title="Throughput vs ITL p99")
    c_tp.set(xlabel="concurrency", ylabel="aggregate output tok/s", xscale="log",
             title="Throughput by concurrency (flat = past the knee)")
    c_e2e.set(xlabel="concurrency", ylabel="seconds (log)", xscale="log", yscale="log",
              title="Per-request time: E2E ≈ TTFT + (out-1) × ITL")
    for x in ax.flat:
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    for x in (c_tp, c_e2e):
        x.set_xticks(levels, labels=levels)

    out = a.out or os.path.splitext(a.csv)[0] + ".png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
