"""Back-of-envelope serving capacity for a model on a GPU (vLLM-style, single GPU).

  python scripts/estimate.py --model Qwen/Qwen3-8B --gpu A10G --input-tokens 4000 --output-tokens 256

Reads the model's config.json (HF hub id or local path). Covers dense and simple-MoE GQA models,
plus hybrid models (e.g. Qwen3.5/3.8: linear-attention layers + periodic full-attention layers),
where only the full-attention layers hold a growing KV cache.
MLA models (DeepSeek) need less KV than estimated here. Override params with --params if unsure.
For hybrid models ALWAYS pass --params: the config-derived count ignores linear-attention layers.
Not modeled: the constant-size recurrent state that each linear-attention layer keeps per sequence.
"""
import argparse, json, os, httpx

GiB = 2**30
# visible GiB, memory bandwidth GB/s, dense bf16 tensor TFLOPS (datasheet, approximate)
GPUS = {
    "A10G": (22.0, 600, 125), "L4": (22.0, 300, 121), "RTX4090": (23.6, 1008, 165),
    "L40S": (44.4, 864, 362), "A100-40G": (39.4, 1555, 312), "A100-80G": (79.2, 2039, 312),
    "H100": (79.2, 3350, 989), "H200": (139.8, 4800, 989),
}
BW_EFF, FLOP_EFF = 0.75, 0.5  # calibrated on A10G + Qwen3-8B (results/logs/baseline.log)


def load_config(model):
    if os.path.exists(model):
        return json.load(open(os.path.join(model, "config.json") if os.path.isdir(model) else model))
    tok = os.getenv("HF_TOKEN")
    r = httpx.get(f"https://huggingface.co/{model}/resolve/main/config.json", follow_redirects=True,
                  headers={"Authorization": f"Bearer {tok}"} if tok else {}, timeout=30)
    r.raise_for_status()
    return r.json()


SKIP = ("vision", "visual", "audio", "talker", "token2wav", "speech", "vit")
NEED = ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")


def text_cfg(cfg, inherited=None):
    """Walk nested multimodal configs (text_config, thinker_config.text_config, ...) down to the LLM's
    config. Scalar keys from outer levels are inherited; inner values win. Skips encoder/talker branches."""
    merged = {**(inherited or {}), **{k: v for k, v in cfg.items() if not isinstance(v, dict)}}
    if all(k in merged for k in NEED):
        return merged
    for k, v in cfg.items():
        if isinstance(v, dict) and not any(s in k for s in SKIP):
            found = text_cfg(v, merged)
            if found:
                return found
    return None


def shape(cfg):
    c = text_cfg(cfg)
    if c is None:
        raise SystemExit(f"couldn't find an LLM config (hidden_size, num_hidden_layers, ...) in config.json; "
                         f"top-level keys: {list(cfg)}")
    h, layers, heads = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
    kv = c.get("num_key_value_heads", heads)
    hd = c.get("head_dim") or h // heads
    # Hybrid models: only full-attention layers have a per-token KV cache.
    # Prefer the explicit per-layer list; fall back to "every Nth layer is full attention".
    lt = c.get("layer_types")
    if lt:
        kv_layers = sum(1 for t in lt if "linear" not in t)
    else:
        kv_layers = layers // c.get("full_attention_interval", 1)
    attn = 2 * h * heads * hd + 2 * h * kv * hd
    n_exp = c.get("num_experts") or c.get("num_local_experts") or 0
    if n_exp:
        one = 3 * h * (c.get("moe_intermediate_size") or c["intermediate_size"])
        mlp_total, mlp_active = n_exp * one, c.get("num_experts_per_tok", 1) * one
    else:
        mlp_total = mlp_active = 3 * h * c["intermediate_size"]
    emb = c["vocab_size"] * h * (1 if c.get("tie_word_embeddings") else 2)
    active = layers * (attn + mlp_active) + emb
    # decode reads one embedding row, not the whole input table (unless tied to the LM head)
    read = active - (0 if c.get("tie_word_embeddings") else c["vocab_size"] * h)
    return {"layers": layers, "kv_layers": kv_layers, "kv_heads": kv, "head_dim": hd,
            "params": layers * (attn + mlp_total) + emb, "active": active, "read": read}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", choices=GPUS, default="A10G")
    ap.add_argument("--vram-gib", type=float)
    ap.add_argument("--bw-gbs", type=float)
    ap.add_argument("--tflops", type=float)
    ap.add_argument("--params", type=float, help="billions; overrides the config-derived count")
    ap.add_argument("--active-params", type=float, help="billions; MoE active params per token (overrides config-derived)")
    ap.add_argument("--weight-bytes", type=float, default=2, help="2=bf16, 1=fp8/int8, 0.5=int4")
    ap.add_argument("--kv-bytes", type=float, default=2, help="2=bf16, 1=fp8 KV cache")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--overhead-gib", type=float, default=1.8, help="activations + CUDA graphs + runtime")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048, help="--max-num-batched-tokens")
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--slo-ttft", type=float, default=2.0, help="seconds, p50")
    ap.add_argument("--slo-itl", type=float, default=0.1, help="seconds, p99")
    a = ap.parse_args()

    vram, bw, tflops = GPUS[a.gpu]
    vram, bw, tflops = a.vram_gib or vram, a.bw_gbs or bw, a.tflops or tflops
    s = shape(load_config(a.model))
    hybrid = s["kv_layers"] < s["layers"]
    if a.params:
        s["params"] = s["active"] = s["read"] = a.params * 1e9
    if a.active_params:  # MoE: params read per decode step / used per prefill token
        s["active"] = s["read"] = a.active_params * 1e9
    i, o = a.input_tokens, a.output_tokens

    weights = s["params"] * a.weight_bytes
    kv_per_tok = 2 * s["kv_layers"] * s["kv_heads"] * s["head_dim"] * a.kv_bytes
    budget = a.gpu_util * vram * GiB - weights - a.overhead_gib * GiB
    kv_tokens = max(0, budget / kv_per_tok)
    max_seqs = min(int(kv_tokens // (i + o)), a.max_num_seqs)
    prefill = tflops * 1e12 * FLOP_EFF / (2 * s["active"])
    itl = lambda b: (s["read"] * a.weight_bytes + b * (i + o / 2) * kv_per_tok) / (bw * 1e9 * BW_EFF)

    print(f"{a.model} on {a.gpu} ({vram} GiB, {bw:.0f} GB/s, {tflops:.0f} TFLOPS), request {i} in / {o} out")
    if hybrid:
        print(f"  note: hybrid model, {s['kv_layers']}/{s['layers']} layers use full attention (hold KV)"
              + ("" if a.params else "; param count ignores linear-attn layers, pass --params"))
    print(f"  1 weights           {s['params']/1e9:.2f}B params x {a.weight_bytes} B = {weights/GiB:.1f} GiB")
    print(f"  2 KV per token      2 x {s['kv_layers']} layers x {s['kv_heads']} kv_heads x {s['head_dim']} dim"
          f" x {a.kv_bytes} B = {kv_per_tok/1024:.0f} KiB")
    print(f"  3 KV budget         {a.gpu_util} x {vram} - {weights/GiB:.1f} - {a.overhead_gib} = "
          f"{budget/GiB:.2f} GiB = {kv_tokens:,.0f} tokens")
    if budget <= 0:
        return print("  !! weights + overhead don't fit: raise --gpu-util, quantize weights, or use a bigger GPU")
    if kv_tokens < a.max_model_len:
        print(f"  !! KV holds fewer than --max-model-len {a.max_model_len} tokens: vLLM will refuse to start")
    print(f"  4 max running reqs  {kv_tokens:,.0f} / {i + o} = {max_seqs}   <- the knee")
    print(f"  5 decode ITL        {s['read']*a.weight_bytes/1e9:.1f} GB / {bw:.0f} GB/s / {BW_EFF} = "
          f"{itl(1)*1000:.0f} ms at c=1")
    print(f"  6 prefill           {prefill:,.0f} tok/s -> TTFT {i/prefill:.2f} s at c=1")
    if max_seqs == 0:
        return print("  !! a single request doesn't fit in the KV cache")

    print(f"\n  {'conc':>4} {'running':>7} {'agg tok/s':>9} {'user tok/s':>10} {'TTFT p50':>9} "
          f"{'ITL p50':>8} {'ITL p99':>8}  SLO")
    usable = 0
    for c in a.concurrency:
        b = min(c, max_seqs)
        cycle = b * i / prefill + o * itl(b)  # GPU time for b requests to finish together
        queue = max(0, c - b) / b * cycle
        ttft = i / prefill * (1 + (b - 1) / 2) + queue
        # p99 lands on a stall once >~1.5% of a request's gaps are other requests' prefill chunks
        stalls = (b - 1) * -(-i // a.chunk) / o
        p99 = itl(b) + (min(i, a.chunk) / prefill if stalls > 0.015 else 0)
        ok = ttft <= a.slo_ttft and p99 <= a.slo_itl
        usable = c if ok else usable
        print(f"  {c:>4} {b:>7} {b*o/cycle:>9.0f} {o/(cycle+queue):>10.1f} {ttft:>8.1f}s "
              f"{itl(b)*1000:>6.0f}ms {p99*1000:>6.0f}ms  {'ok' if ok else '-'}")
    print(f"\n  usable concurrency at TTFT<={a.slo_ttft}s, ITL p99<={a.slo_itl*1000:.0f}ms: {usable or 'none'}")


if __name__ == "__main__":
    main()