# Week 4: Quantization, speculative decoding, vLLM vs SGLang

```bash
make sweep4 && make plot4      # automated: results/week4.csv, results/curve.png
```

## Checklist
- [ ] Quantization: BF16 vs FP8 (dynamic and pre-quantized) vs AWQ. Record weights size, KV tokens (from log), tok/s, TTFT
- [ ] Quality check per quant (speed without quality is meaningless), see below
- [ ] Speculative decoding: ngram first, then EAGLE/draft model. Test at concurrency 1 vs 32
- [ ] Same model on SGLang; compare at each concurrency level
- [ ] (Optional) TensorRT-LLM: heavier setup; do it after SGLang

## Quality check (manual)
```bash
pip install "lm-eval[api]"
lm_eval --model local-chat-completions \
  --model_args model=Qwen/Qwen3-8B,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=16 \
  --tasks gsm8k --limit 200 --apply_chat_template
```
Run the same command against each quantized server and tabulate the score next to the speed.

## What to expect / hypotheses
| Experiment | Hypothesis |
|---|---|
| FP8 | ~1.3–1.8× decode speed on H100/Ada/Blackwell (halves weight bytes). On A100/3090 there is no FP8 compute (weight-only, smaller gain) |
| AWQ/GPTQ (4-bit) | Fastest at low concurrency (bandwidth-bound), gains shrink at high concurrency as it becomes compute-bound |
| Spec decoding | Big win at concurrency 1–4; neutral or negative at high concurrency, since the GPU is already saturated by batching |
| ngram | Only helps when output copies from the prompt (summarization, code edits). `bench.py` uses synthetic prompts, so benchmark with real data too |

**Caveat:** speculative decoding speedups depend on acceptance rate; check `spec_decode` metrics via `curl localhost:8000/metrics | grep spec`.

## Manual commands (debugging / one-off)
```bash
# vLLM with FP8
docker run --rm --gpus all --ipc=host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 8192

# SGLang
docker run --rm --gpus all --ipc=host -p 8000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 30000 --context-length 8192

curl localhost:8000/v1/models        # works on both
curl localhost:8000/metrics | head   # SGLang needs --enable-metrics
docker logs -f <container>
```
Version drift is normal: if a flag is rejected, check `--help` in that image tag and pin the tag in the config.
