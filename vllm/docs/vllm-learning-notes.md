# vLLM Learning Notes

> Interactive step-by-step notes on vLLM, LLM serving, and GPU optimization.
> Built incrementally — new topics appended as we go.

---

## Roadmap

1. **What is vLLM actually doing?** — PagedAttention, KV cache, continuous batching
2. **GPU basics for LLM serving** — VRAM, cores, how a model sits on a GPU
3. **Measuring what's happening** — tokens/sec, TTFT, TPOT, KV cache utilization, `/metrics`
4. **Key vLLM configs** — `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, tensor parallelism, quantization
5. **Optimization techniques** — prefix caching, chunked prefill, speculative decoding (draft models)
6. **When to use what** — decision framework by workload shape

---

# Step 1: What is vLLM actually doing?

## The core problem: LLM generation is weird

When a model generates `"The cat sat on the mat"`:
- It generates **one token at a time**, autoregressively.
- Each new token must **attend to all previous tokens**.
- To avoid recomputing everything, it stores **Key/Value tensors** for every previous token → the **KV cache**.

The KV cache is often **bigger than the model weights** at scale.

## Why naive batching fails

Traditional batching: "wait for 32 requests, run one forward pass, return 32 results."

Breaks for LLMs because:
- Request A wants 10 output tokens, Request B wants 500. Whole batch waits for the slowest.
- New requests arriving mid-generation wait for the current batch to finish.
- Result: bad latency, bad GPU utilization.

## vLLM's 3 core innovations

### 1. PagedAttention (memory trick)
KV cache stored in fixed-size **pages**, not one contiguous block per request.
- Old way: pre-allocate max-length KV cache → tons of waste.
- vLLM: allocate pages on demand → **2–4× more requests fit** in the same GPU memory.

### 2. Continuous batching (scheduling trick)
Every generation step, vLLM decides which requests to keep, drop, or add.
New requests **join mid-flight** — no waiting for the batch to drain.

### 3. Prefix caching (reuse trick)
Two requests sharing a prefix (e.g., same system prompt) reuse the same KV cache pages for the shared part. Huge win for RAG, agents, long system prompts.

## Visual: batching styles

```
Time →

Naive batching:
[Req A: ▓▓▓▓▓▓▓▓░░░░░░░░ done]
[Req B: ▓▓▓▓░░░░░░░░░░░░ done]      ← GPU wasted after B finishes
[Req C: waits...........starts]
[Req D: waits...........starts]

Continuous batching (vLLM):
[Req A: ▓▓▓▓▓▓▓▓ done]
[Req B: ▓▓▓▓ done, Req E: ▓▓▓▓▓▓ starts here]
[Req C: ▓▓▓▓▓▓▓▓▓▓▓▓]                ← GPU always full
[Req D: ▓▓▓▓▓▓ done, Req F: ▓▓▓▓ starts here]
```

## Resources
- 📄 vLLM / PagedAttention paper: https://arxiv.org/abs/2309.06180
- 📝 Original vLLM blog: https://blog.vllm.ai/2023/06/20/vllm.html
- 🎥 Woosuk Kwon talk (~30 min): https://www.youtube.com/watch?v=5ZlavKF_98U
- 📝 Anyscale continuous-batching explainer: https://www.anyscale.com/blog/continuous-batching-llm-inference

---

# Deep Dive: Autoregressive generation & attention

## "One token at a time, autoregressively"

A **token** = a chunk of text (word or subword). `"The cat sat on the mat"` might tokenize as 6 tokens.

**Autoregressive** = generate one token, feed it back as input, generate the next.

```
Step 1: input "The cat sat"           → output " on"
Step 2: input "The cat sat on"        → output " the"
Step 3: input "The cat sat on the"    → output " mat"
Step 4: input "The cat sat on the mat"→ output "."
Step 5: ...                            → output <END>
```

- **Every token = one full forward pass** through the model.
- A 500-token response = 500 forward passes.
- That's why **tokens/sec** is the key generation metric.

## "Each new token attends to all previous tokens"

Modern LLMs are **Transformers**; core operation is **attention**.

> When deciding the next token, the model looks at every previous token and weights how relevant each one is to what comes next.

Example: generating after `"The cat sat on the"`:
```
Previous:  "The"  "cat"  "sat"  "on"  "the"
Weights:   0.05   0.35   0.20   0.15  0.25
              → predicts " mat"
```

Context grows every step. Token 1000 attends to 999 previous tokens.

## Why this needs the KV cache

Naively recomputing all past K/V each step = O(n²) work — infeasible.

Solution: cache K and V for past tokens; only compute K/V for the newest token each step.

## Resources
- 🎥 3Blue1Brown "But what is a GPT?": https://www.youtube.com/watch?v=wjZofJX0v4M
- 🎥 3Blue1Brown "Attention visually explained": https://www.youtube.com/watch?v=eMlx5fFNoYc
- 📝 Jay Alammar "Illustrated GPT-2": https://jalammar.github.io/illustrated-gpt2/

---

# Deep Dive 1: KV cache concretely

## Where K and V come from

Each token, at each transformer layer, produces three vectors:
- **Q (Query)** — "what am I looking for?"
- **K (Key)**   — "what do I offer to searchers?"
- **V (Value)** — "if attended to, here's my content"

Search-index analogy: Q = search term, K = index entries, V = retrieved content.

Only **K and V of past tokens** are cached (they never change). Q is computed fresh for the current token.

## What's physically stored

```
KV cache size per token =
    2                     (K and V)
  × num_layers            (stacked transformer layers)
  × num_kv_heads          (attention heads)
  × head_dim              (dimension per head)
  × bytes_per_value       (2 for fp16/bf16, 1 for fp8/int8)
```

### Llama-3-8B example
| Parameter | Value |
|---|---|
| num_layers | 32 |
| num_kv_heads | 8 (GQA) |
| head_dim | 128 |
| dtype | bf16 (2 bytes) |

```
per_token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB / token
```

## Concrete scale on A100-80GB with Llama-3-8B

| Item | Memory |
|---|---|
| Model weights (bf16) | ~16 GB |
| Activations / overhead | ~4 GB |
| **Available for KV cache** | ~60 GB |
| **Total KV budget** | ~490K tokens |

Split across users:
- 100 users × 4000 tokens = 50 GB → OK
- 500 users × 4000 tokens = 250 GB → won't fit → requests queue

**KV cache is the #1 memory concern, not model weights.**

## Attention variants & their KV cost

- **MHA** (Multi-Head Attention) — 1 KV head per Q head. Largest cache. (Llama-2)
- **GQA** (Grouped-Query) — fewer KV heads shared. Standard now. (Llama-3, Qwen2, Mistral)
- **MQA** (Multi-Query) — just 1 KV head. Smallest, slight quality loss.
- **MLA** (Multi-head Latent) — DeepSeek's compressed latent. Massive savings.

## Generalized formula

```
per_token_bytes = 2 × L × H_kv × D_head × dtype_bytes
```

- `L` = layers
- `H_kv` = `num_key_value_heads` (from model's `config.json`)
- `D_head` = `hidden_size / num_attention_heads`
- `dtype_bytes` = 2 (bf16/fp16), 1 (fp8/int8), 0.5 (int4)

## Reference table (bf16)

| Model | Layers | KV Heads | Head Dim | Per Token | 4K tokens |
|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | **128 KB** | 512 MB |
| Llama-3-70B | 80 | 8 | 128 | **320 KB** | 1.3 GB |
| Qwen2.5-7B | 28 | 4 | 128 | **56 KB** | 224 MB |
| Mistral-7B | 32 | 8 | 128 | **128 KB** | 512 MB |
| Llama-2-7B (MHA) | 32 | 32 | 128 | **512 KB** | 2 GB |
| DeepSeek-V3 (MLA) | 61 | latent | — | ~34 KB | 136 MB |

Note: Llama-2-7B uses **4×** the KV memory of Llama-3-8B despite being similar size.

## Key takeaway

Every vLLM optimization (paging, prefix caching, quantization, GQA) is fundamentally about **fitting more KV cache into limited GPU memory**.

## Resources
- 📝 HF KV-cache blog: https://huggingface.co/blog/kv-cache-quantization
- 📝 Transformer inference arithmetic: https://kipp.ly/transformer-inference-arithmetic/

---

# Deep Dive 2: What is a token, in memory?

Tracing the word `"unbelievable"` from string → GPU memory.

## Stage 1: Text → Token IDs (tokenizer)

Models can't process strings — only integers. The **tokenizer** converts text via a fixed **vocabulary** (~32K–200K entries).

```
"unbelievable"
   │  BPE splits into known subwords
["un", "believ", "able"]
   │  vocab lookup
[  665,  5297,    481 ]   ← token IDs (each is a 32-bit int)
```

- BPE = **Byte-Pair Encoding**, learned by greedily merging frequent character pairs.
- Common words → 1 token; rare/complex words → multiple tokens.

### Rules of thumb
- 1 token ≈ 4 chars ≈ 0.75 English words
- Code/JSON/non-English → more tokens per character
- Emojis, Chinese, Arabic → often multiple tokens per glyph

### Try it
- OpenAI tokenizer viz: https://platform.openai.com/tokenizer
- HF tokenizer playground: https://huggingface.co/spaces/Xenova/the-tokenizer-playground

## Stage 2: Token ID → Embedding vector

Each token ID indexes into an **embedding table**.

```
Embedding table shape: [vocab_size, hidden_size]
Llama-3-8B:            [128,256,   4096]
```

Row 665 = 4096-dim float vector = the model's learned representation of "un".

Memory per token embedding (bf16):
```
4096 × 2 bytes = 8 KB
```

Embedding table itself: `128,256 × 4096 × 2 ≈ 1 GB` of the model.

## Stage 3: Vector flows through transformer layers

At **each of 32 layers**:

```
input vec [4096]
   ├─ LayerNorm
   ├─ Attention: compute Q, K, V (each 4096-d), attend to past K/V → output
   ├─ residual add
   ├─ LayerNorm
   ├─ Feed-forward (MLP): 4096 → 14336 → 4096
   └─ residual add
   ▼
output vec [4096] → next layer
```

After layer 32: one last projection → **logits** of size 128,256 (one score per vocab entry). Softmax → probabilities → sample → next token.

## Stage 4: KV cache — what actually gets stored

During Stage 3, at every layer, K and V vectors are written to the cache.

For Llama-3-8B, per token per layer:
```
K: 8 heads × 128 head_dim × 2 bytes = 2 KB
V:                                   = 2 KB
Total per layer:                       4 KB
Across 32 layers:                    128 KB  ✓ matches earlier calc
```

## Full memory picture for ONE token

| Component | Size | Persistent? |
|---|---|---|
| Token ID | 4 bytes | brief |
| Embedding vector | 8 KB | transient |
| Q per layer | 2 KB | transient |
| **K per layer** | **2 KB** | **cached ✓** |
| **V per layer** | **2 KB** | **cached ✓** |
| Attention output | 8 KB | transient |
| Logits (once at end) | 256 KB | transient |
| **Total persistent KV** | **~128 KB / token** | **yes** |

Only the **KV cache** persists between generation steps. Everything else is discarded and recomputed — that's why KV cache is the memory bottleneck.

## Ties back to vLLM configs

- `--max-model-len 8192` → per-request KV cache upper bound
- `--max-num-seqs 256` → total concurrent KV budget
- `--kv-cache-dtype fp8` → halves KV memory (2 → 1 byte per value)
- `--gpu-memory-utilization 0.9` → % of VRAM vLLM claims for weights + KV

## Resources
- 📝 Jay Alammar "Illustrated Word2Vec": https://jalammar.github.io/illustrated-word2vec/
- 📝 HF BPE tokenizer chapter: https://huggingface.co/learn/nlp-course/chapter6/5
- 🛠️ `tiktoken` (OpenAI tokenizer, offline): `pip install tiktoken`
- 🛠️ `transformers.AutoTokenizer.from_pretrained(...)` for any HF model

---

# Clarification: Tokens vs Embeddings

**They are NOT the same thing** — two different stages of the pipeline.

| | **Token** | **Embedding** |
|---|---|---|
| What | Integer ID (a symbol) | Vector of floats (meaning) |
| Example | `665` | `[0.02, -0.85, ..., 0.11]` (4096 floats) |
| Size | 4 bytes | ~8 KB (bf16) |
| Human-readable? | Yes via vocab ("un") | No |
| Origin | Fixed by tokenizer | Learned during training |
| Analogy | Word in a dictionary | Meaning of the word |

## Pipeline
```
"unbelievable"
      │ tokenizer
      ▼
  [665, 5297, 481]           ← TOKENS (integers)
      │ embedding table lookup
      ▼
  [[...4096 floats...],       ← EMBEDDINGS (vectors)
   [...4096 floats...],
   [...4096 floats...]]
      │ transformer layers
      ▼
  contextual representations
```

An embedding is what a token **becomes** inside the model. Tokens are the discrete units we count/bill; embeddings are the continuous vectors the model computes with.

## Analogy: library
- **Token** = book's call number (`QA76.9.C55`)
- **Embedding** = the book's actual content

## Three flavors of "embedding" you'll hear
| Term | Meaning |
|---|---|
| **Token embedding** | Vector for one token entering layer 1 |
| **Contextual embedding** | Vector for a token after some layers (context-aware) |
| **Sentence/document embedding** | Single vector for a whole text — what "embedding models" like `text-embedding-3` produce; used in RAG |

## Practical usage
- **Tokens** → counting, pricing, context limits, `/metrics`
- **Embeddings** → internal model math (mostly hidden), or explicit RAG/similarity search

---

# Deep Dive 3: PagedAttention — the OS-page analogy

## A. OS virtual memory refresher

OS chops RAM into fixed-size **pages** (4 KB). Each program has a **page table** mapping virtual pages → physical pages.

Benefits:
1. **No contiguity needed** — data scattered in physical RAM looks contiguous virtually.
2. **Lazy allocation** — pages allocated only when touched.
3. **Sharing** — same physical page mapped into multiple programs (shared libs).

vLLM steals all three for the KV cache.

## B. The naive-KV-cache waste

Old frameworks pre-allocated a giant contiguous KV tensor per request, sized for `max_model_len` (e.g., 4096 tokens).

```
Allocated:  [████████████████████████████████████████]  4096 slots
Used:       [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  200 used, 3896 WASTED
```

Types of waste:
- **Internal** — over-provisioned per request
- **Reservation** — leftover gaps between allocations
- **External** — free memory exists but no big contiguous chunk

Measured: only 20–40% of KV memory actually used. GPUs that *could* serve 500 requests served ~100.

## C. PagedAttention design

1. Chop GPU KV memory into **KV blocks** (default **16 tokens** of K,V each, across all layers).
2. Each request keeps a **block table** mapping logical token positions → physical blocks.
3. Allocate blocks **on demand**.

```
Physical KV memory:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │ ...
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

Request A (40 tokens, needs 3 blocks: 16+16+8):
  block table → [B5, B2, B8]      ← scattered, doesn't matter

Request B (20 tokens): → [B1, B6]
Request C (8 tokens):  → [B3]
```

Waste bounded to at most 1 block per request (~4%) vs up to 96% before.

## D. Three OS-inspired superpowers

### 1. No contiguity needed
Custom **PagedAttention CUDA kernel** reads K/V from scattered blocks via the block table.

### 2. Lazy allocation
Grow the block table as generation proceeds. 200-token response = 13 blocks, not 256.

### 3. Sharing → this IS prefix caching
Two requests with same system prompt share physical blocks:

```
Request X: [B7, B3, B9,  B1,  B4 ]
Request Y: [B7, B3, B9,  B12, B15]
             └──shared──┘
```

Copy-on-write if either needs to modify a shared block. Same as OS shared memory.

**Same 4K system prompt across 100 users → stored ONCE, not 100×.**

## E. What this unlocks

| Benefit | Result |
|---|---|
| 2–4× more concurrent requests | More throughput/$ |
| Prefix caching (free) | RAG, agents, chat get huge speedups |
| Preemption (swap KV blocks to CPU RAM) | Graceful degradation vs OOM |
| Beam search / parallel sampling share prompt blocks | Cheap structured generation |

## F. The knob to know

`--block-size N` (default **16**). Smaller = less waste + more overhead; larger = opposite. Default fine 99% of the time.

## One-paragraph summary

> vLLM treats KV cache like OS virtual memory. Fixed-size **blocks** (16 tokens), a **block table** per request, lazy allocation, and shared blocks for common prefixes. Eliminates 60–80% memory waste, enables prefix caching, and fits 2–4× more concurrent requests on the same GPU.

## Resources
- 📄 PagedAttention paper (Section 4 = block table diagrams): https://arxiv.org/abs/2309.06180
- 📝 vLLM blog: https://blog.vllm.ai/2023/06/20/vllm.html
- 🎥 Woosuk Kwon SOSP talk: https://www.youtube.com/watch?v=5ZlavKF_98U

---

# Q&A: Prefix cache lifecycle & sharing rules

## Q1: What counts as "same prompt" for sharing?

**Exact match, prefix-only, at 16-token block boundaries.**

- vLLM hashes each 16-token block.
- Two requests share a block only if it's byte-for-byte identical AND all previous blocks matched too.
- Sharing must be **contiguous from the start** — attention is directional (token N's K/V depends on tokens 1…N), so a diff at token 5 poisons every following block even if later words match.

### Example
```
User A: "You are a helpful assistant. The user asks: What is 2+2?"
User B: "You are a helpful assistant. The user asks: What is the capital of France?"

Shared prefix = 32 tokens = 2 blocks (B0, B1)

User A: [B0][B1][B2 for "2+2?"]
User B: [B0][B1][B2 for "the capital..."]
           ↑    ↑
        physically shared
```

Diverge at token 33? Blocks B0, B1 shared; B2 onward separate.

## Q2: When is KV cache cleared?

There is **no server-side session**. Every API call is independent — client sends full history each turn. vLLM's "memory" is entirely the LRU prefix cache.

### Request lifecycle
1. Request arrives → blocks allocated as it prefills + generates.
2. Request ends (EOS, max_tokens, or disconnect).
3. Its block table is discarded.
4. Physical blocks are **marked freeable but not erased** — they stay in an LRU cache indexed by content hash.

### Eviction
When memory pressure demands new blocks, vLLM evicts **least-recently-used** cached blocks. Blocks currently in use by active requests are **pinned** and never evicted.

### Timeline
```
Time  Event                                    Block state
────  ──────────────────────────────────────  ──────────────────
t=0   User A starts                            computed, in use
t=1   User A finishes                          freed → LRU cached
t=2   User B sends same prefix (10 min later)  HIT — reused ✓
t=3   ...heavy traffic evicts it...            evicted
t=4   User C sends same prefix                 MISS — recompute + recache
```

## Q3: Does A finishing hurt B if B shares the prefix?

**Never.** Blocks in active use are reference-counted and pinned. When A ends, its ref drops; if B still holds it, block stays. If nobody holds it, it moves to LRU cache (still there, just evictable).

**No in-flight request ever loses its KV cache** due to another request finishing.

## Q4: How do "chats" feel fast without sessions?

Client re-sends full history each turn. Turn 2's prompt = Turn 1's exact prefix + new content. That prefix is almost always still in LRU cache → **cache hit → fast**.

If cache was evicted (long idle + heavy traffic), Turn N's TTFT goes up but result is still correct — just recomputes.

## Q5: Does prefix cache survive restart?

**No.** GPU-memory-only. Restart vLLM → cache empty. (Some experimental tools like LMCache persist to CPU RAM / NVMe for very long prompts.)

## Summary

| Question | Answer |
|---|---|
| Same prompt = exact match? | Yes, byte-for-byte |
| Sharing granularity | 16-token blocks |
| Middle-of-prompt sharing? | Never (attention is directional) |
| KV cleared on request end? | Freed but LRU-cached |
| A ending hurts B? | No — pinned while in use |
| When actually evicted? | LRU under memory pressure |
| vLLM knows about sessions? | No — prefix cache does the work |
| Survives restart? | No |

## Knobs
- `--enable-prefix-caching` — on by default in vLLM v0.4+
- Metric: `prefix_cache_hit_rate` in `/metrics`. Good ranges:
  - Chat: 60–90%
  - RAG: 50–80%
  - One-shot random prompts: near 0%

---

# Step 2: GPU basics for LLM serving

## 2.1 CPU vs GPU
| | CPU | GPU |
|---|---|---|
| Cores | 8–128 fast, complex | 1000s simple |
| Best at | Branching, single-thread | Massive parallel matmul |
| Memory | 100s GB RAM, ~50 GB/s | 10s GB VRAM, ~1–3 TB/s |

GPUs win at LLMs because transformers = giant matmuls, done in parallel.

## 2.2 Four things that matter

### a) VRAM
Weights + KV cache + activations must fit here. **Usually the bottleneck.**

| GPU | VRAM |
|---|---|
| L4 | 24 GB |
| A10G | 24 GB |
| A100 40GB | 40 GB |
| A100 80GB | 80 GB |
| L40S | 48 GB |
| H100 80GB | 80 GB |
| H200 141GB | 141 GB |
| B200 | 192 GB |

### b) Memory bandwidth (the real decode bottleneck)
Every generated token reads the whole model from VRAM once.
16 GB model / 2 TB/s ≈ ~125 tokens/sec theoretical ceiling per stream.

| GPU | Bandwidth |
|---|---|
| L4 | 300 GB/s |
| A10G | 600 GB/s |
| A100 80GB | 2.0 TB/s |
| L40S | 864 GB/s |
| H100 80GB | 3.35 TB/s |
| H200 141GB | 4.8 TB/s |
| B200 | 8.0 TB/s |

### c) Compute (tensor cores, TFLOPS)
Matters for prefill (big matmul over the whole prompt).
A100 has NO native fp8 → H100+ dramatic win for fp8 models.

| GPU | bf16 TFLOPS | fp8 TFLOPS |
|---|---|---|
| A100 80GB | 312 | — |
| L40S | 362 | 733 |
| H100 80GB | 989 | 1979 |
| H200 | 989 | 1979 |
| B200 | 2250 | 4500 |

### d) NVLink / interconnect
For multi-GPU tensor parallelism. NVLink = 900 GB/s; PCIe = 64 GB/s. **You want NVLink.**

## 2.3 How a model sits on an A100-80GB (Llama-3-8B)

```
80 GB total
 ├── 16 GB weights (bf16)
 ├── ~1.5 GB CUDA/PyTorch overhead
 ├── ~2 GB activations
 ├── ~4 GB safety margin
 └── ~56.5 GB KV cache pool (~450K tokens of KV budget)
```

`--gpu-memory-utilization 0.9` = use 90% VRAM.

### 70B doesn't fit on 80GB
70B × 2 bytes = 140 GB. Options:
1. Quantize to fp8/int8 → 70 GB (tight KV) → slow
2. Tensor parallelism 2×H100 → 70 GB weights each → healthy
3. H200 141 GB → fits standalone
4. 2× H100 recommended

## 2.4 Prefill vs Decode (critical!)

Every request has TWO phases:

| Phase | What | Bottleneck | Determines |
|---|---|---|---|
| **Prefill** | Process whole prompt in parallel | Compute (tensor cores) | TTFT |
| **Decode** | Generate 1 token at a time | Memory bandwidth | TPOT |

Long prompt / short output (RAG) → prefill-dominated → **compute matters**
Short prompt / long output (chat, codegen) → decode-dominated → **bandwidth matters**

## 2.5 Continuous batching mixes them
vLLM packs prefills (compute-hungry) alongside decodes (bandwidth-hungry) so both hardware resources stay busy. `--enable-chunked-prefill` (default now) chops long prefills so they don't stall decodes.

## 2.6 How to inspect a GPU

### `nvidia-smi`
```
Memory-Usage: 62450 / 81920 MiB   ← 76% used
GPU-Util:     87%                  ← ⚠️ misleading! just "was a kernel running"
Pwr:          120W / 700W          ← low = idle/waiting
```

### `nvidia-smi dmon -s pucm` — live streaming
### `nvtop` — htop-style TUI (highly recommended)
### PyTorch runtime:
```python
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```
### `nsys profile` — Nsight Systems, deep kernel timeline

## 2.7 VRAM sizing formula

```
VRAM = model_weights + overhead + (max_concurrent × avg_tokens × per_token_KV)
```

Example: Llama-3-8B, 100 users, 3000 tokens avg
```
= 16 + 4 + (100 × 3000 × 128 KB)
= 16 + 4 + 38.4 ≈ 58 GB   → fits A100-80GB
```

Reverse: "how many users fit?"
```
concurrent = (VRAM - weights - overhead) / (avg_tokens × per_token_KV)
```
A10G-24GB + Llama-3-8B → ~10 concurrent users only.

## Recap
- VRAM is #1 constraint
- Bandwidth (not compute) caps decode throughput
- Prefill compute-bound, Decode bandwidth-bound
- fp8 needs H100+
- Multi-GPU needs NVLink
- Tools: nvidia-smi, nvtop, nsys

## Resources
- 📝 Horace He, "LLM inference speed of light": https://www.thonking.ai/p/llm-inference-speed-of-light
- 📝 "Making DL Go Brrrr from first principles": https://horace.io/brrr_intro.html
- 📝 H100 whitepaper: https://resources.nvidia.com/en-us-tensor-core

---

# vLLM VRAM Calculator (self-contained)

## Master formula
```
VRAM_needed  ≈  model_weights + overhead + KV_cache_pool
KV_cache_pool = max_concurrent_seqs × avg_tokens_per_seq × per_token_KV
```

## Piece 1: model_weights

```
model_weights = num_parameters × bytes_per_parameter
```

### bytes_per_parameter by dtype

| dtype | bytes | Notes |
|---|---|---|
| fp32 | 4 | rare in inference |
| **fp16 / bf16** | **2** | **default** |
| fp8 | 1 | needs H100+ |
| int8 | 1 | any GPU |
| int4 (AWQ, GPTQ) | 0.5 | small quality loss |
| int3 / int2 | 0.375 / 0.25 | notable quality loss |

**Rule of thumb**: `dtype × params_in_B ≈ GB`. bf16 × 8B = 16 GB. int4 × 70B = 35 GB.

Quick reference — 8B model:
| dtype | Size |
|---|---|
| fp32 | 32 GB |
| bf16 | 16 GB |
| fp8/int8 | 8 GB |
| int4 | 4 GB |

## Piece 2: per_token_KV

```
per_token_KV = 2 × num_layers × num_kv_heads × head_dim × bytes_per_value
head_dim     = hidden_size / num_attention_heads
```

`bytes_per_value` follows `--kv-cache-dtype`:
- bf16 → 2
- fp8/int8 → 1
- int4 → 0.5

**Yes — per_token_KV varies per model.** Architecture (MHA/GQA/MLA), layer count, hidden size all change it.

### Where to find the values

Every HuggingFace model has `config.json`:

```bash
curl -s https://huggingface.co/<org>/<model>/raw/main/config.json | jq \
  '{num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_size}'
```

Mapping:
| Formula | config.json field |
|---|---|
| num_layers | `num_hidden_layers` |
| num_kv_heads | `num_key_value_heads` (falls back to `num_attention_heads` for MHA) |
| head_dim | `hidden_size / num_attention_heads` (or explicit `head_dim`) |

## Piece 3: overhead

```
overhead ≈ 2–6 GB   (use 4 GB for back-of-envelope)
```

Components: CUDA context (~1–2 GB), activations (~1–2 GB), vLLM safety (~1–2 GB), CUDA graphs (~0.5–1 GB).

## Piece 4: KV dtype (`--kv-cache-dtype`)

Store KV cache in a **different** dtype than model weights:

| flag | bytes/value | Effect |
|---|---|---|
| `auto` | matches model | baseline |
| `fp8` | 1 | ~2× KV pool, needs H100+ |
| `fp8_e5m2` | 1 | alternative fp8 encoding |
| `int8` | 1 | ~2× KV pool, works anywhere |

Halving KV = 2× concurrent users. Huge lever.

---

## Worked Example 1: Llama-3.1-8B (GQA)

`config.json`:
```json
{"num_hidden_layers": 32, "num_attention_heads": 32,
 "num_key_value_heads": 8, "hidden_size": 4096}
```

`head_dim = 4096/32 = 128`

**per_token_KV (bf16)** = 2 × 32 × 8 × 128 × 2 = **128 KB**

**weights**:
- bf16: 16 GB
- fp8:  8 GB
- int4: 4 GB

**Total VRAM for 100 users × 3000 tokens:**

| Config | Weights | KV pool | +Overhead | Total |
|---|---|---|---|---|
| bf16 + bf16 KV | 16 | 38.4 | +4 | **58.4 GB** |
| bf16 + fp8 KV | 16 | 19.2 | +4 | **39.2 GB** |
| fp8 + fp8 KV | 8 | 19.2 | +4 | **31.2 GB** |
| int4 + fp8 KV | 4 | 19.2 | +4 | **27.2 GB** |

## Worked Example 2: Qwen2.5-14B (GQA, bigger)

`config.json`:
```json
{"num_hidden_layers": 48, "num_attention_heads": 40,
 "num_key_value_heads": 8, "hidden_size": 5120}
```

`head_dim = 5120/40 = 128`

**per_token_KV (bf16)** = 2 × 48 × 8 × 128 × 2 = **192 KB**  (50% more than Llama-3.1-8B → more layers)

**weights**:
- bf16: 28 GB
- fp8: 14 GB
- int4: 7 GB

**Total VRAM for 50 users × 4000 tokens:**

| Config | Weights | KV pool | +Overhead | Total |
|---|---|---|---|---|
| bf16 + bf16 KV | 28 | 38.4 | +4 | **70.4 GB** (over 90% of A100-80GB) |
| bf16 + fp8 KV | 28 | 19.2 | +4 | **51.2 GB** |
| fp8 + fp8 KV | 14 | 19.2 | +4 | **37.2 GB** |
| int4 + fp8 KV | 7 | 19.2 | +4 | **30.2 GB** (fits L40S-48GB) |

## Contrast: Llama-2-7B (MHA, old)

```json
{"num_hidden_layers": 32, "num_attention_heads": 32,
 "num_key_value_heads": 32, "hidden_size": 4096}
```
`per_token_KV (bf16)` = 2 × 32 × 32 × 128 × 2 = **512 KB / token**

**4× more KV than Llama-3.1-8B** despite similar size. This is why GQA was such a big deal for serving.

## Reverse formula: how many users fit?

```
max_users = (VRAM_GB − weights_GB − overhead_GB) × 1024 / (avg_tokens × per_token_KV_MB)
```

H100-80GB, bf16 Llama-3-8B + fp8 KV, avg 4000 tokens:
```
= (80 − 16 − 4) × 1024 / (4000 × 0.0625)
= 60 × 1024 / 250
≈ 245 concurrent users
```
(0.0625 MB/token = 64 KB with fp8 KV.)

## Python calculator (copy-paste)

```python
def vllm_vram_estimate(
    num_params_B, num_layers, num_kv_heads,
    hidden_size, num_attention_heads,
    weight_dtype_bytes=2,   # 2=bf16, 1=fp8/int8, 0.5=int4
    kv_dtype_bytes=2,
    max_concurrent=100,
    avg_tokens=3000,
    overhead_gb=4,
):
    head_dim = hidden_size / num_attention_heads
    per_token_kv_bytes = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
    weights_gb = num_params_B * weight_dtype_bytes
    kv_pool_gb = (max_concurrent * avg_tokens * per_token_kv_bytes) / (1024**3)
    total_gb = weights_gb + overhead_gb + kv_pool_gb

    print(f"per-token KV: {per_token_kv_bytes/1024:.0f} KB")
    print(f"weights:      {weights_gb:.1f} GB")
    print(f"KV pool:      {kv_pool_gb:.1f} GB  ({max_concurrent}×{avg_tokens} tok)")
    print(f"overhead:     {overhead_gb:.1f} GB")
    print(f"TOTAL:        {total_gb:.1f} GB")
    return total_gb

# Llama-3.1-8B, default
vllm_vram_estimate(8, 32, 8, 4096, 32)

# Qwen2.5-14B, fp8 weights + fp8 KV, 50 users, 4K ctx
vllm_vram_estimate(14, 48, 8, 5120, 40,
    weight_dtype_bytes=1, kv_dtype_bytes=1,
    max_concurrent=50, avg_tokens=4000)
```

---

# Step 3: Measuring what's happening

## 3.1 The four latency numbers

```
send → [queue] → [prefill] → [1st tok] → [decode...decode] → done
       ↑         ↑           ↑                                ↑
    queue     leaves        TTFT                          e2e latency
    time      queue
```

| Metric | Meaning | Good value |
|---|---|---|
| Queue time | Waiting before scheduling | < 50 ms |
| **TTFT** | Send → first token | 200 ms – 2 s |
| **TPOT** | Steady inter-token delay | 10–50 ms |
| E2E latency | Total time | depends on output len |

Derived:
- Per-stream throughput = 1 / TPOT
- Server throughput = Σ across in-flight requests

Targets:
| Use case | TTFT | TPOT |
|---|---|---|
| Chat | <500 ms | <40 ms (25 tok/s) |
| Code completion | <200 ms | <25 ms |
| RAG/summary | <2 s | <40 ms |
| Batch | — | maximize throughput |

## 3.2 vLLM `/metrics` endpoint

Prometheus format at `http://<host>:8000/metrics`.

### Latency histograms
```
vllm:time_to_first_token_seconds_bucket
vllm:time_per_output_token_seconds_bucket
vllm:e2e_request_latency_seconds_bucket
vllm:request_queue_time_seconds_bucket
vllm:request_prefill_time_seconds_bucket
vllm:request_decode_time_seconds_bucket
```
Query as `histogram_quantile(0.95, ...)`. **Alert on p95/p99, never averages.**

### Throughput counters
```
vllm:prompt_tokens_total
vllm:generation_tokens_total
```
Use `rate(...[1m])` for current tokens/sec.

### Request state
```
vllm:num_requests_running    # on GPU right now
vllm:num_requests_waiting    # queued
vllm:num_requests_swapped    # KV evicted to CPU RAM (⚠️)
```

### THE memory metric
```
vllm:gpu_cache_usage_perc    # 0.0–1.0
```
| Value | Meaning |
|---|---|
| <50% | Under-utilized |
| 50–85% | Sweet spot ✓ |
| 85–95% | Tight, prefix cache churn |
| >95% | Preemption imminent ⚠️ |

### Prefix cache hit rate
```
vllm:gpu_prefix_cache_hit_rate
```
| Workload | Expected |
|---|---|
| Chat w/ long sys prompt | 60–90% |
| RAG | 50–80% |
| Agents | 70–95% |
| Random one-off | 0–5% |

### Healthy production snapshot
```
num_requests_running          82 (max=100)
num_requests_waiting          3
num_requests_swapped          0
gpu_cache_usage_perc          0.78
gpu_prefix_cache_hit_rate     0.72
rate(generation_tokens_total) ~3800 tok/s
p95 TTFT                      0.45 s
p95 TPOT                      0.028 s (~36 tok/s)
```

## 3.3 Monitoring stack setup

### Prometheus scrape
```yaml
scrape_configs:
  - job_name: vllm
    scrape_interval: 5s
    static_configs:
      - targets: ['vllm-host:8000']
    metrics_path: /metrics
```

### Grafana
Official dashboard JSON:
📊 https://github.com/vllm-project/vllm/tree/main/examples/online_serving/prometheus_grafana

### Quick live watch (no stack)
```bash
watch -n 1 'curl -s localhost:8000/metrics | \
  grep -E "num_requests_(running|waiting|swapped)|gpu_cache_usage|prefix_cache_hit"'
```

## 3.4 Correlating with nvidia-smi / nvtop

| Symptom | Diagnosis |
|---|---|
| High cache% + high util + healthy p95 | ✅ At capacity |
| High cache% + growing waiting | Scale replicas |
| Low cache% + low util + slow | Non-GPU bottleneck (net, tokenizer, client) |
| High util + low tok/s | Bandwidth-bound; check TPOT |
| Swapped > 0 | Cut max-num-seqs, enable fp8 KV, or bigger GPU |
| Prefix hit dropping | KV pool too small for diversity |

## 3.5 Load testing

### Built-in
```bash
vllm bench serve \
  --model meta-llama/Llama-3.1-8B \
  --dataset-name sharegpt \
  --num-prompts 500 --request-rate 10
```

### More realistic: `guidellm`
```bash
pip install guidellm
guidellm benchmark --target http://localhost:8000 \
  --data prompt_tokens=1024,output_tokens=256 \
  --rate-type sweep
```

## 3.6 Measurement checklist

1. Confirm `/metrics` responds
2. `vllm bench serve` at ~10 rps
3. `nvidia-smi`/`nvtop` in parallel — GPU actually engaged
4. `num_requests_swapped` stays 0
5. Ramp load until p95 TTFT/TPOT exceeds SLO OR waiting grows → capacity ceiling
6. At ceiling, `gpu_cache_usage_perc` should be 80–95%
7. Check prefix hit rate with **real** prompts
8. Save baseline numbers before tuning

## 3.7 The 5 metrics that matter most

1. `num_requests_waiting`
2. `gpu_cache_usage_perc`
3. `gpu_prefix_cache_hit_rate`
4. p95 `time_to_first_token_seconds`
5. p95 `time_per_output_token_seconds`

## Resources
- 📊 vLLM Grafana dashboard: https://github.com/vllm-project/vllm/tree/main/examples/online_serving/prometheus_grafana
- 📝 vLLM metrics docs: https://docs.vllm.ai/en/latest/serving/metrics.html
- 📝 NVIDIA on TTFT/TPOT: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- 🛠️ guidellm: https://github.com/neuralmagic/guidellm

---

# Hands-on Lab: vLLM + Qwen3-8B-AWQ on g4dn.xlarge (T4)

## Setup

- Instance: AWS g4dn.xlarge (NVIDIA T4, 16 GB VRAM, Turing arch)
- Model: `Qwen/Qwen3-8B-AWQ` (int4-quantized ≈ 5 GB weights)
- Prometheus: existing at `http://10.192.12.112:9090`

## ⚠️ Port collision fix
Prometheus is on 9090 → run vLLM on **8000** (or anything else).

## VRAM sanity check
Qwen3-8B: layers=36, kv_heads=8, hidden=4096, head_dim=128
```
per-token KV (bf16) = 2 × 36 × 8 × 128 × 2 = 147 KB
1 × 15K-token req  = 2.15 GB just for its KV
Weights (AWQ int4) ≈ 5 GB
Overhead           ≈ 2 GB
KV pool            ≈ 9 GB → ~4 concurrent 15K reqs, ~30 at 2K avg
```
T4 = no fp8. Use `--kv-cache-dtype auto`.

## Launch command (fixed)
```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --dtype half --served-model-name qwen \
  --gpu-memory-utilization 0.90 --max-model-len 15000 \
  --disable-log-requests
```

## Pre-flight
```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
python -c "import vllm; print(vllm.__version__)"   # want ≥ 0.6
pip install -U vllm    # if needed
sudo apt install -y nvtop
pip install guidellm
```

## Smoke tests
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi."}],"max_tokens":20}'

curl -s http://localhost:8000/metrics | grep -E 'vllm:(num_requests|gpu_cache_usage|prefix_cache_hit)'
```

## Prometheus config (on Prometheus box)
```yaml
scrape_configs:
  - job_name: vllm
    scrape_interval: 5s
    static_configs:
      - targets: ['<g4dn-ip>:8000']
    metrics_path: /metrics
  - job_name: gpu
    scrape_interval: 5s
    static_configs:
      - targets: ['<g4dn-ip>:9400']
```
Reload: `curl -X POST http://10.192.12.112:9090/-/reload`
Verify: `http://10.192.12.112:9090/targets`

## DCGM exporter (GPU hardware metrics)
```bash
sudo docker run -d --gpus all --rm --name dcgm-exporter \
  -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
curl -s http://localhost:9400/metrics | head -20
```
Key metrics:
- `DCGM_FI_DEV_GPU_UTIL` — compute utilization
- `DCGM_FI_DEV_MEM_COPY_UTIL` — **bandwidth utilization (LLM decode ceiling)**
- `DCGM_FI_DEV_FB_USED` — VRAM used
- `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`

## Grafana dashboards
- vLLM: https://raw.githubusercontent.com/vllm-project/vllm/main/examples/online_serving/prometheus_grafana/grafana.json
- DCGM: Grafana.com dashboard ID `12239`

## Live watch (no dashboard needed)
```bash
watch -n 1 'echo "=== GPU ==="; \
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv; \
  echo; echo "=== vLLM ==="; \
  curl -s localhost:8000/metrics 2>/dev/null | \
  grep -E "vllm:(num_requests_(running|waiting|swapped)|gpu_cache_usage_perc|gpu_prefix_cache_hit_rate) " | grep -v "^#"'
```
Or: `nvtop`

## Load tests

### Test 1 — Baseline
```bash
vllm bench serve --model Qwen/Qwen3-8B-AWQ --served-model-name qwen \
  --host localhost --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 20 --request-rate 1
```

### Test 2 — Ramp to find ceiling
```bash
for RATE in 1 2 5 10 20; do
  echo "=== rate=$RATE rps ==="
  vllm bench serve --model Qwen/Qwen3-8B-AWQ --served-model-name qwen \
    --host localhost --port 8000 \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --request-rate $RATE
  sleep 20
done
```
Capacity ceiling = last rate before p95 SLO break OR `num_requests_waiting` grows.

### Test 3 — Realistic mixed
```bash
vllm bench serve --model Qwen/Qwen3-8B-AWQ --served-model-name qwen \
  --host localhost --port 8000 \
  --dataset-name sharegpt --num-prompts 500 --request-rate 5
```

### Test 4 — Long context stress
```bash
vllm bench serve --model Qwen/Qwen3-8B-AWQ --served-model-name qwen \
  --host localhost --port 8000 \
  --dataset-name random --random-input-len 8000 --random-output-len 512 \
  --num-prompts 20 --request-rate 0.5
```

### Test 5 — Prefix cache demo
```bash
python <<'PY'
import requests
SYSTEM = "You are an expert Python programmer. " * 200   # ~1500 tokens
for i in range(50):
    requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "qwen",
        "messages": [{"role":"system","content":SYSTEM},
                     {"role":"user","content":f"Print number {i}."}],
        "max_tokens": 20,
    })
PY
curl -s localhost:8000/metrics | grep prefix_cache_hit_rate
```
Compare hit rate vs. sending 50 unique random prompts.

## Security note
- Open port 8000 + 9400 only to Prometheus & your workstation
- vLLM has NO auth — never internet-exposed
- Consider nginx + basic-auth in front

## Expected numbers on T4 + Qwen3-8B-AWQ
- Cold start: 60–90 s (weight download + load)
- TTFT (512 in): 500 ms – 1.5 s
- TPOT: 30–60 ms (~15–30 tok/s per stream, bandwidth-limited)
- Max concurrency before degrading: ~10–20 seqs
- `gpu_cache_usage_perc` saturates fast at 15K contexts

## Pre-launch checklist
- [ ] vLLM port ≠ 9090
- [ ] `nvidia-smi` shows T4 with ~15 GB free
- [ ] vLLM ≥ 0.6
- [ ] nvtop, guidellm installed
- [ ] SG allows 8000 + 9400 from Prometheus + workstation
- [ ] dcgm-exporter running (optional)
- [ ] Prometheus target UP
- [ ] Smoke curl works
- [ ] `/metrics` non-zero after smoke test

---

# Lab Results: g4dn.xlarge + Qwen2.5-7B-Instruct-AWQ

## Bench results (200 requests, 512-in / 128-out)

| Rate (rps) | Actual req/s | Output tok/s | Peak conc | Med TTFT | Med TPOT | P99 TTFT |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.98 | 125 | 28 | 479 ms | 52 ms | 1.6 s |
| 2 | 1.89 | 242 | 37 | 470 ms | 77 ms | 2.1 s |
| 5 | **1.72** ⚠️ | 220 | 200 | **22.3 s** 💥 | 570 ms | 44.8 s |
| 10 | 4.34 | 555 | 200 | 538 ms | 232 ms | 2.5 s |
| 20 | 4.21 | 539 | 200 | 896 ms | 308 ms | 3.3 s |

### Findings
- **Real throughput ceiling: ~4 rps / ~550 tok/s.**
- **TPOT scales with concurrency**: 52 ms (rate 1) → 308 ms (rate 20) — total tok/s ceiling divided across users.
- **Rate=5 anomaly**: Poisson burst + no concurrency limit → 60s queue spike + preemption. Rate=10 came cleaner because prefix cache warmed (same `seed=0`).
- **Chat SLO capacity** (TTFT<1s, TPOT<60ms → ≥17 tok/s): ~1 rps only.
- **Batch throughput sweet spot**: rate 10 (555 tok/s), but per-user experience is poor.

## Config learned from `vllm:cache_config_info`

| Label | Value | Meaning |
|---|---|---|
| enable_prefix_caching | True | prefix caching ON |
| block_size | 16 | tokens per KV block |
| cache_dtype | auto | KV in bf16 (2 bytes/value) |
| gpu_memory_utilization | 0.9 | 90% of VRAM used by vLLM |
| num_gpu_blocks | 7,728 | total KV blocks |
| **kv_cache_size_tokens** | **123,582** | **total KV token budget** |
| **kv_cache_max_concurrency** | **8.24** | max 15K-token requests |

### Back-of-envelope
- per-token KV = 2 × 28 layers × 4 kv_heads × 128 head_dim × 2 = **56 KB/token**
- 123K tokens × 56 KB ≈ 6.9 GB KV pool
- 200 concurrent 640-tok requests = 128K tokens → **exactly at the wall**, hence rate=5 blowup

### Prefix cache success
```
prefix_cache_queries_total = 307,200
prefix_cache_hits_total    = 189,984  → hit rate = 61.8% ✓
```
Because same `seed=0` reused prompts across runs. Real workloads see less unless system prompts are shared.

## Grafana fixes needed

Old metric → New metric:
```
vllm:gpu_cache_usage_perc → vllm:kv_cache_usage_perc
```

### Useful new panels
```promql
# KV usage
vllm:kv_cache_usage_perc

# Prefix hit rate over 5 min
rate(vllm:prefix_cache_hits_total[5m]) 
  / 
rate(vllm:prefix_cache_queries_total[5m])

# Tokens saved by cache per second
rate(vllm:prompt_tokens_by_source_total{source="local_cache_hit"}[5m])
```

---

# 🗺️ Mental Model: The 4 Worlds of LLM Serving

Every term you'll encounter lives in one of 4 worlds. Know the world, terms stop being confusing.

## Overview
```
     WORLD 1: HARDWARE           WORLD 2: MEMORY
     ─────────────────           ─────────────────
     • VRAM                      • Model weights
     • Bandwidth                 • KV cache
     • Tensor cores / FLOPS      • Blocks / pages
                                 • Prefix cache

     WORLD 3: REQUESTS           WORLD 4: TIME
     ─────────────────           ─────────────────
     • Prompt / tokens           • TTFT
     • Concurrent seqs           • TPOT / ITL
     • Queue depth               • E2E latency
     • Prefill vs decode         • Throughput
     • Preemption
```

## World 1: HARDWARE (what CAN the GPU do?)
| Term | Meaning | Example |
|---|---|---|
| VRAM | GPU's own RAM | 16 GB on T4 |
| Bandwidth | VRAM read speed | 300 GB/s on T4 |
| Tensor cores | Matmul units | Do the math |
| FLOPS | Ops/sec ceiling | Matters for prefill |

## World 2: MEMORY (what's IN the GPU now?)
| Term | Meaning | Your lab |
|---|---|---|
| Model weights | Trained network | 5 GB (AWQ 7B) |
| KV cache | Attention K,V of past tokens | Grows with conversations |
| KV block | 16-token chunk | 7,728 blocks |
| Prefix cache | Freed blocks kept for reuse | 61.8% hit rate |
| `kv_cache_usage_perc` | % of KV pool in use | Watch in Grafana |
| `kv_cache_size_tokens` | Total KV token budget | 123,582 |

## World 3: REQUESTS (what are users sending?)
| Term | Meaning | Your lab |
|---|---|---|
| Prompt / input tokens | User input | 512-token random |
| Output tokens | Model output | 128 tokens |
| Prefill | Process input in parallel | Compute-heavy |
| Decode | Generate output 1-at-a-time | Bandwidth-heavy |
| Concurrent seqs | In-flight requests | Peak 200 at rate=5 |
| Queue depth (`num_requests_waiting`) | Waiting requests | Spiked at rate=5 |
| Preemption | Evict request to make room | Killed rate=5 |
| `num_requests_running` | On GPU right now | Grafana Num Running |

## World 4: TIME (how FAST?)

### Latency timeline
```
send ─┬─→ [queue] ─┬─→ [prefill] ─┬─→ [decode]×N ─┬─→ done
      │            │              │                │
   queued      starts prefill   1st token        last token
      │────────────── TTFT ───────│                │
                                  │─── TPOT ───────│
      │──────────────── E2E latency ───────────────│
```

| Term | Meaning | Your lab (rate=1) |
|---|---|---|
| Queue time | Waiting before scheduling | ~0 ms |
| **TTFT** | Send → first token | 479 ms |
| **TPOT** | Steady inter-token delay | 52 ms = 19 tok/s |
| ITL | Per-token version of TPOT | 32 ms median |
| E2E | Total time | ~7 s |

### Throughput
| Term | Meaning | Your lab (rate=10) |
|---|---|---|
| Request throughput | Completed req/s | 4.34 rps |
| Output tok throughput | Generated tok/s all users | 555 tok/s |
| Total tok throughput | Input + output tok/s | 2,776 tok/s |
| Peak output tok/s | Best 1-sec window | 800 tok/s |

## Causal Chain (unifies everything)
```
HARDWARE constrains → MEMORY constrains → REQUESTS determine → TIME
(VRAM, bandwidth)    (KV pool size)     (concurrency)      (TTFT, TPOT)
```

Debug flow: symptom in World 4 → trace back through 3, 2, 1.

## Rate=5 diagnosis using the model
1. World 4: TTFT 22 s = something stuck
2. World 3: peak conc 200, queue 60 s = server overloaded
3. World 2: 200 × 640 tok = 128K ≈ KV budget of 123K = SATURATED
4. World 1: T4 too small for this concurrency
5. Fix: `--max-concurrency` in World 3 protects World 2

## The 8 daily terms
| # | Term | Signal | Source |
|---|---|---|---|
| 1 | VRAM | hardware size | nvidia-smi |
| 2 | KV cache size tokens | memory budget | cache_config_info |
| 3 | Prefill | compute-heavy input | Grafana prefill panel |
| 4 | Decode | bandwidth-heavy output | Grafana decode panel |
| 5 | TTFT | first-token latency | bench / metrics |
| 6 | TPOT | streaming latency | bench / metrics |
| 7 | Throughput (tok/s) | server capacity | rate(generation_tokens_total) |
| 8 | Queue depth | over/under load | num_requests_waiting |

---

# Self-Test 1 Answers

## Q1: TTFT high, TPOT normal — which bottleneck?

**Answer: World 3 (queue/prefill) — NOT bandwidth.**

- Bandwidth affects **decode → TPOT**. If TPOT is fine, bandwidth is fine.
- TTFT = queue time + prefill time. So the cause is either queue backup (overload) OR long prompt / slow prefill (compute).

| Symptom | Bottleneck |
|---|---|
| TTFT bad, TPOT fine | Queue OR prefill compute |
| TTFT fine, TPOT bad | Decode bandwidth / concurrency batching |
| Both bad | Full saturation |

## Q2: KV usage 30%, throughput bad — where's the bottleneck?

**Answer: NOT memory. Look elsewhere.**

Candidates when KV is underused:
- Client not sending enough load (check `num_requests_running`)
- `--max-num-seqs` too low (cap before KV fills)
- Network / client bottleneck
- One very long generation (bandwidth-bound but single stream)
- CPU-side tokenizer bottleneck

Preemption happens near 100% KV, NOT 30%.

## Q3: Double model size — which metric degrades FIRST?

**Answer: TPOT — immediately, at zero load.**

- 2× params → 2× weight bytes read per decode step → decode is bandwidth-bound → TPOT ~doubles.
- KV pool shrinks (weights ate the VRAM), but that only bites once concurrency saturates it.

Correct degradation order under increasing load:
1. TPOT (instantly, per-user)
2. KV pool ceiling (fewer concurrent users fit)
3. Queue depth (as load exceeds new ceiling)
4. TTFT (queue time rises)

## Q4: running=200, waiting=50, kv_usage=95% — what happens next?

**Answer: preemption cascade.**

1. New request → no free blocks
2. vLLM preempts an in-flight request → KV freed (swapped to CPU RAM or fully evicted)
3. Preempted req back to queue → must **re-prefill** on return
4. Waiting grows further
5. TTFT explodes; TPOT worsens
6. Effective throughput often **drops** (wasted work on re-prefills)

Same cascade that killed rate=5.

## The one table that summarizes everything
| Metric bad | Bottleneck world | Common cause |
|---|---|---|
| TTFT | 3 or 1 (compute) | Queue overflow, long prompt |
| TPOT | 1 (bandwidth) or 3 (concurrency) | Big model, many concurrent |
| Queue depth | 3 | Traffic > capacity |
| KV usage % | 2 | Overprovisioned concurrency |

---

# Self-Test 2 Answers

## Q1: TTFT p99 spikes, median fine
Interpretation: 1% of users have terrible experience; most are fine.

Causes (ranked):
1. **Bursty (Poisson) traffic** — occasional pileup → queue variance
2. Preemption cascades
3. Rare long-prompt requests in mixed workload
4. Cold prefix cache misses on 1% of prompts

Rule: **median = systemic health; P99 = tail/variance.**

## Q2: 14B bf16 vs 14B int4 for chat → int4
Reasoning:
- int4 weights = 7 GB vs 28 GB → 21 GB more KV pool → many more concurrent users
- Decode is bandwidth-bound → int4 reads 4× fewer bytes/step → **TPOT ~4× better**
- Quality stated acceptable → no downside

## Q3: --gpu-memory-utilization 0.9 → 0.95 risk
- Only ~5% more KV budget
- Overhead estimation can be off → **CUDA OOM crashes vLLM mid-request**
- No graceful degradation
- Rule: this is a **safety knob, not a throughput knob**. Prefer fp8 KV for 2× win.

## Q4: prefix_cache_hit_rate = 0.05 for chat with shared system prompts
NOT `--max-num-seqs` (that's concurrency, not cache).

Real causes (ranked):
1. `--enable-prefix-caching = False`
2. KV pool too small → blocks churned before reuse
3. **Hidden per-request variance in prefix** (timestamps, user IDs) — most common IRL
4. JSON key-order differences → different tokens
5. Tokenizer / model swap

Rule: **prefix cache misses despite "same" prompts = hidden variance. Check actual tokens, not semantic content.**

## Meta-lesson
Ask **"what mechanism actually causes this metric?"** before reaching for a config knob. Mechanism → fix.

---

# Step 4: Key vLLM Configs

## Group 1: Essentials
| Flag | Value | When |
|---|---|---|
| `--model` | HF ID / path | always |
| `--served-model-name` | short name | client-facing name |
| `--host / --port` | 0.0.0.0 / 8000 | network binding |
| `--dtype` | `auto` / `half` / `bfloat16` | force fp16 on T4; bf16 on A100+ |

## Group 2: Memory & KV

### `--gpu-memory-utilization` (default 0.9)
Safety knob, not throughput. 0.85 conservative, 0.9 default, 0.95 risky (OOM crash risk).

### `--max-model-len`
Per-request token ceiling. Sets `kv_cache_max_concurrency`. Set to what workload actually needs; smaller = more concurrent users.

### `--kv-cache-dtype`
| Value | Effect | Needs |
|---|---|---|
| auto | matches model | — |
| **fp8** | **2× KV pool** | Ampere+ |
| int8 | 2× KV pool, small quality loss | any GPU |

**Highest single-flag impact on concurrency.**

### `--block-size` (default 16)
Almost never touch.

### `--swap-space` (default 4 GB)
CPU RAM for preempted KV. Prefer capping `--max-num-seqs` to avoid preemption entirely.

## Group 3: Concurrency & Batching

### `--max-num-seqs` (default 256) — **THE main lever**
Max concurrent sequences.
| Value | Effect |
|---|---|
| 16–32 | Great TPOT/user, low ceiling |
| 64–128 | Balance |
| 256+ | Max throughput, TPOT degrades |

Fix for rate=5 disaster: `--max-num-seqs 30`.

### `--max-num-batched-tokens`
Total tokens per forward pass. Usually leave auto.

### `--enable-chunked-prefill` (on by default)
Chops long prefills. Fixes TTFT for long prompts. Always on.

### `--enable-prefix-caching` (on by default)
Prefix caching. Always on.

### `--disable-log-requests`
Quieter under high rps.

## Group 4: Quantization
Usually pre-quantized model. Auto-detected. Manual override:
`--quantization awq | gptq | fp8 | bitsandbytes`

## Group 5: Multi-GPU
- `--tensor-parallel-size N` — split each layer across N GPUs. Needs NVLink.
- `--pipeline-parallel-size N` — stages across GPUs. Cross-node.
- `--distributed-executor-backend mp | ray` — ray for multi-node.

## Group 6: Speculative decoding (preview, covered in Step 5)
- `--speculative-model <path>` — draft model
- `--num-speculative-tokens N`

## Group 7: Server / ops
- `--api-key` — bearer token auth. Set for any real deployment.
- `--trust-remote-code` — for models with custom code
- `--seed` — deterministic output (test only)
- `--chat-template` — override built-in template

## Group 8: Advanced
- `--scheduling-policy fcfs | priority`
- `--preemption-mode recompute | swap`
- `--long-prefill-token-threshold`

## Your annotated lab command
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 --port 9090 \
  --dtype half \                    # T4 fp16
  --served-model-name qwen \
  --gpu-memory-utilization 0.9 \
  --max-model-len 15000             # → concurrency 8.24
```

Suggested adds:
```bash
  --max-num-seqs 30                 # cap → prevents rate=5 blowup
  --api-key sk-lab-secret           # basic auth
```

## Config decision tree

```
KV usage near 95%?
├── Yes → fp8 KV / smaller max-model-len / lower max-num-seqs / smaller quant
└── No → TPOT bad?
         ├── Yes → decode bandwidth-bound: lower max-num-seqs OR speculative decoding OR upgrade GPU
         └── No → TTFT bad?
                  ├── + long prompts → enable-chunked-prefill (probably already on)
                  └── + high load    → max-num-seqs is capping, raise it OR scale replicas
```

## Tuning workflow
1. Start with defaults + firm `--max-num-seqs`
2. Bench at 1, 2, 5, 10 rps
3. Watch `kv_cache_usage_perc` in Grafana
4. Watch p95 TTFT/TPOT against SLO
5. Change ONE flag at a time

## The 10 flags to remember
| Flag | Default | Touch when |
|---|---|---|
| `--model` | required | always |
| `--served-model-name` | HF path | cleaner API |
| `--dtype` | auto | force fp16 on T4 |
| `--gpu-memory-utilization` | 0.9 | rarely |
| `--max-model-len` | model max | set to workload need |
| `--kv-cache-dtype` | auto | fp8 for 2× KV (H100+) |
| `--max-num-seqs` | 256 | **main lat/throughput lever** |
| `--tensor-parallel-size` | 1 | model bigger than 1 GPU |
| `--api-key` | none | any real deployment |
| `--enable-chunked-prefill` | on | leave on |

## Resources
- Engine args: https://docs.vllm.ai/en/latest/serving/engine_args.html
- Perf tuning: https://docs.vllm.ai/en/latest/serving/performance.html

---

# Step 5: Optimization Techniques

## Landscape
```
                TTFT / prefill        TPOT / decode
Model compute   Chunked prefill       Speculative decoding
Memory          Prefix caching        fp8 KV / Quantization
Scheduling      Priority queues       Continuous batching
```

## 5.1 Speculative Decoding (draft models)

Small "draft" model guesses N tokens quickly. Big "target" model verifies them in ONE parallel pass.

```
Draft (Qwen-0.5B): guesses "mat and looked around"     ~5 ms
Target (Qwen-7B): verifies all 4 in 1 pass              ~12 ms
  Accepts first 3, produces its own for the mismatch
Result: 4 tokens in ~17 ms instead of 4 × 40 ms
```

Mathematically identical output to running target alone.

### Config
```bash
--speculative-model Qwen/Qwen2.5-0.5B-Instruct
--num-speculative-tokens 5
```

### Speedup expectations
| Workload | Speedup |
|---|---|
| Code | 2.5–3× |
| JSON | 2–2.5× |
| Chat | 1.5–2× |
| Creative | 1.2–1.5× |

### Variants
- **Vanilla** — separate draft model
- **N-gram** (`[ngram]`) — no draft model; matches prompt repetitions. Great for RAG.
- **EAGLE / Medusa / MLPSpeculator** — draft heads attached to target model.

Example without draft model:
```bash
--speculative-model="[ngram]" --ngram-prompt-lookup-max 4
```

### When it wins
- Draft 10–20× smaller than target
- Predictable outputs (code, structured, similar topics)
- Acceptance rate > 60%

### When it hurts
- Creative writing, high-diversity outputs
- Cost: extra VRAM for draft + KV, CPU overhead

## 5.2 Quantization strategies

### Types
| Method | Bits | Loss | Speed | Notes |
|---|---|---|---|---|
| bf16/fp16 | 16 | baseline | baseline | reference |
| fp8 (w+act) | 8 | ~0% | fast | H100+ |
| fp8 (weights) | 8 | ~0% | fast | Ampere+ |
| int8 (SmoothQuant) | 8 | tiny | fast | any GPU |
| **AWQ int4** | 4 | small | very fast | Ampere+ |
| GPTQ int4 | 4 | small | very fast | Ampere+ |
| GGUF | 3–8 | varies | CPU/GPU | llama.cpp/ollama |

### Model + KV combinations
| Model | KV | Effect |
|---|---|---|
| bf16 | bf16 | baseline |
| **bf16 | fp8** | **same quality, 2× concurrent** |
| fp8 | fp8 | fastest on H100 |
| AWQ int4 | bf16 | small weights, std KV |
| AWQ int4 | fp8 | max concurrency |

## 5.3 Chunked Prefill
Splits long prefills into chunks (~512 tok) so decodes don't freeze.
`--enable-chunked-prefill` (default on). Tune `--max-num-batched-tokens` (2048–4096 for RAG).

## 5.4 Prefix Caching — maximize hit rate
- System prompt FIRST
- No timestamps/user IDs in system prompt
- Stable few-shot examples & RAG chunk order
- Advanced: CPU/disk prefix cache spillover

## 5.5 Continuous Batching Tuning
```bash
--scheduling-policy priority        # allow per-request priority
--preemption-mode recompute         # default; usually best
--preemption-mode swap              # only if huge swap-space AND expensive prefill
```

## 5.6 Structured / Guided Decoding
Force valid JSON / regex / grammar. Zero broken outputs.

Request body:
```json
{"response_format": {"type": "json_object"}}
```
Or schema:
```json
{"response_format": {"type": "json_schema", "json_schema": {"schema": {...}}}}
```

Uses `outlines` / `xgrammar` under hood. Small overhead, eliminates retries.

## 5.7 LoRA hot-swap
```bash
--enable-lora \
--lora-modules '{"name":"cs","path":"lora/cs"}' \
               '{"name":"legal","path":"lora/legal"}'
```
Request: `"model": "cs"`. Share base weights across many fine-tunes.

## Symptom → optimization lookup
| Symptom | First try | Then |
|---|---|---|
| KV saturating | fp8 KV | lower max-num-seqs |
| TPOT slow (decode) | Spec decoding | bigger GPU |
| TTFT slow, long prompts | Chunked prefill | prefix caching |
| TTFT slow, short prompts | Prefix caching | check max-num-seqs |
| Broken JSON | Guided decoding | — |
| Many fine-tunes | LoRA hot-swap | — |
| Model too big | Quantization | tensor parallelism |

## Stacked production example (H100)
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 --max-num-seqs 128 \
  --enable-chunked-prefill --enable-prefix-caching \
  --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
  --num-speculative-tokens 5 \
  --api-key sk-prod-...
```
Effect: ~3× throughput, ~2× lower TPOT, same quality.

## Anti-patterns
- Don't raise gpu-memory-utilization > 0.92
- Don't set num-speculative-tokens > 8
- Don't quantize to int2/int3 for chat
- Don't disable prefix caching
- Don't set max-num-seqs to 1000+

## Resources
- 📄 Spec decoding paper: https://arxiv.org/abs/2302.01318
- 📝 vLLM spec decoding docs: https://docs.vllm.ai/en/latest/serving/speculative_decoding.html
- 📝 Guided gen (outlines): https://github.com/dottxt-ai/outlines

---

# Step 6: NVIDIA "Mastering LLM Inference" — mapped to vLLM

Reference: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

The NVIDIA post is a general survey of inference optimization. Most of what it describes is **already implemented inside vLLM** — you just enable it via a flag. Below is a 1:1 map.

## 6.1 Technique map: NVIDIA blog → vLLM

| NVIDIA technique | What it does | Already in vLLM? | How you use it |
|---|---|---|---|
| **KV caching** | Store K,V of past tokens so decode is O(1) per token, not O(n) | ✅ Core of vLLM | Automatic |
| **PagedAttention** | Non-contiguous KV in fixed blocks (OS-page style) | ✅ vLLM invented it | Automatic. `--block-size` |
| **Continuous / in-flight batching** | Replace finished seqs mid-batch instead of waiting | ✅ Core scheduler | Automatic |
| **Chunked prefill** | Split long prefill so decodes aren't starved | ✅ | `--enable-chunked-prefill` (on) |
| **Prefix / prompt caching** | Reuse KV of shared prefixes across requests | ✅ | `--enable-prefix-caching` (on) |
| **Multi-Query / Grouped-Query Attention (MQA/GQA)** | Fewer KV heads → smaller KV cache | ⚠️ **Model-side.** vLLM just executes what the model defines | Pick a GQA/MQA model (Llama-3, Qwen2.5, Mistral) |
| **FlashAttention / FlashAttention-2/3** | Fused, IO-aware attention kernel | ✅ | Automatic (vLLM auto-picks best backend: FA2/FA3/xFormers). `VLLM_ATTENTION_BACKEND=FLASH_ATTN` to force |
| **KV cache quantization (fp8 / int8)** | Halve KV memory → 2× concurrency | ✅ | `--kv-cache-dtype fp8` |
| **Weight quantization (AWQ/GPTQ/fp8/int8/int4)** | Smaller weights, faster matmul | ✅ | `--quantization awq / gptq / fp8` |
| **Speculative decoding** | Draft model + verify | ✅ | `--speculative-model ...` |
| **Tensor Parallelism** | Split each layer across GPUs | ✅ | `--tensor-parallel-size N` |
| **Pipeline Parallelism** | Split layers into stages across GPUs/nodes | ✅ | `--pipeline-parallel-size N` |
| **Sequence Parallelism** | Split long-sequence activations across GPUs | ⚠️ Partial (training-focused; some paths in vLLM v1) | Rarely a knob you set for inference |
| **Expert Parallelism (MoE)** | Route different experts to different GPUs | ✅ (for MoE models like Mixtral, DeepSeek) | `--enable-expert-parallel` (model-dependent) |
| **In-flight LoRA / multi-LoRA** | Serve many fine-tunes on one base | ✅ | `--enable-lora --lora-modules ...` |
| **Guided / structured decoding** | Constrained JSON/regex/grammar | ✅ | `response_format` in request |
| **Distillation / pruning / sparsity** | Smaller/thinner model | ❌ Offline step | Do before serving; then load in vLLM |

**Rule of thumb:** if NVIDIA lists it as a *serving-time* technique, vLLM has it. If it's a *model-training-time* technique (distillation, pruning, changing attention type), you do it upstream and vLLM just serves the result.

## 6.2 "Optimizing attention is a model thing, right?"

Mostly **yes** — with one important nuance.

Split it into two layers:

### A. Attention *algorithm* (architectural) — MODEL-side
This is baked into the model weights and cannot be changed at serve time:
- **MHA** (Multi-Head Attention) — original, every head has its own K,V. Big KV cache.
- **MQA** (Multi-Query) — one shared K,V for all heads. Tiny KV cache. (PaLM, Falcon-7B)
- **GQA** (Grouped-Query) — heads share K,V in groups. Middle ground. (Llama-2-70B, Llama-3, Qwen2.5, Mistral)
- **MLA** (Multi-head Latent Attention) — DeepSeek-v2/v3. Compresses KV into a latent.
- **Sliding-window attention** — Mistral. Only attend to last W tokens.

👉 If you want a "better attention" you must **choose a different model.** vLLM will detect the config and size KV accordingly (see Deep Dive 1 formula: `2 × L × H_kv × d_head`).

### B. Attention *kernel* (implementation) — SERVER-side
This is how the same math is executed on the GPU. vLLM controls this:
- **FlashAttention v2 / v3** — IO-aware, fused, avoids materializing the N×N attention matrix.
- **PagedAttention kernel** — reads K,V from paged blocks instead of a contiguous tensor.
- **xFormers / Triton / CUTLASS** backends — fallbacks.

vLLM automatically picks the fastest backend for your GPU + model. You rarely touch this, but:
```bash
# force a backend for A/B testing
VLLM_ATTENTION_BACKEND=FLASH_ATTN     # or FLASHINFER, XFORMERS
```

### Summary
| Layer | Who owns it | Example |
|---|---|---|
| Attention *math* (MHA/GQA/MQA/MLA) | Model architect | Pick Llama-3 (GQA) over Llama-2-7B (MHA) |
| Attention *kernel* (Flash, Paged) | vLLM | Automatic |

So your intuition is right: **to change attention, change the model.** vLLM's job is to run whatever attention the model uses as efficiently as possible.

## 6.3 Parallelism types — with inference examples

> **MoE = Mixture-of-Experts.** A model architecture where each layer has many "expert" sub-networks, and a router picks a small subset per token (e.g., 2 of 8). Total parameters are huge but only a fraction activate per token. Examples: Mixtral 8x7B / 8x22B, DeepSeek-V3, Qwen3-MoE.

### vLLM's official strategy (from the [Parallelism & Scaling docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/))

For a **single-model replica**:

| Situation | Strategy |
|---|---|
| Model fits on 1 GPU | No distributed inference. Just run on that GPU. |
| Model too big for 1 GPU, fits on 1 node | **Tensor Parallel** — `--tensor-parallel-size = #GPUs in node` |
| Model too big for 1 node | **TP within node + PP across nodes** — `--tensor-parallel-size = GPUs-per-node`, `--pipeline-parallel-size = #nodes` |

Then, orthogonally, to serve **more traffic**:
- **Data Parallel** (`--data-parallel-size N`) — run N replicas of the whole strategy above
- **Expert Parallel** (`--enable-expert-parallel`) — for MoE models, spread experts across the DP/TP world

### The 5 parallelism types

All 5 apply during **inference/serving** in vLLM.

#### 1. Tensor Parallelism (TP) — split *within* a layer, across GPUs, in the same computation

Example: an MLP weight matrix `A` of shape `[4096, 16384]` is split column-wise into `A1`, `A2` on GPU0/GPU1. Both GPUs compute `X·A1` and `X·A2` for the same forward pass simultaneously, then sync (**all-reduce**) to combine.

```
Same forward pass, sliced across GPUs:
  Q/K/V proj → [GPU0 slice | GPU1 slice] → all-reduce
  MLP        → [GPU0 slice | GPU1 slice] → all-reduce
  (every layer, every step)
```

- **Flag:** `--tensor-parallel-size 2`
- **Needs:** high-bandwidth interconnect (NVLink). All-reduce every layer.
- **Best for:** within a single node.

#### 2. Pipeline Parallelism (PP) — split *across* layers, like an assembly line

Example: 32-layer model → GPU0 holds layers 1–16, GPU1 holds layers 17–32. A request's activations flow GPU0 → GPU1 sequentially.

```
GPU0: layers 1–16  ──activations──▶  GPU1: layers 17–32
```

- **Flag:** `--pipeline-parallel-size 2`
- **Needs:** only point-to-point transfer of activations at stage boundaries — far less inter-GPU chatter than TP. Good for cross-node scaling.
- **Trade-off:** "pipeline bubbles" (GPU0 idle while GPU1 finishes). vLLM's continuous batching keeps the pipe full and hides most of this.

#### 3. Sequence Parallelism (SP) — split ops along the token/sequence dimension

Ops like LayerNorm/dropout/residuals can't be head/weight-sharded the way TP shards attention & MLP, so they get sharded along the **sequence axis** instead. This reduces redundant activation memory that would otherwise be duplicated across TP ranks.

- Usually rides **along with TP automatically** in vLLM — you rarely set it separately.
- Not a manual `--sequence-parallel-size` knob you tune at serve time.

#### 4. Data Parallelism (DP) — full, independent model replicas

Each GPU (or each TP group) holds a **complete copy** of the model and handles different requests. No cross-GPU communication per-token — pure throughput scaling.

Example: 4 GPUs, `--data-parallel-size 4` → 4 complete copies, each serving its own request stream.

- **Flag:** `--data-parallel-size 4`
- **NVIDIA blog note:** The 2023 blog calls DP "less relevant during inference" — that was true then, but current vLLM leans on DP a lot, especially when combined with **EP** for MoE models (see below).
- **Best for:** models that fit on 1 GPU (or 1 TP group) → maximum throughput. Also the backbone for MoE serving.

#### 5. Expert Parallelism (EP) — MoE-specific

Different GPUs each host a **subset of experts**; a router sends each token to the GPU whose expert it needs. Massively cuts per-GPU weight memory for MoE models.

```
GPU0: experts 0-1   GPU1: experts 2-3   GPU2: experts 4-5   GPU3: experts 6-7
Token X → expert 3 → routed to GPU1
Token Y → expert 6 → routed to GPU2
```

- **Flag:** `--enable-expert-parallel`
- **Cost:** all-to-all communication during routing; can imbalance across GPUs. vLLM has an **EPLB (Expert Parallel Load Balancer)** to mitigate.
- **Big deal for:** DeepSeek-V3, Mixtral, Qwen3-MoE.

### How to combine them (typical large-MoE, multi-node)

- **TP within a node** (fast NVLink)
- **PP or DP across nodes** (less bandwidth-hungry)
- **EP layered on top** for the MoE layers specifically

Dense (non-MoE) models on one node usually just need **TP**. Small models that fit on 1 GPU → **DP replicas** for throughput.

### Concrete vLLM commands

```bash
# Small model, 1 GPU
vllm serve Qwen/Qwen2.5-7B-Instruct

# Small model, 8 GPUs for max throughput (data parallel)
vllm serve Qwen/Qwen2.5-7B-Instruct --data-parallel-size 8

# Dense large model, single 4-GPU node → TP
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

# Dense huge model across 2 nodes of 8 GPUs each → TP within, PP across
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray

# MoE model with expert parallel + tensor parallel
vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel

# MoE on multi-node: DP for throughput, EP for expert spread
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --enable-expert-parallel
```

### Does all this apply at serving/inference time?

**Yes** — parallelism (TP/PP/SP/DP/EP), PagedAttention, in-flight batching, quantization, speculative decoding, and attention kernels all operate during inference. The one thing in the NVIDIA doc that does **not** is **distillation** — that's a separate training run done beforehand to produce a smaller model; vLLM just serves the resulting checkpoint.

### Prefill vs decode: which parallelism helps which?

One nuance worth internalizing — prefill and decode benefit **differently** from these techniques:

| Phase | Bound by | Benefits most from |
|---|---|---|
| **Prefill** (process prompt) | Compute (parallel matrix-matrix multiply) | **TP** — parallel compute across GPUs |
| **Decode** (generate 1 tok/step) | Memory bandwidth (matrix-vector, one token) | Things that (a) free memory for bigger batches — **TP** (splits weights → more KV room), **quantization**, **fp8 KV** — or (b) cut the number of sequential steps — **speculative decoding** |

That's why the winning stack for a big model is usually: **TP for compute + KV headroom** + **quantization for more headroom** + **speculative decoding to shrink decode step count** + **continuous batching + prefix caching** to keep GPUs busy.

## 6.4 Mental model: where each NVIDIA technique attacks the bottleneck

```
                        PREFILL (compute-bound)      DECODE (memory-bw bound)
Attention algorithm     GQA/MQA/MLA (model)          GQA/MQA/MLA (model)
Attention kernel        FlashAttention               FlashAttention + PagedAttention
KV memory               —                            PagedAttention, fp8 KV, prefix cache
Batch efficiency        Chunked prefill              Continuous batching
Token throughput        —                            Speculative decoding
Model size              Quantization (AWQ/fp8)       Quantization
Fit big model           TP / PP / EP                 TP / PP / EP
Horizontal scale        Data parallel (replicas)     Data parallel (replicas)
```

## 6.5 TL;DR answers to your questions

1. **Does vLLM already have the NVIDIA blog's techniques?**
   Yes — every *serving-time* technique in that blog is in vLLM. The blog is a general survey; vLLM is one concrete implementation of it. Techniques that are *training-time* (distillation, pruning, changing attention type) happen before serving.

2. **Is optimizing attention a model thing?**
   The **algorithm** (MHA/GQA/MQA/MLA) is model-side — pick a better model. The **kernel** (FlashAttention, PagedAttention) is vLLM-side and mostly automatic. So the interesting server-side attention optimizations (PagedAttention + FlashAttention) are already on.

3. **Parallelism during inference:**
   - **DP (replicas)** — when model fits on 1 GPU (or 1 TP group). Best throughput scaling. Also backbone for MoE serving.
   - **TP** — split each layer within a node over NVLink. Also helps decode by freeing weight memory for more KV.
   - **PP** — split layers into stages, use across nodes (low interconnect needs).
   - **SP** — rides along with TP automatically; not a manual knob.
   - **EP** — MoE models only (`--enable-expert-parallel`).
   All apply at inference. Training-only kinds (ZeRO/FSDP) don't matter for serving. Distillation happens *before* serving.

## Resources
- 📄 NVIDIA blog: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- 📝 vLLM Parallelism & Scaling: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- 📝 vLLM Data Parallel deployment: https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/
- 📝 vLLM Expert Parallel deployment: https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/
- 📄 FlashAttention-2: https://arxiv.org/abs/2307.08691
- 📄 GQA paper: https://arxiv.org/abs/2305.13245
- 📄 MLA (DeepSeek-V2): https://arxiv.org/abs/2405.04434

---
