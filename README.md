# QICCR-LLM

A from-scratch transformer language model in pure Python. No PyTorch, no NumPy, no frameworks — just `array.array('f')` and explicit loops. Educational, inspectable, and mathematically correct.

## Overview

QICCR is a complete decoder-only transformer with manual forward/backward passes, AdamW optimizer, KV cache, rotary embeddings, BPE tokenizer, and beam search. It doesn't compete with production models in speed, but every operation is transparent and step-debuggable.

**At a glance:**

| | |
|---|---|
| Layers | 2 |
| Attention heads | 4 |
| Model dimension | 128 |
| FFN dimension | 256 |
| Vocabulary | 8192 (BPE) |
| Train context | 64 tokens |
| KV cache capacity | 128 tokens |
| Parameters | ~1.3M |

## Requirements

Python 3.8+. Nothing else.

```bash
python --version  # → 3.8+
```

## Quick start

### Train

Create a `treino.txt` file with your training text, then:

```bash
python qiccr_llm.py --train        # 5 epochs (default)
python qiccr_llm.py --train 10     # 10 epochs
```

The model saves checkpoints automatically:
- `qiccr_v61_best_*` — best loss so far
- `qiccr_v61_latest_*` — most recent epoch

### Chat

```bash
python qiccr_llm.py
```

The model loads the best available checkpoint and opens an interactive session:

```
🚀 QICCR-LLM v6.1 | Comandos: 'sair' | 'reset' | 'beam'

🧑 Você: What is machine learning?
🐶 Qiccr: Machine learning is a field of artificial intelligence...

🧑 Você: beam
✅ Beam search: ON

🧑 Você: reset
✅ Cache limpo.
```

**Chat commands:**

| Command | Effect |
|---|---|
| `sair` / `exit` / `quit` | Exit |
| `reset` | Clear KV cache, restart context |
| `beam` | Toggle beam search on/off |

## Configuration

All hyperparameters live in the `Config` class. Edit directly before training:

```python
class Config:
    # Model architecture
    VOCAB_SIZE       = 8192
    D_MODEL          = 128
    N_HEADS          = 4
    N_LAYERS         = 2
    D_FF             = 256
    MAX_SEQ          = 64
    KV_MAX_SEQ       = 128

    # Training
    MAX_TRAIN_STEPS  = 8000
    TRAIN_WINDOW_MIN = 16
    TRAIN_WINDOW_MAX = 48
    BATCH_SIZE       = 4
    WEIGHT_DECAY     = 0.01
    GRAD_CLIP        = 1.0
    FFN_DROPOUT      = 0.1
    LABEL_SMOOTHING  = 0.1
    NOAM_WARMUP      = 4000

    # AdamW
    ADAM_BETA1       = 0.9
    ADAM_BETA2       = 0.999
    ADAM_EPS         = 1e-8

    # Generation
    TEMPERATURE        = 0.75
    TOP_K              = 50
    TOP_P              = 0.90
    REPETITION_PENALTY = 1.1
    BEAM_WIDTH         = 3
```

**Tuning tips:**
- Short texts (~10k chars): reduce `MAX_TRAIN_STEPS` to 2000–3000
- More capacity: increase `D_MODEL` to 256, `N_LAYERS` to 4 (training ~4× slower)
- More creative: `TEMPERATURE = 0.9`, `TOP_P = 0.95`
- More deterministic: `TEMPERATURE = 0.5`, `TOP_K = 20`

## Programmatic use

```python
from qiccr_llm import QICCRLLM, Config, train_model
import random

random.seed(42)
model = QICCRLLM()

# Train
train_model(model, file="my_text.txt", epochs=3)

# Generate
response = model.generate(
    "Once upon a time",
    max_new=100,
    temp=0.8,
    top_k=50,
    top_p=0.92
)
print(response)

# Save / load
model.save("my_model")
model.load("my_model")
```

### Tokenizer

```python
# Train BPE on custom texts
texts = ["first corpus...", "second corpus..."]
model.tokenizer.train(texts, num_merges=3000)

# Encode / decode
tokens = model.tokenizer.encode("hello world")  # → tuple of ints
text   = model.tokenizer.decode(tokens)         # → "hello world"
```

### Multi-turn context

```python
# Full control over KV cache
logits = model.prefill(prompt_tokens)  # populate cache
for _ in range(max_new):
    logits = model.decode_step(token)  # one token at a time
    # ... sample next token

model.kv.clear()  # reset context
```

## How it works

### Data flow

```
Text → BPE Tokenizer → Token IDs
Token IDs → Embedding lookup → D_MODEL vectors
Vectors → [TransformerBlock × N_LAYERS] → Representations
Final representation → LayerNorm → Logit projection
Logits → Softmax → Token probabilities → Sampled token
```

### 1. BPE tokenizer

Starts from all 256 bytes as base vocabulary. Learns frequent adjacent pairs from training text using a min-heap for O(n log n) merge application. Special tokens: `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3).

### 2. Flat weight storage

All parameters live in a single `array.array('f')`. A `WeightAllocator` assigns sequential offsets at initialization — each layer knows where its weights begin. This mimics a tensor store without NumPy.

```
[tok_embed | q0 k0 v0 o0 up0 down0 ln10 ln20 | q1 k1 ... | ln_final | logit_scale]
 ←── VOCAB*D ──→ ←────────── LAYER 0 ────────────→ ←─ LAYER 1 ─→ ←2D→ ←1→
```

### 3. Multi-head attention with RoPE

Each token projects Q, K, V. Queries and keys receive **rotary positional embeddings** — position is encoded by rotating pairs of dimensions:

```
q_rot[2i]   = q[2i]·cos(θ) - q[2i+1]·sin(θ)
q_rot[2i+1] = q[2i]·sin(θ) + q[2i+1]·cos(θ)
θ = pos / 10000^(2i/d_head)
```

This gives the model **relative** position awareness, not absolute.

### 4. KV cache

During inference, computed keys and values are stored and reused. Each entry carries a timestamp (original position). Base keys are stored **unrotated** — rotation is applied on retrieval using the correct timestamp, regardless of storage order.

### 5. Manual backward pass

Gradients flow explicitly through every operation in reverse:

```
∂L/∂logits → ∂L/∂final → ∂L/∂LayerNorm → ∂L/∂blocks → ∂L/∂embeddings
```

Inside each transformer block:
1. FFN down → GELU derivative → FFN up
2. LayerNorm 2 + residual
3. Output projection
4. Attention: ∂L/∂probs → softmax backward → ∂L/∂scores → ∂L/∂Q,∂K,∂V
5. Inverse RoPE on Q and K gradients
6. Q, K, V projections
7. LayerNorm 1 + residual

### 6. AdamW optimizer

Full implementation with:
- First/second moment estimates (m, v)
- Bias correction: `m̂ = m / (1 - β₁ᵗ)`
- Decoupled weight decay (AdamW, not Adam)
- Global L2 gradient clipping

### 7. Text generation

Two strategies:

**Sampling** (default): prefill prompt → decode token-by-token with top-K, top-P, and repetition penalty.

**Beam search** (`beam` command): maintains B parallel hypotheses, expands each with top-B candidates per step, prunes to best B by cumulative log-probability. Uses cache cloning for efficient branching.

## Checkpoint format

Three files per checkpoint:

| File | Content |
|---|---|
| `*_fp32.bin` | Model weights (raw float32) |
| `*_optim.json.gz` | Adam state (m, v, step count) |
| `*_meta.json.gz` | BPE vocabulary, merges, step count |

```python
model = QICCRLLM()
model.load("qiccr_v61_best")   # no extension needed
```

## Performance expectations

This is **pure Python** — no vectorization, no GPU. On a modern CPU:

| Operation | Approximate time |
|---|---|
| 1 training step (window ~32) | 5–15s |
| Generate 1 token | 2–5s |
| 500-step epoch | ~1–2 hours |

The bottleneck is the lm_head vocabulary projection: `VOCAB_SIZE × D_MODEL` operations in Python per forward pass (~1M operations). Convergence happens — loss decreases — but iteration speed limits extensive experimentation.

## Architecture

```
QICCR-LLM v6.1
┌─────────────────────────────────┐
│      Token Embedding            │
│      (Vocab → 128)              │
└────────────┬────────────────────┘
             │
   ┌─────────▼─────────┐
   │ Transformer Block  │  ×2
   │                    │
   │  LayerNorm →       │
   │  Multi-Head Attn   │
   │  (4 heads + RoPE)  │
   │  + Residual →      │
   │  LayerNorm →       │
   │  FFN (GELU, 256)   │
   │  + Residual        │
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │   Final LayerNorm  │
   │   Logit Projection │
   │   Softmax / Beam   │
   └────────────────────┘
```

## Limitations

- **Performance**: Python-loop attention and vocabulary projection are the main bottlenecks. No GPU path exists in this implementation.
- **KV cache at training**: Only used during inference. Training always computes full attention over the window.
- **Model capacity**: ~1.3M parameters at default settings — sufficient for local pattern memorization in short texts, limited for generalization.

## Known fixes in v6.1

- **KV cache clone**: Added missing `d_head` parameter to `KVCache.__init__` inside `clone()`, preventing silent shape corruption during beam search.
- **Backward pass indices**: Causal mask in backward now correctly uses `ts <= pos_local[i]` (relative positions), matching the forward pass constraint.
- **Removed unused defaultdict**: Cleaned up an import that was never used.

## License

Free for educational and research use.
