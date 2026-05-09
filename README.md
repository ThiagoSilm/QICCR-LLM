# QICCR-LLM

**A minimal transformer language model built entirely from scratch in pure Python — no PyTorch, no TensorFlow, no CUDA.**

QICCR-LLM is a research-oriented LLM implementation designed to prove that meaningful language modeling doesn't require industrial-scale infrastructure. Every component — backpropagation, attention, BPE tokenization, KV cache, RoPE embeddings, AdamW — is implemented by hand using only Python's standard library.

---

## Philosophy

Modern LLMs are trained on warehouse-scale hardware with billions of parameters and petabytes of data. QICCR takes the opposite bet: that a small model trained efficiently, with full control over every gradient, can serve as a serious research platform and eventually a practical alternative for constrained environments.

The human brain doesn't memorize every book ever written. It learns to reason, then reaches for information when needed. QICCR is built around that same intuition.

---

## Architecture

- **Staged training** — model starts small (1 layer, d=64) and grows to final size (2 layers, d=128) via weight interpolation, preserving learned representations across expansion
- **Rotary Positional Embeddings (RoPE)** — implemented from scratch with cached sin/cos tables
- **KV Cache** — per-layer, per-head cache with timestamp-based eviction for efficient autoregressive generation
- **BPE Tokenizer** — heap-accelerated merge algorithm trained on your own corpus
- **AdamW with Noam scheduling** — gradient clipping, weight decay, label smoothing
- **Beam search + sampling** — top-k, top-p, temperature, repetition penalty
- **No external ML dependencies** — `math`, `array`, `json`, `gzip`, `heapq`

---

## Quickstart

```bash
# Train on your own text corpus
python qiccr.py --train

# Chat with a trained model
python qiccr.py
```

Place your training data in `treino.txt`. The tokenizer trains on your corpus before the first epoch begins.

---

## Training Stages

| Stage | Layers | d_model | d_ff | Heads | Steps |
|-------|--------|---------|------|-------|-------|
| 1 — Base | 1 | 64 | 128 | 2 | 2000 |
| 2 — Expansion | 2 | 128 | 256 | 4 | 4000 |
| 3 — Fine-tuning | 2 | 128 | 256 | 4 | 2000 |

All hyperparameters are centralized in the `Config` class and require no external config files.

---

## Who This Is For

**Researchers** interested in studying transformer internals without framework abstraction — every forward pass, every gradient, every weight update is readable Python.

**Educators** who want a codebase where students can trace exactly what happens during backpropagation through attention.

**Engineers** working in constrained environments — embedded systems, edge devices, air-gapped infrastructure — where PyTorch is not an option.

**Experimenters** who want to test architectural ideas at minimal cost before scaling them up.

---

## Contributing

Contributions are welcome. The codebase is intentionally dense — no unnecessary abstraction. If you want to contribute:

- Open an issue describing what you want to change and why
- Keep changes scoped — one idea per PR
- If you're adding a new component, implement it without external ML dependencies
- Optimize for clarity over brevity where the two conflict

Areas of active interest:
- Mixture of Experts routing for sparse activation
- Early exit mechanisms for adaptive compute
- Retrieval-augmented generation via external index
- Quantization to int8 using only Python arrays
- Multi-document context beyond current KV cache limits

---

## License

GNU General Public License v3.0

This software is free. You may use, study, modify, and distribute it under the terms of the GPL v3. Any derivative work must remain open under the same license.

---

## Status

Pre-release. The architecture is stable and the training loop is functional. No pretrained weights are distributed — you bring your own data. This is intentional.

---

*Built without frameworks. Trained without clusters. Runs anywhere Python runs.*
