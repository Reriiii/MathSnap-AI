# MathSnap-AI — Guidelines for Understanding Context & Model Design

## 1. System Overview

MathSnap-AI is a **Handwritten Mathematical Expression Recognition (HMER)** system that converts images of handwritten math into LaTeX. It uses **Mini-CoMER**, a compact encoder-decoder architecture with 6.39M parameters, optimized for single-GPU inference.

**End-to-end flow:**

```
Image → Preprocessing → DenseNet Encoder → Transformer Decoder (with ARM) → LaTeX
```

---

## 2. Context Flow

Context in MathSnap-AI refers to how information is represented and passed between stages.

### 2.1 Input Context

- **Raw input:** A user-uploaded image of handwritten math.
- **Preprocessing** (in `backend/main.py`):
  1. Convert to grayscale
  2. Otsu binarization (automatic thresholding)
  3. Background detection and inversion (check border pixels)
  4. Aspect-ratio-preserving resize to fit within **128×512**
  5. Normalize pixel values to `[0, 1]`
  6. Generate a binary mask (`False` = valid pixel, `True` = padding)
- **Output:** Tensor `[1, 1, H, W]` + mask `[1, H, W]`

### 2.2 Encoder Context (Visual Features)

The **DenseNet encoder** (`models/encoder.py`) transforms the image into a spatial feature map that captures symbol shapes and layout.

| Stage | Operation | Output Shape |
|-------|-----------|--------------|
| Initial conv | 1→48 channels, 7×7, stride 2 | `[b, 48, H/2, W/2]` |
| Dense Block 1 | 16 bottleneck layers, growth=24 | `[b, 432, H/2, W/2]` |
| Transition 1 | 1×1 conv (0.5 reduction) + avg pool | `[b, 216, H/4, W/4]` |
| Dense Block 2 | 16 bottleneck layers | `[b, 600, H/4, W/4]` |
| Transition 2 | 1×1 conv + avg pool | `[b, 300, H/8, W/8]` |
| Dense Block 3 | 16 bottleneck layers | `[b, 684, H/8, W/8]` |
| Projection | 684 → 256 | `[b, h, w, 256]` |
| 2D Pos Encoding | Sinusoidal (height & width) | `[b, h, w, 256]` |
| LayerNorm | Normalize | `[b, h, w, 256]` |

**Key design decisions:**
- Dense connections let deeper layers access raw edge/stroke features from earlier layers.
- 2D positional encoding preserves spatial layout so the decoder knows *where* a symbol is, not just *what* it looks like.
- The mask propagates through pooling layers to track which spatial positions are real vs. padded.

### 2.3 Decoder Context (Sequence Generation)

The **Transformer decoder** (`models/decoder.py`) autoregressively generates LaTeX tokens conditioned on the encoder features.

**Context sources at each decoding step:**

1. **Token embeddings** — the LaTeX tokens generated so far (`[SOS, tok1, tok2, ...]`)
2. **1D positional encoding** — position within the output sequence (max 500)
3. **Self-attention** — causal (masked) attention over prior tokens; gives the decoder language-model context
4. **Cross-attention** — attends to encoder feature map; binds tokens to image regions
5. **ARM coverage** — cumulative attention history that biases the model away from re-attending already-covered regions

**Decoder configuration:**

| Parameter | Value |
|-----------|-------|
| Layers | 3 |
| Attention heads | 8 |
| d_model | 256 |
| FFN inner dim | 1024 |
| Max sequence length | 200 tokens |

### 2.4 ARM Context (Attention Refinement)

The **Attention Refinement Module** (`models/transformer/arm.py`) solves the *coverage problem* — without it, the decoder may repeatedly attend to prominent symbols while ignoring others.

**How ARM works:**

1. **Accumulate** cross-attention weights across all previous time steps → coverage vector
2. **Refine** with a 5×5 convolution + ReLU + 1×1 projection (32 intermediate channels)
3. **Apply masked BatchNorm** (only over valid, non-padded positions)
4. **Subtract** the coverage bias from attention logits before softmax

This encourages the model to explore un-attended image regions at each new step, improving recognition of multi-symbol expressions.

---

## 3. Model Design

### 3.1 Architecture Diagram

```
Input Image [b, 1, H, W]
       │
       ▼
┌──────────────────────┐
│   DenseNet Encoder   │  3 dense blocks, growth_rate=24
│   + 2D Pos Encoding  │  Output: [b, h, w, 256]
│   + LayerNorm        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│         Transformer Decoder (×3 layers)      │
│  ┌─────────────────────────────────────────┐ │
│  │ Self-Attention (causal)                 │ │
│  │     ↓                                   │ │
│  │ Cross-Attention ← ARM coverage bias     │ │
│  │     ↓                                   │ │
│  │ Feed-Forward (256 → 1024 → 256)         │ │
│  └─────────────────────────────────────────┘ │
└──────────┬───────────────────────────────────┘
           │
           ▼
   Output Projection (256 → 114 vocab)
           │
           ▼
   LaTeX Token Sequence
```

### 3.2 Vocabulary & Tokenization

Defined in `data/vocab.py`. 114 total tokens:

| Category | Examples |
|----------|----------|
| Special (4) | `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` |
| Digits | 0–9 |
| Letters | a–z, A–Z, Greek (α, β, γ, θ, π, ...) |
| Operators | `+`, `-`, `=`, `<`, `>`, `×`, `÷`, `≤`, `≥` |
| Functions | `sin`, `cos`, `tan`, `log`, `lim` |
| Structural | `frac`, `sqrt`, `^`, `_`, `{`, `}` |
| Symbols | `∞`, `∈`, `→`, `Σ`, `∫`, `∏` |

- **Encoding:** `”x ^ { 2 }”` → `[SOS, x, ^, {, 2, }, EOS]`
- **Decoding:** Strip special tokens, join with spaces

### 3.3 Decoding Strategies

| Strategy | Method | When to Use |
|----------|--------|-------------|
| **Greedy** | `argmax` at each step | Default for real-time inference (fast) |
| **Beam search** | Track top-k candidates | Higher accuracy, beam_size=10, length penalty configurable |

### 3.4 Positional Encoding Design

Two complementary positional encodings preserve spatial and sequential structure:

- **ImgPosEnc (2D)** — applied to encoder output. Uses sinusoidal functions over normalized (x, y) coordinates. Handles variable image sizes by computing positions from cumulative masks.
- **WordPosEnc (1D)** — applied to decoder embeddings. Standard sinusoidal encoding (temperature=10000) for up to 500 positions.

---

## 4. Configuration Reference

All hyperparameters are defined in `config.py` as dataclasses.

### Data

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `img_height` | 128 | Max image height after resize |
| `img_width` | 512 | Max image width after resize |
| `batch_size` | 64 | Training batch size |
| `max_seq_len` | 200 | Maximum LaTeX token length |
| `scale_range` | 0.7–1.4 | Data augmentation scale factor |

### Model

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `d_model` | 256 | Shared embedding dimension |
| `growth_rate` | 24 | DenseNet channel growth per layer |
| `num_layers` | 16 | Bottleneck layers per dense block |
| `nhead` | 8 | Attention heads |
| `num_decoder_layers` | 3 | Transformer decoder depth |
| `dim_feedforward` | 1024 | FFN inner dimension |
| `dc` | 32 | ARM intermediate channels |
| `beam_size` | 10 | Beam search width |

### Training

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Gradient clipping | 5.0 |
| Mixed precision | FP16 enabled |
| Epochs | 300 (early stopped at 230) |

---

## 5. Key Files

| File | Role |
|------|------|
| `models/model.py` | Top-level CoMER class, greedy/beam decode, `build_model` |
| `models/encoder.py` | DenseNet feature extractor + 2D pos encoding |
| `models/decoder.py` | Transformer decoder + embeddings + output projection |
| `models/pos_enc.py` | `WordPosEnc` (1D) and `ImgPosEnc` (2D) |
| `models/transformer/attention.py` | Multi-head attention with ARM integration |
| `models/transformer/arm.py` | Attention Refinement Module |
| `models/transformer/transformer_decoder.py` | Stacked decoder layers |
| `data/vocab.py` | Vocabulary encode/decode (114 tokens) |
| `config.py` | All hyperparameters as dataclasses |
| `backend/main.py` | FastAPI server, preprocessing, `/predict` endpoint |

---

## 6. Design Principles

1. **Compact by design** — Mini-CoMER achieves 68% parameter reduction (6.39M vs ~20M) by using a narrower DenseNet (growth_rate=24) and fewer decoder layers (3). This enables real-time inference on consumer GPUs.

2. **Coverage-aware decoding** — The ARM prevents the decoder from fixating on dominant symbols. Without it, expressions like `x^{2} + y^{2} = r^{2}` might miss the `r^{2}` entirely.

3. **Spatial awareness** — 2D positional encoding lets the model distinguish spatially meaningful structures (fractions, superscripts, subscripts) that depend on vertical and horizontal position.

4. **Robust preprocessing** — Otsu binarization + background detection handles diverse input conditions (photos, scans, dark/light backgrounds) without manual tuning.

5. **Separation of concerns** — Encoder handles visual feature extraction, decoder handles language modeling, ARM handles attention coverage. Each component can be tuned or replaced independently.
