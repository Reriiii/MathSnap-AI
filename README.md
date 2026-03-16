# MathSnap AI — Handwritten Mathematical Expression Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**An end-to-end system for recognizing handwritten mathematical expressions and converting them into LaTeX, powered by a compact neural architecture (Mini-CoMER) and a modern web interface.**

[Live Demo](https://mathsnap-ai.vercel.app) · [Backend API](https://quoctk31-mathsnap-ai.hf.space) · [Report Bug](https://github.com/Reriiii/hmer-using-graph-transformer/issues)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Mini-CoMER Model](#mini-comer-model)
  - [DenseNet Encoder](#densenet-encoder)
  - [Transformer Decoder with ARM](#transformer-decoder-with-arm)
  - [Attention Refinement Module (ARM)](#attention-refinement-module-arm)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [System Design](#system-design)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (React + Vite)](#frontend-react--vite)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
- [Deployment](#deployment)
  - [Backend → Hugging Face Spaces](#backend--hugging-face-spaces)
  - [Frontend → Vercel](#frontend--vercel)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

**MathSnap AI** addresses the problem of Handwritten Mathematical Expression Recognition (HMER) — automatically converting images of handwritten math into structured LaTeX markup. This project implements **Mini-CoMER**, a lightweight variant of the CoMER architecture ([Zhao et al., 2022](https://arxiv.org/abs/2207.04410)), optimized for deployment on resource-constrained environments with a single consumer-grade GPU.

### Key Contributions

1. **Compact Architecture Study**: Systematic reduction of CoMER's architecture (6.39M vs ~20M+ parameters), analyzing the trade-offs between model size and recognition accuracy for practical deployment scenarios.
2. **End-to-End Application**: Complete pipeline from training to inference to deployment, including a production-ready web interface with real-time LaTeX rendering.
3. **Deployment Pipeline**: Automated deployment workflow using Hugging Face Spaces (backend) and Vercel (frontend), enabling free-tier cloud hosting with GPU inference support.
4. **Comprehensive Analytics Dashboard**: Interactive visualization of dataset statistics, training metrics, and model architecture details.

---

## Architecture

### Mini-CoMER Model

Mini-CoMER follows an encoder-decoder paradigm with an Attention Refinement Module (ARM) for coverage-based attention correction. The architecture is adapted from [CoMER](https://github.com/Green-Wood/CoMER) with reduced capacity for efficient single-GPU training and inference.

```
Input Image [b, 1, H, W]
        │
        ▼
┌─────────────────────┐
│   DenseNet Encoder   │  3 dense blocks, growth_rate=24, 16 layers/block
│   + 2D Pos Encoding  │  Output: [b, h, w, 256]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Transformer Decoder  │  3 layers, 8 heads, d_ff=1024
│   + ARM Coverage     │  Cross-coverage + Self-coverage
└─────────┬───────────┘
          │
          ▼
   LaTeX Token Sequence
```

**Model Summary:**

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Shared** | d_model | 256 |
| **Encoder** | Architecture | DenseNet-B |
| | Growth rate | 24 |
| | Dense blocks | 3 |
| | Layers per block | 16 |
| | Bottleneck | Yes (4x growth rate) |
| | Reduction factor | 0.5 |
| | Output channels | 684 → projected to 256 |
| **Decoder** | Layers | 3 |
| | Attention heads | 8 |
| | FFN dimension | 1,024 |
| | Dropout | 0.3 |
| **ARM** | Intermediate channels (dc) | 32 |
| | Cross-coverage | Enabled |
| | Self-coverage | Enabled |
| | Kernel size | 5x5 |
| **Total** | Parameters | **6.39M** |

### DenseNet Encoder

The encoder employs a DenseNet-B (bottleneck) architecture with three dense blocks separated by transition layers. Each bottleneck layer consists of a 1x1 convolution (expanding to 4x growth rate) followed by a 3x3 convolution. Transition layers perform 1x1 convolution with 0.5 reduction and 2x2 average pooling.

The raw feature maps are projected to `d_model=256` dimensions via a 1x1 convolution, followed by 2D sinusoidal positional encoding and LayerNorm.

**Feature extraction pipeline:**

```
Input [b, 1, H, W]
  → Conv2d(1, 48, 7x7, stride=2) + BN + ReLU
  → MaxPool2d(2x2)
  → DenseBlock1 (16 bottleneck layers, growth_rate=24)
  → Transition1 (reduction=0.5, AvgPool 2x2)
  → DenseBlock2 (16 bottleneck layers, growth_rate=24)
  → Transition2 (reduction=0.5, AvgPool 2x2)
  → DenseBlock3 (16 bottleneck layers, growth_rate=24)
  → BN
  → Conv2d(684, 256, 1x1)        # Feature projection
  → Rearrange [b, d, h, w] → [b, h, w, d]
  → 2D Positional Encoding
  → LayerNorm
Output [b, h, w, 256]
```

### Transformer Decoder with ARM

The decoder is a standard Transformer decoder augmented with the Attention Refinement Module. It operates autoregressively, generating one LaTeX token at a time conditioned on the encoded image features and previously generated tokens.

- **Word Embedding**: Learned embedding (vocab_size x 256) + LayerNorm
- **Positional Encoding**: 1D sinusoidal encoding (max 500 positions)
- **Decoding**: Greedy (default for inference) or Beam Search (configurable beam_size=10, length penalty alpha=1.0)

### Attention Refinement Module (ARM)

ARM addresses the coverage problem in attention-based HMER — where the model may repeatedly attend to the same image regions or miss certain regions entirely. It computes cumulative attention maps and refines them through convolutional layers:

```
Previous cross-attention maps (accumulated)  ─┐
Current self-attention maps (accumulated)    ─┤
                                               ▼
                                    Concatenate [2 x nhead, h, w]
                                         │
                                    Conv2d(5x5) + ReLU
                                         │
                                    Conv2d(1x1)
                                         │
                                    MaskBatchNorm2d
                                         │
                                    Coverage bias → added to attention logits
```

---

## Dataset

The model is trained and evaluated on the **CROHME** (Competition on Recognition of Online Handwritten Mathematical Expressions) dataset, combining samples from CROHME 2013, 2016, and 2019 competitions.

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 21,855 | 80.7% |
| Validation | 1,702 | 6.3% |
| Test | 3,499 | 12.9% |
| **Total** | **27,056** | **100%** |

**Vocabulary**: 114 tokens (110 LaTeX tokens + 4 special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`)

**Token Categories**:
- Digits (0-9), Lowercase/Uppercase Latin letters
- Greek letters (alpha, beta, gamma, delta, theta, pi, sigma, ...)
- Operators (+, -, =, <, >, times, div, leq, geq, ...)
- Functions (sin, cos, tan, log, lim, ...)
- Structural tokens (frac, sqrt, ^, _, {, }, ...)
- Delimiters, Symbols (infty, in, rightarrow, sum, int, prod, ...)

**Sequence Length Statistics**:
- Mean: ~12 tokens | Median: ~10 tokens
- 95th percentile: ~30 tokens
- Maximum: ~200 tokens

**Preprocessing Pipeline**:
1. Convert to grayscale
2. Otsu binarization
3. Background detection (border pixel analysis) and invert if white background
4. Aspect-ratio-preserving resize to fit within 128x512 pixels
5. Normalize to [0, 1] float tensor

---

## Training

Training was conducted on a single **NVIDIA RTX 5060 Ti (16 GB VRAM)** with the following configuration:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (beta1=0.9, beta2=0.999) |
| Learning rate | 1e-4 (initial) |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Loss function | Cross-entropy |
| Batch size | 64 |
| Mixed precision | FP16 (torch.cuda.amp) |
| Gradient clipping | Max norm = 5.0 |
| Epochs trained | 230 / 300 (early stopped) |
| Training time | ~48 hours |

**Learning Rate Schedule Events:**

| Epoch | New LR | Trigger |
|-------|--------|---------|
| 0 | 1.0e-4 | Initial |
| 82 | 5.0e-5 | Plateau (patience=10) |
| 131 | 2.5e-5 | Plateau |
| 176 | 1.25e-5 | Plateau |
| 211 | 6.25e-6 | Plateau |

---

## Results

### Expression Recognition Rate (ExpRate)

| Model | Params | CROHME Test ExpRate |
|-------|--------|---------------------|
| CoMER (Zhao et al., 2022) | ~20M+ | 59.33% |
| **Mini-CoMER (Ours)** | **6.39M** | **47.12%** |

- **Best epoch**: 194 / 300
- **Best validation ExpRate**: 47.12%

### Analysis

The 12-point gap relative to the original CoMER is attributable to:

1. **Reduced decoder depth** (3 layers vs. 6): fewer layers of attention refinement limit the model's ability to resolve complex structural dependencies.
2. **Incomplete training** (230/300 epochs): training was terminated before full convergence due to diminishing returns.
3. **Single-GPU constraints**: smaller effective batch size compared to the original multi-GPU distributed training setup.
4. **No auxiliary losses**: the original CoMER benefits from ICAL's auxiliary losses (SCCM, FusionModule); this implementation uses only standard cross-entropy.

Despite the lower accuracy, Mini-CoMER achieves a **68% parameter reduction** while retaining the core architectural innovations (ARM coverage mechanism), demonstrating the feasibility of deploying attention-based HMER models in resource-constrained environments.

---

## System Design

### Backend (FastAPI)

The inference backend is built with FastAPI, providing a RESTful API for image-to-LaTeX conversion.

**Tech Stack:**
- Python 3.10+
- PyTorch 2.5 (CUDA / CPU)
- FastAPI + Uvicorn
- OpenCV (image preprocessing)
- PIL / NumPy

**Inference Pipeline:**
```
Client uploads image
  → FastAPI receives multipart/form-data
  → PIL opens image
  → Preprocessing (grayscale, binarize, resize, normalize)
  → Model.greedy_decode(img_tensor, mask_tensor)
  → Vocab.decode(predicted_indices)
  → Return {"latex": "x ^ { 2 } + y ^ { 2 } = r ^ { 2 }"}
```

### Frontend (React + Vite)

The web interface provides an interactive environment for uploading handwritten math images and viewing recognition results.

**Tech Stack:**
- React 18.3 + TypeScript 5.6
- Vite 6.3 (build tool)
- Tailwind CSS 4 (styling)
- Radix UI + shadcn/ui (component library)
- KaTeX 0.16 (LaTeX rendering)
- Recharts 2.15 (data visualization)
- Motion (animations)
- next-themes (dark/light mode)

**Pages:**

| Page | Description |
|------|-------------|
| **Home** | Landing page with feature highlights and workflow overview |
| **Convert** | Core functionality: image upload (drag-and-drop, paste, file picker), side-by-side LaTeX code editor and KaTeX preview with font-size and zoom controls |
| **Dashboard** | Interactive analytics with 4 tabs: Overview (KPIs), Dataset (token frequency, sequence lengths, complexity), Model (architecture diagram, parameter distribution), Training (loss curves, ExpRate, LR schedule) |
| **Settings** | Theme preferences and configuration |

**Responsive Breakpoints:**

| Breakpoint | Layout |
|------------|--------|
| Desktop (>=1024px) | 3-panel: Image sidebar + Code + Preview |
| Tablet (768-1024px) | Compact upload strip + 50/50 Code/Preview split |
| Mobile (<768px) | Stacked layout with tab switching (Code/Preview) |

---

## Installation

### Prerequisites

- **Node.js** >= 18.0
- **Python** >= 3.10
- **CUDA** >= 11.8 (optional, for GPU inference)
- **Git LFS** (for model weights)

### Local Development

**1. Clone the repository:**

```bash
git clone https://github.com/Reriiii/hmer-using-graph-transformer.git
cd hmer-using-graph-transformer
```

**2. Install frontend dependencies:**

```bash
npm install
```

**3. Install backend dependencies:**

```bash
pip install -r backend/requirements.txt
```

**4. Configure environment:**

```bash
cp .env.example .env
# Edit .env to set VITE_API_URL (default: http://localhost:8000)
```

**5. Start the backend:**

```bash
python backend/main.py
# Server starts at http://localhost:8000
```

**6. Start the frontend (separate terminal):**

```bash
npm run dev
# Opens at http://localhost:5173
```

---

## Deployment

### Backend → Hugging Face Spaces

1. Create a new Space on [huggingface.co](https://huggingface.co/new-space) with **Docker SDK** (Blank template).

2. Prepare deployment files:
   ```bash
   python deployment/huggingface/setup_hf_space.py
   ```

3. Push to Hugging Face:
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/YOUR-USERNAME/mathsnap-ai
   cd mathsnap-ai
   cp -r ../deployment/huggingface/* .
   git lfs track "*.pt"
   git add . && git commit -m "Deploy Mini-CoMER backend"
   git push
   ```

4. Wait for Docker build (~5-10 min). The API will be available at:
   ```
   https://YOUR-USERNAME-mathsnap-ai.hf.space
   ```

### Frontend → Vercel

1. Update `.env`:
   ```
   VITE_API_URL=https://YOUR-USERNAME-mathsnap-ai.hf.space
   ```

2. Deploy:
   ```bash
   npm install -g vercel
   vercel --prod
   ```

   Or connect your GitHub repository via the [Vercel Dashboard](https://vercel.com) for automatic deployments.

---

## Project Structure

```
hmer/
├── backend/                    # FastAPI inference server
│   ├── main.py                 #   API endpoints (/predict, /health)
│   ├── analyze_dataset.py      #   Dataset statistics generator
│   ├── vocab.json              #   Token vocabulary (114 tokens)
│   └── requirements.txt        #   Python dependencies
│
├── models/                     # PyTorch model architecture
│   ├── model.py                #   CoMER main class + greedy/beam decode
│   ├── encoder.py              #   DenseNet encoder + 2D positional encoding
│   ├── decoder.py              #   Transformer decoder + word embedding
│   ├── pos_enc.py              #   1D (WordPosEnc) + 2D (ImgPosEnc)
│   └── transformer/
│       ├── arm.py              #   Attention Refinement Module
│       ├── attention.py        #   Multi-head attention with ARM
│       └── transformer_decoder.py  # Decoder layer stack
│
├── data/                       # Data utilities
│   └── vocab.py                #   Vocabulary encode/decode
│
├── config.py                   # Model, data, and training configuration
│
├── checkpoints/                # Model weights (Git LFS)
│   └── model_weights.pt        #   Best checkpoint (25.1 MB)
│
├── dataset/                    # CROHME dataset (not in repo)
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/                        # React frontend source
│   ├── main.tsx                #   Entry point
│   ├── app/
│   │   ├── App.tsx             #   Root component + routing
│   │   ├── components/
│   │   │   ├── home-page.tsx
│   │   │   ├── convert-page.tsx
│   │   │   ├── dashboard-page.tsx
│   │   │   ├── settings-page.tsx
│   │   │   ├── header.tsx
│   │   │   ├── latex-preview.tsx
│   │   │   └── ui/             #   50+ Radix/shadcn components
│   │   └── data/
│   │       ├── dataset-stats.json
│   │       └── training-metrics.json
│   └── vite-env.d.ts
│
├── deployment/
│   └── huggingface/            # HF Spaces deployment files
│       ├── Dockerfile
│       ├── app.py
│       ├── requirements.txt
│       ├── README.md
│       └── setup_hf_space.py
│
├── package.json
├── vite.config.ts
├── vercel.json
├── .env.example
└── README.md
```

---

## Configuration

All model and training hyperparameters are centralized in `config.py`:

```python
@dataclass
class ModelConfig:
    d_model: int = 256
    growth_rate: int = 24           # DenseNet growth rate
    num_layers: int = 16            # Layers per dense block
    nhead: int = 8                  # Attention heads
    num_decoder_layers: int = 3     # Transformer decoder layers
    dim_feedforward: int = 1024     # FFN hidden dimension
    dropout: float = 0.3
    dc: int = 32                    # ARM intermediate channels
    cross_coverage: bool = True
    self_coverage: bool = True

@dataclass
class DataConfig:
    h_hi: int = 128                 # Max image height
    w_hi: int = 512                 # Max image width
    max_seq_len: int = 200          # Max LaTeX token sequence
    batch_size: int = 64
```

---

## API Reference

### `POST /predict`

Upload an image and receive the predicted LaTeX string.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@equation.png"
```

**Response:**
```json
{
  "latex": "x ^ { 2 } + y ^ { 2 } = r ^ { 2 }"
}
```

### `GET /health`

Check backend status.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "vocab_size": 114,
  "model_params": 6393714
}
```

---

## Acknowledgments

- **CoMER**: Zhao, W., Gao, L., Yan, Z., Peng, S., Du, L., & Zhang, Z. (2022). *CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition*. ECCV 2022. [Paper](https://arxiv.org/abs/2207.04410) | [Code](https://github.com/Green-Wood/CoMER)
- **ICAL**: [https://github.com/qingzhenduyu/ICAL](https://github.com/qingzhenduyu/ICAL) — DenseNet encoder implementation
- **CROHME Dataset**: Mouchere, H., Viard-Gaudin, C., Zanibbi, R., & Garain, U. — Competition on Recognition of Online Handwritten Mathematical Expressions
- **KaTeX**: [https://katex.org](https://katex.org) — Fast LaTeX rendering in the browser
- **shadcn/ui**: [https://ui.shadcn.com](https://ui.shadcn.com) — UI component library

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with PyTorch, React, and FastAPI</sub>
</div>
