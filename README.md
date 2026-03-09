# Graph-to-Graph HMER

A PyTorch reproduction of "Graph-to-Graph: Towards Accurate and Interpretable Online Handwritten Mathematical Expression Recognition" (Wu et al., AAAI 2021).

## Overview

This project implements the G2G model that formulates handwritten mathematical expression recognition as a graph-to-graph learning problem, achieving state-of-the-art results on CROHME datasets.

### Key Features

- **Graph-based representation** of both input strokes and output SLT
- **GNN encoder** with modified Graph Attention Networks
- **GNN decoder** with hierarchical structure awareness
- **Sub-graph attention** for explicit symbol segmentation
- **End-to-end training** with multiple supervision signals

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## Installation

```bash
# Create and activate environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Project Structure

```
g2g-hmer/
├── data/
│   ├── download.py          # Script to download CROHME data
│   └── preprocess.py         # Data preprocessing
├── models/
│   ├── encoder.py            # Encoder GNN
│   ├── decoder.py            # Decoder GNN
│   ├── attention.py          # Sub-graph attention
│   └── g2g.py                # Full G2G model
├── utils/
│   ├── graph_builder.py      # Graph construction
│   ├── losses.py             # Loss functions
│   └── metrics.py            # Evaluation metrics
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
└── configs/
    └── default.yaml          # Configuration file
```

## Data Preparation

1. Download CROHME dataset:
```bash
python data/download.py --output_dir ./data/crohme
```

2. Preprocess data:
```bash
python data/preprocess.py --data_dir ./data/crohme --output_dir ./data/processed
```

## Training

Train the G2G model:

```bash
python train.py --config configs/default.yaml
```

Key hyperparameters (from paper):
- Encoder/Decoder blocks: 3 layers each
- Feature extraction: 4 blocks (BiGRU + TCN)
- Embedding dimension: C=400, C'=256
- Optimizer: Adam with lr=5e-4
- Loss weights: λ1=λ2=λ6=0.5, λ3=λ4=1, λ5=0.3

## Evaluation

Evaluate on CROHME test sets:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --test_set 2014
```

## Results

Reproduction results on CROHME (ExpRate %):

| Model | CROHME 2014 | CROHME 2016 |
|-------|-------------|-------------|
| Paper | 54.46 | 52.05 |
| Ours  | TBD | TBD |

## Architecture Details

### Source Graph (Input)
- **Nodes**: Handwriting strokes
- **Edges**: Spatial (LOS algorithm) + temporal connections
- **Features**: BiGRU + TCN encoding with masking

### Target Graph (Output)
- **Nodes**: SLT nodes (symbols + end-child markers)
- **Edges**: Grandparent-to-child, parent-to-child, sibling edges
- **Structure**: Modified SLT with depth-first traversal

### Training Objectives
1. Node classification (encoder): Symbol categories
2. Edge classification (encoder): Spatial relationships
3. Node generation (decoder): Target symbols
4. Edge generation (decoder): Structural relations
5. Sub-graph attention: Symbol segmentation
6. Sub-graph supervision: Symbol-level alignment

## Citation

```bibtex
@inproceedings{wu2021graph,
  title={Graph-to-Graph: Towards Accurate and Interpretable Online Handwritten Mathematical Expression Recognition},
  author={Wu, Jin-Wen and Yin, Fei and Zhang, Yan-Ming and Zhang, Xu-Yao and Liu, Cheng-Lin},
  booktitle={AAAI},
  year={2021}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Original paper by Wu et al. from National Laboratory of Pattern Recognition, Institute of Automation of Chinese Academy of Sciences.