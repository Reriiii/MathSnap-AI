---
title: MathSnap AI - Mini-CoMER
emoji: ✏️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MathSnap AI — Mini-CoMER HMER Backend

Handwritten Mathematical Expression Recognition using Mini-CoMER (DenseNet Encoder + Transformer Decoder with ARM).

- **Model**: Mini-CoMER (6.39M params)
- **Architecture**: DenseNet encoder (growth_rate=24, 16 layers) + Transformer decoder (d_model=256, 3 layers, 8 heads) with Attention Refinement Module
- **Dataset**: CROHME 2013/2016/2019 (27,056 samples)
- **Best ExpRate**: 47.12% (epoch 194/300)
