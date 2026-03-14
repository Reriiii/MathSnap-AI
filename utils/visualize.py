"""
Visualization utilities for HMER training.

Generates training curves, attention heatmaps, and sample predictions.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_curves(
    history: dict,
    output_dir: str = "outputs",
    filename: str = "training_curves.png"
):
    """
    Plot training and validation loss/metric curves.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'exprate', 'bleu', etc.
                 Each value is a list of values per epoch.
        output_dir: directory to save plots
        filename: output filename
    """
    os.makedirs(output_dir, exist_ok=True)

    has_ctc = 'train_ctc_loss' in history and any(v > 0 for v in history.get('train_ctc_loss', []))
    n_rows = 3 if has_ctc else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    fig.suptitle('HMER Training Progress', fontsize=16, fontweight='bold')

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # 1. Loss curves
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ExpRate curves
    ax = axes[0, 1]
    if 'exprate' in history:
        ax.plot(epochs, history['exprate'], 'g-', label='ExpRate', linewidth=2)
    if 'exprate_1' in history:
        ax.plot(epochs, history['exprate_1'], 'g--', label='ExpRate@1', alpha=0.7)
    if 'exprate_2' in history:
        ax.plot(epochs, history['exprate_2'], 'g:', label='ExpRate@2', alpha=0.7)
    # Mark peak ExpRate
    if 'exprate' in history and history['exprate']:
        peak_val = max(history['exprate'])
        peak_ep = history['exprate'].index(peak_val) + 1
        ax.axvline(peak_ep, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.annotate(f'Peak {peak_val:.1f}%', xy=(peak_ep, peak_val),
                    xytext=(5, -15), textcoords='offset points', fontsize=8, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ExpRate (%)')
    ax.set_title('Expression Recognition Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. BLEU curves
    ax = axes[1, 0]
    if 'bleu' in history:
        ax.plot(epochs, history['bleu'], 'm-', label='BLEU-4', linewidth=2)
    if 'bleu_1' in history:
        ax.plot(epochs, history['bleu_1'], 'm--', label='BLEU-1', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BLEU (%)')
    ax.set_title('BLEU Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Learning rate
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'c-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    # 5. CTC loss (only if present and non-zero)
    if has_ctc:
        ax = axes[2, 0]
        ax.plot(epochs, history['train_ctc_loss'], 'orange', label='CTC Loss', linewidth=2)
        if 'train_ce_loss' in history:
            ax.plot(epochs, history['train_ce_loss'], 'b--', label='CE Loss', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('CTC vs CE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Token accuracy
        ax = axes[2, 1]
        if 'token_accuracy' in history:
            ax.plot(epochs, history['token_accuracy'], 'teal', label='Token Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Token Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_sample_predictions(
    images: np.ndarray,
    predictions: List[str],
    targets: List[str],
    output_dir: str = "outputs",
    filename: str = "sample_predictions.png",
    num_samples: int = 8,
    image_paths: List[str] = None,
):
    """
    Display sample predictions alongside ground truth.

    Args:
        images: [N, H, W] or [N, 1, H, W] grayscale images (numpy)
        predictions: predicted LaTeX strings
        targets: ground truth LaTeX strings
        output_dir: directory to save plot
        filename: output filename
        num_samples: number of samples to show
        image_paths: optional list of source image file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    n = min(num_samples, len(predictions))
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]

        # Handle image dimensions
        img = images[i]
        if img.ndim == 3:
            img = img.squeeze(0)  # Remove channel dim

        # Denormalize (from [-1,1] to [0,1])
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        ax.imshow(img, cmap='gray', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])

        # Color code: green for match, red for mismatch
        is_correct = predictions[i].strip() == targets[i].strip()
        color = 'green' if is_correct else 'red'

        pred_display = predictions[i] if len(predictions[i]) < 80 else predictions[i][:77] + "..."
        tgt_display = targets[i] if len(targets[i]) < 80 else targets[i][:77] + "..."

        # Build title with optional file path
        title_lines = []
        if image_paths and i < len(image_paths):
            path_display = os.path.basename(image_paths[i])
            title_lines.append(f"File: {image_paths[i]}")
        title_lines.append(f"GT:   {tgt_display}")
        title_lines.append(f"Pred: {pred_display}")

        ax.set_title(
            "\n".join(title_lines),
            fontsize=9,
            color=color,
            loc='left'
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def plot_attention_heatmap(
    image: np.ndarray,
    attention_weights: np.ndarray,
    tokens: List[str],
    output_dir: str = "outputs",
    filename: str = "attention_heatmap.png",
    top_k_steps: int = 8,
):
    """
    Visualize decoder attention weights overlaid on the input image.

    Args:
        image: [H, W] or [1, H, W] grayscale image (numpy)
        attention_weights: [T, S] attention from decoder to encoder features
        tokens: decoded tokens corresponding to each timestep
        output_dir: directory to save
        filename: output filename
        top_k_steps: number of decoding steps to show
    """
    os.makedirs(output_dir, exist_ok=True)

    if image.ndim == 3:
        image = image.squeeze(0)

    # Denormalize
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)

    T = min(top_k_steps, len(tokens), attention_weights.shape[0])
    cols = 4
    rows = (T + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle('Attention Heatmaps', fontsize=14, fontweight='bold')

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    H, W = image.shape
    # Infer spatial dims of attention
    S = attention_weights.shape[1]
    # Try to find H_feat, W_feat such that H_feat * W_feat = S
    feat_h = int(np.sqrt(S * H / W))
    feat_w = S // feat_h if feat_h > 0 else S

    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]

        if i < T:
            ax.imshow(image, cmap='gray', aspect='auto')

            # Reshape attention to 2D and overlay
            attn = attention_weights[i]
            try:
                attn_2d = attn.reshape(feat_h, feat_w)
            except ValueError:
                attn_2d = attn.reshape(1, -1)

            # Resize attention map to image size using simple interpolation
            from PIL import Image as PILImage
            attn_resized = np.array(
                PILImage.fromarray(attn_2d.astype(np.float32)).resize(
                    (W, H), PILImage.BILINEAR
                )
            )

            ax.imshow(attn_resized, cmap='jet', alpha=0.4, aspect='auto')
            token_label = tokens[i] if i < len(tokens) else "?"
            ax.set_title(f"t={i}: '{token_label}'", fontsize=10)
        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved to {save_path}")


def save_history(history: dict, output_dir: str = "outputs", filename: str = "history.json"):
    """Save training history to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {save_path}")


def load_history(output_dir: str = "outputs", filename: str = "history.json") -> dict:
    """Load training history from JSON."""
    load_path = os.path.join(output_dir, filename)
    with open(load_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate visualization from saved history
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    history = load_history(output_dir)
    plot_training_curves(history, output_dir)
    print("Visualizations generated!")