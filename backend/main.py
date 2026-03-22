"""
FastAPI backend for CoMER HMER model inference.
Receives an image, preprocesses it, runs greedy decode, returns LaTeX.
"""

import io
import os
import sys

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add project root to path so we can import models/data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.vocab import Vocab
from models.model import build_model
from config import Config

# ============================================================
# App setup
# ============================================================
app = FastAPI(title="MathSnap AI - HMER Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Model loading (once at startup)
# ============================================================
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BACKEND_DIR, "vocab.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "model_weights.pt")

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading vocab from {VOCAB_PATH}")
vocab = Vocab.from_file(VOCAB_PATH)

print(f"Building model ({config.model.d_model}d, {config.model.num_decoder_layers}L)...")
model = build_model(config, len(vocab)).to(device)

print(f"Loading weights from {WEIGHTS_PATH}")
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=False))
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model ready: {num_params/1e6:.2f}M params on {device}")


# ============================================================
# Image preprocessing (must match training pipeline)
# ============================================================
def preprocess_image(pil_image: Image.Image) -> tuple:
    """Preprocess uploaded image to match CoMER training format.

    Training uses: black background, white foreground, grayscale,
    variable size within h_hi x w_hi bounds.

    Handles diverse inputs: photos of paper, colored backgrounds,
    low contrast, noise, etc.

    Returns:
        img_tensor: [1, 1, H, W] float32
        mask_tensor: [1, H, W] bool (False = valid, True = padding)
    """
    # Convert to grayscale numpy
    img = np.array(pil_image.convert("L"))  # [H, W] uint8

    # --- Noise reduction ---
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # --- Adaptive thresholding (handles uneven lighting from photos) ---
    # Try adaptive first, fall back to Otsu
    adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )

    # Also compute Otsu for comparison
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use adaptive if image has uneven lighting (high std in local means),
    # otherwise use Otsu (cleaner for screenshots)
    local_means = cv2.blur(img, (64, 64))
    lighting_variation = np.std(local_means)
    binary = adaptive if lighting_variation > 15 else otsu

    # --- Detect and normalize background ---
    h, w = binary.shape
    border = np.concatenate([
        binary[0, :], binary[-1, :],
        binary[:, 0], binary[:, -1]
    ])
    bg_is_white = np.mean(border) > 128

    # Ensure black background, white foreground (matching training data)
    if bg_is_white:
        binary = 255 - binary

    # --- Remove small noise blobs ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # --- Crop to content (remove excess border) ---
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        pad = 8  # small padding around content
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        binary = binary[y1:y2, x1:x2]

    # --- Scale to fit within bounds (preserve aspect ratio) ---
    h, w = binary.shape
    h_hi, w_hi = config.data.h_hi, config.data.w_hi
    scale = min(h_hi / h, w_hi / w, 1.0)
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = binary.shape

    # To tensor
    img_tensor = torch.from_numpy(binary).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
    mask_tensor = torch.zeros(1, h, w, dtype=torch.bool)  # no padding

    return img_tensor.to(device), mask_tensor.to(device)


# ============================================================
# API endpoints
# ============================================================
@app.post("/predict")
async def predict_latex(file: UploadFile = File(...)):
    """Receive image, return predicted LaTeX string."""
    image_data = await file.read()
    pil_image = Image.open(io.BytesIO(image_data))

    img_tensor, mask_tensor = preprocess_image(pil_image)

    with torch.no_grad():
        pred_indices = model.greedy_decode(
            img_tensor, mask_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=150,
        )

    latex_tokens = vocab.decode(pred_indices[0], remove_special=True)

    return {"latex": latex_tokens}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "vocab_size": len(vocab),
        "model_params": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
