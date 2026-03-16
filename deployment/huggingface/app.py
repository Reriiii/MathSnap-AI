"""
MathSnap AI — Mini-CoMER HMER Backend for Hugging Face Spaces.
Receives an image, preprocesses it, runs greedy decode, returns LaTeX.
"""

import io
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from data.vocab import Vocab
from models.model import build_model
from config import Config

# ============================================================
# App setup
# ============================================================
app = FastAPI(title="MathSnap AI - Mini-CoMER Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Model loading (once at startup)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "checkpoints", "model_weights.pt")

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
    """Preprocess uploaded image to match training format."""
    img = np.array(pil_image.convert("L"))

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    border = np.concatenate([
        binary[0, :], binary[-1, :],
        binary[:, 0], binary[:, -1]
    ])
    bg_is_white = np.mean(border) > 128

    if bg_is_white:
        binary = 255 - binary

    h_hi, w_hi = config.data.h_hi, config.data.w_hi
    scale = min(h_hi / h, w_hi / w, 1.0)
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = binary.shape

    img_tensor = torch.from_numpy(binary).float().unsqueeze(0).unsqueeze(0) / 255.0
    mask_tensor = torch.zeros(1, h, w, dtype=torch.bool)

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


@app.get("/")
async def root():
    return {"message": "MathSnap AI - Mini-CoMER HMER Backend", "status": "running"}
