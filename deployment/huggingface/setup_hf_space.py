"""
Script to prepare Hugging Face Space directory.
Copies all required files from the project into the deployment folder.

Usage:
    cd D:/Workplace/hmer
    python deployment/huggingface/setup_hf_space.py
"""

import shutil
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HF_DIR = Path(__file__).resolve().parent

print(f"Project root: {PROJECT_ROOT}")
print(f"HF Space dir: {HF_DIR}")


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied: {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(HF_DIR)}")


def copy_dir(src, dst, pattern="*.py"):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob(pattern):
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        print(f"  Copied: {f.relative_to(PROJECT_ROOT)} -> {target.relative_to(HF_DIR)}")


print("\n1. Copying model code (models/)...")
copy_dir(PROJECT_ROOT / "models", HF_DIR / "models")

print("\n2. Copying data utilities (data/)...")
copy_dir(PROJECT_ROOT / "data", HF_DIR / "data")

print("\n3. Copying config.py...")
copy_file(PROJECT_ROOT / "config.py", HF_DIR / "config.py")

print("\n4. Copying vocab.json...")
copy_file(PROJECT_ROOT / "backend" / "vocab.json", HF_DIR / "vocab.json")

print("\n5. Copying model weights...")
weights_src = PROJECT_ROOT / "checkpoints" / "model_weights.pt"
weights_dst = HF_DIR / "checkpoints" / "model_weights.pt"
if weights_src.exists():
    copy_file(weights_src, weights_dst)
    size_mb = weights_src.stat().st_size / 1024 / 1024
    print(f"     Size: {size_mb:.1f} MB")
else:
    print(f"  WARNING: {weights_src} not found!")

print("\n[OK] HF Space directory ready!")
print(f"\nFiles in {HF_DIR}:")
for f in sorted(HF_DIR.rglob("*")):
    if f.is_file() and "__pycache__" not in str(f):
        size = f.stat().st_size
        unit = "KB" if size < 1024 * 1024 else "MB"
        val = size / 1024 if unit == "KB" else size / (1024 * 1024)
        print(f"  {f.relative_to(HF_DIR)} ({val:.1f} {unit})")
