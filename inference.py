"""
inference.py — Test NAMER trên 1 ảnh bất kỳ
============================================
Cách dùng:

  # Ảnh cụ thể:
  python inference.py --image path/to/image.png

  # Random 1 ảnh từ val set:
  python inference.py

  # Checkpoint cụ thể:
  python inference.py --checkpoint checkpoints/best.pth

  # In toạ độ VAT detection chi tiết:
  python inference.py --verbose
"""
import argparse, random, sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import Config
from data import Vocabulary
from data.dataset import _parse_label_file, _split3
from models import NAMER


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='NAMER inference on a single image')
parser.add_argument('--image',      type=str, default=r"D:\dataset\HME100K\images\1824.png",
                    help='Path to image file.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint .pth (default: checkpoints/best.pth or latest ep*.pth)')
parser.add_argument('--vocab',      type=str, default='./vocab.json')
parser.add_argument('--verbose',    action='store_true',
                    help='Print detailed VAT detection map')
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
cfg    = Config()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


def _find_checkpoint():
    if args.checkpoint:
        return Path(args.checkpoint)
    ckpt_dir = Path(cfg.checkpoint_dir)
    best = ckpt_dir / 'best.pth'
    if best.exists():
        return best
    ep_files = sorted(ckpt_dir.glob('ep*.pth'))
    if ep_files:
        return ep_files[-1]
    raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')


ckpt_path = _find_checkpoint()
print(f'Checkpoint: {ckpt_path}')

# ── Load vocab ────────────────────────────────────────────────────────────────
vocab_path = Path(args.vocab)
if not vocab_path.exists():
    sys.exit(f'[ERROR] Vocab not found: {vocab_path}')
vocab = Vocabulary.load(str(vocab_path))

# ── Load model ────────────────────────────────────────────────────────────────
model = NAMER(
    vocab_size=len(vocab), d=cfg.d_model, heads=cfg.nhead,
    pgd_layers=cfg.pgd_layers, drop=0.0,
).to(DEVICE)

ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
ep  = ckpt.get('epoch', '?')
er  = ckpt.get('exprate', 0.0)
print(f'Loaded: epoch={ep}  val_ExpRate={er*100:.2f}%')
print(f'Vocab : {len(vocab)} tokens\n')

# ── Preprocess ────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((cfg.img_h, cfg.img_w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Select image ─────────────────────────────────────────────────────────────
gt_latex = gt_toks = None
img_path = Path(args.image)

if not img_path.exists():
    print(f'Image {img_path} not found — picking random from val set...')
    all_s = _parse_label_file(cfg.label_file, cfg.data_root, cfg.max_len)
    _, val_s, _ = _split3(all_s, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    sample   = random.choice(val_s)
    img_path = Path(sample['img_path'])
    gt_latex = sample['latex']
    gt_toks  = sample['tokens']

print(f'Image: {img_path}')
if gt_latex:
    print(f'GT   : {gt_latex}')

# ── Inference ─────────────────────────────────────────────────────────────────
try:
    img = Image.open(img_path).convert('RGB')
except Exception as e:
    sys.exit(f'[ERROR] Cannot open image: {e}')

tensor = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred_ids = model(tensor, token_ids=None)[0]

pred_toks  = vocab.decode(pred_ids)
pred_latex = ' '.join(pred_toks)
print(f'Pred : {pred_latex}')
if gt_toks is not None:
    print(f'Match: {pred_toks == gt_toks}')

# ── VAT detection ─────────────────────────────────────────────────────────────
print('\n── VAT Detection ──')
with torch.no_grad():
    f8x, f16x   = model.enc(tensor)
    probs, _    = model.vat(f8x, f16x)

none_idx = vocab.none_idx
pred_map = probs[0].argmax(dim=0)
detected = (pred_map != none_idx).nonzero(as_tuple=False)

if detected.size(0) == 0:
    print('  VAT: no tokens detected (model not trained enough)')
else:
    print(f'  VAT: {detected.size(0)} tokens detected')
    if args.verbose:
        for pos in detected:
            r, c = pos[0].item(), pos[1].item()
            tid  = pred_map[r, c].item()
            tok  = vocab.i2t.get(tid, '?')
            conf = probs[0, tid, r, c].item()
            print(f'    [{r:2d},{c:2d}]  {tok:20s}  conf={conf:.3f}')
    else:
        tids = [pred_map[p[0], p[1]].item() for p in detected]
        toks = [vocab.i2t.get(t, '?') for t in tids]
        col_order = detected[:, 1].argsort()
        print(f'  Tokens (L→R): {" ".join(toks[i] for i in col_order.tolist())}')
        print(f'  (use --verbose for per-position confidence)')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n── Summary ──')
if gt_toks:
    print(f'  GT   ({len(gt_toks):3d} tokens): {" ".join(gt_toks[:20])}{"..." if len(gt_toks)>20 else ""}')
print(f'  Pred ({len(pred_toks):3d} tokens): {" ".join(pred_toks[:20])}{"..." if len(pred_toks)>20 else ""}')