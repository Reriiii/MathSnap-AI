"""
eval_checkpoint.py — Evaluate a saved NAMER checkpoint

Usage:
    python eval_checkpoint.py --ckpt checkpoints/best.pth
    python eval_checkpoint.py --ckpt checkpoints/best.pth --split test
    python eval_checkpoint.py --ckpt checkpoints/best.pth --split val --batch_size 64
"""
import argparse
from functools import partial

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from config import Config
from data import build_datasets
from data.dataset import _collate
from models import NAMER
from utils.metrics import compute_exprate
from utils.matching import make_vat_targets


@torch.no_grad()
def evaluate(model, loader, vocab, device, split='val'):
    model.eval()
    preds, gts = [], []
    vat_tp = vat_fp = vat_fn = vat_det_total = 0
    n_samples = 0
    none_idx = vocab.none_idx

    pbar = tqdm(loader, desc=f"[{split}]", dynamic_ncols=True, unit='batch')
    for batch in pbar:
        images       = batch['image'].to(device)
        token_ids_gt = batch['token_ids'].to(device)

        f8x, f16x    = model.enc(images)
        vat_probs, _ = model.vat(f8x, f16x)
        pred_seqs    = model._infer(f16x, vat_probs)

        # VAT diagnostic
        vat_pred_map = vat_probs.argmax(dim=1)
        B = images.size(0)
        for b in range(B):
            pred_map = vat_pred_map[b]
            gt_tids  = token_ids_gt[b]
            gt_set   = set(t.item() for t in gt_tids
                           if t.item() not in (vocab.pad_idx, vocab.sos_idx,
                                                vocab.eos_idx, none_idx))
            det_mask = (pred_map != none_idx)
            det_tids = pred_map[det_mask].cpu().tolist()
            det_set  = set(det_tids)
            tp = len(det_set & gt_set)
            fp = len(det_tids) - tp
            fn = len(gt_set)   - tp
            vat_tp += tp; vat_fp += fp; vat_fn += fn
            vat_det_total += len(det_tids)
            n_samples += 1

        for ids, gt_toks in zip(pred_seqs, batch['tokens']):
            preds.append(vocab.decode(ids))
            gts.append(gt_toks)

    pbar.close()

    er, l1, l2 = compute_exprate(preds, gts)
    prec    = vat_tp / max(vat_tp + vat_fp, 1)
    rec     = vat_tp / max(vat_tp + vat_fn, 1)
    avg_det = vat_det_total / max(n_samples, 1)

    print(f"\n{'='*50}")
    print(f"  Split      : {split}")
    print(f"  ExpRate    : {er*100:.2f}%")
    print(f"  ExpRate ≤1 : {l1*100:.2f}%")
    print(f"  ExpRate ≤2 : {l2*100:.2f}%")
    print(f"  VAT prec   : {prec*100:.1f}%")
    print(f"  VAT rec    : {rec*100:.1f}%")
    print(f"  VAT avg_det: {avg_det:.1f}")
    print(f"{'='*50}\n")
    return er, l1, l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True,  help='Path to checkpoint .pth')
    parser.add_argument('--split',      default='val',  choices=['val', 'test', 'train'])
    parser.add_argument('--batch_size', type=int, default=None, help='Override config batch size')
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"Split  : {args.split}")

    train_ds, val_ds, test_ds, vocab = build_datasets(cfg)
    ds = {'train': train_ds, 'val': val_ds, 'test': test_ds}[args.split]

    bs  = args.batch_size or cfg.batch_size
    col = partial(_collate, pad_idx=vocab.pad_idx)
    loader = DataLoader(ds, bs, shuffle=False,
                        collate_fn=col,
                        num_workers=cfg.num_workers,
                        pin_memory=(device.type == 'cuda'),
                        persistent_workers=(cfg.num_workers > 0),
                        prefetch_factor=(2 if cfg.num_workers > 0 else None))

    model = NAMER(vocab_size=len(vocab), d=cfg.d_model, heads=cfg.nhead,
                  pgd_layers=cfg.pgd_layers, drop=cfg.drop).to(device)

    ckpt       = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt['model']
    # torch.compile adds "_orig_mod." prefix — strip it for plain model loading
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        print("Stripped _orig_mod. prefix (torch.compile checkpoint)")
    model.load_state_dict(state_dict)
    saved_ep = ckpt.get('epoch', '?')
    saved_er = ckpt.get('exprate', 0)
    print(f"Loaded : epoch={saved_ep}  saved_exprate={saved_er*100:.2f}%\n")

    evaluate(model, loader, vocab, device, split=args.split)


if __name__ == '__main__':
    main()