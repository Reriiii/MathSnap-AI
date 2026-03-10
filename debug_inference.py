"""
Deep diagnostic: measure each stage of the NAMER pipeline on val set.
Stage 1: VAT detection quality (recall, precision, accuracy per token)
Stage 2: PGD SCH correction quality 
Stage 3: path_selection quality
"""
import torch, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from config import Config
from data import build_datasets, Vocabulary
from data.dataset import _collate
from functools import partial
from torch.utils.data import DataLoader
from models import NAMER
from utils.metrics import _edit_dist

cfg = Config()
device = torch.device('cuda')
_, val_ds, _, vocab = build_datasets(cfg)
col = partial(_collate, pad_idx=vocab.pad_idx)
ldr = DataLoader(val_ds, 32, shuffle=False, collate_fn=col, num_workers=0)

model = NAMER(vocab_size=len(vocab), d=cfg.d_model, heads=cfg.nhead,
              pgd_layers=cfg.pgd_layers, drop=0.0).to(device)
ckpt = torch.load('checkpoints/ep020_0.05.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
ep = ckpt.get('epoch', '?')
er = ckpt.get('exprate', 0)
print(f'Loaded best.pth, epoch={ep}, exprate={er*100:.2f}%')

none_idx = vocab.none_idx
sos_idx = vocab.sos_idx
eos_idx = vocab.eos_idx
pad_idx = vocab.pad_idx

# ── Stage-by-stage analysis on first 500 samples ──
N_SAMPLES = 500
n_batches = (N_SAMPLES + 31) // 32

# Counters
vat_total_gt_tokens = 0
vat_total_detected = 0
vat_correct_detected = 0  # detected AND correct class
vat_missed = 0            # GT token not detected at all

# Token-level stats
gt_token_counts = {}
detected_token_counts = {}

# ExpRate per stage
exact_match_vat_only = 0   # just VAT sorted by column, no PGD
exact_match_full = 0       # full pipeline
total = 0
edit_dist_vat_only = []
edit_dist_full = []

# Length mismatch
len_mismatches = []

print(f'\nAnalyzing {N_SAMPLES} val samples...\n')

for bi, batch in enumerate(ldr):
    if bi >= n_batches:
        break
    imgs = batch['image'].to(device)
    B = imgs.size(0)
    
    with torch.no_grad():
        f8x, f16x = model.enc(imgs)
        probs, logits = model.vat(f8x, f16x)
        # Full pipeline
        results = model(imgs, token_ids=None)
    
    for b in range(B):
        total += 1
        gt_toks = batch['tokens'][b]
        gt_ids = batch['token_ids'][b]
        
        # GT tokens (excluding SOS, EOS, PAD)
        gt_valid = [t.item() for t in gt_ids if t.item() not in (sos_idx, eos_idx, pad_idx)]
        gt_set = {}
        for t in gt_valid:
            gt_set[t] = gt_set.get(t, 0) + 1
            
        vat_total_gt_tokens += len(gt_valid)
        
        # VAT predictions
        pred_map = probs[b].argmax(dim=0)  # [H, W]
        tok_mask = (pred_map != none_idx)
        pos = tok_mask.nonzero(as_tuple=False)
        
        if pos.size(0) > 0:
            col_order = pos[:, 1].argsort()
            pos = pos[col_order]
            det_ids = [pred_map[p[0], p[1]].item() for p in pos]
        else:
            det_ids = []
        
        vat_total_detected += len(det_ids)
        
        # Count correct detections (token exists in GT, matching count)
        det_set = {}
        for t in det_ids:
            det_set[t] = det_set.get(t, 0) + 1
        
        for t, cnt in gt_set.items():
            detected_cnt = det_set.get(t, 0)
            vat_correct_detected += min(cnt, detected_cnt)
            if detected_cnt < cnt:
                vat_missed += (cnt - detected_cnt)
        
        # VAT-only prediction (sorted by column, no PGD)
        vat_toks = [vocab.i2t.get(t, '?') for t in det_ids 
                    if t not in (sos_idx, eos_idx, pad_idx, none_idx)]
        
        # Full pipeline prediction
        full_toks = vocab.decode(results[b]) if b < len(results) else []
        
        # Edit distances
        d_vat = _edit_dist(vat_toks, gt_toks)
        d_full = _edit_dist(full_toks, gt_toks)
        edit_dist_vat_only.append(d_vat)
        edit_dist_full.append(d_full)
        
        if d_vat == 0: exact_match_vat_only += 1
        if d_full == 0: exact_match_full += 1
        
        # Length mismatch
        len_mismatches.append(len(det_ids) - len(gt_valid))
        
        # Token frequency
        for t in gt_toks:
            gt_token_counts[t] = gt_token_counts.get(t, 0) + 1
        for t in vat_toks:
            detected_token_counts[t] = detected_token_counts.get(t, 0) + 1

# ── Print results ──
print('='*60)
print('STAGE 1: VAT Detection Quality')
print('='*60)
recall = vat_correct_detected / max(vat_total_gt_tokens, 1)
precision = vat_correct_detected / max(vat_total_detected, 1)
print(f'  GT tokens (total):     {vat_total_gt_tokens}')
print(f'  Detected (total):      {vat_total_detected}')
print(f'  Correct detections:    {vat_correct_detected}')
print(f'  Missed GT tokens:     {vat_missed}')
print(f'  Recall:    {recall*100:.1f}%')
print(f'  Precision: {precision*100:.1f}%')
print(f'  Avg len mismatch:     {sum(len_mismatches)/len(len_mismatches):.1f} tokens')
print(f'  Median len mismatch:  {sorted(len_mismatches)[len(len_mismatches)//2]} tokens')

# Most missed tokens
print(f'\n  Most over-detected tokens (VAT detects more than GT):')
for t in sorted(gt_token_counts.keys(), 
                key=lambda t: detected_token_counts.get(t, 0) - gt_token_counts.get(t, 0), 
                reverse=True)[:10]:
    gt_c = gt_token_counts[t]
    det_c = detected_token_counts.get(t, 0)
    print(f'    {t:15s}: GT={gt_c:5d}  Det={det_c:5d}  diff={det_c-gt_c:+d}')

print(f'\n  Most under-detected tokens (VAT misses):')
for t in sorted(gt_token_counts.keys(), 
                key=lambda t: detected_token_counts.get(t, 0) - gt_token_counts.get(t, 0))[:10]:
    gt_c = gt_token_counts[t]
    det_c = detected_token_counts.get(t, 0)
    print(f'    {t:15s}: GT={gt_c:5d}  Det={det_c:5d}  diff={det_c-gt_c:+d}')

print(f'\n{"="*60}')
print('STAGE 2: Pipeline Comparison')
print('='*60)
avg_ed_vat = sum(edit_dist_vat_only) / total
avg_ed_full = sum(edit_dist_full) / total
print(f'  VAT-only (no PGD):')
print(f'    ExpRate:         {exact_match_vat_only/total*100:.2f}%')
print(f'    Avg edit dist:   {avg_ed_vat:.2f}')
print(f'  Full pipeline (VAT+PGD+path_sel):')
print(f'    ExpRate:         {exact_match_full/total*100:.2f}%')
print(f'    Avg edit dist:   {avg_ed_full:.2f}')
print(f'  PGD improvement:   {avg_ed_vat - avg_ed_full:+.2f} edit dist')

# Distribution of edit distances
print(f'\n  Edit distance distribution (full pipeline):')
for thresh in [0, 1, 2, 3, 5, 10]:
    cnt = sum(1 for d in edit_dist_full if d <= thresh)
    print(f'    d<={thresh:2d}: {cnt:4d} ({cnt/total*100:.1f}%)')

print(f'\n  Edit distance distribution (VAT-only):')
for thresh in [0, 1, 2, 3, 5, 10]:
    cnt = sum(1 for d in edit_dist_vat_only if d <= thresh)
    print(f'    d<={thresh:2d}: {cnt:4d} ({cnt/total*100:.1f}%)')
