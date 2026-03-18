"""
Analyze CROHME dataset and generate statistics JSON for the dashboard.
Reads train/val/test CSVs, computes token frequencies, sequence lengths,
category breakdowns, dataset sources, and formula complexity.
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "src" / "app" / "data" / "dataset-stats.json"
TRAINING_OUTPUT_PATH = PROJECT_ROOT / "src" / "app" / "data" / "training-metrics.json"

# ============================================================
# Token category mapping
# ============================================================
TOKEN_CATEGORIES = {
    "Digits": list("0123456789"),
    "Lowercase": list("abcdefghijklmnopqrstuvwxyz"),
    "Uppercase": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "Greek": [
        "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta",
        "\\lambda", "\\mu", "\\phi", "\\pi", "\\sigma",
        "\\Delta", "\\Pi", "\\omega", "\\rho", "\\epsilon",
        "\\zeta", "\\eta", "\\iota", "\\kappa", "\\nu",
        "\\xi", "\\tau", "\\upsilon", "\\chi", "\\psi",
    ],
    "Operators": [
        "+", "-", "=", "<", ">", "\\cdot", "\\div", "\\times",
        "\\pm", "\\neq", "\\geq", "\\leq", "\\lt", "\\gt",
        "\\approx", "\\equiv", "\\propto",
    ],
    "Functions": [
        "\\sin", "\\cos", "\\tan", "\\log", "\\lim",
        "\\exp", "\\ln", "\\max", "\\min",
    ],
    "Structural": [
        "\\frac", "\\sqrt", "^", "_", "{", "}", "\\limits",
        "\\prime", "\\hat", "\\bar", "\\dot", "\\vec",
        "\\overline", "\\underline", "\\widetilde",
    ],
    "Delimiters": [
        "(", ")", "[", "]", "\\{", "\\}", "|",
        "\\lfloor", "\\rfloor", "\\lceil", "\\rceil",
        "\\left", "\\right",
    ],
    "Symbols": [
        "\\infty", "\\in", "\\rightarrow", "\\exists", "\\forall",
        "\\ldots", "\\cdots", "!", ".", ",", ";", ":",
        "\\sum", "\\int", "\\prod", "\\cup", "\\cap",
        "\\subset", "\\supset", "\\emptyset", "\\partial",
        "\\nabla", "\\perp", "\\angle", "\\triangle",
    ],
}

# Build reverse mapping: token -> category
TOKEN_TO_CATEGORY = {}
for cat, tokens in TOKEN_CATEGORIES.items():
    for t in tokens:
        TOKEN_TO_CATEGORY[t] = cat


def get_category(token: str) -> str:
    if token in TOKEN_TO_CATEGORY:
        return TOKEN_TO_CATEGORY[token]
    if token.startswith("\\"):
        return "Symbols"
    if len(token) == 1 and token.isalpha():
        return "Uppercase" if token.isupper() else "Lowercase"
    return "Other"


# ============================================================
# Read CSVs
# ============================================================
def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyze():
    splits = {}
    all_formulas = []

    for split_name in ["train", "val", "test"]:
        csv_path = DATA_DIR / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        rows = read_csv(csv_path)
        splits[split_name] = rows
        all_formulas.extend([(split_name, r) for r in rows])
        print(f"  {split_name}: {len(rows)} samples")

    total = sum(len(v) for v in splits.values())
    print(f"  Total: {total} samples\n")

    # --------------------------------------------------------
    # 1. Split distribution
    # --------------------------------------------------------
    split_dist = []
    for name in ["train", "val", "test"]:
        count = len(splits.get(name, []))
        split_dist.append({
            "name": name,
            "count": count,
            "percentage": round(count / total * 100, 1),
        })

    # --------------------------------------------------------
    # 2. Token frequency
    # --------------------------------------------------------
    freq_total = Counter()
    freq_per_split = {s: Counter() for s in splits}

    seq_lengths_per_split = {s: [] for s in splits}

    for split_name, row in all_formulas:
        latex = row.get("latex", "")
        tokens = latex.strip().split()
        freq_total.update(tokens)
        freq_per_split[split_name].update(tokens)
        seq_lengths_per_split[split_name].append(len(tokens))

    token_freq_list = []
    for token, total_count in freq_total.most_common():
        token_freq_list.append({
            "token": token,
            "total": total_count,
            "train": freq_per_split.get("train", {}).get(token, 0),
            "val": freq_per_split.get("val", {}).get(token, 0),
            "test": freq_per_split.get("test", {}).get(token, 0),
            "category": get_category(token),
        })

    # --------------------------------------------------------
    # 3. Sequence length distribution
    # --------------------------------------------------------
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30),
            (31, 40), (41, 50), (51, 75), (76, 100), (101, 200)]
    bin_labels = [f"{a}-{b}" for a, b in bins]

    seq_histogram = []
    for i, (lo, hi) in enumerate(bins):
        entry = {"bin": bin_labels[i]}
        for s in ["train", "val", "test"]:
            lengths = seq_lengths_per_split.get(s, [])
            entry[s] = sum(1 for l in lengths if lo <= l <= hi)
        seq_histogram.append(entry)

    def calc_stats(lengths):
        if not lengths:
            return {}
        s = sorted(lengths)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "mean": round(sum(s) / n, 1),
            "median": s[n // 2],
            "p25": s[int(n * 0.25)],
            "p75": s[int(n * 0.75)],
            "p95": s[int(n * 0.95)],
        }

    seq_stats = []
    for s in ["train", "val", "test"]:
        stats = calc_stats(seq_lengths_per_split.get(s, []))
        stats["split"] = s
        seq_stats.append(stats)

    all_lengths = []
    for v in seq_lengths_per_split.values():
        all_lengths.extend(v)
    overall_stats = calc_stats(all_lengths)
    overall_stats["split"] = "all"
    seq_stats.append(overall_stats)

    # --------------------------------------------------------
    # 4. Token categories
    # --------------------------------------------------------
    cat_summary = defaultdict(lambda: {"tokens": set(), "totalFrequency": 0})
    for tf in token_freq_list:
        cat = tf["category"]
        cat_summary[cat]["tokens"].add(tf["token"])
        cat_summary[cat]["totalFrequency"] += tf["total"]

    token_categories = []
    for cat in ["Digits", "Lowercase", "Uppercase", "Greek", "Operators",
                "Functions", "Structural", "Delimiters", "Symbols", "Other"]:
        if cat in cat_summary:
            token_categories.append({
                "category": cat,
                "tokenCount": len(cat_summary[cat]["tokens"]),
                "totalFrequency": cat_summary[cat]["totalFrequency"],
                "tokens": sorted(cat_summary[cat]["tokens"]),
            })

    # --------------------------------------------------------
    # 5. Dataset sources
    # --------------------------------------------------------
    source_counter = defaultdict(lambda: defaultdict(int))
    for split_name, row in all_formulas:
        img_path = row.get("image_path", "")
        # Extract CROHME year from path
        match = re.search(r"CROHME(\d{4})", img_path)
        if match:
            source = f"CROHME {match.group(1)}"
        else:
            source = "Unknown"
        source_counter[source][split_name] += 1

    dataset_sources = []
    for source in sorted(source_counter.keys()):
        for split in ["train", "val", "test"]:
            count = source_counter[source].get(split, 0)
            if count > 0:
                dataset_sources.append({
                    "source": source,
                    "count": count,
                    "split": split,
                })

    # --------------------------------------------------------
    # 6. Complexity metrics
    # --------------------------------------------------------
    constructs = ["\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\log"]
    construct_counts = {c: 0 for c in constructs}
    total_unique_tokens = []
    total_nesting = []

    for _, row in all_formulas:
        latex = row.get("latex", "")
        tokens = latex.strip().split()
        unique = len(set(tokens))
        total_unique_tokens.append(unique)

        # Nesting depth
        depth = 0
        max_depth = 0
        for t in tokens:
            if t == "{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif t == "}":
                depth = max(0, depth - 1)
        total_nesting.append(max_depth)

        for c in constructs:
            if c in tokens:
                construct_counts[c] += 1

    n_formulas = len(all_formulas)
    complexity = {
        "avgUniqueTokens": round(sum(total_unique_tokens) / max(n_formulas, 1), 1),
        "avgNestingDepth": round(sum(total_nesting) / max(n_formulas, 1), 2),
        "maxNestingDepth": max(total_nesting) if total_nesting else 0,
        "constructUsage": [
            {
                "construct": c,
                "count": construct_counts[c],
                "percentage": round(construct_counts[c] / max(n_formulas, 1) * 100, 1),
            }
            for c in constructs
        ],
    }

    # --------------------------------------------------------
    # Assemble output
    # --------------------------------------------------------
    result = {
        "totalSamples": total,
        "splits": split_dist,
        "tokenFrequency": token_freq_list,
        "sequenceLength": {
            "histogram": seq_histogram,
            "stats": seq_stats,
        },
        "tokenCategories": token_categories,
        "datasetSources": dataset_sources,
        "complexityMetrics": complexity,
    }

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Dataset stats written to {OUTPUT_PATH}")
    return result


# ============================================================
# Generate training metrics from real outputs/history.json
# ============================================================
HISTORY_PATH = PROJECT_ROOT / "outputs" / "history.json"


def generate_training_metrics():
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_losses = history["train_loss"]
    lrs = history["lr"]
    trained_epochs = len(train_losses)

    # Average val loss and exprate across CROHME 2014/2016/2019
    val_loss_keys = ["2014_val_loss", "2016_val_loss", "2019_val_loss"]
    exprate_keys = ["2014_exprate", "2016_exprate", "2019_exprate"]

    # Detect LR change events
    lr_events = [{"epoch": 1, "newLr": lrs[0], "reason": "Initial"}]
    for i in range(1, len(lrs)):
        if lrs[i] != lrs[i - 1]:
            lr_events.append({
                "epoch": i + 1,
                "newLr": lrs[i],
                "reason": "Plateau detected",
            })

    # Build per-epoch records
    epochs = []
    best_epoch = 1
    best_exprate = 0.0

    for i in range(trained_epochs):
        val_loss = sum(history[k][i] for k in val_loss_keys) / len(val_loss_keys)
        exprate = sum(history[k][i] for k in exprate_keys) / len(exprate_keys)

        if exprate > best_exprate:
            best_exprate = exprate
            best_epoch = i + 1

        epochs.append({
            "epoch": i + 1,
            "trainLoss": round(train_losses[i], 4),
            "valLoss": round(val_loss, 4),
            "expRate": round(exprate, 2),
            "lr": lrs[i],
        })

    best_exprate = round(best_exprate, 2)

    result = {
        "epochs": epochs,
        "bestEpoch": best_epoch,
        "bestExpRate": best_exprate,
        "totalEpochs": trained_epochs,
        "trainedEpochs": trained_epochs,
        "lrEvents": lr_events,
    }

    os.makedirs(TRAINING_OUTPUT_PATH.parent, exist_ok=True)
    with open(TRAINING_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Training metrics written to {TRAINING_OUTPUT_PATH}")
    print(f"  {trained_epochs} epochs, best ExpRate: {best_exprate}% at epoch {best_epoch}")


if __name__ == "__main__":
    print("=== Analyzing CROHME Dataset ===")
    analyze()
    print("\n=== Generating Training Metrics ===")
    generate_training_metrics()
    print("\nDone!")
