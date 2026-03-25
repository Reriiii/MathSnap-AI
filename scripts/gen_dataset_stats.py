"""Generate real dataset-stats.json from CoMER training data."""
import json
import os
from collections import Counter, defaultdict

DATA_DIR = "E:/Workspace/CoMER/data/data"
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "app", "data", "dataset-stats.json")

# Token categories
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

TOKEN_TO_CAT = {}
for cat, toks in TOKEN_CATEGORIES.items():
    for t in toks:
        TOKEN_TO_CAT[t] = cat


def get_category(token):
    if token in TOKEN_TO_CAT:
        return TOKEN_TO_CAT[token]
    if token.startswith("\\"):
        return "Symbols"
    if len(token) == 1 and token.isalpha():
        return "Uppercase" if token.isupper() else "Lowercase"
    return "Other"


def read_captions(path):
    """Read caption.txt: filename<TAB>tok1 tok2 ..."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            samples.append({"name": parts[0], "tokens": parts[1].split()})
    return samples


def main():
    split_files = {
        "train": os.path.join(DATA_DIR, "train", "caption.txt"),
        "val_2014": os.path.join(DATA_DIR, "2014", "caption.txt"),
        "val_2016": os.path.join(DATA_DIR, "2016", "caption.txt"),
        "val_2019": os.path.join(DATA_DIR, "2019", "caption.txt"),
    }

    formulas = []  # (dashboard_split, source, tokens)
    freq_total = Counter()
    freq_per_split = {"train": Counter(), "val": Counter()}
    seq_lengths = {"train": [], "val": []}

    for key, path in split_files.items():
        dashboard_split = "train" if key == "train" else "val"
        source = "CROHME Train" if key == "train" else f"CROHME {key.split('_')[1]}"
        samples = read_captions(path)
        for s in samples:
            toks = s["tokens"]
            formulas.append((dashboard_split, source, toks))
            freq_total.update(toks)
            freq_per_split[dashboard_split].update(toks)
            seq_lengths[dashboard_split].append(len(toks))

    total = len(formulas)
    train_count = len(seq_lengths["train"])
    val_count = len(seq_lengths["val"])

    print(f"Total: {total} samples ({train_count} train, {val_count} val)")
    print(f"Unique tokens: {len(freq_total)}")

    # 1. Split distribution
    split_dist = [
        {"name": "train", "count": train_count,
         "percentage": round(train_count / total * 100, 1)},
        {"name": "val", "count": val_count,
         "percentage": round(val_count / total * 100, 1)},
    ]

    # 2. Token frequency
    token_freq = []
    for token, cnt in freq_total.most_common():
        token_freq.append({
            "token": token,
            "total": cnt,
            "train": freq_per_split["train"].get(token, 0),
            "val": freq_per_split["val"].get(token, 0),
            "category": get_category(token),
        })

    # 3. Sequence length histogram
    bins = [
        (1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30),
        (31, 40), (41, 50), (51, 75), (76, 100), (101, 200), (201, 300),
    ]

    seq_hist = []
    for lo, hi in bins:
        entry = {"bin": f"{lo}-{hi}"}
        for s in ["train", "val"]:
            entry[s] = sum(1 for l in seq_lengths[s] if lo <= l <= hi)
        seq_hist.append(entry)

    def calc_stats(lengths):
        if not lengths:
            return {}
        s = sorted(lengths)
        n = len(s)
        return {
            "min": s[0], "max": s[-1],
            "mean": round(sum(s) / n, 1),
            "median": s[n // 2],
            "p25": s[int(n * 0.25)],
            "p75": s[int(n * 0.75)],
            "p95": s[int(n * 0.95)],
        }

    seq_stats = []
    for s in ["train", "val"]:
        st = calc_stats(seq_lengths[s])
        st["split"] = s
        seq_stats.append(st)
    all_len = seq_lengths["train"] + seq_lengths["val"]
    overall = calc_stats(all_len)
    overall["split"] = "all"
    seq_stats.append(overall)

    # 4. Token categories
    cat_summary = defaultdict(lambda: {"tokens": set(), "totalFrequency": 0})
    for tf in token_freq:
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

    # 5. Dataset sources
    source_counter = defaultdict(lambda: defaultdict(int))
    for split, source, _ in formulas:
        source_counter[source][split] += 1

    dataset_sources = []
    for source in sorted(source_counter.keys()):
        for split in ["train", "val"]:
            count = source_counter[source].get(split, 0)
            if count > 0:
                dataset_sources.append({
                    "source": source, "count": count, "split": split,
                })

    # 6. Complexity metrics
    constructs = ["\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\log"]
    construct_counts = {c: 0 for c in constructs}
    unique_tokens_list = []
    nesting_list = []

    for _, _, toks in formulas:
        unique_tokens_list.append(len(set(toks)))
        depth = max_d = 0
        for t in toks:
            if t == "{":
                depth += 1
                max_d = max(max_d, depth)
            elif t == "}":
                depth = max(0, depth - 1)
        nesting_list.append(max_d)
        for c in constructs:
            if c in toks:
                construct_counts[c] += 1

    n = len(formulas)
    complexity = {
        "avgUniqueTokens": round(sum(unique_tokens_list) / n, 1),
        "avgNestingDepth": round(sum(nesting_list) / n, 2),
        "maxNestingDepth": max(nesting_list),
        "constructUsage": [
            {
                "construct": c,
                "count": construct_counts[c],
                "percentage": round(construct_counts[c] / n * 100, 1),
            }
            for c in constructs
        ],
    }

    # Assemble
    result = {
        "totalSamples": total,
        "splits": split_dist,
        "tokenFrequency": token_freq,
        "sequenceLength": {"histogram": seq_hist, "stats": seq_stats},
        "tokenCategories": token_categories,
        "datasetSources": dataset_sources,
        "complexityMetrics": complexity,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Written to {OUTPUT}")
    print(f"Seq length: mean={overall['mean']}, median={overall['median']}, max={overall['max']}")
    print(f"Nesting: avg={complexity['avgNestingDepth']}, max={complexity['maxNestingDepth']}")
    print(f"Token categories: {[c['category'] + ':' + str(c['tokenCount']) for c in token_categories]}")


if __name__ == "__main__":
    main()
