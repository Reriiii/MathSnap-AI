"""
SymLG (Symbol Level Label Graph) parser and LaTeX converter.

Parses .lg files from CROHME dataset and converts the symbol graph
representation into LaTeX strings for sequence-to-sequence training.
"""

import os
import re
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_symlg(filepath: str) -> dict:
    """
    Parse a SymLG (.lg) file into structured data.

    Returns:
        dict with keys:
            - 'objects': dict mapping id -> label
            - 'relations': list of (src_id, dst_id, relation_type)
            - 'iud': the expression identifier
    """
    objects = {}
    relations = []
    iud = ""

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # IUD identifier
            if line.startswith("# IUD,"):
                iud = line.split(",", 1)[1].strip()
                continue

            # Skip other comments
            if line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            if parts[0] == "O" and len(parts) >= 4:
                # Object: O, <id>, <label>, <confidence>, <spatial_chain>
                obj_id = parts[1]
                label = parts[2]
                objects[obj_id] = label

            elif parts[0] == "R" and len(parts) >= 5:
                # Relation: R, <src_id>, <dst_id>, <relation_type>, <confidence>
                src_id = parts[1]
                dst_id = parts[2]
                rel_type = parts[3]
                relations.append((src_id, dst_id, rel_type))

    return {
        'objects': objects,
        'relations': relations,
        'iud': iud
    }


def symlg_to_latex(parsed: dict) -> str:
    """
    Convert parsed SymLG data to a LaTeX token string.

    Builds a tree from the relation graph and traverses it to produce
    a space-separated LaTeX token sequence.
    """
    objects = parsed['objects']
    relations = parsed['relations']

    if not objects:
        return ""

    # Build adjacency: for each node, store its children by relation type
    # children[node_id] = {'Right': [...], 'Sup': [...], 'Sub': [...], 'Above': [...], 'Below': [...]}
    children = defaultdict(lambda: defaultdict(list))
    child_set = set()  # track which nodes appear as children

    for src_id, dst_id, rel_type in relations:
        children[src_id][rel_type].append(dst_id)
        child_set.add(dst_id)

    # Find root: the node that never appears as a child
    all_nodes = set(objects.keys())
    roots = all_nodes - child_set

    if not roots:
        # Fallback: pick the first object
        root = list(objects.keys())[0]
    else:
        root = list(roots)[0]

    # Check if this is a fraction expression
    # Fractions are indicated by a '-' symbol with Above and Below relations
    def _is_fraction_bar(node_id):
        label = objects.get(node_id, "")
        node_children = children.get(node_id, {})
        return label == "-" and ("Above" in node_children or "Below" in node_children)

    def _traverse(node_id, visited=None):
        """Recursively traverse the symbol tree to build LaTeX tokens."""
        if visited is None:
            visited = set()

        if node_id in visited:
            return []

        visited.add(node_id)

        tokens = []
        node_children = children.get(node_id, {})
        label = objects.get(node_id, "")

        # Check if this node is a fraction bar
        if _is_fraction_bar(node_id):
            tokens.append("\\frac")
            # Above = numerator
            tokens.append("{")
            for above_id in node_children.get("Above", []):
                tokens.extend(_traverse(above_id, visited))
            tokens.append("}")
            # Below = denominator
            tokens.append("{")
            for below_id in node_children.get("Below", []):
                tokens.extend(_traverse(below_id, visited))
            tokens.append("}")
        else:
            # Regular symbol
            tokens.append(label)

        # Handle superscript
        if "Sup" in node_children:
            tokens.append("^")
            tokens.append("{")
            for sup_id in node_children["Sup"]:
                tokens.extend(_traverse(sup_id, visited))
            tokens.append("}")

        # Handle subscript
        if "Sub" in node_children:
            tokens.append("_")
            tokens.append("{")
            for sub_id in node_children["Sub"]:
                tokens.extend(_traverse(sub_id, visited))
            tokens.append("}")

        # Handle right continuation
        if "Right" in node_children:
            for right_id in node_children["Right"]:
                tokens.extend(_traverse(right_id, visited))

        return tokens

    tokens = _traverse(root)

    # Clean up: remove empty braces, normalize spacing
    return " ".join(tokens)


def _find_matching_image(lg_path: Path, img_base_dir: Path, split: str) -> Optional[Path]:
    """
    Find the corresponding image for a .lg file.
    Images and labels share the same filename stem but may be in different subdirectories.
    """
    stem = lg_path.stem
    # Search in all subdirectories of img_base_dir/split/
    split_dir = img_base_dir / split
    if not split_dir.exists():
        return None

    for img_path in split_dir.rglob(f"{stem}.png"):
        return img_path

    return None


def preprocess_dataset(
    raw_dir: str = "dataset/raw",
    output_dir: str = "dataset/processed",
    exclude_artificial: bool = True
) -> Dict[str, List[dict]]:
    """
    Process all SymLG files and create CSV files mapping images to LaTeX labels.

    Args:
        raw_dir: path to raw dataset directory
        output_dir: path to output processed data
        exclude_artificial: if True, skip Artificial_data subdirectories

    Returns:
        dict mapping split name -> list of {image_path, latex} entries
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    symlg_dir = raw_path / "SymLG"
    img_dir = raw_path / "IMG"

    result = {}

    for split in ["train", "val", "test"]:
        split_symlg = symlg_dir / split
        if not split_symlg.exists():
            continue

        entries = []
        errors = []

        # Find all .lg files recursively
        lg_files = sorted(split_symlg.rglob("*.lg"))

        for lg_file in lg_files:
            # Skip artificial data if requested
            if exclude_artificial and "Artificial_data" in str(lg_file):
                continue

            # Find matching image
            img_path = _find_matching_image(lg_file, img_dir, split)
            if img_path is None:
                errors.append(f"No image found for: {lg_file}")
                continue

            # Parse SymLG and convert to LaTeX
            try:
                parsed = parse_symlg(str(lg_file))
                latex = symlg_to_latex(parsed)

                if not latex.strip():
                    errors.append(f"Empty LaTeX for: {lg_file}")
                    continue

                entries.append({
                    'image_path': str(img_path),
                    'latex': latex
                })
            except Exception as e:
                errors.append(f"Error processing {lg_file}: {e}")

        # Write CSV
        csv_path = out_path / f"{split}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'latex'])
            writer.writeheader()
            writer.writerows(entries)

        # Write errors log
        if errors:
            err_path = out_path / f"{split}_errors.log"
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(errors))

        result[split] = entries
        print(f"[{split}] Processed {len(entries)} samples, {len(errors)} errors")

    return result


def collect_all_tokens(csv_path: str) -> List[str]:
    """Read a processed CSV and collect all unique LaTeX tokens."""
    tokens = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for tok in row['latex'].split():
                tokens.add(tok)
    return sorted(tokens)


if __name__ == "__main__":
    # Quick test: parse and convert a single file
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "dataset/raw/SymLG/train/CROHME2019_train/001-equation000.lg"

    parsed = parse_symlg(filepath)
    print(f"IUD: {parsed['iud']}")
    print(f"Objects ({len(parsed['objects'])}):")
    for obj_id, label in parsed['objects'].items():
        print(f"  {obj_id} -> {label}")
    print(f"\nRelations ({len(parsed['relations'])}):")
    for src, dst, rel in parsed['relations']:
        print(f"  {src} --{rel}--> {dst}")

    latex = symlg_to_latex(parsed)
    print(f"\nLaTeX: {latex}")

    # Full preprocessing
    print("\n--- Full Preprocessing ---")
    preprocess_dataset()
