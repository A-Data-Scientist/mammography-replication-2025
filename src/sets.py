#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split images into Train / Val / Test folders.

Supports:
A) Subfolder-labeled input (input_dir/<label>/*.png)
B) CSV-labeled input (filename_col, label_col), with optional group_col to avoid patient leakage

Outputs:
output_dir/
  train/<label>/*.png
  val/<label>/*.png
  test/<label>/*.png

Example (CSV + groups):
python split_images.py ^
  --input_dir ../data/mini-MIAS/images_png ^
  --csv ../data/mini-MIAS/labels.csv ^
  --filename_col filename --label_col label --group_col patient_id ^
  --ext .png ^
  --output_dir ../data/mini-MIAS/splits ^
  --train 0.7 --val 0.15 --test 0.15 --seed 42 --copy

Example (subfolders):
python split_images.py ^
  --input_dir ../data/mini-MIAS/images_by_class ^
  --output_dir ../data/mini-MIAS/splits ^
  --train 0.7 --val 0.15 --test 0.15 --seed 42 --copy
"""

import argparse, os, sys, shutil, glob, random
from collections import defaultdict, Counter
from pathlib import Path

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def copy_or_move(src, dst, move=False):
    safe_mkdir(os.path.dirname(dst))
    if move:
        shutil.move(src, dst)
    else:
        # copy2 keeps timestamps; ok to use copy if you want speed
        shutil.copy2(src, dst)

def normalize_ext(name, ext):
    # Ensure expected extension, lowercased
    p = Path(name)
    if ext and p.suffix.lower() != ext.lower():
        return (p.stem + ext.lower())
    return p.name.lower()

def stratified_split(items, labels, train_ratio, val_ratio, test_ratio, seed=42):
    """
    items: list of IDs
    labels: list of class labels, same length
    Returns: train_ids, val_ids, test_ids
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"
    random.seed(seed)

    # group items by label
    by_label = defaultdict(list)
    for it, lab in zip(items, labels):
        by_label[lab].append(it)

    train, val, test = set(), set(), set()
    for lab, lab_items in by_label.items():
        n = len(lab_items)
        random.shuffle(lab_items)
        n_train = int(round(train_ratio * n))
        n_val   = int(round(val_ratio * n))
        # remainder to test
        n_test  = max(0, n - n_train - n_val)
        train.update(lab_items[:n_train])
        val.update(lab_items[n_train:n_train+n_val])
        test.update(lab_items[n_train+n_val:n_train+n_val+n_test])

    return list(train), list(val), list(test)

def grouped_split(rows, group_key, label_key, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Group-aware split (e.g., by patient_id).
    rows: list of dicts { 'group': ..., 'label': ..., 'file': ... }
    Returns index sets for train/val/test rows.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    random.seed(seed)

    # collect groups with labels (use majority label per group for stratification)
    group_to_files = defaultdict(list)
    group_to_labels = defaultdict(list)
    for i, r in enumerate(rows):
        g = r[group_key]
        group_to_files[g].append(i)
        group_to_labels[g].append(r[label_key])

    # derive a single label per group (majority)
    group_labels = {}
    for g, labs in group_to_labels.items():
        most_common = Counter(labs).most_common(1)[0][0]
        group_labels[g] = most_common

    # stratify groups
    groups = list(group_to_files.keys())
    g_labels = [group_labels[g] for g in groups]
    train_g, val_g, test_g = stratified_split(groups, g_labels, train_ratio, val_ratio, test_ratio, seed)

    # expand to row indices
    train_idx, val_idx, test_idx = set(), set(), set()
    for g in train_g:
        train_idx.update(group_to_files[g])
    for g in val_g:
        val_idx.update(group_to_files[g])
    for g in test_g:
        test_idx.update(group_to_files[g])

    return train_idx, val_idx, test_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing images (flat or class-subfolders)")
    ap.add_argument("--csv", default=None, help="Optional CSV with filename/label (and optional group)")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--group_col", default=None, help="Optional group column to avoid leakage (e.g., patient_id)")
    ap.add_argument("--ext", default=".png", help="Expected image extension when using CSV (e.g., .png)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    ap.add_argument("--limit_per_class", type=int, default=None, help="Optional cap for quick smoke tests")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    rows = []  # list of dicts: {file,label,group?,src_path}
    labels_set = set()

    if args.csv:
        # CSV-driven collection
        import csv
        csv_path = Path(args.csv)
        if not csv_path.exists():
            eprint(f"[ERR] CSV not found: {csv_path}")
            sys.exit(2)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = normalize_ext(r[args.filename_col], args.ext)
                label = str(r[args.label_col]).strip()
                group = r[args.group_col].strip() if args.group_col and r.get(args.group_col) is not None else None
                # look for file in input_dir (case-insensitive)
                cand = input_dir / fname
                if not cand.exists():
                    # try to find case-insensitively
                    matches = list(input_dir.glob(fname.lower()))
                    if matches:
                        cand = matches[0]
                if not cand.exists():
                    # last resort: linear search by stem
                    stem = Path(fname).stem.lower()
                    m = [p for p in input_dir.glob("**/*") if p.is_file() and p.stem.lower() == stem]
                    if m:
                        cand = m[0]
                if not cand.exists():
                    eprint(f"[WARN] Missing file for CSV row: {fname} (label={label})")
                    continue
                rows.append({"file": cand, "label": label, "group": group})
                labels_set.add(label)
    else:
        # Subfolder-labeled collection
        for lab_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
            label = lab_dir.name
            files = list(lab_dir.glob("**/*"))
            count = 0
            for p in files:
                if p.is_file():
                    rows.append({"file": p, "label": label, "group": None})
                    count += 1
            if count > 0:
                labels_set.add(label)

    if not rows:
        eprint("[ERR] No images found. Check input_dir / CSV / extensions.")
        sys.exit(2)

    # Optional cap for quick tests
    if args.limit_per_class is not None:
        by_lab = defaultdict(list)
        for i, r in enumerate(rows):
            by_lab[r["label"]].append(i)
        keep_idx = set()
        for lab, idxs in by_lab.items():
            random.Random(args.seed).shuffle(idxs)
            keep_idx.update(idxs[:args.limit_per_class])
        rows = [rows[i] for i in sorted(keep_idx)]

    # Build splits
    if args.group_col:
        # Require group for each row; fill None with filename stem fallback
        for r in rows:
            if not r["group"]:
                r["group"] = r["file"].stem  # fallback to filename stem
        train_idx, val_idx, test_idx = grouped_split(
            rows, "group", "label", args.train, args.val, args.test, args.seed
        )
        train_rows = [rows[i] for i in sorted(train_idx)]
        val_rows   = [rows[i] for i in sorted(val_idx)]
        test_rows  = [rows[i] for i in sorted(test_idx)]
    else:
        items = list(range(len(rows)))
        labels = [r["label"] for r in rows]
        tr, va, te = stratified_split(items, labels, args.train, args.val, args.test, args.seed)
        train_rows = [rows[i] for i in tr]
        val_rows   = [rows[i] for i in va]
        test_rows  = [rows[i] for i in te]

    # Copy/move files
    def dump(split_name, split_rows):
        for r in split_rows:
            label = r["label"]
            src = str(r["file"])
            dst = str(output_dir / split_name / label / Path(src).name.lower())
            copy_or_move(src, dst, move=args.move)

    dump("train", train_rows)
    dump("val",   val_rows)
    dump("test",  test_rows)

    # Report
    def stats(name, split_rows):
        c = Counter([r["label"] for r in split_rows])
        return f"{name}: total={len(split_rows)} " + ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
    print(stats("train", train_rows))
    print(stats("val",   val_rows))
    print(stats("test",  test_rows))

    # Sanity: no overlap
    train_set = set([str(r["file"]).lower() for r in train_rows])
    val_set   = set([str(r["file"]).lower() for r in val_rows])
    test_set  = set([str(r["file"]).lower() for r in test_rows])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    print("No overlap between splits. Done.")

if __name__ == "__main__":
    main()
