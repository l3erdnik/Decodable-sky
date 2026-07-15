#!/usr/bin/env python
"""
Decodable Sky -- object x prompt table of a single PCA coordinate at one layer.

For a chosen model, layer, and principal component, write a table whose ROWS are the
objects and COLUMNS are the prompts, with each cell the value of that PC in the
last-token residual stream for (object, prompt). Reads the <model>_pca128.npz produced
by extract_pca.py; the prompt file supplies the column labels (and must be the one used
at extraction, so the prompt indices line up).

Usage
-----
    python pc_table.py --model mistrallarge123b --layer 80 --pc 2
    python pc_table.py --model qwen235b --layer 94 --pc 1 --indir PCA128

Output: <out>/<model>_L<layer>_PC<pc>.csv
    columns: object, type, then one column per prompt ("p00: <text>", ...)
"""
import argparse
import csv
import os

import numpy as np


def load_prompts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().strip('"')
            if s:
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model key / file prefix (reads <model>_pca128.npz)")
    ap.add_argument("--layer", type=int, required=True, help="hidden-state index (0 = embeddings)")
    ap.add_argument("--pc", type=int, default=1, help="principal component, 1-based (default 1)")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--prompts", default="data/astro_prompts_location.csv")
    ap.add_argument("--out", default=".", help="output folder")
    args = ap.parse_args()

    d = np.load(os.path.join(args.indir, f"{args.model}_pca128.npz"), allow_pickle=True)
    pca = d["pca"]
    n_layers, n_pc = pca.shape[1], pca.shape[2]
    if not 0 <= args.layer < n_layers:
        ap.error(f"--layer must be in [0, {n_layers - 1}] for this model")
    if not 1 <= args.pc <= n_pc:
        ap.error(f"--pc must be in [1, {n_pc}]")

    obj = d["obj_ids"].astype(int)
    pr = d["pr_ids"].astype(int)
    names = d["names"].astype(str)
    types = d["types"].astype(str)
    prompts = load_prompts(args.prompts)
    n_obj, n_prompt = len(names), len(prompts)

    val = pca[:, args.layer, args.pc - 1].astype(np.float32)   # PC value per sample
    mat = np.full((n_obj, n_prompt), np.nan, np.float32)
    for s in range(len(obj)):
        mat[obj[s], pr[s]] = val[s]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.model}_L{args.layer}_PC{args.pc}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object", "type"] + [f"p{i:02d}: {prompts[i]}" for i in range(n_prompt)])
        for j in range(n_obj):
            w.writerow([names[j], types[j]] + [round(float(mat[j, i]), 6) for i in range(n_prompt)])
    print(f"saved {out_path}  ({n_obj} objects x {n_prompt} prompts)")


if __name__ == "__main__":
    main()
