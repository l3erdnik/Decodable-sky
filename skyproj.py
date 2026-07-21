#!/usr/bin/env python
"""
Decodable Sky -- optimal linear projection of residual-stream PCA onto the sky.

For one model at one layer, take the top-N PCA directions and fit an ordinary
least-squares map onto the raw 3-D sky unit vector (sinDec, cosDec sinRA,
cosDec cosRA). This is the "best linear read-out" of sky position at that layer;
it is an in-sample fit (no leave-one-out) -- the point is the projection itself,
not held-out validation (see correlations.py sky_r2 for the LOO/R^2 story).

Target handling (important)
---------------------------
The regression has an INTERCEPT and the target is the RAW unit vector -- it is
NOT centered or whitened. Centering is absorbed by the intercept and reversed in
the prediction, so the fitted px,py,pz live in the ORIGINAL sky-coordinate space
and are directly comparable to the target tx,ty,tz. (If the target were whitened
for the fit, the predictions would have to be un-whitened before saving; here
there is nothing to undo.)

Outputs (into --out, one row per (object, prompt) sample / per object)
----------------------------------------------------------------------
  <model>_skyproj_coords_L<L>.csv     object, type, prompt_id, px, py, pz, tx, ty, tz
  <model>_skyproj_objstats_L<L>.csv   object, type, n_prompts, var, mse
      With predictions normalized to the unit sphere (p_hat = p / |p|):
        var = mean_i |p_hat_i - mean_i(p_hat)|^2   (prompt-to-prompt spread)
        mse = mean_i |p_hat_i - t|^2               (squared chord error to truth)
"""
import argparse
import csv
import os

import numpy as np


def fit_projection(pca, obj_ids, yunit, layer, n):
    """In-sample OLS (with intercept) from top-n PCA at `layer` onto raw unit vectors."""
    from sklearn.linear_model import LinearRegression
    z = pca[:, layer, :n].astype(np.float64)
    target = yunit[obj_ids].astype(np.float64)              # raw unit vector per sample
    pred = LinearRegression(fit_intercept=True).fit(z, target).predict(z)
    return pred, target


def objstats(pred, target, obj_ids, types, names):
    """Per-object spread/error of the unit-normalized predictions."""
    rows = []
    for g in np.unique(obj_ids):
        m = obj_ids == g
        p = pred[m]
        pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)   # onto unit sphere
        t = target[m][0]                                              # constant per object
        var = float(((pn - pn.mean(0)) ** 2).sum(1).mean())
        mse = float(((pn - t) ** 2).sum(1).mean())
        rows.append([names[g], types[g], int(m.sum()), round(var, 6), round(mse, 6)])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model key / file prefix (e.g. mistrallarge123b)")
    ap.add_argument("--layer", type=int, required=True, help="layer index into the pca array")
    ap.add_argument("--npca", type=int, default=8, help="number of leading PCA directions to fit (default 8)")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--out", default="projection_fit", help="output folder")
    args = ap.parse_args()

    d = np.load(os.path.join(args.indir, f"{args.model}_pca128.npz"), allow_pickle=True)
    pca = d["pca"]
    n_layers = pca.shape[1]
    if not 0 <= args.layer < n_layers:
        ap.error(f"--layer {args.layer} out of range 0..{n_layers - 1}")
    n = min(args.npca, pca.shape[2])
    obj_ids = d["obj_ids"].astype(int)
    pr_ids = d["pr_ids"].astype(int)
    yunit = d["Yunit"].astype(np.float64)
    types = d["types"].astype(str)
    names = d["names"].astype(str)

    pred, target = fit_projection(pca, obj_ids, yunit, args.layer, n)

    os.makedirs(args.out, exist_ok=True)
    coords_path = os.path.join(args.out, f"{args.model}_skyproj_coords_L{args.layer}.csv")
    with open(coords_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "type", "prompt_id", "px", "py", "pz", "tx", "ty", "tz"])
        for i in range(pred.shape[0]):
            w.writerow([names[obj_ids[i]], types[obj_ids[i]], int(pr_ids[i])]
                       + [round(float(x), 6) for x in pred[i]]
                       + [round(float(x), 6) for x in target[i]])

    stats_path = os.path.join(args.out, f"{args.model}_skyproj_objstats_L{args.layer}.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "type", "n_prompts", "var", "mse"])
        w.writerows(objstats(pred, target, obj_ids, types, names))

    print(f"saved {coords_path}  ({pred.shape[0]} samples)")
    print(f"saved {stats_path}   ({int(np.unique(obj_ids).size)} objects, npca={n}, layer={args.layer})")


if __name__ == "__main__":
    main()
