#!/usr/bin/env python
"""
Decodable Sky -- per-direction correlation / density tables from residual-stream PCA.

Reads a `<model>_pca128.npz` produced by extract_pca.py and, for a chosen regime,
writes a per-layer table over the first N PCA directions. It always reads the 128-
component file; --npca (default 128) is how many of those directions are used/shown.

Regimes
-------
  sky_cov_scaled  sky target; rescale each layer AS A WHOLE so PC-1 has variance 1.
                  density_j = sum_a Cov(w_j, U_a)^2 ,  w = (PCA scores)/std(PC-1)
                  -> keeps each direction's relative variance (distinguishes
                     "small well-correlated" from "large poorly-correlated").
  sky_cov_equal   sky target; whiten each PCA direction to variance 1.
                  density_j = sum_a corr(z_j, U_a)^2  (per-direction squared canonical corr;
                  the columns sum to sum of squared canonical correlations, in [0, 3]).
  type_cov_equal  object-type target; whiten each PCA direction to variance 1.
                  density_j = corr(z_j, T)^2 ,  T = (star:+1, constellation:-1, other:0)
                  standardized to mean 0 / variance 1.
  sky_r2          first N PCA directions; leave-one-object-out OLS (no ridge) predicting
                  the raw sky unit vector; per-layer R^2, mean and median angular error.

Sky target for the density regimes is whitened to Cov = I_3 so a perfect isometric
fit puts unit mass on a direction; sky_r2 uses the raw unit vectors.

Output: <out>/<model>_<regime>_<N>.csv
  sky_cov_scaled / sky_cov_equal / type_cov_equal :  layer, trace, d1..dN
  sky_r2                                          :  layer, r2, angerr_deg, angerr_median_deg
"""
import argparse
import csv
import os

import numpy as np
from scipy.linalg import eigh

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]
REGIMES = ["sky_cov_scaled", "sky_cov_equal", "type_cov_equal", "sky_r2"]


def whiten_sky(u):
    uc = u - u.mean(0)
    c = (uc.T @ uc) / uc.shape[0]
    ev, v = eigh(c)
    return uc @ (v @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ v.T)  # Cov -> I_3


def density_table(pca, obj, yunit, types, regime, n):
    n_samples, n_layers, _ = pca.shape
    if regime == "type_cov_equal":
        tv = np.where(types == "star", 1.0, np.where(types == "constellation", -1.0, 0.0))
        target = (tv[obj] - tv[obj].mean()) / (tv[obj].std() + 1e-12)   # (N,) mean 0 var 1
        target = target[:, None]                                       # (N, 1)
    else:
        target = whiten_sky(yunit[obj].astype(np.float64))             # (N, 3) Cov = I

    rows = []
    for L in range(n_layers):
        z = pca[:, L, :n].astype(np.float64)
        zc = z - z.mean(0)
        if regime == "sky_cov_scaled":
            scale = zc[:, 0].std() + 1e-12          # whole-layer rescale by PC-1 std
            w = zc / scale
        else:                                        # *_cov_equal: per-direction whitening
            w = zc / (zc.std(0) + 1e-12)
        kmat = (target.T @ w) / n_samples            # (n_targets, n)  = Cov(target_a, w_j)
        diag = (kmat ** 2).sum(0)                     # length n
        rows.append([L, round(float(diag.sum()), 6)] + [round(float(x), 7) for x in diag])
    header = ["layer", "trace"] + [f"d{j + 1}" for j in range(n)]
    return header, rows


def sky_r2_table(pca, obj, yunit, n):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    n_samples, n_layers, _ = pca.shape
    u = yunit[obj].astype(np.float64)                # raw unit vectors
    groups = np.unique(obj)
    rows = []
    for L in range(n_layers):
        z = pca[:, L, :n].astype(np.float64)
        pred = np.zeros_like(u)
        for g in groups:                              # leave-one-object-out
            te = obj == g
            pred[te] = LinearRegression().fit(z[~te], u[~te]).predict(z[te])
        r2 = r2_score(u, pred, multioutput="uniform_average")
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
        ang = np.degrees(np.arccos(np.clip((pn * u).sum(1), -1, 1)))   # per-object error
        rows.append([L, round(float(r2), 6),
                     round(float(ang.mean()), 4), round(float(np.median(ang)), 4)])
    return ["layer", "r2", "angerr_deg", "angerr_median_deg"], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model key / file prefix (e.g. " + ", ".join(MODELS) + ")")
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--npca", type=int, default=128, help="number of leading PCA directions to use")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--out", default="correlations", help="output folder")
    args = ap.parse_args()

    d = np.load(os.path.join(args.indir, f"{args.model}_pca128.npz"), allow_pickle=True)
    pca = d["pca"]
    n = min(args.npca, pca.shape[2])
    obj = d["obj_ids"].astype(int)
    yunit = d["Yunit"].astype(np.float32)
    types = d["types"].astype(str)

    if args.regime == "sky_r2":
        header, rows = sky_r2_table(pca, obj, yunit, n)
    else:
        header, rows = density_table(pca, obj, yunit, types, args.regime, n)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.model}_{args.regime}_{n}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {out_path}  ({len(rows)} layers, {len(header)} cols)")


if __name__ == "__main__":
    main()
