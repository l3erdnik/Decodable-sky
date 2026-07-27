#!/usr/bin/env python
"""
Decodable Sky -- per-layer LOO R^2 of the optimal top-k PCA -> sky projection.

For each model and every layer, take the top-k subset-refit PCA scores and fit the
optimal linear projection (leave-one-object-out OLS) onto two sky parametrizations:

  xyz    the 3-D unit vector (x=sinDec, y=cosDec sinRA, z=cosDec cosRA)
  radec  the 2-D (RA_deg, Dec_deg) pair; RA is fit raw -- its 0/360 discontinuity
         is intentional.

Test list
---------
All objects EXCEPT the 11 constellations whose single catalogue point sits on the
RA=0/360 seam or on a celestial pole, where the centroid RA is ill-defined:
  Ursa Minor, Cepheus, Cassiopeia, Andromeda, Pegasus, Pisces, Cetus, Sculptor,
  Phoenix, Tucana, Octans  (all stored as "<name> constellation").
So stars + "other" + the 77 well-defined constellations are kept.

PCA basis
---------
Like correlations_noconst.py, the stored 128-component PCA (fit by extract_pca.py on
ALL objects) is RE-FIT on the test-list subset per layer: re-center on the subset
mean and rotate by the subset's own eigenvectors, so the components are orthonormal
and variance-ordered ON THE SUBSET. The top k of those are the predictors.

Whitening and the two R^2 flavours
----------------------------------
Each target is centered and whitened to Cov = I on the subset before the fit (as
requested). Because OLS is linear, fitting the whitened target and un-whitening the
prediction is identical to fitting the raw target, so:
  * the per-coordinate columns (r2_x, r2_y, r2_z, r2_ra, r2_dec) are ordinary R^2 on
    the raw coordinate -- how decodable that individual coordinate is;
  * the whole-target columns (r2_xyz, r2_radec) are the uniform-average R^2 over the
    WHITENED components, i.e. the fraction of isotropic (unit-variance-per-axis) sky
    variance explained -- the fair aggregate that makes the 3-D and 2-D models
    comparable. It is NOT the mean of the per-coordinate columns.

Output (one file per model, into --out):
  <model>_loo_projfit_k<k>.csv
    layer, r2_xyz, r2_x, r2_y, r2_z, r2_radec, r2_ra, r2_dec
"""
import argparse
import csv
import os

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]

EXCLUDE = {  # constellations on the RA=0 seam or a pole -> ill-defined centroid RA
    "Ursa Minor constellation", "Cepheus constellation", "Cassiopeia constellation",
    "Andromeda constellation", "Pegasus constellation", "Pisces constellation",
    "Cetus constellation", "Sculptor constellation", "Phoenix constellation",
    "Tucana constellation", "Octans constellation",
}


def whiten_matrix(y):
    """Return (mean, W) such that (y - mean) @ W has covariance I."""
    m = y.mean(0)
    yc = y - m
    c = (yc.T @ yc) / yc.shape[0]
    ev, v = eigh(c)
    w = v @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ v.T
    return m, w


def subset_pca(z):
    """Re-center and rotate stored scores to the subset's own variance-ordered PCA."""
    zc = z - z.mean(0)
    c = (zc.T @ zc) / zc.shape[0]
    ev, v = eigh(c)
    return zc @ v[:, ::-1]                       # columns variance-descending on subset


def load_radec(path, names):
    ra, dec = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ra[r["name"]] = float(r["l_ra_deg"])
            dec[r["name"]] = float(r["a_dec_deg"])
    missing = [nm for nm in names if nm not in ra]
    if missing:
        raise KeyError(f"{len(missing)} npz objects absent from {path}: {missing[:5]}")
    return (np.array([ra[nm] for nm in names]), np.array([dec[nm] for nm in names]))


def loo_predict(z, target, obj):
    """Leave-one-object-out OLS prediction of raw `target` from `z`."""
    pred = np.zeros_like(target, dtype=np.float64)
    for g in np.unique(obj):
        te = obj == g
        pred[te] = LinearRegression().fit(z[~te], target[~te]).predict(z[te])
    return pred


def target_r2(z, y_raw, obj):
    """(whole whitened-uniform R^2, per-raw-coordinate R^2 list) for one target."""
    pred = loo_predict(z, y_raw, obj)
    per = r2_score(y_raw, pred, multioutput="raw_values")
    m, w = whiten_matrix(y_raw)
    whole = r2_score((y_raw - m) @ w, (pred - m) @ w, multioutput="uniform_average")
    return float(whole), [float(p) for p in per]


def run_model(model, indir, objects, out, k):
    d = np.load(os.path.join(indir, f"{model}_pca128.npz"), allow_pickle=True)
    pca_all = d["pca"]
    obj_all = d["obj_ids"].astype(int)
    names = d["names"].astype(str)
    yunit = d["Yunit"].astype(np.float64)
    ra_o, dec_o = load_radec(objects, names)

    if k > pca_all.shape[2]:                              # cropped PCA128 file (fewer comps)
        print(f"[{model}] file has only {pca_all.shape[2]} PCA comps; using k={pca_all.shape[2]}", flush=True)
        k = pca_all.shape[2]
    keep = (~np.isin(names, list(EXCLUDE)))[obj_all]      # drop the 11 seam/pole objects
    obj = obj_all[keep]
    u = yunit[obj]                                        # (N,3) raw unit vector
    radec = np.column_stack([ra_o[obj], dec_o[obj]])      # (N,2) raw RA, Dec
    n_layers = pca_all[keep].shape[1]
    print(f"[{model}] {keep.sum()} samples / {int(np.unique(obj).size)} objects, "
          f"{n_layers} layers, top-{k} PCA", flush=True)

    rows = []
    for L in range(n_layers):
        z = subset_pca(pca_all[keep, L, :].astype(np.float64))[:, :k]
        xyz_whole, (r2x, r2y, r2z) = target_r2(z, u, obj)
        rd_whole, (r2ra, r2dec) = target_r2(z, radec, obj)
        rows.append([L, round(xyz_whole, 6), round(r2x, 6), round(r2y, 6), round(r2z, 6),
                     round(rd_whole, 6), round(r2ra, 6), round(r2dec, 6)])

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{model}_loo_projfit_k{k}.csv")
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["layer", "r2_xyz", "r2_x", "r2_y", "r2_z", "r2_radec", "r2_ra", "r2_dec"])
        wtr.writerows(rows)
    best = max(rows, key=lambda r: r[1])
    print(f"  saved {os.path.basename(path)}  ({len(rows)} layers); "
          f"best xyz R2={best[1]} at layer {best[0]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, help="single model key (default: all 7)")
    ap.add_argument("--k", type=int, default=8, help="number of top subset-PCA directions used")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="loo_projfit", help="output folder")
    args = ap.parse_args()

    for model in ([args.model] if args.model else MODELS):
        run_model(model, args.indir, args.objects, args.out, args.k)


if __name__ == "__main__":
    main()
