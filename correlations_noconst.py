#!/usr/bin/env python
"""
Decodable Sky -- non-constellation, subset-refit LOO decoding + density tables.

Companion to correlations.py, restricted to NON-constellation objects (stars and
"other"; every object whose type == "constellation" is dropped, along with all of
its prompt samples).

The global .npz PCA basis is fit by extract_pca.py on ALL objects, so on the
non-constellation subset its components are neither mean-zero, nor mutually
uncorrelated, nor variance-ordered. This script therefore RE-FITS the PCA on the
subset: per layer it re-centers on the subset mean (the bias) and rotates the
128 stored scores by the subset's own eigenvectors, giving components that are
orthonormal and variance-ordered ON THE SUBSET. All tables below use that basis.

The sky is parametrized two symmetric ways -- the 3-D unit vector (sinDec,
cosDec sinRA, cosDec cosRA) and the 2-D (RA_deg, Dec_deg) pair -- each centered and
whitened to Cov = I on the subset. Their direct sum is one common 5-D sky embedding;
the xyz vs RA/Dec "models" are just its 3-D and 2-D components.

Outputs per model (into --out):

  <model>_noconst_commonality_k<k>.csv   commonality decomposition of how much of the
                                  top-k subset-PCA subspace the sky embedding explains
                                  (k in --ks, default 4, 8, 16). Leave-one-object-out
                                  OLS predicts the k PC scores FROM the sky features;
                                  R^2 is uniform-averaged over the k PC outputs.
     layer, k, r2_full, r2_xyz, r2_radec, unique_xyz, unique_radec, shared
     * r2_full  : predictors = the 5-D embedding (xyz + RA/Dec together).
     * r2_xyz   : predictors = the 3-D whitened unit vector only.
     * r2_radec : predictors = the 2-D whitened (RA, Dec) only. RA fit raw -- the
                  0/360 discontinuity is intentional.
     * commonality (Newton-Spurrell):
          unique_xyz   = r2_full - r2_radec   (only the Cartesian encoding explains)
          unique_radec = r2_full - r2_xyz     (only the RA/Dec encoding explains)
          shared       = r2_xyz + r2_radec - r2_full   (<0 = suppression)

  <model>_noconst_skyxyz_cov_scaled_<N>.csv   sky_cov_scaled density on the subset
  <model>_noconst_radec_cov_scaled_<N>.csv    PCA, one file per target: the 3-D sky
  <model>_noconst_ra_cov_scaled_<N>.csv       unit vector, the 2-D (RA,Dec) pair, and
  <model>_noconst_dec_cov_scaled_<N>.csv      RA and Dec each on their own (1-D).
     Whole-layer rescale by PC-1 std; density_j = sum_a Cov(w_j, target_a)^2 with
     the target whitened to Cov = I. On the subset-refit basis the trace (sum over
     PC directions) is bounded by the number of targets: <=3 xyz, <=2 RA/Dec,
     <=1 RA-only / Dec-only. run_model prints the max trace per target to verify.

RA/Dec are read from data/astro_objects.csv (l_ra_deg, a_dec_deg) and joined to the
.npz objects by name, so this stays correct regardless of row ordering.
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


def whiten_target(y):
    """Center and whiten an (N, d) target so its covariance is I_d."""
    yc = y - y.mean(0)
    c = (yc.T @ yc) / yc.shape[0]
    ev, v = eigh(c)
    return yc @ (v @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ v.T)


def subset_pca(z):
    """Re-center (bias) and rotate the stored scores to a PCA fit ON this sample.

    Returns scores whose columns are orthonormal-direction projections ordered by
    descending variance on `z` itself (not on the full dataset the basis came from).
    """
    zc = z - z.mean(0)                                  # subtract the subset mean (bias)
    c = (zc.T @ zc) / zc.shape[0]
    ev, v = eigh(c)                                      # eigenvalues ascending
    return zc @ v[:, ::-1]                               # rotate; columns -> variance-descending


def load_radec(path, names):
    """Return (ra_deg, dec_deg) aligned to `names`, joined by object name."""
    ra, dec = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ra[r["name"]] = float(r["l_ra_deg"])
            dec[r["name"]] = float(r["a_dec_deg"])
    missing = [nm for nm in names if nm not in ra]
    if missing:
        raise KeyError(f"{len(missing)} npz objects absent from {path}: {missing[:5]}")
    return (np.array([ra[nm] for nm in names], np.float64),
            np.array([dec[nm] for nm in names], np.float64))


def loo_predict(z, target, obj):
    """Leave-one-object-out OLS prediction of `target` (>=2-D) from `z`."""
    pred = np.zeros_like(target, dtype=np.float64)
    for g in np.unique(obj):
        te = obj == g
        pred[te] = LinearRegression().fit(z[~te], target[~te]).predict(z[te])
    return pred


def commonality_row(Y, obj, xxyz, xradec, L, k):
    """Commonality decomposition of how well the sky embedding explains subspace `Y`.

    `Y` is the top-k subset-PCA subspace (the dependent variable). Leave-one-object-out
    OLS predicts it FROM the 3-D whitened xyz features, from the 2-D whitened (RA,Dec)
    features, and from their 5-D union; R^2 is uniform-averaged over the k PC outputs.
    """
    xfull = np.column_stack([xxyz, xradec])
    r2f = r2_score(Y, loo_predict(xfull, Y, obj), multioutput="uniform_average")
    r2x = r2_score(Y, loo_predict(xxyz, Y, obj), multioutput="uniform_average")
    r2r = r2_score(Y, loo_predict(xradec, Y, obj), multioutput="uniform_average")
    return [L, k, round(float(r2f), 6), round(float(r2x), 6), round(float(r2r), 6),
            round(float(r2f - r2r), 6),      # unique_xyz
            round(float(r2f - r2x), 6),      # unique_radec
            round(float(r2x + r2r - r2f), 6)]  # shared


def density_row(s, tw, n, L):
    """One layer row of the sky_cov_scaled density table on subset-PCA scores `s`."""
    z = s[:, :n]
    zc = z - z.mean(0)
    w = zc / (zc[:, 0].std() + 1e-12)                   # whole-layer rescale by PC-1 std
    kmat = (tw.T @ w) / s.shape[0]                       # (d, n) = Cov(target_a, w_j)
    diag = (kmat ** 2).sum(0)
    return [L, round(float(diag.sum()), 6)] + [round(float(x), 7) for x in diag]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  saved {os.path.basename(path)}  ({len(rows)} layers, {len(header)} cols)", flush=True)


def run_model(model, indir, objects, out, npca, ks):
    d = np.load(os.path.join(indir, f"{model}_pca128.npz"), allow_pickle=True)
    pca_all = d["pca"]
    obj_all = d["obj_ids"].astype(int)
    yunit = d["Yunit"].astype(np.float64)
    types = d["types"].astype(str)
    names = d["names"].astype(str)
    ra, dec = load_radec(objects, names)

    keep = (types != "constellation")[obj_all]          # drop constellations (keep star/other)
    pca = pca_all[keep]
    obj = obj_all[keep]
    u = yunit[obj]
    ra_s, dec_s = ra[obj], dec[obj]
    n_layers = pca.shape[1]
    ndens = min(npca, pca.shape[2])
    ks = [k for k in ks if k <= pca.shape[2]]
    print(f"[{model}] kept {pca.shape[0]} samples / {int(np.unique(obj).size)} "
          f"non-constellation objects, {n_layers} layers; refit-PCA LOO on k={ks}", flush=True)

    dens_targets = {                                    # target whitened to Cov = I per density
        "skyxyz": whiten_target(u),
        "radec": whiten_target(np.column_stack([ra_s, dec_s])),
        "ra": whiten_target(ra_s[:, None]),
        "dec": whiten_target(dec_s[:, None]),
    }
    xxyz, xradec = dens_targets["skyxyz"], dens_targets["radec"]  # commonality predictors
    comm_rows = {k: [] for k in ks}
    dens_rows = {tag: [] for tag in dens_targets}
    for L in range(n_layers):
        s = subset_pca(pca[:, L, :].astype(np.float64))  # subset-refit scores, this layer
        for tag, tw in dens_targets.items():
            dens_rows[tag].append(density_row(s, tw, ndens, L))
        for k in ks:
            comm_rows[k].append(commonality_row(s[:, :k], obj, xxyz, xradec, L, k))

    bounds = {"skyxyz": 3, "radec": 2, "ra": 1, "dec": 1}    # trace should not exceed dim
    for tag, rows in dens_rows.items():
        mx = max(r[1] for r in rows)
        flag = "" if mx <= bounds[tag] + 1e-6 else "  !! EXCEEDS BOUND"
        print(f"  density[{tag:>6}] max trace over layers = {mx:.4f}  (bound {bounds[tag]}){flag}", flush=True)

    os.makedirs(out, exist_ok=True)
    comm_header = ["layer", "k", "r2_full", "r2_xyz", "r2_radec",
                   "unique_xyz", "unique_radec", "shared"]
    for k in ks:
        write_csv(os.path.join(out, f"{model}_noconst_commonality_k{k}.csv"), comm_header, comm_rows[k])
    dens_header = ["layer", "trace"] + [f"d{j + 1}" for j in range(ndens)]
    for tag, rows in dens_rows.items():
        write_csv(os.path.join(out, f"{model}_noconst_{tag}_cov_scaled_{ndens}.csv"), dens_header, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="single model key (default: all 7)", choices=MODELS)
    ap.add_argument("--ks", default="4,8,16", help="comma-separated top-k PC subspace sizes (commonality target)")
    ap.add_argument("--npca", type=int, default=128, help="components shown in the density tables")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="correlations_noconst", help="output folder")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    for model in ([args.model] if args.model else MODELS):
        run_model(model, args.indir, args.objects, args.out, args.npca, ks)


if __name__ == "__main__":
    main()
