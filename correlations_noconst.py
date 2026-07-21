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

Outputs per model (into --out):

  <model>_noconst_r2_k<k>.csv     leave-one-object-out OLS on the top-k subset-PCA
                                  subspace (k in --ks, default 4 and 8). Columns:
     layer, xyz_r2, xyz_angerr_deg, xyz_angerr_med_deg,
            radec_angerr_deg, radec_angerr_med_deg, ra_r2, dec_r2
     * xyz_*   : joint 3-D fit of the sky unit vector (sinDec, cosDec sinRA,
                 cosDec cosRA); R^2 (uniform over 3 outputs) + angular error.
     * radec_* : joint fit of (RA_deg, Dec_deg); angular error of the unit vector
                 rebuilt from the predicted (RA, Dec). RA fit raw -- the 0/360
                 discontinuity is intentional.
     * ra_r2 / dec_r2 : RA and Dec fit SEPARATELY as 1-D targets. (Under OLS the
                 per-column fit of a joint (RA,Dec) regression is identical, so
                 these also equal the components of the radec joint fit.)

  <model>_noconst_skyxyz_cov_scaled_<N>.csv   sky_cov_scaled density on the subset
  <model>_noconst_radec_cov_scaled_<N>.csv    PCA; 3-D sky / 2-D (RA,Dec) targets.
     Whole-layer rescale by PC-1 std; density_j = sum_a Cov(w_j, target_a)^2 with
     the target whitened to Cov = I. On the subset-refit basis the trace is bounded
     by the number of targets again (<=3 for xyz, <=2 for RA/Dec).

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


def radec_to_unit(ra_deg, dec_deg):
    """(RA, Dec) in degrees -> unit vector (sinDec, cosDec sinRA, cosDec cosRA)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.sin(dec), np.cos(dec) * np.sin(ra), np.cos(dec) * np.cos(ra)], 1)


def loo_predict(z, target, obj):
    """Leave-one-object-out OLS prediction of `target` (>=2-D) from `z`."""
    pred = np.zeros_like(target, dtype=np.float64)
    for g in np.unique(obj):
        te = obj == g
        pred[te] = LinearRegression().fit(z[~te], target[~te]).predict(z[te])
    return pred


def _angerr(pred_unit, u):
    """Per-object angular error (deg) between predicted and true unit vectors."""
    pn = pred_unit / (np.linalg.norm(pred_unit, axis=1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((pn * u).sum(1), -1, 1)))


def r2_row(zk, obj, u, ra, dec, L):
    """One layer row of the LOO R^2 table for the top-k subset-PCA subspace `zk`."""
    # xyz: joint 3-D fit of the raw unit vector
    pxyz = loo_predict(zk, u, obj)
    r2_xyz = r2_score(u, pxyz, multioutput="uniform_average")
    ang_x = _angerr(pxyz, u)
    # (RA, Dec): joint fit, angular error from the rebuilt unit vector
    prd = loo_predict(zk, np.column_stack([ra, dec]), obj)
    ang_r = _angerr(radec_to_unit(prd[:, 0], prd[:, 1]), u)
    # RA and Dec fit separately as 1-D targets
    r2_ra = r2_score(ra, loo_predict(zk, ra[:, None], obj)[:, 0])
    r2_dec = r2_score(dec, loo_predict(zk, dec[:, None], obj)[:, 0])
    return [L, round(float(r2_xyz), 6),
            round(float(ang_x.mean()), 4), round(float(np.median(ang_x)), 4),
            round(float(ang_r.mean()), 4), round(float(np.median(ang_r)), 4),
            round(float(r2_ra), 6), round(float(r2_dec), 6)]


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

    u_w = whiten_target(u)
    rd_w = whiten_target(np.column_stack([ra_s, dec_s]))
    r2_rows = {k: [] for k in ks}
    dens_xyz, dens_rd = [], []
    for L in range(n_layers):
        s = subset_pca(pca[:, L, :].astype(np.float64))  # subset-refit scores, this layer
        dens_xyz.append(density_row(s, u_w, ndens, L))
        dens_rd.append(density_row(s, rd_w, ndens, L))
        for k in ks:
            r2_rows[k].append(r2_row(s[:, :k], obj, u, ra_s, dec_s, L))

    os.makedirs(out, exist_ok=True)
    r2_header = ["layer", "xyz_r2", "xyz_angerr_deg", "xyz_angerr_med_deg",
                 "radec_angerr_deg", "radec_angerr_med_deg", "ra_r2", "dec_r2"]
    for k in ks:
        write_csv(os.path.join(out, f"{model}_noconst_r2_k{k}.csv"), r2_header, r2_rows[k])
    dens_header = ["layer", "trace"] + [f"d{j + 1}" for j in range(ndens)]
    write_csv(os.path.join(out, f"{model}_noconst_skyxyz_cov_scaled_{ndens}.csv"), dens_header, dens_xyz)
    write_csv(os.path.join(out, f"{model}_noconst_radec_cov_scaled_{ndens}.csv"), dens_header, dens_rd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="single model key (default: all 7)", choices=MODELS)
    ap.add_argument("--ks", default="4,8", help="comma-separated PCA-subspace sizes for the LOO fit")
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
