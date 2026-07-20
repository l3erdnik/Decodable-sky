#!/usr/bin/env python
"""
Decodable Sky -- non-constellation LOO decoding + density tables.

Companion to correlations.py, restricted to NON-constellation objects (stars and
"other"; every object whose type == "constellation" is dropped, along with all of
its prompt samples). For a given model it produces four per-layer tables:

  1. sky_xyz_r2   leave-one-object-out OLS predicting the raw 3-D sky unit vector
                  (sinDec, cosDec sinRA, cosDec cosRA); per-layer R^2 (uniform over
                  the 3 outputs) plus mean / median angular error.
  2. radec_r2     leave-one-object-out OLS predicting (RA_deg, Dec_deg) directly.
                  RA is fit as a raw scalar -- its 0/360 discontinuity is intentional.
                  Reports SEPARATE R^2 for RA and for Dec, plus the angular error of
                  the unit vector rebuilt from the predicted (RA, Dec).
  3. skyxyz_cov_scaled   sky_cov_scaled density table with the 3-D sky target.
  4. radec_cov_scaled    sky_cov_scaled density table with the 2-D (RA, Dec) target.

The density regime mirrors correlations.py `sky_cov_scaled`:
  target is whitened to Cov = I (I_3 for xyz, I_2 for RA/Dec); each layer is rescaled
  AS A WHOLE so PC-1 has variance 1 (w = scores / std(PC-1)); then
  density_j = sum_a Cov(w_j, target_a)^2 . This keeps each direction's relative
  variance (distinguishes "small well-correlated" from "large poorly-correlated").

RA/Dec are read from data/astro_objects.csv (l_ra_deg, a_dec_deg) and joined to the
.npz objects by name, so this stays correct regardless of row ordering.

Output (per model, into --out):
  <model>_noconst_sky_xyz_r2_<N>.csv          layer, r2, angerr_deg, angerr_median_deg
  <model>_noconst_radec_r2_<N>.csv            layer, r2_ra, r2_dec, angerr_deg, angerr_median_deg
  <model>_noconst_skyxyz_cov_scaled_<N>.csv   layer, trace, d1..dN
  <model>_noconst_radec_cov_scaled_<N>.csv    layer, trace, d1..dN
"""
import argparse
import csv
import os

import numpy as np
from scipy.linalg import eigh

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]


def whiten_target(y):
    """Center and whiten an (N, d) target so its covariance is I_d."""
    yc = y - y.mean(0)
    c = (yc.T @ yc) / yc.shape[0]
    ev, v = eigh(c)
    return yc @ (v @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ v.T)


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
    """Leave-one-object-out OLS prediction of `target` from `z`."""
    from sklearn.linear_model import LinearRegression
    pred = np.zeros_like(target)
    for g in np.unique(obj):
        te = obj == g
        pred[te] = LinearRegression().fit(z[~te], target[~te]).predict(z[te])
    return pred


def sky_xyz_r2_table(pca, obj, u, n):
    from sklearn.metrics import r2_score
    rows = []
    for L in range(pca.shape[1]):
        z = pca[:, L, :n].astype(np.float64)
        pred = loo_predict(z, u, obj)
        r2 = r2_score(u, pred, multioutput="uniform_average")
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
        ang = np.degrees(np.arccos(np.clip((pn * u).sum(1), -1, 1)))
        rows.append([L, round(float(r2), 6),
                     round(float(ang.mean()), 4), round(float(np.median(ang)), 4)])
    return ["layer", "r2", "angerr_deg", "angerr_median_deg"], rows


def radec_r2_table(pca, obj, ra, dec, u_true, n):
    from sklearn.metrics import r2_score
    target = np.column_stack([ra, dec])          # RA fit raw (discontinuity intentional)
    rows = []
    for L in range(pca.shape[1]):
        z = pca[:, L, :n].astype(np.float64)
        pred = loo_predict(z, target, obj)
        r2_ra = r2_score(ra, pred[:, 0])
        r2_dec = r2_score(dec, pred[:, 1])
        pu = radec_to_unit(pred[:, 0], pred[:, 1])
        ang = np.degrees(np.arccos(np.clip((pu * u_true).sum(1), -1, 1)))
        rows.append([L, round(float(r2_ra), 6), round(float(r2_dec), 6),
                     round(float(ang.mean()), 4), round(float(np.median(ang)), 4)])
    return ["layer", "r2_ra", "r2_dec", "angerr_deg", "angerr_median_deg"], rows


def cov_scaled_table(pca, target, n):
    """sky_cov_scaled density: whole-layer rescale by PC-1 std, sum_a Cov(w_j, t_a)^2."""
    n_samples = pca.shape[0]
    tw = whiten_target(target.astype(np.float64))     # Cov -> I
    rows = []
    for L in range(pca.shape[1]):
        z = pca[:, L, :n].astype(np.float64)
        zc = z - z.mean(0)
        w = zc / (zc[:, 0].std() + 1e-12)             # whole-layer rescale by PC-1 std
        kmat = (tw.T @ w) / n_samples                 # (d, n) = Cov(target_a, w_j)
        diag = (kmat ** 2).sum(0)
        rows.append([L, round(float(diag.sum()), 6)] + [round(float(x), 7) for x in diag])
    header = ["layer", "trace"] + [f"d{j + 1}" for j in range(n)]
    return header, rows


def run_model(model, indir, objects, out, npca):
    d = np.load(os.path.join(indir, f"{model}_pca128.npz"), allow_pickle=True)
    pca = d["pca"]
    n = min(npca, pca.shape[2])
    obj_all = d["obj_ids"].astype(int)
    yunit = d["Yunit"].astype(np.float64)
    types = d["types"].astype(str)
    names = d["names"].astype(str)
    ra, dec = load_radec(objects, names)

    keep_obj = types != "constellation"               # drop constellations (keep star/other)
    keep = keep_obj[obj_all]
    pca, obj = pca[keep], obj_all[keep]
    u = yunit[obj]
    ra_s, dec_s = ra[obj], dec[obj]
    n_obj = int(np.unique(obj).size)
    print(f"[{model}] kept {pca.shape[0]} samples / {n_obj} non-constellation objects, "
          f"{pca.shape[1]} layers, npca={n}", flush=True)

    tables = {
        "sky_xyz_r2":        sky_xyz_r2_table(pca, obj, u, n),
        "radec_r2":          radec_r2_table(pca, obj, ra_s, dec_s, u, n),
        "skyxyz_cov_scaled": cov_scaled_table(pca, u, n),
        "radec_cov_scaled":  cov_scaled_table(pca, np.column_stack([ra_s, dec_s]), n),
    }
    os.makedirs(out, exist_ok=True)
    for tag, (header, rows) in tables.items():
        path = os.path.join(out, f"{model}_noconst_{tag}_{n}.csv")
        with open(path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(header)
            wtr.writerows(rows)
        print(f"  saved {path}  ({len(rows)} layers, {len(header)} cols)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="single model key (default: all 7)", choices=MODELS)
    ap.add_argument("--npca", type=int, default=128, help="number of leading PCA directions to use")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="correlations_noconst", help="output folder")
    args = ap.parse_args()

    for model in ([args.model] if args.model else MODELS):
        run_model(model, args.indir, args.objects, args.out, args.npca)


if __name__ == "__main__":
    main()
