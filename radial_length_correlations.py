#!/usr/bin/env python
"""
Per-object "mean prediction-vector length" correlations, for every model with a
<model>_pca128.npz in PCA128/.

For each model:
  1. Pick the layer with the best (smallest) MEDIAN angular error at npca=8,
     read from correlations/<model>_sky_r2_8.csv (angerr_median_deg column).
  2. At that layer, take the top-8 PCA directions and build leave-one-object-out
     OLS predictions of the 3D sky unit vector -- computed exactly as in the github
     correlations.py sky_r2 ("sky_r2_table") function: LinearRegression (OLS with
     intercept) trained on every OTHER object's samples, predicting the held-out
     object's samples; angular error = angle between the L2-normalized prediction
     and the (unit) target. (numpy lstsq on an intercept-augmented design is
     identical to sklearn LinearRegression(fit_intercept=True).)
  3. Per object: mean-square angular error MSE_o (mean of angerr^2 over its prompts)
     and mean prediction-vector length meanlen_o (mean of ||pred|| over its prompts).
  4. Eliminate objects whose MSE_o exceeds 2x the across-object average MSE.
  5. Over the surviving objects, correlate meanlen_o with:
       (1) distance to object          Spearman, stars only (stars with a distance)
       (2) text-corpus frequency (zipf) Pearson,  stars only
       (3) text-corpus frequency (zipf) Pearson,  constellations only
       (4) negative apparent magnitude  Pearson,  stars only
       (5) "constellation" indicator    Pearson,  all surviving
       (6) "star" indicator             Pearson,  all surviving
       (7) "other" indicator            Pearson,  all surviving

Object metadata (Vmag, distance_ly, zipf) is model-independent and read from
astro_metadata_full.csv, produced by build_star_metadata.py (SIMBAD parallax
distances for every star + wordfreq frequencies).

Outputs (under radial_corr/):
  corr_summary.csv          one row per model: layer, kept counts, the 7 correlations
                            (r + p) and their subset sizes.
  <model>_perobject.csv     per-object detail: type, kept flag, mse_angerr, mean_len,
                            Vmag, distance_ly, zipf.
"""
import csv
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "gptoss120b_harmony",
          "llama33_70b", "mixtral8x22b", "mistrallarge123b", "glm45air"]
NPCA = 8
PCA_DIR = "PCA128"
CORR_DIR = "correlations"
META_FILE = "astro_metadata_full.csv"
OUT_DIR = "radial_corr"


def load_meta(path):
    """name -> {Vmag, distance_ly, zipf} as floats (NaN when blank)."""
    def f(s):
        s = (s or "").strip()
        return float(s) if s else float("nan")
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["name"]] = {"Vmag": f(r.get("Vmag")),
                               "distance_ly": f(r.get("distance_ly")),
                               "zipf": f(r.get("zipf"))}
    return meta


def pick_layer(model):
    """Layer with the smallest median angular error at npca=8."""
    path = os.path.join(CORR_DIR, f"{model}_sky_r2_8.csv")
    best_L, best_med = None, float("inf")
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            med = float(r["angerr_median_deg"])
            if med < best_med:
                best_med, best_L = med, int(r["layer"])
    return best_L, best_med


def loo_predictions(z, u, obj):
    """LOO-by-object OLS (intercept) predictions, one 3D vector per sample."""
    pred = np.zeros_like(u)
    for g in np.unique(obj):
        te = obj == g
        Xtr = np.hstack([z[~te], np.ones((int((~te).sum()), 1))])
        coef, *_ = np.linalg.lstsq(Xtr, u[~te], rcond=None)
        Xte = np.hstack([z[te], np.ones((int(te.sum()), 1))])
        pred[te] = Xte @ coef
    return pred


def corr(fn, x, y):
    """Return (r, p, n) for paired finite values; (nan, nan, n) if n < 3."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), n
    r, p = fn(x, y)
    return float(r), float(p), n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = load_meta(META_FILE)
    summary = []
    for model in MODELS:
        npz = os.path.join(PCA_DIR, f"{model}_pca128.npz")
        if not os.path.exists(npz):
            print(f"[skip] {model}: no {npz}")
            continue
        d = np.load(npz, allow_pickle=True)
        pca = d["pca"]
        obj = d["obj_ids"].astype(int)
        names = d["names"].astype(str)
        types = d["types"].astype(str)
        Yunit = d["Yunit"].astype(np.float64)

        L, med = pick_layer(model)
        z = pca[:, L, :NPCA].astype(np.float64)
        u = Yunit[obj]                                  # per-sample unit target
        pred = loo_predictions(z, u, obj)

        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
        angerr = np.degrees(np.arccos(np.clip((pn * u).sum(1), -1, 1)))   # per sample
        length = np.linalg.norm(pred, axis=1)                             # per sample

        n_obj = len(names)
        mse = np.zeros(n_obj); meanlen = np.zeros(n_obj)
        for oi in range(n_obj):
            m = obj == oi
            mse[oi] = np.mean(angerr[m] ** 2)
            meanlen[oi] = np.mean(length[m])

        avg_mse = mse.mean()
        keep = mse <= 2.0 * avg_mse                     # drop MSE > 2x average

        # per-object detail table
        with open(os.path.join(OUT_DIR, f"{model}_perobject.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "type", "kept", "mse_angerr_deg2", "mean_len",
                        "Vmag", "distance_ly", "zipf"])
            for oi in range(n_obj):
                mt = meta.get(names[oi], {})
                w.writerow([names[oi], types[oi], int(keep[oi]),
                            round(float(mse[oi]), 6), round(float(meanlen[oi]), 8),
                            mt.get("Vmag", ""), mt.get("distance_ly", ""),
                            mt.get("zipf", "")])

        # kept-object arrays
        kidx = np.where(keep)[0]
        ktype = types[kidx]
        kml = meanlen[kidx]
        kdist = np.array([meta.get(names[i], {}).get("distance_ly", np.nan) for i in kidx])
        kzipf = np.array([meta.get(names[i], {}).get("zipf", np.nan) for i in kidx])
        kvmag = np.array([meta.get(names[i], {}).get("Vmag", np.nan) for i in kidx])
        is_star = ktype == "star"
        is_const = ktype == "constellation"
        is_other = ktype == "other"

        c1 = corr(spearmanr, kml[is_star],  kdist[is_star])   # distance, stars (Spearman)
        c2 = corr(pearsonr,  kml[is_star],  kzipf[is_star])   # zipf, stars (Pearson)
        c3 = corr(pearsonr,  kml[is_const], kzipf[is_const])  # zipf, constellations (Pearson)
        c4 = corr(pearsonr,  kml[is_star], -kvmag[is_star])   # -Vmag, stars
        c5 = corr(pearsonr,  kml, is_const.astype(float))     # const indicator, all
        c6 = corr(pearsonr,  kml, is_star.astype(float))      # star indicator, all
        c7 = corr(pearsonr,  kml, is_other.astype(float))     # other indicator, all

        row = {
            "model": model, "layer": L, "median_angerr_deg": round(med, 4),
            "n_total": n_obj, "n_kept": int(keep.sum()),
            "n_dropped": int((~keep).sum()),
            "dist_star_spearman_r": c1[0], "dist_star_p": c1[1], "dist_star_n": c1[2],
            "zipf_star_pearson_r": c2[0], "zipf_star_p": c2[1], "zipf_star_n": c2[2],
            "zipf_const_pearson_r": c3[0], "zipf_const_p": c3[1], "zipf_const_n": c3[2],
            "negVmag_star_pearson_r": c4[0], "negVmag_star_p": c4[1], "negVmag_star_n": c4[2],
            "const_ind_pearson_r": c5[0], "const_ind_p": c5[1], "const_ind_n": c5[2],
            "star_ind_pearson_r": c6[0], "star_ind_p": c6[1], "star_ind_n": c6[2],
            "other_ind_pearson_r": c7[0], "other_ind_p": c7[1], "other_ind_n": c7[2],
        }
        summary.append(row)
        print(f"[{model}] L={L} med={med:.3f} kept={int(keep.sum())}/{n_obj} "
              f"dist_r={c1[0]:.3f} zipfS_r={c2[0]:.3f} zipfC_r={c3[0]:.3f} "
              f"-Vmag_r={c4[0]:.3f} const_r={c5[0]:.3f} star_r={c6[0]:.3f} other_r={c7[0]:.3f}")

    def rr(x):
        return "" if isinstance(x, float) and np.isnan(x) else (round(x, 6) if isinstance(x, float) else x)
    if summary:
        cols = list(summary[0].keys())
        with open(os.path.join(OUT_DIR, "corr_summary.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for row in summary:
                w.writerow([rr(row[c]) for c in cols])
        print(f"\nsaved {os.path.join(OUT_DIR, 'corr_summary.csv')} ({len(summary)} models)")


if __name__ == "__main__":
    main()
