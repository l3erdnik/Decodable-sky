#!/usr/bin/env python
"""
Decodable Sky -- per-direction correlation / density tables from residual-stream PCA.

Reads a `<model>_pca128.npz` produced by extract_pca.py and, for a chosen regime,
writes a per-layer table over the first N directions. It always reads the 128-
component file; --npca (default 128) is how many leading directions are used/shown.

PCA-basis regimes (directions = PCA components, ordered by variance)
-------------------------------------------------------------------
  sky_cov_scaled  sky target; rescale each layer AS A WHOLE so PC-1 has variance 1.
                  d_j = sum_a Cov(w_j, U_a)^2 ,  w = (PCA scores)/std(PC-1)
  sky_cov_equal   sky target; whiten each PCA direction to variance 1.
                  d_j = sum_a corr(z_j, U_a)^2   (per-direction squared canonical corr)
  type_cov_equal  object-type target; whiten each PCA direction to variance 1.
  sky_r2          first N PCA directions; leave-one-object-out OLS (no ridge) predicting
                  the raw sky unit vector; per-layer R^2 and mean angular error.

LDA-basis regimes (directions = LDA discriminants, ordered by eigenvalue)
------------------------------------------------------------------------
The LDA basis diagonalizes the between-object scatter S_B in the metric of the
within-object scatter S_W (v_i^T S_W v_j = delta_ij, v_i^T S_B v_j = lambda_i).
S_W = within-object scatter (variation across an object's prompts). S_B is the
WITHIN-PROMPT between-object scatter: for each prompt, the scatter of the objects
around that prompt's mean, summed over prompts -- i.e. objects are always compared
under an identical prompt, so the prompt main-effect cancels. No ridge on S_W.
Directions are scaled so the top eigenvalue is 1 (weight lambda_j/lambda_1).
Layer 0 (the raw last-token embedding) is object-independent, so its within-object
scatter is rank-deficient; the LDA regimes therefore start at layer 1.

  lda_eigs        per layer: lambda_1 (raw) and the scaled spectrum e_j = lambda_j/lambda_1.
  sky_cov_lda     d_j = (lambda_j/lambda_1) * R2_j , R2_j = fraction of LDA direction j's
                  between-object variance explained by the whitened sky target (Cov=I_3).
  type_cov_lda    same with the standardized object-type target.

Output: <out>/<model>_<regime>_<N>.csv
  density regimes : layer, trace, d1..dN
  sky_r2          : layer, r2, angerr_deg
  lda_eigs        : layer, lam1, e1..eN
"""
import argparse
import csv
import os

import numpy as np
from scipy.linalg import eigh

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]
REGIMES = ["sky_cov_scaled", "sky_cov_equal", "type_cov_equal", "sky_r2",
           "sky_cov_lda", "type_cov_lda", "lda_eigs"]


def whiten(m):
    """Return a copy of m (n, k) centered with covariance = I_k."""
    mc = m - m.mean(0)
    c = (mc.T @ mc) / mc.shape[0]
    ev, v = eigh(np.atleast_2d(c))
    return mc @ (v @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ v.T)


def type_vector(types):
    tv = np.where(types == "star", 1.0, np.where(types == "constellation", -1.0, 0.0))
    return (tv - tv.mean()) / (tv.std() + 1e-12)


# ---------------------------------------------------------------- PCA-basis density
def density_pca(pca, obj, yunit, types, regime, n):
    n_samples = pca.shape[0]
    if regime == "type_cov_equal":
        target = type_vector(types[obj])[:, None]
    else:
        target = whiten(yunit[obj].astype(np.float64))
    rows = []
    for L in range(pca.shape[1]):
        z = pca[:, L, :n].astype(np.float64)
        zc = z - z.mean(0)
        if regime == "sky_cov_scaled":
            w = zc / (zc[:, 0].std() + 1e-12)          # whole-layer rescale by PC-1 std
        else:                                          # *_cov_equal: per-direction whitening
            w = zc / (zc.std(0) + 1e-12)
        kmat = (target.T @ w) / n_samples
        diag = (kmat ** 2).sum(0)
        rows.append([L, round(float(diag.sum()), 6)] + [round(float(x), 7) for x in diag])
    return ["layer", "trace"] + [f"d{j + 1}" for j in range(n)], rows


# ---------------------------------------------------------------- sky_r2 (LOO OLS)
def sky_r2(pca, obj, yunit, n):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    u = yunit[obj].astype(np.float64)
    groups = np.unique(obj)
    rows = []
    for L in range(pca.shape[1]):
        z = pca[:, L, :n].astype(np.float64)
        pred = np.zeros_like(u)
        for g in groups:
            te = obj == g
            pred[te] = LinearRegression().fit(z[~te], u[~te]).predict(z[te])
        r2 = r2_score(u, pred, multioutput="uniform_average")
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
        ang = float(np.degrees(np.arccos(np.clip((pn * u).sum(1), -1, 1))).mean())
        rows.append([L, round(float(r2), 6), round(ang, 4)])
    return ["layer", "r2", "angerr_deg"], rows


# ---------------------------------------------------------------- LDA-basis regimes
def lda_basis(z, obj, pr):
    """Generalized eigenbasis of within-prompt S_B in the within-object metric S_W."""
    k = z.shape[1]
    sw = np.zeros((k, k))
    for o in np.unique(obj):
        zc = z[obj == o] - z[obj == o].mean(0)         # within-object (across prompts)
        sw += zc.T @ zc
    sb = np.zeros((k, k))
    for p in np.unique(pr):
        zc = z[pr == p] - z[pr == p].mean(0)           # within-prompt (across objects)
        sb += zc.T @ zc
    lam, v = eigh(sb, sw)                               # ascending; S_W assumed PD (no ridge)
    order = np.argsort(lam)[::-1]
    return lam[order], v[:, order]


def lda_table(pca, obj, pr, yunit, types, regime, n):
    uo = np.unique(obj)
    n_obj = len(uo)
    usky = whiten(yunit[uo].astype(np.float64))         # (n_obj, 3), Cov = I
    ttype = type_vector(types[uo])                       # (n_obj,)
    rows = []
    for L in range(1, pca.shape[1]):                     # skip layer 0 (embedding; S_W singular)
        z = pca[:, L, :n].astype(np.float64)
        lam, v = lda_basis(z, obj, pr)
        scaled = lam / (lam[0] + 1e-12)                  # top eigenvalue -> 1
        if regime == "lda_eigs":
            rows.append([L, round(float(lam[0]), 6)] + [round(float(x), 7) for x in scaled])
            continue
        a = (z - z.mean(0)) @ v                           # LDA scores (N, n)
        objmean = np.zeros((n_obj, n))
        np.add.at(objmean, obj, a)
        objmean /= np.bincount(obj)[:, None]
        ac = objmean - objmean.mean(0)
        var_j = (ac ** 2).mean(0) + 1e-12                # between-object variance per direction
        if regime == "sky_cov_lda":
            m = (ac.T @ usky) / n_obj                    # Cov(objmean_j, U_a)
            r2 = (m ** 2).sum(1) / var_j                 # fraction of var_j explained by sky
        else:                                            # type_cov_lda
            m = (ac.T @ ttype) / n_obj
            r2 = (m ** 2) / var_j
        diag = scaled * r2
        rows.append([L, round(float(diag.sum()), 6)] + [round(float(x), 7) for x in diag])
    key = "e" if regime == "lda_eigs" else "d"
    lead = "lam1" if regime == "lda_eigs" else "trace"
    return ["layer", lead] + [f"{key}{j + 1}" for j in range(n)], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model key / file prefix (e.g. " + ", ".join(MODELS) + ")")
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--npca", type=int, default=128, help="number of leading directions to use")
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--out", default="correlations", help="output folder")
    args = ap.parse_args()

    d = np.load(os.path.join(args.indir, f"{args.model}_pca128.npz"), allow_pickle=True)
    pca = d["pca"]
    n = min(args.npca, pca.shape[2])
    obj = d["obj_ids"].astype(int)
    pr = d["pr_ids"].astype(int)
    yunit = d["Yunit"].astype(np.float32)
    types = d["types"].astype(str)

    if args.regime == "sky_r2":
        header, rows = sky_r2(pca, obj, yunit, n)
    elif args.regime in ("sky_cov_lda", "type_cov_lda", "lda_eigs"):
        header, rows = lda_table(pca, obj, pr, yunit, types, args.regime, n)
    else:
        header, rows = density_pca(pca, obj, yunit, types, args.regime, n)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.model}_{args.regime}_{n}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {out_path}  ({len(rows)} layers, {len(header)} cols)")


if __name__ == "__main__":
    main()
