#!/usr/bin/env python
"""
Decodable Sky -- geometric vs cartographic sky code (RSA seam discriminator).

Motivation
----------
"xyz model" and "RA/Dec model" are not two rival hypotheses -- they are two
coordinate charts of the SAME point on the sphere. Linear xyz is exactly the
degree-1 spherical harmonics (a smooth, topology-respecting basis); raw RA/Dec is
a chart with a 0/360 discontinuity. Global R^2 / commonality mostly measures the
shared smooth structure both charts capture, so it ties and is biased toward xyz
(the more expressive basis). The two hypotheses only genuinely DISAGREE where the
chart is singular: the RA=0/360 seam (and the poles). A geometric code keeps
RA=2 deg and RA=358 deg objects CLOSE; a cartographic code (one that encodes the
catalogue NUMBER) pushes them ~356 apart. So we test there.

Method (second-order RSA, symmetric, no linear-decodability assumption)
-----------------------------------------------------------------------
Per layer, per object: mean top-k subset-refit PCA score (25 prompts averaged).
  D_neural  = pairwise Euclidean distance of object centroids in top-k PCA space
  D_sphere  = great-circle angle between objects (the geometric model)
  seam      = |dRA| - min(|dRA|, 360-|dRA|)   (>0 ONLY for seam-straddling pairs;
              the cartographic-UNIQUE feature -- how much raw RA over-separates a
              pair beyond their true angular separation)
Standardised distance-matrix regression on the upper triangle:
  D_neural ~ beta_sphere * D_sphere + beta_seam * seam
  * beta_seam ~ 0  -> geometric code (seam-straddling pairs are NOT pushed apart)
  * beta_seam > 0  -> cartographic code (the raw RA number is encoded)
p_seam is a label-permutation test (permute objects, recompute beta_seam).

Two sanity numbers per layer:
  ceiling   split-half (across prompts) correlation of the neural RSM -- how much
            of the object geometry is reliable at all (an upper bound on any RSA fit).
  r_sphere / r_rawchart  first-order RSA: corr of D_neural with the sphere RSM vs
            with a raw-(RA,Dec) Euclidean RSM.

--controls prints, at each model's best layer, beta_seam for SYNTHETIC neural data
with a known code (pure sphere / pure raw chart / a 50-50 mix), confirming the test
has power: the cartographic controls must show a large positive beta_seam.

Output (into --out, a SEPARATE folder):
  <model>_geom_vs_chart_k<k>.csv
    layer, rsa_r2, beta_sphere, beta_seam, p_seam, r_sphere, r_rawchart, ceiling
"""
import argparse
import csv
import os

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]


def subset_pca(z):
    zc_ = z - z.mean(0)
    c = (zc_.T @ zc_) / zc_.shape[0]
    _, v = eigh(c)
    return zc_ @ v[:, ::-1]


def load_radec(path, names):
    ra, dec = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ra[r["name"]] = float(r["l_ra_deg"])
            dec[r["name"]] = float(r["a_dec_deg"])
    return ra, dec


def upper(M):
    return M[np.triu_indices(M.shape[0], 1)]


def zc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def reg_beta_r2(y, preds):
    """Standardised OLS of y on preds; return (coeffs incl intercept, R^2)."""
    yz = zc(y)
    X = np.column_stack([np.ones_like(yz)] + [zc(p) for p in preds])
    b, _, _, _ = np.linalg.lstsq(X, yz, rcond=None)
    yh = X @ b
    r2 = 1 - ((yz - yh) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    return b, r2


def seam_regression(Dn_vec, ds, se, rng, nperm):
    """Return (betas, rsa_r2, p_seam). p_seam from an object-label permutation."""
    b, r2 = reg_beta_r2(Dn_vec, [ds, se])
    Dn = squareform(Dn_vec)
    n = Dn.shape[0]
    null = np.empty(nperm)
    for i in range(nperm):
        p = rng.permutation(n)
        bb, _ = reg_beta_r2(upper(Dn[np.ix_(p, p)]), [ds, se])
        null[i] = bb[2]
    p_seam = (np.sum(np.abs(null) >= abs(b[2])) + 1) / (nperm + 1)
    return b, r2, p_seam


def object_centroids(pca, obj, L, k):
    """Per-object mean top-k subset-refit PCA score, plus two split-half means."""
    s = subset_pca(pca[:, L, :].astype(np.float64))[:, :k]
    ids = np.unique(obj)
    C = np.array([s[obj == i].mean(0) for i in ids])
    Ca = np.array([s[obj == i][:12].mean(0) for i in ids])
    Cb = np.array([s[obj == i][12:].mean(0) for i in ids])
    return C, Ca, Cb


def sky_matrices(nm, ra, dec):
    RA = np.array([ra[n] for n in nm])
    DE = np.array([dec[n] for n in nm])
    U = np.column_stack([np.cos(np.radians(DE)) * np.cos(np.radians(RA)),
                         np.cos(np.radians(DE)) * np.sin(np.radians(RA)),
                         np.sin(np.radians(DE))])
    Dsph = np.degrees(np.arccos(np.clip(U @ U.T, -1, 1)))
    dRA = np.abs(RA[:, None] - RA[None, :])
    seam = dRA - np.minimum(dRA, 360 - dRA)
    Draw = squareform(pdist(np.column_stack([zc(RA), zc(DE)])))
    return U, RA, DE, upper(Dsph), upper(seam), upper(Draw)


def print_controls(U, RA, DE, ds, se):
    """Synthetic neural with a known code -> verify the seam test has power."""
    print("  -- power controls (beta_seam: ~0 if code geometric, >0 if cartographic) --")
    synth = {
        "geometric (sphere)":     U,
        "cartographic (raw)":     np.column_stack([zc(RA), zc(DE)]),
        "mix (sphere+RA ramp)":   np.column_stack([U, 0.7 * zc(RA)]),
    }
    for label, C in synth.items():
        b, _ = reg_beta_r2(upper(squareform(pdist(C))), [ds, se])
        print(f"     {label:24s} beta_sphere={b[1]:+.3f}  beta_seam={b[2]:+.3f}")


def run_model(model, indir, objects, out, k, nperm, controls):
    d = np.load(os.path.join(indir, f"{model}_pca128.npz"), allow_pickle=True)
    names = d["names"].astype(str)
    keep = (d["types"].astype(str) != "constellation")[d["obj_ids"]]  # non-constellation only
    pca = d["pca"][keep]
    obj = d["obj_ids"][keep]
    if k > pca.shape[2]:                                  # cropped PCA128 file (fewer comps)
        print(f"[{model}] file has only {pca.shape[2]} PCA comps; using k={pca.shape[2]}", flush=True)
        k = pca.shape[2]
    ra, dec = load_radec(objects, names)
    nm = [names[i] for i in np.unique(obj)]
    U, RA, DE, ds, se, draw = sky_matrices(nm, ra, dec)
    n_layers = pca.shape[1]
    print(f"[{model}] {len(nm)} objects, {n_layers} layers, k={k}, "
          f"seam_pairs={(se > 1).sum()}, corr(sphere,seam)={np.corrcoef(ds, se)[0, 1]:+.2f}", flush=True)
    if controls:
        print_controls(U, RA, DE, ds, se)

    rows = []
    for L in range(n_layers):
        C, Ca, Cb = object_centroids(pca, obj, L, k)
        Dn = upper(squareform(pdist(C)))
        b, r2, p = seam_regression(Dn, ds, se, np.random.default_rng(1000 + L), nperm)
        r_sph = np.corrcoef(Dn, ds)[0, 1]
        r_raw = np.corrcoef(Dn, draw)[0, 1]
        ceil = np.corrcoef(upper(squareform(pdist(Ca))), upper(squareform(pdist(Cb))))[0, 1]
        rows.append([L, round(r2, 5), round(b[1], 4), round(b[2], 4), round(p, 4),
                     round(r_sph, 4), round(r_raw, 4), round(ceil, 4)])

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{model}_geom_vs_chart_k{k}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "rsa_r2", "beta_sphere", "beta_seam", "p_seam",
                    "r_sphere", "r_rawchart", "ceiling"])
        w.writerows(rows)
    best = max(rows, key=lambda r: r[1])
    print(f"  saved {os.path.basename(path)}  best RSA layer L{best[0]}: "
          f"beta_sphere={best[2]} beta_seam={best[3]} (p={best[4]}) "
          f"rsa_r2={best[1]} ceiling={best[7]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="mistrallarge123b,qwen235b",
                    help="comma list (default the two clearest models); 'all' for all 7")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--nperm", type=int, default=500)
    ap.add_argument("--indir", default="PCA128")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="geometry_vs_chart")
    ap.add_argument("--controls", action="store_true", help="print power controls per model")
    args = ap.parse_args()
    models = MODELS if args.models == "all" else args.models.split(",")
    for m in models:
        run_model(m, args.indir, args.objects, args.out, args.k, args.nperm, args.controls)


if __name__ == "__main__":
    main()
