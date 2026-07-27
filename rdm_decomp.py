#!/usr/bin/env python
"""
Decodable Sky -- squared-distance RDM decomposition (equal-footing coordinates).

Neural RDM = SQUARED Euclidean distance between per-object centroids in the top-k
subset-refit PCA space (non-constellation objects only), the space scaled AS A WHOLE
so PC-1 has variance 1 (the sky_cov_scaled convention -- relative PC variances are
preserved, so the dominant directions dominate the distances). Because squared
distances are ADDITIVE over a direct sum of features, we regress the neural RDM onto
the squared-distance RDM of each sky coordinate, every coordinate standardised to
unit variance:

    D2_neural ~ b0 + b_x D2_x + b_y D2_y + b_z D2_z + b_dec D2_dec + b_ra D2_ra

  x, y, z  = the 3-D unit vector (x=sinDec, y=cosDec sinRA, z=cosDec cosRA)
  dec, ra  = the raw angles in degrees (RA discontinuous at 0/360 -- intentional)

The intercept absorbs the isotropic baseline from non-sky neural variance. The two
charts share almost everything (x is a near-duplicate of dec; raw ra overlaps the
geometric longitude y,z), so individual coefficients are collinear -- read the
decomposition through ra's UNIQUE incremental R^2 instead:

    dr2_ra_unique = R2(x,y,z,dec,ra) - R2(x,y,z,dec)     (the isolated cartographic
                                                          / raw-RA-number signal)
~0  -> geometric code (only the smooth circular longitude is present)
>0  -> the raw RA number (its 0/360 seam) is separately encoded.
p from an object-label permutation of the neural RDM.

The Gram / correlation matrix among the five coordinate RDMs (printed once; it
depends only on the sky coordinates, so it is identical across models) shows how
non-orthogonal the coordinates are.

Output (one file per model, into --out):
  <model>_rdm_decomp_k<k>.csv
    layer, r2, dr2_ra_unique, b0, b_x, b_y, b_z, b_dec, b_ra
"""
import argparse
import csv
import os

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

MODELS = ["qwen32b", "qwen235b", "gptoss120b", "llama33_70b",
          "mixtral8x22b", "mistrallarge123b", "glm45air"]
COORDS5 = ["x", "y", "z", "dec", "ra"]


def subset_pca(z):
    zc = z - z.mean(0)
    c = (zc.T @ zc) / zc.shape[0]
    _, v = eigh(c)
    return zc @ v[:, ::-1]


def upper(M):
    return M[np.triu_indices(M.shape[0], 1)]


def std(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def d2(f):
    """Squared-distance RDM (upper triangle) of a standardised scalar coordinate."""
    fs = std(f)
    return upper((fs[:, None] - fs[None, :]) ** 2)


def load_radec(path, names):
    ra, dec = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ra[r["name"]] = float(r["l_ra_deg"])
            dec[r["name"]] = float(r["a_dec_deg"])
    return np.array([ra[n] for n in names]), np.array([dec[n] for n in names])


def coord_rdms(RA, DE):
    """The five standardised coordinate squared-distance RDMs, columns = COORDS5."""
    coords = {
        "x": np.sin(np.radians(DE)),
        "y": np.cos(np.radians(DE)) * np.sin(np.radians(RA)),
        "z": np.cos(np.radians(DE)) * np.cos(np.radians(RA)),
        "dec": DE, "ra": RA,
    }
    return np.column_stack([d2(coords[c]) for c in COORDS5])


def r2_of(design, y):
    b, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    yh = design @ b
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b


def print_gram(M):
    cos = M.T @ M / np.outer(np.linalg.norm(M, axis=0), np.linalg.norm(M, axis=0))
    corr = np.corrcoef(M.T)
    print("Coordinate-RDM Gram (cosine; 0=orthogonal) [model-independent]:")
    print("        " + "".join(f"{c:>8}" for c in COORDS5))
    for i, c in enumerate(COORDS5):
        print(f"  {c:>4}: " + "".join(f"{cos[i, j]:>8.3f}" for j in range(5)))
    print("Coordinate-RDM centered correlation (regression collinearity):")
    print("        " + "".join(f"{c:>8}" for c in COORDS5))
    for i, c in enumerate(COORDS5):
        print(f"  {c:>4}: " + "".join(f"{corr[i, j]:>8.3f}" for j in range(5)))


def run_model(model, indir, objects, out, k, nperm, show_gram, rng):
    d = np.load(os.path.join(indir, f"{model}_pca128.npz"), allow_pickle=True)
    names = d["names"].astype(str)
    keep = (d["types"].astype(str) != "constellation")[d["obj_ids"]]
    pca = d["pca"][keep]
    obj = d["obj_ids"][keep]
    if k > pca.shape[2]:                                  # cropped PCA128 file (fewer comps)
        print(f"[{model}] file has only {pca.shape[2]} PCA comps; using k={pca.shape[2]}", flush=True)
        k = pca.shape[2]
    RA_all, DE_all = load_radec(objects, names)
    ids = np.unique(obj)
    RA, DE = RA_all[ids], DE_all[ids]
    M = coord_rdms(RA, DE)
    if show_gram:
        print_gram(M)
        print()

    i_ra = COORDS5.index("ra")
    Dfull = np.column_stack([np.ones(M.shape[0]), M])
    Dno = np.column_stack([np.ones(M.shape[0]), np.delete(M, i_ra, axis=1)])

    rows, cache = [], {}
    for L in range(pca.shape[1]):
        s = subset_pca(pca[:, L, :].astype(np.float64))[:, :k]
        s = s / (s[:, 0].std() + 1e-12)                       # scale whole space so PC-1 var=1
        C = np.array([s[obj == i].mean(0) for i in ids])
        yv = upper(squareform(pdist(C)) ** 2)                 # SQUARED distances
        cache[L] = yv
        r2f, b = r2_of(Dfull, yv)
        r2n, _ = r2_of(Dno, yv)
        rows.append([L, round(r2f, 5), round(r2f - r2n, 5)] + [round(float(v), 5) for v in b])

    best = max(rows, key=lambda r: r[1])
    Lb = best[0]
    # permutation p for ra's unique dR2 at the best RSA layer
    obs, Draw2 = best[2], squareform(cache[Lb])
    null = np.empty(nperm)
    for i in range(nperm):
        p = rng.permutation(len(ids))                     # permute OBJECTS (RDM rows/cols)
        yp = upper(Draw2[np.ix_(p, p)])
        null[i] = r2_of(Dfull, yp)[0] - r2_of(Dno, yp)[0]
    p_ra = (np.sum(null >= obs) + 1) / (nperm + 1)

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{model}_rdm_decomp_k{k}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "r2", "dr2_ra_unique", "b0", "b_x", "b_y", "b_z", "b_dec", "b_ra"])
        w.writerows(rows)
    print(f"[{model}] best RSA layer L{Lb}: r2={best[1]:.3f}  "
          f"dr2_ra_unique={best[2]:+.4f} (p={p_ra:.3f})  b_ra={best[8]:+.3f}  -> {os.path.basename(path)}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help="'all' (default) or comma-separated model keys")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--indir", default="PCA128", help="folder holding <model>_pca128.npz")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="geometry_vs_chart")
    args = ap.parse_args()
    models = MODELS if args.models == "all" else args.models.split(",")
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        run_model(m, args.indir, args.objects, args.out, args.k, args.nperm, show_gram=(i == 0), rng=rng)


if __name__ == "__main__":
    main()
