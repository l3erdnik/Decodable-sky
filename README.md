# Decodable Sky

Probing whether large language models internally represent **astronomical objects by
their position on the sky**. Each object (a star, constellation, or deep-sky object)
is labelled by a unit direction vector `(sin Dec, cos Dec sin RA, cos Dec cos RA)`, and
we ask how linearly decodable that direction is from a model's residual stream.

The pipeline has two steps:

* **`extract_pca.py`** — for a chosen model, record the top-K PCA components of the
  last-token residual stream (per layer, K set with `--npca`, default 128), in pure
  autocomplete mode, over a grid of location-style prompts × objects. Produces one
  compact `<model>_pca128.npz` per model.
* **`correlations.py`** — turn those `.npz` files into per-layer, per-direction
  correlation / density tables that show *where* (which PCA directions, which layers)
  the sky-direction (or object-type) signal lives.
* **`pc_table.py`** — small utility: for one model/layer/principal-component, dump an
  objects × prompts table of that PC's value (handy for eyeballing what a single
  direction encodes).
* **`extract_pca_gptoss_harmony.py`** — a gpt-oss-specific variant of the extractor.
  gpt-oss is post-trained on the *harmony* format, so bare-text autocomplete is out of
  distribution; this wraps each prompt in a minimal harmony conversation and seeds the
  phrase into the assistant `final` channel (empty `analysis` channel before it). It
  writes `gptoss120b_harmony_pca128.npz`, a drop-in for `correlations.py`.

## Install

```bash
pip install -r requirements.txt
# FP8 checkpoints (e.g. Qwen3-235B-*-FP8) additionally need:
pip install "kernels>=0.12,<0.13"
```

## Usage

### 1. Extract PCA features

```bash
# a model from the built-in registry
python extract_pca.py --model qwen32b

# choose how many PCA components to keep (default 128)
python extract_pca.py --model qwen32b --npca 64

# multi-GPU sharding for large models
CUDA_VISIBLE_DEVICES=0,1 python extract_pca.py --model mistrallarge123b --batch 16

# any other HuggingFace causal LM
python extract_pca.py --hf-repo some/other-model --name mymodel
```

Output: `pca/<name>_pca<K>.npz` (where `K` is `--npca`) containing

| key | shape | meaning |
|---|---|---|
| `pca` | `(n_samples, n_layers+1, K)` | top-K PCA of the standardized residual stream, per layer |
| `obj_ids` | `(n_samples,)` | index into `names` for each sample |
| `pr_ids` | `(n_samples,)` | prompt-template index for each sample |
| `Yunit` | `(n_objects, 3)` | unit sky-direction of each object |
| `types` | `(n_objects,)` | `star` / `constellation` / `other` |
| `names` | `(n_objects,)` | object name |

`n_samples = n_prompts × n_objects`.

### 2. Correlation / density tables

`correlations.py` reads a `<model>_pca128.npz` (always the 128-component file; `--npca`
selects how many leading directions to use/show) and writes one per-layer table:

```bash
python correlations.py --model qwen235b --regime sky_cov_scaled --npca 128 \
    --indir PCA128 --out correlations
python correlations.py --model qwen235b --regime sky_r2 --npca 8
```

Output: `<out>/<model>_<regime>_<N>.csv`. The four regimes:

| regime | target | PCA metric | columns |
|---|---|---|---|
| `sky_cov_scaled` | sky direction (whitened to Cov = I₃) | rescale each layer as a whole so PC-1 has variance 1 — keeps relative variances | `layer, trace, d1..dN` |
| `sky_cov_equal` | sky direction (whitened) | whiten each PCA direction to variance 1 — per-direction squared canonical correlation | `layer, trace, d1..dN` |
| `type_cov_equal` | object type (star +1 / constellation −1 / other 0, standardized) | whiten each PCA direction | `layer, trace, d1..dN` |
| `sky_r2` | raw sky unit vector | first N directions, leave-one-object-out OLS (no ridge) | `layer, r2, angerr_deg, angerr_median_deg` |

For the density regimes `d_j = Σ_a Cov(w_j, target_a)²` on the (metric-transformed)
PCA direction `j`; `trace` is their sum.

## Models

The registry in `extract_pca.py` and `correlations.py` (`--model` keys): `qwen32b`,
`qwen235b`, `gptoss120b`, `llama33_70b`, `mixtral8x22b`, `mistrallarge123b`, `glm45air`.
`extract_pca.py` also accepts any other causal LM via `--hf-repo`.

## Data

* `data/astro_objects.csv` — 188 objects (named stars with V < 2.5, all 88 constellations,
  and 15 deep-sky objects) with equatorial coordinates and the unit direction vector.
* `data/astro_prompts_location.csv` — location-style prompt templates; `X` marks where the
  object name is substituted.
