# Decodable Sky

Probing whether large language models internally represent **astronomical objects by
their position on the sky**. Each object (a star, constellation, or deep-sky object)
is labelled by a unit direction vector `(sin Dec, cos Dec sin RA, cos Dec cos RA)`, and
we ask how linearly decodable that direction is from a model's residual stream.

This repository currently contains the **feature-extraction** step: for a chosen model,
it records the top-128 PCA components of the last-token residual stream (per layer),
in pure autocomplete mode, over a grid of location-style prompts × objects. The compact
per-model `.npz` files it produces are the input to the downstream probing / LDA analyses.

## Install

```bash
pip install -r requirements.txt
# FP8 checkpoints (e.g. Qwen3-235B-*-FP8) additionally need:
pip install "kernels>=0.12,<0.13"
```

## Usage

```bash
# a model from the built-in registry
python extract_pca.py --model qwen32b

# multi-GPU sharding for large models
CUDA_VISIBLE_DEVICES=0,1 python extract_pca.py --model mistrallarge --batch 16

# any other HuggingFace causal LM
python extract_pca.py --hf-repo some/other-model --name mymodel
```

Output: `pca128/<name>_pca128.npz` containing

| key | shape | meaning |
|---|---|---|
| `pca` | `(n_samples, n_layers+1, 128)` | top-128 PCA of the standardized residual stream, per layer |
| `obj_ids` | `(n_samples,)` | index into `names` for each sample |
| `pr_ids` | `(n_samples,)` | prompt-template index for each sample |
| `Yunit` | `(n_objects, 3)` | unit sky-direction of each object |
| `types` | `(n_objects,)` | `star` / `constellation` / `other` |
| `names` | `(n_objects,)` | object name |

`n_samples = n_prompts × n_objects`.

## Models

The registry in `extract_pca.py` (`--model` keys): `qwen32b`, `qwen235b`, `gptoss120b`,
`llama70b`, `mixtral8x22b`, `mistrallarge`, `glm45air`. Any other causal LM works via
`--hf-repo`.

## Data

* `data/astro_objects.csv` — 188 objects (named stars with V < 2.5, all 88 constellations,
  and 15 deep-sky objects) with equatorial coordinates and the unit direction vector.
* `data/astro_prompts_location.csv` — location-style prompt templates; `X` marks where the
  object name is substituted.
