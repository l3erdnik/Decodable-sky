#!/usr/bin/env python
"""
Decodable Sky -- residual-stream PCA extractor.

For a chosen causal LLM, run every (location-prompt x astronomical-object) string
through the model in pure autocomplete mode (no chat template), grab the last-token
residual stream at every layer, and keep the top-K PCA components per layer
(K is set with --npca, default 128). The result is one compact .npz per model that
later probing / LDA analyses read.

Usage
-----
    python extract_pca.py --model qwen32b
    python extract_pca.py --model qwen32b --npca 64
    python extract_pca.py --model llama70b --out pca --batch 32
    python extract_pca.py --hf-repo some/other-model --name mymodel   # any HF causal LM

Requirements
------------
    pip install torch transformers scikit-learn numpy
  * FP8 checkpoints (e.g. Qwen3-235B-*-FP8) also need:  pip install "kernels>=0.12,<0.13"
  * Multi-GPU: export CUDA_VISIBLE_DEVICES=0,1,...  -- weights are sharded with
    device_map="auto"; the residual stream is read the same way regardless of sharding.

Output (<out>/<name>_pca<K>.npz)
--------------------------------
    pca      float16 (n_samples, n_layers+1, K)    -- top-K PCA of the standardized
                                                      residual stream, per layer
    obj_ids  int     (n_samples,)  index into `names` for each sample
    pr_ids   int     (n_samples,)  index of the prompt template for each sample
    Yunit    float32 (n_objects, 3) unit direction (sin Dec, cosDec sinRA, cosDec cosRA)
    types    str     (n_objects,)  "star" / "constellation" / "other"
    names    str     (n_objects,)  object name
"""
import argparse
import csv
import os

import numpy as np

# Registry of the models compared in this project. Any of these -- or any other
# HuggingFace causal LM via --hf-repo -- can be selected.
MODELS = {
    "qwen32b":      "Qwen/Qwen3-32B",
    "qwen235b":     "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "gptoss120b":   "openai/gpt-oss-120b",
    "llama70b":     "unsloth/Llama-3.3-70B-Instruct",
    "mixtral8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistrallarge": "mistralai/Mistral-Large-Instruct-2411",
    "glm45air":     "zai-org/GLM-4.5-Air",
}


def load_objects(path):
    names, types, y = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names.append(r["name"])
            types.append(r["type"])
            y.append([float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])])
    return names, types, np.asarray(y, np.float32)


def load_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().strip('"')
            if s:
                prompts.append(s)
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Extract top-128 residual-stream PCA for an LLM.")
    ap.add_argument("--model", choices=sorted(MODELS), help="model key from the built-in registry")
    ap.add_argument("--hf-repo", help="any HuggingFace causal-LM repo id (overrides --model)")
    ap.add_argument("--name", help="output prefix (defaults to the model key / repo name)")
    ap.add_argument("--prompts", default="data/astro_prompts_location.csv")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="pca", help="output directory")
    ap.add_argument("--npca", type=int, default=128, help="number of top PCA components to keep")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--max-memory-gib", type=int, default=None,
                    help="per-GPU weight budget for device_map (leave unset for auto)")
    args = ap.parse_args()

    if not args.hf_repo and not args.model:
        ap.error("choose a model with --model <key> or --hf-repo <repo>")
    if args.npca < 1:
        ap.error("--npca must be >= 1")
    repo = args.hf_repo or MODELS[args.model]
    name = args.name or args.model or repo.split("/")[-1]
    os.makedirs(args.out, exist_ok=True)

    import torch
    from sklearn.decomposition import PCA
    from transformers import AutoModelForCausalLM, AutoTokenizer

    names, types, yunit = load_objects(args.objects)
    prompts = load_prompts(args.prompts)

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad so the last token aligns across a batch

    max_memory = None
    if args.max_memory_gib and torch.cuda.is_available():
        max_memory = {i: f"{args.max_memory_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype="auto", device_map="auto", max_memory=max_memory,
    ).eval()
    n_layers = model.config.num_hidden_layers + 1  # +1 for the embedding layer
    hidden = model.config.hidden_size
    print(f"[{name}] {repo}: {n_layers} hidden states, hidden={hidden}, "
          f"{len(prompts)} prompts x {len(names)} objects", flush=True)

    # Build every (prompt, object) string: "X" in the template is replaced by the name.
    texts, obj_ids, pr_ids = [], [], []
    for pi, p in enumerate(prompts):
        for oi, nm in enumerate(names):
            texts.append(p.replace("X", nm))
            obj_ids.append(oi)
            pr_ids.append(pi)
    n = len(texts)

    acts = np.zeros((n, n_layers, hidden), np.float16)
    with torch.no_grad():
        for s in range(0, n, args.batch):
            batch = texts[s:s + args.batch]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            for li, h in enumerate(hs):
                acts[s:s + len(batch), li, :] = h[:, -1, :].to(torch.float16).cpu().numpy()
            if (s // args.batch) % 10 == 0:
                print(f"  {s + len(batch)}/{n}", flush=True)

    # Free the model, then reduce each layer to its top-128 PCA components.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    npca = min(args.npca, hidden, n - 1)
    print(f"computing top-{npca} PCA per layer ...", flush=True)
    pca = np.zeros((n, n_layers, npca), np.float16)
    for li in range(n_layers):
        x = acts[:, li, :].astype(np.float32)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)          # standardize features
        pca[:, li, :] = PCA(n_components=npca, random_state=0).fit_transform(x).astype(np.float16)

    out_path = os.path.join(args.out, f"{name}_pca{npca}.npz")
    np.savez_compressed(
        out_path, pca=pca,
        obj_ids=np.asarray(obj_ids), pr_ids=np.asarray(pr_ids),
        Yunit=yunit, types=np.asarray(types), names=np.asarray(names),
    )
    print(f"saved {out_path}  shape={pca.shape}", flush=True)


if __name__ == "__main__":
    main()
