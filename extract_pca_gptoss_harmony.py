#!/usr/bin/env python
"""
Decodable Sky -- residual-stream PCA extractor for gpt-oss in its native HARMONY format.

gpt-oss was post-trained almost exclusively on the harmony response format, so feeding it
bare text (as extract_pca.py does for the bf16 models) is out of distribution. This variant
wraps each (prompt x object) string in a minimal-but-valid harmony conversation and reads the
residual stream at the point where the model is *continuing* the phrase inside its `final`
channel -- the closest in-distribution analogue of pure autocomplete:

    <|start|>system ... Reasoning: low ... <|end|>
    <|start|>user<|message|><|end|>                                   # minimal user turn
    <|start|>assistant<|channel|>analysis<|message|><|end|>           # empty analysis channel
    <|start|>assistant<|channel|>final<|message|>{prompt with object} # <- last token read here

Everything else (last-token residual per layer -> top-128 PCA) matches extract_pca.py, so the
output .npz is a drop-in for correlations.py. Output name defaults to gptoss120b_harmony_pca128
so it does not overwrite the naive-format run.

Usage
-----
    CUDA_VISIBLE_DEVICES=0,1 python extract_pca_gptoss_harmony.py
"""
import argparse
import csv
import os

import numpy as np

REPO = "openai/gpt-oss-120b"
NPCA = 128
# empty analysis channel, then open the assistant `final` channel to be seeded with the phrase
SEED_MID = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"


def load_objects(path):
    names, types, y = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names.append(r["name"]); types.append(r["type"])
            y.append([float(r["x_sinA"]), float(r["y_cosAsinL"]), float(r["z_cosAcosL"])])
    return names, types, np.asarray(y, np.float32)


def load_prompts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().strip('"')
            if s:
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="gptoss120b_harmony", help="output prefix")
    ap.add_argument("--prompts", default="data/astro_prompts_location.csv")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="pca")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--user-content", default="", help="content of the minimal user turn")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from sklearn.decomposition import PCA
    from transformers import AutoModelForCausalLM, AutoTokenizer

    names, types, yunit = load_objects(args.objects)
    prompts = load_prompts(args.prompts)

    tok = AutoTokenizer.from_pretrained(REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"        # left-pad so the seeded phrase's last token aligns
    tok.truncation_side = "left"     # if ever too long, drop the system prefix, never the phrase

    # harmony prefix up to "<|start|>assistant" (system auto-injected by the template)
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": args.user_content}],
        tokenize=False, add_generation_prompt=True, reasoning_effort=args.reasoning_effort,
    )

    def wrap(phrase):
        return prefix + SEED_MID + phrase

    # sanity check on one example before the heavy load
    example = wrap(prompts[0].replace("X", names[0]))
    ids0 = tok(example, add_special_tokens=False)["input_ids"]
    ch, msg = (tok.convert_tokens_to_ids(t) for t in ("<|channel|>", "<|message|>"))
    # 2 channels (analysis, final); 4 messages (system, user, analysis, final)
    assert ids0.count(ch) == 2 and ids0.count(msg) == 4, f"harmony structure off: {example!r}"
    print(f"example ({len(ids0)} tok): {example!r}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(REPO, dtype="auto", device_map="auto").eval()
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    print(f"loaded: {n_layers} hidden states, hidden={hidden}, "
          f"{len(prompts)} prompts x {len(names)} objects", flush=True)

    texts, obj_ids, pr_ids = [], [], []
    for pi, p in enumerate(prompts):
        for oi, nm in enumerate(names):
            texts.append(wrap(p.replace("X", nm))); obj_ids.append(oi); pr_ids.append(pi)
    n = len(texts)

    acts = np.zeros((n, n_layers, hidden), np.float16)
    with torch.no_grad():
        for s in range(0, n, args.batch):
            batch = texts[s:s + args.batch]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            for li, h in enumerate(hs):
                acts[s:s + len(batch), li, :] = h[:, -1, :].to(torch.float16).cpu().numpy()
            if (s // args.batch) % 10 == 0:
                print(f"  {s + len(batch)}/{n}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("computing PCA per layer ...", flush=True)
    pca = np.zeros((n, n_layers, NPCA), np.float16)
    for li in range(n_layers):
        x = acts[:, li, :].astype(np.float32)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        k = min(NPCA, x.shape[1], x.shape[0] - 1)
        pca[:, li, :k] = PCA(n_components=k, random_state=0).fit_transform(x).astype(np.float16)

    out_path = os.path.join(args.out, f"{args.name}_pca{NPCA}.npz")
    np.savez_compressed(
        out_path, pca=pca,
        obj_ids=np.asarray(obj_ids), pr_ids=np.asarray(pr_ids),
        Yunit=yunit, types=np.asarray(types), names=np.asarray(names),
    )
    print(f"saved {out_path}  shape={pca.shape}", flush=True)


if __name__ == "__main__":
    main()
