#!/usr/bin/env python
"""
Decodable Sky -- celestial-coordinate recall test.

A behavioural counterpart to the residual-stream probing: instead of asking
*where the model represents* an object, this asks the model *what it knows*.
For every non-constellation object X in the catalogue, each thinking/reasoning
model is prompted, in its own chat template with reasoning enabled:

    "Give the right ascension and declination of X with no commentary."

and the full answer is recorded. `radec_score.py` then parses the RA/Dec and
computes the great-circle angular error against the true J2000 position.

Only the reasoning-capable models from the project registry are meaningful here
(the plain instruct models -- llama33_70b, mixtral8x22b, mistrallarge123b -- have
no thinking mode and are omitted).

Usage
-----
    python radec_recall.py --model qwen32b
    python radec_recall.py --model qwen235b            # runs on 2 GPUs (tp=2)
    python radec_recall.py --model gptoss120b --objects data/astro_objects.csv

Requirements
------------
    pip install vllm            # plus the project requirements.txt
  vLLM is used (not plain transformers) because these are large FP8 / MXFP4
  checkpoints generating long reasoning traces; it handles the quant formats and
  batched decoding far better. On Blackwell (B200/B300) the FlashInfer backend is
  selected automatically and JIT-compiles kernels on first load (needs nvcc + a
  C compiler on PATH).

Output (<out>/radec_answers_<model>.csv)
----------------------------------------
    name, type, true_ra_deg, true_dec_deg, answer
  where `answer` is the model's full generated text (reasoning + committed answer).
"""
import argparse
import csv
import os

# vLLM forces the FlashInfer attention backend on Blackwell regardless of this,
# but on other GPUs it keeps us on a prebuilt (no-JIT) kernel path.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

# Reasoning-capable subset of the project registry, with the generation settings
# each family needs. tp = tensor-parallel GPUs; ctk = chat_template_kwargs (this
# is where each family's thinking / reasoning mode is switched on).
MODELS = {
    "qwen32b": dict(
        hf="Qwen/Qwen3-32B", tp=1,
        ctk={"enable_thinking": True},
        temp=0.6, top_p=0.95, top_k=20,
    ),
    "qwen235b": dict(
        # Dedicated "Thinking" variant: the chat template always emits <think>,
        # so no enable_thinking toggle is passed (it would error).
        hf="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", tp=2,
        ctk={},
        temp=0.6, top_p=0.95, top_k=20,
    ),
    "gptoss120b": dict(
        # Harmony format: reasoning lives in the analysis channel, the answer in
        # the final channel. reasoning_effort high.
        hf="openai/gpt-oss-120b", tp=1,
        ctk={"reasoning_effort": "high"},
        temp=1.0, top_p=1.0, top_k=-1,
    ),
    "glm45air": dict(
        # GLM-4.5 hybrid reasoning (thinking on by default). tp=1: at tp=2 the
        # FlashInfer MoE kernel rejects the per-expert intermediate size (64) as
        # not a multiple of 128; the full 106B bf16 (~212 GB) fits one 275 GB GPU.
        hf="zai-org/GLM-4.5-Air", tp=1,
        ctk={},
        temp=0.6, top_p=0.95, top_k=-1,
    ),
}

PROMPT = "Give the right ascension and declination of {X} with no commentary."
MAX_TOKENS = 8192


def load_objects(path):
    """Return [(name, type, true_ra_deg, true_dec_deg)] excluding constellations."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "constellation":
                continue
            rows.append((r["name"], r["type"],
                         float(r["l_ra_deg"]), float(r["a_dec_deg"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ask a thinking model for each object's RA/Dec.")
    ap.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="reasoning-capable model key from the project registry")
    ap.add_argument("--objects", default="data/astro_objects.csv")
    ap.add_argument("--out", default="recall", help="output directory")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    objs = load_objects(args.objects)
    os.makedirs(args.out, exist_ok=True)
    print(f"[{args.model}] {cfg['hf']}: {len(objs)} objects", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["hf"],
        tensor_parallel_size=cfg["tp"],
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        trust_remote_code=True,
        # Blackwell FlashInfer autotune wants nvcc; skip it (kernels still JIT).
        kernel_config={"enable_flashinfer_autotune": False},
    )
    sp = SamplingParams(temperature=cfg["temp"], top_p=cfg["top_p"],
                        top_k=cfg["top_k"], max_tokens=MAX_TOKENS)

    convos = [[{"role": "user", "content": PROMPT.format(X=name)}]
              for (name, _t, _ra, _dec) in objs]
    chat_kwargs = {"chat_template_kwargs": cfg["ctk"]} if cfg["ctk"] else {}
    outs = llm.chat(convos, sp, **chat_kwargs)

    out_csv = os.path.join(args.out, f"radec_answers_{args.model}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "true_ra_deg", "true_dec_deg", "answer"])
        for (name, typ, ra, dec), o in zip(objs, outs):
            w.writerow([name, typ, f"{ra:.4f}", f"{dec:.4f}", o.outputs[0].text])
    print(f"saved {out_csv}  ({len(objs)} rows)", flush=True)


if __name__ == "__main__":
    main()
