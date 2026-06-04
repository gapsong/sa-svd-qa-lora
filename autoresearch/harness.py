#!/usr/bin/env python
"""
Autoresearch harness (FIXED, do not modify during research).

Modeled on karpathy/autoresearch: one mutable file, one metric, fixed budget.
Here the mutable file is autoresearch/init_fn.py and the metric is
WikiText-subset perplexity of a 2-bit QA-LoRA fine-tuned SmolLM2-135M.

Contract with init_fn.build_init(weight, rank, group_size, seed):
  returns (lora_A, lora_B) as float32 tensors of shapes
  (rank, in_features // group_size) and (out_features, rank).
  The harness derives the base as residual = W - B * expand(A), so the
  step-0 function ALWAYS equals W regardless of what init_fn returns.
  The research surface is purely the adapter parameterization.

Everything else is fixed: model, data, steps, batch, lr, rank, group size,
eval. Prints exactly one line starting with METRIC on success.

Usage: python autoresearch/harness.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "autoresearch"))

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


@torch.no_grad()
def evaluate_wikitext_subset(peft_model, tokenizer, *, max_tokens: int,
                             stride: int = 512, max_length: int = 1024) -> float:
    """Same sliding-window NLL as pipeline.evaluate_wikitext, on a token cap.

    Keeps the known overlap double-counting quirk so numbers stay internally
    comparable; the loop only ever compares harness runs to harness runs.
    """
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens]
    enc = enc.to(peft_model.device)

    nlls, n_tokens = [], 0
    peft_model.eval()
    for begin in range(0, enc.size(1), stride):
        end = min(begin + max_length, enc.size(1))
        trg_len = end - begin
        input_ids = enc[:, begin:end]
        out = peft_model(input_ids, labels=input_ids.clone())
        nlls.append(out.loss * trg_len)
        n_tokens += trg_len
        if end == enc.size(1):
            break
    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    # Fixed budget knobs. Change them only by editing program.md first and
    # restarting the whole journal; mid-stream changes break comparability.
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--n-train-samples", type=int, default=2400)
    p.add_argument("--eval-tokens", type=int, default=60_000)
    args = p.parse_args()

    t0 = time.time()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import init_fn  # THE MUTABLE FILE
    from pipeline import attach_qalora_2bit, train
    from sa_svd import find_target_linears, write_adapter_weights

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    # Apply the candidate init. Residual derived here: function preserved.
    names = find_target_linears(model, TARGETS)
    adapters = {}
    for name in names:
        layer = model.get_submodule(name)
        W = layer.weight.data.to(torch.float32)
        out_f, in_f = W.shape
        n_groups = in_f // args.group_size

        lora_A, lora_B = init_fn.build_init(
            W, rank=args.rank, group_size=args.group_size, seed=args.seed
        )
        lora_A = lora_A.to(device=W.device, dtype=torch.float32)
        lora_B = lora_B.to(device=W.device, dtype=torch.float32)
        if lora_A.shape != (args.rank, n_groups) or lora_B.shape != (out_f, args.rank):
            raise ValueError(
                f"init_fn shape contract violated on {name}: "
                f"A {tuple(lora_A.shape)} != {(args.rank, n_groups)} or "
                f"B {tuple(lora_B.shape)} != {(out_f, args.rank)}"
            )
        component = (lora_B @ lora_A).repeat_interleave(args.group_size, dim=1)
        layer.weight.data = (W - component).to(layer.weight.dtype)
        adapters[name] = {"lora_A": lora_A, "lora_B": lora_B}

    peft_model = attach_qalora_2bit(
        model, tokenizer, rank=args.rank, group_size=args.group_size,
        seed=args.seed,
    )
    n = write_adapter_weights(peft_model, adapters)
    if n != len(names):
        raise RuntimeError(f"wrote {n} adapter layers, expected {len(names)}")

    data = load_dataset("tatsu-lab/alpaca", split=f"train[:{args.n_train_samples}]")
    train(peft_model, tokenizer, data, args)

    ppl = evaluate_wikitext_subset(peft_model, tokenizer,
                                   max_tokens=args.eval_tokens)
    print(f"METRIC val_ppl={ppl:.4f} seed={args.seed} "
          f"steps={args.max_steps} time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
