#!/usr/bin/env python
"""
Reproduce the headline result: SA-SVD repairs a 2-bit SmolLM2-1.7B where the
standard QA-LoRA baseline fails.

What it does
------------
For each method in {baseline, sa_svd}:
  1. Load SmolLM2-1.7B (FP16).
  2. (sa_svd only) Replace target linear weights with their SA-SVD residual and
     stash the adapter components.
  3. Quantize to 2-bit with GPTQ.
  4. Attach a QA-LoRA adapter; (sa_svd) overwrite A/B with the SA-SVD components.
  5. Fine-tune on a 10k-sample Alpaca subset.
  6. Evaluate WikiText-2 perplexity (the honest repair signal).

Results are written to results/results.json and rendered by
scripts/plot_results.py into assets/hero.png.

Expected outcome (rank=16, group=16), matching the thesis:
    baseline QA-LoRA : WikiText PPL ~170  (still broken)
    SA-SVD           : WikiText PPL ~26   (repaired)

Hardware: single 24 GB GPU (e.g. RTX 4090). Approx 1-2 h for both runs.

Usage:
    python scripts/reproduce.py --method both
    python scripts/reproduce.py --method sa_svd --max-steps 200   # quicker
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]
RESULTS_PATH = ROOT / "results" / "results.json"

# Per-model max learning rate (thesis Table 4.1). Qwen2 trains unstably at 1e-4
# (loss spikes), so it needs 3e-5; TinyLlama and SmolLM2 use 1e-4. Matched by
# substring against --model-id. An explicit --lr overrides this.
DEFAULT_LR = 1e-4
MODEL_LR = {"Qwen2-1.5B": 3e-5}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method",
                   choices=["baseline", "sa_svd", "both", "residual_random",
                            "random_matched"],
                   default="both",
                   help="residual_random is the E2 ablation arm "
                        "(research/red-team.md): SA-SVD residual base, but "
                        "PEFT default adapter init instead of SA-SVD A/B. "
                        "random_matched is the E4 arm: SA-SVD spectrum with "
                        "random orthonormal directions.")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=None,
                   help="Max learning rate. If unset, uses the thesis per-model "
                        "default (3e-5 for Qwen2, else 1e-4).")
    p.add_argument("--n-train-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-path", type=Path, default=RESULTS_PATH,
                   help="Where to accumulate the results JSON "
                        "(default: results/results.json).")
    return p.parse_args()


def run_one(method: str, args) -> dict:
    """Train + evaluate one method. Imports heavy deps lazily so --help is fast."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sa_svd import apply_sa_svd_to_base, find_target_linears, write_adapter_weights

    torch.manual_seed(args.seed)
    print(f"\n=== Method: {method} ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    sa_svd_adapters = None
    if method in ("sa_svd", "residual_random"):
        # residual_random swaps in the residual base exactly like sa_svd but
        # then discards the adapter components, leaving PEFT's default init.
        names = find_target_linears(model, TARGETS)
        print(f"Applying SA-SVD to {len(names)} layers ...")
        sa_svd_adapters = apply_sa_svd_to_base(
            model, names, rank=args.rank, group_size=args.group_size
        )
    elif method == "random_matched":
        # E4: SA-SVD's singular values, random orthonormal directions.
        sys.path.insert(0, str(ROOT / "research"))
        from e4_random_matched import apply_random_matched_to_base

        names = find_target_linears(model, TARGETS)
        print(f"Applying random-matched init to {len(names)} layers ...")
        sa_svd_adapters = apply_random_matched_to_base(
            model, names, rank=args.rank, group_size=args.group_size,
            seed=args.seed,
        )

    # ----- Quantize to 2-bit (GPTQ) and attach a QA-LoRA adapter. -----
    # The GPTQ + QA-LoRA wiring uses the public PEFT / GPTQModel API exactly as
    # in examples/qalora_finetuning. See README "How the pieces fit" for the
    # exact calls; kept out of this file so the SA-SVD logic stays front-and-centre.
    from pipeline import attach_qalora_2bit, evaluate_wikitext, train  # local helper

    peft_model = attach_qalora_2bit(
        model, tokenizer, rank=args.rank, group_size=args.group_size,
        seed=args.seed,
    )

    if method in ("sa_svd", "random_matched"):
        n = write_adapter_weights(peft_model, sa_svd_adapters)
        print(f"Wrote {method} init into {n} adapter layers.")

    # ----- Fine-tune on Alpaca subset. -----
    data = load_dataset("tatsu-lab/alpaca", split=f"train[:{args.n_train_samples}]")
    train(peft_model, tokenizer, data, args)

    # ----- Evaluate WikiText-2 perplexity. -----
    ppl = evaluate_wikitext(peft_model, tokenizer)
    print(f"[{method}] WikiText-2 perplexity: {ppl:.2f}")
    return {"method": method, "model": args.model_id, "rank": args.rank,
            "group_size": args.group_size, "wikitext_ppl": ppl}


def main():
    args = parse_args()
    methods = ["baseline", "sa_svd"] if args.method == "both" else [args.method]

    if args.lr is None:
        args.lr = next((lr for key, lr in MODEL_LR.items() if key in args.model_id),
                       DEFAULT_LR)
        print(f"Using learning rate {args.lr:g} for {args.model_id}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.results_path.exists():
        results = json.loads(args.results_path.read_text())

    # Results are keyed by model so multiple models can accumulate in one file:
    # {model_name: {method: {...}}}.
    model_key = args.model_id.split("/")[-1]
    for m in methods:
        results.setdefault(model_key, {})[m] = run_one(m, args)
        args.results_path.write_text(json.dumps(results, indent=2))
        print(f"Saved -> {args.results_path}")

    print("\nDone. Render the chart with: python scripts/plot_results.py")


if __name__ == "__main__":
    main()
