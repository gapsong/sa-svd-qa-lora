#!/usr/bin/env python
"""
Render results/results.json into assets/hero.png.

The chart is the first thing in the README: two bars, baseline vs SA-SVD,
WikiText-2 perplexity, log scale, with the ~30 usable-LM threshold marked.

Usage:
    python scripts/plot_results.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "results.json"
OUT_PATH = ROOT / "assets" / "hero.png"

USABLE_THRESHOLD = 30.0  # a usable LM has WikiText PPL below ~30


def main():
    if not RESULTS_PATH.exists():
        sys.exit(f"No results at {RESULTS_PATH}. Run scripts/reproduce.py first.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # results.json is keyed {model: {method: {...}}} so several models can sit
    # side by side as grouped bars.
    results = json.loads(RESULTS_PATH.read_text())
    methods = ("baseline", "sa_svd")
    labels = {"baseline": "QA-LoRA (random init)", "sa_svd": "SA-SVD (ours)"}
    colors = {"baseline": "#C0392B", "sa_svd": "#16579B"}

    models = list(results.keys())
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6, 2.6 * len(models) + 1.2), 4.2))
    ppls = []
    for i, model in enumerate(models):
        for j, meth in enumerate(methods):
            if meth not in results[model]:
                continue
            ppl = results[model][meth]["wikitext_ppl"]
            ppls.append(ppl)
            x = i + (j - 0.5) * width
            ax.bar(x, ppl, width=width, color=colors[meth],
                   label=labels[meth] if i == 0 else None)
            ax.text(x, ppl * 1.05, f"{ppl:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("WikiText-2 perplexity  (lower is better, log scale)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)

    first = results[models[0]][next(iter(results[models[0]]))]
    rank, gs = first.get("rank", "?"), first.get("group_size", "?")
    ax.set_title(f"2-bit QA-LoRA fine-tune, WikiText-2  (rank={rank}, group={gs})")

    ax.axhline(USABLE_THRESHOLD, ls="--", lw=1, color="#555555")
    ax.text(-0.5 + 0.02, USABLE_THRESHOLD * 0.97, "usable LM threshold",
            ha="left", va="top", fontsize=8, color="#555555")

    # Explicit limits: keep the threshold line in frame and leave headroom for
    # the value labels so they do not collide with the title.
    ax.set_ylim(min(USABLE_THRESHOLD * 0.65, min(ppls) * 0.55),
                max(ppls) * 1.45)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
