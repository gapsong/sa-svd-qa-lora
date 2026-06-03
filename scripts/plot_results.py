#!/usr/bin/env python
"""
Render results/results.json into assets/hero.png.

The chart is the first thing in the README: two bars, baseline vs SA-SVD,
WikiText-2 perplexity, log scale so the gap between ~170 and ~26 is legible.

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

    results = json.loads(RESULTS_PATH.read_text())
    order = [m for m in ("baseline", "sa_svd") if m in results]
    labels = {"baseline": "QA-LoRA\n(random init)", "sa_svd": "SA-SVD\n(ours)"}
    colors = {"baseline": "#C0392B", "sa_svd": "#16579B"}

    names = [labels[m] for m in order]
    ppls = [results[m]["wikitext_ppl"] for m in order]
    bar_colors = [colors[m] for m in order]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(names, ppls, color=bar_colors, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("WikiText-2 perplexity  (lower is better, log scale)")

    rank = results[order[0]].get("rank", "?")
    gs = results[order[0]].get("group_size", "?")
    ax.set_title(f"2-bit SmolLM2-1.7B repair  (rank={rank}, group={gs})")

    ax.axhline(USABLE_THRESHOLD, ls="--", lw=1, color="#555555")
    ax.text(len(order) - 0.5, USABLE_THRESHOLD * 1.05,
            "usable LM threshold", ha="right", va="bottom",
            fontsize=8, color="#555555")

    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, ppl * 1.05,
                f"{ppl:.1f}", ha="center", va="bottom", fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
