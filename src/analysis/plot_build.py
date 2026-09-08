#!/usr/bin/env python3
"""Compare a random deck against a generated one, as value distributions.

Runs src/deck/build.py and draws both results as ridgelines on a shared axis,
so the question "did generating actually make the cards more equal?" is
answered by looking at how much the rows overlap.

    python -m src.analysis.plot_build --target 32
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.deck.build import build
from src.analysis.plot_card_values import (BASELINE, GRID, INK, MUTED, RAMP,
                                           SURFACE, ridgeline)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', type=float, default=32.0)
    p.add_argument('--cards', type=int, default=20)
    p.add_argument('--calibration', type=int, default=60)
    p.add_argument('--candidates', type=int, default=4000)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='analysis_random_vs_generated.png')
    a = p.parse_args()

    deck, gen_result, gen_means, base_result, base_means, model = build(
        a.target, a.cards, a.calibration, a.candidates,
        pool=20, hands=200, rounds=10, placements=300,
        seed=a.seed, iterations=a.iterations)

    gen, base = gen_result['middle'], base_result['middle']
    gen_order = sorted(range(len(gen)), key=lambda c: -gen[c].mean())
    base_order = sorted(range(len(base)), key=lambda c: -base[c].mean())

    lo = min(v.min() for v in list(gen) + list(base))
    hi = max(v.max() for v in list(gen) + list(base))
    pad = (hi - lo) * 0.04
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9), facecolor=SURFACE, sharex=True)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    ridgeline(axes[0], base, base_order,
              f"{a.cards} random cards   spread {base_means.max()-base_means.min():.2f}",
              xlim, 48)
    ridgeline(axes[1], gen, gen_order,
              f"{a.cards} generated cards   spread {gen_means.max()-gen_means.min():.2f}",
              xlim, 48)

    fig.suptitle(
        "Does generating make the cards equal?  Card value distributions, "
        "competent-player band\n"
        f"random: spread {base_means.max()-base_means.min():.2f}, "
        f"sd {base_means.std():.2f}   ·   "
        f"generated: spread {gen_means.max()-gen_means.min():.2f}, "
        f"sd {gen_means.std():.2f}   ·   "
        f"each card measured over {len(gen[0])} hands",
        fontsize=12, color=INK, x=0.012, ha='left', y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"\n  wrote {a.out}")


if __name__ == '__main__':
    main()
