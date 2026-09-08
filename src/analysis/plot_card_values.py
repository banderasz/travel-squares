#!/usr/bin/env python3
"""Draw each card's value distribution, to see how similar the cards are.

A ridgeline: one row per card, ordered by mean, so overlap between cards is
visible at a glance. Two panels — the value under random placement, and under
the competent-player middle band.

Colour is sequential (one blue hue, light to dark by rank), not 20 categorical
hues: the cards are an ordered magnitude, and 20 distinct hues would be
indistinguishable under colour-vision deficiency anyway. Every row is also
directly labelled, so identity never depends on colour.

    python -m src.analysis.plot_card_values --deck decks/pirate_120_balanced.json
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.deck.evaluate import measure

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
# Blue ramp, ordinal range: nothing lighter than step 250 on a light surface.
# Validated: ordinal, light surface #fcfcfb — monotone L, all gaps >= 0.06,
# light end 2.06:1 vs surface, hue spread 4 degrees. All checks pass.
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']


def ridgeline(ax, values, order, title, xlim, bins):
    overlap = 1.9
    for row, card in enumerate(order):
        v = values[card]
        density, edges = np.histogram(v, bins=bins, range=xlim, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        colour = RAMP[int(row / max(len(order) - 1, 1) * (len(RAMP) - 1))]
        base = len(order) - row
        ax.fill_between(centres, base, base + density * overlap / max(density.max(), 1e-9),
                        color=colour, alpha=0.92, linewidth=0.8,
                        edgecolor=SURFACE, zorder=row)
        ax.plot([v.mean(), v.mean()], [base, base + 0.55], color=SURFACE,
                linewidth=1.6, zorder=row + 0.5)
        ax.text(xlim[0] + 0.12, base + 0.30, f"card {card}", fontsize=7.5,
                color=INK, va='center', zorder=100)
        ax.text(xlim[1] - 0.12, base + 0.30, f"{v.mean():.2f}", fontsize=7.5,
                color=MUTED, va='center', ha='right', zorder=100)

    grand = np.mean([values[c].mean() for c in order])
    ax.axvline(grand, color=BASELINE, linewidth=1.2, linestyle='--', zorder=0)
    ax.set_title(title, fontsize=10.5, color=INK, loc='left', pad=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.4, len(order) + 2.4)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.set_xlabel('hand value', fontsize=8.5, color=MUTED)
    ax.grid(axis='x', color=GRID, linewidth=0.6, zorder=-1)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deck', default='decks/pirate_120_balanced.json')
    p.add_argument('--cards', type=int, default=20)
    p.add_argument('--pool', type=int, default=20)
    p.add_argument('--hands', type=int, default=200)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--placements', type=int, default=300)
    p.add_argument('--bins', type=int, default=48)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='card_value_distributions.png')
    a = p.parse_args()

    r = measure(a.deck, a.cards, a.pool, a.hands, a.rounds, a.placements, seed=a.seed)
    full, middle = r['full'], r['middle']
    order = sorted(range(r['n_cards']), key=lambda c: -middle[c].mean())

    lo = min(v.min() for v in full + middle)
    hi = max(v.max() for v in full + middle)
    pad = (hi - lo) * 0.04
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9), facecolor=SURFACE, sharex=True)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    ridgeline(axes[0], full, order, 'Random placement  (mean of all layouts)',
              xlim, a.bins)
    ridgeline(axes[1], middle, order,
              'Competent player  (middle band, p50–p85)', xlim, a.bins)

    fs, ms = [full[c].mean() for c in order], [middle[c].mean() for c in order]
    fig.suptitle(
        f"Per-card value distributions — {r['n_cards']} cards, "
        f"{len(full[0])} hands each\n"
        f"spread between cards: {max(fs)-min(fs):.2f} random, "
        f"{max(ms)-min(ms):.2f} competent   ·   "
        f"within a card: ~{np.mean([v.std() for v in middle]):.2f}",
        fontsize=12, color=INK, x=0.012, ha='left', y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"  wrote {a.out}")


if __name__ == '__main__':
    main()
