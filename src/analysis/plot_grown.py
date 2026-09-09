#!/usr/bin/env python3
"""Compare the grown decks against a random one, as per-card value distributions.

Three panels on a shared axis: a random 120-card deck, and the two decks grown
by src/deck/grow.py. One row per card, ordered by mean. How tightly the rows
stack up is the answer to "are these cards equal?".

Below them, the same information collapsed: a histogram of the 120 card means
per deck, where the width of each bump is the spread between cards.

    python -m src.analysis.plot_grown
"""
import argparse
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.deck.evaluate import measure
from src.deck.tune import random_card
from src.deck_io import dumps_deck

SURFACE, INK, MUTED, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
# validated ordinal blue ramp (monotone L, gaps >= 0.06, light end 2.06:1)
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
# The two grown decks are the subject and take categorical slots 1 and 2
# (validated all-pairs: CVD dE 24.7, contrast >= 3:1). The random deck is
# context, so it wears the de-emphasis gray rather than a third hue.
DECK_COLOURS = ['#c3c2b7', '#eb6834', '#2a78d6']


def ridge(ax, values, title, xlim, bins=44):
    order = sorted(range(len(values)), key=lambda c: -values[c].mean())
    n = len(order)
    for row, card in enumerate(order):
        v = values[card]
        dens, edges = np.histogram(v, bins=bins, range=xlim, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        colour = RAMP[int(row / max(n - 1, 1) * (len(RAMP) - 1))]
        base = n - row
        ax.fill_between(centres, base, base + dens * 3.0 / max(dens.max(), 1e-9),
                        color=colour, alpha=0.9, linewidth=0.35,
                        edgecolor=SURFACE, zorder=row)
    means = np.array([v.mean() for v in values])
    ax.set_title(f"{title}\nspread {means.max()-means.min():.2f}   sd {means.std():.3f}",
                 fontsize=10, color=INK, loc='left', pad=8)
    ax.set_xlim(*xlim); ax.set_ylim(0.5, n + 4)
    ax.set_yticks([]); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.grid(axis='x', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel('hand value', fontsize=8.5, color=MUTED)
    return means


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rounds', type=int, default=60)
    p.add_argument('--out', default='analysis_grown_decks.png')
    a = p.parse_args()

    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    handle.write(dumps_deck([random_card() for _ in range(120)])); handle.close()

    decks = [('120 random cards', handle.name),
             ('grown — analytic heuristic', 'decks/grown_analytic.json'),
             ('grown — measured heuristic', 'decks/grown_measured.json')]
    results = [measure(path, 120, 20, 200, a.rounds, 300, seed=77)['middle']
               for _, path in decks]

    lo = min(v.min() for r in results for v in r)
    hi = max(v.max() for r in results for v in r)
    pad = (hi - lo) * 0.03
    xlim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(15, 13), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.0], hspace=0.22, wspace=0.10)

    all_means = []
    for i, ((title, _), values) in enumerate(zip(decks, results)):
        ax = fig.add_subplot(gs[0, i]); ax.set_facecolor(SURFACE)
        all_means.append(ridge(ax, values, title, xlim))

    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(SURFACE)
    for (title, _), means, colour in zip(decks, all_means, DECK_COLOURS):
        ax.hist(means, bins=40, alpha=0.8, color=colour, label=title,
                edgecolor=SURFACE, linewidth=1.2)
    ax.set_title("the 120 card means — a narrower bump means the cards are more equal",
                 fontsize=10, color=INK, loc='left', pad=8)
    ax.set_xlabel('mean value of a card', fontsize=8.5, color=MUTED)
    ax.set_ylabel('cards', fontsize=8.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(BASELINE)
    ax.grid(axis='y', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9, loc='upper right')
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle("Grown decks: is any card clearly better?   "
                 f"random spread {all_means[0].max()-all_means[0].min():.2f}  ->  "
                 f"analytic {all_means[1].max()-all_means[1].min():.2f}  ->  "
                 f"measured {all_means[2].max()-all_means[2].min():.2f}",
                 fontsize=13, color=INK, x=0.008, ha='left', y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(a.out, dpi=140, facecolor=SURFACE)
    print(f"  wrote {a.out}")


if __name__ == '__main__':
    main()
