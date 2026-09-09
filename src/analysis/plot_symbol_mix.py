#!/usr/bin/env python3
"""Compare the symbol make-up of the grown decks against the older 20_25 deck.

Balancing card values could have succeeded by making every card the same, which
would be a worse game. These two views check it did not:

  left   average count of each symbol per card, against the generation weight.
         Shows whether the grown decks drifted toward particular symbols.
  right  how many symbols a card carries. A wide spread means the deck still
         holds sparse and dense cards, not 120 copies of the average.

    python -m src.analysis.plot_symbol_mix
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.deck.value_model import features
from src.symbols import Symbols, NUMBER_OF_SYMBOLS_IN_PLAY as TOTAL

SURFACE, INK, MUTED, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'
# categorical slots 1-3, validated all-pairs on the light surface
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
NAME2SYM = {'food': Symbols.CIRCLE, 'treasure': Symbols.SQUARE, 'weapon': Symbols.STAR,
            'coin': Symbols.SUN, 'parrot': Symbols.DIAMOND, 'rum': Symbols.TRIANGLE,
            'rat': Symbols.X, 'snake': Symbols.MOON, 'mask': Symbols.SKULL}


def tidy(ax, xlabel='', ylabel=''):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(BASELINE)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='analysis_symbol_mix.png')
    a = p.parse_args()

    decks = {'20_25 (older deck)': 'decks/pirate_20_25.json',
             'grown — analytic': 'decks/grown_analytic.json',
             'grown — measured': 'decks/grown_measured.json'}
    feats = {k: [features(c) for c in json.load(open(p))] for k, p in decks.items()}

    names = list(NAME2SYM) + ['arrows']
    target = [16 * NAME2SYM[n].weight / TOTAL for n in NAME2SYM]
    target.append(16 * sum(s.weight for s in Symbols.arrows()) / TOTAL)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1.7, 1]})

    ax = axes[0]
    x = np.arange(len(names))
    width = 0.26
    for i, (label, fs) in enumerate(feats.items()):
        vals = [np.mean([f['n_' + n] for f in fs]) for n in names]
        ax.bar(x + (i - 1) * width, vals, width * 0.9, label=label,
               color=SERIES[i], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.plot(x, target, 'D', color=INK, markersize=5, zorder=5,
            label='generation weight (target)', linestyle='none')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
    tidy(ax, ylabel='average count per card')
    ax.set_title("Symbol mix — did balancing skew the deck?",
                 fontsize=11, color=INK, loc='left', pad=10)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    ax = axes[1]
    bins = np.arange(-0.5, 17.5, 1)
    for i, (label, fs) in enumerate(feats.items()):
        per_card = [sum(f['n_' + n] for n in names) for f in fs]
        ax.hist(per_card, bins=bins, histtype='step', linewidth=2.2,
                color=SERIES[i], label=f"{label}  (sd {np.std(per_card):.2f})", zorder=3)
    ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
    tidy(ax, xlabel='symbols on a card', ylabel='cards')
    ax.set_title("Variety — are the cards still different from each other?",
                 fontsize=11, color=INK, loc='left', pad=10)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"  wrote {a.out}")


if __name__ == '__main__':
    main()
