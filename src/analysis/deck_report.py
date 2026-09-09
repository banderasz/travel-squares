#!/usr/bin/env python3
"""The four numbers that say whether a deck plays well.

    python -m src.analysis.deck_report --deck decks/grown_t20.json

Two of them should be LARGE (the game should reward playing well) and two
should be SMALL (it should not reward being lucky). Everything is in points of
final score, so they are directly comparable to each other.
"""
import argparse

import numpy as np

from src.deck.evaluate import measure
from src.deck.simulation import evaluate_placements_batch, get_paths, load_cards


def report(deck_path, hands=400, placements=1000, rounds=40, seed=0):
    cd, paths = load_cards(deck_path), get_paths()
    rng = np.random.default_rng(seed)

    # --- skill side: take fixed hands and lay them out many different ways ---
    deal = [rng.choice(cd['n_cards'], 6, replace=False) for _ in range(hands)]
    ids = np.repeat(np.asarray(deal, dtype=np.int32), placements, axis=0)
    rots = rng.integers(0, 4, size=(len(ids), 6)).astype(np.int8)
    pth = rng.integers(0, len(paths), size=len(ids)).astype(np.int32)
    tb = rng.integers(0, 32, size=len(ids)).astype(np.int8)
    s = np.asarray(evaluate_placements_batch(ids, rots, pth, tb, paths, cd))
    s = s.reshape(len(deal), placements)
    s.sort(axis=1)

    lo, hi = int(0.50 * placements), int(0.85 * placements)
    placement_sd = float(s.std(axis=1).mean())
    skill_gap = float((s[:, hi] - s[:, lo]).mean())
    luck_sd = float(s[:, lo:hi].mean(axis=1).std())

    # --- fairness side: what each card is worth ---
    m = measure(deck_path, None, 20, 200, rounds, 300, seed=seed + 1)
    per_card, means = m['middle'], m['middle_mean']
    best, worst = int(means.argmax()), int(means.argmin())
    b, w = per_card[best], per_card[worst]
    upset = float((rng.choice(w, 200000) > rng.choice(b, 200000)).mean())

    return {
        'mean': float(means.mean()),
        'card_spread': float(means.max() - means.min()),
        'card_sd': float(means.std()),
        'placement_sd': placement_sd,
        'skill_gap': skill_gap,
        'luck_sd': luck_sd,
        'upset': upset,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deck', default='decks/grown_t20.json')
    p.add_argument('--rounds', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    r = report(a.deck, rounds=a.rounds, seed=a.seed)

    def line(label, value, want, note):
        print(f"  {label:<26} {value:>7.2f}   {want:<8} {note}")

    print(f"\n  {a.deck} — average score {r['mean']:.1f}\n")
    print(f"  {'metric':<26} {'value':>7}   {'want':<8} what it means")
    print(f"  {'-'*26} {'-'*7}   {'-'*8} {'-'*40}")
    line("placement matters", r['placement_sd'], "BIG",
         "how much the same hand swings on layout")
    line("reward for playing well", r['skill_gap'], "BIG",
         "good play vs adequate play")
    line("luck of the draw", r['luck_sd'], "small",
         "how much the cards dealt decide it")
    line("best card - worst card", r['card_spread'], "small",
         "is any card plainly better")
    print()
    print(f"  skill-to-luck ratio        {r['skill_gap']/max(r['luck_sd'],1e-9):>7.1f}   "
          f"BIG      >3 means play beats the deal")
    print(f"  worst card upset rate      {r['upset']*100:>6.1f}%   ~50%     "
          f"how often the worst card still wins")
    print()


if __name__ == '__main__':
    main()
