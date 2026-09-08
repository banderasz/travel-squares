#!/usr/bin/env python3
"""Measure the distribution of values for every card in a deck.

A card does not have a value, it has a distribution. This produces that
distribution by playing each card in many hands and keeping every score.

  placement  one board: a path, a rotation per card, a stacking order.
  hand       6 cards, scored over M sampled placements. Every score is kept.
             Two summaries come out of it:
               full     mean of all M          — placement chosen at random
               middle   mean of the p50-p85 band — a competent player: avoids
                        the bad layouts, does not find the perfect one
  pool       P cards that play each other. H random hands are drawn from it,
             so each card collects roughly H*6/P values per round.
  round      the deck is re-partitioned into fresh pools. Partitions are chosen
             to even out how often each pair of cards has already met, so no
             pair is over- or under-sampled.

    python -m src.deck.evaluate --deck decks/pirate_120_balanced.json --cards 20
"""
import argparse
import json
from itertools import combinations

import numpy as np

from src.deck.simulation import evaluate_placements_batch, get_paths, load_cards

HAND = 6


def _partition(n_cards, pool, met, rng, tries=40):
    """Split cards into pools, preferring splits whose pairs have met least."""
    best, best_cost = None, None
    for _ in range(tries):
        order = rng.permutation(n_cards)
        pools = [order[i:i + pool] for i in range(0, n_cards, pool)]
        pools = [p for p in pools if len(p) >= HAND]
        cost = sum(met[a, b] for p in pools for a, b in combinations(sorted(p), 2))
        if best_cost is None or cost < best_cost:
            best, best_cost = pools, cost
    for p in best:
        for a, b in combinations(sorted(p), 2):
            met[a, b] += 1
    return best


def measure(deck_path, n_cards=None, pool=20, hands_per_pool=200, rounds=10,
            placements=300, lo=0.50, hi=0.85, seed=0):
    card_data = load_cards(deck_path)
    paths = get_paths()
    total_cards = card_data['n_cards'] if n_cards is None else min(n_cards, card_data['n_cards'])
    pool = min(pool, total_cards)

    rng = np.random.default_rng(seed)
    met = np.zeros((total_cards, total_cards), dtype=np.int32)
    full = [[] for _ in range(total_cards)]
    middle = [[] for _ in range(total_cards)]
    raw_sample = [[] for _ in range(total_cards)]

    lo_i, hi_i = int(placements * lo), int(placements * hi)

    for _ in range(rounds):
        for members in _partition(total_cards, pool, met, rng):
            hands = np.array([rng.choice(members, HAND, replace=False)
                              for _ in range(hands_per_pool)], dtype=np.int32)

            ids = np.repeat(hands, placements, axis=0)
            rots = rng.integers(0, 4, size=(len(ids), HAND)).astype(np.int8)
            pth = rng.integers(0, len(paths), size=len(ids)).astype(np.int32)
            tb = rng.integers(0, 32, size=len(ids)).astype(np.int8)

            scores = np.asarray(
                evaluate_placements_batch(ids, rots, pth, tb, paths, card_data)
            ).reshape(len(hands), placements)
            scores.sort(axis=1)

            hand_full = scores.mean(axis=1)
            hand_middle = scores[:, lo_i:hi_i].mean(axis=1)

            for h, hand in enumerate(hands):
                for card in hand:
                    full[card].append(hand_full[h])
                    middle[card].append(hand_middle[h])
                    if len(raw_sample[card]) < 4000:
                        raw_sample[card].extend(scores[h, ::20].tolist())

    return {
        'full': [np.array(v) for v in full],
        'middle': [np.array(v) for v in middle],
        'raw': [np.array(v) for v in raw_sample],
        'n_cards': total_cards,
        'settings': dict(pool=pool, hands_per_pool=hands_per_pool, rounds=rounds,
                         placements=placements, lo=lo, hi=hi),
    }


def report(result):
    full, middle = result['full'], result['middle']
    n = result['n_cards']
    print(f"\n  {n} cards, {len(full[0])} values each "
          f"({result['settings']['placements']} placements per hand)\n")
    print(f"  {'card':>4} {'full mean':>10} {'sd':>6} | {'middle mean':>12} {'sd':>6} "
          f"| {'gap':>6}")
    print(f"  {'-'*4} {'-'*10} {'-'*6} | {'-'*12} {'-'*6} | {'-'*6}")
    fm = np.array([v.mean() for v in full])
    mm = np.array([v.mean() for v in middle])
    for c in np.argsort(-mm):
        print(f"  {c:>4} {fm[c]:>10.2f} {full[c].std():>6.2f} | "
              f"{mm[c]:>12.2f} {middle[c].std():>6.2f} | {mm[c]-fm[c]:>6.2f}")
    print(f"\n  full   : spread {fm.max()-fm.min():.2f}  sd across cards {fm.std():.2f}")
    print(f"  middle : spread {mm.max()-mm.min():.2f}  sd across cards {mm.std():.2f}")
    return fm, mm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deck', default='decks/pirate_120_balanced.json')
    p.add_argument('--cards', type=int, default=20, help='Use only the first N cards')
    p.add_argument('--pool', type=int, default=20)
    p.add_argument('--hands', type=int, default=200)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--placements', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='', help='Write per-card values to this .npz')
    a = p.parse_args()

    result = measure(a.deck, a.cards, a.pool, a.hands, a.rounds, a.placements, seed=a.seed)
    report(result)
    if a.out:
        np.savez_compressed(
            a.out,
            **{f'full_{i}': v for i, v in enumerate(result['full'])},
            **{f'middle_{i}': v for i, v in enumerate(result['middle'])},
            **{f'raw_{i}': v for i, v in enumerate(result['raw'])},
        )
        print(f"\n  values -> {a.out}")


if __name__ == '__main__':
    main()
