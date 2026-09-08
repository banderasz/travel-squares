#!/usr/bin/env python3
"""How many values does each card end up with? Work it out before running.

Structure of the measurement, from the inside out:

  placement  one concrete board: a path, a rotation per card, a stacking order.
             Scored exactly.
  hand       6 cards. Sample M placements and keep every score. The p50-p85
             band gives the "competent player" point value; the full set's
             spread says how much placement skill matters for that hand.
  pool       a group of P cards (e.g. 20) that play against each other. Draw H
             random 6-card hands from the pool. Every hand value is credited to
             its 6 cards, so each card collects H*6/P values per pool.
  round      re-partition the deck into fresh pools and repeat, choosing
             partitions so no pair of cards meets far more often than any other.

    python -m src.deck.sampling_plan
    python -m src.deck.sampling_plan --cards 120 --pool 20 --hands 200
"""
import argparse
from itertools import combinations

# ---- measured on decks/pirate_120_balanced.json, M1 Max, 10 threads ----
THROUGHPUT = 18.7e6      # placements/sec
SIGMA_PLACEMENT = 6.47   # score spread across placements within one hand
SIGMA_BAND = 1.86        # spread inside the kept p50-p85 band
SIGMA_HAND = 3.90        # spread of a card's hand value across different partners
SIGMA_CARD = 0.82        # spread of true value between cards — the signal
CARD_RANGE = 3.34        # best minus worst card, same measurement

HAND = 6


def plan(n_cards, pool, hands_per_pool, rounds, placements, lo=0.50, hi=0.85):
    pools_per_round = max(1, n_cards // pool)
    hands_total = pools_per_round * hands_per_pool * rounds

    # Each hand credits 6 cards, spread over the pool's P members.
    values_per_card = rounds * hands_per_pool * HAND / pool
    kept = int(placements * (hi - lo))

    se_hand_internal = SIGMA_BAND / max(kept, 1) ** 0.5
    se_per_value = (SIGMA_HAND ** 2 + se_hand_internal ** 2) ** 0.5
    se_card = se_per_value / max(values_per_card, 1) ** 0.5

    # Pair coverage: each hand puts C(6,2)=15 pairs together; a pool of P has
    # C(P,2) possible pairs. Only pairs sharing a pool can ever meet.
    pairs_per_hand = len(list(combinations(range(HAND), 2)))
    pair_slots = hands_total * pairs_per_hand
    pairs_reachable = pools_per_round * rounds * len(list(combinations(range(pool), 2)))
    all_pairs = len(list(combinations(range(n_cards), 2)))

    return {
        'values_per_card': values_per_card,
        'hands_total': hands_total,
        'placements_total': hands_total * placements,
        'seconds': hands_total * placements / THROUGHPUT,
        'se_card': se_card,
        'signal_to_noise': SIGMA_CARD / se_card,
        'resolvable_gap': 2 * (2 ** 0.5) * se_card,
        'avg_pair_meetings': pair_slots / max(pairs_reachable, 1),
        'pair_coverage': min(1.0, pairs_reachable / all_pairs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cards', type=int, default=120)
    p.add_argument('--pool', type=int, default=20, help='Cards playing each other')
    p.add_argument('--hands', type=int, default=200, help='Hands drawn per pool')
    p.add_argument('--placements', type=int, default=300, help='Placements per hand')
    a = p.parse_args()

    print(f"\n  {a.cards} cards, pools of {a.pool}, {a.hands} hands per pool, "
          f"{a.placements} placements per hand")
    print(f"  measured: partner noise {SIGMA_HAND}, real card spread "
          f"{SIGMA_CARD} (range {CARD_RANGE})\n")
    print(f"  {'rounds':>7} {'values':>8} {'hands':>9} {'placements':>13} {'time':>8} "
          f"{'SE':>7} {'resolves':>9} {'pair':>7} {'pairs':>7}")
    print(f"  {'':>7} {'/card':>8} {'':>9} {'':>13} {'':>8} {'':>7} {'gap of':>9} "
          f"{'meets':>7} {'covered':>7}")
    print(f"  {'-'*7} {'-'*8} {'-'*9} {'-'*13} {'-'*8} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")
    for r in (1, 2, 5, 10, 20, 50):
        s = plan(a.cards, a.pool, a.hands, r, a.placements)
        flag = "" if s['signal_to_noise'] >= 4 else ("  <- too noisy" if s['signal_to_noise'] < 2 else "  <- marginal")
        print(f"  {r:>7} {s['values_per_card']:>8.0f} {s['hands_total']:>9,} "
              f"{s['placements_total']:>13,} {s['seconds']:>7.1f}s {s['se_card']:>7.3f} "
              f"{s['resolvable_gap']:>9.2f} {s['avg_pair_meetings']:>7.1f} "
              f"{s['pair_coverage']*100:>6.0f}%{flag}")
    print()


if __name__ == '__main__':
    main()
