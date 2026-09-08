#!/usr/bin/env python3
"""Analyze which cards are best using trimmed (p50-p75) placement data.

For each card, computes its marginal contribution: how much better are combos
that include this card vs combos that don't?

Usage:
    python analyze_card_value.py --store data/placements_20 --shards 256
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.sim import PlacementStore, ShardedPlacementStore, SYM_MAP, load_cards


def load_card_details(json_path, n_cards=20):
    """Load card quarter details for display."""
    with open(json_path) as f:
        raw = json.load(f)

    sym_display = {
        'anchor': 'Anc', 'shark': 'Shk', 'rat': 'Rat', 'kraken': 'Kra',
        'map': 'Map', 'coin': 'Coi', 'rum': 'Rum', 'parrot': 'Par',
        'spyglass': 'Spy', 'arrow_up': '↑', 'arrow_down': '↓',
        'arrow_left': '←', 'arrow_right': '→'
    }
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    qshort = ['TL', 'TR', 'BL', 'BR']

    cards = []
    for ci in range(min(n_cards, len(raw))):
        card = raw[ci]
        quarters = {}
        for qi, qn in enumerate(qnames):
            syms = card['card']['quarters'][qn]
            quarters[qshort[qi]] = [sym_display.get(s, s) for s in syms]
        cards.append(quarters)
    return cards


def analyze_cards(store, n_cards, trim_low, trim_high, json_path):
    print(f"\n  Scanning {store.n_shards} shards, trimming to p{trim_low}-p{trim_high} per combo...\n")

    # Per-card accumulators: score_sum, count (weighted by group membership)
    card_score_sum = np.zeros(n_cards, dtype=np.float64)
    card_score_sq = np.zeros(n_cards, dtype=np.float64)
    card_count = np.zeros(n_cards, dtype=np.int64)

    # Per-combo stats for ranking
    combo_data = []  # (cards_list, mean, std, count)

    trimming = trim_low > 0 or trim_high < 100
    t_start = time.time()
    total_records = 0

    for shard_id in range(store.n_shards):
        data = store.load_shard(shard_id)
        if len(data) == 0:
            continue

        combo_keys = PlacementStore.combo_key_from_packed(data['card_ids'])
        scores_all = data['score'].astype(np.float64)

        order = np.argsort(combo_keys)
        keys_sorted = combo_keys[order]
        scores_sorted = scores_all[order]
        data_sorted = data[order]
        boundaries = np.concatenate([
            [0],
            np.where(np.diff(keys_sorted) != 0)[0] + 1,
            [len(data)]
        ])

        for b in range(len(boundaries) - 1):
            start, end = boundaries[b], boundaries[b + 1]
            sz = end - start
            if sz < 4:
                continue

            grp_scores = scores_sorted[start:end]

            if trimming:
                lo_val = np.percentile(grp_scores, trim_low)
                hi_val = np.percentile(grp_scores, trim_high)
                mask = (grp_scores >= lo_val) & (grp_scores <= hi_val)
                grp_scores = grp_scores[mask]
                sz = len(grp_scores)
                if sz < 2:
                    continue

            grp_mean = grp_scores.mean()
            grp_std = grp_scores.std()
            total_records += sz

            cards = PlacementStore.unpack_cards_scalar(data_sorted['card_ids'][start])
            cards_sorted = sorted(cards)

            combo_data.append((cards_sorted, grp_mean, grp_std, sz))

            # Accumulate per-card stats
            for c in cards_sorted:
                if c < n_cards:
                    card_score_sum[c] += grp_mean * sz
                    card_score_sq[c] += (grp_mean ** 2) * sz
                    card_count[c] += sz

        if (shard_id + 1) % 100 == 0 or shard_id == store.n_shards - 1:
            elapsed = time.time() - t_start
            pct = (shard_id + 1) / store.n_shards * 100
            sys.stdout.write(
                f"\r  Shard {shard_id+1}/{store.n_shards} ({pct:.0f}%) "
                f"| {len(combo_data):,} combos | {total_records:,} records "
                f"| {elapsed:.0f}s   ")
            sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s\n")

    # ============================================================
    # CARD RANKINGS
    # ============================================================
    card_avg = np.zeros(n_cards, dtype=np.float64)
    card_var = np.zeros(n_cards, dtype=np.float64)
    for c in range(n_cards):
        if card_count[c] > 0:
            card_avg[c] = card_score_sum[c] / card_count[c]
            card_var[c] = card_score_sq[c] / card_count[c] - card_avg[c] ** 2

    ranking = np.argsort(-card_avg)

    # Load card details
    card_details = load_card_details(json_path, n_cards)

    print(f"  {'='*70}")
    print(f"  CARD RANKINGS (p{trim_low}-p{trim_high} trimmed, {total_records:,} records)")
    print(f"  {'='*70}")
    print(f"  {'#':<4} {'Card':<6} {'Avg':>7} {'Std':>6} {'Combos':>7}  Quarters")
    print(f"  {'-'*4} {'-'*6} {'-'*7} {'-'*6} {'-'*7}  {'-'*40}")

    global_mean = card_avg[ranking].mean()
    for rank, cid in enumerate(ranking):
        q = card_details[cid]
        q_str = ' | '.join(f"{k}:{','.join(v)}" for k, v in q.items() if v)
        delta = card_avg[cid] - global_mean
        print(f"  {rank+1:<4} {cid:<6} {card_avg[cid]:>7.2f} {card_var[cid]**0.5:>6.2f} "
              f"{card_count[cid]:>7,}  {q_str}")

    # ============================================================
    # MARGINAL CONTRIBUTION: with vs without each card
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  MARGINAL CONTRIBUTION — each card's impact")
    print(f"  {'='*70}")

    # For each card, compute mean of combos containing it vs not containing it
    all_means = np.array([c[1] for c in combo_data])
    all_sizes = np.array([c[3] for c in combo_data])

    # Build card membership arrays
    card_in_combo = np.zeros((len(combo_data), n_cards), dtype=bool)
    for i, (cards, _, _, _) in enumerate(combo_data):
        for c in cards:
            if c < n_cards:
                card_in_combo[i, c] = True

    marginal = []
    for c in range(n_cards):
        in_mask = card_in_combo[:, c]
        out_mask = ~in_mask
        if in_mask.sum() > 0 and out_mask.sum() > 0:
            mean_in = np.average(all_means[in_mask], weights=all_sizes[in_mask])
            mean_out = np.average(all_means[out_mask], weights=all_sizes[out_mask])
            marginal.append((c, mean_in, mean_out, mean_in - mean_out))
        else:
            marginal.append((c, card_avg[c], global_mean, 0))

    marginal.sort(key=lambda x: -x[3])

    print(f"  {'#':<4} {'Card':<6} {'With':>8} {'Without':>8} {'Delta':>7}  {'Bar'}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*7}  {'-'*30}")
    max_delta = max(abs(m[3]) for m in marginal)
    for rank, (cid, mean_in, mean_out, delta) in enumerate(marginal):
        bar_len = int(abs(delta) / max_delta * 25)
        bar = ('█' * bar_len) if delta >= 0 else ('▒' * bar_len)
        sign = '+' if delta >= 0 else ''
        print(f"  {rank+1:<4} {cid:<6} {mean_in:>8.2f} {mean_out:>8.2f} {sign}{delta:>6.2f}  {bar}")

    # ============================================================
    # BEST/WORST CARD PAIRS
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  BEST CARD PAIRS (synergies)")
    print(f"  {'='*70}")

    pair_sum = np.zeros((n_cards, n_cards), dtype=np.float64)
    pair_count = np.zeros((n_cards, n_cards), dtype=np.int64)

    for cards, mean, std, sz in combo_data:
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                a, b = cards[i], cards[j]
                if a < n_cards and b < n_cards:
                    pair_sum[a, b] += mean * sz
                    pair_count[a, b] += sz

    pair_avg = np.full((n_cards, n_cards), np.nan)
    mask = pair_count > 0
    pair_avg[mask] = pair_sum[mask] / pair_count[mask]

    # Find best and worst pairs
    pairs = []
    for a in range(n_cards):
        for b in range(a + 1, n_cards):
            if pair_count[a, b] > 0:
                # Synergy = pair avg - (card_a avg + card_b avg) / 2 ... but simpler:
                # Just rank by pair average
                pairs.append((a, b, pair_avg[a, b], pair_count[a, b]))

    pairs.sort(key=lambda x: -x[2])

    print(f"  {'#':<4} {'Pair':<12} {'Avg':>7} {'Records':>9}")
    print(f"  {'-'*4} {'-'*12} {'-'*7} {'-'*9}")
    for rank, (a, b, avg, cnt) in enumerate(pairs[:15]):
        print(f"  {rank+1:<4} ({a:>2},{b:>3})    {avg:>7.2f} {cnt:>9,}")
    print(f"  ...")
    for rank, (a, b, avg, cnt) in enumerate(pairs[-10:]):
        r = len(pairs) - 10 + rank + 1
        print(f"  {r:<4} ({a:>2},{b:>3})    {avg:>7.2f} {cnt:>9,}")

    # ============================================================
    # WHY: analyze card properties
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  CARD PROPERTIES — WHY SOME CARDS ARE BETTER")
    print(f"  {'='*70}")

    # Score each card's raw symbol values
    score_map = {
        'Anc': 1, 'Spy': 1, 'Map': 1, 'Coi': 1, 'Rum': 1, 'Par': 1,
        'Shk': -1, 'Rat': -1, 'Kra': -1,
        '↑': 0.5, '↓': 0.5, '←': 0.5, '→': 0.5  # arrows have conditional value
    }

    print(f"  {'#':<4} {'Card':<5} {'SimAvg':>7} {'RawVal':>7} {'Arrows':>7} {'Neg':>5} {'Pos':>5}  Quarters")
    print(f"  {'-'*4} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5}  {'-'*40}")

    for rank, cid in enumerate(ranking):
        q = card_details[cid]
        raw_val = 0
        n_arrows = 0
        n_pos = 0
        n_neg = 0
        for qname, syms in q.items():
            for s in syms:
                v = score_map.get(s, 0)
                raw_val += v
                if s in ('↑', '↓', '←', '→'):
                    n_arrows += 1
                elif v > 0:
                    n_pos += 1
                elif v < 0:
                    n_neg += 1

        q_str = ' | '.join(f"{k}:{','.join(v)}" for k, v in q.items() if v)
        print(f"  {rank+1:<4} {cid:<5} {card_avg[cid]:>7.2f} {raw_val:>7.1f} {n_arrows:>7} "
              f"{n_neg:>5} {n_pos:>5}  {q_str}")


def main():
    p = argparse.ArgumentParser(description='Analyze card values from trimmed placement data')
    p.add_argument('--store', default='data/placements_20')
    p.add_argument('--shards', type=int, default=256)
    p.add_argument('--n-cards', type=int, default=20)
    p.add_argument('--trim-low', type=int, default=50)
    p.add_argument('--trim-high', type=int, default=75)
    p.add_argument('--json', default='pirate_cards/pirate_20_25.json')
    a = p.parse_args()

    store = ShardedPlacementStore(a.store, n_shards=a.shards)

    print("=" * 70)
    print(f"  CARD VALUE ANALYSIS — p{a.trim_low}–p{a.trim_high} player skill")
    print("=" * 70)

    analyze_cards(store, a.n_cards, a.trim_low, a.trim_high, a.json)


if __name__ == '__main__':
    main()
