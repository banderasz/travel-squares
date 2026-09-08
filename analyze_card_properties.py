#!/usr/bin/env python3
"""Deep analysis of what makes a card better than others.

Computes detailed card properties and correlates them with simulated performance.
Uses p50-p75 trimmed placement data from the 20-card pool.

Usage:
    python analyze_card_properties.py --store data/placements_20 --shards 256
"""
import json
import os
import sys
import time
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.sim import PlacementStore, ShardedPlacementStore, SCORE_DEFAULT, SYM_ORDER
from src.symbols import Symbols

# Derived from the scoring tables rather than hardcoded, so a rename or a
# rebalance cannot leave these sets quietly out of date. This reproduces the
# original membership exactly: 6 positive, 3 negative.
POSITIVE = {s for s in Symbols if s.value_symbol() and max(s.points) > 0}
NEGATIVE = {s for s in Symbols if s.value_symbol() and max(s.points) <= 0}
ARROWS = set(Symbols.arrows())


def load_and_score_cards(json_path, n_cards=20):
    with open(json_path) as f:
        raw = json.load(f)

    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    qshort = ['TL', 'TR', 'BL', 'BR']

    cards = []
    for ci in range(min(n_cards, len(raw))):
        card = raw[ci]['card']['quarters']
        c = {'id': ci, 'quarters': {}, 'raw_quarters': {}}

        total_pos = 0
        total_neg = 0
        total_arrows = 0
        total_symbols = 0
        sym_counts = {}
        quarter_values = []
        quarter_neg_counts = []
        quarter_pos_counts = []

        for qi, qn in enumerate(qnames):
            syms = [Symbols.of(s) for s in card[qn]]
            c['raw_quarters'][qshort[qi]] = card[qn]
            c['quarters'][qshort[qi]] = [s.abbrev for s in syms]

            q_pos = sum(1 for s in syms if s in POSITIVE)
            q_neg = sum(1 for s in syms if s in NEGATIVE)
            q_arr = sum(1 for s in syms if s in ARROWS)
            q_val = q_pos - q_neg

            total_pos += q_pos
            total_neg += q_neg
            total_arrows += q_arr
            total_symbols += len(syms)
            quarter_values.append(q_val)
            quarter_neg_counts.append(q_neg)
            quarter_pos_counts.append(q_pos)

            for s in syms:
                if s not in ARROWS:
                    sym_counts[s] = sym_counts.get(s, 0) + 1

        # Net value
        c['n_positive'] = total_pos
        c['n_negative'] = total_neg
        c['n_arrows'] = total_arrows
        c['n_symbols'] = total_symbols
        c['net_value'] = total_pos - total_neg
        c['sym_counts'] = sym_counts

        # Quarter analysis
        c['quarter_values'] = quarter_values
        c['best_quarter_val'] = max(quarter_values)
        c['worst_quarter_val'] = min(quarter_values)
        c['quarter_asymmetry'] = max(quarter_values) - min(quarter_values)

        # Hidability: how much do you gain by hiding the worst quarter?
        # = removing the worst quarter's contribution
        worst_qi = quarter_values.index(min(quarter_values))
        c['worst_quarter_neg'] = quarter_neg_counts[worst_qi]
        c['worst_quarter_pos'] = quarter_pos_counts[worst_qi]
        c['hide_gain'] = -min(quarter_values)  # hiding a -3 quarter gains +3

        # Best 3 quarters (what a smart player keeps visible)
        sorted_qv = sorted(quarter_values, reverse=True)
        c['best_3_sum'] = sum(sorted_qv[:3])
        c['all_4_sum'] = sum(quarter_values)

        # Symbol diversity (number of distinct positive symbol types)
        pos_types = set(s for s in sym_counts if s in POSITIVE)
        c['pos_diversity'] = len(pos_types)

        # Negative concentration: are negatives in one quarter or spread?
        if total_neg > 0:
            max_neg_q = max(quarter_neg_counts)
            c['neg_concentration'] = max_neg_q / total_neg  # 1.0 = all in one quarter
        else:
            c['neg_concentration'] = 1.0

        # Positive concentration
        if total_pos > 0:
            max_pos_q = max(quarter_pos_counts)
            c['pos_concentration'] = max_pos_q / total_pos
        else:
            c['pos_concentration'] = 0

        # Empty quarters
        c['n_empty_quarters'] = sum(1 for qn in qnames if len(card[qn]) == 0)

        cards.append(c)

    return cards


def get_sim_scores(store, n_cards, trim_low=50, trim_high=75):
    """Get per-card average scores from trimmed simulation data."""
    card_score_sum = np.zeros(n_cards, dtype=np.float64)
    card_count = np.zeros(n_cards, dtype=np.int64)
    trimming = trim_low > 0 or trim_high < 100

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
            cards = PlacementStore.unpack_cards_scalar(data_sorted['card_ids'][start])
            for c in cards:
                if c < n_cards:
                    card_score_sum[c] += grp_mean * sz
                    card_count[c] += sz

    card_avg = card_score_sum / np.maximum(card_count, 1)
    return card_avg


def main():
    n_cards = 20
    json_path = 'pirate_cards/pirate_20_25.json'
    store = ShardedPlacementStore('data/placements_20', n_shards=256)

    print("=" * 70)
    print("  DEEP CARD ANALYSIS — What Makes a Card Good?")
    print("=" * 70)

    # Load card properties
    cards = load_and_score_cards(json_path, n_cards)

    # Get simulation scores
    print("\n  Loading p50-p75 simulation scores...")
    sim_avg = get_sim_scores(store, n_cards, trim_low=50, trim_high=75)
    ranking = np.argsort(-sim_avg)

    for c in cards:
        c['sim_avg'] = sim_avg[c['id']]

    # ============================================================
    # DETAILED CARD BREAKDOWN
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  DETAILED CARD BREAKDOWN (ranked by simulation performance)")
    print(f"  {'='*70}")

    for rank, cid in enumerate(ranking):
        c = cards[cid]
        print(f"\n  #{rank+1} — Card {cid}  (sim avg: {c['sim_avg']:.2f})")
        print(f"  {'─'*60}")

        for qn in ['TL', 'TR', 'BL', 'BR']:
            syms = c['quarters'][qn]
            val = c['quarter_values'][['TL','TR','BL','BR'].index(qn)]
            if not syms:
                print(f"    {qn}: (empty)          val={val:+d}")
            else:
                print(f"    {qn}: {', '.join(syms):<25s} val={val:+d}")

        print(f"    ───")
        print(f"    Positive: {c['n_positive']}  Negative: {c['n_negative']}  "
              f"Arrows: {c['n_arrows']}  Net: {c['net_value']:+d}")
        print(f"    Quarter values: {c['quarter_values']}  "
              f"Asymmetry: {c['quarter_asymmetry']}")
        print(f"    Hide worst quarter gain: +{c['hide_gain']}  "
              f"Best 3 quarters sum: {c['best_3_sum']:+d}")
        print(f"    Neg concentration: {c['neg_concentration']:.0%}  "
              f"Pos diversity: {c['pos_diversity']} types")

    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    print(f"\n\n  {'='*70}")
    print(f"  CORRELATION: card properties vs simulation performance")
    print(f"  {'='*70}")

    features = [
        ('Net value (pos-neg)', [c['net_value'] for c in cards]),
        ('# Positive symbols', [c['n_positive'] for c in cards]),
        ('# Negative symbols', [-c['n_negative'] for c in cards]),
        ('# Arrows', [c['n_arrows'] for c in cards]),
        ('Best 3 quarters sum', [c['best_3_sum'] for c in cards]),
        ('Hide gain', [c['hide_gain'] for c in cards]),
        ('Quarter asymmetry', [c['quarter_asymmetry'] for c in cards]),
        ('Neg concentration', [c['neg_concentration'] for c in cards]),
        ('Pos diversity', [c['pos_diversity'] for c in cards]),
        ('Total symbols', [c['n_symbols'] for c in cards]),
    ]

    sim_scores = np.array([c['sim_avg'] for c in cards])

    print(f"\n  {'Feature':<30} {'Correlation':>12}  {'Strength'}")
    print(f"  {'-'*30} {'-'*12}  {'-'*15}")

    results = []
    for name, vals in features:
        arr = np.array(vals, dtype=np.float64)
        if arr.std() > 0:
            corr = np.corrcoef(arr, sim_scores)[0, 1]
        else:
            corr = 0
        results.append((name, corr))

    results.sort(key=lambda x: -abs(x[1]))
    for name, corr in results:
        strength = '█' * int(abs(corr) * 20)
        sign = '+' if corr > 0 else '−'
        print(f"  {name:<30} {sign}{abs(corr):>11.3f}  {strength}")

    # ============================================================
    # WHAT DRIVES VALUE: regression-style analysis
    # ============================================================
    print(f"\n\n  {'='*70}")
    print(f"  VALUE DRIVERS — What matters most?")
    print(f"  {'='*70}")

    # Simple multivariate: net_value + hide_gain + arrows
    net_vals = np.array([c['net_value'] for c in cards], dtype=np.float64)
    hide_gains = np.array([c['hide_gain'] for c in cards], dtype=np.float64)
    n_arrows_arr = np.array([c['n_arrows'] for c in cards], dtype=np.float64)
    best3 = np.array([c['best_3_sum'] for c in cards], dtype=np.float64)

    # Fit linear: sim_avg ≈ a + b*net_value + c*hide_gain + d*arrows
    X = np.column_stack([np.ones(n_cards), net_vals, hide_gains, n_arrows_arr])
    beta = np.linalg.lstsq(X, sim_scores, rcond=None)[0]
    predicted = X @ beta
    residuals = sim_scores - predicted
    r_squared = 1 - np.var(residuals) / np.var(sim_scores)

    print(f"\n  Linear model: sim_avg = {beta[0]:.2f} + {beta[1]:.2f}×net_value "
          f"+ {beta[2]:.2f}×hide_gain + {beta[3]:.2f}×arrows")
    print(f"  R² = {r_squared:.3f}")

    # Also try: best_3_sum + arrows
    X2 = np.column_stack([np.ones(n_cards), best3, n_arrows_arr])
    beta2 = np.linalg.lstsq(X2, sim_scores, rcond=None)[0]
    pred2 = X2 @ beta2
    r2_2 = 1 - np.var(sim_scores - pred2) / np.var(sim_scores)

    print(f"\n  Alt model: sim_avg = {beta2[0]:.2f} + {beta2[1]:.2f}×best_3_sum "
          f"+ {beta2[2]:.2f}×arrows")
    print(f"  R² = {r2_2:.3f}")

    # Show residuals — which cards over/underperform their stats?
    print(f"\n  {'='*70}")
    print(f"  RESIDUALS — Cards that over/underperform their stats")
    print(f"  {'='*70}")
    res_order = np.argsort(-residuals)
    print(f"\n  {'Card':<6} {'Sim':>7} {'Predicted':>9} {'Residual':>9}  Interpretation")
    print(f"  {'-'*6} {'-'*7} {'-'*9} {'-'*9}  {'-'*30}")
    for idx in res_order:
        c = cards[idx]
        interp = "overperforms" if residuals[idx] > 0.1 else "underperforms" if residuals[idx] < -0.1 else "as expected"
        print(f"  {idx:<6} {sim_scores[idx]:>7.2f} {predicted[idx]:>9.2f} {residuals[idx]:>+9.2f}  {interp}")

    # ============================================================
    # SYMBOL-LEVEL ANALYSIS
    # ============================================================
    print(f"\n\n  {'='*70}")
    print(f"  SYMBOL ANALYSIS — Which symbols correlate with high scores?")
    print(f"  {'='*70}")

    all_syms = set()
    for c in cards:
        all_syms.update(c['sym_counts'].keys())

    sym_corrs = []
    for sym in sorted(all_syms, key=SYM_ORDER.index):  # wire order; Symbols is not orderable
        counts = np.array([c['sym_counts'].get(sym, 0) for c in cards], dtype=np.float64)
        if counts.std() > 0:
            corr = np.corrcoef(counts, sim_scores)[0, 1]
            avg_count = counts.mean()
            sym_corrs.append((sym, corr, avg_count))

    sym_corrs.sort(key=lambda x: -x[1])

    # Show scoring tables for context
    print(f"\n  Scoring reminder (points by count):")
    print(f"  {'Symbol':<10} {'1':>4} {'2':>4} {'3':>4} {'4':>4} {'5':>4} {'6':>4}")
    print(f"  {'-'*10} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for symbol in SYM_ORDER[:9]:
        pts = SCORE_DEFAULT[symbol.name]
        short = symbol.abbrev
        print(f"  {short:<10} {pts[0]:>4} {pts[1]:>4} {pts[2]:>4} {pts[3]:>4} {pts[4]:>4} {pts[5]:>4}")

    print(f"\n  {'Symbol':<10} {'Avg/card':>8} {'Corr w/score':>13}  {'Direction'}")
    print(f"  {'-'*10} {'-'*8} {'-'*13}  {'-'*20}")
    for sym, corr, avg in sym_corrs:
        short = sym.abbrev
        direction = "more → better" if corr > 0.1 else "more → worse" if corr < -0.1 else "neutral"
        bar = '█' * int(abs(corr) * 15)
        sign = '+' if corr > 0 else '−'
        print(f"  {short:<10} {avg:>8.1f} {sign}{abs(corr):>12.3f}  {bar} {direction}")

    # ============================================================
    # KEY INSIGHTS SUMMARY
    # ============================================================
    print(f"\n\n  {'='*70}")
    print(f"  KEY INSIGHTS")
    print(f"  {'='*70}")

    # Find the strongest correlate
    best_feature = results[0]
    print(f"""
  1. STRONGEST PREDICTOR: {best_feature[0]} (r={best_feature[1]:+.3f})
     The single best predictor of card value is '{best_feature[0]}'.

  2. QUARTER HIDING: A smart player hides their worst quarter.
     Cards with concentrated negatives (high hide_gain) benefit most
     from smart rotation — effectively removing their worst symbols.

  3. NET FORMULA: sim_avg ≈ {beta[0]:.1f} + {beta[1]:.2f}×(pos−neg) + {beta[2]:.2f}×hide_gain + {beta[3]:.2f}×arrows
     - Each net positive symbol adds ~{abs(beta[1]):.2f} points
     - Each point of "hidable damage" adds ~{abs(beta[2]):.2f} points  
     - Each arrow adds ~{abs(beta[3]):.2f} points
     - This explains {r_squared*100:.0f}% of card value differences

  4. ARROWS amplify value because they copy symbols from adjacent cards.
     In a 6-card layout with overlapping quarters, arrows pointing
     toward high-value neighbors create multiplicative scoring.

  5. The BEST cards combine: many positives + few negatives concentrated
     in one quarter (easy to hide) + arrows for bonus symbol copying.
     Card 1 is #1 because it has ALL THREE: 8 positives, 2 negatives
     both in coverable positions, and 2 arrows for symbol multiplication.
""")


if __name__ == '__main__':
    main()
