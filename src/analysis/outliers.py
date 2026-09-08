#!/usr/bin/env python3
"""Analyze placement outliers from sharded storage.

For each card combo with enough samples, identifies placements where
the score is unusually low (or high) compared to the combo's mean.

Usage:
    python analyze_outliers.py [--store data/placements_sharded] [--shards 4096]
                               [--min-count 5] [--threshold 2.0]
    python analyze_outliers.py --combo 33,29,79,1,14,109     # single combo
"""
import argparse
import sys
import time
import numpy as np

from src.deck.simulation import (
    PLACEMENT_DTYPE, PlacementStore, ShardedPlacementStore
)


def analyze_combo(store, cards, show_outliers=10, threshold=2.0):
    """Analyze a specific card combination."""
    stats = store.combo_stats(cards)
    if stats is None:
        print(f"  No records found for combo {sorted(cards)}")
        return

    scores = stats['scores']
    records = stats['records']
    n = stats['count']
    mean, std = stats['mean'], stats['std']

    print(f"\n  Cards: {stats['cards']}")
    print(f"  Records: {n:,}")
    print(f"  Mean: {mean:.2f}  Std: {std:.2f}  "
          f"Range: [{stats['min']}, {stats['max']}]  "
          f"IQR: [{stats['p25']:.0f}, {stats['p75']:.0f}]")

    if n < 3 or std < 0.01:
        print("  (too few records or zero variance for outlier analysis)")
        return

    # Score distribution
    vals, counts = np.unique(scores.astype(int), return_counts=True)
    max_bar = 40
    max_count = counts.max()
    print(f"\n  Distribution:")
    for v, c in zip(vals, counts):
        bar = '█' * int(c / max_count * max_bar)
        z = (v - mean) / std
        marker = ' ◄' if abs(z) > threshold else ''
        print(f"    {v:>4}: {bar} {c:>6} ({c/n*100:5.1f}%) z={z:+.1f}{marker}")

    # Bottom outliers (low scores)
    cutoff_low = mean - threshold * std
    low_mask = scores < cutoff_low
    n_low = low_mask.sum()
    print(f"\n  Low outliers (score < {cutoff_low:.1f}, z < -{threshold}): {n_low:,} ({n_low/n*100:.1f}%)")

    if n_low > 0 and show_outliers > 0:
        low_idx = np.where(low_mask)[0]
        low_scores = scores[low_idx]
        worst_order = np.argsort(low_scores)[:show_outliers]
        print(f"  {'#':<4} {'Cards':<30} {'Rots':<16} {'Path':>7} {'TB':>3} {'Score':>5} {'Z':>6}")
        print(f"  {'-'*4} {'-'*30} {'-'*16} {'-'*7} {'-'*3} {'-'*5} {'-'*6}")
        for rank, oi in enumerate(worst_order):
            idx = low_idx[oi]
            rec = records[idx]
            cards_list = PlacementStore.unpack_cards_scalar(rec['card_ids'])
            rots = PlacementStore.unpack_rots_scalar(rec['rotations'])
            z = (scores[idx] - mean) / std
            print(f"  {rank+1:<4} {','.join(map(str,cards_list)):<30} "
                  f"{','.join(map(str,rots)):<16} "
                  f"{rec['path_idx']:>7} {rec['tb_config']:>3} "
                  f"{rec['score']:>5} {z:>+6.1f}")

    # Top outliers (high scores)
    cutoff_high = mean + threshold * std
    high_mask = scores > cutoff_high
    n_high = high_mask.sum()
    print(f"\n  High outliers (score > {cutoff_high:.1f}, z > +{threshold}): {n_high:,} ({n_high/n*100:.1f}%)")

    if n_high > 0 and show_outliers > 0:
        high_idx = np.where(high_mask)[0]
        high_scores = scores[high_idx]
        best_order = np.argsort(-high_scores)[:show_outliers]
        print(f"  {'#':<4} {'Cards':<30} {'Rots':<16} {'Path':>7} {'TB':>3} {'Score':>5} {'Z':>6}")
        print(f"  {'-'*4} {'-'*30} {'-'*16} {'-'*7} {'-'*3} {'-'*5} {'-'*6}")
        for rank, oi in enumerate(best_order):
            idx = high_idx[oi]
            rec = records[idx]
            cards_list = PlacementStore.unpack_cards_scalar(rec['card_ids'])
            rots = PlacementStore.unpack_rots_scalar(rec['rotations'])
            z = (scores[idx] - mean) / std
            print(f"  {rank+1:<4} {','.join(map(str,cards_list)):<30} "
                  f"{','.join(map(str,rots)):<16} "
                  f"{rec['path_idx']:>7} {rec['tb_config']:>3} "
                  f"{rec['score']:>5} {z:>+6.1f}")


def scan_all_shards(store, min_count=5, threshold=2.0, top_combos=20):
    """Scan all shards to find combos with the highest within-combo variance
    and the most extreme outliers."""
    print(f"\n  Scanning {store.n_shards} shards for combos with {min_count}+ records...")

    all_results = []
    t_start = time.time()
    total_records = 0
    total_combos = 0

    for shard_id in range(store.n_shards):
        for combo_key, records in store.iter_shard_combos(shard_id, min_count=min_count):
            scores = records['score'].astype(np.float64)
            n = len(scores)
            mean = scores.mean()
            std = scores.std()
            total_records += n
            total_combos += 1

            if std > 0:
                # Track worst z-score in this combo
                z_scores = (scores - mean) / std
                worst_z = z_scores.min()
                best_z = z_scores.max()
                score_range = int(scores.max()) - int(scores.min())

                cards = PlacementStore.unpack_cards_scalar(records['card_ids'][0])
                cards_sorted = sorted(cards)

                all_results.append({
                    'combo_key': combo_key,
                    'cards': cards_sorted,
                    'count': n,
                    'mean': mean,
                    'std': std,
                    'min': int(scores.min()),
                    'max': int(scores.max()),
                    'range': score_range,
                    'worst_z': worst_z,
                    'best_z': best_z,
                })

        if (shard_id + 1) % 100 == 0:
            elapsed = time.time() - t_start
            pct = (shard_id + 1) / store.n_shards * 100
            sys.stdout.write(f"\r  Shard {shard_id+1}/{store.n_shards} ({pct:.0f}%) "
                             f"| {total_combos:,} combos | {total_records:,} records   ")
            sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\n  Scanned in {elapsed:.1f}s: {total_combos:,} combos, {total_records:,} records\n")

    if not all_results:
        print("  No combos with enough records found.")
        return

    # Sort by worst z-score (most extreme low outliers)
    all_results.sort(key=lambda r: r['worst_z'])

    print(f"  {'='*70}")
    print(f"  Top {top_combos} combos with most extreme LOW outliers")
    print(f"  {'='*70}")
    print(f"  {'Cards':<30} {'N':>6} {'Mean':>6} {'Std':>5} {'Min':>4} {'Max':>4} {'Worst Z':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*8}")
    for r in all_results[:top_combos]:
        print(f"  {str(r['cards']):<30} {r['count']:>6} {r['mean']:>6.1f} {r['std']:>5.1f} "
              f"{r['min']:>4} {r['max']:>4} {r['worst_z']:>+8.2f}")

    # Sort by highest variance
    all_results.sort(key=lambda r: -r['std'])
    print(f"\n  {'='*70}")
    print(f"  Top {top_combos} combos with highest placement variance")
    print(f"  {'='*70}")
    print(f"  {'Cards':<30} {'N':>6} {'Mean':>6} {'Std':>5} {'Range':>5} {'Min':>4} {'Max':>4}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*4} {'-'*4}")
    for r in all_results[:top_combos]:
        print(f"  {str(r['cards']):<30} {r['count']:>6} {r['mean']:>6.1f} {r['std']:>5.1f} "
              f"{r['range']:>5} {r['min']:>4} {r['max']:>4}")


def main():
    p = argparse.ArgumentParser(description='Analyze placement outliers from sharded store')
    p.add_argument('--store', default='data/placements_sharded',
                   help='Sharded store directory')
    p.add_argument('--shards', type=int, default=4096,
                   help='Number of shards')
    p.add_argument('--combo', type=str, default=None,
                   help='Specific combo to analyze (comma-separated card IDs)')
    p.add_argument('--min-count', type=int, default=5,
                   help='Minimum records per combo for scan mode')
    p.add_argument('--threshold', type=float, default=2.0,
                   help='Z-score threshold for outliers')
    p.add_argument('--top', type=int, default=20,
                   help='Number of top combos to show in scan mode')
    a = p.parse_args()

    store = ShardedPlacementStore(a.store, n_shards=a.shards)

    print("=" * 70)
    print("  PLACEMENT OUTLIER ANALYSIS")
    print("=" * 70)

    if a.combo:
        cards = [int(x) for x in a.combo.split(',')]
        assert len(cards) == 6, "Must specify exactly 6 card IDs"
        analyze_combo(store, cards, threshold=a.threshold)
    else:
        scan_all_shards(store, min_count=a.min_count, threshold=a.threshold, top_combos=a.top)


if __name__ == '__main__':
    main()
