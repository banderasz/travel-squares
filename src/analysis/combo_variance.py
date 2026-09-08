#!/usr/bin/env python3
"""Analyze within-combo score variance from sharded placement data.

Scans all shards to find card combinations with multiple placements,
then reports how much scores vary for the same 6 cards.

Usage:
    python analyze_combo_variance.py [--store data/placements_sharded] [--shards 4096]
"""
import argparse
import sys
import time
import numpy as np

from src.deck.simulation import PlacementStore, ShardedPlacementStore


def scan_combo_variance(store, min_count=2, trim_low=0, trim_high=100):
    """Scan all shards, collect per-combo stats.
    
    trim_low/trim_high: percentile range to keep per combo.
    E.g. trim_low=50, trim_high=75 keeps only scores between p50 and p75,
    simulating a player who's better than median but not perfect.
    """
    trimming = trim_low > 0 or trim_high < 100
    trim_label = f" (keeping p{trim_low}-p{trim_high} per combo)" if trimming else ""
    print(f"\n  Scanning {store.n_shards} shards for combos with {min_count}+ placements{trim_label}...\n")

    # Accumulators
    combo_counts = []   # (count, mean, std, min, max, range, cards)
    total_records = 0
    total_combos = 0
    n_singletons = 0
    group_sizes = []

    # Per-group-size stats for ANOVA
    within_var_weighted = 0.0
    within_n = 0
    global_sum = 0.0
    global_sum_sq = 0.0
    global_n = 0

    t_start = time.time()

    for shard_id in range(store.n_shards):
        data = store.load_shard(shard_id)
        if len(data) == 0:
            continue

        # Compute combo keys for all records in this shard
        combo_keys = PlacementStore.combo_key_from_packed(data['card_ids'])
        scores_all = data['score'].astype(np.float64)

        # Group by combo key
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
            total_combos += 1

            if sz < min_count:
                n_singletons += 1
                continue

            grp_scores = scores_sorted[start:end]

            # Apply percentile trimming if requested
            if trimming and sz >= 4:
                lo_val = np.percentile(grp_scores, trim_low)
                hi_val = np.percentile(grp_scores, trim_high)
                mask = (grp_scores >= lo_val) & (grp_scores <= hi_val)
                grp_scores = grp_scores[mask]
                sz = len(grp_scores)
                if sz < 2:
                    n_singletons += 1
                    continue

            grp_mean = grp_scores.mean()
            grp_std = grp_scores.std()
            grp_min = int(grp_scores.min())
            grp_max = int(grp_scores.max())
            grp_range = grp_max - grp_min

            total_records += sz
            group_sizes.append(sz)

            # Accumulate global stats (after trimming)
            global_sum += grp_scores.sum()
            global_sum_sq += (grp_scores ** 2).sum()
            global_n += sz

            # Within-group variance for ANOVA
            within_var_weighted += grp_std ** 2 * sz
            within_n += sz

            cards = PlacementStore.unpack_cards_scalar(data_sorted['card_ids'][start])
            cards_sorted_list = sorted(cards)

            combo_counts.append((sz, grp_mean, grp_std, grp_min, grp_max, grp_range, cards_sorted_list))

        if (shard_id + 1) % 200 == 0 or shard_id == store.n_shards - 1:
            elapsed = time.time() - t_start
            pct = (shard_id + 1) / store.n_shards * 100
            sys.stdout.write(
                f"\r  Shard {shard_id+1}/{store.n_shards} ({pct:.0f}%) "
                f"| {len(combo_counts):,} multi-combos | {total_records:,} records "
                f"| {elapsed:.0f}s   ")
            sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s\n")

    if not combo_counts:
        print("  No combos with multiple placements found.")
        return

    group_sizes = np.array(group_sizes)
    n_multi = len(combo_counts)

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    global_mean = global_sum / global_n
    global_var = global_sum_sq / global_n - global_mean ** 2
    within_var = within_var_weighted / within_n if within_n > 0 else 0
    between_var = global_var - within_var

    print(f"  {'='*70}")
    print(f"  COMBO VARIANCE ANALYSIS — {global_n:,} total placements")
    print(f"  {'='*70}")
    print(f"  Total unique combos:     {total_combos:,}")
    print(f"  Singletons (1 record):   {n_singletons:,} ({n_singletons/total_combos*100:.1f}%)")
    print(f"  Multi-placement combos:  {n_multi:,} ({n_multi/total_combos*100:.1f}%)")
    print(f"  Records in multi-combos: {total_records:,}")
    print(f"")
    print(f"  Group sizes: mean={group_sizes.mean():.1f}, median={np.median(group_sizes):.0f}, "
          f"max={group_sizes.max()}, p95={np.percentile(group_sizes,95):.0f}, "
          f"p99={np.percentile(group_sizes,99):.0f}")

    # Size distribution
    size_bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 1000]
    print(f"\n  Group size distribution:")
    prev = 0
    for b in size_bins:
        mask = (group_sizes > prev) & (group_sizes <= b)
        cnt = mask.sum()
        if cnt > 0:
            label = f"{prev+1}" if b == prev + 1 else f"{prev+1}-{b}"
            print(f"    {label:>8}: {cnt:>8,} combos ({cnt/n_multi*100:5.1f}%)")
        prev = b
    mask = group_sizes > size_bins[-1]
    if mask.sum() > 0:
        print(f"    >{size_bins[-1]:>6}: {mask.sum():>8,} combos ({mask.sum()/n_multi*100:5.1f}%)")

    # ============================================================
    # VARIANCE DECOMPOSITION (ANOVA)
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  VARIANCE DECOMPOSITION (ANOVA)")
    print(f"  {'='*70}")
    print(f"  Overall: mean={global_mean:.2f}, std={global_var**0.5:.2f}, var={global_var:.3f}")
    print(f"  Total variance:          {global_var:>8.3f}  (100%)")
    print(f"  Between-combo variance:  {between_var:>8.3f}  ({between_var/global_var*100:.1f}%)")
    print(f"  Within-combo variance:   {within_var:>8.3f}  ({within_var/global_var*100:.1f}%)")
    print(f"")
    print(f"  → {within_var/global_var*100:.1f}% of score variance = HOW you place the cards")
    print(f"  → {between_var/global_var*100:.1f}% of score variance = WHICH cards you have")

    # ============================================================
    # WITHIN-COMBO STATISTICS
    # ============================================================
    stds = np.array([c[2] for c in combo_counts])
    means = np.array([c[1] for c in combo_counts])
    ranges = np.array([c[5] for c in combo_counts])
    sizes = np.array([c[0] for c in combo_counts])

    # Weight by group size for unbiased estimates
    print(f"\n  {'='*70}")
    print(f"  WITHIN-COMBO SCORE VARIATION ({n_multi:,} combos)")
    print(f"  {'='*70}")
    print(f"  Within-combo std:  mean={stds.mean():.2f}, median={np.median(stds):.2f}, "
          f"p25={np.percentile(stds,25):.2f}, p75={np.percentile(stds,75):.2f}")
    print(f"  Score range:       mean={ranges.mean():.1f}, median={np.median(ranges):.0f}, "
          f"max={ranges.max()}")
    print(f"  Combo mean score:  mean={means.mean():.2f}, std={means.std():.2f}, "
          f"min={means.min():.1f}, max={means.max():.1f}")

    # ============================================================
    # COMBOS WITH HIGHEST VARIANCE
    # ============================================================
    # Sort by std (descending), but only for combos with enough data
    min_for_ranking = max(5, int(np.percentile(sizes, 50)))
    big_enough = [(i, c) for i, c in enumerate(combo_counts) if c[0] >= min_for_ranking]

    if big_enough:
        big_enough.sort(key=lambda x: -x[1][2])  # sort by std desc
        print(f"\n  {'='*70}")
        print(f"  TOP 30 COMBOS — HIGHEST PLACEMENT VARIANCE (≥{min_for_ranking} records)")
        print(f"  {'='*70}")
        print(f"  {'Cards':<35} {'N':>5} {'Mean':>6} {'Std':>5} {'Min':>4} {'Max':>4} {'Range':>5}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*5}")
        for _, (sz, m, s, mn, mx, rng, cards) in big_enough[:30]:
            print(f"  {str(cards):<35} {sz:>5} {m:>6.1f} {s:>5.1f} {mn:>4} {mx:>4} {rng:>5}")

        # LOWEST VARIANCE
        big_enough.sort(key=lambda x: x[1][2])  # sort by std asc
        print(f"\n  {'='*70}")
        print(f"  TOP 30 COMBOS — LOWEST PLACEMENT VARIANCE (≥{min_for_ranking} records)")
        print(f"  {'='*70}")
        print(f"  {'Cards':<35} {'N':>5} {'Mean':>6} {'Std':>5} {'Min':>4} {'Max':>4} {'Range':>5}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*5}")
        for _, (sz, m, s, mn, mx, rng, cards) in big_enough[:30]:
            print(f"  {str(cards):<35} {sz:>5} {m:>6.1f} {s:>5.1f} {mn:>4} {mx:>4} {rng:>5}")

    # ============================================================
    # HIGHEST AND LOWEST SCORING COMBOS
    # ============================================================
    if big_enough:
        big_enough.sort(key=lambda x: -x[1][1])  # sort by mean desc
        print(f"\n  {'='*70}")
        print(f"  TOP 30 COMBOS — HIGHEST MEAN SCORE (≥{min_for_ranking} records)")
        print(f"  {'='*70}")
        print(f"  {'Cards':<35} {'N':>5} {'Mean':>6} {'Std':>5} {'Min':>4} {'Max':>4} {'Range':>5}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*5}")
        for _, (sz, m, s, mn, mx, rng, cards) in big_enough[:30]:
            print(f"  {str(cards):<35} {sz:>5} {m:>6.1f} {s:>5.1f} {mn:>4} {mx:>4} {rng:>5}")

        big_enough.sort(key=lambda x: x[1][1])  # sort by mean asc
        print(f"\n  {'='*70}")
        print(f"  TOP 30 COMBOS — LOWEST MEAN SCORE (≥{min_for_ranking} records)")
        print(f"  {'='*70}")
        print(f"  {'Cards':<35} {'N':>5} {'Mean':>6} {'Std':>5} {'Min':>4} {'Max':>4} {'Range':>5}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*5}")
        for _, (sz, m, s, mn, mx, rng, cards) in big_enough[:30]:
            print(f"  {str(cards):<35} {sz:>5} {m:>6.1f} {s:>5.1f} {mn:>4} {mx:>4} {rng:>5}")

    # ============================================================
    # BIGGEST SPREAD: best vs worst placement of same cards
    # ============================================================
    if big_enough:
        big_enough.sort(key=lambda x: -x[1][5])  # sort by range desc
        print(f"\n  {'='*70}")
        print(f"  TOP 30 COMBOS — BIGGEST BEST-vs-WORST SPREAD (≥{min_for_ranking} records)")
        print(f"  {'='*70}")
        print(f"  {'Cards':<35} {'N':>5} {'Mean':>6} {'Std':>5} {'Min':>4} {'Max':>4} {'Range':>5}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*5}")
        for _, (sz, m, s, mn, mx, rng, cards) in big_enough[:30]:
            print(f"  {str(cards):<35} {sz:>5} {m:>6.1f} {s:>5.1f} {mn:>4} {mx:>4} {rng:>5}")

    # ============================================================
    # VARIANCE vs MEAN: does placement matter more for good or bad hands?
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  VARIANCE BY MEAN SCORE BUCKET")
    print(f"  {'='*70}")
    mean_buckets = [(-20, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 50)]
    print(f"  {'Mean range':<15} {'Combos':>8} {'Records':>10} {'Avg Std':>8} {'Avg Range':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
    for lo, hi in mean_buckets:
        mask = (means >= lo) & (means < hi)
        if mask.sum() == 0:
            continue
        bucket_stds = stds[mask]
        bucket_ranges = ranges[mask]
        bucket_sizes = sizes[mask]
        print(f"  [{lo:>3}, {hi:>3})     {mask.sum():>8,} {bucket_sizes.sum():>10,} "
              f"{bucket_stds.mean():>8.2f} {bucket_ranges.mean():>10.1f}")


def main():
    p = argparse.ArgumentParser(description='Analyze within-combo score variance')
    p.add_argument('--store', default='data/placements_sharded')
    p.add_argument('--shards', type=int, default=4096)
    p.add_argument('--min-count', type=int, default=2)
    p.add_argument('--trim-low', type=int, default=0,
                   help='Drop scores below this percentile per combo (default: 0)')
    p.add_argument('--trim-high', type=int, default=100,
                   help='Drop scores above this percentile per combo (default: 100)')
    a = p.parse_args()

    store = ShardedPlacementStore(a.store, n_shards=a.shards)

    print("=" * 70)
    if a.trim_low > 0 or a.trim_high < 100:
        print(f"  WITHIN-COMBO SCORE VARIANCE — TRIMMED p{a.trim_low}–p{a.trim_high}")
        print(f"  Simulating a player between p{a.trim_low} and p{a.trim_high} skill")
    else:
        print("  WITHIN-COMBO SCORE VARIANCE ANALYSIS")
    print("=" * 70)

    scan_combo_variance(store, min_count=a.min_count,
                        trim_low=a.trim_low, trim_high=a.trim_high)


if __name__ == '__main__':
    main()
