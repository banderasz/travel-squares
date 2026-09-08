#!/usr/bin/env python3
"""Placement variance analysis with a reduced card pool.

Uses only 20 cards so C(20,6) = 38,760 combos — with 1B samples that's
~25,800 placements per combo, enough for robust within-combo statistics.

Usage:
    python analyze_placement_variance_20.py [--samples 1000000000] [--threads 8]
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.sim import (
    load_cards, get_paths, evaluate_placements_batch,
    generate_and_eval, PlacementStore
)
import numba as nb


def run(n_samples, n_card_pool, seed, n_threads, batch_size):
    if n_threads > 0:
        nb.set_num_threads(n_threads)

    from math import comb
    n_combos_possible = comb(n_card_pool, 6)

    print("=" * 70)
    print("  PLACEMENT VARIANCE — REDUCED CARD POOL")
    print("=" * 70)
    print(f"  Card pool: {n_card_pool} cards (from 120)")
    print(f"  Possible combos: C({n_card_pool},6) = {n_combos_possible:,}")
    print(f"  Samples: {n_samples:,}")
    print(f"  Expected per combo: {n_samples / n_combos_possible:,.0f}")
    print(f"  Numba threads: {nb.get_num_threads()}")

    cd = load_cards('pirate_cards/pirate_20_25.json')
    paths = get_paths()
    n_paths = len(paths)

    # Warmup JIT
    print("  Compiling JIT...", end=' ', flush=True)
    t0 = time.time()
    wc = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    wr = np.zeros((1, 6), dtype=np.int32)
    _ = evaluate_placements_batch(wc, wr, np.array([0], dtype=np.int32),
                                  np.array([0], dtype=np.int32), paths[:2], cd)
    print(f"done in {time.time()-t0:.1f}s")

    # ============================================================
    # Generate samples in batches, accumulate per-combo stats
    # ============================================================
    # Use online/streaming stats: for each combo, track count, sum, sum_sq, min, max
    combo_count = np.zeros(n_combos_possible + 1, dtype=np.int64)
    combo_sum = np.zeros(n_combos_possible + 1, dtype=np.float64)
    combo_sum_sq = np.zeros(n_combos_possible + 1, dtype=np.float64)
    combo_min = np.full(n_combos_possible + 1, 127, dtype=np.int8)
    combo_max = np.full(n_combos_possible + 1, -128, dtype=np.int8)

    # Build combo_key → index mapping
    # For 20 cards, enumerate all C(20,6) combos and assign indices
    from itertools import combinations
    card_pool = list(range(n_card_pool))
    combo_to_idx = {}
    idx_to_combo = {}
    for idx, combo in enumerate(combinations(card_pool, 6)):
        key = 0
        for i, c in enumerate(combo):
            key |= c << (i * 8)
        combo_to_idx[key] = idx
        idx_to_combo[idx] = combo

    # Build lookup array for fast key→index (use a hash approach)
    # Since keys are sparse in int64 space, use a dict-based approach in numpy
    # For speed, convert to sorted arrays for np.searchsorted
    all_keys = np.array(sorted(combo_to_idx.keys()), dtype=np.int64)
    all_indices = np.array([combo_to_idx[k] for k in all_keys], dtype=np.int64)

    print(f"  Built combo index: {len(combo_to_idx):,} entries")
    print(f"\n  Generating {n_samples:,} random placements...\n")

    rng = np.random.default_rng(seed)
    t_start = time.time()
    generated = 0
    global_sum = 0.0
    global_sum_sq = 0.0

    while generated < n_samples:
        chunk = min(batch_size, n_samples - generated)

        # Generate random placements using only the card pool (vectorized, no Python loop)
        rand = rng.random((chunk, n_card_pool))
        card_ids = np.argpartition(rand, 6, axis=1)[:, :6].astype(np.int32)

        rots = rng.integers(0, 4, size=(chunk, 6), dtype=np.int32)
        pi = rng.integers(0, n_paths, size=chunk, dtype=np.int32)
        tb = rng.integers(0, 32, size=chunk, dtype=np.int32)

        scores = evaluate_placements_batch(card_ids, rots, pi, tb, paths, cd)

        # Compute combo keys (sorted)
        sorted_cards = np.sort(card_ids, axis=1)
        cs = sorted_cards.astype(np.int64)
        combo_keys = cs[:, 0] | (cs[:, 1] << 8) | (cs[:, 2] << 16) | \
                     (cs[:, 3] << 24) | (cs[:, 4] << 32) | (cs[:, 5] << 40)

        # Map keys to indices
        positions = np.searchsorted(all_keys, combo_keys)
        indices = all_indices[positions]

        # Accumulate stats
        sc = scores
        sc_int = np.round(scores).astype(np.int8)
        global_sum += sc.sum()
        global_sum_sq += (sc ** 2).sum()

        np.add.at(combo_count, indices, 1)
        np.add.at(combo_sum, indices, sc)
        np.add.at(combo_sum_sq, indices, sc ** 2)
        np.minimum.at(combo_min, indices, sc_int)
        np.maximum.at(combo_max, indices, sc_int)

        generated += chunk
        elapsed = time.time() - t_start
        rate = generated / elapsed
        eta = (n_samples - generated) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  [{generated:,}/{n_samples:,}] {generated/n_samples*100:.0f}% "
            f"| {rate:,.0f}/s | ETA {eta:.0f}s   ")
        sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s ({n_samples / elapsed:,.0f}/s)\n")

    # ============================================================
    # ANALYSIS
    # ============================================================
    mask = combo_count[:n_combos_possible] > 0
    active = np.where(mask)[0]
    counts = combo_count[active]
    sums = combo_sum[active]
    sum_sqs = combo_sum_sq[active]
    mins = combo_min[active].astype(np.float64)
    maxs = combo_max[active].astype(np.float64)

    means = sums / counts
    variances = sum_sqs / counts - means ** 2
    variances = np.maximum(variances, 0)  # numerical safety
    stds = np.sqrt(variances)
    ranges = maxs - mins

    global_mean = global_sum / n_samples
    global_var = global_sum_sq / n_samples - global_mean ** 2

    # Within-combo variance (weighted average)
    within_var = np.sum(variances * counts) / counts.sum()
    between_var = global_var - within_var

    print(f"  {'='*70}")
    print(f"  RESULTS — {n_samples:,} samples, {len(active):,} combos")
    print(f"  {'='*70}")
    print(f"  Combos covered: {len(active):,} / {n_combos_possible:,} "
          f"({len(active)/n_combos_possible*100:.1f}%)")
    print(f"  Placements/combo: mean={counts.mean():,.0f}, "
          f"median={np.median(counts):,.0f}, "
          f"min={counts.min()}, max={counts.max()}")

    print(f"\n  {'='*70}")
    print(f"  VARIANCE DECOMPOSITION (ANOVA)")
    print(f"  {'='*70}")
    print(f"  Overall: mean={global_mean:.3f}, std={global_var**0.5:.3f}, var={global_var:.3f}")
    print(f"  Total variance:          {global_var:>8.3f}  (100%)")
    print(f"  Between-combo variance:  {between_var:>8.3f}  ({between_var/global_var*100:.1f}%)")
    print(f"  Within-combo variance:   {within_var:>8.3f}  ({within_var/global_var*100:.1f}%)")
    print(f"")
    print(f"  → {within_var/global_var*100:.1f}% of score variance = HOW you place the cards")
    print(f"  → {between_var/global_var*100:.1f}% of score variance = WHICH cards you have")

    print(f"\n  {'='*70}")
    print(f"  WITHIN-COMBO STATISTICS")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, arr in [
        ("Combo mean score", means),
        ("Placement std", stds),
        ("Score range", ranges),
    ]:
        print(f"  {label:<25} {arr.mean():>8.2f} {np.median(arr):>8.2f} "
              f"{arr.std():>8.2f} {arr.min():>8.2f} {arr.max():>8.2f}")

    print(f"\n  Correlation(mean, std) = {np.corrcoef(means, stds)[0,1]:.3f}")
    print(f"  Correlation(mean, range) = {np.corrcoef(means, ranges)[0,1]:.3f}")

    # ============================================================
    # VARIANCE BY HAND QUALITY
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  PLACEMENT VARIANCE BY HAND QUALITY")
    print(f"  {'='*70}")
    buckets = [(-30, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 50)]
    print(f"  {'Mean range':<15} {'Combos':>8} {'Placements':>12} {'Avg Std':>8} {'Avg Range':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")
    for lo, hi in buckets:
        bmask = (means >= lo) & (means < hi)
        if bmask.sum() == 0:
            continue
        print(f"  [{lo:>3}, {hi:>3})     {bmask.sum():>8,} {counts[bmask].sum():>12,} "
              f"{stds[bmask].mean():>8.2f} {ranges[bmask].mean():>10.1f}")

    # ============================================================
    # TOP COMBOS
    # ============================================================
    def show_top(title, order, n=20):
        print(f"\n  {'='*70}")
        print(f"  {title}")
        print(f"  {'='*70}")
        print(f"  {'#':<4} {'Cards':<30} {'N':>7} {'Mean':>7} {'Std':>6} {'Min':>5} {'Max':>5} {'Range':>6}")
        print(f"  {'-'*4} {'-'*30} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
        for rank, oi in enumerate(order[:n]):
            combo_idx = active[oi]
            cards = list(idx_to_combo[combo_idx])
            print(f"  {rank+1:<4} {str(cards):<30} {counts[oi]:>7,} {means[oi]:>7.2f} "
                  f"{stds[oi]:>6.2f} {mins[oi]:>5.0f} {maxs[oi]:>5.0f} {ranges[oi]:>6.0f}")

    show_top("HIGHEST PLACEMENT VARIANCE", np.argsort(-stds))
    show_top("LOWEST PLACEMENT VARIANCE", np.argsort(stds))
    show_top("HIGHEST MEAN SCORE", np.argsort(-means))
    show_top("LOWEST MEAN SCORE", np.argsort(means))
    show_top("BIGGEST BEST-vs-WORST SPREAD", np.argsort(-ranges))

    # ============================================================
    # DISTRIBUTION OF WITHIN-COMBO STD
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  DISTRIBUTION OF PLACEMENT STD")
    print(f"  {'='*70}")
    hist_vals, hist_edges = np.histogram(stds, bins=25)
    max_bar = 45
    max_count = hist_vals.max()
    for i in range(len(hist_vals)):
        lo = hist_edges[i]
        hi = hist_edges[i + 1]
        bar = '█' * int(hist_vals[i] / max_count * max_bar)
        print(f"  {lo:>5.2f}-{hi:>5.2f}: {bar} {hist_vals[i]:>5} ({hist_vals[i]/len(active)*100:5.1f}%)")


def main():
    p = argparse.ArgumentParser(description='Placement variance with reduced card pool')
    p.add_argument('--samples', type=int, default=1_000_000_000,
                   help='Number of random placements (default: 1B)')
    p.add_argument('--cards', type=int, default=20,
                   help='Number of cards in pool (default: 20)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=500_000,
                   help='Batch size for generation (default: 500K)')
    a = p.parse_args()

    run(a.samples, a.cards, a.seed, a.threads, a.batch_size)


if __name__ == '__main__':
    main()
