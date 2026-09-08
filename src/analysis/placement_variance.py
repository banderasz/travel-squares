#!/usr/bin/env python3
"""Placement variance analysis: how much does score vary for the same 6 cards?

Picks N random 6-card combos, generates M random placements for each,
and measures how much the score varies purely from placement choices.

Usage:
    python analyze_placement_variance.py [--combos 200] [--placements 100000]
                                         [--threads 8] [--seed 42]
"""
import argparse
import sys
import time
import numpy as np

from src.deck.simulation import (
    load_cards, get_paths, evaluate_placements_batch
)
import numba as nb


def run_analysis(n_combos, n_placements, seed, n_threads):
    if n_threads > 0:
        nb.set_num_threads(n_threads)

    print("=" * 70)
    print("  PLACEMENT VARIANCE ANALYSIS")
    print("  Same cards, different placements — how much does the score vary?")
    print("=" * 70)
    print(f"  Combos: {n_combos} | Placements/combo: {n_placements:,} | Seed: {seed}")
    print(f"  Numba threads: {nb.get_num_threads()}")

    cd = load_cards('decks/pirate_20_25.json')
    paths = get_paths()
    n_cards, n_paths = cd['n_cards'], len(paths)
    print(f"  {n_cards} cards, {n_paths:,} paths")
    print(f"  Placement space per combo: {n_paths:,} × 4^6 × 2^5 "
          f"= {n_paths * 4096 * 32:,.0f}")

    # Warmup JIT
    print("  Compiling JIT...", end=' ', flush=True)
    t0 = time.time()
    wc = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    wr = np.zeros((1, 6), dtype=np.int32)
    _ = evaluate_placements_batch(wc, wr, np.array([0], dtype=np.int32),
                                  np.array([0], dtype=np.int32), paths[:2], cd)
    print(f"done in {time.time()-t0:.1f}s")

    # Pick random card combos
    rng = np.random.default_rng(seed)
    combos = []
    seen = set()
    while len(combos) < n_combos:
        cards = tuple(sorted(int(c) for c in rng.choice(n_cards, 6, replace=False)))
        if cards not in seen:
            seen.add(cards)
            combos.append(list(cards))

    # ============================================================
    # Evaluate each combo
    # ============================================================
    all_means = []
    all_stds = []
    all_ranges = []
    all_iqrs = []
    all_mins = []
    all_maxs = []
    all_p5 = []
    all_p95 = []
    all_cards = []

    print(f"\n  Evaluating {n_combos} combos × {n_placements:,} placements each...\n")
    t_start = time.time()

    for i, cards in enumerate(combos):
        # Generate random placements for this combo
        ci = np.tile(np.array(cards, dtype=np.int32), (n_placements, 1))
        rots = rng.integers(0, 4, size=(n_placements, 6), dtype=np.int32)
        pi = rng.integers(0, n_paths, size=n_placements, dtype=np.int32)
        tb = rng.integers(0, 32, size=n_placements, dtype=np.int32)

        scores = evaluate_placements_batch(ci, rots, pi, tb, paths, cd)
        sc = scores  # keep as float for precision

        mean = sc.mean()
        std = sc.std()
        p5 = np.percentile(sc, 5)
        p25 = np.percentile(sc, 25)
        p75 = np.percentile(sc, 75)
        p95 = np.percentile(sc, 95)
        sc_min = sc.min()
        sc_max = sc.max()

        all_means.append(mean)
        all_stds.append(std)
        all_ranges.append(sc_max - sc_min)
        all_iqrs.append(p75 - p25)
        all_mins.append(sc_min)
        all_maxs.append(sc_max)
        all_p5.append(p5)
        all_p95.append(p95)
        all_cards.append(cards)

        if (i + 1) % 10 == 0 or i == n_combos - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_combos - i - 1) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  [{i+1}/{n_combos}] {(i+1)/n_combos*100:.0f}% "
                f"| {rate:.1f} combos/s | ETA {eta:.0f}s   ")
            sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s ({n_combos * n_placements / elapsed:,.0f} evals/s)\n")

    # Convert to arrays
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    all_ranges = np.array(all_ranges)
    all_iqrs = np.array(all_iqrs)
    all_mins = np.array(all_mins)
    all_maxs = np.array(all_maxs)
    all_p5 = np.array(all_p5)
    all_p95 = np.array(all_p95)

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print(f"  {'='*70}")
    print(f"  PLACEMENT VARIANCE SUMMARY — {n_combos} random combos")
    print(f"  {'='*70}")
    print(f"")
    print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, arr in [
        ("Combo mean score", all_means),
        ("Placement std", all_stds),
        ("Score range", all_ranges),
        ("IQR (p25-p75)", all_iqrs),
        ("90% range (p5-p95)", all_p95 - all_p5),
    ]:
        print(f"  {label:<25} {arr.mean():>8.2f} {np.median(arr):>8.2f} "
              f"{arr.std():>8.2f} {arr.min():>8.2f} {arr.max():>8.2f}")

    # ============================================================
    # HOW MUCH DOES PLACEMENT MATTER?
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  HOW MUCH DOES PLACEMENT MATTER?")
    print(f"  {'='*70}")
    avg_std = all_stds.mean()
    avg_range = all_ranges.mean()
    avg_mean = all_means.mean()
    between_std = all_means.std()
    print(f"  Average placement std:        {avg_std:.2f} points")
    print(f"  Average best-vs-worst range:  {avg_range:.1f} points")
    print(f"  Average 90% range:            {(all_p95 - all_p5).mean():.1f} points")
    print(f"  Average combo mean:           {avg_mean:.2f} points")
    print(f"  Between-combo std:            {between_std:.2f} points")
    print(f"")
    print(f"  Placement std / combo mean:   {avg_std / abs(avg_mean) * 100:.0f}%")
    print(f"  Within-combo var share:       "
          f"{avg_std**2 / (avg_std**2 + between_std**2) * 100:.1f}%")
    print(f"  Between-combo var share:      "
          f"{between_std**2 / (avg_std**2 + between_std**2) * 100:.1f}%")

    # ============================================================
    # VARIANCE BY COMBO QUALITY (mean score buckets)
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  PLACEMENT VARIANCE BY HAND QUALITY")
    print(f"  {'='*70}")
    buckets = [(-30, -5), (-5, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50)]
    print(f"  {'Mean score':<15} {'N':>5} {'Avg Std':>8} {'Avg Range':>10} {'Avg IQR':>8} {'Avg 90%':>8}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for lo, hi in buckets:
        mask = (all_means >= lo) & (all_means < hi)
        if mask.sum() == 0:
            continue
        print(f"  [{lo:>3}, {hi:>3})     {mask.sum():>5} {all_stds[mask].mean():>8.2f} "
              f"{all_ranges[mask].mean():>10.1f} {all_iqrs[mask].mean():>8.2f} "
              f"{(all_p95[mask] - all_p5[mask]).mean():>8.2f}")

    # ============================================================
    # EXTREME COMBOS
    # ============================================================
    order_high_var = np.argsort(-all_stds)
    order_low_var = np.argsort(all_stds)

    print(f"\n  {'='*70}")
    print(f"  TOP 20 COMBOS — HIGHEST PLACEMENT VARIANCE")
    print(f"  {'='*70}")
    print(f"  {'#':<4} {'Cards':<35} {'Mean':>6} {'Std':>5} {'Range':>6} {'IQR':>5} {'Min':>5} {'Max':>5}")
    print(f"  {'-'*4} {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")
    for rank, idx in enumerate(order_high_var[:20]):
        print(f"  {rank+1:<4} {str(all_cards[idx]):<35} {all_means[idx]:>6.1f} "
              f"{all_stds[idx]:>5.2f} {all_ranges[idx]:>6.0f} {all_iqrs[idx]:>5.1f} "
              f"{all_mins[idx]:>5.0f} {all_maxs[idx]:>5.0f}")

    print(f"\n  {'='*70}")
    print(f"  TOP 20 COMBOS — LOWEST PLACEMENT VARIANCE")
    print(f"  {'='*70}")
    print(f"  {'#':<4} {'Cards':<35} {'Mean':>6} {'Std':>5} {'Range':>6} {'IQR':>5} {'Min':>5} {'Max':>5}")
    print(f"  {'-'*4} {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")
    for rank, idx in enumerate(order_low_var[:20]):
        print(f"  {rank+1:<4} {str(all_cards[idx]):<35} {all_means[idx]:>6.1f} "
              f"{all_stds[idx]:>5.2f} {all_ranges[idx]:>6.0f} {all_iqrs[idx]:>5.1f} "
              f"{all_mins[idx]:>5.0f} {all_maxs[idx]:>5.0f}")

    # Highest and lowest mean combos
    order_high_mean = np.argsort(-all_means)
    order_low_mean = np.argsort(all_means)

    print(f"\n  {'='*70}")
    print(f"  TOP 20 COMBOS — HIGHEST MEAN SCORE")
    print(f"  {'='*70}")
    print(f"  {'#':<4} {'Cards':<35} {'Mean':>6} {'Std':>5} {'Range':>6} {'IQR':>5} {'Min':>5} {'Max':>5}")
    print(f"  {'-'*4} {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")
    for rank, idx in enumerate(order_high_mean[:20]):
        print(f"  {rank+1:<4} {str(all_cards[idx]):<35} {all_means[idx]:>6.1f} "
              f"{all_stds[idx]:>5.2f} {all_ranges[idx]:>6.0f} {all_iqrs[idx]:>5.1f} "
              f"{all_mins[idx]:>5.0f} {all_maxs[idx]:>5.0f}")

    print(f"\n  {'='*70}")
    print(f"  TOP 20 COMBOS — LOWEST MEAN SCORE")
    print(f"  {'='*70}")
    print(f"  {'#':<4} {'Cards':<35} {'Mean':>6} {'Std':>5} {'Range':>6} {'IQR':>5} {'Min':>5} {'Max':>5}")
    print(f"  {'-'*4} {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")
    for rank, idx in enumerate(order_low_mean[:20]):
        print(f"  {rank+1:<4} {str(all_cards[idx]):<35} {all_means[idx]:>6.1f} "
              f"{all_stds[idx]:>5.2f} {all_ranges[idx]:>6.0f} {all_iqrs[idx]:>5.1f} "
              f"{all_mins[idx]:>5.0f} {all_maxs[idx]:>5.0f}")

    # ============================================================
    # DISTRIBUTION OF WITHIN-COMBO STD
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  DISTRIBUTION OF PLACEMENT STD ACROSS {n_combos} COMBOS")
    print(f"  {'='*70}")
    vals, counts = np.histogram(all_stds, bins=20)
    max_bar = 40
    max_count = vals.max()
    for i in range(len(vals)):
        lo = counts[i]
        hi = counts[i + 1]
        bar = '█' * int(vals[i] / max_count * max_bar)
        print(f"  {lo:>5.1f}-{hi:>5.1f}: {bar} {vals[i]:>4} ({vals[i]/n_combos*100:4.1f}%)")

    # Correlation: does higher mean → higher variance?
    corr = np.corrcoef(all_means, all_stds)[0, 1]
    print(f"\n  Correlation(mean, placement_std) = {corr:.3f}")
    corr_range = np.corrcoef(all_means, all_ranges)[0, 1]
    print(f"  Correlation(mean, range) = {corr_range:.3f}")


def main():
    p = argparse.ArgumentParser(description='Placement variance analysis')
    p.add_argument('--combos', type=int, default=200,
                   help='Number of random card combos to test (default: 200)')
    p.add_argument('--placements', type=int, default=100_000,
                   help='Random placements per combo (default: 100,000)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-cards', type=int, default=0,
                   help='Use only first N cards (0=all 120). E.g. 20 → C(20,6)=38,760 combos')
    p.add_argument('--samples', type=int, default=0,
                   help='Generate this many random samples from N cards and group by combo. '
                        'Mutually exclusive with --combos mode.')
    a = p.parse_args()

    if a.samples > 0:
        run_sampled_analysis(a.samples, a.n_cards or 20, a.seed, a.threads)
    else:
        run_analysis(a.combos, a.placements, a.seed, a.threads)


if __name__ == '__main__':
    main()
