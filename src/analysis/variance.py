"""Analyze placement variance: how much does score vary for the same 6 cards
depending on rotation, path, and top/bottom placement?"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.deck.simulation import (
    PLACEMENT_DTYPE, PlacementStore, load_cards, get_paths,
    evaluate_placements_batch
)
import numba as nb
nb.set_num_threads(8)

print("=" * 70)
print("  VARIANCE ANALYSIS: Same 6 cards, different placements")
print("=" * 70)

# ============================================================
# PART 1: Sample from 5B stored data for ANOVA decomposition
# ============================================================
store = PlacementStore('data/placements')
n_total = store.count()
print(f"\n  Total placements in store: {n_total:,}")

if n_total > 0:
    print(f"\n--- Part 1: Variance decomposition (sampled from stored data) ---")
    data = store.load_mmap()

    # Quick global stats from a 10M sample
    SAMPLE = min(10_000_000, n_total)
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(n_total, SAMPLE, replace=False))

    t0 = time.time()
    sample_scores = data['score'][idx].astype(np.float64)
    sample_cards_packed = data['card_ids'][idx]
    print(f"  Sampled {SAMPLE:,} records in {time.time()-t0:.1f}s")

    total_mean = sample_scores.mean()
    total_var = sample_scores.var()
    print(f"  Overall: mean={total_mean:.3f}, std={total_var**0.5:.3f}, var={total_var:.3f}")

    # Unpack and sort card combos to get unordered keys
    t0 = time.time()
    cards_unpacked = PlacementStore.unpack_cards_array(sample_cards_packed)  # (N, 6)
    cards_sorted = np.sort(cards_unpacked, axis=1)
    cs = cards_sorted.astype(np.int64)
    combo_key = cs[:,0] | (cs[:,1]<<8) | (cs[:,2]<<16) | (cs[:,3]<<24) | (cs[:,4]<<32) | (cs[:,5]<<40)

    # Group by sorted combo key
    sort_order = np.argsort(combo_key)
    combo_sorted = combo_key[sort_order]
    scores_sorted = sample_scores[sort_order]
    boundaries = np.concatenate([[0], np.where(np.diff(combo_sorted) != 0)[0] + 1, [SAMPLE]])
    group_sizes = np.diff(boundaries)
    n_groups = len(group_sizes)
    multi_mask = group_sizes > 1
    n_multi = multi_mask.sum()
    print(f"  {n_groups:,} unique combos, {n_multi:,} with 2+ placements (took {time.time()-t0:.1f}s)")

    if n_multi > 0:
        # ANOVA: within-group variance
        within_var_weighted = 0.0
        within_n = 0
        group_stds = []
        group_ranges = []

        for i in range(n_groups):
            sz = group_sizes[i]
            if sz < 2:
                continue
            start = boundaries[i]
            grp = scores_sorted[start:start+sz]
            gv = grp.var()
            within_var_weighted += gv * sz
            within_n += sz
            group_stds.append(grp.std())
            group_ranges.append(grp.max() - grp.min())

        within_var = within_var_weighted / within_n
        between_var = total_var - within_var
        group_stds = np.array(group_stds)
        group_ranges = np.array(group_ranges)

        print(f"\n  {'='*60}")
        print(f"  VARIANCE DECOMPOSITION (ANOVA) — {SAMPLE:,} sample")
        print(f"  {'='*60}")
        print(f"  Total variance:          {total_var:>8.3f}  (100%)")
        print(f"  Between-combo variance:  {between_var:>8.3f}  ({between_var/total_var*100:.1f}%)")
        print(f"  Within-combo variance:   {within_var:>8.3f}  ({within_var/total_var*100:.1f}%)")
        print(f"")
        print(f"  → {within_var/total_var*100:.1f}% of score variance = HOW you place the cards")
        print(f"  → {between_var/total_var*100:.1f}% of score variance = WHICH cards you have")
        print(f"\n  Within-combo stats ({n_multi:,} combos with 2+ placements):")
        print(f"    Mean within-std:   {group_stds.mean():.2f}")
        print(f"    Median within-std: {np.median(group_stds):.2f}")
        print(f"    Mean score range:  {group_ranges.mean():.1f}")
        print(f"    Max score range:   {group_ranges.max():.0f}")
        print(f"    Group size: mean={group_sizes[multi_mask].mean():.1f}, max={group_sizes.max()}")

    del data  # release mmap

# ============================================================
# PART 2: Focused evaluation — fixed cards, vary everything else
# ============================================================
print(f"\n\n{'='*70}")
print(f"  PART 2: Controlled experiment — fixed cards, random placements")
print(f"{'='*70}")

cd = load_cards('pirate_cards/pirate_20_25.json')
paths = get_paths()
n_cards, n_paths = cd['n_cards'], len(paths)

# Warmup
wc = np.array([[0,1,2,3,4,5]], dtype=np.int32)
wr = np.zeros((1,6), dtype=np.int32)
wp = np.array([0], dtype=np.int32)
wt = np.array([0], dtype=np.int32)
_ = evaluate_placements_batch(wc, wr, wp, wt, paths[:2], cd)

N_EVAL = 1_000_000

combos = [
    ("Top 6 cards",     [33, 29, 79, 1, 14, 109]),
    ("Mid cards",       [38, 62, 9, 47, 2, 68]),
    ("Bottom 6 cards",  [43, 12, 103, 32, 107, 88]),
    ("Random mix A",    [10, 45, 91, 3, 72, 116]),
    ("Random mix B",    [42, 78, 105, 29, 64, 87]),
    ("No-arrow hand",   [118, 67, 48, 49, 96, 54]),
    ("All-arrow hand",  [33, 45, 9, 112, 14, 109]),
]

print(f"  Evaluating {N_EVAL:,} random placements per combo...\n")

results = []
for name, cards in combos:
    rng = np.random.default_rng(42)
    ci = np.tile(np.array(cards, dtype=np.int32), (N_EVAL, 1))
    rots = rng.integers(0, 4, size=(N_EVAL, 6), dtype=np.int32)
    pi = rng.integers(0, n_paths, size=N_EVAL, dtype=np.int32)
    tb = rng.integers(0, 32, size=N_EVAL, dtype=np.int32)

    t0 = time.time()
    scores = evaluate_placements_batch(ci, rots, pi, tb, paths, cd)
    elapsed = time.time() - t0
    sc = np.round(scores).astype(int)
    results.append((name, cards, sc))

    vals, counts = np.unique(sc, return_counts=True)
    max_bar = 35
    max_count = counts.max()

    print(f"  {name}: {cards}")
    print(f"    Mean={sc.mean():.2f}  Std={sc.std():.2f}  Range=[{sc.min()}, {sc.max()}]  "
          f"IQR=[{np.percentile(sc,25):.0f}, {np.percentile(sc,75):.0f}]")
    print(f"    Distribution:")
    for v, c in zip(vals, counts):
        bar = '█' * int(c / max_count * max_bar)
        print(f"      {v:>3}: {bar} {c/N_EVAL*100:.1f}%")
    print()

# ============================================================
# PART 3: Decompose variance sources — path vs rotation vs tb
# ============================================================
print(f"\n{'='*70}")
print(f"  PART 3: What drives placement variance?")
print(f"{'='*70}")

cards_test = [33, 29, 79, 1, 14, 109]
N = 200_000

rng = np.random.default_rng(99)
ci = np.tile(np.array(cards_test, dtype=np.int32), (N, 1))

# Baseline: everything random
rots_rand = rng.integers(0, 4, size=(N, 6), dtype=np.int32)
pi_rand = rng.integers(0, n_paths, size=N, dtype=np.int32)
tb_rand = rng.integers(0, 32, size=N, dtype=np.int32)
sc_all = evaluate_placements_batch(ci, rots_rand, pi_rand, tb_rand, paths, cd)

# Fix path (same path), vary rotation + tb
fixed_path = np.full(N, 0, dtype=np.int32)
sc_fix_path = evaluate_placements_batch(ci, rots_rand, fixed_path, tb_rand, paths, cd)

# Fix rotation, vary path + tb
fixed_rots = np.zeros((N, 6), dtype=np.int32)
sc_fix_rot = evaluate_placements_batch(ci, fixed_rots, pi_rand, tb_rand, paths, cd)

# Fix tb, vary path + rotation
fixed_tb = np.zeros(N, dtype=np.int32)
sc_fix_tb = evaluate_placements_batch(ci, rots_rand, pi_rand, fixed_tb, paths, cd)

# Fix all except path
sc_only_path = evaluate_placements_batch(ci, fixed_rots, pi_rand, fixed_tb, paths, cd)

# Fix all except rotation
sc_only_rot = evaluate_placements_batch(ci, rots_rand, np.full(N, 0, dtype=np.int32), fixed_tb, paths, cd)

# Fix all except tb
sc_only_tb = evaluate_placements_batch(ci, fixed_rots, fixed_path, tb_rand, paths, cd)

print(f"\n  Cards: {cards_test}")
print(f"  {N:,} placements each\n")
print(f"  {'Source':<30} {'Mean':>7} {'Std':>7} {'Var':>7}  {'% of total':>10}")
print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}  {'-'*10}")
total_v = np.var(sc_all)
for label, sc in [
    ("All random (total)",          sc_all),
    ("Only path varies",            sc_only_path),
    ("Only rotation varies",        sc_only_rot),
    ("Only tb varies",              sc_only_tb),
    ("Path + rot (fix tb)",         sc_fix_tb),
    ("Path + tb (fix rot)",         sc_fix_rot),
    ("Rot + tb (fix path)",         sc_fix_path),
]:
    v = np.var(sc)
    print(f"  {label:<30} {np.mean(sc):>7.2f} {np.std(sc):>7.2f} {v:>7.2f}  {v/total_v*100:>9.1f}%")

# ============================================================
# Summary
# ============================================================
print(f"\n\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Combo':<20} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5} {'Range':>6} {'IQR':>5}")
print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
for name, cards, sc in results:
    iqr = np.percentile(sc, 75) - np.percentile(sc, 25)
    print(f"  {name:<20} {sc.mean():>6.2f} {sc.std():>6.2f} {sc.min():>5} {sc.max():>5} "
          f"{sc.max()-sc.min():>6} {iqr:>5.0f}")

avg_std = np.mean([r[2].std() for r in results])
avg_range = np.mean([r[2].max()-r[2].min() for r in results])
print(f"\n  Key findings:")
print(f"    • Placement std ≈ {avg_std:.1f} points (average across combos)")
print(f"    • Best vs worst placement of same cards: ~{avg_range:.0f} point spread")
print(f"    • That's {avg_range/13*100:.0f}% of a typical hand score (~13)")
print(f"    • For context: companion card std ≈ 2.3 points (from DB analysis)")
print(f"    • → Placement choice and card choice contribute roughly equally to variance")
