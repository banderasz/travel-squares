# Splitting the Problem: Corrected Analysis

## The Correction: Intra-Card Arrows Become Cross-Card Dependencies

An **intra-card arrow** (e.g., arrow_right in TL pointing to TR of the same card) becomes a **cross-card dependency** when another card covers the target quarter:

```
Card A at (0,0):                    Card B placed at (1,0) on top:
  +--------+---------+               +--------+==========+
  | TL     | TR      |               | TL     || B's    ||
  | ->rght | shark   |   ------->    | ->rght || BL     ||  <- B covers A's TR
  | coin   | map     |               | coin   ||        ||
  +--------+---------+               +--------+==========+
  | BL     | BR      |               | BL     | BR      |
  +--------+---------+               +--------+---------+

Arrow now points to Card B's BL quarter, NOT Card A's TR.
Card A's arrow contribution depends on Card B's ROTATION!
```

### Measured Impact (50,000 random arrangements with actual card data)

**Cross-card dependency pairs per arrangement:**

| Pairs | Frequency | Percent |
|-------|-----------|---------|
| 0 | 19,910 | 39.8% |
| 1 | 20,231 | 40.5% |
| 2 | 7,949 | 15.9% |
| 3 | 1,704 | 3.4% |
| 4+ | 206 | 0.4% |

**Largest dependency cluster:**

| Cluster Size | Frequency | Percent |
|-------------|-----------|---------|
| 1 (all independent) | 19,910 | 39.8% |
| 2 cards | 23,519 | 47.0% |
| 3 cards | 5,434 | 10.9% |
| 4+ cards | 1,137 | 2.3% |

**Arrow statistics per arrangement (average):**

| Metric | Average |
|--------|---------|
| Visible arrows | 3.39 |
| Independent (target on same card, not covered) | 2.50 (73.7%) |
| **Dependent** (target covered by different card) | **0.89 (26.3%)** |

About **26% of visible arrows** have their target covered by a different card. This happens in ~60% of arrangements.

---

## Why Naive Factoring Fails

Trying to compute each card's contribution independently:
```
total_score = contribution(card1, rot1) + contribution(card2, rot2) + ...
```

Doesn't work because `contribution(cardA, rotA)` depends on `rotB` when card A's arrow points to a cell covered by card B. Treating dependent cards as clusters and iterating joint rotations only gives ~5x speedup (clusters grow too large).

---

## The Hybrid Solution: Base Score + Arrow Adjustment (3,600x speedup)

### Key Insight: Split the Score

```
total_score = base_symbol_score + arrow_bonus_score
```

**base_symbol_score**: Count of non-arrow symbols in visible quarters
- Each card's contribution is **ALWAYS independent** of other cards' rotations
- Can always be factored: O(6 x 4) = O(24) per visibility config

**arrow_bonus_score**: Extra symbols copied by arrows from adjacent cells
- Only ~0.89 dependent arrows per arrangement create cross-card dependencies
- Each involves at most 2 cards: iterate O(4x4) = O(16) pair rotations

### Algorithm

```python
for each position_path:                    # 729,529 paths
    for each top_bottom_combo:              # 32 combos
        
        # 1. Determine visibility
        visibility = compute_visibility()   # O(~20)
        
        # 2. BASE SCORE: fully factored, always independent
        for each card k:
            for rot r in 0..3:
                base_contrib[k][r] = precomputed[card_k][r][visible_mask_k]
            avg_base[k] = mean(base_contrib[k])
        total_base = sum(avg_base)
        
        # 3. ARROW BONUS: handle dependencies only
        for each dependent arrow (card_a -> card_b):
            for rot_a in 0..3:
                for rot_b in 0..3:
                    bonus[rot_a][rot_b] = resolve_arrow(...)
            avg_bonus += mean(bonus)
        
        total_score = total_base + avg_bonus
```

### Performance

| Approach | Ops | Time (C, 2ns/op) | Speedup |
|----------|-----|-------------------|---------|
| Naive (rebuild per leaf) | 6.12 x 10^12 | 204 min | 1x |
| DFS + precomputed deltas | 1.20 x 10^12 | 40 min | 5x |
| **Hybrid base+arrow** | **1.68 x 10^9** | **3.4 sec** | **3,641x** |

### Why It Works

Arrows are only ~10% of the total score. And only 26% of arrows are cross-card dependent. So 97.4% of the scoring work uses the fast factored path, and only 2.6% needs expensive pair-wise iteration.

---

## Precomputation: 17 KB total, fits in L1 cache

| Table | Size |
|-------|------|
| quarter_scores[120][4][4] | 7.5 KB |
| quarter_symbols[120][4][4][4] | 7.5 KB |
| card_arrows[120][4] | ~2 KB |
| has_arrows[120] | 120 B |
