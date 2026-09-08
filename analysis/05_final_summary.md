# Final Summary: Optimal Simulation Strategy

## Problem

Given 120 pirate cards (each a 2×2 grid of symbol quarters), exhaustively evaluate all possible 6-card placements to determine each card's scoring contribution. Cards can be rotated, placed on top or below existing cards, and arrows copy symbols from adjacent visible quarters.

## Scale

| Dimension | Count |
|-----------|-------|
| Fixed 6 cards, position paths (geometry) | 729,529 |
| × rotations (4 per card) | × 4^6 = 4,096 |
| × top/bottom (cards 2-6) | × 2^5 = 32 |
| **Total placements per 6-card set** | **95.6 billion** |
| Card-set selections from 120 | Sampled (Monte Carlo) |

## Optimal Approach: Three-Layer Decomposition

### Layer 1: Position Path Enumeration (Outer Loop)

Enumerate the **729,529 unique position paths** via DFS on the 2×2 grid geometry. Each path defines which grid cells each card occupies and how they overlap.

- Use incremental footprint tracking (add/undo 4 cells per step)
- Store the path as a sequence of 6 positions
- This is the outermost loop and runs fast (~microseconds per path)

### Layer 2: Top/Bottom Visibility (Middle Loop)

For each position path, iterate **32 top/bottom combinations** (2^5 for cards 2-6). Each combination determines **visibility**: which quarter of which card is the topmost at each grid cell.

- Compute visibility mask per card: which of its 4 quarters are visible
- Identify any arrows whose target cell is covered by a different card (cross-card dependencies)
- O(~20) operations per combination

### Layer 3: Rotation Scoring — Hybrid Factored Evaluation (Inner "Loop")

**Do NOT iterate 4^6 = 4,096 rotation combinations.** Instead, split the score:

```
total_score = base_symbol_score + arrow_bonus
```

**Base symbol score** (non-arrow symbols in visible quarters):
- Each card's contribution depends ONLY on its own rotation
- All 6 cards are independent → evaluate each card's 4 rotations separately
- Cost: O(6 × 4) = **O(24)** lookups from precomputed table

**Arrow bonus** (symbols copied by arrows):
- Independent arrows (target on same card, not covered): fold into base, O(4) per arrow
- Dependent arrows (~0.89 per arrangement on average, target covered by different card): iterate that pair's joint rotations, O(4 × 4) = O(16) per dependent arrow
- Cost: ~O(28) average

**Total per visibility config: ~72 ops** instead of ~49,000.

### Combined

```
729,529 position paths
  × 32 top/bottom combos
  × ~72 ops per combo
= 1.68 billion operations
≈ 3.4 seconds in C/Rust (@ 2ns/op)
```

## Architecture

```
Startup (once):
  Load 120 cards from JSON
  Precompute for each card × 4 rotations:
    - quarter_symbols[120][4][4] → symbol lists per quarter
    - quarter_score[120][4][4]   → score contribution per quarter  
    - card_arrows[120][4]        → arrow positions and directions
  Total: ~17 KB, fits in L1 cache

Per 6-card set:
  ┌─ DFS over position paths (729,529)
  │   ├─ Loop over 32 top/bottom combos
  │   │   ├─ Compute visibility masks (O(20))
  │   │   ├─ Base score: 6 cards × 4 rotations, averaged (O(24))
  │   │   ├─ Arrow bonus: ~1 dependent arrow × 16 pair combos (O(28))
  │   │   └─ Record score per card (O(6))
  │   └─ Undo top/bottom state
  └─ Undo position state

Card-set sampling (outer):
  Monte Carlo or stratified sampling of which 6 cards to use
  → accumulate per-card score statistics
```

## Speedup Chain

| What | Factor |
|------|--------|
| Position path DFS instead of flat enumeration | Structural |
| Factored base scoring (independent rotations) | 4^6/24 = **170×** |
| Precomputed lookup tables (L1-cached) | ~3× vs computed |
| Only iterate dependent arrow pairs | ~5× vs all arrows |
| **Combined vs naive brute force** | **~3,600×** |

## Performance Per 6-Card Set

| Metric | Value |
|--------|-------|
| Operations | 1.68 × 10^9 |
| Time (C/Rust, 2ns/op) | **~3.4 seconds** |
| Time (Python+Numba) | ~15 seconds |

## Full 120-Card Analysis (Monte Carlo Sampling)

| Samples | Time (C/Rust) | Statistical Quality |
|---------|---------------|-------------------|
| 1,000 sets | 57 min | Rough card ranking |
| 10,000 sets | 9.4 hours | Confident averages |
| Stratified (100/card = 12,000) | 3.3 hours | All cards well-covered |

## Key Implementation Notes

1. **Position path DFS**: track footprint as a set of occupied cells; at each level compute valid next positions (cells that overlap the footprint); push/pop positions on a stack.

2. **Visibility computation**: for each grid cell, the topmost card layer wins. With 6 cards and the top/bottom ordering, this is a simple priority check per cell.

3. **Arrow dependency detection**: for each visible arrow, check if its target grid cell belongs to a different card than the arrow's card. If yes, it's a dependent arrow — record the (source_card, target_card) pair.

4. **Factored base scoring**: precompute `quarter_score[card_id][rotation][quarter_idx]` as the sum of points for all non-arrow symbols in that quarter. For each visible quarter of each card, look up the 4 rotation values and average them.

5. **Dependent arrow resolution**: for each dependent arrow pair (card_a, card_b), iterate 16 joint rotation combos. For each combo, resolve the arrow (look up target quarter's non-arrow symbols) and accumulate the bonus. Average over 16 combos.

6. **Recording**: accumulate total scores per card_id across all sampled arrangements. After all samples, compute mean/median/percentiles per card.
