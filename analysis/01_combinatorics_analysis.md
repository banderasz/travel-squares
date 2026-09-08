# Combinatorics Analysis: Pirate Card Placement Simulation

## Input Data

- **Total cards**: 120
- **Cards per game setup**: 6 (placed sequentially)
- **Card structure**: 2×2 grid of quarters (top_left, top_right, bottom_left, bottom_right)
- **Symbols per quarter**: 0–4 (from a set of 13 distinct symbols)
- **Rotations per card**: 4 (0°, 90°, 180°, 270°)
- **Placement**: on top of or below existing card stack (2 options)

### Symbol Inventory (13 types)

| Symbol     | Count | Type     |
|------------|-------|----------|
| rat        | 160   | standard |
| anchor     | 141   | standard |
| spyglass   | 120   | standard |
| shark      | 119   | standard |
| map        | 102   | standard |
| coin       | 101   | standard |
| rum        | 81    | standard |
| kraken     | 62    | standard |
| parrot     | 42    | standard |
| arrow_left | 34    | arrow    |
| arrow_up   | 31    | arrow    |
| arrow_right| 29    | arrow    |
| arrow_down | 25    | arrow    |

### Quarter Size Distribution

| Symbols in Quarter | Occurrences (out of 480 quarters) |
|---|---|
| 0 | 19 |
| 1 | 86 |
| 2 | 198 |
| 3 | 143 |
| 4 | 34 |

---

## Placement Mechanics

A card occupies a **2×2 area** on the grid. When placing a new card, it must overlap **at least one quarter** with any existing card in the arrangement. The new card can be placed either:
- **On top**: hides the overlapping quarter(s) of the card below
- **Below**: the existing card hides the overlapping quarter(s) of the new card

### Relative Positions (Card 2 vs Card 1)

With Card 1 at origin (0,0), Card 2 can be placed at 9 relative positions:

```
(-1,-1) → 1 quarter overlap    (-1,0) → 2 quarters overlap    (-1,1) → 1 quarter overlap
 (0,-1) → 2 quarters overlap    (0,0) → 4 quarters overlap     (0,1) → 2 quarters overlap
 (1,-1) → 1 quarter overlap     (1,0) → 2 quarters overlap     (1,1) → 1 quarter overlap
```

### Valid Positions Growth (by arrangement type)

| Cards Placed | Footprint (cells) | Valid Positions |
|---|---|---|
| **Diagonal (worst case spread)** | | |
| 1 | 4 | 9 |
| 2 | 7 | 14 |
| 3 | 10 | 19 |
| 4 | 13 | 24 |
| 5 | 16 | 29 |
| 6 | 19 | 34 |
| **Full overlap (best case)** | | |
| 1–6 | 4 | 9 |
| **Adjacent horizontal** | | |
| 1 | 4 | 9 |
| 2 | 6 | 12 |
| 3 | 8 | 15 |
| 4 | 10 | 18 |
| 5 | 12 | 21 |
| 6 | 14 | 24 |

Average positions for card 3 (across all card-2 placements): **12.6** (range: 9–14)

---

## Total Combination Count

### Breakdown per Step

| Step | Card Choices | Rotations | Positions | Top/Bottom | Subtotal per Step |
|------|-------------|-----------|-----------|------------|------------------|
| Card 1 | 120 | 4 | 1 | — | 480 |
| Card 2 | 119 | 4 | 9 | 2 | 8,568 |
| Card 3 | 118 | 4 | ~12–16 | 2 | 11,328–15,104 |
| Card 4 | 117 | 4 | ~16–23 | 2 | 14,976–21,528 |
| Card 5 | 116 | 4 | ~20–30 | 2 | 18,560–27,840 |
| Card 6 | 115 | 4 | ~24–37 | 2 | 22,080–34,040 |

### Total Combinations

| Scenario | Total | Note |
|----------|-------|------|
| **Lower bound** (full overlap, 9 positions always) | **2.04 × 10²²** | All cards stacked on same spot |
| **Average case** (~growing positions) | **~1.27 × 10²⁴** | Typical gameplay spread |
| **Upper bound** (diagonal, max spread) | **~2.39 × 10²⁵** | Maximum spread arrangement |

### For Reference

- Just card selection order (no rotation/position): **2.63 × 10¹²** (2.6 trillion)
- With rotations added: **1.08 × 10¹⁶** (10.8 quadrillion)
- Card selection is itself P(120,6) = 120 × 119 × 118 × 117 × 116 × 115 = **2,629,976,731,200**

---

## Feasibility Assessment

| Scenario | Operations | Time @ 10⁹ ops/sec | Time @ 10⁹ ops/sec (parallel, 1000 cores) |
|----------|-----------|--------------------|--------------------------------------------|
| Lower bound | 2.04 × 10²² | 645,000 years | 645 years |
| Average case | 1.27 × 10²⁴ | 40 million years | 40,000 years |
| Upper bound | 2.39 × 10²⁵ | 758 million years | 758,000 years |

### Verdict (Full 120-Card Selection)

**Exhaustive brute-force simulation is completely infeasible.** Even with the most optimistic assumptions (smallest search space, fastest hardware, heavy parallelism), the problem space is at minimum ~10²² — far beyond what any computer can enumerate.

---

## Fixed 6 Cards, All Placements (Exact Count)

If we **fix the 6 cards and their order** (removing card selection entirely), and only enumerate all possible placements (positions, rotations, top/bottom), the exact count is:

### **95,620,825,088 (~95.6 billion) placement sequences**

| Step | Unique Footprint Shapes | Avg Positions | Multiplier per Step | Cumulative Sequences |
|------|------------------------|---------------|--------------------|--------------------|
| Card 1 | 1 | 1 | 4 rot × 1 pos = 4 | 4 |
| Card 2 | 5 | 9.0 | 4 rot × 9 pos × 2 top/bot = 72 | 288 |
| Card 3 | 25 | 12.6 | 4 rot × 12.6 pos × 2 = ~100.4 | 28,928 |
| Card 4 | 131 | 15.8 | 4 rot × 15.8 pos × 2 = ~126.4 | 3,655,680 |
| Card 5 | 698 | 18.8 | 4 rot × 18.8 pos × 2 = ~150.8 | 550,518,784 |
| Card 6 | 3,772 | 21.7 | 4 rot × 21.7 pos × 2 = ~173.6 | 95,620,825,088 |

### Decomposition

```
Position-only paths (ignoring rotation & top/bottom): 729,529
× 4⁶ rotations:                                      × 4,096
× 2⁵ top/bottom choices (cards 2–6):                  × 32
= 95,620,825,088
```

### Feasibility of Fixed-Card Exhaustive Search

| Ops/sec | Time |
|---------|------|
| 10⁸ (Python+Numba) | ~16 minutes |
| 10⁹ (C/Rust) | ~96 seconds |
| 10¹⁰ (SIMD/GPU) | ~10 seconds |

**Verdict: Exhaustive enumeration of all placements for a fixed set of 6 cards IS feasible!** At optimized C speed, it takes under 2 minutes. This opens the door to a hybrid approach: sample card selections, but exhaustively evaluate each selection's placements.

---

## Key Reductions to Consider

### Symmetry Reductions
- **First card rotation**: The first card's rotation can be fixed (factor of 4 saved) since we can always normalize the first card's orientation.
- **Card order equivalence**: If we only care about the final arrangement (not the sequence), many orderings produce identical states. However, top/bottom layering makes order matter.

### Practical Reductions
- **Position-dependent pruning**: Many positions yield identical visible symbol counts due to symmetry.
- **Card equivalence**: Cards with identical quarter contents (after rotation) can be grouped.
- **Monte Carlo sampling**: Instead of exhaustive enumeration, sample random arrangements and compute statistics.

### Reduction Impact
Even with aggressive symmetry reduction (÷4 for first rotation, ÷some factor for equivalent layouts), we'd reduce by perhaps 10–100×, leaving ~10²⁰+ operations — still infeasible.

**Conclusion: A sampling-based approach is mandatory.**
