# Pruning & Equivalence Analysis: Reducing the Search Space

## The Goal

Starting from **729,529 position paths × 4,096 rotations × 32 top/bottom = 95.6 billion** total placements for 6 fixed cards, identify which branches produce identical outcomes so we can iterate only distinct outcomes and multiply by weight.

All optimizations must be O(1) or O(small constant) at simulation time — no expensive checks in the hot loop.

---

## Key Findings from Data Analysis

### 1. Rotation Symmetry: Almost Zero Benefit

**All 120 cards have 4 distinct rotations.** No card has rotational symmetry (180° or 90°). This means we cannot skip any rotation based on the card alone.

When we consider which quarters are hidden by overlap:
- **1 quarter hidden**: 4.00 distinct rotations on average → **no savings**
- **2 quarters hidden**: 4.00 distinct rotations → **no savings**  
- **3 quarters hidden**: 3.98 distinct rotations → **negligible savings**
- **4 quarters hidden** (full overlap): 1.00 distinct rotations → **4× savings** (but rare)

**Verdict: Rotation deduplication is NOT a significant optimization.** The cards are too diverse — different rotations almost always produce different visible symbol sets.

### 2. Arrow Distribution

| Metric | Count |
|--------|-------|
| Cards with NO arrows | 34 / 120 (28.3%) |
| Cards with arrows | 86 / 120 (71.7%) |
| Intra-card arrows (point within same card) | 92 (77.3%) |
| Inter-card arrows (point to adjacent card) | 27 (22.7%) |

Most arrows (77%) point within the same card, meaning they resolve independently of placement. Only **27 arrows across all cards** point to an adjacent card's quarter — these are the ones that create cross-card dependencies.

### 3. Empty Quarters: Rare

Only **19 / 480 quarters (4.0%)** are empty. Top/bottom equivalence (when overlap is with an empty quarter) happens rarely — **not worth optimizing**.

### 4. Position Collapse by Overlap Mask: **THE BIG WIN**

When a new card is placed, what matters for scoring is **which of its 4 quarters are hidden** (the "overlap mask"), not the exact grid position. Multiple positions can produce the same overlap mask.

| Card Step | Avg Positions | Avg Unique Overlap Masks | Collapse Factor |
|-----------|--------------|-------------------------|-----------------|
| Card 2 | 9.0 | 9.0 | 1.0× |
| Card 3 | 12.6 | 9.9 | **1.3×** |
| Card 4 | 15.8 | 10.4 | **1.5×** |
| Card 5 | 18.8 | 10.8 | **1.7×** |
| Card 6 | 21.7 | 11.1 | **2.0×** |

For card 2 against just card 1 (a simple 2×2), every position gives a unique mask (9 positions = 9 masks). But as the footprint grows, multiple positions produce the same overlap pattern.

**However**, this is more nuanced: the overlap mask of the new card is only half the story. We also need to know which OLD quarters are being overlapped (for top/bottom semantics). The full mask is `(new_card_hidden_quarters, old_surface_hidden_quarters)`, and these are more often unique.

---

## The Real Question: What Produces Identical Scoring Outcomes?

Two placements produce the **same score** if and only if the **final visible symbol multiset** is identical after arrow resolution. This depends on:

1. **Which quarters of each card are visible** (determined by all positions + top/bottom of all 6 cards)
2. **Which rotation each card is in** (determines symbol content of each quarter)
3. **Arrow resolution** (arrows copy symbols from adjacent visible quarters)

The dependencies form a complex chain. Let's identify what's **safe to prune** with O(1) checks:

---

## Practical Pruning Strategies (Ranked by Impact)

### Strategy A: Position Collapse by Overlap Mask (1.3×–2.0×)

**How it works**: Instead of iterating all positions, group positions by their overlap mask. For each unique mask, pick one representative position and record the multiplicity.

**At simulation time**:
```python
# Precomputed at each step:
for mask, count in unique_masks:   # ~10 masks instead of ~20 positions
    place_card(mask)
    score = evaluate()
    total += score * count
```

**Correctness concern**: Positions with the same overlap mask hide the same quarters of the new card. But they may affect DIFFERENT quarters of the existing arrangement. If the new card is placed on top, the old quarters being covered are different — but the visible result is determined by the new card's content, which is the same for same-mask positions (same quarters of new card are visible).

**Wait — this is actually fully correct for "on top" placement**, because when placing on top, we only care about which new card quarters are visible (the hidden ones are irrelevant, and the old card quarters beneath are replaced). The visible new-card quarters are determined by the new-card overlap mask alone.

**For "below" placement**, what matters is which OLD quarters remain visible and which NEW quarters are exposed in non-overlapping positions. The new-card mask alone determines this too.

**BUT**: the exact position also affects WHERE the visible quarters are on the grid, which matters for arrows from other cards. An arrow in card 1 pointing right might reach the new card in one position but not another, even with the same overlap mask.

**Verdict**: Position collapse is **only safe when no inter-card arrows are involved**. With 27 inter-card arrows across 120 cards, this optimization applies to a subset of cases. A runtime check (`has_inter_arrows[card_id]` flag) can gate it.

### Strategy B: Full-Overlap Rotation Skip (4× on full overlap)

When a card is placed with **full 4-quarter overlap** (position (0,0) relative to another card), all its quarters are hidden (if below) or it completely covers the card (if on top). 

- **On top**: the new card's content completely replaces the old → rotation matters.
- **Below**: the new card is completely hidden → rotation is irrelevant → **4× savings**.

Full overlap is 1 out of 9+ positions, so this saves `1/9 × 3/4 = 8%` but only on "below" placements.

### Strategy C: Arrow-Free Card Fast Path (minor)

For the 34 cards with no arrows:
- Arrow resolution can be skipped for this card's quarters (just count symbols directly).
- This is a micro-optimization in the arrow resolution phase, not a branch reduction.

### Strategy D: First Card Rotation Normalization (4×)

The **first card's rotation can be fixed** without loss of generality. Every arrangement with card 1 at rotation R is equivalent to the arrangement with card 1 at rotation 0 and all other cards adjusted. Since we're counting total symbols (rotationally invariant scoring), we can fix card 1's rotation.

**Wait** — this is only true if the scoring function is symmetric. If arrows from card 1 point to specific adjacent quarters, rotating card 1 changes which arrows point where, producing different scores. So this is **NOT safe** in general.

It IS safe if card 1 has no arrows. Card 1 has no arrows in 28.3% of cases.

---

## Combined Impact Estimate

| Strategy | Condition | Reduction | Scope |
|----------|-----------|-----------|-------|
| Position collapse (no arrows) | No inter-card arrows involved | 1.3×–2.0× | ~30% of cards |
| Full-overlap below rotation skip | Card placed at same position, below | 4× | ~5% of placements |
| Arrow-free card 1 rotation fix | Card 1 has no arrows | 4× on card 1 | 28% of card-1 choices |
| Arrow resolution skip | Card has no arrows + no arrows point to it | Micro | 28% of cards |

**Combined best-case reduction: ~2–3× for typical 6-card arrangements.**

---

## Most Impactful Structural Optimization

The biggest win isn't branch pruning — it's **the computation model**:

### Separate Position-Path Enumeration from Rotation/Top-Bottom

Since we showed there are only **729,529 position paths** and the rotation/top-bottom multiplier (×131,072) accounts for 99.99% of the branches:

**Approach**: For each position path (which defines the geometry — which quarters of which cards are visible), evaluate the combinatorics of rotations and top/bottom analytically:

1. **Enumerate the 729,529 position paths** (each takes ~microseconds)
2. **For each path**, determine the visibility structure: which quarters of which cards are exposed
3. **For rotation/top-bottom**, the visibility determines which specific symbols appear — iterate only the 4⁶ × 2⁵ = 131,072 rotation+layer combos, but with the geometry fixed
4. Many rotation combos can be pruned per position path based on the specific cards involved

This separates the **geometric** problem (729K paths) from the **content** problem (which symbols are visible), allowing the inner loop over rotations to be highly optimized with precomputed card data.

**Estimated effective search**: 729,529 × ~50,000 (pruned rotations/layers) ≈ **36 billion** — down from 95.6 billion.

---

## Recommendation

The most **performant and correct** approach is:

1. **Don't try to prune rotations/positions** — the cards are too diverse for meaningful equivalence classes
2. **Focus on making each evaluation blazing fast** — bitwise symbol counting, precomputed rotation tables, cache-friendly layout
3. **Use the 729,529 position-path enumeration** as the outer loop, with rotation/layer as the inner loop
4. **Apply micro-optimizations**: skip arrow resolution for arrow-free cards, precompute all rotation variants

The raw 95.6 billion evaluations at ~20ns each (Rust/C with SIMD) = **~32 minutes per 6-card set**. With the structural separation and fast inner loop, this is achievable.
