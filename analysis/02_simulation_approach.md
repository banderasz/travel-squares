# Simulation Approach: Pirate Card Placement

## Problem Summary

Given 120 pirate cards (each a 2×2 grid of symbol quarters), simulate placing 6 cards in a stack with overlaps, resolve arrow symbols, count visible symbols, and evaluate each card's contribution to scoring. The exhaustive search space is ~10²² to ~10²⁵ — brute force is impossible.

---

## Recommended Approach: Monte Carlo Sampling + Bitwise Encoding

### Strategy Overview

1. **Encode card data as compact bitmasks** for maximum throughput
2. **Use Monte Carlo random sampling** to evaluate millions of random 6-card arrangements
3. **Accumulate per-card statistics** across all sampled arrangements
4. **Implement in C or Rust** (or optimized NumPy/Numba) for performance

---

## Phase 1: Data Encoding

### Card Representation

Each quarter has 0–4 symbols. There are 9 non-arrow symbols and 4 arrow types.

**Option A — Bitfield per quarter (fastest for overlap resolution):**

```
Per quarter (16 bits):
  Bits 0–8:   symbol presence flags (anchor, shark, rat, kraken, map, coin, rum, parrot, spyglass)
  Bits 9–12:  arrow flags (up, down, left, right)
  Bit  13:    unused
  Bits 14–15: symbol count encoded (for multi-occurrence tracking)
```

**Problem**: Symbols can repeat (e.g., ["rat", "rat"]). A presence bitmask loses count info.

**Option B — Count-based encoding (handles duplicates):**

```
Per quarter: array of 13 uint4 counts (13 symbols × 4 bits = 52 bits = 7 bytes)
Per card: 4 quarters × 7 bytes = 28 bytes
```

This fits in a single 256-bit SIMD register or two 128-bit registers.

**Option C — Symbol list with fixed-size array:**

```
Per quarter: uint8[4] symbols (symbol ID 0-12, with 255 = empty)
Per card: 4 quarters × 4 bytes = 16 bytes
```

Simplest and most cache-friendly. Duplicates are naturally handled.

**Recommended: Option C** — 16 bytes per card, trivially fits in cache, and symbol counting is a simple loop.

### Rotation Precomputation

Precompute all 4 rotations for each card at load time:

```
Rotation 0°:   [TL, TR, BL, BR]
Rotation 90°:  [BL, TL, BR, TR] (each quarter's internal symbols also remapped: arrow_right→arrow_up, etc.)
Rotation 180°: [BR, BL, TR, TL] (arrows reversed)
Rotation 270°: [TR, BR, TL, BL] (arrows remapped)
```

**Important**: When rotating, arrows must also rotate:
- 90° CW: right→down, down→left, left→up, up→right
- 180°: right→left, left→right, up→down, down→up
- 270° CW: right→up, up→left, left→down, down→right

Store as `rotated_cards[120][4]` = 480 card variants, each 16 bytes = **7.5 KB total**. Trivially fits in L1 cache.

---

## Phase 2: Board State Representation

### Grid Model

The board is a grid of quarter-cells. After 6 cards (each 2×2), the maximum extent is:
- Worst case: ~7×7 grid of quarter-cells (diagonal spread)
- Typical: ~5×5 grid

**Board representation**: A **layered grid** where each cell tracks visibility.

```
struct Cell {
    uint8_t symbol_ids[4];  // up to 4 symbols (from topmost visible card)
    uint8_t card_index;     // which card occupies this cell (top of stack)
    uint8_t layer;          // Z-order layer
};
```

**Simpler approach — just track what's visible:**

Since we only need the *final visible symbols*, we can represent the board as:
- A dictionary/array mapping (x, y) grid positions to `(card_index, quarter_index, layer)`
- The topmost layer at each position wins

For 6 cards with max ~19 occupied cells, this is tiny.

### Efficient Visibility Resolution

When placing a card on top:
```
for each of the 4 quarters of the new card:
    board[x][y] = new_card_quarter  (overwrites whatever was there)
```

When placing a card below:
```
for each of the 4 quarters of the new card:
    if board[x][y] is empty:
        board[x][y] = new_card_quarter
    // else: existing card stays visible (it's on top)
```

This is O(4) per card placement — **extremely fast**.

---

## Phase 3: Arrow Resolution

After all 6 cards are placed and visibility is determined:

1. For each visible quarter containing arrows:
   - Look at the adjacent quarter in the arrow's direction
   - Collect all **non-arrow** symbols from that adjacent quarter (only if visible)
   - Add those symbols to the current quarter's symbol list
2. Arrows are **not chained** — only direct adjacent symbols are copied

**Direction mapping:**
```
arrow_right: (x+1, y) — quarter to the right
arrow_left:  (x-1, y) — quarter to the left
arrow_up:    (x, y-1) — quarter above
arrow_down:  (x, y+1) — quarter below
```

**Important**: The arrow points to the adjacent *quarter cell* in the grid, not to another quarter of the same card. So arrow_right in TL points to TR, arrow_right in TR points to the TL of a card placed to the right, etc.

**Implementation**: Single pass over all visible quarters, O(total_visible_quarters) ≈ O(19 max).

---

## Phase 4: Scoring & Statistics

### Symbol Counting
After arrow resolution, count all visible symbols across the board. This yields a count per symbol type.

### Point Calculation
Apply the scoring map (to be provided) to convert symbol counts to points.

### Per-Card Accumulation
For each sampled arrangement, record which 6 cards were used and the total score:
```python
for each card_id in arrangement:
    card_scores[card_id].append(score)
    card_usage_count[card_id] += 1
```

After all samples:
```python
for each card:
    avg_score = mean(card_scores[card_id])
    median_score = median(card_scores[card_id])
    contribution = avg_score_with_card - avg_score_without_card  # marginal value
```

---

## Phase 5: Sampling Strategy

### Basic Monte Carlo
```
for sample in range(NUM_SAMPLES):
    1. Pick 6 random cards (without replacement)
    2. For each card (in order):
       a. Pick random rotation (0-3)
       b. Pick random valid position (overlapping existing footprint)
       c. Pick random layer (top or bottom)
    3. Resolve visibility
    4. Resolve arrows
    5. Count symbols, compute score
    6. Record score for each card used
```

### Sample Count Recommendations

| Samples | Time Estimate (optimized C) | Statistical Quality |
|---------|---------------------------|-------------------|
| 1 million | ~1 second | Rough ranking |
| 10 million | ~10 seconds | Good averages |
| 100 million | ~2 minutes | High confidence |
| 1 billion | ~20 minutes | Publication quality |

At ~50–100 ns per simulation (placement + resolution + scoring), these are achievable.

### Enhanced Sampling: Stratified by Card

To ensure every card is well-represented:
```
for each card_id in 0..119:
    for sample in range(SAMPLES_PER_CARD):
        Force card_id as one of the 6 cards
        Randomly fill the other 5
        Run simulation
        Record score attributed to card_id
```

With 100,000 samples per card × 120 cards = 12 million total samples. Very feasible.

---

## Implementation Plan

### Language Choice

| Language | Pros | Cons | Estimated Speed |
|----------|------|------|-----------------|
| **C** | Fastest, full SIMD control | Manual memory management | ~20–50 ns/sim |
| **Rust** | Safe, fast, good ergonomics | Steeper learning curve | ~20–50 ns/sim |
| **Python + Numba** | Easy to write, JIT compiled | Some overhead, limited control | ~200–500 ns/sim |
| **Python + NumPy** | Vectorized operations | Hard to vectorize this problem | ~1–5 μs/sim |
| **Pure Python** | Easiest | Far too slow | ~50–200 μs/sim |

**Recommended: Python with Numba JIT** for rapid development with near-C performance.  
**Alternative: Rust** if maximum performance and correctness guarantees are desired.

### Architecture

```
┌─────────────────────────┐
│     Card Data Loader    │  ← Reads JSON, precomputes rotations
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│   Simulation Engine     │  ← Monte Carlo loop (Numba/C/Rust)
│  ┌───────────────────┐  │
│  │ Card Selection    │  │  ← Random 6-card draw
│  │ Placement Engine  │  │  ← Position + rotation + layer
│  │ Visibility Solver │  │  ← Top-layer extraction
│  │ Arrow Resolver    │  │  ← Symbol copy from adjacent cells
│  │ Symbol Counter    │  │  ← Count per type
│  │ Scorer            │  │  ← Apply point map
│  └───────────────────┘  │
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│   Statistics Collector  │  ← Per-card score accumulation
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│   Results Analyzer      │  ← Rankings, distributions, reports
└─────────────────────────┘
```

### Key Data Structures

```python
# Card data (precomputed, all rotations)
# Shape: (120, 4, 4, 4) → [card_id][rotation][quarter][symbol_slot]
# dtype: uint8, symbol IDs 0-12 (255 = empty)
cards = np.zeros((120, 4, 4, 4), dtype=np.uint8)

# Board state during simulation
# Using offset coordinates to handle negative positions
# Board: 12×12 grid (generous), each cell = (card_layer, quarter_data)
board_symbols = np.zeros((12, 12, 4), dtype=np.uint8)  # symbol slots
board_layer = np.full((12, 12), -1, dtype=np.int8)      # -1 = empty

# Per-card statistics
card_total_score = np.zeros(120, dtype=np.float64)
card_count = np.zeros(120, dtype=np.int64)
```

### Optimization Techniques

1. **Precompute all rotated cards** at startup (480 variants × 16 bytes = 7.5 KB)
2. **Fixed-size board array** (no dynamic allocation in hot loop)
3. **Inline placement + visibility** (no function call overhead with Numba)
4. **Batch random number generation** (generate all random values for a batch upfront)
5. **Avoid branches** in the inner loop where possible
6. **Track footprint incrementally** (maintain bounding box and occupied set as cards are placed)
7. **Use rejection sampling** for valid positions (pick random position in bounding box + margin, retry if no overlap — fast because overlap is common)

---

## Validation Plan

1. **Small-scale exhaustive test**: With 6 cards (not 120), do exhaustive enumeration and compare against Monte Carlo results
2. **Convergence check**: Plot card rankings at 1M, 10M, 100M samples — should stabilize
3. **Known-answer test**: Manually compute score for a specific 6-card arrangement, verify simulation matches
4. **Arrow resolution test**: Create test cards with known arrow configurations, verify resolution

---

## Summary

| Aspect | Decision |
|--------|----------|
| **Search strategy** | Monte Carlo sampling (exhaustive is 10²²+ operations) |
| **Sample count** | 10M–100M for reliable card rankings |
| **Language** | Python + Numba (or Rust for max performance) |
| **Card encoding** | uint8[4] per quarter, 16 bytes per card |
| **Board** | Fixed 12×12 grid of quarter-cells |
| **Arrow resolution** | Single post-placement pass, non-chained |
| **Statistics** | Per-card mean/median score, marginal contribution |
| **Expected runtime** | 10 seconds to 20 minutes depending on sample count |
