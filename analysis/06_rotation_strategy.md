# Rotation Strategy Analysis: Random vs Deliberate Placement

## Executive Summary

The current simulation assumes each card is placed in a **random rotation** (25% chance each).
In reality, a player **chooses** how to rotate each card. This analysis quantifies the impact
of smart rotation on card rankings.

### Key Finding

- **Average gain from smart rotation: +1.61 points per card** (standalone, one quarter hidden)
- Maximum gain for a single card: +5.75 points (Card 105)
- **99 cards** shift ≥5 rank positions between random and smart rotation
- **80 cards** shift ≥10 rank positions

**Biggest winners from smart rotation:**

- Card 46: rank 86→30 (+56 positions, gain=+3.25)
- Card 17: rank 92→46 (+46 positions, gain=+2.50)
- Card 119: rank 82→39 (+43 positions, gain=+3.00)
- Card 52: rank 94→56 (+38 positions, gain=+2.50)
- Card 12: rank 118→85 (+33 positions, gain=+3.00)

**Biggest losers (cards that benefit least, fall in relative ranking):**

- Card 91: rank 22→76 (-54 positions, gain=+0.25)
- Card 114: rank 29→80 (-51 positions, gain=+0.50)
- Card 76: rank 62→102 (-40 positions, gain=+0.50)
- Card 84: rank 33→73 (-40 positions, gain=+0.75)
- Card 92: rank 39→77 (-38 positions, gain=+1.00)

---

## How Rotation Works

Each card is a 2×2 grid of quarters (TL, TR, BL, BR). Each quarter contains symbols.
When cards overlap, typically **one quarter gets covered** by the neighboring card.
The covered quarter's symbols become invisible and don't count toward scoring.

**Random rotation**: The simulation tries all 4 rotations with equal probability.
This means a quarter full of rats has a 25% chance of being the one that's visible in each position.

**Smart rotation**: A player rotates the card so that the quarter with the **worst symbols**
(rats, sharks) faces the direction most likely to be covered, and the quarter with the
**best symbols** (coins, map, rum) stays visible.

---

## Why Some Cards Benefit More

The key factor is **quarter asymmetry** — how unevenly the good and bad symbols are distributed.

**Correlation between quarter asymmetry and rotation gain: r = 0.241**

| Card Type | Rotation Gain | Example |
|---|---|---|
| **Highly asymmetric** (all bad in one quarter) | +3 to +6 pts | Card 105: TL=Coi,Map,Par / BL=Shk,Rat |
| **Moderately asymmetric** | +1 to +3 pts | Most cards |
| **Symmetric** (good/bad evenly spread) | +0.2 to +0.5 pts | Card 91: even distribution |

---

## Full Ranking Comparison

### Cards sorted by smart-rotation rank (top 30)

| Smart Rank | Random Rank | Change | Card | Random Score | Smart Score | Gain | Asymmetry | Best Quarter | Worst Quarter |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 34 | +33 🔺 | 105 | -1.8 | 4.0 | +5.75 | 1.79 | TL: Coi, Map, Par | BL: Shk, ↑, Rat |
| 2 | 5 | +3  | 1 | 0.5 | 3.0 | +2.50 | 1.12 | TL: Par, →, Coi, Rum | BR: Rat, Anc |
| 3 | 2 | -1  | 29 | 0.8 | 3.0 | +2.25 | 1.48 | TR: Anc, Spy, Par | TL: Shk |
| 4 | 3 | -1  | 78 | 0.8 | 3.0 | +2.25 | 1.80 | TR: Coi, Spy, Rum | BL: Rat, Kra |
| 5 | 9 | +4  | 99 | 0.2 | 3.0 | +2.75 | 2.12 | BL: Map, Anc, Coi | BR: Rat, Kra, ←, Kra |
| 6 | 11 | +5 🔺 | 33 | 0.0 | 2.0 | +2.00 | 0.83 | TL: Rum, ←, Spy | BL: Coi, Kra |
| 7 | 6 | -1  | 68 | 0.5 | 2.0 | +1.50 | 1.09 | TR: Anc, Spy | TL: Shk, → |
| 8 | 4 | -4  | 79 | 0.8 | 2.0 | +1.25 | 1.12 | BR: Coi, Par, Spy | TL: →, Coi, Shk |
| 9 | 14 | +5 🔺 | 100 | -0.8 | 2.0 | +2.75 | 1.48 | TR: Par, Spy, Coi | BR: Shk, ↑ |
| 10 | 1 | -9 🔻 | 109 | 1.0 | 2.0 | +1.00 | 1.09 | BL: Anc, Rum, Spy | BR: Rat, Coi |
| 11 | 10 | -1  | 10 | 0.0 | 1.0 | +1.00 | 1.30 | BL: Anc, Map, Spy | TL: Rat, Coi |
| 12 | 23 | +11 🔺 | 11 | -1.5 | 1.0 | +2.50 | 1.64 | BL: Coi, Spy, Rum | TL: Rat |
| 13 | 7 | -6 🔻 | 14 | 0.2 | 1.0 | +0.75 | 1.12 | TL: Rum, Anc | BR: Map, Kra, Shk |
| 14 | 24 | +10 🔺 | 21 | -1.5 | 1.0 | +2.50 | 2.05 | TL: Map, Rum, Coi | TR: Rat, Rat |
| 15 | 32 | +17 🔺 | 39 | -1.8 | 1.0 | +2.75 | 0.83 | TR: Coi | TL: Kra |
| 16 | 13 | -3  | 50 | -0.8 | 1.0 | +1.75 | 1.79 | BR: Spy, Rum, Anc | TR: Rat, Par, Rat, Shk |
| 17 | 25 | +8 🔺 | 56 | -1.5 | 1.0 | +2.50 | 1.64 | BL: Rum, Coi, Spy | TL: Shk |
| 18 | 8 | -10 🔻 | 72 | 0.2 | 1.0 | +0.75 | 1.12 | BR: Coi, Anc | BL: Kra, Rum, Kra |
| 19 | 52 | +33 🔺 | 86 | -2.2 | 1.0 | +3.25 | 1.92 | TL: Map, Spy, Coi | BL: Rat, Rat |
| 20 | 12 | -8 🔻 | 97 | -0.5 | 1.0 | +1.50 | 1.87 | BL: Spy, Coi, Par, → | BR: ↓, Kra, Rat |
| 21 | 28 | +7 🔺 | 110 | -1.5 | 1.0 | +2.50 | 1.48 | BL: Anc, Shk, Coi, Rum | TL: Rat, ↓, Rat |
| 22 | 40 | +18 🔺 | 2 | -2.2 | 0.0 | +2.25 | 1.09 | TR: Rum, Par | TL: Kra, Rat, ↑, Spy |
| 23 | 19 | -4  | 9 | -1.2 | 0.0 | +1.25 | 0.83 | TR: ←, Anc, Map | TL:  |
| 24 | 35 | +11 🔺 | 13 | -2.0 | 0.0 | +2.00 | 1.30 | TR: Spy, Map | BR: ↑, Kra |
| 25 | 42 | +17 🔺 | 23 | -2.2 | 0.0 | +2.25 | 1.80 | BL: Coi, Anc, Anc | TL: Kra, Rat |
| 26 | 43 | +17 🔺 | 24 | -2.2 | 0.0 | +2.25 | 1.12 | TL: Coi, Rum | BR: Rat, ← |
| 27 | 44 | +17 🔺 | 25 | -2.2 | 0.0 | +2.25 | 1.41 | TR: Anc, Anc | TL: Rat, Rat |
| 28 | 17 | -11 🔻 | 38 | -1.0 | 0.0 | +1.00 | 0.83 | TR: Anc, Rat, Spy, Map | BR: Coi, Shk |
| 29 | 46 | +17 🔺 | 40 | -2.2 | 0.0 | +2.25 | 1.80 | BR: Spy, ←, Spy, Spy | TR: Rat, Rat |
| 30 | 86 | +56 🔺 | 46 | -3.2 | 0.0 | +3.25 | 1.22 | TL: Spy, Shk, Coi | TR: Rat, Rat |

### Bottom 30

| Smart Rank | Random Rank | Change | Card | Random Score | Smart Score | Gain | Worst Quarter |
|---|---|---|---|---|---|---|---|
| 91 | 84 | -7 🔻 | 41 | -3.2 | -2.0 | +1.25 | TL: Shk |
| 92 | 71 | -21 🔻 | 42 | -2.8 | -2.0 | +0.75 | TL: Kra, ←, Rat |
| 93 | 85 | -8 🔻 | 45 | -3.2 | -2.0 | +1.25 | TL: Rat |
| 94 | 72 | -22 🔻 | 51 | -2.8 | -2.0 | +0.75 | BL: ↑ |
| 95 | 114 | +19 🔺 | 55 | -4.0 | -2.0 | +2.00 | BL: Rat |
| 96 | 73 | -23 🔻 | 58 | -2.8 | -2.0 | +0.75 | TR: Kra, Coi, Shk |
| 97 | 103 | +6 🔺 | 60 | -3.8 | -2.0 | +1.75 | TL: Rat |
| 98 | 95 | -3  | 61 | -3.5 | -2.0 | +1.50 | TR: Rat, Coi, Rat |
| 99 | 104 | +5 🔺 | 66 | -3.8 | -2.0 | +1.75 | TL: Rat |
| 100 | 96 | -4  | 70 | -3.5 | -2.0 | +1.50 | BL: Rat |
| 101 | 89 | -12 🔻 | 74 | -3.2 | -2.0 | +1.25 | BL: Rat, Rat |
| 102 | 62 | -40 🔻 | 76 | -2.5 | -2.0 | +0.50 | TR: Coi, Rat, Shk |
| 103 | 74 | -29 🔻 | 85 | -2.8 | -2.0 | +0.75 | BL: Rat |
| 104 | 75 | -29 🔻 | 89 | -2.8 | -2.0 | +0.75 | TL: Shk, Coi, → |
| 105 | 106 | +1  | 98 | -3.8 | -2.0 | +1.75 | BR: Rat |
| 106 | 91 | -15 🔻 | 101 | -3.2 | -2.0 | +1.25 | TR: ← |
| 107 | 117 | +10 🔺 | 103 | -4.2 | -2.0 | +2.25 | BL: Shk, Rat, Rat |
| 108 | 81 | -27 🔻 | 106 | -3.0 | -2.0 | +1.00 | TR:  |
| 109 | 99 | -10 🔻 | 108 | -3.5 | -2.0 | +1.50 | TL: Shk |
| 110 | 107 | -3  | 113 | -3.8 | -2.0 | +1.75 | BL: Shk, Shk, Shk |
| 111 | 93 | -18 🔻 | 18 | -3.5 | -3.0 | +0.50 | TR: Shk |
| 112 | 83 | -29 🔻 | 26 | -3.2 | -3.0 | +0.25 | TR: Rat |
| 113 | 112 | -1  | 31 | -4.0 | -3.0 | +1.00 | TL: Rat |
| 114 | 102 | -12 🔻 | 35 | -3.8 | -3.0 | +0.75 | BR: Rat, Shk |
| 115 | 116 | +1  | 88 | -4.2 | -3.0 | +1.25 | BR: Rat, Rat |
| 116 | 98 | -18 🔻 | 93 | -3.5 | -3.0 | +0.50 | TR: Shk, Rat |
| 117 | 105 | -12 🔻 | 96 | -3.8 | -3.0 | +0.75 | BR: Rat |
| 118 | 100 | -18 🔻 | 117 | -3.5 | -3.0 | +0.50 | BL: Shk, Rat |
| 119 | 119 | 0  | 43 | -5.0 | -4.0 | +1.00 | BL: Rat, Shk, Shk, Rat |
| 120 | 120 | 0  | 107 | -5.8 | -4.0 | +1.75 | TL: Shk, Shk, Rat |

---

## Biggest Rank Movers: Case Studies

### Card 46 — CLIMBER 🔺 (rank 86→30, +56)

```
  TL: Spy, Shk, Coi                  (net: +1) ← PROTECT
  TR: Rat, Rat                       (net: -2) ← HIDE THIS
  BL: Map, Kra                       (net: +0)
  BR: ↑, Spy, ←                      (net: +1)
```

- **Random rotation score**: -3.2
- **Smart rotation score**: 0.0 (gain: +3.25)
- **Quarter asymmetry**: 1.22
- **Why it climbs**: The worst quarter (TR) has concentrated negative symbols.
  Smart rotation hides this quarter, removing the damage. Random rotation
  exposes it 25% of the time unnecessarily.

### Card 91 — FALLER 🔻 (rank 22→76, -54)

```
  TL: ↓, Shk, Coi                    (net: +0) ← HIDE THIS
  TR: Anc, Kra                       (net: +0)
  BL: ↑, Map                         (net: +1)
  BR: Rum, Coi                       (net: +2) ← PROTECT
```

- **Random rotation score**: -1.2
- **Smart rotation score**: -1.0 (gain: +0.25)
- **Quarter asymmetry**: 0.83
- **Why it falls**: This card has evenly distributed symbols — rotation
  doesn't help much. Other cards gain more from smart rotation, pushing this one down.

### Card 114 — FALLER 🔻 (rank 29→80, -51)

```
  TL: Spy, Shk, Anc                  (net: +1)
  TR: Rum                            (net: +1)
  BL: Shk, Rat, Coi                  (net: -1) ← HIDE THIS
  BR: Coi, Rum                       (net: +2) ← PROTECT
```

- **Random rotation score**: -1.5
- **Smart rotation score**: -1.0 (gain: +0.50)
- **Quarter asymmetry**: 1.09
- **Why it falls**: This card has evenly distributed symbols — rotation
  doesn't help much. Other cards gain more from smart rotation, pushing this one down.

### Card 17 — CLIMBER 🔺 (rank 92→46, +46)

```
  TL: Coi, Map, Kra                  (net: +1)
  TR: ↓, Shk                         (net: -1)
  BL: Shk, Rat, Rat                  (net: -3) ← HIDE THIS
  BR: Anc, Anc                       (net: +2) ← PROTECT
```

- **Random rotation score**: -3.5
- **Smart rotation score**: -1.0 (gain: +2.50)
- **Quarter asymmetry**: 1.92
- **Why it climbs**: The worst quarter (BL) has concentrated negative symbols.
  Smart rotation hides this quarter, removing the damage. Random rotation
  exposes it 25% of the time unnecessarily.

### Card 119 — CLIMBER 🔺 (rank 82→39, +43)

```
  TL: Coi                            (net: +1)
  TR: Rat, Shk, Rat                  (net: -3) ← HIDE THIS
  BL: Map, Kra                       (net: +0)
  BR: Map, Coi, ←                    (net: +2) ← PROTECT
```

- **Random rotation score**: -3.0
- **Smart rotation score**: 0.0 (gain: +3.00)
- **Quarter asymmetry**: 1.87
- **Why it climbs**: The worst quarter (TR) has concentrated negative symbols.
  Smart rotation hides this quarter, removing the damage. Random rotation
  exposes it 25% of the time unnecessarily.

### Card 76 — FALLER 🔻 (rank 62→102, -40)

```
  TL: Rum                            (net: +1)
  TR: Coi, Rat, Shk                  (net: -1) ← HIDE THIS
  BL: Map, Spy                       (net: +2) ← PROTECT
  BR: Anc, Map                       (net: +2)
```

- **Random rotation score**: -2.5
- **Smart rotation score**: -2.0 (gain: +0.50)
- **Quarter asymmetry**: 1.22
- **Why it falls**: This card has evenly distributed symbols — rotation
  doesn't help much. Other cards gain more from smart rotation, pushing this one down.

### Card 84 — FALLER 🔻 (rank 33→73, -40)

```
  TL: Rat                            (net: -1) ← HIDE THIS
  TR: ↓, Par, Coi                    (net: +2) ← PROTECT
  BL: Par, Anc, Kra                  (net: +1)
  BR: Anc                            (net: +1)
```

- **Random rotation score**: -1.8
- **Smart rotation score**: -1.0 (gain: +0.75)
- **Quarter asymmetry**: 1.09
- **Why it falls**: This card has evenly distributed symbols — rotation
  doesn't help much. Other cards gain more from smart rotation, pushing this one down.

### Card 52 — CLIMBER 🔺 (rank 94→56, +38)

```
  TL: →, →, Coi                      (net: +1) ← PROTECT
  TR: Spy, →                         (net: +1)
  BL: Rat, Rat                       (net: -2) ← HIDE THIS
  BR: Map, Shk, Rum                  (net: +1)
```

- **Random rotation score**: -3.5
- **Smart rotation score**: -1.0 (gain: +2.50)
- **Quarter asymmetry**: 1.30
- **Why it climbs**: The worst quarter (BL) has concentrated negative symbols.
  Smart rotation hides this quarter, removing the damage. Random rotation
  exposes it 25% of the time unnecessarily.

### Card 92 — FALLER 🔻 (rank 39→77, -38)

```
  TL: Rat, Shk, Spy                  (net: -1) ← HIDE THIS
  TR: ↓, Map, Par                    (net: +2) ← PROTECT
  BL: Anc, Shk                       (net: +0)
  BR: Spy, ↑                         (net: +1)
```

- **Random rotation score**: -2.0
- **Smart rotation score**: -1.0 (gain: +1.00)
- **Quarter asymmetry**: 1.12
- **Why it falls**: This card has evenly distributed symbols — rotation
  doesn't help much. Other cards gain more from smart rotation, pushing this one down.

### Card 64 — FALLER 🔻 (rank 27→63, -36)

```
  TL: Map, Kra, Map                  (net: +1) ← PROTECT
  TR: Rat, Spy                       (net: +0) ← HIDE THIS
  BL: Shk, Rum                       (net: +0)
  BR: Coi, Shk, Rum                  (net: +1)
```

- **Random rotation score**: -1.5
- **Smart rotation score**: -1.0 (gain: +0.50)
- **Quarter asymmetry**: 0.50
- **Why it falls**: This card has evenly distributed symbols — rotation
  doesn't help much. Other cards gain more from smart rotation, pushing this one down.

---

## Statistical Summary

| Metric | Value |
|---|---|
| Cards analyzed | 120 |
| Avg gain from smart rotation | +1.61 pts |
| Median gain | +1.50 pts |
| Max gain | +5.75 pts (Card 105) |
| Min gain | +0.25 pts (Card 26) |
| Cards shifting ≥5 ranks | 99 |
| Cards shifting ≥10 ranks | 80 |
| Rank correlation (random vs smart) | 0.8167 |
| Asymmetry↔Gain correlation | 0.241 |
| Spearman ρ (random vs smart ranking) | 0.8167 |

## Answer: Can a card be bad with random placement and good with deliberate placement?

**12 cards cross from the bottom half to the top half** with smart rotation:

- Card 46: random rank 86 → smart rank 30 (gain: +3.25)
- Card 112: random rank 64 → smart rank 37 (gain: +2.50)
- Card 119: random rank 82 → smart rank 39 (gain: +3.00)
- Card 0: random rank 66 → smart rank 40 (gain: +1.75)
- Card 3: random rank 67 → smart rank 41 (gain: +1.75)
- Card 5: random rank 68 → smart rank 42 (gain: +1.75)
- Card 6: random rank 69 → smart rank 43 (gain: +1.75)
- Card 7: random rank 70 → smart rank 44 (gain: +1.75)
- Card 17: random rank 92 → smart rank 46 (gain: +2.50)
- Card 34: random rank 77 → smart rank 51 (gain: +2.00)
- Card 52: random rank 94 → smart rank 56 (gain: +2.50)
- Card 57: random rank 87 → smart rank 59 (gain: +2.25)

**12 cards cross from the top half to the bottom half:**

- Card 114: random rank 29 → smart rank 80 (gain: +0.50)
- Card 104: random rank 55 → smart rank 79 (gain: +1.25)
- Card 95: random rank 54 → smart rank 78 (gain: +1.25)
- Card 92: random rank 39 → smart rank 77 (gain: +1.00)
- Card 91: random rank 22 → smart rank 76 (gain: +0.25)
- Card 87: random rank 38 → smart rank 74 (gain: +1.00)
- Card 84: random rank 33 → smart rank 73 (gain: +0.75)
- Card 83: random rank 51 → smart rank 72 (gain: +1.25)
- Card 80: random rank 50 → smart rank 69 (gain: +1.25)
- Card 67: random rank 49 → smart rank 64 (gain: +1.25)
- Card 64: random rank 27 → smart rank 63 (gain: +0.50)
- Card 62: random rank 26 → smart rank 61 (gain: +0.50)

### The Verdict

- **Top 10 overlap**: 7/10 cards are in the top 10 under both strategies
  - Random top 10: [1, 10, 14, 29, 68, 72, 78, 79, 99, 109]
  - Smart top 10: [1, 29, 33, 68, 78, 79, 99, 100, 105, 109]
  - Entered smart top 10: [33, 100, 105]
  - Dropped from top 10: [10, 14, 72]

- **Bottom 10 overlap**: 4/10 cards remain in both

**Conclusion**: Smart rotation changes the **middle rankings** significantly (±5-15 positions)
but the **best and worst cards remain mostly the same**. A card that's terrible with random
rotation won't become good with smart rotation — its symbols are still bad. But a card with
concentrated bad symbols in one quarter is **less bad** than the random simulation suggests,
because a smart player hides that quarter.

The biggest impact is on **asymmetric cards** — those with all their rats/sharks in one quarter.
These cards are **undervalued by the random-rotation simulation** and would climb 5-15 ranks
with deliberate placement.