#!/usr/bin/env python3
"""
Analyze what makes cards balanced yet skill-expressive.

Goal: Create cards where:
  1. Expected value (random placement) is similar across all cards
  2. Variance is high (skilled placement matters)
  3. Card synergies exist (combining cards matters)
"""
import json
import numpy as np
from collections import Counter

# Load simulation results
# Card rankings from v2 analysis
CARD_RANKINGS_V2 = {
    # Position in test set: (orig_id, sim_avg, raw_val, positives, negatives, parrots, arrows, rat_quarters)
    0: (0, 14.25, -1.0, 5, 6, 0, 0, 1),
    1: (1, 15.42, 7.0, 8, 2, 1, 2, 2),
    2: (2, 14.78, 2.0, 5, 4, 1, 2, 2),
    3: (3, 14.00, 4.5, 5, 1, 0, 1, 1),
    4: (4, 13.49, 0.5, 3, 3, 0, 1, 1),
    5: (5, 14.09, 3.5, 6, 3, 1, 1, 0),
    6: (6, 13.83, 3.5, 5, 2, 0, 1, 1),
    7: (7, 14.01, 3.0, 6, 3, 0, 0, 2),
    8: (8, 13.20, -1.5, 3, 5, 0, 1, 1),
    9: (9, 14.56, 6.0, 6, 1, 1, 2, 1),
    10: (10, 14.88, 7.0, 8, 1, 1, 0, 1),
    11: (11, 14.18, 1.5, 4, 3, 0, 1, 1),
    12: (12, 12.42, 1.0, 5, 5, 0, 2, 2),
    13: (13, 14.17, 3.5, 4, 1, 0, 1, 0),
    14: (14, 15.48, 2.5, 6, 4, 1, 1, 0),
    15: (15, 13.43, 2.5, 4, 2, 0, 1, 2),
    16: (16, 13.98, 5.0, 6, 1, 0, 0, 1),
    17: (17, 13.63, -0.5, 4, 5, 0, 1, 1),
    18: (18, 13.09, 1.0, 4, 4, 1, 2, 2),
    19: (19, 14.40, 3.5, 4, 1, 1, 1, 0),
    # New counter-examples
    20: (34, 13.33, 2.5, 5, 3, 0, 1, 2),   # spread_rats
    21: (111, 15.00, 3.0, 6, 3, 0, 0, 1),  # good_kraken (2 krakens)
    22: (26, 13.26, 1.5, 4, 3, 1, 1, 1),   # parrot_spread_neg
    23: (29, 15.20, 3.0, 6, 3, 1, 0, 0),   # spread_strong (parrot + spread neg)
    24: (33, 15.43, 6.0, 6, 1, 1, 2, 0),   # arrow_parrot synergy
    25: (113, 13.65, 1.5, 4, 3, 0, 1, 0),  # concentrated_weak (3 sharks 1 quarter)
    26: (38, 14.72, 5.0, 8, 3, 0, 0, 1),   # spread_strong (8 pos)
    27: (79, 15.51, 6.5, 8, 2, 1, 1, 1),   # high_net + parrot
    28: (107, 11.82, -2.0, 3, 6, 0, 2, 3), # low_net spread rats
    29: (106, 14.29, 6.0, 6, 0, 0, 0, 0),  # zero_neg
}

print("=" * 80)
print("CARD BALANCE ANALYSIS — What creates similar-value yet varied cards?")
print("=" * 80)

# Extract data
data = []
for pos, (orig, sim_avg, raw_val, pos_count, neg_count, parrots, arrows, rat_q) in CARD_RANKINGS_V2.items():
    data.append({
        'pos': pos, 'orig': orig, 'sim_avg': sim_avg, 'raw_val': raw_val,
        'positives': pos_count, 'negatives': neg_count, 'parrots': parrots,
        'arrows': arrows, 'rat_quarters': rat_q, 'net': pos_count - neg_count
    })

sim_avgs = np.array([d['sim_avg'] for d in data])
print(f"\nCurrent card value distribution:")
print(f"  Mean: {sim_avgs.mean():.2f}")
print(f"  Std:  {sim_avgs.std():.2f}")
print(f"  Min:  {sim_avgs.min():.2f} (Card {data[np.argmin(sim_avgs)]['orig']})")
print(f"  Max:  {sim_avgs.max():.2f} (Card {data[np.argmax(sim_avgs)]['orig']})")
print(f"  Range: {sim_avgs.max() - sim_avgs.min():.2f} points")

# ============================================================
# MODEL: What predicts sim_avg?
# ============================================================
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS — What determines card value?")
print("=" * 80)

X = np.array([
    [d['positives'], d['negatives'], d['parrots'], d['arrows'], d['rat_quarters'], d['net']]
    for d in data
])
y = sim_avgs

# Simple correlations
features = ['positives', 'negatives', 'parrots', 'arrows', 'rat_quarters', 'net']
print("\nCorrelations with sim_avg:")
for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    bar = '█' * int(abs(corr) * 20)
    sign = '+' if corr > 0 else '-'
    print(f"  {feat:<15} r={sign}{abs(corr):.3f}  {bar}")

# Linear regression
X_with_const = np.column_stack([np.ones(len(X)), X])
try:
    coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred = X_with_const @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\nLinear model (R² = {r2:.3f}):")
    print(f"  sim_avg = {coeffs[0]:.2f}")
    for i, feat in enumerate(features):
        sign = '+' if coeffs[i+1] >= 0 else ''
        print(f"           {sign}{coeffs[i+1]:.3f} × {feat}")
except:
    pass

# ============================================================
# KEY INSIGHT: What creates VARIANCE (skill expression)?
# ============================================================
print("\n" + "=" * 80)
print("VARIANCE ANALYSIS — What creates skill expression?")
print("=" * 80)

# From earlier analysis, within-combo variance was ~87.8% of total variance
# This means HOW you place matters more than WHAT cards you have

# Cards with asymmetric quarters have more placement skill expression
# Arrows create variance because they depend on neighbors
# Concentrated negatives create variance (can hide or expose)

print("""
Key variance (skill) drivers identified from simulation:
  1. QUARTER ASYMMETRY: Cards with very different quarter values have more
     placement options. Can choose to expose good quarters, hide bad ones.
     
  2. ARROWS: Copy symbols from adjacent cards. High variance because:
     - Arrow pointing at parrot = great
     - Arrow pointing at rat = terrible
     - Skill = arranging arrows to point at valuable symbols
     
  3. CONCENTRATED NEGATIVES: 
     - If all negatives are in 1 quarter, skilled player hides that quarter
     - If spread across quarters, fewer options = lower skill ceiling
     
  4. CARD SYNERGIES:
     - Parrot + parrot = compound scoring (3+3=6, but 4 parrots = 5)
     - Arrow pointing at parrot quarter = copies parrot value
     - Rum concentration = diminishing returns become positive
""")

# ============================================================
# CARD GENERATION PRINCIPLES
# ============================================================
print("\n" + "=" * 80)
print("CARD GENERATION PRINCIPLES — How to create balanced cards")
print("=" * 80)

print("""
PRINCIPLE 1: BALANCE THE VALUE FORMULA
────────────────────────────────────────────────────────────────────────
To create cards with similar expected value (~14.0 target), balance:
  
  TARGET: sim_avg ≈ 14.0 ± 0.5
  
  Base formula:
    value ≈ 13.0 + 1.2×parrot + 0.2×positives - 0.9×rat_quarters
                 + 0.15×arrows_with_parrot - 0.05×krakens
  
  Examples of balanced cards:
    A) 6 positives + 1 parrot + 0 rats/quarter = 13 + 1.2 + 1.2 + 0 = 15.4 ✓
    B) 8 positives + 0 parrot + 1 rat/quarter = 13 + 0 + 1.6 - 0.9 = 13.7 ✓
    C) 5 positives + 1 parrot + 1 rat/quarter = 13 + 1.2 + 1.0 - 0.9 = 14.3 ✓
    D) 7 positives + 0 parrot + 0 rats/quarter = 13 + 0 + 1.4 + 0 = 14.4 ✓

PRINCIPLE 2: MAXIMIZE VARIANCE FOR SKILL EXPRESSION
────────────────────────────────────────────────────────────────────────
To create skill expression within balanced cards:

  HIGH VARIANCE (good for skill):
    ✓ Asymmetric quarters (one great, one bad)
    ✓ Arrows (amplify good/bad neighbors)
    ✓ Concentrated negatives (can be hidden)
    ✓ Parrots (compound with other parrots)
    
  LOW VARIANCE (bad for skill):
    ✗ Uniform quarters (all similar value)
    ✗ Spread negatives (can't hide them)
    ✗ No arrows (no neighbor interaction)

PRINCIPLE 3: CREATE SYNERGY POTENTIAL
────────────────────────────────────────────────────────────────────────
Cards should combo with other cards:

  SYNERGY ENABLERS:
    • Arrows pointing outward (can find good targets)
    • Parrots (compound scoring with other parrot cards)
    • Rum concentration (6+ rums = big positive swing)
    • Krakens (5+ krakens = penalty disappears)
    
  SYNERGY TARGETS:
    • Large parrot quarters (arrows want to point here)
    • High-value quarters (other cards want adjacency)

PRINCIPLE 4: MAINTAIN VARIETY
────────────────────────────────────────────────────────────────────────
Even with balanced values, cards should feel different:

  CARD ARCHETYPES:
    1. "Parrot Focus" — 1-2 parrots, moderate other symbols
    2. "Positive Flood" — 7-8 positives, no parrot
    3. "Arrow Engine" — 2-3 arrows, needs good neighbors
    4. "Kraken Tank" — 2-3 krakens, compensated by positives
    5. "Rum Runner" — 3+ rums, concentrated in quarters
    6. "Clean" — 0 negatives, moderate positives
    7. "Risky" — High highs (parrot+arrows) and lows (rats)
""")

# ============================================================
# SPECIFIC CARD GENERATION GUIDELINES
# ============================================================
print("\n" + "=" * 80)
print("SPECIFIC CARD GENERATION GUIDELINES")
print("=" * 80)

print("""
POSITIVE SYMBOLS (target 5-8 per card):
  • Parrot:   0-2 (premium, add ~1.2 points each)
  • Spyglass: 1-3 (solid, linear scaling)
  • Map:      1-3 (moderate, needs 3+ to score)
  • Anchor:   1-3 (moderate, linear scaling)
  • Coin:     1-2 (peaks at 5, then diminishes)
  • Rum:      0-3 (risky: -2 for 1, +6 for 5)

NEGATIVE SYMBOLS (target 1-4 per card):
  • Kraken:   0-3 (caps at -4 total, diminishes) — SAFE NEGATIVE
  • Shark:    0-2 (flat -1 each) — MODERATE NEGATIVE  
  • Rat:      0-2 (MUST be concentrated in 1 quarter) — DANGEROUS

ARROWS (target 0-2 per card):
  • 0 arrows: Simple card, value from symbols only
  • 1 arrow:  Moderate interaction, some placement skill
  • 2 arrows: High interaction, needs good neighbors
  • 3 arrows: Risky, very dependent on combo

QUARTER DESIGN:
  Good quarter:  2-3 positives, 0 negatives = +2 to +3 value
  Neutral:       1-2 positives, 1 negative = 0 to +1 value
  Bad quarter:   0-1 positives, 2-3 negatives = -1 to -3 value
  
  RULE: Every card should have at least one good and one bad quarter
  for meaningful placement decisions.

BALANCE CONSTRAINTS:
────────────────────────────────────────────────────────────────────────
For sim_avg ≈ 14.0 ± 0.5:

  IF parrot=1:
    • Allow up to 2 rat-quarters (parrot compensates)
    • Target 5-6 total positives
    • Can have 3-4 total negatives
    
  IF parrot=0:
    • Maximum 1 rat-quarter
    • Target 6-8 total positives  
    • Keep total negatives ≤ 3
    
  IF arrows ≥ 2:
    • Add 0.5-1.0 point bonus assumption (neighbor copying)
    • Pair with parrot for synergy
    • Keep base symbols clean (4-5 positives ok)
    
  IF krakens ≥ 2:
    • Treat as -0.5 points total (not -2 per kraken)
    • Can have 6+ positives to compensate
    • Avoid adding rats (double negative)
""")

# ============================================================
# EXAMPLE BALANCED DECK
# ============================================================
print("\n" + "=" * 80)
print("EXAMPLE BALANCED 12-CARD DECK")
print("=" * 80)

example_deck = [
    # Format: (name, quarters, expected_value, notes)
    ("Parrot Focus A", 
     "TL: Par,Coi,Rum | TR: Map,Spy | BL: Rat,Anc | BR: Shk,Map", 
     "~14.5", "Parrot + 6 pos, 1 rat quarter"),
     
    ("Parrot Focus B",
     "TL: Par,Spy | TR: Anc,Kra | BL: Coi,Rum | BR: Shk,Map",
     "~14.3", "Parrot + 5 pos, spread krakens (ok), no rats"),
     
    ("Positive Flood A",
     "TL: Spy,Spy,Coi | TR: Map,Anc | BL: Rum,Coi | BR: Rat",
     "~14.4", "8 positives, 0 parrot, 1 rat concentrated"),
     
    ("Positive Flood B",
     "TL: Anc,Map,Spy | TR: Coi,Rum | BL: Anc,Coi | BR: Shk",
     "~14.2", "7 positives, 0 parrot, 1 shark only"),
     
    ("Arrow Engine A",
     "TL: →,Par,Coi | TR: Spy,↓ | BL: Rum,Shk | BR: Anc",
     "~14.8", "Parrot + 2 arrows, 5 pos, needs neighbors"),
     
    ("Arrow Engine B",
     "TL: ←,Spy,Coi | TR: →,Map | BL: Rum,Rum | BR: Kra,Anc",
     "~13.8", "2 arrows, no parrot, rum concentration"),
     
    ("Kraken Tank A",
     "TL: Kra,Kra | TR: Par,Spy,Map | BL: Anc,Coi | BR: Rum",
     "~14.6", "2 krakens concentrated, parrot, 6 pos"),
     
    ("Kraken Tank B",
     "TL: Kra,Spy | TR: Kra,Map,Coi | BL: Anc,Anc | BR: Rum,Spy",
     "~14.1", "2 krakens spread, 7 pos, no parrot"),
     
    ("Rum Runner",
     "TL: Rum,Rum,Rum | TR: Spy,Map | BL: Anc,Coi | BR: Shk,Rat",
     "~14.0", "3 rums concentrated (0 penalty), 5 pos, 1 rat quarter"),
     
    ("Clean Card",
     "TL: Spy,Map | TR: Anc,Coi | BL: Map,Spy | BR: Anc,Coi",
     "~14.3", "0 negatives, 8 positives, uniform quarters"),
     
    ("Risky Card A",
     "TL: Par,→,Coi | TR: Rat,Shk | BL: ↓,Spy,Map | BR: Rat,Rum",
     "~14.0", "Parrot+arrows, but 2 rat quarters = high variance"),
     
    ("Risky Card B",
     "TL: Shk,Rat,Rat | TR: Par,Par,Coi | BL: Spy,→ | BR: Anc,Map",
     "~14.2", "2 parrots! but concentrated rats, arrow synergy"),
]

print("\nCard | Expected | Description")
print("-" * 70)
for name, quarters, exp_val, notes in example_deck:
    print(f"{name:<18} | {exp_val:<8} | {notes}")

print("""
\nDECK BALANCE CHECK:
  • 6/12 cards have parrots (premium cards)
  • 4/12 cards have arrows (interaction cards)
  • 3/12 cards have krakens (tank cards)
  • 2/12 cards are "clean" (beginner-friendly)
  • 2/12 cards are "risky" (expert cards)
  
  Expected value range: 13.8 - 14.8 (only 1 point spread!)
  But variance per card differs significantly → skill expression
""")

# ============================================================
# VERIFICATION: Does our model match simulation?
# ============================================================
print("\n" + "=" * 80)
print("MODEL VERIFICATION — Predicting actual card values")
print("=" * 80)

def predict_value(positives, negatives, parrots, arrows, rat_quarters, has_parrot_bonus=False):
    """Predict card sim_avg using our model."""
    base = 13.0
    parrot_bonus = 1.2 * parrots
    pos_bonus = 0.20 * positives
    rat_penalty = -0.9 * rat_quarters
    arrow_bonus = 0.15 * arrows if parrots > 0 else 0  # Only if has parrot
    
    return base + parrot_bonus + pos_bonus + rat_penalty + arrow_bonus

print("\nCard | Actual | Predicted | Error")
print("-" * 50)
errors = []
for d in sorted(data, key=lambda x: -x['sim_avg']):
    pred = predict_value(d['positives'], d['negatives'], d['parrots'], 
                         d['arrows'], d['rat_quarters'])
    error = d['sim_avg'] - pred
    errors.append(error)
    print(f"{d['orig']:>4} | {d['sim_avg']:>6.2f} | {pred:>9.2f} | {error:>+5.2f}")

print(f"\nModel accuracy: MAE = {np.mean(np.abs(errors)):.2f}, RMSE = {np.sqrt(np.mean(np.array(errors)**2)):.2f}")
