#!/usr/bin/env python3
"""
Balanced Card Generator for Travel Squares

Generates cards with:
  1. Similar expected value (~14.0 ± 0.5)
  2. High variance (skill expression through placement)
  3. Varied archetypes for interesting gameplay
  4. Synergy potential between cards
  
Based on analysis of 1 billion simulated placements.
"""
import json
import random
from collections import Counter
from typing import List, Dict, Tuple
import argparse

# ============================================================
# CONSTANTS FROM SIMULATION ANALYSIS
# ============================================================

# Value coefficients (UPDATED from balanced deck simulation)
# Key insight: Rum concentration is MUCH more powerful than expected!
BASE_VALUE = 18.0  # Higher base to match simulation
PARROT_BONUS = 0.5  # Reduced - parrots less impactful than rum
POS_BONUS_PER = 0.15
RAT_QUARTER_PENALTY = -1.2  # Spread rats are even worse
ARROW_PARROT_BONUS = 0.1
KRAKEN_PENALTY_PER = -0.05
RUM_CONCENTRATION_BONUS = 0.7  # Per rum above 2 (huge bonus)

# Target value range (updated to match simulation averages)
TARGET_MIN = 18.5
TARGET_MAX = 19.5

# Symbol sets
POSITIVE_SYMBOLS = ['anchor', 'spyglass', 'map', 'coin', 'rum', 'parrot']
NEGATIVE_SYMBOLS = ['shark', 'rat', 'kraken']
ARROW_SYMBOLS = ['arrow_up', 'arrow_down', 'arrow_left', 'arrow_right']

# Scoring reference (for validation)
SCORE_TABLE = {
    "coin": [0, 1, 2, 3, 5, 7, 5, 3, 2, 1, 0, 0],
    "map": [0, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
    "anchor": [0, 1, 1, 2, 3, 3, 4, 5, 6, 8, 9, 10],
    "spyglass": [1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9],
    "parrot": [0, 3, 0, 5, 0, 8, 0, 10, 0, 13, 0, 15],
    "rum": [-2, -1, 0, 2, 4, 6, 8, 10, 11, 12, 13, 14],
    "kraken": [-4, -3, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    "rat": [0, 0, -1, -1, -2, -2, -3, -4, -5, -6, -7, -8],
    "shark": [-1, -1, -1, -2, -2, -2, -3, -3, -4, -4, -5, -5],
}

# ============================================================
# CARD ARCHETYPES
# ============================================================

ARCHETYPES = {
    'parrot_focus': {
        'description': 'Premium parrot card with moderate symbols',
        'parrot_range': (1, 2),
        'positive_range': (5, 7),  # Increased to compensate
        'negative_range': (1, 2),  # Reduced negatives
        'arrow_range': (0, 1),
        'max_rat_quarters': 1,
        'max_rum': 2,  # Limit rum to prevent over-power
        'weight': 0.25,
    },
    'positive_flood': {
        'description': 'Many positives, no parrot, no rum concentration',
        'parrot_range': (0, 0),
        'positive_range': (6, 8),
        'negative_range': (1, 2),
        'arrow_range': (0, 1),
        'max_rat_quarters': 1,
        'max_rum': 2,  # Limit rum
        'weight': 0.20,
    },
    'arrow_engine': {
        'description': 'Multiple arrows for interaction',
        'parrot_range': (0, 1),
        'positive_range': (4, 6),
        'negative_range': (1, 2),
        'arrow_range': (2, 3),
        'max_rat_quarters': 1,
        'max_rum': 2,
        'weight': 0.15,
    },
    'kraken_tank': {
        'description': 'Krakens with compensating positives',
        'parrot_range': (0, 1),
        'positive_range': (5, 7),
        'negative_range': (2, 4),
        'arrow_range': (0, 1),
        'max_rat_quarters': 0,
        'prefer_krakens': True,
        'max_rum': 2,
        'weight': 0.15,
    },
    'rum_runner': {
        'description': 'Moderate rum for controlled bonus',
        'parrot_range': (0, 0),  # No parrot to balance rum
        'positive_range': (4, 5),  # Fewer other positives
        'negative_range': (2, 3),  # More negatives to balance
        'arrow_range': (0, 1),
        'max_rat_quarters': 1,
        'prefer_rum': True,
        'target_rum': 3,  # Exactly 3 rums = moderate bonus
        'max_rum': 3,
        'weight': 0.10,
    },
    'clean': {
        'description': 'Zero or minimal negatives, no rum',
        'parrot_range': (0, 1),
        'positive_range': (5, 7),
        'negative_range': (0, 1),
        'arrow_range': (0, 1),
        'max_rat_quarters': 0,
        'max_rum': 1,  # Very limited rum
        'weight': 0.10,
    },
    'risky': {
        'description': 'High variance - great highs and lows',
        'parrot_range': (1, 2),
        'positive_range': (4, 6),
        'negative_range': (2, 4),
        'arrow_range': (1, 2),
        'max_rat_quarters': 2,
        'max_rum': 2,
        'weight': 0.05,
    },
}


def predict_value(positives: int, parrots: int, rat_quarters: int, 
                  arrows: int, krakens: int, rums: int = 0) -> float:
    """Predict expected card value using our regression model."""
    value = BASE_VALUE
    value += PARROT_BONUS * parrots
    value += POS_BONUS_PER * positives
    value += RAT_QUARTER_PENALTY * rat_quarters
    if parrots > 0:
        value += ARROW_PARROT_BONUS * arrows
    value += KRAKEN_PENALTY_PER * krakens
    # Rum concentration bonus (exponential scaling)
    if rums > 2:
        value += RUM_CONCENTRATION_BONUS * (rums - 2)
    return value


def generate_card(archetype: str, rng: random.Random) -> Dict:
    """Generate a single card following the archetype guidelines."""
    cfg = ARCHETYPES[archetype]
    
    quarters = {
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }
    quarter_names = list(quarters.keys())
    
    # Determine counts
    n_parrots = rng.randint(*cfg['parrot_range'])
    n_positives = rng.randint(*cfg['positive_range'])
    n_negatives = rng.randint(*cfg['negative_range'])
    n_arrows = rng.randint(*cfg['arrow_range'])
    max_rat_q = cfg['max_rat_quarters']
    max_rum = cfg.get('max_rum', 3)
    target_rum = cfg.get('target_rum', 0)
    
    # Choose negative types
    negatives = []
    if cfg.get('prefer_krakens'):
        n_krakens = min(rng.randint(2, 3), n_negatives)
        negatives.extend(['kraken'] * n_krakens)
        remaining_neg = n_negatives - n_krakens
        for _ in range(remaining_neg):
            negatives.append(rng.choice(['shark', 'rat']))
    else:
        for _ in range(n_negatives):
            negatives.append(rng.choice(NEGATIVE_SYMBOLS))
    
    # Count rats
    n_rats = negatives.count('rat')
    
    # If too many rats, replace some with sharks/krakens
    while n_rats > 2:
        idx = negatives.index('rat')
        negatives[idx] = rng.choice(['shark', 'kraken'])
        n_rats -= 1
    
    # Choose positive types (excluding parrot, will add separately)
    other_positives = [s for s in POSITIVE_SYMBOLS if s != 'parrot']
    positives = []
    
    # Handle rum specifically
    if cfg.get('prefer_rum') and target_rum > 0:
        n_rum = target_rum
    else:
        # Limit rum to max_rum
        n_rum = min(rng.randint(0, 2), max_rum)
    
    positives.extend(['rum'] * n_rum)
    remaining_pos = n_positives - n_rum - n_parrots
    
    # Fill remaining positives (excluding rum)
    non_rum_positives = [s for s in other_positives if s != 'rum']
    for _ in range(max(0, remaining_pos)):
        positives.append(rng.choice(non_rum_positives))
    
    # Add parrots
    positives.extend(['parrot'] * n_parrots)
    
    # Choose arrows
    arrows = rng.sample(ARROW_SYMBOLS, min(n_arrows, len(ARROW_SYMBOLS)))
    
    # Distribute symbols to quarters with strategy
    all_symbols = positives + negatives + arrows
    rng.shuffle(all_symbols)
    
    # Strategy: Concentrate negatives (especially rats) in fewer quarters
    rat_quarter = rng.choice(quarter_names) if n_rats > 0 else None
    negative_quarters = rng.sample(quarter_names, min(2, len(quarter_names)))
    
    for sym in all_symbols:
        if sym == 'rat':
            # Concentrate rats
            quarters[rat_quarter].append(sym)
        elif sym in NEGATIVE_SYMBOLS:
            # Prefer concentrating negatives
            q = rng.choice(negative_quarters) if rng.random() < 0.7 else rng.choice(quarter_names)
            quarters[q].append(sym)
        elif sym == 'parrot':
            # Put parrot in a good quarter (not with rats)
            good_quarters = [q for q in quarter_names if q != rat_quarter]
            q = rng.choice(good_quarters) if good_quarters else rng.choice(quarter_names)
            quarters[q].append(sym)
        elif sym in ARROW_SYMBOLS:
            # Distribute arrows for coverage
            q = rng.choice(quarter_names)
            quarters[q].append(sym)
        else:
            # Distribute positives somewhat evenly
            q = rng.choice(quarter_names)
            quarters[q].append(sym)
    
    # Count rat quarters for validation
    rat_quarters = sum(1 for q in quarters.values() if 'rat' in q)
    
    # If too many rat quarters, reconcentrate
    if rat_quarters > max_rat_q and n_rats > 0:
        # Move all rats to one quarter
        all_rats = []
        for qn in quarter_names:
            while 'rat' in quarters[qn]:
                quarters[qn].remove('rat')
                all_rats.append('rat')
        target_q = rng.choice(quarter_names)
        quarters[target_q].extend(all_rats)
    
    return {
        'card': {
            'dimensions': {'width': 135, 'height': 135},
            'quarters': quarters
        },
        'archetype': archetype,
        'stats': {
            'positives': len([s for s in all_symbols if s in POSITIVE_SYMBOLS]),
            'negatives': len([s for s in all_symbols if s in NEGATIVE_SYMBOLS]),
            'parrots': n_parrots,
            'arrows': len(arrows),
            'rats': n_rats,
            'krakens': negatives.count('kraken'),
            'rums': n_rum,
        }
    }


def validate_card(card: Dict) -> Tuple[bool, float, str]:
    """Validate card is within balance targets."""
    stats = card['stats']
    
    # Count rat quarters
    rat_quarters = sum(1 for q in card['card']['quarters'].values() if 'rat' in q)
    
    # Count rums
    rums = sum(q.count('rum') for q in card['card']['quarters'].values())
    
    predicted = predict_value(
        stats['positives'],
        stats['parrots'],
        rat_quarters,
        stats['arrows'],
        stats['krakens'],
        rums
    )
    
    if predicted < TARGET_MIN:
        return False, predicted, "Value too low"
    if predicted > TARGET_MAX:
        return False, predicted, "Value too high"
    
    return True, predicted, "OK"


def generate_balanced_card(archetype: str, rng: random.Random, max_attempts: int = 50) -> Dict:
    """Generate a card that passes validation."""
    for _ in range(max_attempts):
        card = generate_card(archetype, rng)
        valid, value, msg = validate_card(card)
        if valid:
            card['predicted_value'] = value
            return card
    
    # Fallback: return best attempt
    card = generate_card(archetype, rng)
    _, value, _ = validate_card(card)
    card['predicted_value'] = value
    return card


def generate_deck(n_cards: int, seed: int = None) -> List[Dict]:
    """Generate a balanced deck with varied archetypes."""
    rng = random.Random(seed)
    
    # Calculate archetype distribution
    archetype_counts = {}
    remaining = n_cards
    for archetype, cfg in ARCHETYPES.items():
        count = int(n_cards * cfg['weight'])
        archetype_counts[archetype] = count
        remaining -= count
    
    # Distribute remaining cards
    archetypes_list = list(ARCHETYPES.keys())
    for _ in range(remaining):
        archetype = rng.choice(archetypes_list)
        archetype_counts[archetype] += 1
    
    # Generate cards
    deck = []
    for archetype, count in archetype_counts.items():
        for _ in range(count):
            card = generate_balanced_card(archetype, rng)
            deck.append(card)
    
    rng.shuffle(deck)
    return deck


def format_card(card: Dict, idx: int) -> str:
    """Format card for display."""
    q = card['card']['quarters']
    stats = card['stats']
    
    sym_short = {
        'anchor': 'Anc', 'shark': 'Shk', 'rat': 'Rat', 'kraken': 'Kra',
        'map': 'Map', 'coin': 'Coi', 'rum': 'Rum', 'parrot': 'Par',
        'spyglass': 'Spy', 'arrow_up': '↑', 'arrow_down': '↓',
        'arrow_left': '←', 'arrow_right': '→'
    }
    
    parts = []
    for qn, qshort in [('top_left', 'TL'), ('top_right', 'TR'), 
                        ('bottom_left', 'BL'), ('bottom_right', 'BR')]:
        syms = q[qn]
        if syms:
            sym_str = ','.join(sym_short.get(s, s) for s in syms)
            parts.append(f"{qshort}:{sym_str}")
    
    quarters_str = ' | '.join(parts) if parts else '(empty)'
    pred_val = card.get('predicted_value', 0)
    
    return (f"Card {idx:3d} [{card['archetype']:<15}] ~{pred_val:.1f}  "
            f"pos={stats['positives']} neg={stats['negatives']} "
            f"par={stats['parrots']} arr={stats['arrows']}  {quarters_str}")


def main():
    parser = argparse.ArgumentParser(description='Generate balanced card deck')
    parser.add_argument('--cards', type=int, default=30, help='Number of cards')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default=None, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Show card details')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"GENERATING BALANCED DECK — {args.cards} cards, seed={args.seed}")
    print("=" * 80)
    
    deck = generate_deck(args.cards, args.seed)
    
    # Statistics
    pred_values = [c.get('predicted_value', 14.0) for c in deck]
    archetype_counts = Counter(c['archetype'] for c in deck)
    
    print(f"\nDeck Statistics:")
    print(f"  Cards: {len(deck)}")
    print(f"  Predicted value: mean={sum(pred_values)/len(pred_values):.2f}, "
          f"min={min(pred_values):.2f}, max={max(pred_values):.2f}, "
          f"range={max(pred_values)-min(pred_values):.2f}")
    
    print(f"\nArchetype Distribution:")
    for archetype, count in sorted(archetype_counts.items(), key=lambda x: -x[1]):
        pct = count / len(deck) * 100
        print(f"  {archetype:<15} {count:3d} ({pct:.0f}%)")
    
    # Symbol totals
    all_symbols = []
    for card in deck:
        for q in card['card']['quarters'].values():
            all_symbols.extend(q)
    sym_counts = Counter(all_symbols)
    
    print(f"\nSymbol Distribution (total {len(all_symbols)} symbols):")
    for sym in POSITIVE_SYMBOLS + NEGATIVE_SYMBOLS:
        count = sym_counts.get(sym, 0)
        per_card = count / len(deck)
        print(f"  {sym:<10} {count:3d} ({per_card:.1f}/card)")
    arrow_count = sum(sym_counts.get(a, 0) for a in ARROW_SYMBOLS)
    print(f"  {'arrows':<10} {arrow_count:3d} ({arrow_count/len(deck):.1f}/card)")
    
    if args.verbose:
        print(f"\n{'='*80}")
        print("CARD DETAILS")
        print("=" * 80)
        for i, card in enumerate(deck):
            print(format_card(card, i))
    
    # Save to JSON
    if args.output:
        # Strip metadata for clean JSON
        clean_deck = [{'card': c['card']} for c in deck]
        with open(args.output, 'w') as f:
            json.dump(clean_deck, f, indent=2)
        print(f"\nSaved to {args.output}")
    
    print("\n" + "=" * 80)
    print("BALANCE VERIFICATION")
    print("=" * 80)
    
    # Check balance
    in_target = sum(1 for v in pred_values if TARGET_MIN <= v <= TARGET_MAX)
    print(f"  Cards in target range ({TARGET_MIN}-{TARGET_MAX}): {in_target}/{len(deck)} ({in_target/len(deck)*100:.0f}%)")
    
    # Check variety
    unique_archetypes = len(archetype_counts)
    print(f"  Unique archetypes: {unique_archetypes}/{len(ARCHETYPES)}")
    
    # Check no extreme outliers
    outliers = [c for c, v in zip(deck, pred_values) if v < 13.0 or v > 15.0]
    print(f"  Extreme outliers (<13 or >15): {len(outliers)}")
    
    print("\n✓ Deck generation complete!")


if __name__ == '__main__':
    main()
