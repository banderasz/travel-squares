#!/usr/bin/env python3
"""
Balanced Card Generator v3 — Formula-based with Simulation Validation

Process:
1. Define a value formula calibrated from simulation data
2. Generate ~500 cards targeting ±0.5 of target value
3. Simulate all cards to get actual values
4. Select ~120 cards with truly similar values AND good symbol distribution

Key insights from simulation analysis:
- RUM concentration: 3+ rums = +2-3 points (game-changing)
- RAT distribution: spread rats = -1.2 per quarter affected
- Parrot: +0.5-1.0 per parrot (less than expected)
- Kraken: penalty caps at ~-0.5 total (nearly free)
- Arrows: depends on neighbors, assume +0.5 per arrow with good card
"""
import json
import random
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.symbols import Symbols  # noqa: E402

# ============================================================
# SCORING TABLES — derived from src/symbols.py, the single source of truth
# ============================================================
SCORING_SYMBOLS = [x for x in Symbols if x.value_symbol()]

# Points by count 0..11 (symbols.py carries a 13th saturating bucket we ignore here)
SYMBOL_POINTS = {x: x.points[:12] for x in SCORING_SYMBOLS}

# Target distribution: the generation weights, with the four arrows pooled.
SYMBOL_DISTRIBUTION = {x: x.weight for x in SCORING_SYMBOLS}
SYMBOL_DISTRIBUTION["arrow"] = sum(x.weight for x in Symbols.arrows())

# Normalize to per-card values (16 slots per card)
TOTAL_WEIGHT = sum(SYMBOL_DISTRIBUTION.values())
SYMBOLS_PER_CARD = {k: v / TOTAL_WEIGHT * 16 for k, v in SYMBOL_DISTRIBUTION.items()}
NOTHING_PER_CARD = 16 - sum(SYMBOLS_PER_CARD.values())

# Note: rum is deliberately not in POSITIVE_SYMBOLS — it is a penalty at low
# counts. That was the original modelling choice and is preserved.
POSITIVE_SYMBOLS = [Symbols.CIRCLE, Symbols.STAR, Symbols.SQUARE, Symbols.SUN, Symbols.DIAMOND]
NEGATIVE_SYMBOLS = [Symbols.MOON, Symbols.X, Symbols.SKULL]
ARROW_SYMBOLS = Symbols.arrows()
RUM = Symbols.TRIANGLE

# ============================================================
# VALUE FORMULA (calibrated from simulation)
# ============================================================

def calculate_card_value(card: Dict) -> float:
    """
    Calculate expected card value using formula calibrated from simulation.
    
    Base formula: Sum of expected marginal contribution of each symbol
    adjusted for:
    - Rum concentration bonus
    - Rat quarter penalty
    - Parrot bonus
    - Kraken cap
    - Arrow expected contribution
    """
    quarters = card['card']['quarters']
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    # Count symbols
    all_symbols = []
    for qn in qnames:
        all_symbols.extend(quarters.get(qn, []))
    
    counts = Counter(all_symbols)
    
    # Base value: expected contribution in a 6-card game
    # Each symbol contributes its marginal expected value
    base_value = 13.0  # Baseline from simulation
    
    # Positive symbols (approximately +0.2 each)
    for sym in POSITIVE_SYMBOLS:
        if sym == Symbols.DIAMOND:
            base_value += counts.get(sym, 0) * 0.8  # Parrot bonus
        else:
            base_value += counts.get(sym, 0) * 0.25
    
    # Rum: complex scoring, approximated
    rum_count = counts.get(Symbols.TRIANGLE, 0)
    if rum_count == 1:
        base_value -= 0.3  # Slight penalty for 1 rum
    elif rum_count == 2:
        base_value += 0.3  # Neutral to slight positive
    elif rum_count >= 3:
        base_value += 0.7 * rum_count  # Big bonus for concentration
    
    # Negative symbols
    # Kraken: penalty caps, so treat as ~-0.2 each (diminishes)
    kraken_count = counts.get(Symbols.SKULL, 0)
    if kraken_count == 1:
        base_value -= 0.5
    elif kraken_count >= 2:
        base_value -= 0.7  # Cap the penalty
    
    # Shark: flat -0.3 each
    base_value -= counts.get(Symbols.MOON, 0) * 0.3
    
    # Rat: depends on distribution across quarters
    rat_count = counts.get(Symbols.X, 0)
    rat_quarters = sum(1 for qn in qnames if Symbols.X in quarters.get(qn, []))
    if rat_quarters > 0:
        # Penalty per quarter with rats (spread is worse)
        base_value -= rat_quarters * 0.9
    
    # Arrows: expected +0.4 each in good combos
    arrow_count = sum(counts.get(a, 0) for a in ARROW_SYMBOLS)
    base_value += arrow_count * 0.4
    
    return base_value


def calculate_symbol_value(symbol: str, context_counts: Dict[str, int] = None) -> float:
    """Calculate marginal value of a single symbol."""
    if context_counts is None:
        context_counts = {}
    
    if symbol in ['arrow_up', 'arrow_down', 'arrow_left', 'arrow_right']:
        return 0.4
    elif symbol == Symbols.DIAMOND:
        return 0.8
    elif symbol == Symbols.TRIANGLE:
        rum_count = context_counts.get(Symbols.TRIANGLE, 0)
        if rum_count >= 2:
            return 1.0  # Bonus for concentration
        elif rum_count == 1:
            return 0.3
        else:
            return -0.3  # First rum is often bad
    elif symbol == Symbols.SKULL:
        kraken_count = context_counts.get(Symbols.SKULL, 0)
        if kraken_count >= 1:
            return -0.1  # Diminishing penalty
        return -0.5
    elif symbol == Symbols.MOON:
        return -0.3
    elif symbol == Symbols.X:
        return -0.9  # Approximate (depends on quarter spread)
    elif symbol in POSITIVE_SYMBOLS:
        return 0.25
    return 0.0


# ============================================================
# CARD GENERATION
# ============================================================

def generate_card_symbols(rng: random.Random, target_value: float, 
                         tolerance: float = 0.5) -> Dict:
    """Generate a card targeting a specific value."""
    
    # Build weighted symbol list based on distribution
    symbol_weights = []
    for sym, per_card in SYMBOLS_PER_CARD.items():
        if sym == 'arrow':
            # Split arrows evenly
            for arrow in ARROW_SYMBOLS:
                symbol_weights.append((arrow, per_card / 4))
        else:
            symbol_weights.append((sym, per_card))
    
    # Add "nothing" slots
    symbol_weights.append((None, NOTHING_PER_CARD))
    
    symbols = [sw[0] for sw in symbol_weights]
    weights = [sw[1] for sw in symbol_weights]
    
    max_attempts = 100
    for _ in range(max_attempts):
        # Generate 16 symbols
        card_symbols = rng.choices(symbols, weights=weights, k=16)
        
        # Count arrows and limit to 4 max
        arrow_count = sum(1 for s in card_symbols if s in ARROW_SYMBOLS)
        if arrow_count > 4:
            # Replace excess arrows with None
            excess = arrow_count - 4
            for i in range(len(card_symbols)):
                if excess <= 0:
                    break
                if card_symbols[i] in ARROW_SYMBOLS:
                    card_symbols[i] = None
                    excess -= 1
        
        # Distribute to quarters
        quarters = {
            'top_left': [s for s in card_symbols[0:4] if s],
            'top_right': [s for s in card_symbols[4:8] if s],
            'bottom_left': [s for s in card_symbols[8:12] if s],
            'bottom_right': [s for s in card_symbols[12:16] if s],
        }
        
        card = {'card': {'dimensions': {'width': 135, 'height': 135}, 'quarters': quarters}}
        
        value = calculate_card_value(card)
        if abs(value - target_value) <= tolerance:
            card['calculated_value'] = value
            return card
    
    # Return last attempt if none match exactly
    card['calculated_value'] = value
    return card


def optimize_card_for_value(card: Dict, target_value: float, 
                           rng: random.Random, max_iterations: int = 50) -> Dict:
    """Iteratively adjust card to get closer to target value."""
    current_value = calculate_card_value(card)
    best_card = card
    best_diff = abs(current_value - target_value)
    
    quarters = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    non_arrow_symbols = [None] + list(SYMBOL_POINTS.keys())  # Don't add arrows in optimization
    
    for _ in range(max_iterations):
        if best_diff < 0.3:
            break
            
        # Make a copy
        new_card = {
            'card': {
                'dimensions': {'width': 135, 'height': 135},
                'quarters': {q: list(card['card']['quarters'][q]) for q in quarters}
            }
        }
        
        # Randomly modify one symbol
        quarter = rng.choice(quarters)
        if new_card['card']['quarters'][quarter]:
            idx = rng.randrange(len(new_card['card']['quarters'][quarter]))
            
            # Determine direction to adjust
            if current_value < target_value:
                # Need to increase value - add positive or remove negative
                good_symbols = [s for s in non_arrow_symbols if calculate_symbol_value(s) > 0]
                new_card['card']['quarters'][quarter][idx] = rng.choice(good_symbols) if good_symbols else None
            else:
                # Need to decrease value - add negative or remove positive
                bad_symbols = [s for s in non_arrow_symbols if calculate_symbol_value(s) < 0.2]
                new_card['card']['quarters'][quarter][idx] = rng.choice(bad_symbols) if bad_symbols else None
            
            # Clean up None values
            new_card['card']['quarters'][quarter] = [s for s in new_card['card']['quarters'][quarter] if s]
        
        new_value = calculate_card_value(new_card)
        new_diff = abs(new_value - target_value)
        
        if new_diff < best_diff:
            best_card = new_card
            best_diff = new_diff
            current_value = new_value
            card = new_card
    
    best_card['calculated_value'] = calculate_card_value(best_card)
    return best_card


def concentrate_rats(card: Dict, rng: random.Random) -> Dict:
    """Move all rats to a single quarter to minimize penalty."""
    quarters = card['card']['quarters']
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    # Collect all rats
    all_rats = []
    for qn in qnames:
        while Symbols.X in quarters[qn]:
            quarters[qn].remove(Symbols.X)
            all_rats.append(Symbols.X)
    
    # Put all rats in one quarter
    if all_rats:
        target_q = rng.choice(qnames)
        quarters[target_q].extend(all_rats)
    
    card['calculated_value'] = calculate_card_value(card)
    return card


# ============================================================
# DECK GENERATION AND SELECTION
# ============================================================

def generate_candidate_deck(n_cards: int, target_value: float, 
                           tolerance: float, seed: int) -> List[Dict]:
    """Generate candidate cards all targeting similar value."""
    rng = random.Random(seed)
    cards = []
    
    for i in range(n_cards):
        # Generate initial card
        card = generate_card_symbols(rng, target_value, tolerance * 2)
        
        # Optimize toward target
        card = optimize_card_for_value(card, target_value, rng)
        
        # Concentrate rats for better value
        card = concentrate_rats(card, rng)
        
        cards.append(card)
    
    return cards


def calculate_distribution_deviation(cards: List[Dict]) -> float:
    """Calculate how far the deck is from target distribution."""
    if not cards:
        return float('inf')
    
    # Count all symbols
    all_symbols = []
    for card in cards:
        for q in card['card']['quarters'].values():
            all_symbols.extend(q)
    
    counts = Counter(all_symbols)
    n_cards = len(cards)
    total_symbols = len(all_symbols)
    
    deviation = 0.0
    for sym, target_per_card in SYMBOLS_PER_CARD.items():
        if sym == 'arrow':
            actual = sum(counts.get(a, 0) for a in ARROW_SYMBOLS)
        else:
            actual = counts.get(sym, 0)
        
        expected = target_per_card * n_cards
        # Normalize deviation
        if expected > 0:
            deviation += ((actual - expected) / expected) ** 2
    
    return deviation


def select_balanced_cards(candidates: List[Dict], n_select: int, 
                         target_value: float, value_tolerance: float) -> List[Dict]:
    """
    Greedily select cards that:
    1. Have values close to target
    2. Together achieve good symbol distribution
    """
    # Filter by value tolerance first
    valid_candidates = [c for c in candidates 
                       if abs(c.get('calculated_value', 0) - target_value) <= value_tolerance]
    
    if len(valid_candidates) < n_select:
        print(f"Warning: Only {len(valid_candidates)} cards within value tolerance")
        valid_candidates = sorted(candidates, 
                                 key=lambda c: abs(c.get('calculated_value', 0) - target_value))[:n_select * 2]
    
    selected = []
    remaining = list(valid_candidates)
    
    for i in range(n_select):
        if not remaining:
            break
            
        best_idx = 0
        best_deviation = float('inf')
        
        for idx, card in enumerate(remaining):
            test_selection = selected + [card]
            deviation = calculate_distribution_deviation(test_selection)
            
            # Also penalize cards far from target value
            value_penalty = abs(card.get('calculated_value', 0) - target_value) * 0.1
            total_score = deviation + value_penalty
            
            if total_score < best_deviation:
                best_deviation = total_score
                best_idx = idx
        
        selected.append(remaining.pop(best_idx))
    
    return selected


# ============================================================
# MAIN
# ============================================================

def format_card(card: Dict, idx: int) -> str:
    """Format card for display."""
    q = card['card']['quarters']
    
    sym_short = {symbol: symbol.abbrev for symbol in Symbols if symbol.abbrev}

    parts = []
    for qn, qshort in [('top_left', 'TL'), ('top_right', 'TR'), 
                        ('bottom_left', 'BL'), ('bottom_right', 'BR')]:
        syms = q.get(qn, [])
        if syms:
            sym_str = ','.join(sym_short.get(s, s) for s in syms)
            parts.append(f"{qshort}:{sym_str}")
    
    quarters_str = ' | '.join(parts) if parts else '(empty)'
    calc_val = card.get('calculated_value', 0)
    
    return f"Card {idx:3d} (val={calc_val:.1f})  {quarters_str}"


def main():
    parser = argparse.ArgumentParser(description='Generate balanced card deck v3')
    parser.add_argument('--candidates', type=int, default=500, 
                       help='Number of candidate cards to generate')
    parser.add_argument('--select', type=int, default=120, 
                       help='Number of cards to select for final deck')
    parser.add_argument('--target-value', type=float, default=17.0,
                       help='Target card value')
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='Value tolerance for candidate generation')
    parser.add_argument('--select-tolerance', type=float, default=0.8,
                       help='Value tolerance for final selection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default=None, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Show card details')
    args = parser.parse_args()
    
    print("=" * 80)
    print("BALANCED CARD GENERATOR v3 — Formula-based with Distribution Matching")
    print("=" * 80)
    
    # Generate candidates
    print(f"\nGenerating {args.candidates} candidate cards (target value: {args.target_value})...")
    candidates = generate_candidate_deck(
        args.candidates, args.target_value, args.tolerance, args.seed
    )
    
    # Analyze candidates
    cand_values = [c.get('calculated_value', 0) for c in candidates]
    print(f"Candidate values: mean={np.mean(cand_values):.2f}, "
          f"std={np.std(cand_values):.2f}, "
          f"min={np.min(cand_values):.2f}, max={np.max(cand_values):.2f}")
    
    in_tolerance = sum(1 for v in cand_values 
                      if abs(v - args.target_value) <= args.select_tolerance)
    print(f"Candidates within ±{args.select_tolerance} of target: {in_tolerance}/{len(candidates)}")
    
    # Select balanced deck
    print(f"\nSelecting {args.select} cards with balanced distribution...")
    selected = select_balanced_cards(
        candidates, args.select, args.target_value, args.select_tolerance
    )
    
    # Final statistics
    sel_values = [c.get('calculated_value', 0) for c in selected]
    print(f"\n{'='*80}")
    print("FINAL DECK STATISTICS")
    print("=" * 80)
    print(f"Cards selected: {len(selected)}")
    print(f"Value: mean={np.mean(sel_values):.2f}, "
          f"std={np.std(sel_values):.2f}, "
          f"min={np.min(sel_values):.2f}, max={np.max(sel_values):.2f}, "
          f"range={np.max(sel_values)-np.min(sel_values):.2f}")
    
    # Symbol distribution
    all_symbols = []
    for card in selected:
        for q in card['card']['quarters'].values():
            all_symbols.extend(q)
    
    counts = Counter(all_symbols)
    print(f"\nSymbol Distribution (total {len(all_symbols)} symbols across {len(selected)} cards):")
    print(f"{'Symbol':<12} {'Actual':>8} {'Target':>8} {'Per Card':>10} {'Target/Card':>12}")
    print("-" * 60)
    
    for sym, target in SYMBOLS_PER_CARD.items():
        if sym == 'arrow':
            actual = sum(counts.get(a, 0) for a in ARROW_SYMBOLS)
        else:
            actual = counts.get(sym, 0)
        expected = target * len(selected)
        per_card = actual / len(selected)
        print(f"{sym:<12} {actual:>8} {expected:>8.0f} {per_card:>10.2f} {target:>12.2f}")
    
    deviation = calculate_distribution_deviation(selected)
    print(f"\nDistribution deviation score: {deviation:.4f}")
    
    if args.verbose:
        print(f"\n{'='*80}")
        print("CARD DETAILS")
        print("=" * 80)
        for i, card in enumerate(sorted(selected, key=lambda c: -c.get('calculated_value', 0))):
            print(format_card(card, i))
    
    # Save to JSON
    if args.output:
        # Serialise symbols by stable ID (see src/deck_io.py)
        from src.deck_io import dumps_deck
        clean_deck = [
            {'card': {
                'dimensions': c['card'].get('dimensions', {'width': 135, 'height': 135}),
                'quarters': {qn: [s.name for s in syms]
                             for qn, syms in c['card']['quarters'].items()},
            }}
            for c in selected
        ]
        with open(args.output, 'w') as f:
            f.write(dumps_deck(clean_deck))
        print(f"\nSaved to {args.output}")
    
    print("\n✓ Deck generation complete!")
    print("\nNext step: Run simulation to verify actual values")
    print(f"  python -m src.simulation.sim --json {args.output or 'deck.json'} "
          f"--out data/placements_verify --placements 100000000 --sharded --n-shards 256 "
          f"--n-cards {len(selected)}")


if __name__ == '__main__':
    main()
