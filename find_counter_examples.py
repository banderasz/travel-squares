#!/usr/bin/env python3
"""Find cards in the 120-card pool that could disprove the findings about good cards.

Analyzes all 120 cards to find counter-examples for each hypothesis.
"""
import json

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.simulation.sim import SCORE_DEFAULT  # noqa: E402  (keyed by stable symbol ID)
from src.symbols import Symbols  # noqa: E402

# Derived from the scoring tables so a rename or rebalance cannot leave these
# stale. Reproduces the original membership: 6 positive, 3 negative.
POSITIVE = {x for x in Symbols if x.value_symbol() and max(x.points) > 0}
NEGATIVE = {x for x in Symbols if x.value_symbol() and max(x.points) <= 0}
ARROWS = set(Symbols.arrows())

def analyze_card(card_data, card_id):
    """Extract properties from a card."""
    quarters = card_data['card']['quarters']
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    total_pos = 0
    total_neg = 0
    total_arrows = 0
    sym_counts = {}
    quarter_neg = []
    quarter_pos = []
    quarter_vals = []
    
    for qn in qnames:
        syms = [Symbols.of(x) for x in quarters[qn]]
        qp = sum(1 for s in syms if s in POSITIVE)
        qn_neg = sum(1 for s in syms if s in NEGATIVE)
        qa = sum(1 for s in syms if s in ARROWS)
        
        total_pos += qp
        total_neg += qn_neg
        total_arrows += qa
        quarter_pos.append(qp)
        quarter_neg.append(qn_neg)
        quarter_vals.append(qp - qn_neg)
        
        for s in syms:
            if s not in ARROWS:
                sym_counts[s] = sym_counts.get(s, 0) + 1
    
    pos_types = set(s for s in sym_counts if s in POSITIVE)
    
    return {
        'id': card_id,
        'n_pos': total_pos,
        'n_neg': total_neg,
        'n_arrows': total_arrows,
        'net': total_pos - total_neg,
        'sym_counts': sym_counts,
        'pos_diversity': len(pos_types),
        'quarter_vals': quarter_vals,
        'quarter_neg': quarter_neg,
        'best_3_sum': sum(sorted(quarter_vals, reverse=True)[:3]),
        'worst_quarter': min(quarter_vals),
        'hide_gain': -min(quarter_vals),
        'n_parrots': sym_counts.get(Symbols.DIAMOND, 0),
        'n_rats': sym_counts.get(Symbols.X, 0),
        'n_krakens': sym_counts.get(Symbols.SKULL, 0),
        'n_sharks': sym_counts.get(Symbols.MOON, 0),
        'neg_concentrated': max(quarter_neg) == total_neg if total_neg > 0 else True,
    }


def main():
    with open('pirate_cards/pirate_20_25.json') as f:
        cards_raw = json.load(f)
    
    cards = [analyze_card(c, i) for i, c in enumerate(cards_raw)]
    
    print("=" * 70)
    print("  FINDING COUNTER-EXAMPLES TO GOOD CARD HYPOTHESES")
    print("=" * 70)
    
    # ================================================================
    # Finding 1: Positive count matters most
    # Counter: Cards with few positives but should be good (high diversity, parrots)
    # Counter: Cards with many positives but should be bad (low diversity, rats)
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 1: Positive count matters most")
    print("  Counter-examples: few positives but high quality, or many but low quality")
    print("=" * 70)
    
    # Few positives (<=4) but high diversity and/or parrots
    few_pos_good = [c for c in cards if c['n_pos'] <= 4 and 
                   (c['pos_diversity'] >= 4 or c['n_parrots'] >= 1) and c['n_neg'] <= 2]
    print(f"\n  Few positives (≤4) but high diversity or parrots, low negatives:")
    for c in sorted(few_pos_good, key=lambda x: -x['pos_diversity'])[:10]:
        print(f"    Card {c['id']:>3}: {c['n_pos']} pos, {c['n_neg']} neg, "
              f"diversity={c['pos_diversity']}, parrots={c['n_parrots']}, "
              f"syms={dict(c['sym_counts'])}")
    
    # Many positives (>=7) but concentrated in one type (low diversity)
    many_pos_bad = [c for c in cards if c['n_pos'] >= 6 and c['pos_diversity'] <= 3]
    print(f"\n  Many positives (≥6) but low diversity (≤3 types):")
    for c in sorted(many_pos_bad, key=lambda x: -x['n_pos'])[:10]:
        print(f"    Card {c['id']:>3}: {c['n_pos']} pos, diversity={c['pos_diversity']}, "
              f"syms={dict(c['sym_counts'])}")
    
    # ================================================================
    # Finding 2: Diversity pays
    # Counter: Low diversity cards that might still be good
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 2: Symbol diversity pays")
    print("  Counter-examples: low diversity but many positives and few negatives")
    print("=" * 70)
    
    low_div_good = [c for c in cards if c['pos_diversity'] <= 2 and 
                   c['n_pos'] >= 5 and c['n_neg'] <= 1]
    print(f"\n  Low diversity (≤2) but many positives and few negatives:")
    for c in sorted(low_div_good, key=lambda x: -c['n_pos'])[:10]:
        print(f"    Card {c['id']:>3}: {c['n_pos']} pos, {c['n_neg']} neg, "
              f"diversity={c['pos_diversity']}, syms={dict(c['sym_counts'])}")
    
    # ================================================================
    # Finding 3: Parrots are best
    # Counter: Cards with multiple parrots but still bad (many negatives)
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 3: Parrots are the best symbol")
    print("  Counter-examples: multiple parrots but many negatives")
    print("=" * 70)
    
    parrot_but_bad = [c for c in cards if c['n_parrots'] >= 2 and c['n_neg'] >= 4]
    print(f"\n  Multiple parrots (≥2) but many negatives (≥4):")
    for c in sorted(parrot_but_bad, key=lambda x: -x['n_parrots'])[:10]:
        print(f"    Card {c['id']:>3}: parrots={c['n_parrots']}, "
              f"neg={c['n_neg']} (rats={c['n_rats']}, sharks={c['n_sharks']}, krakens={c['n_krakens']}), "
              f"net={c['net']:+d}")
    
    # ================================================================
    # Finding 4: Rats are worst
    # Counter: Many rats but still net positive (many good symbols)
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 4: Rats are the worst symbol")
    print("  Counter-examples: many rats but net positive or compensated")
    print("=" * 70)
    
    rats_but_ok = [c for c in cards if c['n_rats'] >= 3 and c['n_pos'] >= 5]
    print(f"\n  Many rats (≥3) but many positives (≥5):")
    for c in sorted(rats_but_ok, key=lambda x: -x['n_rats'])[:10]:
        print(f"    Card {c['id']:>3}: rats={c['n_rats']}, pos={c['n_pos']}, "
              f"net={c['net']:+d}, syms={dict(c['sym_counts'])}")
    
    # ================================================================
    # Finding 5: Krakens are ok (capped penalty)
    # Counter: Multiple krakens making card bad
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 5: Krakens have capped penalty")
    print("  Counter-examples: high kraken count + other negatives")
    print("=" * 70)
    
    many_krakens = [c for c in cards if c['n_krakens'] >= 3]
    print(f"\n  Many krakens (≥3):")
    for c in sorted(many_krakens, key=lambda x: -x['n_krakens'])[:10]:
        print(f"    Card {c['id']:>3}: krakens={c['n_krakens']}, "
              f"other_neg={c['n_neg']-c['n_krakens']}, pos={c['n_pos']}, net={c['net']:+d}")
    
    # ================================================================
    # Finding 6: Hide gain is negatively correlated
    # Counter: High hide gain that actually works (concentrated negatives + strong positives)
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 6: Hide gain is negatively correlated (needing to hide = bad)")
    print("  Counter-examples: high hide gain but overall strong card")
    print("=" * 70)
    
    high_hide_good = [c for c in cards if c['hide_gain'] >= 3 and c['best_3_sum'] >= 5]
    print(f"\n  High hide gain (≥3) AND strong best-3 quarters (≥5):")
    for c in sorted(high_hide_good, key=lambda x: -x['best_3_sum'])[:10]:
        print(f"    Card {c['id']:>3}: hide_gain={c['hide_gain']}, best_3={c['best_3_sum']}, "
              f"quarters={c['quarter_vals']}, net={c['net']:+d}")
    
    # ================================================================
    # Finding 7: Arrows are uncorrelated
    # Counter: Cards with many arrows AND good base symbols
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINDING 7: Arrows are nearly uncorrelated with value")
    print("  Counter-examples: many arrows + strong base symbols")
    print("=" * 70)
    
    many_arrows_good = [c for c in cards if c['n_arrows'] >= 3 and c['n_pos'] >= 5 and c['n_neg'] <= 2]
    print(f"\n  Many arrows (≥3) AND many positives, few negatives:")
    for c in sorted(many_arrows_good, key=lambda x: -x['n_arrows'])[:10]:
        print(f"    Card {c['id']:>3}: arrows={c['n_arrows']}, pos={c['n_pos']}, neg={c['n_neg']}, "
              f"net={c['net']:+d}, syms={dict(c['sym_counts'])}")
    
    many_arrows_bad = [c for c in cards if c['n_arrows'] >= 3 and c['n_neg'] >= 4]
    print(f"\n  Many arrows (≥3) AND many negatives (≥4):")
    for c in sorted(many_arrows_bad, key=lambda x: -c['n_neg'])[:10]:
        print(f"    Card {c['id']:>3}: arrows={c['n_arrows']}, pos={c['n_pos']}, neg={c['n_neg']}, "
              f"net={c['net']:+d}")
    
    # ================================================================
    # Summary: Best candidates to add to test set
    # ================================================================
    print("\n" + "=" * 70)
    print("  RECOMMENDED CARDS TO ADD FOR TESTING (cards 20-119)")
    print("=" * 70)
    
    # Cards that could challenge findings
    candidates = set()
    
    # Cards with unusual arrow counts
    for c in cards[20:]:
        if c['n_arrows'] >= 3:
            candidates.add(c['id'])
    
    # Cards with many parrots
    for c in cards[20:]:
        if c['n_parrots'] >= 2:
            candidates.add(c['id'])
    
    # Cards with concentrated negatives but good positives
    for c in cards[20:]:
        if c['hide_gain'] >= 3 and c['best_3_sum'] >= 4:
            candidates.add(c['id'])
    
    # Cards with many krakens
    for c in cards[20:]:
        if c['n_krakens'] >= 2:
            candidates.add(c['id'])
    
    # Cards with many rats but also positives
    for c in cards[20:]:
        if c['n_rats'] >= 3 and c['n_pos'] >= 4:
            candidates.add(c['id'])
    
    # Low-diversity high-positive cards
    for c in cards[20:]:
        if c['pos_diversity'] <= 2 and c['n_pos'] >= 5:
            candidates.add(c['id'])
    
    # High-diversity low-positive cards
    for c in cards[20:]:
        if c['pos_diversity'] >= 5 and c['n_pos'] <= 5:
            candidates.add(c['id'])
    
    print(f"\n  Candidate card IDs from 20-119: {sorted(candidates)}")
    print(f"  Total candidates: {len(candidates)}")
    
    # Pick 10 most interesting
    interesting = []
    for cid in sorted(candidates):
        c = cards[cid]
        score = 0
        if c['n_arrows'] >= 3: score += 2
        if c['n_parrots'] >= 2: score += 2
        if c['hide_gain'] >= 3: score += 1
        if c['n_krakens'] >= 2: score += 1
        if c['n_rats'] >= 3 and c['n_pos'] >= 4: score += 2
        interesting.append((cid, score, c))
    
    interesting.sort(key=lambda x: -x[1])
    
    print(f"\n  TOP 10 MOST INTERESTING CARDS TO ADD:")
    print(f"  {'ID':>4} {'Pos':>4} {'Neg':>4} {'Arr':>4} {'Par':>4} {'Rat':>4} {'Kra':>4} {'Div':>4} {'Hide':>5} {'Why'}")
    print(f"  {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*5} {'-'*30}")
    for cid, score, c in interesting[:10]:
        why = []
        if c['n_arrows'] >= 3: why.append(f"arrows={c['n_arrows']}")
        if c['n_parrots'] >= 2: why.append(f"parrots={c['n_parrots']}")
        if c['n_rats'] >= 3: why.append(f"rats={c['n_rats']}")
        if c['n_krakens'] >= 2: why.append(f"krakens={c['n_krakens']}")
        if c['hide_gain'] >= 3: why.append(f"hide_gain={c['hide_gain']}")
        print(f"  {cid:>4} {c['n_pos']:>4} {c['n_neg']:>4} {c['n_arrows']:>4} "
              f"{c['n_parrots']:>4} {c['n_rats']:>4} {c['n_krakens']:>4} "
              f"{c['pos_diversity']:>4} {c['hide_gain']:>5}  {', '.join(why)}")


if __name__ == '__main__':
    main()
