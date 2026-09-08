#!/usr/bin/env python3
"""Find counter-examples to disprove the updated findings from 30-card analysis."""
import json
from collections import Counter

from src.symbols import Symbols

# Load all 120 cards
with open('decks/pirate_20_25.json') as f:
    all_cards = json.load(f)

# Derived from the scoring tables; reproduces the original 6 positive /
# 3 negative membership. See the naming contract in src/symbols.py.
POSITIVES = {x for x in Symbols if x.value_symbol() and max(x.points) > 0}
NEGATIVES = {x for x in Symbols if x.value_symbol() and max(x.points) <= 0}
ARROWS = set(Symbols.arrows())

def analyze_card(idx):
    """Analyze a single card and return its properties."""
    card = all_cards[idx]['card']['quarters']
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    all_symbols = []
    quarters = {}
    for qn in qnames:
        syms = [Symbols.of(x) for x in card.get(qn, [])]
        quarters[qn] = syms
        all_symbols.extend(syms)
    
    counts = Counter(all_symbols)
    
    # Count by type
    pos_count = sum(counts[s] for s in POSITIVES)
    neg_count = sum(counts[s] for s in NEGATIVES)
    arrow_count = sum(counts[s] for s in ARROWS)
    rat_count = counts.get(Symbols.X, 0)
    kraken_count = counts.get(Symbols.SKULL, 0)
    shark_count = counts.get(Symbols.MOON, 0)
    parrot_count = counts.get(Symbols.DIAMOND, 0)
    
    # Count quarters with rats
    rat_quarters = sum(1 for qn in qnames if Symbols.X in quarters[qn])
    
    # Count quarters with ANY negatives
    neg_quarters = sum(1 for qn in qnames 
                       if any(s in NEGATIVES for s in quarters[qn]))
    
    # Quarter values
    def quarter_val(syms):
        val = 0
        for s in syms:
            if s in POSITIVES:
                val += 1
            elif s in NEGATIVES:
                val -= 1
        return val
    
    q_vals = [quarter_val(quarters[qn]) for qn in qnames]
    worst_quarter = min(q_vals)
    best_3_sum = sum(sorted(q_vals, reverse=True)[:3])
    hide_gain = best_3_sum - sum(q_vals)  # = -worst_quarter if worst < 0
    
    # Diversity
    pos_types = set(s for s in all_symbols if s in POSITIVES)
    
    # Net value
    net = pos_count - neg_count
    
    # Find which quarter has the most rats
    rat_per_quarter = {qn: quarters[qn].count(Symbols.X) for qn in qnames}
    max_rats_in_quarter = max(rat_per_quarter.values())
    rats_concentrated = (rat_count > 0 and max_rats_in_quarter >= rat_count * 0.67)
    
    return {
        'idx': idx,
        'pos': pos_count,
        'neg': neg_count,
        'arrows': arrow_count,
        'rats': rat_count,
        'krakens': kraken_count,
        'sharks': shark_count,
        'parrots': parrot_count,
        'rat_quarters': rat_quarters,
        'neg_quarters': neg_quarters,
        'q_vals': q_vals,
        'worst_q': worst_quarter,
        'best_3': best_3_sum,
        'hide_gain': hide_gain,
        'diversity': len(pos_types),
        'net': net,
        'rats_concentrated': rats_concentrated,
        'max_rats_in_quarter': max_rats_in_quarter,
        'counts': counts
    }

# Analyze all cards


def main():
    cards_analysis = [analyze_card(i) for i in range(len(all_cards))]

    # Cards already in the 30-card test set
    existing = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                25, 28, 36, 48, 52, 59, 71, 72, 94, 97]
    existing_set = set(existing)

    print("=" * 80)
    print("COUNTER-EXAMPLES TO TEST NEW FINDINGS")
    print("=" * 80)

    candidates = []

    # FINDING 1: Rat distribution matters more than count
    # COUNTER: Find cards with 3+ concentrated rats (should be "okay" per theory)
    # AND cards with 2 spread rats that are "bad"
    print("\n" + "=" * 80)
    print("FINDING 1: RAT DISTRIBUTION > RAT COUNT")
    print("=" * 80)

    print("\n3+ rats CONCENTRATED in 1 quarter (should be survivable per theory):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['rats'] >= 3 and c['max_rats_in_quarter'] >= 3:
            print(f"  Card {c['idx']:3d}: {c['rats']} rats, max {c['max_rats_in_quarter']} in one quarter, "
                  f"pos={c['pos']}, net={c['net']}, parrots={c['parrots']}")
            candidates.append((c['idx'], 'concentrated_rats'))

    print("\n2 rats SPREAD (2 quarters) + decent positives (should be bad per theory):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['rats'] == 2 and c['rat_quarters'] == 2 and c['pos'] >= 5:
            print(f"  Card {c['idx']:3d}: 2 rats in 2 quarters, pos={c['pos']}, net={c['net']}, parrots={c['parrots']}")
            candidates.append((c['idx'], 'spread_rats'))

    # FINDING 2: Krakens are okay
    # COUNTER: Find cards with 4+ krakens
    print("\n" + "=" * 80)
    print("FINDING 2: KRAKENS HAVE CAPPED PENALTY")
    print("=" * 80)

    print("\n4+ krakens (extreme test of kraken ceiling):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['krakens'] >= 4:
            print(f"  Card {c['idx']:3d}: {c['krakens']} krakens, pos={c['pos']}, net={c['net']}")
            candidates.append((c['idx'], 'many_krakens'))

    print("\n2-3 krakens + good positives (≥6) + no/few rats (should be GOOD per theory):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if 2 <= c['krakens'] <= 3 and c['pos'] >= 6 and c['rats'] <= 1:
            print(f"  Card {c['idx']:3d}: {c['krakens']} krakens, pos={c['pos']}, rats={c['rats']}, net={c['net']}")
            candidates.append((c['idx'], 'good_kraken'))

    # FINDING 3: Parrots are premium
    # COUNTER: Find cards with parrots that should fail (spread negatives)
    print("\n" + "=" * 80)
    print("FINDING 3: PARROT PREMIUM")
    print("=" * 80)

    print("\nParrot + spread negatives (≥3 neg quarters) - parrot vs spread test:")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['parrots'] >= 1 and c['neg_quarters'] >= 3:
            print(f"  Card {c['idx']:3d}: {c['parrots']} parrots, neg_quarters={c['neg_quarters']}, "
                  f"rats={c['rats']}, net={c['net']}")
            candidates.append((c['idx'], 'parrot_spread_neg'))

    print("\nNO parrot but excellent stats (≥7 pos, ≤2 neg, concentrated neg):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['parrots'] == 0 and c['pos'] >= 7 and c['neg'] <= 2 and c['neg_quarters'] <= 1:
            print(f"  Card {c['idx']:3d}: 0 parrots, pos={c['pos']}, neg={c['neg']}, net={c['net']}")
            candidates.append((c['idx'], 'no_parrot_good'))

    # FINDING 4: Arrows are neutral
    # COUNTER: Many arrows + clean base (no rats, few negatives)
    print("\n" + "=" * 80)
    print("FINDING 4: ARROWS ARE NEUTRAL")
    print("=" * 80)

    print("\n3+ arrows + clean base (≤1 rat, ≤2 total neg):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['arrows'] >= 3 and c['rats'] <= 1 and c['neg'] <= 2:
            print(f"  Card {c['idx']:3d}: {c['arrows']} arrows, rats={c['rats']}, neg={c['neg']}, pos={c['pos']}")
            candidates.append((c['idx'], 'clean_arrows'))

    print("\n2+ arrows + parrots (arrow-parrot synergy test):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['arrows'] >= 2 and c['parrots'] >= 1 and c['rats'] <= 1:
            print(f"  Card {c['idx']:3d}: {c['arrows']} arrows, {c['parrots']} parrots, rats={c['rats']}, net={c['net']}")
            candidates.append((c['idx'], 'arrow_parrot'))

    # FINDING 5: Negative concentration
    # COUNTER: ALL negatives in 1 quarter but still bad
    print("\n" + "=" * 80)
    print("FINDING 5: NEGATIVE CONCENTRATION ENABLES HIDING")
    print("=" * 80)

    print("\nAll negatives in 1 quarter (neg_quarters=1) but bad positives (≤4 pos):")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['neg'] >= 3 and c['neg_quarters'] == 1 and c['pos'] <= 4:
            print(f"  Card {c['idx']:3d}: neg_q=1, neg={c['neg']}, pos={c['pos']}, net={c['net']}")
            candidates.append((c['idx'], 'concentrated_weak'))

    print("\nSpread negatives (≥3 quarters) but many positives (≥6):")  
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['neg_quarters'] >= 3 and c['pos'] >= 6:
            print(f"  Card {c['idx']:3d}: neg_q={c['neg_quarters']}, neg={c['neg']}, pos={c['pos']}, net={c['net']}")
            candidates.append((c['idx'], 'spread_strong'))

    # FINDING 6: Raw value not predictive
    # COUNTER: Very high raw value cards, very low raw value cards
    print("\n" + "=" * 80)
    print("FINDING 6: RAW VALUE NOT PREDICTIVE")  
    print("=" * 80)

    print("\nHighest net value (≥6) not already tested:")
    for c in sorted(cards_analysis, key=lambda x: -x['net']):
        if c['idx'] in existing_set:
            continue
        if c['net'] >= 6:
            print(f"  Card {c['idx']:3d}: net={c['net']}, pos={c['pos']}, neg={c['neg']}, parrots={c['parrots']}")
            candidates.append((c['idx'], 'high_net'))

    print("\nLowest net value (≤-2) with some positives:")
    for c in sorted(cards_analysis, key=lambda x: x['net']):
        if c['idx'] in existing_set:
            continue
        if c['net'] <= -2 and c['pos'] >= 3:
            print(f"  Card {c['idx']:3d}: net={c['net']}, pos={c['pos']}, neg={c['neg']}")
            candidates.append((c['idx'], 'low_net'))

    # SPECIAL: Zero negatives (baseline)
    print("\n" + "=" * 80)
    print("SPECIAL: ZERO NEGATIVES (baseline test)")
    print("=" * 80)

    print("\nCards with 0 negatives:")
    for c in cards_analysis:
        if c['idx'] in existing_set:
            continue
        if c['neg'] == 0:
            print(f"  Card {c['idx']:3d}: 0 neg, pos={c['pos']}, arrows={c['arrows']}, parrots={c['parrots']}")
            candidates.append((c['idx'], 'zero_neg'))

    # Select best candidates
    print("\n" + "=" * 80)
    print("RECOMMENDED NEW CARDS TO ADD")
    print("=" * 80)

    # Deduplicate and select
    from collections import defaultdict
    by_reason = defaultdict(list)
    for idx, reason in candidates:
        if idx not in existing_set:
            by_reason[reason].append(idx)

    # Select diverse set
    selected = []
    reason_priority = [
        ('concentrated_rats', 2),      # Test rat concentration theory
        ('spread_rats', 1),            # Test spread rats are bad
        ('many_krakens', 2),           # Test kraken ceiling
        ('good_kraken', 1),            # Krakens + good positives
        ('parrot_spread_neg', 2),      # Parrot vs spread test
        ('no_parrot_good', 1),         # No parrot but good
        ('clean_arrows', 1),           # Pure arrow test
        ('arrow_parrot', 1),           # Arrow-parrot synergy
        ('concentrated_weak', 1),      # Concentrated but weak positives
        ('spread_strong', 1),          # Spread but strong positives
        ('high_net', 1),               # High raw value
        ('low_net', 1),                # Low raw value
        ('zero_neg', 1),               # Zero negatives baseline
    ]

    for reason, count in reason_priority:
        available = [i for i in by_reason[reason] if i not in selected]
        selected.extend(available[:count])

    # Limit to 10 new cards (keeping 20 original)
    selected = selected[:10]

    print(f"\nSelected {len(selected)} new cards to replace last 10 counter-examples:")
    for idx in selected:
        c = cards_analysis[idx]
        reason_str = ', '.join(set(r for i, r in candidates if i == idx))
        print(f"  Card {idx:3d}: pos={c['pos']}, neg={c['neg']}, rats={c['rats']}, "
              f"krakens={c['krakens']}, parrots={c['parrots']}, rat_q={c['rat_quarters']}, "
              f"net={c['net']} [{reason_str}]")

    # Create new 30-card set: original 20 + 10 new counter-examples  
    new_30 = list(range(20)) + selected
    print(f"\nNEW 30-CARD TEST SET: {new_30}")

    # Save to new JSON
    output_cards = [all_cards[i] for i in new_30]
    with open('decks/pirate_30_test_v2.json', 'w') as f:
        json.dump(output_cards, f, indent=2)
    print(f"\nSaved to decks/pirate_30_test_v2.json")

    # Print card details for the new cards
    print("\n" + "=" * 80)
    print("NEW CARD DETAILS")
    print("=" * 80)
    qmap = {'top_left': 'TL', 'top_right': 'TR', 'bottom_left': 'BL', 'bottom_right': 'BR'}
    sym_short = {x.name: x.abbrev for x in Symbols if x.abbrev}

    for pos, orig_idx in enumerate(new_30[20:], start=20):
        card = all_cards[orig_idx]['card']['quarters']
        parts = []
        for qn in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            syms = card.get(qn, [])
            if syms:
                sym_str = ','.join(sym_short.get(s, s) for s in syms)
                parts.append(f"{qmap[qn]}:{sym_str}")
        c = cards_analysis[orig_idx]
        print(f"Pos {pos} (orig {orig_idx}): {' | '.join(parts)}")
        print(f"   pos={c['pos']}, neg={c['neg']}, rats={c['rats']}, krakens={c['krakens']}, "
              f"parrots={c['parrots']}, net={c['net']}")


if __name__ == '__main__':
    main()
