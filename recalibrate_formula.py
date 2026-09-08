#!/usr/bin/env python3
"""
Recalibrate the card value formula using actual simulation data.

Process:
1. Load simulation results from data/placements_verify
2. Extract card features from pirate_cards/pirate_120_balanced.json
3. Fit a linear regression to find the best coefficients
4. Output the calibrated formula
"""
import json
import numpy as np
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.sim import ShardedPlacementStore, PlacementStore

# Symbol categories
POSITIVE_SYMBOLS = ['anchor', 'spyglass', 'map', 'coin', 'parrot']
NEGATIVE_SYMBOLS = ['shark', 'rat', 'kraken']
ARROW_SYMBOLS = ['arrow_up', 'arrow_down', 'arrow_left', 'arrow_right']

def extract_features(card):
    """Extract numerical features from a card."""
    quarters = card['card']['quarters']
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    all_symbols = []
    for qn in qnames:
        all_symbols.extend(quarters.get(qn, []))
    
    counts = Counter(all_symbols)
    
    # Basic counts
    n_positives = sum(counts.get(s, 0) for s in POSITIVE_SYMBOLS)
    n_negatives = sum(counts.get(s, 0) for s in NEGATIVE_SYMBOLS)
    n_arrows = sum(counts.get(s, 0) for s in ARROW_SYMBOLS)
    n_parrots = counts.get('parrot', 0)
    n_rums = counts.get('rum', 0)
    n_krakens = counts.get('kraken', 0)
    n_rats = counts.get('rat', 0)
    n_sharks = counts.get('shark', 0)
    n_anchors = counts.get('anchor', 0)
    n_maps = counts.get('map', 0)
    n_coins = counts.get('coin', 0)
    n_spyglasses = counts.get('spyglass', 0)
    
    # Rat quarters (spread penalty)
    rat_quarters = sum(1 for qn in qnames if 'rat' in quarters.get(qn, []))
    
    # Rum concentration bonus
    rum_bonus = max(0, n_rums - 2)  # Bonus kicks in at 3+ rums
    
    # Total symbols
    total_symbols = len(all_symbols)
    
    return {
        'total_symbols': total_symbols,
        'n_positives': n_positives,
        'n_negatives': n_negatives,
        'n_arrows': n_arrows,
        'n_parrots': n_parrots,
        'n_rums': n_rums,
        'n_krakens': n_krakens,
        'n_rats': n_rats,
        'n_sharks': n_sharks,
        'n_anchors': n_anchors,
        'n_maps': n_maps,
        'n_coins': n_coins,
        'n_spyglasses': n_spyglasses,
        'rat_quarters': rat_quarters,
        'rum_bonus': rum_bonus,
    }


def load_simulation_results(store_path, n_shards, n_cards):
    """Load simulation results and compute per-card averages."""
    store = ShardedPlacementStore(store_path, n_shards=n_shards)
    
    # Accumulate per-card statistics
    card_score_sum = np.zeros(n_cards, dtype=np.float64)
    card_count = np.zeros(n_cards, dtype=np.int64)
    
    print(f"Loading simulation data from {store_path}...")
    for shard_id in range(n_shards):
        data = store.load_shard(shard_id)
        if len(data) == 0:
            continue
        
        # Extract card IDs and scores
        for record in data:
            cards = PlacementStore.unpack_cards_scalar(record['card_ids'])
            score = record['score']
            
            for card_id in cards:
                if card_id < n_cards:
                    card_score_sum[card_id] += score
                    card_count[card_id] += 1
        
        if (shard_id + 1) % 50 == 0:
            print(f"  Processed shard {shard_id + 1}/{n_shards}")
    
    # Compute averages
    card_avg = np.zeros(n_cards, dtype=np.float64)
    for c in range(n_cards):
        if card_count[c] > 0:
            card_avg[c] = card_score_sum[c] / card_count[c]
    
    return card_avg, card_count


def fit_linear_model(features_list, targets):
    """Fit a linear regression model."""
    # Build feature matrix
    feature_names = [
        'total_symbols', 'n_positives', 'n_negatives', 'n_arrows',
        'n_parrots', 'n_rums', 'n_krakens', 'n_rats', 'n_sharks',
        'rat_quarters', 'rum_bonus'
    ]
    
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(targets)
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Fit using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    
    # Predictions
    y_pred = X_with_intercept @ coeffs
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    return coeffs, ['intercept'] + feature_names, r2, y_pred


def main():
    # Load cards
    with open('pirate_cards/pirate_120_balanced.json') as f:
        cards = json.load(f)
    
    n_cards = len(cards)
    print(f"Loaded {n_cards} cards")
    
    # Extract features
    features_list = [extract_features(card) for card in cards]
    
    # Load simulation results
    card_avg, card_count = load_simulation_results(
        'data/placements_verify', n_shards=256, n_cards=n_cards
    )
    
    # Filter to cards with enough samples
    min_samples = 100
    valid_mask = card_count >= min_samples
    valid_indices = np.where(valid_mask)[0]
    
    print(f"\nCards with {min_samples}+ samples: {len(valid_indices)}/{n_cards}")
    
    if len(valid_indices) < 10:
        print("Not enough data for reliable regression. Using all cards with any data.")
        valid_mask = card_count > 0
        valid_indices = np.where(valid_mask)[0]
    
    valid_features = [features_list[i] for i in valid_indices]
    valid_targets = [card_avg[i] for i in valid_indices]
    
    print(f"\nSimulation value statistics:")
    print(f"  Mean: {np.mean(valid_targets):.2f}")
    print(f"  Std:  {np.std(valid_targets):.2f}")
    print(f"  Min:  {np.min(valid_targets):.2f}")
    print(f"  Max:  {np.max(valid_targets):.2f}")
    
    # Fit model
    coeffs, feature_names, r2, predictions = fit_linear_model(valid_features, valid_targets)
    
    print(f"\n{'='*60}")
    print("CALIBRATED FORMULA")
    print(f"{'='*60}")
    print(f"R² = {r2:.4f}")
    print(f"\nCard Value = {coeffs[0]:.2f}")
    
    for i, name in enumerate(feature_names[1:], 1):
        sign = '+' if coeffs[i] >= 0 else ''
        print(f"            {sign}{coeffs[i]:.3f} × {name}")
    
    # Simplified formula (most important features)
    print(f"\n{'='*60}")
    print("SIMPLIFIED FORMULA (key features only)")
    print(f"{'='*60}")
    
    # Correlations
    print("\nFeature correlations with simulation value:")
    for i, name in enumerate(feature_names[1:]):
        feature_vals = [f[name] for f in valid_features]
        corr = np.corrcoef(feature_vals, valid_targets)[0, 1]
        bar = '█' * int(abs(corr) * 20)
        sign = '+' if corr > 0 else '-'
        print(f"  {name:<16} r={sign}{abs(corr):.3f}  {bar}")
    
    # Predict vs actual
    print(f"\n{'='*60}")
    print("MODEL VALIDATION")
    print(f"{'='*60}")
    
    errors = np.array(valid_targets) - np.array(predictions)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Show some examples
    print(f"\n{'Card':<6} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print("-" * 36)
    sorted_indices = np.argsort(-np.array(valid_targets))[:10]
    for idx in sorted_indices:
        card_idx = valid_indices[idx]
        print(f"{card_idx:<6} {valid_targets[idx]:>8.1f} {predictions[idx]:>10.1f} {errors[idx]:>+8.1f}")
    
    # Generate Python code for the formula
    print(f"\n{'='*60}")
    print("PYTHON CODE FOR CALIBRATED FORMULA")
    print(f"{'='*60}")
    print("""
def calculate_card_value_calibrated(card):
    '''Calculate card value using calibrated formula.'''
    quarters = card['card']['quarters']
    qnames = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    all_symbols = []
    for qn in qnames:
        all_symbols.extend(quarters.get(qn, []))
    
    counts = Counter(all_symbols)
    
    # Feature extraction
    total_symbols = len(all_symbols)
    n_positives = sum(counts.get(s, 0) for s in ['anchor', 'spyglass', 'map', 'coin', 'parrot'])
    n_negatives = sum(counts.get(s, 0) for s in ['shark', 'rat', 'kraken'])
    n_arrows = sum(counts.get(s, 0) for s in ['arrow_up', 'arrow_down', 'arrow_left', 'arrow_right'])
    n_parrots = counts.get('parrot', 0)
    n_rums = counts.get('rum', 0)
    n_krakens = counts.get('kraken', 0)
    n_rats = counts.get('rat', 0)
    n_sharks = counts.get('shark', 0)
    rat_quarters = sum(1 for qn in qnames if 'rat' in quarters.get(qn, []))
    rum_bonus = max(0, n_rums - 2)
    """)
    
    print(f"    # Calibrated coefficients (R² = {r2:.3f})")
    print(f"    value = {coeffs[0]:.2f}")
    for i, name in enumerate(feature_names[1:], 1):
        if abs(coeffs[i]) > 0.01:
            sign = '+' if coeffs[i] >= 0 else ''
            print(f"    value {sign}= {coeffs[i]:.3f} * {name}")
    
    print("    return value")


if __name__ == '__main__':
    main()
