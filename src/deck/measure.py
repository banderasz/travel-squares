"""Measure what each card in a deck is actually worth, quickly.

The slow, exact answer comes from src/deck/simulation.py in --placements mode.
This is the fast approximation used to drive the calibrate/generate loop.

Estimator
---------
Naively you would deal random 6-card hands and average the score of the hands
each card appears in. That works but is noisy: hand scores have a spread of
~3.5 points, almost all of it from *which other five cards* came along.

Instead this pairs. It samples a handful of 5-card **contexts**, and for each
context evaluates every remaining card as the sixth. Every card therefore faces
the same opponents, so the shared component cancels and what is left is the
card's own contribution. That buys roughly an order of magnitude in precision
per unit of compute.
"""
import numpy as np

from src.deck.simulation import evaluate_batch, get_paths, load_cards

CONTEXT_SIZE = 5


def measure_deck(deck_path, contexts=24, paths=40, seed=0, card_data=None, path_cache=None):
    """Return per-card mean score for a deck.

    contexts  number of shared 5-card contexts (more = less noise)
    paths     random position paths per evaluation (more = less placement noise)

    Pass card_data/path_cache to avoid reloading them in a loop.
    """
    cd = card_data if card_data is not None else load_cards(deck_path)
    all_paths = path_cache if path_cache is not None else get_paths()
    n_cards = cd['n_cards']
    if n_cards <= CONTEXT_SIZE:
        raise ValueError(f"deck needs more than {CONTEXT_SIZE} cards, got {n_cards}")

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(all_paths), min(paths, len(all_paths)), replace=False)
    chosen_paths = np.ascontiguousarray(all_paths[sample])

    hands, owners = [], []
    for _ in range(contexts):
        context = rng.choice(n_cards, CONTEXT_SIZE, replace=False)
        in_context = set(int(c) for c in context)
        for card in range(n_cards):
            if card in in_context:
                continue
            hands.append((card, *(int(c) for c in context)))
            owners.append(card)

    scores = np.asarray(evaluate_batch(hands, chosen_paths, cd), dtype=np.float64)
    owners = np.asarray(owners)

    total = np.zeros(n_cards)
    count = np.zeros(n_cards)
    np.add.at(total, owners, scores)
    np.add.at(count, owners, 1)
    seen = count > 0
    per_card = np.full(n_cards, np.nan)
    per_card[seen] = total[seen] / count[seen]

    return {
        'per_card': per_card,
        'samples_per_card': int(count[seen].min()) if seen.any() else 0,
        'mean': float(np.nanmean(per_card)),
        'std': float(np.nanstd(per_card)),
        'spread': float(np.nanmax(per_card) - np.nanmin(per_card)),
        'n_cards': n_cards,
        'n_hands': len(hands),
    }


def summarise(result, label=""):
    p = result['per_card']
    order = np.argsort(-p)
    best, worst = order[0], order[-1]
    return (f"{label}mean={result['mean']:.2f}  std={result['std']:.2f}  "
            f"spread={result['spread']:.2f}  "
            f"(best card {best} @ {p[best]:.2f}, worst {worst} @ {p[worst]:.2f})")
