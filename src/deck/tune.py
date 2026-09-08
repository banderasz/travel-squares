#!/usr/bin/env python3
"""Iteratively fit the value model and generate a balanced deck.

The loop
--------
    generate a deck at the target value using the current model
      -> measure what the cards are actually worth (src/deck/measure.py)
      -> refit the model against those measurements
      -> repeat

The point is that the model starts out wrong, so the first deck is only roughly
balanced. Each round the model gets closer to the simulation, so the decks get
tighter. Two numbers say whether it is working:

    spread   max - min measured card value. This is the goal: small spread
             means no card is plainly better than another.
    offset   measured mean - target. How well you can dial the deck's power
             level up or down.

Usage:
    python -m src.deck.tune --target 12 --cards 30 --iterations 5
    python -m src.deck.tune --target 16 --cards 30      # a stronger deck
"""
import argparse
import json
import random

import numpy as np

from src.cards import Card
from src.symbols import Symbols
from src.deck_io import card_to_dict, dumps_deck
from src.deck.measure import measure_deck
from src.deck.simulation import get_paths, load_cards
from src.deck.value_model import DEFAULT_MODEL_PATH, ValueModel


# src/deck/simulation.py allocates room for this many arrows per card
# (a_qi is shaped [cards, rotations, 4]). Random sampling can exceed it, which
# blows up in load_cards, so candidates carrying more are rejected here.
MAX_ARROWS_PER_CARD = 4


def random_card():
    """A weighted-random card the simulation can actually load."""
    while True:
        card = card_to_dict(Card.generate_card())
        arrows = sum(
            1 for syms in card['card']['quarters'].values()
            for s in syms if Symbols.of(s) in Symbols.arrows()
        )
        if arrows <= MAX_ARROWS_PER_CARD:
            return card


def generate_deck(model, aim, n_cards, tolerance, candidates, rng):
    """Sample random cards and keep the n the model puts closest to `aim`.

    The aim is clamped into the range the pool can actually supply. Without
    that, asking for a value no random card reaches just returns the n most
    extreme cards in the pool — which is how a mis-calibrated model produces a
    deck nowhere near what was requested.

    Returns the deck and the aim actually used.
    """
    cards = [random_card() for _ in range(candidates)]
    predicted = model.predict_many(cards)

    low, high = np.quantile(predicted, [0.05, 0.95])
    clamped = float(np.clip(aim, low, high))

    order = np.argsort(np.abs(predicted - clamped))[:n_cards]
    return [cards[i] for i in order], clamped


def random_deck(n_cards, rng):
    """A deck of unfiltered random cards — wide value range, for calibration."""
    return [random_card() for _ in range(n_cards)]


def _measure(deck, deck_out, contexts, paths, seed, all_paths):
    with open(deck_out, 'w') as handle:
        handle.write(dumps_deck(deck))
    return measure_deck(deck_out, contexts=contexts, paths=paths, seed=seed,
                        card_data=load_cards(deck_out), path_cache=all_paths)


def run(target, n_cards, iterations, candidates, tolerance,
        contexts, paths, deck_out, model_path, seed, ridge, damping,
        seed_decks, on_target):
    rng = random.Random(seed)
    all_paths = get_paths()
    history_cards, history_values = [], []

    print(f"\n  target={target}  cards={n_cards}  iterations={iterations}"
          f"  (contexts={contexts}, paths={paths}, ridge={ridge})\n")

    # Seed with an unfiltered random deck. Target-matched decks are, by design,
    # nearly uniform in value, which leaves the fit almost nothing to learn
    # from; this one spans the whole range and carries the slope information.
    seed_results = []
    for s_i in range(seed_decks):
        d = random_deck(n_cards, rng)
        r = _measure(d, deck_out, contexts, paths, seed + s_i, all_paths)
        history_cards += d
        history_values += list(r['per_card'])
        seed_results.append(r)
    seed_result = seed_results[-1]
    model = ValueModel.fit(history_cards, np.array(history_values), ridge=ridge)

    print(f"  {'iter':<6} {'spread':>7} {'std':>6} {'mean':>7} {'offset':>7} "
          f"{'R2':>6} {'MAE':>6} {'n fit':>6} {'aim':>7}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    q = model.score(history_cards, np.array(history_values))
    print(f"  {'seed':<6} {seed_result['spread']:>7.2f} {seed_result['std']:>6.2f} "
          f"{seed_result['mean']:>7.2f} {seed_result['mean']-target:>+7.2f} "
          f"{q['r2']:>6.3f} {q['mae']:>6.2f} {len(history_cards):>6}")

    # A card is worth less among strong cards than among weak ones, so a deck
    # built to look like `target` does not measure as `target`. Refitting the
    # intercept to chase that overshoots and oscillates. Instead the fit supplies
    # the slopes (which control spread) and this damped correction supplies the
    # level: aim higher when the deck measured low.
    aim = target
    history_aim = []
    best = None
    for i in range(1, iterations + 1):
        deck, used_aim = generate_deck(model, aim, n_cards, tolerance, candidates, rng)
        result = _measure(deck, deck_out, contexts, paths, seed, all_paths)

        # Deliberately not refitting on this deck. Its cards were chosen to have
        # near-identical predicted value, so they carry almost no feature
        # variation; folding them in makes the slopes lurch between rounds and
        # the aim correction chase a model that has already moved. The slopes
        # come from the diverse seed decks; the loop only moves the aim.
        q = model.score(history_cards, np.array(history_values))

        print(f"  {i:<6} {result['spread']:>7.2f} {result['std']:>6.2f} "
              f"{result['mean']:>7.2f} {result['mean']-target:>+7.2f} "
              f"{q['r2']:>6.3f} {q['mae']:>6.2f} {len(history_cards):>6} {used_aim:>7.2f}")

        # Secant step. A fixed damping cannot work here: the model's predicted
        # values span a much narrower range than the measured ones, so moving
        # the aim by 1 can move the measured mean by 7 or more, and any constant
        # gain either crawls or oscillates. Estimating that gain from the last
        # two rounds and solving for the target converges in a few steps.
        history_aim.append((used_aim, result['mean']))
        if len(history_aim) >= 2:
            (a0, m0), (a1, m1) = history_aim[-2], history_aim[-1]
            if abs(m1 - m0) > 1e-6 and abs(a1 - a0) > 1e-9:
                gain = (m1 - m0) / (a1 - a0)
                aim = a1 + (target - m1) / gain
            else:
                aim = used_aim + (target - result['mean']) * damping
        else:
            aim = used_aim + (target - result['mean']) * damping

        # A tightly balanced deck at the wrong power level is not what was
        # asked for, so rank on missing the target first and only then on spread.
        offset = abs(result['mean'] - target)
        key = (offset > on_target, round(offset + result['spread'], 2))
        if best is None or key < best[0]:
            best = (key, result['spread'], result['mean'], list(deck))

    model.save(model_path)
    _, spread, mean, deck = best
    with open(deck_out, 'w') as handle:
        handle.write(dumps_deck(deck))

    print(f"\n  best deck: spread={spread:.2f}, mean={mean:.2f} "
          f"(target {target}, off by {mean - target:+.2f})")
    print(f"  deck  -> {deck_out}")
    print(f"  model -> {model_path}")
    print(f"\n  {model}")
    return spread


def _exists(path):
    try:
        open(path).close()
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', type=float, default=12.0,
                   help='Desired average card value. Lower = weaker deck.')
    p.add_argument('--cards', type=int, default=30, help='Cards in the deck')
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--candidates', type=int, default=4000,
                   help='Random cards sampled per iteration to choose from')
    p.add_argument('--tolerance', type=float, default=0.4,
                   help='Keep candidates within this much of the target')
    p.add_argument('--contexts', type=int, default=32,
                   help='Shared 5-card contexts when measuring (more = less noise)')
    p.add_argument('--paths', type=int, default=60,
                   help='Position paths per evaluation (more = less noise)')
    p.add_argument('--deck-out', default='decks/tuned.json')
    p.add_argument('--model', default=DEFAULT_MODEL_PATH)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ridge', type=float, default=1.0,
                   help='L2 penalty on the fit; higher = more conservative')
    p.add_argument('--seed-decks', type=int, default=2,
                   help='Random decks measured up front to fit the initial model')
    p.add_argument('--on-target', type=float, default=1.0,
                   help='Count a deck as on target if its mean is within this much')
    p.add_argument('--damping', type=float, default=0.8,
                   help='How hard to correct the aim toward the target each round')
    a = p.parse_args()

    run(a.target, a.cards, a.iterations, a.candidates, a.tolerance,
        a.contexts, a.paths, a.deck_out, a.model, a.seed, a.ridge, a.damping,
        a.seed_decks, a.on_target)


if __name__ == '__main__':
    main()
