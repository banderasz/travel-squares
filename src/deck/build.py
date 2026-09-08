#!/usr/bin/env python3
"""Generate a deck of near-equal cards at a chosen value, and check it worked.

    1. calibrate  measure a batch of purely random cards and fit the linear
                  value model to what they actually scored
    2. generate   sample many random cards, keep the ones the model puts near
                  the target — this is where the power level is chosen
    3. verify     measure the generated deck the same way and compare it to the
                  random baseline

Everything uses the fast sampled measurement in src/deck/evaluate.py, so a full
build takes seconds.

    python -m src.deck.build --target 38 --cards 20
"""
import argparse
import tempfile

import numpy as np

from src.deck_io import dumps_deck
from src.deck.evaluate import measure
from src.deck.tune import random_card
from src.deck.value_model import ValueModel


def _write(cards):
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    handle.write(dumps_deck(cards))
    handle.close()
    return handle.name


def _measure_cards(cards, pool, hands, rounds, placements, seed):
    result = measure(_write(cards), len(cards), pool, hands, rounds, placements, seed=seed)
    return result, np.array([v.mean() for v in result['middle']])


def build(target, n_cards, calibration, candidates, pool, hands, rounds,
          placements, seed, iterations):
    rng = np.random.default_rng(seed)

    print(f"\n  1. calibrating on {calibration} random cards")
    cal_cards = [random_card() for _ in range(calibration)]
    cal_result, cal_means = _measure_cards(cal_cards, pool, hands, rounds, placements, seed)
    model = ValueModel.fit(cal_cards, cal_means, ridge=1.0)
    q = model.score(cal_cards, cal_means)
    print(f"     random cards span {cal_means.min():.2f} to {cal_means.max():.2f} "
          f"(spread {cal_means.max()-cal_means.min():.2f}, sd {cal_means.std():.2f})")
    print(f"     model R2={q['r2']:.3f}  MAE={q['mae']:.2f}")

    baseline_result = measure(_write(cal_cards[:n_cards]), n_cards, pool, hands,
                              rounds, placements, seed=seed)
    baseline = np.array([v.mean() for v in baseline_result['middle']])

    print(f"\n  2. generating {n_cards} cards near {target}")
    aim, best = target, None
    for i in range(1, iterations + 1):
        pooled = [random_card() for _ in range(candidates)]
        predicted = model.predict_many(pooled)
        low, high = np.quantile(predicted, [0.02, 0.98])
        used = float(np.clip(aim, low, high))
        pick = np.argsort(np.abs(predicted - used))[:n_cards]
        deck = [pooled[i] for i in pick]

        result, means = _measure_cards(deck, pool, hands, rounds, placements, seed)
        spread, mean = means.max() - means.min(), means.mean()
        print(f"     iter {i}: aim {used:6.2f} -> measured mean {mean:6.2f} "
              f"(off {mean-target:+.2f}), spread {spread:.2f}, sd {means.std():.2f}")

        key = (abs(mean - target) > 1.0, round(abs(mean - target) + spread, 2))
        if best is None or key < best[0]:
            best = (key, deck, result, means)
        aim = used + (target - mean)   # model units and measured units differ; step directly

    _, deck, result, means = best
    print(f"\n  3. result")
    print(f"     random 20   : spread {baseline.max()-baseline.min():.2f}  "
          f"sd {baseline.std():.2f}  mean {baseline.mean():.2f}")
    print(f"     generated 20: spread {means.max()-means.min():.2f}  "
          f"sd {means.std():.2f}  mean {means.mean():.2f}  (target {target})")
    return deck, result, means, baseline_result, baseline, model


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', type=float, default=38.0)
    p.add_argument('--cards', type=int, default=20)
    p.add_argument('--calibration', type=int, default=60)
    p.add_argument('--candidates', type=int, default=4000)
    p.add_argument('--pool', type=int, default=20)
    p.add_argument('--hands', type=int, default=200)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--placements', type=int, default=300)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='decks/generated.json')
    a = p.parse_args()

    deck, *_ = build(a.target, a.cards, a.calibration, a.candidates, a.pool,
                     a.hands, a.rounds, a.placements, a.seed, a.iterations)
    with open(a.out, 'w') as handle:
        handle.write(dumps_deck(deck))
    print(f"\n  deck -> {a.out}\n")


if __name__ == '__main__':
    main()
