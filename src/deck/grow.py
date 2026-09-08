#!/usr/bin/env python3
"""Grow a large deck of near-equal cards, checkpointing every step.

Runs a replace-the-worst loop:

    measure the whole deck
      -> record it
      -> refit the heuristic on everything measured so far
      -> throw out the cards furthest from the target
      -> generate replacements the heuristic thinks are on target
      -> repeat

Every iteration is written to a checkpoint before the next begins, so the run
can be killed at any point: the deck on disk is always a complete, usable one,
and restarting picks up exactly where it stopped.

Two heuristics can drive the selection:

    measured   a linear model refitted from the accumulated measurements.
               13 free numbers, knows about hiding and arrows because it only
               ever sees real results.
    analytic   the marginal value derived from the scoring tables by averaging
               over what five random cards are likely to bring. Its shape is
               fixed by theory; only a scale and offset are fitted, so it can
               price a card it has never seen.

    python -m src.deck.grow --heuristic measured --cards 120
    python -m src.deck.grow --heuristic analytic --cards 120
"""
import argparse
import json
import os
import signal
import tempfile
import time

import numpy as np
from scipy.stats import binom

from src.deck.evaluate import measure
from src.deck.tune import random_card
from src.deck.value_model import ValueModel, feature_matrix, features
from src.deck_io import dumps_deck
from src.symbols import Symbols, NUMBER_OF_SYMBOLS_IN_PLAY as TOTAL

CAP = 11          # the simulation saturates counts here
SLOTS = 16
SCORING_SYMBOLS = [s for s in Symbols if s.value_symbol()]
_stop = False


def _on_signal(*_):
    global _stop
    _stop = True
    print("\n  stop requested — finishing this iteration, then saving", flush=True)


def analytic_tables(others=5):
    """Marginal value of holding k of each symbol, given `others` random cards.

    Averages the scoring table over the distribution of what the other cards
    bring, so the non-linearity of the table is priced correctly. Needs no
    measurements, which is the point: it can value a card never seen before.
    """
    tables = {}
    for s in SCORING_SYMBOLS:
        p = s.weight / TOTAL
        pts = np.array(s.points, dtype=float)
        d = binom.pmf(np.arange(SLOTS * others + 1), SLOTS * others, p)
        tables[s] = np.array([
            sum(d[i] * pts[min(i + k, CAP)] for i in range(len(d)))
            for k in range(SLOTS + 1)
        ])
    return tables


ANALYTIC = analytic_tables()


def analytic_raw(card):
    counts = {s: 0 for s in SCORING_SYMBOLS}
    for syms in card['card']['quarters'].values():
        for name in syms:
            s = Symbols.of(name)
            if s.value_symbol():
                counts[s] += 1
    return sum(ANALYTIC[s][counts[s]] for s in SCORING_SYMBOLS)


class Heuristic:
    """Prices a card. Either kind is refitted from whatever has been measured."""

    def __init__(self, kind):
        self.kind = kind
        self.model = None        # measured: ValueModel
        self.scale = (0.0, 1.0)  # analytic: offset, slope

    def fit(self, cards, values):
        if self.kind == 'measured':
            self.model = ValueModel.fit(cards, np.asarray(values), ridge=1.0)
        else:
            x = np.array([analytic_raw(c) for c in cards])
            slope, offset = np.polyfit(x, np.asarray(values), 1)
            self.scale = (float(offset), float(slope))

    def predict(self, cards):
        if self.kind == 'measured':
            if self.model is None:
                return np.zeros(len(cards))
            return self.model.predict_many(cards)
        offset, slope = self.scale
        return offset + slope * np.array([analytic_raw(c) for c in cards])

    def state(self):
        if self.kind == 'measured':
            return None if self.model is None else {
                'intercept': self.model.intercept,
                'coefficients': self.model.coefficients}
        return {'offset': self.scale[0], 'slope': self.scale[1]}

    def load(self, state):
        if state is None:
            return
        if self.kind == 'measured':
            self.model = ValueModel(state['intercept'], state['coefficients'])
        else:
            self.scale = (state['offset'], state['slope'])


def measure_deck(cards, pool, hands, rounds, placements, seed):
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    handle.write(dumps_deck(cards))
    handle.close()
    try:
        result = measure(handle.name, len(cards), pool, hands, rounds, placements, seed=seed)
    finally:
        os.unlink(handle.name)
    return np.array([v.mean() for v in result['middle']])


def propose(heuristic, aim, n, candidates):
    pool = [random_card() for _ in range(candidates)]
    predicted = heuristic.predict(pool)
    low, high = np.quantile(predicted, [0.02, 0.98])
    order = np.argsort(np.abs(predicted - float(np.clip(aim, low, high))))[:n]
    return [pool[i] for i in order]


def save(path, deck, history, heuristic, seen_cards, seen_values, iteration, target):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump({
            'iteration': iteration,
            'target': target,
            'deck': deck,
            'history': history,
            'heuristic_kind': heuristic.kind,
            'heuristic_state': heuristic.state(),
            'seen_cards': seen_cards,
            'seen_values': seen_values,
        }, handle)
    os.replace(tmp, path)          # atomic: a killed run never leaves a partial file


def run(kind, n_cards, target, iterations, replace, candidates,
        pool, hands, rounds, placements, checkpoint, deck_out, seed):
    heuristic = Heuristic(kind)
    deck, history, seen_cards, seen_values, start, = [], [], [], [], 0

    if os.path.exists(checkpoint):
        with open(checkpoint) as handle:
            state = json.load(handle)
        deck = state['deck']
        history = state['history']
        seen_cards = state['seen_cards']
        seen_values = state['seen_values']
        start = state['iteration']
        target = state['target'] if target is None else target
        heuristic.load(state['heuristic_state'])
        print(f"  resuming {kind} from iteration {start} "
              f"({len(seen_cards)} cards measured so far)", flush=True)

    if not deck:
        deck = [random_card() for _ in range(n_cards)]

    for i in range(start + 1, start + iterations + 1):
        if _stop:
            break
        t0 = time.time()
        values = measure_deck(deck, pool, hands, rounds, placements, seed + i)

        seen_cards += deck
        seen_values += [float(v) for v in values]
        heuristic.fit(seen_cards, seen_values)

        if target is None:                     # self-levelling: aim at the deck itself
            target = float(np.median(values))

        spread = float(values.max() - values.min())
        history.append({'iteration': i, 'spread': spread,
                        'std': float(values.std()), 'mean': float(values.mean()),
                        'target': target, 'measured': len(seen_cards)})
        print(f"  [{kind:>8}] iter {i:>4}  spread {spread:6.2f}  sd {values.std():5.2f}  "
              f"mean {values.mean():6.2f}  ({time.time()-t0:.1f}s)", flush=True)

        # keep the cards nearest the target, replace the rest
        keep = np.argsort(np.abs(values - target))[:n_cards - replace]
        deck = [deck[i] for i in keep] + propose(heuristic, target, replace, candidates)

        save(checkpoint, deck, history, heuristic, seen_cards, seen_values, i, target)
        with open(deck_out, 'w') as handle:
            handle.write(dumps_deck(deck))

    print(f"  [{kind}] stopped at iteration {history[-1]['iteration'] if history else start}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--heuristic', choices=['measured', 'analytic'], required=True)
    p.add_argument('--cards', type=int, default=120)
    p.add_argument('--target', type=float, default=None,
                   help='Value to aim for. Default: the deck levels itself.')
    p.add_argument('--iterations', type=int, default=100000)
    p.add_argument('--replace', type=int, default=12, help='Cards swapped per iteration')
    p.add_argument('--candidates', type=int, default=4000)
    p.add_argument('--pool', type=int, default=20)
    p.add_argument('--hands', type=int, default=200)
    p.add_argument('--rounds', type=int, default=40)
    p.add_argument('--placements', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint', default='')
    p.add_argument('--deck-out', default='')
    a = p.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    checkpoint = a.checkpoint or f'data/grow_{a.heuristic}.json'
    deck_out = a.deck_out or f'decks/grown_{a.heuristic}.json'
    os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
    run(a.heuristic, a.cards, a.target, a.iterations, a.replace, a.candidates,
        a.pool, a.hands, a.rounds, a.placements, checkpoint, deck_out, a.seed)


if __name__ == '__main__':
    main()
