#!/usr/bin/env python3
"""Grow a large deck of near-equal cards, checkpointing every step.

The loop:

    measure the whole deck
      -> pool the result with every earlier measurement of the same card
      -> refit the heuristic on all of it
      -> replace only the cards whose pooled value is worse than the target by
         more than the measurement can explain
      -> repeat

Two things make this different from a naive replace-the-worst loop, both
learned from watching that version fail:

**Evidence is pooled per card.** A card keeps every value it has ever
measured, so a card that has survived twenty iterations is known twenty times
more precisely than a newcomer. Judging on the latest measurement alone means
keeping whichever cards got lucky, watching them regress, and calling the
resulting bounce progress.

**A card is only replaced on significant evidence.** Once the deck is tight,
the spread between cards is smaller than the noise in a single measurement, and
swapping on that noise is not just useless — it actively churns good cards out.
A card must miss the target by more than `--sigma` standard errors to go.

Every iteration is written atomically before the next begins, so the run can be
killed at any point: the deck on disk is complete, and restarting continues
from the same place.

    python -m src.deck.grow --heuristic measured --cards 120
    python -m src.deck.grow --heuristic analytic --cards 120
"""
import argparse
import hashlib
import json
import os
import signal
import tempfile
import time

import numpy as np
from scipy.stats import binom

from src.deck.evaluate import measure
from src.deck.tune import random_card
from src.deck.value_model import ValueModel
from src.deck_io import dumps_deck
from src.symbols import Symbols, NUMBER_OF_SYMBOLS_IN_PLAY as TOTAL

CAP = 11
SLOTS = 16
SCORING_SYMBOLS = [s for s in Symbols if s.value_symbol()]
_stop = False


def _on_signal(*_):
    global _stop
    _stop = True
    print("\n  stop requested — finishing this iteration, then saving", flush=True)


def card_key(card):
    """Stable identity for a card, so its measurements accumulate across runs."""
    q = card['card']['quarters']
    canon = ";".join(f"{name}:{','.join(sorted(q.get(name, [])))}"
                     for name in sorted(q))
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def analytic_tables(others=5):
    """Marginal value of holding k of each symbol, given `others` random cards."""
    tables = {}
    for s in SCORING_SYMBOLS:
        p = s.weight / TOTAL
        pts = np.array(s.points, dtype=float)
        d = binom.pmf(np.arange(SLOTS * others + 1), SLOTS * others, p)
        tables[s] = np.array([sum(d[i] * pts[min(i + k, CAP)] for i in range(len(d)))
                              for k in range(SLOTS + 1)])
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
    def __init__(self, kind):
        self.kind = kind
        self.model = None
        self.scale = (0.0, 1.0)

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
                'intercept': self.model.intercept, 'coefficients': self.model.coefficients}
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
        r = measure(handle.name, len(cards), pool, hands, rounds, placements,
                    seed=seed, keep_values=False)
    finally:
        os.unlink(handle.name)
    return r


def propose(heuristic, aim, n, candidates):
    pool = [random_card() for _ in range(candidates)]
    predicted = heuristic.predict(pool)
    low, high = np.quantile(predicted, [0.02, 0.98])
    order = np.argsort(np.abs(predicted - float(np.clip(aim, low, high))))[:n]
    return [pool[i] for i in order]


def save(path, state):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(state, handle)
    os.replace(tmp, path)


def run(kind, n_cards, target, iterations, max_replace, sigma, candidates,
        pool, hands, rounds, placements, checkpoint, deck_out, seed):
    heuristic = Heuristic(kind)
    # records[key] = {'card': …, 'values': [means], 'sds': [per-measurement sd]}
    records, deck_keys, history, start = {}, [], [], 0

    if os.path.exists(checkpoint):
        with open(checkpoint) as handle:
            st = json.load(handle)
        records = st['records']
        deck_keys = st['deck_keys']
        history = st['history']
        start = st['iteration']
        target = st['target'] if target is None else target
        heuristic.load(st['heuristic_state'])
        print(f"  resuming {kind} at iteration {start}: {len(records)} distinct cards known",
              flush=True)

    if not deck_keys:
        for c in [random_card() for _ in range(n_cards)]:
            records[card_key(c)] = {'card': c, 'values': [], 'sds': []}
        deck_keys = list(records)

    hands_per_card = rounds * hands * 6 // pool
    print(f"  [{kind}] {n_cards} cards, {hands_per_card:,} hands per card per iteration,"
          f" replace<={max_replace} at >{sigma} sigma", flush=True)

    for i in range(start + 1, start + iterations + 1):
        if _stop:
            break
        t0 = time.time()
        deck = [records[k]['card'] for k in deck_keys]
        result = measure_deck(deck, pool, hands, rounds, placements, seed + i)

        for j, k in enumerate(deck_keys):
            records[k]['values'].append(float(result['middle_mean'][j]))
            records[k]['sds'].append(
                float(result['middle_std'][j] / np.sqrt(max(result['count'][j], 1))))

        # pooled estimate per card, and how well it is known
        est = np.array([np.mean(records[k]['values']) for k in deck_keys])
        n_obs = np.array([len(records[k]['values']) for k in deck_keys])
        se = np.array([np.mean(records[k]['sds']) / np.sqrt(len(records[k]['sds']))
                       for k in deck_keys])

        if target is None:
            target = float(np.median(est))

        all_cards = [r['card'] for r in records.values() if r['values']]
        all_vals = [float(np.mean(r['values'])) for r in records.values() if r['values']]
        heuristic.fit(all_cards, all_vals)

        latest = result['middle_mean']
        history.append({'iteration': i,
                        'spread_pooled': float(est.max() - est.min()),
                        'sd_pooled': float(est.std()),
                        'spread_latest': float(latest.max() - latest.min()),
                        'mean': float(est.mean()), 'target': target,
                        'median_obs': int(np.median(n_obs)),
                        'typical_se': float(np.median(se)),
                        'distinct_cards': len(records)})

        # replace only where the miss is bigger than the uncertainty
        z = np.abs(est - target) / np.maximum(se, 1e-9)
        doomed = [j for j in np.argsort(-z)[:max_replace] if z[j] > sigma]
        print(f"  [{kind:>8}] iter {i:>4}  pooled spread {est.max()-est.min():5.2f}  "
              f"sd {est.std():4.2f}  se {np.median(se):.3f}  obs/card {int(np.median(n_obs)):>3}  "
              f"replacing {len(doomed):>2}  ({time.time()-t0:.0f}s)", flush=True)

        if doomed:
            keep = [k for j, k in enumerate(deck_keys) if j not in set(doomed)]
            fresh = propose(heuristic, target, len(doomed), candidates)
            for c in fresh:
                records.setdefault(card_key(c), {'card': c, 'values': [], 'sds': []})
            deck_keys = keep + [card_key(c) for c in fresh]

        save(checkpoint, {'iteration': i, 'target': target, 'records': records,
                          'deck_keys': deck_keys, 'history': history,
                          'heuristic_kind': kind,
                          'heuristic_state': heuristic.state()})
        with open(deck_out, 'w') as handle:
            handle.write(dumps_deck([records[k]['card'] for k in deck_keys]))

    print(f"  [{kind}] stopped at iteration {history[-1]['iteration'] if history else start}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--heuristic', choices=['measured', 'analytic'], required=True)
    p.add_argument('--cards', type=int, default=120)
    p.add_argument('--target', type=float, default=None)
    p.add_argument('--iterations', type=int, default=100000)
    p.add_argument('--replace', type=int, default=6,
                   help='Most cards replaced in one iteration')
    p.add_argument('--sigma', type=float, default=2.0,
                   help='Only replace a card missing the target by this many standard errors')
    p.add_argument('--candidates', type=int, default=6000)
    p.add_argument('--pool', type=int, default=20)
    p.add_argument('--hands', type=int, default=200)
    p.add_argument('--rounds', type=int, default=1200,
                   help='Measurement precision. 1200 ~= 72,000 hands per card.')
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
    run(a.heuristic, a.cards, a.target, a.iterations, a.replace, a.sigma, a.candidates,
        a.pool, a.hands, a.rounds, a.placements, checkpoint, deck_out, a.seed)


if __name__ == '__main__':
    main()
