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

    python -m src.deck.grow --cards 120 --target 20
"""
import argparse
import hashlib
import json
import os
import signal
import tempfile
import time

import numpy as np

from src.deck.evaluate import measure
from src.deck.tune import random_card
from src.deck.value_model import ValueModel
from src.deck_io import dumps_deck

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


class Heuristic:
    """Prices a card so candidates can be filtered without measuring them.

    A linear model over symbol counts, refitted on every measurement taken so
    far. There used to be an analytic alternative here, deriving marginal value
    from the scoring tables instead of from data; it was measurably worse (0.864
    against 0.900 predictive accuracy, and a final deck spread of 1.85 against
    0.40) and is kept only in the git history. Its supposed advantage — pricing
    a card configuration never observed — does not apply to a linear model,
    which extrapolates to any count.
    """

    def __init__(self):
        self.model = None

    def fit(self, cards, values):
        self.model = ValueModel.fit(cards, np.asarray(values), ridge=1.0)

    def predict(self, cards):
        if self.model is None:
            return np.zeros(len(cards))
        return self.model.predict_many(cards)

    def state(self):
        return None if self.model is None else {
            'intercept': self.model.intercept, 'coefficients': self.model.coefficients}

    def load(self, state):
        if state is not None:
            self.model = ValueModel(state['intercept'], state['coefficients'])


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


def run(n_cards, bench, target, iterations, max_replace, sigma, candidates,
        pool, hands, rounds, placements, checkpoint, deck_out, seed, init_from,
        calibration_rounds):
    # The roster carries `bench` more cards than the deck ships. Freshly
    # generated cards land on the bench, get measured there, and only join the
    # shipped deck if they earn it. Without that, every iteration's reported
    # spread included cards with a single observation to their name.
    roster_size = n_cards + bench
    heuristic = Heuristic()
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
        print(f"  resuming at iteration {start}: {len(records)} distinct cards known",
              flush=True)

    if not deck_keys:
        # Bootstrap. Without this the first roster is unfiltered random, and the
        # loop spends ~20 iterations dragging it to the target six cards at a
        # time — a phase where the reported spread describes a mixture of old
        # and new cards rather than the balance of anything.
        if init_from and os.path.exists(init_from):
            with open(init_from) as handle:
                prior = json.load(handle)
            heuristic.load(prior.get('heuristic_state'))
            print(f"  seeded the heuristic from {init_from}", flush=True)

        # Measure a batch, fit the heuristic, build the opening roster from it.
        #
        # One round only. Repeating this oscillates: a heuristic fitted on a weak
        # deck builds an over-strong one, refitting on that builds an over-weak
        # one, and so on (12.6 -> 28.8 -> 4.8 -> 24.0 when tried with four). The
        # main loop is stable precisely because it replaces a few cards at a
        # time, so it damps the same feedback instead of amplifying it. Starting
        # high and walking down is fine; starting from a full rebuild is not.
        calib = (propose(heuristic, target, roster_size, candidates)
                 if heuristic.state() else [random_card() for _ in range(roster_size)])
        for step in range(max(1, calibration_rounds)):
            cal_values = list(measure_deck(calib, pool, hands, rounds,
                                           placements, seed + step)['middle_mean'])
            heuristic.fit(calib, np.array(cal_values))
            if target is None:
                target = float(np.median(cal_values))
            print(f"  calibration {step + 1}/{calibration_rounds}: "
                  f"deck mean {np.mean(cal_values):6.2f}, target {target:.2f}", flush=True)
            calib = propose(heuristic, target, roster_size, candidates)

        for c in calib:
            records[card_key(c)] = {'card': c, 'values': [], 'sds': []}
        deck_keys = list(records)

    hands_per_card = rounds * hands * 6 // pool
    print(f"  roster {roster_size} -> ships best {n_cards}, "
          f"{hands_per_card:,} hands per card per iteration, "
          f"replace<={max_replace} at >{sigma} sigma", flush=True)

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

        # the deck we would ship: the n_cards closest to target
        rank = np.argsort(np.abs(est - target))
        shipped, benched = rank[:n_cards], rank[n_cards:]
        ship_est = est[shipped]

        all_cards = [r['card'] for r in records.values() if r['values']]
        all_vals = [float(np.mean(r['values'])) for r in records.values() if r['values']]
        heuristic.fit(all_cards, all_vals)

        latest = result['middle_mean']
        history.append({'iteration': i,
                        'spread_pooled': float(ship_est.max() - ship_est.min()),
                        'sd_pooled': float(ship_est.std()),
                        'spread_roster': float(est.max() - est.min()),
                        'spread_latest': float(latest.max() - latest.min()),
                        'mean': float(est.mean()), 'target': target,
                        'median_obs': int(np.median(n_obs)),
                        'typical_se': float(np.median(se)),
                        'distinct_cards': len(records)})

        # No aim correction. The heuristic is refitted every iteration on
        # measurements taken in the *current* deck, so asking it for `target`
        # already means "a card that would measure target here", and swapping
        # such cards in walks the deck to the target on its own. Correcting the
        # aim from what fresh cards score does not work: a weak card in a strong
        # deck still scores high, because the value is a hand value and includes
        # its five companions. That offset is mostly the deck's level, and
        # steering on it oscillates instead of converging.
        ship_mean = float(ship_est.mean())

        # cut only from the bench, and only where the miss beats the uncertainty
        z = np.abs(est - target) / np.maximum(se, 1e-9)
        bench_by_z = sorted(benched, key=lambda j: -z[j])
        doomed = [j for j in bench_by_z[:max_replace] if z[j] > sigma]
        print(f"  iter {i:>4}  mean {ship_mean:6.2f} (target {target:.1f})  "
              f"spread {ship_est.max()-ship_est.min():5.2f}  sd {ship_est.std():4.2f}  "
              f"obs/card {int(np.median(n_obs)):>3}  replacing {len(doomed):>2}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

        if doomed:
            keep = [k for j, k in enumerate(deck_keys) if j not in set(doomed)]
            fresh = propose(heuristic, target, len(doomed), candidates)
            for c in fresh:
                records.setdefault(card_key(c), {'card': c, 'values': [], 'sds': []})
            deck_keys = keep + [card_key(c) for c in fresh]

        save(checkpoint, {'iteration': i, 'target': target, 'records': records,
                          'deck_keys': deck_keys, 'history': history,
                          'heuristic_state': heuristic.state()})
        with open(deck_out, 'w') as handle:
            handle.write(dumps_deck([records[deck_keys[j]]['card'] for j in shipped]))

    print(f"  stopped at iteration {history[-1]['iteration'] if history else start}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cards', type=int, default=120, help='Deck size to ship')
    p.add_argument('--bench', type=int, default=12,
                   help='Extra cards carried and measured but not shipped')
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
    p.add_argument('--calibration-rounds', type=int, default=1,
                   help='Roster rebuilds before the main loop. More than 1 oscillates.')
    p.add_argument('--init-from', default='',
                   help="Another run's checkpoint to borrow the heuristic from")
    p.add_argument('--checkpoint', default='')
    p.add_argument('--deck-out', default='')
    a = p.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    checkpoint = a.checkpoint or 'data/grow.json'
    deck_out = a.deck_out or 'decks/grown.json'
    os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
    run(a.cards, a.bench, a.target, a.iterations, a.replace, a.sigma, a.candidates,
        a.pool, a.hands, a.rounds, a.placements, checkpoint, deck_out, a.seed,
        a.init_from, a.calibration_rounds)


if __name__ == '__main__':
    main()
