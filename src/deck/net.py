#!/usr/bin/env python3
"""A differentiable stand-in for the simulator: 6 cards -> what the hand scores.

Why a hand and not a card: adding up the six cards' individual values explains
only about 15% of what a hand actually scores. The rest is the cards interacting
— which is also what makes the draft a real decision. A per-card value cannot
represent it; a model over the whole hand can.

Why a hand and not a deck: a hand's score depends on those six cards and nothing
else, so the hand is the complete unit of physics. A deck only decides which
hands occur, and that you can compute by sampling. Labelled hands also arrive
at ~200,000/sec while labelled decks take seconds each.

The model is permutation invariant — a hand is a set, so the six cards are
encoded separately, allowed to attend to each other, then pooled.

Note this is NOT faster than the simulator; both are microseconds per hand. Its
point is being differentiable, so card contents can be optimised by gradient
descent instead of by sampling and filtering.

    python -m src.deck.net --train 40000
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from src.deck.simulation import evaluate_placements_batch, get_paths, load_cards
from src.deck.tune import random_card
from src.deck_io import dumps_deck
from src.symbols import Symbols

HAND = 6
QUARTERS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
SYMBOL_LIST = [s for s in Symbols if s is not Symbols.NOTHING]
SYM_INDEX = {s: i for i, s in enumerate(SYMBOL_LIST)}
CARD_DIM = len(QUARTERS) * len(SYMBOL_LIST)      # 4 x 13


def encode_card(card):
    """A card as counts of each symbol in each quarter."""
    out = np.zeros((len(QUARTERS), len(SYMBOL_LIST)), dtype=np.float32)
    for qi, name in enumerate(QUARTERS):
        for raw in card['card']['quarters'].get(name, []):
            out[qi, SYM_INDEX[Symbols.of(raw)]] += 1
    return out.reshape(-1)


class HandNet(nn.Module):
    def __init__(self, width=128, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(CARD_DIM, width), nn.GELU(), nn.Linear(width, width))
        enc = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True,
                                         dropout=0.0, norm_first=True)
        self.mix = nn.TransformerEncoder(enc, layers)   # cards see each other
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x):                # x: (batch, 6, CARD_DIM)
        h = self.mix(self.embed(x))
        return self.head(h.mean(dim=1)).squeeze(-1)   # mean-pool = order-independent


class Oracle:
    """Labels hands using the real simulator."""

    def __init__(self, pool_size=3000, placements=200, seed=0, lo=0.50, hi=0.85):
        self.cards = [random_card() for _ in range(pool_size)]
        path = '/tmp/_net_pool.json'
        with open(path, 'w') as handle:
            handle.write(dumps_deck(self.cards))
        self.cd = load_cards(path)
        self.paths = get_paths()
        self.encoded = np.stack([encode_card(c) for c in self.cards])
        self.placements, self.lo, self.hi = placements, lo, hi
        self.rng = np.random.default_rng(seed)

    def batch(self, n_hands):
        n, m = n_hands, self.placements
        hands = np.array([self.rng.choice(len(self.cards), HAND, replace=False)
                          for _ in range(n)], dtype=np.int32)
        ids = np.repeat(hands, m, axis=0)
        rots = self.rng.integers(0, 4, size=(len(ids), HAND)).astype(np.int8)
        pth = self.rng.integers(0, len(self.paths), size=len(ids)).astype(np.int32)
        tb = self.rng.integers(0, 32, size=len(ids)).astype(np.int8)
        s = np.asarray(evaluate_placements_batch(ids, rots, pth, tb,
                                                 self.paths, self.cd)).reshape(n, m)
        s.sort(axis=1)
        y = s[:, int(self.lo * m):int(self.hi * m)].mean(axis=1)
        return torch.from_numpy(self.encoded[hands]), torch.from_numpy(y.astype(np.float32))


def train(steps=40000, batch=256, lr=3e-4, seed=0, out='data/handnet.pt'):
    torch.manual_seed(seed)
    oracle = Oracle(seed=seed)
    net = HandNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    xv, yv = oracle.batch(4000)                    # held-out, never trained on
    print(f"  training on freshly simulated hands; {sum(p.numel() for p in net.parameters()):,} parameters\n")
    print(f"  {'step':>7} {'train mse':>10} {'val R2':>8} {'val mae':>8} {'elapsed':>8}")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = oracle.batch(batch)                 # every batch is new data
        loss = nn.functional.mse_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                pred = net(xv)
                r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
                mae = (yv - pred).abs().mean()
            net.train()
            print(f"  {step:>7} {loss.item():>10.3f} {r2.item():>8.3f} {mae.item():>8.3f} "
                  f"{time.time()-t0:>7.0f}s")
    torch.save({'state': net.state_dict()}, out)
    print(f"\n  saved -> {out}")
    return net, oracle


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', type=int, default=20000, help='training steps')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--out', default='data/handnet.pt')
    a = p.parse_args()
    train(a.train, a.batch, out=a.out)


if __name__ == '__main__':
    main()
