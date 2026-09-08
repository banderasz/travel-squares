"""A fittable estimate of what a card is worth.

The simulation is the ground truth but is far too slow to consult while
generating thousands of candidate cards. So generation uses a cheap linear
model over card features, and that model is refitted against measured values —
see src/deck/tune.py for the loop that does it.

This replaces a hand-tuned formula of magic numbers and special cases. Being a
plain linear model means it can actually be fitted, and the coefficients live in
a JSON file rather than in the source.
"""
import json
from collections import Counter
from typing import Dict, List

import numpy as np

from src.symbols import Symbols

QUARTER_NAMES = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
DEFAULT_MODEL_PATH = 'decks/value_model.json'

# Features the value is regressed on. Mostly plain symbol counts, plus two that
# capture the non-linear rules: rats hurt more when spread across quarters, and
# rum only pays once you have three or more.
FEATURE_NAMES = (
    'n_food', 'n_treasure', 'n_weapon', 'n_coin', 'n_parrot',
    'n_rum', 'n_rat', 'n_snake', 'n_mask',
    'n_arrows', 'rat_quarters', 'rum_concentration',
)

_COUNT_FEATURES = {
    'n_food': Symbols.CIRCLE,
    'n_treasure': Symbols.SQUARE,
    'n_weapon': Symbols.STAR,
    'n_coin': Symbols.SUN,
    'n_parrot': Symbols.DIAMOND,
    'n_rum': Symbols.TRIANGLE,
    'n_rat': Symbols.X,
    'n_snake': Symbols.MOON,
    'n_mask': Symbols.SKULL,
}


def as_symbol(value) -> Symbols:
    """Accept either a Symbols member or a name from card JSON."""
    return value if isinstance(value, Symbols) else Symbols.of(value)


def features(card: Dict) -> Dict[str, float]:
    """Feature vector for one card, in the deck JSON shape."""
    quarters = card['card']['quarters']
    per_quarter = {
        name: [as_symbol(s) for s in quarters.get(name, [])]
        for name in QUARTER_NAMES
    }
    counts = Counter(s for syms in per_quarter.values() for s in syms)

    out = {name: counts.get(symbol, 0) for name, symbol in _COUNT_FEATURES.items()}
    out['n_arrows'] = sum(counts.get(a, 0) for a in Symbols.arrows())
    out['rat_quarters'] = sum(1 for syms in per_quarter.values() if Symbols.X in syms)
    # Rum scores nothing below three, then climbs steeply.
    out['rum_concentration'] = max(0, out['n_rum'] - 2)
    return out


def feature_matrix(cards: List[Dict]) -> np.ndarray:
    rows = [features(c) for c in cards]
    return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=np.float64)


class ValueModel:
    """value = intercept + sum(coef_i * feature_i)"""

    def __init__(self, intercept: float = 0.0, coefficients: Dict[str, float] = None):
        self.intercept = intercept
        self.coefficients = coefficients or {name: 0.0 for name in FEATURE_NAMES}

    def predict(self, card: Dict) -> float:
        f = features(card)
        return self.intercept + sum(self.coefficients.get(n, 0.0) * f[n] for n in FEATURE_NAMES)

    def predict_many(self, cards: List[Dict]) -> np.ndarray:
        vec = np.array([self.coefficients.get(n, 0.0) for n in FEATURE_NAMES])
        return feature_matrix(cards) @ vec + self.intercept

    @classmethod
    def fit(cls, cards: List[Dict], measured: np.ndarray, ridge: float = 1.0) -> "ValueModel":
        """Least-squares fit of card features against measured values.

        Ridge-regularised by default. Without it, fitting 13 parameters to a
        deck whose cards were *selected* to have near-identical value gives
        almost no feature variation to work with, and plain least squares
        responds by inventing enormous offsetting coefficients. The penalty is
        not applied to the intercept, which must stay free to track the deck's
        overall level.
        """
        X = feature_matrix(cards)
        y = np.asarray(measured, dtype=np.float64)
        keep = ~np.isnan(y)
        X, y = X[keep], y[keep]

        centre = X.mean(axis=0)
        Xc = X - centre
        n_features = Xc.shape[1]
        gram = Xc.T @ Xc + ridge * np.eye(n_features)
        slopes = np.linalg.solve(gram, Xc.T @ (y - y.mean()))
        intercept = float(y.mean() - slopes @ centre)
        return cls(intercept, dict(zip(FEATURE_NAMES, slopes.tolist())))

    def score(self, cards: List[Dict], measured: np.ndarray) -> Dict[str, float]:
        """How well the model reproduces the measurements."""
        y = np.asarray(measured, dtype=np.float64)
        keep = ~np.isnan(y)
        pred, y = self.predict_many(cards)[keep], y[keep]
        resid = y - pred
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return {
            'mae': float(np.abs(resid).mean()),
            'rmse': float(np.sqrt((resid ** 2).mean())),
            'r2': float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float('nan'),
            'bias': float(resid.mean()),
        }

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        with open(path, 'w') as handle:
            json.dump({'intercept': self.intercept, 'coefficients': self.coefficients},
                      handle, indent=2)
            handle.write('\n')

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "ValueModel":
        with open(path) as handle:
            data = json.load(handle)
        return cls(data['intercept'], data['coefficients'])

    def __str__(self):
        terms = sorted(self.coefficients.items(), key=lambda kv: -abs(kv[1]))
        body = "  ".join(f"{n}={v:+.2f}" for n, v in terms)
        return f"intercept={self.intercept:.2f}  {body}"
