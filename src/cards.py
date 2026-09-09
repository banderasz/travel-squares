"""The card model: a Card is a 2x2 grid of Quarters, each holding symbols.

Kept separate from deck generation and from deck I/O because every module
needs it — generation, simulation, analysis and board reconstruction alike.
"""
import collections
from collections import Counter
from itertools import combinations
from random import choices
from typing import List, Dict, Tuple

from src.symbols import Symbols

class Quarter:
    def __init__(self, symbols: List[Symbols]):
        self.symbols = symbols


    @staticmethod
    def generate_quarter() -> "Quarter":
        """Four slots drawn independently by symbol weight.

        There used to be a rule here rejecting any quarter with more than one
        parrot, rerolling the whole quarter. It capped parrot at 4 per card and
        left it about 6% rarer than its weight of 2 implies. Removed, so every
        symbol now appears at its stated rate.
        """
        return Quarter(choices([symbol for symbol in Symbols],
                               weights=[symbol.weight for symbol in Symbols], k=4))

    def is_empty(self):
        return not [symbol for symbol in self.symbols if symbol != Symbols.NOTHING]

    def __str__(self):
        return str([symbol.display for symbol in self.symbols if symbol.display])

    def __repr__(self):
        return str(self)

    def __eq__(self, other: "Quarter"):
        return collections.Counter(self.symbols) == collections.Counter(other.symbols)

class Card:
    def __init__(self, top_left: Quarter = Quarter(list()),
                 top_right: Quarter = Quarter(list()),
                 bottom_left: Quarter = Quarter(list()),
                 bottom_right: Quarter = Quarter(list())):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.used_quarters = len([quarter for quarter in [top_left, top_right, bottom_left, bottom_right] if not quarter.is_empty()])

    def quarters(self) -> Dict[str, Quarter]:
        return {
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_left": self.bottom_left,
            "bottom_right": self.bottom_right
            }

    def quarter_coordinates(self) -> Dict[Tuple[int, int], Quarter]:
        return {
            (0,0): self.top_left,
            (0,1): self.top_right,
            (1,0): self.bottom_left,
            (1,1): self.bottom_right
            }

    def symbols(self) -> List[Symbols]:
        return [*self.top_left.symbols, *self.top_right.symbols, *self.bottom_left.symbols, *self.bottom_right.symbols]

    def generate_permutations(self):
        combinations_3 = list(combinations(self.quarters().items(), 3))
        combinations_2 = [combination for combination in combinations(self.quarters().items(), 2) if
                          len(set(combination[0][0]) & set(combination[1][0])) > 2]
        combinations_1 = list(combinations(self.quarters().items(), 1))
        variations = list(combinations(self.quarters().items(), 4)) + combinations_2 + combinations_3 + combinations_1
        card_variations = [Card.from_items(variation) for variation in variations]
        return card_variations


    @staticmethod
    def generate_card():
        return Card(**{quarter_name: Quarter.generate_quarter() for quarter_name in
                     ["top_left", "top_right", "bottom_left", "bottom_right"]})

    @staticmethod
    def from_items(items: Tuple[Tuple[str, Quarter]]) -> "Card":
        return Card(**{name:quarter for name, quarter in items})

    def __str__(self):
        """Human-readable summary. To serialise a card, use src.deck_io."""
        return "Card(" + ", ".join(
            f"{name}={quarter}" for name, quarter in self.quarters().items()
        ) + ")"

    def __eq__(self, other: "Card"):
        return (self.top_left == other.top_left
                and self.top_right == other.top_right
                and self.bottom_left == other.bottom_left
                and self.bottom_right == other.bottom_right)
