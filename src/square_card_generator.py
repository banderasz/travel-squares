import collections
from collections import Counter
from itertools import combinations
from random import random, choices
from typing import List, Dict, Tuple

import numpy as np
from scipy import signal

from card_rules import CardRules, ArrowBonusPoints
from symbols import Symbols, NUMBER_OF_SYMBOLS_IN_PLAY
import matplotlib.pyplot as plt



class Quarter:
    def __init__(self, symbols: List[Symbols]):
        self.symbols = symbols


    @staticmethod
    def generate_quarter() -> "Quarter":
        symbols = choices([symbol for symbol in Symbols], weights=[symbol.weight for symbol in Symbols], k=4)
        while Counter(symbols)[Symbols.DIAMOND] > 1:
            symbols = choices([symbol for symbol in Symbols], weights=[symbol.weight for symbol in Symbols], k=4)
        return Quarter(symbols)

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
        return """
        {
            "card": {
                "dimensions": 
                {
                    "width": 135,
                    "height": 135
                },
                "quarters": <quarters>
            }
        }
        """.replace("<quarters>", str(self.quarters()).replace("'", '"'))

    def __eq__(self, other: "Card"):
        return (self.top_left == other.top_left
                and self.top_right == other.top_right
                and self.bottom_left == other.bottom_left
                and self.bottom_right == other.bottom_right)


class CardGenerator:
    def __init__(self):
        self.minimum_card_point, self.maximum_card_point = MINIMUM_POINT, MAXIMUM_POINT

    def generate_card(self) -> Tuple[Card, int]:
        card = Card.generate_card()
        card_point = self.calculate_card_point(card)
        while not self.minimum_card_point <= card_point <= self.maximum_card_point:
            card = Card.generate_card()
            card_point = self.calculate_card_point(card)
        return card, card_point


    @staticmethod
    def calculate_card_point(card: Card) -> int:
        card_variations = card.generate_permutations()
        replaced_arrow_variations = [CardGenerator.replace_internal_arrows(card) for card in card_variations]
        points_of_variation = [CardGenerator.calculate_card_points(card) for card in replaced_arrow_variations]
        max_point = max(points_of_variation)
        chosen_index = points_of_variation.index(max(points_of_variation))
        card.used_quarters = card_variations[chosen_index].used_quarters
        return max_point

    @staticmethod
    def replace_internal_arrows(card: Card):
        coordinate_to_quarter = {(0,0): "top_left",
                                 (0,1): "top_right",
                                 (1,0): "bottom_left",
                                 (1,1): "bottom_right"}
        internal_card_coordinates = [item[0] for item in card.quarter_coordinates().items() if item[1].symbols]

        new_card = collections.defaultdict(lambda: Quarter([]))
        quarter_coordinates = card.quarter_coordinates()
        for coordinate, quarter in quarter_coordinates.items():
            for symbol in quarter.symbols:
                if symbol.name in [arrow.name for arrow in Symbols.arrows()]:
                    y, x = coordinate
                    arrow_y, arrow_x = Symbols.arrow_coordinate(symbol)
                    new_y, new_x = y + arrow_y, x + arrow_x
                    if (new_y, new_x) in internal_card_coordinates:
                        for copied_symbol in quarter_coordinates[(new_y, new_x)].symbols:
                            if copied_symbol not in Symbols.arrows() and copied_symbol is not Symbols.NOTHING:
                                new_card[coordinate_to_quarter[coordinate]].symbols.append(copied_symbol)
                    else:
                        new_card[coordinate_to_quarter[coordinate]].symbols.append(symbol)
                else:
                    new_card[coordinate_to_quarter[coordinate]].symbols.append(symbol)
        return Card.from_items(new_card.items())

    @staticmethod
    def calculate_card_points(card: Card) -> int:
        points = 0
        counter = Counter(card.symbols())
        random_quarter_probs = ArrowBonusPoints.calculate_card_points(card)
        for symbol in Symbols:
            if symbol is not Symbols.NOTHING and symbol not in Symbols.arrows():
                symbol_prob = np.zeros(counter[symbol] + 1)
                symbol_prob[-1] = 1
                total_symbol_prob = np.convolve(symbol_prob, random_quarter_probs[symbol])
                if card.used_quarters == 4 or card.used_quarters == 3:
                    total_symbol_prob = total_symbol_prob
                elif card.used_quarters == 2:
                    total_symbol_prob = signal.deconvolve(total_symbol_prob, symbol.calculate_half_quarter_probability())[0]
                elif card.used_quarters == 1:
                    total_symbol_prob = signal.deconvolve(total_symbol_prob, symbol.quarter_probabilities)[0]
                points += symbol.calculate_expected_value_from_dist(total_symbol_prob)
        return points


def statistics(cards):
    list_of_symbols = []
    for card in cards:
        for quarter in card.quarters().values():
            list_of_symbols.extend(quarter.symbols)
    return Counter(list_of_symbols)



def similar_weights(counter: Dict[Symbols, int]) -> bool:
    for shape, value in counter.items():
        real_weight = round(counter[shape]/NUMBER_OF_CARDS/16*NUMBER_OF_SYMBOLS_IN_PLAY,2)
        if shape not in [Symbols.ARROW_LEFT, Symbols.ARROW_UP, Symbols.ARROW_DOWN, Symbols.ARROW_RIGHT] and abs(shape.weight - real_weight) > min(1,real_weight*0.1):
            return False
    return True

def deviation_score(counter: Dict[Symbols, int], selected_cards: int):
    real_arrow_weight = sum(
        [round(counter[arrow] / selected_cards / 16 * NUMBER_OF_SYMBOLS_IN_PLAY, 2) for arrow in Symbols.arrows()])
    expected_arrow_weight = sum([arrow.weight for arrow in Symbols.arrows()])
    deviation = (expected_arrow_weight - real_arrow_weight) ** 2
    for symbol in Symbols:
        if symbol not in Symbols.arrows():
            real_weight = round(counter[symbol] / selected_cards / 16 * NUMBER_OF_SYMBOLS_IN_PLAY, 2)
            deviation += (symbol.weight - real_weight) ** 2
    return deviation


if __name__ == "__main__":
    NUMBER_OF_CARDS = 120
    TOTAL_GENERATED_CARDS = 2000

    MINIMUM_POINT = 18
    MAXIMUM_POINT = 22

    cards, values = zip(*[CardGenerator().generate_card() for _ in range(TOTAL_GENERATED_CARDS)])

    plt.hist(values, bins=[0+i for i in range(int(min(values)),int(max(values)+1))], edgecolor='black', alpha=0.7)
    plt.show()

    counter = statistics(cards)
    for symbol in Symbols:
        print(f"{str(symbol)}: {round(counter[symbol] / len(cards) / 16 * NUMBER_OF_SYMBOLS_IN_PLAY, 2)} vs {symbol.weight}")

    cards = list(cards)
    chosen_cards = []
    for _ in range(NUMBER_OF_CARDS):
        best_card_index = 0
        best_score = 100000000
        for remaining_card_index in range(len(cards)):
            counter = statistics(chosen_cards + [cards[remaining_card_index]])
            deviation_value = deviation_score(counter, len(chosen_cards) + 1)
            if deviation_value < best_score:
                best_score = deviation_value
                best_card_index = remaining_card_index
        chosen_cards.append(cards.pop(best_card_index))
        # counter = statistics(chosen_cards)
        # for symbol in Symbols:
        #     print(f"{str(symbol)}: {round(counter[symbol] / len(chosen_cards) / 16 * NUMBER_OF_SYMBOLS_IN_PLAY, 2)} vs {symbol.weight}")





    counter = statistics(chosen_cards)
    for symbol in Symbols:
        print(f"{str(symbol)}: {round(counter[symbol]/NUMBER_OF_CARDS/16*NUMBER_OF_SYMBOLS_IN_PLAY,2)} vs {symbol.weight}")

    chosen_values = [CardGenerator.calculate_card_point(card) for card in chosen_cards]
    # plt.hist(chosen_values, bins=[0+i for i in range(int(min(chosen_values)),int(max(chosen_values)+1))], edgecolor='black', alpha=0.7)
    # plt.show()
    with open("cards.json", "w") as file:
        file.write("[\n")
        file.write(",\n".join([str(card) for card in chosen_cards]))
        file.write("\n]")


    test_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.X, Symbols.STAR, Symbols.DIAMOND]),
                      "top_right": Quarter([Symbols.MOON, Symbols.MOON, Symbols.ARROW_RIGHT]),
                     "bottom_left": Quarter([Symbols.SQUARE, Symbols.X]),
                      "bottom_right": Quarter([Symbols.ARROW_LEFT, Symbols.ARROW_LEFT])
                      })
    print(CardGenerator.calculate_card_point(test_card))

    # test_card = Card(**{"top_left": Quarter([Symbols.STAR, Symbols.SQUARE]),
    #                   "top_right": Quarter([Symbols.SQUARE, Symbols.TRIANGLE]),
    #                  "bottom_left": Quarter([Symbols.STAR, Symbols.DIAMOND]),
    #                   "bottom_right": Quarter([Symbols.CIRCLE, Symbols.CIRCLE, Symbols.X])
    #                   })
    # print(CardGenerator.calculate_card_point(test_card))