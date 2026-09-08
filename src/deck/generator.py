import collections
from collections import Counter
from itertools import combinations
from random import random, choices
from typing import List, Dict, Tuple

import numpy as np
from scipy import signal

from src.deck.rules import CardRules, ArrowBonusPoints
from src.symbols import Symbols, NUMBER_OF_SYMBOLS_IN_PLAY
from src.cards import Card, Quarter
from src.deck_io import save_deck
import matplotlib.pyplot as plt

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
    NUMBER_OF_CARDS = 12
    TOTAL_GENERATED_CARDS = 100
    batch_size = 2

    MINIMUM_POINT = 5
    MAXIMUM_POINT = 7

    cards, values = zip(*[CardGenerator().generate_card() for _ in range(TOTAL_GENERATED_CARDS)])

    plt.hist(values, bins=[0+i for i in range(int(min(values)),int(max(values)+1))], edgecolor='black', alpha=0.7)
    plt.show()

    counter = statistics(cards)
    for symbol in Symbols:
        print(f"{str(symbol)}: {round(counter[symbol] / len(cards) / 16 * NUMBER_OF_SYMBOLS_IN_PLAY, 2)} vs {symbol.weight}")

    cards = list(cards)



    batches = [cards[i:i + batch_size] for i in range(0, len(cards), batch_size)]

    # chosen_cards = cards
    chosen_cards = []
    for _ in range(NUMBER_OF_CARDS//batch_size):
        best_card_index = 0
        best_score = 100000000
        for remaining_batch_index in range(len(batches)):
            counter = statistics(chosen_cards + batches[remaining_batch_index])
            deviation_value = deviation_score(counter, len(chosen_cards) + 1)
            if deviation_value < best_score:
                best_score = deviation_value
                best_card_index = remaining_batch_index
        chosen_cards.extend(batches.pop(best_card_index))
        counter = statistics(chosen_cards)
        for symbol in Symbols:
            print(f"{str(symbol)}: {round(counter[symbol] / len(chosen_cards) / 16 * NUMBER_OF_SYMBOLS_IN_PLAY, 2)} vs {symbol.weight}")


    used_quarters = [card.used_quarters for card in chosen_cards]

    symbol_in_card_hist = collections.defaultdict(lambda : [])
    symbol_in_quarter_hist = collections.defaultdict(lambda: [])
    counter_per_card = [Counter(card.symbols()) for card in chosen_cards]
    counter_per_quarter = []
    for card in chosen_cards:
        for quarter in card.quarters().values():
            counter_per_quarter.append(Counter(quarter.symbols))

    for card in counter_per_card:
        for key, value in card.items():
            symbol_in_card_hist[key].append(value)

    for quarter in counter_per_quarter:
        for key, value in quarter.items():
            symbol_in_quarter_hist[key].append(value)


    plt.hist(used_quarters)
    plt.show()

    n_cols = 3
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, (symbol, values) in enumerate(symbol_in_card_hist.items()):
        print(f"Card: {symbol}: {collections.Counter(values)}")
        ax = axes[i]
        ax.hist(values, bins=[0 + i for i in range(int(min(values)), int(max(values) + 2))])
        ax.set_title(symbol.name)

    n_cols = 3
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, (symbol, values) in enumerate(symbol_in_quarter_hist.items()):
        print(f"Quarter: {symbol}: {collections.Counter(values)}")
        ax = axes[i]
        ax.hist(values, bins=[0 + i for i in range(int(min(values)), int(max(values) + 2))])
        ax.set_title(symbol.name)

    plt.show()




    counter = statistics(chosen_cards)
    for symbol in Symbols:
        print(f"{str(symbol)}: {round(counter[symbol]/NUMBER_OF_CARDS/16*NUMBER_OF_SYMBOLS_IN_PLAY,2)} vs {symbol.weight}")

    chosen_values = [CardGenerator.calculate_card_point(card) for card in chosen_cards]
    # plt.hist(chosen_values, bins=[0+i for i in range(int(min(chosen_values)),int(max(chosen_values)+1))], edgecolor='black', alpha=0.7)
    # plt.show()
    save_deck(f"src/cards_{MINIMUM_POINT}_{MAXIMUM_POINT}.json", chosen_cards)


    test_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.CIRCLE, Symbols.CIRCLE, Symbols.CIRCLE]),
                      "top_right": Quarter([]),
                     "bottom_left": Quarter([]),
                      "bottom_right": Quarter([])
                      })
    print(CardGenerator.calculate_card_point(test_card))

    test_card = Card(**{"top_left": Quarter([Symbols.DIAMOND, Symbols.ARROW_RIGHT, Symbols.SUN]),
                      "top_right": Quarter([Symbols.DIAMOND, Symbols.ARROW_LEFT]),
                     "bottom_left": Quarter([Symbols.CIRCLE, Symbols.CIRCLE, Symbols.X]),
                      "bottom_right": Quarter([Symbols.ARROW_UP, Symbols.ARROW_UP, Symbols.X, Symbols.MOON])
                      })
    print(CardGenerator.calculate_card_point(test_card))