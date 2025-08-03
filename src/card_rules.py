import abc
import collections
import enum
from collections import defaultdict
from typing import Tuple, Dict

from symbols import Symbols
import numpy as np
from scipy import signal

class AbstractCardRule(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate_card_points(card: "Card") -> int:
        pass



class ArrowBonusPoints(AbstractCardRule):

    @staticmethod
    def calculate_card_points(card: "Card") -> np.array:
        independent_quarters_to_count = list(ArrowBonusPoints.same_quarter_arrow_points(card).values())
        extra_symbols = defaultdict(lambda : np.ones(1))
        if independent_quarters_to_count:
            for symbol in Symbols:
                if symbol not in Symbols.arrows():
                    probabilities_to_convolve = [ArrowBonusPoints.add_x_zeros_between_values(symbol.quarter_probabilities, quarter_duplicates - 1) for quarter_duplicates in independent_quarters_to_count]
                    current_probability = probabilities_to_convolve[0]
                    for i in range(1, len(probabilities_to_convolve)):
                        current_probability = np.convolve(current_probability, probabilities_to_convolve[i])
                    extra_symbols[symbol] = current_probability
        return extra_symbols

    @staticmethod
    def same_quarter_arrow_points(card: "Card") -> Dict[Tuple[int, int], int]:
        """
        Extra points if there is one square where multiple arrows are pointing.
        """
        quarter_counter = collections.defaultdict(lambda: 0)
        for coordinate, quarter in card.quarter_coordinates().items():
            if quarter.is_empty():
                quarter_counter[coordinate] += 1
            for symbol in quarter.symbols:
                if symbol in Symbols.arrows():
                    y, x = coordinate
                    arrow_y, arrow_x = Symbols.arrow_coordinate(symbol)
                    quarter_counter[(y + arrow_y, x + arrow_x)] += 1

        return quarter_counter

    @staticmethod
    def add_x_zeros_between_values(original_array: np.ndarray, num_zeros: int) -> np.ndarray:
        new_size = original_array.size + num_zeros * (original_array.size - 1)
        new_array = np.zeros(new_size, dtype=original_array.dtype)
        step = num_zeros + 1
        new_array[::step] = original_array
        return new_array

class GoodQuarterToMultiply(AbstractCardRule):

    @staticmethod
    def calculate_card_points(card: "Card") -> int:
        extra_points = 0
        for name, quarter in card.quarters().items():
            extra_points += GoodQuarterToMultiply.calculate_quarter_points(quarter)
        return extra_points

    @staticmethod
    def calculate_quarter_points(quarter: "Quarter") -> int:
        extra_point = 0
        symbol_set = set(quarter.symbols)
        if len(symbol_set.intersection(set(Symbols.bad_symbols_to_multiply()))) == 0:
            counter = collections.Counter(quarter.symbols)
            for symbol, value in counter.items():
                current_value = symbol.calculate_expected_value(value, 3*4*5+4)
                extra_value = symbol.calculate_expected_value(value*2, 3*4*4+4)
                if extra_value > current_value:
                    extra_point += (extra_value - current_value)

        return extra_point/2


class CardRules(enum.Enum):
    ARROW_BONUS_POINTS = ArrowBonusPoints
    GOOD_QUARTER_TO_MULTIPLY = GoodQuarterToMultiply

    def __init__(self, card_rule: AbstractCardRule):
        self.card_rule = card_rule

    def calculate_points(self, card: "Card") -> int:
        return self.card_rule.calculate_card_points(card)
