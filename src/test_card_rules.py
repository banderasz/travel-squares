from unittest import TestCase

from src.card_rules import ArrowBonusPoints
from src.square_card_generator import Card, Quarter
from src.symbols import Symbols


class TestArrowBonusPoints(TestCase):
    def test_calculate_card_points(self):
        test_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.X, Symbols.STAR, Symbols.DIAMOND]),
                            "top_right": Quarter([Symbols.MOON, Symbols.MOON, Symbols.ARROW_RIGHT]),
                            "bottom_left": Quarter([]),
                            "bottom_right": Quarter([Symbols.ARROW_LEFT, Symbols.ARROW_LEFT])
                            })
        expected_quarter_map = {(1, 0): 3, (0, 2): 1 }
        quarter_map = ArrowBonusPoints.same_quarter_arrow_points(test_card)
        assert quarter_map == expected_quarter_map

    def test(self):
        test_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.X, Symbols.STAR, Symbols.DIAMOND]),
                            "top_right": Quarter([Symbols.MOON, Symbols.MOON, Symbols.ARROW_RIGHT]),
                            "bottom_left": Quarter([]),
                            "bottom_right": Quarter([Symbols.ARROW_LEFT, Symbols.ARROW_LEFT])
                            })
        extra_symbols = ArrowBonusPoints.calculate_card_points(test_card)
        assert False
