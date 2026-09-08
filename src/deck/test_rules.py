from unittest import TestCase

from src.deck.rules import ArrowBonusPoints
from src.cards import Card, Quarter
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

    def test_calculate_card_points_returns_normalised_distributions(self):
        test_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.X, Symbols.STAR, Symbols.DIAMOND]),
                            "top_right": Quarter([Symbols.MOON, Symbols.MOON, Symbols.ARROW_RIGHT]),
                            "bottom_left": Quarter([]),
                            "bottom_right": Quarter([Symbols.ARROW_LEFT, Symbols.ARROW_LEFT])
                            })
        extra_symbols = ArrowBonusPoints.calculate_card_points(test_card)

        assert extra_symbols, "expected a distribution per symbol"
        for symbol, distribution in extra_symbols.items():
            self.assertAlmostEqual(
                distribution.sum(), 1.0, places=9,
                msg=f"distribution for {symbol} is not normalised",
            )
