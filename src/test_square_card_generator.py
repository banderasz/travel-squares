from unittest import TestCase

from src.square_card_generator import Card, Quarter, CardGenerator
from src.symbols import Symbols


class TestCardGenerator(TestCase):
    def test_replace_internal_arrows(self):
        test_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.X, Symbols.STAR, Symbols.DIAMOND]),
                            "top_right": Quarter([Symbols.MOON, Symbols.MOON, Symbols.ARROW_RIGHT]),
                            "bottom_left": Quarter([Symbols.SQUARE, Symbols.ARROW_DOWN]),
                            "bottom_right": Quarter([Symbols.ARROW_LEFT, Symbols.ARROW_LEFT])
                            })
        expected_replaced_card = Card(**{"top_left": Quarter([Symbols.CIRCLE, Symbols.X, Symbols.STAR, Symbols.DIAMOND]),
                            "top_right": Quarter([Symbols.MOON, Symbols.MOON, Symbols.ARROW_RIGHT]),
                            "bottom_left": Quarter([Symbols.SQUARE, Symbols.ARROW_DOWN]),
                            "bottom_right": Quarter([Symbols.SQUARE, Symbols.SQUARE])
                            })
        replaced_card = CardGenerator.replace_internal_arrows(test_card)
        assert replaced_card == expected_replaced_card
