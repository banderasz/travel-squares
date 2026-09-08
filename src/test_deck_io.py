"""Tests for deck serialisation.

The point of deck_io is that a deck written today survives the next rename, and
that decks written before the last one still load. Both are checked here.
"""
import json
import os
import tempfile
from unittest import TestCase

from src.deck_io import card_from_dict, card_to_dict, load_deck, save_deck
from src.cards import Card, Quarter
from src.symbols import Symbols

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_card():
    return Card(
        top_left=Quarter([Symbols.CIRCLE, Symbols.MOON]),
        top_right=Quarter([Symbols.ARROW_RIGHT]),
        bottom_left=Quarter([]),
        bottom_right=Quarter([Symbols.SKULL, Symbols.SUN, Symbols.DIAMOND]),
    )


class TestRoundTrip(TestCase):
    def test_save_then_load_preserves_the_deck(self):
        deck = [sample_card(), sample_card()]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deck.json")
            save_deck(path, deck)
            reloaded = load_deck(path)
        self.assertEqual(len(reloaded), len(deck))
        for original, loaded in zip(deck, reloaded):
            self.assertEqual(original, loaded)

    def test_written_files_use_stable_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deck.json")
            save_deck(path, [sample_card()])
            raw = json.load(open(path))
        names = [n for q in raw[0]["card"]["quarters"].values() for n in q]
        self.assertIn("CIRCLE", names)
        for name in names:
            self.assertIn(name, Symbols.__members__,
                          f"{name!r} is not a stable ID; a rename would strand this file")

    def test_empty_slots_are_not_written_as_nothing(self):
        """NOTHING's display is "" but its stable name is truthy, so a naive
        truthiness filter would emit "NOTHING" into every quarter."""
        card = Card(top_left=Quarter([Symbols.CIRCLE, Symbols.NOTHING]))
        names = card_to_dict(card)["card"]["quarters"]["top_left"]
        self.assertEqual(names, ["CIRCLE"])


class TestVocabularyCompatibility(TestCase):
    def test_reads_legacy_pirate_names(self):
        entry = {"card": {"dimensions": {}, "quarters": {
            "top_left": ["anchor", "shark"], "top_right": ["kraken"],
            "bottom_left": [], "bottom_right": ["spyglass", "map"]}}}
        card = card_from_dict(entry)
        self.assertEqual(card.top_left.symbols, [Symbols.CIRCLE, Symbols.MOON])
        self.assertEqual(card.bottom_right.symbols, [Symbols.STAR, Symbols.SQUARE])

    def test_reads_current_display_names(self):
        entry = {"card": {"dimensions": {}, "quarters": {
            "top_left": ["food", "snake"], "top_right": [],
            "bottom_left": [], "bottom_right": []}}}
        card = card_from_dict(entry)
        self.assertEqual(card.top_left.symbols, [Symbols.CIRCLE, Symbols.MOON])

    def test_all_checked_in_decks_load(self):
        import glob
        paths = sorted(
            glob.glob(os.path.join(REPO_ROOT, "decks", "*.json"))
            + glob.glob(os.path.join(REPO_ROOT, "pirate_cards", "*.json"))
        )
        self.assertTrue(paths, "expected some deck files in the repo")
        for path in paths:
            with self.subTest(deck=os.path.basename(path)):
                self.assertTrue(load_deck(path), f"{path} loaded empty")
