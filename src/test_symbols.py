"""Tests for the symbol naming contract described in src/symbols.py.

These exist because the contract has already been broken once: a rename left
most of the repo reading card files whose symbol names no longer matched, and
nothing failed.
"""
import os
import re
import unittest
from unittest import TestCase

from src.symbols import LEGACY_NAMES, NUMBER_OF_SYMBOLS_IN_PLAY, Symbols

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_MAIN = os.path.join(REPO_ROOT, "pirate_sim_rs", "src", "main.rs")

# The rename that prompted all this. Verified against the diff between commits
# 8151842 and d28967b, in which only the display strings changed.
PIRATE_NAMES = {
    "anchor": Symbols.CIRCLE,
    "map": Symbols.SQUARE,
    "rum": Symbols.TRIANGLE,
    "spyglass": Symbols.STAR,
    "rat": Symbols.X,
    "shark": Symbols.MOON,
    "parrot": Symbols.DIAMOND,
    "kraken": Symbols.SKULL,
    "coin": Symbols.SUN,
}


class TestSymbolResolution(TestCase):
    def test_resolves_stable_ids(self):
        for symbol in Symbols:
            self.assertIs(Symbols.of(symbol.name), symbol)

    def test_resolves_display_names(self):
        for symbol in Symbols:
            if symbol.display:
                self.assertIs(Symbols.of(symbol.display), symbol)

    def test_resolves_legacy_pirate_names(self):
        """The detection model emits these as class labels; they must keep working."""
        for name, expected in PIRATE_NAMES.items():
            self.assertIs(Symbols.of(name), expected, f"{name} should resolve to {expected.name}")

    def test_resolution_is_case_and_space_insensitive(self):
        self.assertIs(Symbols.of("  KrAkEn "), Symbols.SKULL)

    def test_unknown_name_raises_with_guidance(self):
        with self.assertRaises(ValueError) as ctx:
            Symbols.of("banana")
        self.assertIn("banana", str(ctx.exception))
        self.assertIn("Valid names are", str(ctx.exception))

    def test_every_legacy_entry_names_a_real_member(self):
        for member_name in LEGACY_NAMES:
            self.assertIn(member_name, Symbols.__members__)


class TestSymbolTables(TestCase):
    def test_points_tables_are_13_long(self):
        """Points are multiplied elementwise against 13-long probability vectors."""
        for symbol in Symbols:
            self.assertEqual(len(symbol.points), 13, f"{symbol.name} has a wrong-length table")

    def test_weights_sum_to_symbols_in_play(self):
        self.assertEqual(sum(s.weight for s in Symbols), NUMBER_OF_SYMBOLS_IN_PLAY)

    def test_scoring_symbols_are_the_non_arrow_non_empty_ones(self):
        scoring = [s for s in Symbols if s.value_symbol()]
        self.assertEqual(len(scoring), 9)
        self.assertNotIn(Symbols.NOTHING, scoring)
        for arrow in Symbols.arrows():
            self.assertNotIn(arrow, scoring)


class TestSimulationWireFormat(TestCase):
    """The simulation's symbol indices are baked into ~200 GB of placement
    shards under data/ and into pirate_sim_rs. They must never move."""

    EXPECTED_INDICES = {
        "anchor": 0, "shark": 1, "rat": 2, "kraken": 3, "map": 4,
        "coin": 5, "rum": 6, "parrot": 7, "spyglass": 8,
        "arrow_up": 9, "arrow_down": 10, "arrow_left": 11, "arrow_right": 12,
    }

    def test_indices_match_the_original_order(self):
        from src.simulation.sim import SYM_MAP

        for legacy_name, index in self.EXPECTED_INDICES.items():
            self.assertEqual(
                SYM_MAP[Symbols.of(legacy_name).name], index,
                f"{legacy_name} moved from wire index {index}",
            )

    def test_scoring_matches_symbols(self):
        from src.simulation.sim import MAX_CT, SCORE_DEFAULT, SYM_ORDER

        for symbol in SYM_ORDER[:9]:
            self.assertEqual(SCORE_DEFAULT[symbol.name], symbol.points[:MAX_CT + 1])


@unittest.skipUnless(os.path.exists(RUST_MAIN), "pirate_sim_rs not present")
class TestRustParity(TestCase):
    """The Rust port keeps its own copies of the scoring table and the name
    mapping. Nothing at runtime forces them to agree with src/symbols.py, and a
    silent divergence would make the two simulators disagree, so check here."""

    @classmethod
    def setUpClass(cls):
        with open(RUST_MAIN) as handle:
            cls.source = handle.read()

    def test_scoring_table_matches(self):
        from src.simulation.sim import MAX_CT, SYM_ORDER

        block = re.search(r"const SCORING[^=]*=\s*\[(.*?)\n\];", self.source, re.S)
        self.assertIsNotNone(block, "could not locate SCORING in main.rs")
        rows = re.findall(r"\[([-0-9.,\s]+)\]", block.group(1))
        self.assertEqual(len(rows), 9, "expected 9 scoring rows")

        for index, row in enumerate(rows):
            values = [float(v) for v in row.replace("\n", "").split(",") if v.strip()]
            expected = [float(v) for v in SYM_ORDER[index].points[:MAX_CT + 1]]
            self.assertEqual(
                values, expected,
                f"Rust SCORING row {index} disagrees with {SYM_ORDER[index].name}",
            )

    def test_name_mapping_matches(self):
        from src.simulation.sim import SYM_MAP

        body = re.search(r"fn sym_id.*?\{(.*?)\n\}", self.source, re.S).group(1)
        arms = re.findall(r'((?:"[A-Za-z_]+"\s*\|?\s*)+)=>\s*(\d+)', body)
        self.assertTrue(arms, "could not parse sym_id arms")

        for names, index in arms:
            for name in re.findall(r'"([A-Za-z_]+)"', names):
                self.assertEqual(
                    SYM_MAP[Symbols.of(name).name], int(index),
                    f"Rust maps {name!r} to {index}, Python disagrees",
                )
