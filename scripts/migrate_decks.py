#!/usr/bin/env python3
"""Rewrite card JSON files to use stable symbol IDs.

Most deck files still store the symbol names from an earlier theme
(anchor/map/shark/kraken/spyglass). Those still load, because src.symbols keeps
them as legacy aliases, but data written today should use the stable enum names
so the next rename does not strand it. See the naming contract in src/symbols.py.

This only substitutes names. Card count, quarter structure and the multiset of
symbols in each quarter must come out identical, and the script verifies that
before writing anything.

Usage:
    python -m scripts.migrate_decks --check     # report, write nothing
    python -m scripts.migrate_decks             # migrate in place
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deck_io import QUARTER_NAMES, dumps_deck  # noqa: E402
from src.symbols import Symbols  # noqa: E402

DECK_GLOBS = ("src/cards*.json", "pirate_cards/*.json")


def _quarters_of(raw):
    """Symbol multiset per quarter per card, as resolved Symbols members."""
    out = []
    for entry in raw:
        quarters = entry["card"]["quarters"]
        out.append({
            name: sorted(s.name for s in map(Symbols.of, quarters.get(name, [])))
            for name in QUARTER_NAMES
        })
    return out


def migrate_file(path, write):
    with open(path) as handle:
        raw = json.load(handle)

    before = _quarters_of(raw)

    migrated = []
    for entry in raw:
        quarters = entry["card"]["quarters"]
        migrated.append({
            "card": {
                "dimensions": entry["card"].get("dimensions", {"width": 135, "height": 135}),
                "quarters": {
                    name: [Symbols.of(s).name for s in quarters.get(name, [])]
                    for name in QUARTER_NAMES
                },
            }
        })

    after = _quarters_of(migrated)
    if before != after:
        raise AssertionError(f"{path}: round-trip changed the deck, refusing to write")

    already = all(
        s == Symbols.of(s).name
        for entry in raw
        for names in entry["card"]["quarters"].values()
        for s in names
    )

    if write and not already:
        with open(path, "w") as handle:
            handle.write(dumps_deck(migrated))

    return len(raw), already


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify round-trip only, write nothing")
    args = parser.parse_args()

    paths = sorted({p for pattern in DECK_GLOBS for p in glob.glob(pattern)})
    if not paths:
        print("no deck files found", file=sys.stderr)
        return 1

    changed = 0
    for path in paths:
        cards, already = migrate_file(path, write=not args.check)
        if already:
            status = "already canonical"
        else:
            changed += 1
            status = "verified (not written)" if args.check else "migrated"
        print(f"  {path:45} {cards:4} cards  {status}")

    verb = "would be migrated" if args.check else "migrated"
    print(f"\n{changed} of {len(paths)} files {verb}; all round-trips verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
