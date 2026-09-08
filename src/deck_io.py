"""Reading and writing decks of cards.

This is the single I/O boundary for card JSON. Everything that loads a deck
should come through here rather than parsing the files itself, because this is
where symbol names get resolved.

Symbols are **written** using their stable enum name (``CIRCLE``, ``SQUARE``, …)
and **read** through :meth:`Symbols.of`, which also accepts the current display
names and every historical name. That combination is what makes a future rename
free: existing files keep loading, and newly written files are not stamped with
a name that is expected to change. See the module docstring in ``src.symbols``.

The on-disk shape is preserved from the original generator so existing readers
(``src/simulation/sim.py``, the root analysis scripts) keep working::

    [
      {"card": {"dimensions": {"width": 135, "height": 135},
                "quarters": {"top_left": ["CIRCLE", "MOON"], ...}}}
    ]
"""
import json
from typing import Dict, List

from src.cards import Card, Quarter
from src.symbols import Symbols

QUARTER_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
DEFAULT_DIMENSIONS = {"width": 135, "height": 135}


def parse_quarter(names: List[str]) -> Quarter:
    """Build a Quarter from a list of symbol names in any supported vocabulary."""
    return Quarter([Symbols.of(name) for name in names])


def quarter_to_names(quarter: Quarter) -> List[str]:
    """Serialise a Quarter to stable symbol IDs.

    Empty slots are dropped. Note the test is against ``Symbols.NOTHING`` and not
    against a falsy name: NOTHING's *display* is the empty string, but its stable
    name is the truthy "NOTHING", so a truthiness check would write it out.
    """
    return [s.name for s in quarter.symbols if s is not Symbols.NOTHING]


def card_to_dict(card: Card) -> Dict:
    return {
        "card": {
            "dimensions": dict(DEFAULT_DIMENSIONS),
            "quarters": {
                name: quarter_to_names(quarter)
                for name, quarter in card.quarters().items()
            },
        }
    }


def card_from_dict(entry: Dict) -> Card:
    quarters = entry["card"]["quarters"]
    return Card(**{name: parse_quarter(quarters.get(name, [])) for name in QUARTER_NAMES})


def dumps_deck(entries: List[Dict]) -> str:
    """Render deck entries as JSON, one card per block and one line per quarter set.

    Deliberately not plain ``json.dump(indent=…)``: that puts every symbol on its
    own line, which turns a symbol rename into a diff of tens of thousands of
    lines and makes these files impossible to review. Keeping each card's
    quarters on a single line means a rename shows up as one changed line per card.
    """
    blocks = []
    for entry in entries:
        card = entry["card"]
        dimensions = json.dumps(card.get("dimensions") or DEFAULT_DIMENSIONS)
        quarters = json.dumps(card["quarters"])
        blocks.append(
            "  {\n"
            '    "card": {\n'
            f'      "dimensions": {dimensions},\n'
            f'      "quarters": {quarters}\n'
            "    }\n"
            "  }"
        )
    return "[\n" + ",\n".join(blocks) + "\n]\n"


def load_deck(path: str) -> List[Card]:
    """Load a deck, accepting stable IDs, display names or legacy names."""
    with open(path) as handle:
        raw = json.load(handle)
    return [card_from_dict(entry) for entry in raw]


def save_deck(path: str, cards: List[Card]) -> None:
    """Write a deck using stable symbol IDs."""
    with open(path, "w") as handle:
        handle.write(dumps_deck([card_to_dict(card) for card in cards]))
