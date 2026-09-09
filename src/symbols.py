"""Symbol definitions — the single source of truth for what a symbol is.

Naming contract
---------------
Each symbol has three kinds of name, and they must not be confused:

1. The **enum member name** (``CIRCLE``, ``SQUARE``, …) is a permanent, opaque
   identifier. It is what gets written to disk. Treat it as an arbitrary ID, not
   as a description of the artwork — never rename it.
2. ``display`` is the current human-facing name (``food``, ``treasure``, …). It
   belongs to the current theme and is expected to change.
3. ``LEGACY_NAMES`` holds every name a symbol has previously had, plus any label
   an external system emits for it.

To rename a symbol: change its ``display`` and append the previous value to
``LEGACY_NAMES``. Nothing else needs to change, and no stored data is invalidated.

Read names with :meth:`Symbols.of`, which accepts any of the three forms. Write
names as ``symbol.name`` (the stable ID). Never key data off ``display``.

This matters because it has already gone wrong once: a previous rename
(anchor->food, map->treasure, shark->snake, kraken->mask, spyglass->weapon)
left most of the repo reading card files whose symbol names no longer matched,
with no error raised anywhere.
"""
import enum
from typing import Dict, List, Tuple

# Total symbol weight across a card's 16 slots, NOTHING included.
NUMBER_OF_SYMBOLS_IN_PLAY = 96

# Every name a symbol has previously been known by, keyed by its stable enum name.
# These stay here forever: old card JSON files still use them, and the trained
# object-detection model emits them as its class labels, so they cannot be
# retired without retraining it.
#
# When you rename a symbol, append its previous `display` value here.
LEGACY_NAMES: Dict[str, Tuple[str, ...]] = {
    "CIRCLE":   ("anchor",),
    "SQUARE":   ("map", "gem"),        # gem: the artwork asset name
    "TRIANGLE": (),           # rum: never renamed
    "STAR":     ("spyglass",),
    "X":        (),           # rat: never renamed
    "MOON":     ("shark",),
    "DIAMOND":  (),           # parrot: never renamed
    "SKULL":    ("kraken",),
    "SUN":      (),           # coin: never renamed
}


class Symbols(enum.Enum):
    """Member layout: (display, weight, points).

    ``weight`` is the relative chance of the symbol filling one of a card's 16
    slots; the weights sum to NUMBER_OF_SYMBOLS_IN_PLAY.

    ``points`` is the score for holding 0..12 of the symbol, where index 12 is a
    saturating "12 or more" bucket. Index 0 is the score for holding *none*, and
    three of these are deliberately non-zero (mask -4, snake -1, weapon +1),
    giving every board a constant -4 offset.
    """

    CIRCLE = ('food', 7, [0, 1, 1, 2, 3, 3, 4, 5, 6, 8, 9, 10, 10])
    SQUARE = ('treasure', 5, [0, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 14])
    TRIANGLE = ('rum', 4, [-2, -1, 0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 14])
    STAR = ('weapon', 6, [1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9])
    X = ('rat', 8, [0, 0, -1, -1, -2, -2, -3, -4, -5, -6, -7, -8, -8])
    MOON = ('snake', 6, [-1, -1, -1, -2, -2, -2, -3, -3, -4, -4, -5, -5, -5])
    DIAMOND = ('parrot', 2, [0, 3, 0, 5, 0, 8, 0, 10, 0, 13, 0, 15, 0])
    SKULL = ('mask', 3, [-4, -3, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    SUN = ('coin', 5, [0, 1, 2, 3, 5, 7, 5, 3, 2, 1, 0, 0, 0])
    ARROW_LEFT = ('arrow_left', 1.5, [0] * 13)
    ARROW_RIGHT = ('arrow_right', 1.5, [0] * 13)
    ARROW_UP = ('arrow_up', 1.5, [0] * 13)
    ARROW_DOWN = ('arrow_down', 1.5, [0] * 13)
    NOTHING = ('', 44, [0] * 13)

    def __init__(self, display: str, weight: float, points: List[int]):
        self.display = display
        self.weight = weight
        self.points = points

    def __str__(self):
        return self.display

    def __repr__(self):
        return self.display

    def __eq__(self, other):
        # Return NotImplemented rather than raising on a non-Symbol: comparing
        # against a bare string used to blow up with AttributeError, including
        # inside `in` tests against a set of Symbols.
        if not isinstance(other, Symbols):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def value_symbol(self) -> bool:
        """True for the nine scoring symbols — not arrows, not an empty slot."""
        return self is not Symbols.NOTHING and self not in Symbols.arrows()

    @property
    def aliases(self) -> Tuple[str, ...]:
        """Names this symbol used to be known by. See LEGACY_NAMES."""
        return LEGACY_NAMES.get(self.name, ())

    @property
    def abbrev(self) -> str:
        """Short label for table headers, derived so it follows a rename.

        Arrows get a direction code (AL/AR/AU/AD) because they would all
        collapse to "Arr" otherwise.
        """
        if self in Symbols.arrows():
            return "A" + self.name.rsplit("_", 1)[1][0].upper()
        return self.display[:3].title() if self.display else ""

    @staticmethod
    def of(name: str) -> "Symbols":
        """Resolve a symbol from its stable ID, current display name, or any legacy name.

        Use this for every name arriving from outside the code — card JSON files,
        model predictions, CLI arguments. Matching is case-insensitive.

        Raises ValueError on an unknown name rather than returning None, so a
        future rename fails loudly instead of silently reading nothing.
        """
        try:
            return _SYMBOL_LOOKUP[name.strip().casefold()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"Unknown symbol name {name!r}. Valid names are: "
                + ", ".join(sorted(_SYMBOL_LOOKUP))
            ) from None

    @staticmethod
    def arrows() -> List["Symbols"]:
        return [Symbols.ARROW_LEFT, Symbols.ARROW_UP, Symbols.ARROW_RIGHT, Symbols.ARROW_DOWN]


def _build_symbol_lookup() -> Dict[str, "Symbols"]:
    """Index every symbol by stable ID, display name and legacy name.

    Collisions are a hard error: if a new display name shadows another symbol's
    legacy name, resolution would become order-dependent and silently wrong.
    """
    lookup: Dict[str, Symbols] = {}
    for symbol in Symbols:
        names = [symbol.name, *symbol.aliases]
        if symbol.display:  # NOTHING has no display name
            names.append(symbol.display)
        for name in names:
            key = name.casefold()
            if key in lookup and lookup[key] is not symbol:
                raise ValueError(
                    f"Symbol name {name!r} maps to both {lookup[key].name} and {symbol.name}"
                )
            lookup[key] = symbol
    return lookup


_SYMBOL_LOOKUP: Dict[str, Symbols] = _build_symbol_lookup()

assert sum(symbol.weight for symbol in Symbols) == NUMBER_OF_SYMBOLS_IN_PLAY

# Points are multiplied against 13-long count vectors elsewhere; keep them aligned.
assert all(len(symbol.points) == 13 for symbol in Symbols), \
    "every points table must have 13 entries (counts 0..12)"

# Abbreviations are derived from display names, so a rename could collide them.
_abbrevs = [s.abbrev for s in Symbols if s.abbrev]
assert len(_abbrevs) == len(set(_abbrevs)), \
    f"symbol abbreviations are not unique: {sorted(_abbrevs)}"
