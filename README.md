# travel-squares

Design tooling for a card game. Each card is a 2×2 grid of **quarters**, each
holding 0–4 **symbols**. Cards are laid down in an overlapping stack, so later
cards hide parts of earlier ones, and arrows copy symbols from adjacent visible
quarters. You score on the multiset of symbols still visible at the end.

The repo exists to do two things:

1. **Generate a deck** where cards are close in value, so which card you want
   depends on the situation rather than one card being plainly better — whether
   because of its own strength or because a simple rule ("always take the most
   parrots") wins.
2. **Generate labelled training images** from that deck, for training an
   object detector.

Everything here serves one of those. If something doesn't, it should go.

---

## Symbol names

The game has been re-themed once and will be again, so names come in three layers.
`src/symbols.py` is the single source of truth.

| Layer | Example | Changes? |
|---|---|---|
| **Stable ID** — the enum member name | `CIRCLE` | Never. This is what gets written to disk. |
| **`display`** — the current theme label | `food` | Freely. |
| **`LEGACY_NAMES`** — every previous name | `anchor` | Append-only. |

Read incoming names with `Symbols.of(...)`, which accepts all three and raises on
anything unknown. Write names as `symbol.name`. Never key data off `display`.

**To rename a symbol:** change its `display`, append the old value to
`LEGACY_NAMES`. That's the whole change — stored data stays valid.

Current mapping, for reading older material:

| Stable ID | Current | Previously |
|---|---|---|
| `CIRCLE` | food | anchor |
| `SQUARE` | treasure | map |
| `TRIANGLE` | rum | — |
| `STAR` | weapon | spyglass |
| `X` | rat | — |
| `MOON` | snake | shark |
| `DIAMOND` | parrot | — |
| `SKULL` | mask | kraken |
| `SUN` | coin | — |

---

## Layout

```
src/
  symbols.py        the symbol table, naming contract and resolver
  cards.py          Card / Quarter
  deck_io.py        read and write deck JSON
  deck/             generating a balanced deck
  analysis/         checking that the deck is actually balanced
  training_images/  rendering labelled images from a deck
decks/              generated decks
pirate_cards/       card artwork (made in Illustrator) + the shipped decks
scripts/            one-off maintenance
```

## 1. Generating a deck — `src/deck/`

- **`balance.py`** — the generator. Builds candidate cards by random sampling
  against a value formula, keeps those near the target, then selects a final set
  with a good symbol spread.
- **`calibrate.py`** — refits that value formula against real simulation results.
- **`simulation.py`** — the ground truth. For a fixed set of 6 cards it evaluates
  placements across 729,529 position paths × 4,096 rotations × 32 top/bottom
  orderings, resolving overlaps and arrows, and reports what each card is
  actually worth. numpy + numba. Two modes: `--scenarios` (rank cards) and
  `--placements` (Monte Carlo sampling into sharded binary stores).
- **`gameplay.py`** — simulates *playing*: turns, card choice with lookahead, and
  opponent behaviour. Used to check that no simple strategy dominates.

```bash
python -m src.deck.balance --candidates 500 --select 120 --output decks/new.json
python -m src.deck.simulation --json decks/new.json --scenarios 200
```

## 2. Checking the deck — `src/analysis/`

All of these read the simulation output and answer "is any card plainly better?"

- **`placement_variance.py`** — the core check. For a fixed hand, how much does
  the score move purely from *how* you place the cards? High within-hand variance
  means placement skill matters more than the deal. `--n-cards` restricts the pool.
- **`card_value.py`** — each card's marginal contribution: are hands containing it
  better than hands without it?
- **`card_properties.py`** — correlates card features (positives, negatives,
  arrows, how hideable the bad quarter is) against measured value.
- **`combo_variance.py`** — score spread within a single 6-card combination.
- **`outliers.py`** — placements that score unusually far from their combo's mean.
- **`counter_examples.py`** — actively tries to falsify the current conclusions.
- **`card_balance.py`** — what makes a card balanced yet skill-expressive.
- **`deck_stats.py`** — symbol distribution across a deck, a sanity check on
  generation.
- **`scenario.py`** — renders one specific placement step by step, for checking
  a score by hand.
- **`bench.py`** — simulation throughput, worth running before a long job.

## 3. Training images — `src/training_images/`

- **`generator.py`** — renders boards from a deck plus the card artwork, with
  rotation, perspective and lighting variation, and writes COCO-style
  annotations. Symbol labels are stable IDs.
- **`composer.py`** — lays out cards and computes bounding boxes.
- **`effects.py`** — camera effects.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

Run from the repo root — `src` is a package.

```bash
python -m unittest discover -s src -t . -p 'test_*.py'
```

## Generated data

Not in git, all reproducible.

| Path | Produced by |
|---|---|
| `data/placements*`, `data/*.db`, `data/paths.*` | `src/deck/simulation.py` |
| `src/training_images/training_data*/` | `src/training_images/generator.py` |

## Removed, and why

Recoverable from git history if needed.

- **Rust simulator** (`pirate_sim_rs/`) — 2.2× faster on full path sets, but only
  implemented one of the simulation's modes and was a second copy of the scoring
  tables to keep in sync. `git show 5b27764:pirate_sim_rs/src/main.rs`
- **Analytic scoring** (`card_rules.py`, `square_card_generator.py`) — computed
  card value by convolving symbol probability distributions. Superseded by random
  sampling plus simulation, and already imported by nothing but its own tests.
  Removing it cut `symbols.py` from 758 to 175 lines and its import from 1.1 s
  to 0.02 s.
- **Recognition** (`src/recognition/`, `model/`, `resources/`) — detector
  training and reading a board back from a photo. Training didn't work well
  enough, and photo scoring is a separate product from generating the images.
- **Pelican website** — published the probability analysis that's now gone.
- **Old findings** (`CARD_BALANCE_GUIDE.md`, `analysis_results_v2.txt`,
  `good_card_findings*.txt`) — produced before the symbol rename and the point
  scale change, so their numbers were wrong.

## Known rough edges

- The analysis scripts have `sys.path.insert` preambles instead of relying on
  the package.
- `src/deck/simulation.py` documents an auto-select between its two parallel
  modes; the code never enables batch mode. That's fine — batch measured 4.3×
  slower — but the docstring is wrong.
