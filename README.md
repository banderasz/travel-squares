# travel-squares

Design and balancing work for a pirate-themed card game ("square pirates").

Each card is a 2×2 grid of **quarters**; each quarter holds 0–4 **symbols**. Cards
are placed in an overlapping stack, so later cards hide parts of earlier ones.
Arrows copy symbols from adjacent visible quarters. You score on the multiset of
symbols still visible at the end, via a per-symbol count → points table.

The repo holds four largely independent workstreams that grew at different times.
They share the game concept but **not** the code — read the caveat on symbol
vocabularies below before assuming otherwise.

---

## Symbol names

The game has been re-themed once, and will be again, so symbol names come in
three layers. `src/symbols.py` is the single source of truth.

| Layer | Example | Changes? |
|---|---|---|
| **Stable ID** — the enum member name | `CIRCLE` | Never. This is what gets written to disk. |
| **`display`** — the current theme label | `food` | Freely. |
| **`LEGACY_NAMES`** — every previous name | `anchor` | Append-only. |

Read every incoming name with `Symbols.of(...)`, which accepts all three forms
and raises on anything unknown. Write names as `symbol.name`. Never key data off
`display`.

**To rename a symbol:** change its `display` and append the old value to
`LEGACY_NAMES`. That is the whole change — no stored data is invalidated, and old
deck files keep loading.

The current mapping, for reference when reading older files or analysis notes:

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

The legacy names are permanent: the trained detection model emits them as its
class labels, so they cannot be retired without retraining it.

Two caveats when reading older material:

- **`CARD_BALANCE_GUIDE.md`, `analysis_results_v2.txt` and `good_card_findings*.txt`
  predate the rename** and use the pirate names throughout. Translate with the
  table above.
- The **simulation's integer symbol indices are a wire format**, baked into the
  placement shards under `data/`. `SYM_ORDER` in
  `sim.py` pins them; never reorder.

---

## 1. Scoring model and static site

The original analytical model: exact probability distributions over scores,
computed by convolving per-symbol PMFs rather than sampling.

- `src/symbols.py` — the `Symbols` enum: weights, per-count point tables, colours.
- `src/card_rules.py` — scoring rules; `ArrowBonusPoints` handles arrow copying.
- `src/square_card_generator.py` — generates cards within a target point range.
- `src/simulate_game.py`, `src/statistics.py`, `src/deck_analyzer.py` — Monte-Carlo
  cross-checks and summary stats. The `src/*.csv` files are their output.
- `content/` + `pelicanconf.py` — a small Pelican site writing up the symbol analysis.

```bash
pelican content          # -> output/ (gitignored)
```

## 2. Board detection from photos

Read a physical board from a photograph and reconstruct which symbols are visible.

- `src/training_data_generator/training-data-generator.py` — renders synthetic
  boards from `pirate_cards/*.png` with rotation, perspective and lighting
  effects, emitting COCO-style annotations. `image_effects.py`,
  `new_image_generator.py` support it.
- `src/training_data_generator/train_yolo.py` — converts those annotations to
  YOLO format and trains YOLOv8. See `README_YOLO.md`.
- `src/gcloud_stuff.py` — converts annotations to a Vertex AI JSONL manifest.
- `src/using_cloud_model.py` / `using_local_model.py` — run inference against
  Vertex AI, or a local TensorFlow Serving container.
- `start_model.sh` — serves `model/square_pirates_edge` via Docker (picks an
  ARM64 or x86 image automatically).
- `src/processing_detected_images.py` — turns raw detections into a `Board`:
  filters by confidence, assigns symbols to their best-overlapping quarter, and
  links quarters into a grid.
- `src/Scorer.py` — scores a reconstructed board.

**Known gap:** detection on a real photo is still well short of the hand-labelled
board. The model returns 20 raw quarter boxes with low confidence (max 0.81, most
below 0.55), and no threshold recovers the 10 expected quarters (0.4 → 14,
0.5 → 8). Overlapping duplicates are never merged because
`Detection.total_overlap_with_others` is an unimplemented stub. `test_real_scenario`
therefore only asserts a lower bound.

## 3. Placement simulation

The combinatorial core: for a fixed set of 6 cards, evaluate placements across
729,529 position paths × 4,096 rotations × 32 top/bottom orderings (~95.6 billion).
`analysis/01`–`06` work through the combinatorics, pruning and equivalence
arguments that make this tractable.

`src/simulation/sim.py` is the implementation — numpy + numba. It precomputes card
data into L1-sized arrays, caches DFS position paths, union-finds arrow groups, and
persists to SQLite or sharded binary stores. Its scoring and symbol indices derive
from `Symbols`.

Two modes:

- **scenario** (`--scenarios`) — pick random 6-card hands, evaluate each across the
  path set, rank cards by mean score into SQLite.
- **Monte Carlo placements** (`--placements`, `--sharded`) — sample individual
  placements into binary shards. This is what produced everything under
  `data/placements_*` and every conclusion in the deck-balancing work.

The point tables here are the current ones. `src/symbols.py` previously carried a
different, much larger scale (`treasure` = n² up to 144); that has been replaced.

```bash
# expects to run from the repo root
python -m src.simulation.sim --help
```

There was also a Rust + rayon port (`pirate_sim_rs/`), removed in favour of keeping
one implementation. Benchmarked on an M1 Max at 10 threads, one scenario over the
full 729,529-path set: **Rust 55.5 s vs Python 122.2 s** on identical cards. Real,
but it only implemented the scenario mode, and on smaller sampled path counts
Python was actually slightly faster (100k paths: 82 s vs 89 s). Not worth a second
copy of the algorithm to keep in sync. Recover it with
`git show 5b27764:pirate_sim_rs/src/main.rs`.

Worth knowing: Rust scaled *sublinearly* with path count (7× the paths for 2.4× the
time) where Python is roughly linear — it was amortising repeated work across paths
in a way `sim.py` does not. Porting that idea to Python is the open performance win.

## 4. Deck balancing

Root-level scripts that read the placement stores produced by workstream 3 and
ask which cards are good, and whether a balanced deck is possible.

- `analyze_card_value.py`, `analyze_card_properties.py` — marginal contribution
  of each card; correlate card features against simulated performance.
- `analyze_placement_variance*.py`, `analyze_combo_variance.py`,
  `analyze_variance.py`, `analyze_outliers.py` — how much does score depend on
  placement skill rather than on the cards dealt?
- `find_counter_examples*.py` — actively try to falsify the conclusions.
- `generate_balanced_deck.py`, `generate_balanced_deck_v3.py`,
  `recalibrate_formula.py` — fit a card-value formula and generate decks from it.
- `reshard_placements.py`, `visualize_scenario.py` — data plumbing and debugging.

Conclusions are written up in `CARD_BALANCE_GUIDE.md`, `analysis_results_v2.txt`
and `good_card_findings*.txt`. Headline finding: symbol *distribution* across
quarters matters more than symbol *count*, and rum concentration alone swings
expected value by ~3 points.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# YOLO training only (torch, ultralytics, opencv):
pip install -r src/training_data_generator/requirements_yolo.txt
```

## Tests

Run from the repo root — `src` is imported as a package, so `src.`-prefixed
imports are required throughout.

```bash
python -m unittest discover -s src -t . -p 'test_*.py'
```

The board-detection tests can render what they detected. That is opt-in, because
it opens image-viewer windows:

```bash
SHOW_VISUALS=1 python -m unittest src.test_processing_detected_images
```

## Generated data

None of this is in git; all of it is reproducible. It reached ~264 GB.

| Path | Produced by |
|---|---|
| `data/placements*`, `data/*.db`, `data/paths.*` | `src/simulation/sim.py` |
| `src/training_data_generator/training_data*/` | `training-data-generator.py` |
| `src/training_data_generator/yolo_dataset/`, `runs/` | `train_yolo.py` |
| `src/square-pirates.jsonl` | `src/gcloud_stuff.py` |
| `output/` | `pelican content` |

## Known rough edges

- Root analysis scripts carry `sys.path.insert` preambles instead of being a package.
- Several scripts are versioned by filename (`find_counter_examples{,_v2}`,
  `generate_balanced_deck{,_v3}` with no v2, `analyze_placement_variance{,_20}`).
- `Detection.total_overlap_with_others` is a stub.
- `src/statistics.py` does not run: `df = create_df()` is commented out but `df`
  is used further down. Pre-existing.
- `src/conv_test.py` does not parse — it ends mid-expression (`q = Symbols.CIRCLE.`).
  Abandoned scratch file.
- Nothing renders the card artwork; `pirate_cards/*.png` are produced in Adobe
  Illustrator, so a regenerated deck will not match the existing images.
