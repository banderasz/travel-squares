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

## ⚠️ Two different symbol vocabularies

This is the single most confusing thing in the repo. The game was re-themed
partway through, and both versions still live here:

| | Symbols | Used by |
|---|---|---|
| **Original** (`src/symbols.py`) | food, treasure, rum, weapon, rat, snake, parrot, mask, coin | `card_rules.py`, `square_card_generator.py`, `simulate_game.py`, `statistics.py`, the Pelican site |
| **Current** (`SYM_MAP` in `sim.py`) | anchor, shark, rat, kraken, map, coin, rum, parrot, spyglass | `src/simulation/sim.py`, `pirate_sim_rs/`, all root `analyze_*` scripts, `pirate_cards/*.json` |

Only rum, rat, parrot and coin appear in both. The scoring tables are also
independent. Workstreams 1 and 3 below are therefore **not** comparable
without a translation step.

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

- `src/simulation/sim.py` — numpy + numba implementation. Precomputes card data
  into L1-sized arrays, caches DFS position paths, union-finds arrow groups, and
  persists to SQLite or sharded binary stores.
- `pirate_sim_rs/` — a Rust + rayon port of the same algorithm, reading the same
  JSON and `.npy` path cache and writing the same schema.

Both are current and neither is authoritative; the Rust version is faster, while
the analysis scripts import `PlacementStore` and friends from `sim.py`. Note the
scoring tables are duplicated between them (and in several root scripts) — see
"Known rough edges".

```bash
# both expect to run from the repo root
python -m src.simulation.sim --help
cargo run --release --manifest-path pirate_sim_rs/Cargo.toml -- --help
```

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
| `data/placements*`, `data/*.db`, `data/paths.*` | `src/simulation/sim.py`, `pirate_sim_rs` |
| `src/training_data_generator/training_data*/` | `training-data-generator.py` |
| `src/training_data_generator/yolo_dataset/`, `runs/` | `train_yolo.py` |
| `src/square-pirates.jsonl` | `src/gcloud_stuff.py` |
| `output/` | `pelican content` |
| `pirate_sim_rs/target/` | `cargo build` |

## Known rough edges

- The two symbol vocabularies described above.
- Scoring tables are duplicated in seven places (`sim.py`, `main.rs`,
  `find_counter_examples.py`, `generate_balanced_deck_v3.py`,
  `analyze_combo_variance.py`, `analyze_card_properties.py`,
  `visualize_scenario.py`) plus `src/symbols.py` and `symbol_points.py`. Symbol
  name lists are hardcoded in ten files. Changing a rule means changing all of them.
- Root analysis scripts carry `sys.path.insert` preambles instead of being a package.
- Several scripts are versioned by filename (`find_counter_examples{,_v2}`,
  `generate_balanced_deck{,_v3}` with no v2, `analyze_placement_variance{,_20}`).
- `Detection.total_overlap_with_others` is a stub.
