#!/usr/bin/env python3
"""
Visualize one specific scenario for manual score validation.

Shows:
  1. The 6 cards and their quarter contents
  2. A specific path (card positions on the grid)
  3. A specific z-ordering (top/bottom config)
  4. Which quarters are visible after overlapping
  5. Visible symbol counts and scores
  6. Final score for this single configuration
  7. The simulation's expected score (averaged over all rotations, paths, tb configs)
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.simulation.sim import (load_cards, get_paths, evaluate_scenario,
                                 SYM_MAP, SYM_ORDER, SCORE_DEFAULT, Z_STACKS,
                                 _make_z_stacks, symbol_index)
from src.symbols import Symbols

# ── Config ──────────────────────────────────────────────────
CARD_IDS   = [0, 5, 3, 2, 4, 1]     # 6 cards to use
PATH_IDX   = 0                        # which path
TB_IDX     = 0                        # which top/bottom config (0-31)
ROTATIONS  = [0, 0, 0, 0, 0, 0]      # rotation per card (0-3, 0=original)

# ── Symbol display ──────────────────────────────────────────
# Keyed by stable symbol ID so a rename does not silently mislabel the output.
# (These were pirate icons; they now follow the current display names.)
SYM_ICONS = {
    'CIRCLE': '🍖', 'MOON': '🐍', 'X': '🐀', 'SKULL': '🎭',
    'SQUARE': '💰', 'SUN': '🪙', 'TRIANGLE': '🍺', 'DIAMOND': '🦜',
    'STAR': '⚔️ ', 'ARROW_UP': '⬆️ ', 'ARROW_DOWN': '⬇️ ',
    'ARROW_LEFT': '⬅️ ', 'ARROW_RIGHT': '➡️ ',
}
SYM_SHORT = {symbol.name: symbol.abbrev for symbol in Symbols if symbol.abbrev}
INV_SYM = {v:k for k,v in SYM_MAP.items()}
QNAMES = ['top_left','top_right','bottom_left','bottom_right']

# ── Rotation helper ─────────────────────────────────────────
ARROW_ROT = {9:12, 12:10, 10:11, 11:9}
ROT_MAP_ARR = np.array([2, 0, 3, 1], dtype=np.int8)

def rotate_quarters(q_syms, times):
    """Rotate quarter symbol lists. q_syms = [TL, TR, BL, BR] as symbol id lists."""
    cur = q_syms
    for _ in range(times):
        new = [None]*4
        for i in range(4):
            old = cur[ROT_MAP_ARR[i]]
            new[i] = [ARROW_ROT.get(s, s) for s in old]
        cur = new
    return cur

# ── Load data ───────────────────────────────────────────────
with open('pirate_cards/pirate_20_25.json') as f:
    raw_cards = json.load(f)

cd = load_cards('pirate_cards/pirate_20_25.json')
paths_all = get_paths()

print("=" * 72)
print("  SCENARIO VISUALIZATION — Manual Score Validation")
print("=" * 72)

# ── 1. Show the 6 cards ────────────────────────────────────
print(f"\n{'─'*72}")
print("  STEP 1: The 6 Cards (before rotation)")
print(f"{'─'*72}")

cards_raw = []
for i, cid in enumerate(CARD_IDS):
    q = raw_cards[cid]['card']['quarters']
    syms = {qn: q[qn] for qn in QNAMES}
    cards_raw.append(syms)
    
    print(f"\n  Card {cid} (slot {i}):")
    tl = ', '.join(SYM_SHORT.get(s,s) for s in syms['top_left']) or '(empty)'
    tr = ', '.join(SYM_SHORT.get(s,s) for s in syms['top_right']) or '(empty)'
    bl = ', '.join(SYM_SHORT.get(s,s) for s in syms['bottom_left']) or '(empty)'
    br = ', '.join(SYM_SHORT.get(s,s) for s in syms['bottom_right']) or '(empty)'
    w = max(len(tl), len(bl)) + 4
    print(f"    ┌{'─'*w}┬{'─'*w}┐")
    print(f"    │ {tl:<{w-1}}│ {tr:<{w-1}}│")
    print(f"    ├{'─'*w}┼{'─'*w}┤")
    print(f"    │ {bl:<{w-1}}│ {br:<{w-1}}│")
    print(f"    └{'─'*w}┴{'─'*w}┘")

# ── 2. Show the path (card positions) ──────────────────────
print(f"\n{'─'*72}")
print("  STEP 2: Card Positions (Path)")
print(f"{'─'*72}")

path = paths_all[PATH_IDX]
print(f"\n  Path index: {PATH_IDX}")
print(f"  Card positions (px, py) — each card is 2×2 cells:")
for i in range(6):
    print(f"    Slot {i} (Card {CARD_IDS[i]}): position ({path[i,0]}, {path[i,1]})")

# Build a visual grid showing card placement
all_cells = {}
for i in range(6):
    px, py = int(path[i,0]), int(path[i,1])
    for qi in range(4):
        cx = px + (qi & 1)
        cy = py + (qi >> 1)
        if (cx, cy) not in all_cells:
            all_cells[(cx,cy)] = []
        all_cells[(cx,cy)].append((i, qi))

min_x = min(c[0] for c in all_cells)
max_x = max(c[0] for c in all_cells)
min_y = min(c[1] for c in all_cells)
max_y = max(c[1] for c in all_cells)

print(f"\n  Grid layout (which card slot occupies each cell, last placed = on top):")
print(f"       ", end='')
for x in range(min_x, max_x+1):
    print(f"  x={x:+d} ", end='')
print()
for y in range(min_y, max_y+1):
    print(f"  y={y:+d}  ", end='')
    for x in range(min_x, max_x+1):
        if (x,y) in all_cells:
            slots = all_cells[(x,y)]
            labels = ','.join(str(s) for s,_ in slots)
            print(f"  [{labels:^5}]", end='')
        else:
            print(f"  {'':^7}", end='')
    print()

# ── 3. Z-ordering (top/bottom config) ──────────────────────
print(f"\n{'─'*72}")
print("  STEP 3: Z-Ordering (Top/Bottom Config)")
print(f"{'─'*72}")

z_stack = Z_STACKS[TB_IDX]
print(f"\n  TB config: {TB_IDX} (binary: {TB_IDX:05b})")
print(f"  Z-stack (bottom → top): {list(z_stack)}")
print(f"  Meaning: slot {z_stack[0]} placed first (bottom), ... slot {z_stack[5]} last (top)")

# ── 4. Compute visibility ──────────────────────────────────
print(f"\n{'─'*72}")
print("  STEP 4: Visibility (what's visible after stacking)")
print(f"{'─'*72}")

# Build grid to determine visibility
GRID_OFF = 7
GRID_SZ = 15
grid = np.full(GRID_SZ * GRID_SZ, -1, dtype=np.int8)
gqi = np.zeros(GRID_SZ * GRID_SZ, dtype=np.int8)

for zi in range(6):
    k = z_stack[zi]
    px, py = int(path[k, 0]), int(path[k, 1])
    for qi in range(4):
        idx = (px + (qi & 1) + GRID_OFF) * GRID_SZ + (py + (qi >> 1) + GRID_OFF)
        grid[idx] = k
        gqi[idx] = qi

# Determine visibility mask per card
vis = np.zeros(6, dtype=np.int8)
for idx in range(GRID_SZ * GRID_SZ):
    if grid[idx] >= 0:
        vis[grid[idx]] |= (1 << gqi[idx])

QN = ['TL', 'TR', 'BL', 'BR']
print(f"\n  Visibility per card (which quarters are visible):")
for k in range(6):
    v = vis[k]
    visible_qs = [QN[qi] for qi in range(4) if v & (1 << qi)]
    hidden_qs = [QN[qi] for qi in range(4) if not (v & (1 << qi))]
    print(f"    Slot {k} (Card {CARD_IDS[k]}): visible={visible_qs}, hidden={hidden_qs}, mask={v:04b}")

# Show the visibility grid
print(f"\n  Top-view grid (slot.quarter that's visible at each cell):")
print(f"       ", end='')
for x in range(min_x, max_x+1):
    print(f"  x={x:+d} ", end='')
print()
for y in range(min_y, max_y+1):
    print(f"  y={y:+d}  ", end='')
    for x in range(min_x, max_x+1):
        idx = (x + GRID_OFF) * GRID_SZ + (y + GRID_OFF)
        if 0 <= idx < GRID_SZ*GRID_SZ and grid[idx] >= 0:
            k = grid[idx]
            qi = gqi[idx]
            print(f"  [{k}.{QN[qi]:>2}]", end='')
        elif (x,y) in all_cells:
            print(f"  [  ×  ]", end='')
        else:
            print(f"  {'':^7}", end='')
    print()

# ── 5. Count visible symbols for given rotations ───────────
print(f"\n{'─'*72}")
print(f"  STEP 5: Symbol Counts (rotation = {ROTATIONS})")
print(f"{'─'*72}")

# Build rotated quarter contents for each card
rotated = []
for i, cid in enumerate(CARD_IDS):
    q = raw_cards[cid]['card']['quarters']
    q_ids = [[symbol_index(s) for s in q[qn]] for qn in QNAMES]
    q_rot = rotate_quarters(q_ids, ROTATIONS[i])
    rotated.append(q_rot)

# Show rotated cards with visibility highlighting
print(f"\n  Rotated cards with visible quarters marked [✓] or hidden [×]:")
for i in range(6):
    v = vis[i]
    print(f"\n    Slot {i} (Card {CARD_IDS[i]}, rot={ROTATIONS[i]}):")
    for qi in range(4):
        is_vis = bool(v & (1 << qi))
        syms = [INV_SYM[s] for s in rotated[i][qi]]
        sym_str = ', '.join(SYM_SHORT[s] for s in syms) or '(empty)'
        mark = '✓' if is_vis else '×'
        print(f"      {QN[qi]:>2} [{mark}]: {sym_str}")

# Count visible symbols (ignoring arrows, no cross-card arrow resolution)
sym_counts = np.zeros(9, dtype=int)  # 9 regular symbols
cross_arrows = []

for i in range(6):
    v = vis[i]
    cid = CARD_IDS[i]
    r = ROTATIONS[i]
    px, py = int(path[i, 0]), int(path[i, 1])
    
    for qi in range(4):
        if not (v & (1 << qi)):
            continue
        for sid in rotated[i][qi]:
            if sid < 9:
                sym_counts[sid] += 1
            else:
                # Arrow: check if it points to another card
                arrow_dir = sid - 9  # 0=up,1=down,2=left,3=right
                dx = [0, 0, -1, 1][arrow_dir]
                dy = [-1, 1, 0, 0][arrow_dir]
                src_cx = px + (qi & 1)
                src_cy = py + (qi >> 1)
                tgt_cx = src_cx + dx
                tgt_cy = src_cy + dy
                tgt_idx = (tgt_cx + GRID_OFF) * GRID_SZ + (tgt_cy + GRID_OFF)
                dir_name = ['↑','↓','←','→'][arrow_dir]
                if 0 <= tgt_idx < GRID_SZ*GRID_SZ and grid[tgt_idx] >= 0:
                    tk = grid[tgt_idx]
                    tqi = gqi[tgt_idx]
                    if tk != i:
                        # Cross-card arrow!
                        tgt_syms = [INV_SYM[s] for s in rotated[tk][tqi] if s < 9]
                        cross_arrows.append((i, qi, dir_name, tk, tqi, tgt_syms))
                        for sid2 in rotated[tk][tqi]:
                            if sid2 < 9:
                                sym_counts[sid2] += 1
                        print(f"    → Cross-card arrow: slot {i} {QN[qi]} {dir_name} → slot {tk} {QN[tqi]} adds {[SYM_SHORT[s] for s in tgt_syms]}")
                    else:
                        # Same-card arrow (already counted in cbm)
                        tgt_syms = [INV_SYM[s] for s in rotated[i][tqi] if s < 9]
                        if v & (1 << tqi):
                            for sid2 in rotated[i][tqi]:
                                if sid2 < 9:
                                    sym_counts[sid2] += 1
                            print(f"    → Same-card arrow: slot {i} {QN[qi]} {dir_name} → slot {i} {QN[tqi]} adds {[SYM_SHORT[s] for s in tgt_syms]}")

# ── 6. Score calculation ───────────────────────────────────
print(f"\n{'─'*72}")
print("  STEP 6: Score Calculation")
print(f"{'─'*72}")

print(f"\n  {'Symbol':<10} {'Count':>5} {'Score':>6}  Scoring table")
print(f"  {'─'*10} {'─'*5} {'─'*6}  {'─'*30}")
total = 0
for si, symbol in enumerate(SYM_ORDER[:9]):
    name = symbol.display
    cnt = int(sym_counts[si])
    pts_table = SCORE_DEFAULT[symbol.name]
    clamped = min(cnt, 11)
    score = pts_table[clamped]
    total += score
    table_str = str(pts_table[:min(cnt+3, 12)])
    marker = f"  ← [{cnt}]={score}"
    print(f"  {name:<10} {cnt:>5} {score:>+6}  {table_str}{marker}")

print(f"\n  TOTAL SCORE (this specific config): {total}")

# ── 7. Compare with simulation ─────────────────────────────
print(f"\n{'─'*72}")
print("  STEP 7: Simulation Expected Score")
print(f"{'─'*72}")

# Run sim on a small number of paths
small_paths = paths_all[:100]
sim_score = evaluate_scenario(CARD_IDS, small_paths, cd)
print(f"\n  Simulation mean score (100 paths, all rotations, all tb configs): {sim_score:.3f}")
print(f"  Your manual config score: {total}")
print(f"  (The simulation averages over all 4^6=4096 rotation combos × 32 tb configs × all paths)")
print(f"  So the manual score is ONE of those ~130K+ configs per path.")

# ── 8. For the specific path, show the distribution ────────
print(f"\n{'─'*72}")
print("  STEP 8: Score Distribution for This Path")
print(f"{'─'*72}")

# Compute score for all 32 tb × all 4^6 rotation combos for this one path
from collections import Counter
all_scores = []
for tb in range(32):
    zs = Z_STACKS[tb]
    # Compute visibility
    g = np.full(GRID_SZ*GRID_SZ, -1, dtype=np.int8)
    gq = np.zeros(GRID_SZ*GRID_SZ, dtype=np.int8)
    for zi in range(6):
        k = zs[zi]
        px, py = int(path[k,0]), int(path[k,1])
        for qi in range(4):
            idx2 = (px + (qi&1) + GRID_OFF)*GRID_SZ + (py + (qi>>1) + GRID_OFF)
            g[idx2] = k
            gq[idx2] = qi
    v = np.zeros(6, dtype=np.int8)
    for idx2 in range(GRID_SZ*GRID_SZ):
        if g[idx2] >= 0:
            v[g[idx2]] |= (1 << gq[idx2])

    # All rotation combos (4^6 = 4096)
    for combo in range(4096):
        rots = [(combo >> (2*i)) & 3 for i in range(6)]
        sc = np.zeros(9, dtype=int)
        for i in range(6):
            q = raw_cards[CARD_IDS[i]]['card']['quarters']
            q_ids = [[symbol_index(s) for s in q[qn]] for qn in QNAMES]
            q_rot = rotate_quarters(q_ids, rots[i])
            for qi in range(4):
                if v[i] & (1 << qi):
                    for sid in q_rot[qi]:
                        if sid < 9:
                            sc[sid] += 1
                        else:
                            # Arrow resolution
                            ad = sid - 9
                            dx = [0,0,-1,1][ad]
                            dy = [-1,1,0,0][ad]
                            px2 = int(path[i,0]) + (qi&1)
                            py2 = int(path[i,1]) + (qi>>1)
                            tx = px2+dx+GRID_OFF
                            ty = py2+dy+GRID_OFF
                            if 0<=tx<GRID_SZ and 0<=ty<GRID_SZ:
                                tidx2 = tx*GRID_SZ+ty
                                if g[tidx2]>=0:
                                    tk = g[tidx2]
                                    tqi2 = gq[tidx2]
                                    q2 = raw_cards[CARD_IDS[tk]]['card']['quarters']
                                    q2_ids = [[symbol_index(s2) for s2 in q2[qn]] for qn in QNAMES]
                                    q2_rot = rotate_quarters(q2_ids, rots[tk])
                                    for sid2 in q2_rot[tqi2]:
                                        if sid2 < 9:
                                            sc[sid2] += 1
        total_s = 0
        for si, symbol in enumerate(SYM_ORDER[:9]):
            c = min(int(sc[si]), 11)
            total_s += SCORE_DEFAULT[symbol.name][c]
        all_scores.append(total_s)

mean_s = np.mean(all_scores)
print(f"\n  Brute-force mean (path {PATH_IDX}, all rot×tb): {mean_s:.4f}")

# Use sim for same single path
sim_single = evaluate_scenario(CARD_IDS, paths_all[PATH_IDX:PATH_IDX+1], cd)
print(f"  Simulation result  (path {PATH_IDX}, all rot×tb): {sim_single:.4f}")
print(f"  Match: {'✓ YES' if abs(mean_s - sim_single) < 0.01 else '✗ NO — MISMATCH!'} (diff = {abs(mean_s-sim_single):.6f})")

hist = Counter(all_scores)
print(f"\n  Score distribution (32×4096 = {len(all_scores)} configs):")
print(f"    Min: {min(all_scores)}, Max: {max(all_scores)}, Mean: {mean_s:.2f}")
for score in sorted(hist.keys()):
    pct = 100 * hist[score] / len(all_scores)
    bar = '█' * int(pct * 0.5)
    if pct >= 0.5:
        print(f"    {score:>4}: {hist[score]:>5} ({pct:5.1f}%) {bar}")
