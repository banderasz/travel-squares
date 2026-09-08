"""
High-performance pirate card placement simulation.

Architecture:
  1. All card data precomputed into tight numpy arrays (67 KB, fits L1 cache)
  2. Position paths precomputed via DFS and cached to disk
  3. Numba JIT-compiled inner loop: per-symbol PMF convolution
     - Phase 1: batch visibility + cross-card arrow detection (merged, single grid pass)
     - Fast path (n_cross==0 per tb): direct 4×12 rotation convolution
     - Slow path (n_cross>0): union-find groups + grouped convolution
  4. Parallelism: prange over paths, one scenario at a time.
  5. SQLite persistence for resumable runs

Notes:
  --threads 8 skips the 2 efficiency cores on an M1 Max (avoids a prange
  barrier bottleneck); --threads 10 was slightly faster in practice.

  --sim-batch >1 switches to parallelising over scenarios instead of paths.
  Do not: it measured 4.3x slower (356s vs 82s on 4 scenarios x 100k paths).
  The default leaves it off, despite an "auto-select" that never triggers.
"""
import json
import os
import pickle
import sqlite3
import time
import argparse
import random
import sys
import numpy as np
import numba as nb

from src.symbols import Symbols

# ============================================================
# CONSTANTS
# ============================================================
# Symbol index order.
#
# THIS IS A WIRE FORMAT. These integers are baked into every placement shard
# under data/ (~200 GB). Reordering them silently reinterprets all existing
# simulation output. Add new symbols at the end; never move an existing one.
#
# The trailing comments are the names these indices were originally written
# with, kept for cross-referencing the older analysis scripts.
SYM_ORDER = (
    Symbols.CIRCLE,       # 0  anchor
    Symbols.MOON,         # 1  shark
    Symbols.X,            # 2  rat
    Symbols.SKULL,        # 3  kraken
    Symbols.SQUARE,       # 4  map
    Symbols.SUN,          # 5  coin
    Symbols.TRIANGLE,     # 6  rum
    Symbols.DIAMOND,      # 7  parrot
    Symbols.STAR,         # 8  spyglass
    Symbols.ARROW_UP,     # 9
    Symbols.ARROW_DOWN,   # 10
    Symbols.ARROW_LEFT,   # 11
    Symbols.ARROW_RIGHT,  # 12
)
SYM_MAP = {symbol.name: index for index, symbol in enumerate(SYM_ORDER)}
# Display names in wire-index order, for analysis scripts that report by index.
SYM_NAMES = [symbol.display for symbol in SYM_ORDER]
N_SYM = 9
MAX_CT = 11


def symbol_index(name):
    """Map a symbol name from card JSON to its wire index.

    Accepts stable IDs, current display names and legacy names, so decks written
    in any vocabulary load. See src/symbols.py for the naming contract.
    """
    return SYM_MAP[Symbols.of(name).name]
ARROW_DIR_DX = np.array([0, 0, -1, 1], dtype=np.int8)   # up, down, left, right
ARROW_DIR_DY = np.array([-1, 1, 0, 0], dtype=np.int8)
Q_DX = np.array([0, 1, 0, 1], dtype=np.int8)  # TL TR BL BR
Q_DY = np.array([0, 0, 1, 1], dtype=np.int8)
ROT_MAP = np.array([2, 0, 3, 1], dtype=np.int8)
ARROW_ROT_MAP = {9:12, 12:10, 10:11, 11:9}

# Scoring comes from src/symbols.py, which is the single source of truth.
# Symbols.points is indexed by count 0..12 with a saturating final bucket; the
# simulation only models counts 0..MAX_CT, so the tail is trimmed.
SCORE_DEFAULT = {
    symbol.name: symbol.points[:MAX_CT + 1] for symbol in SYM_ORDER[:N_SYM]
}

# ============================================================
# DATA LOADING
# ============================================================
def _rotate_card(quarters):
    new = []
    for i in range(4):
        old = quarters[ROT_MAP[i]]
        new.append([ARROW_ROT_MAP.get(s, s) for s in old])
    return new

def load_cards(json_path):
    """Load cards → dict of numpy arrays for Numba."""
    with open(json_path) as f:
        raw = json.load(f)
    n = len(raw)
    qnames = ['top_left','top_right','bottom_left','bottom_right']

    scoring = np.zeros((N_SYM, MAX_CT+1), dtype=np.float64)
    for name, pts in SCORE_DEFAULT.items():
        s = SYM_MAP.get(name)
        if s is not None and s < N_SYM:
            for i, p in enumerate(pts[:MAX_CT+1]):
                scoring[s, i] = p

    nac = np.zeros((n, 4, 4, N_SYM), dtype=np.int8)  # non_arrow_counts[card,rot,qi,sym]
    a_qi = np.full((n, 4, 4), -1, dtype=np.int8)       # arrow source quarters
    a_dx = np.zeros((n, 4, 4), dtype=np.int8)
    a_dy = np.zeros((n, 4, 4), dtype=np.int8)
    n_arr = np.zeros((n, 4), dtype=np.int8)

    for ci, card in enumerate(raw):
        q = card['card']['quarters']
        rot0 = [[symbol_index(s) for s in q[qn]] for qn in qnames]
        cur = rot0
        for r in range(4):
            if r > 0:
                cur = _rotate_card(cur)
            ai = 0
            for qi in range(4):
                for s in cur[qi]:
                    if s < N_SYM:
                        nac[ci, r, qi, s] += 1
                    else:
                        idx = s - 9  # 0=up,1=down,2=left,3=right
                        a_qi[ci, r, ai] = qi
                        a_dx[ci, r, ai] = ARROW_DIR_DX[idx]
                        a_dy[ci, r, ai] = ARROW_DIR_DY[idx]
                        ai += 1
            n_arr[ci, r] = ai

    # Precompute counts_by_mask[card,rot,mask,sym] — includes independent arrows
    cbm = np.zeros((n, 4, 16, N_SYM), dtype=np.int8)
    for ci in range(n):
        for r in range(4):
            for mask in range(16):
                for qi in range(4):
                    if mask & (1 << qi):
                        for s in range(N_SYM):
                            cbm[ci, r, mask, s] += nac[ci, r, qi, s]
                for ai in range(n_arr[ci, r]):
                    src = a_qi[ci, r, ai]
                    if not (mask & (1 << src)):
                        continue
                    tx = Q_DX[src] + a_dx[ci, r, ai]
                    ty = Q_DY[src] + a_dy[ci, r, ai]
                    if 0 <= tx <= 1 and 0 <= ty <= 1:
                        tq = tx + ty * 2
                        if mask & (1 << tq):
                            for s in range(N_SYM):
                                cbm[ci, r, mask, s] += nac[ci, r, tq, s]

    # Precompute which cards have any arrows (any rotation)
    card_has_arrows = np.zeros(n, dtype=np.bool_)
    for ci in range(n):
        for r in range(4):
            if n_arr[ci, r] > 0:
                card_has_arrows[ci] = True
                break

    return dict(n_cards=n, scoring=scoring, nac=nac, cbm=cbm,
                a_qi=a_qi, a_dx=a_dx, a_dy=a_dy, n_arr=n_arr,
                card_has_arrows=card_has_arrows)

# ============================================================
# POSITION PATH ENUMERATION
# ============================================================
def _card_cells(px, py):
    return [(px,py),(px+1,py),(px,py+1),(px+1,py+1)]

def _enum_paths():
    all_paths = []
    positions = [(0,0)]
    footprint = set(_card_cells(0,0))
    def dfs(d):
        if d == 6:
            all_paths.append([p for p in positions])
            return
        mn_x = min(x for x,_ in footprint)-1
        mx_x = max(x for x,_ in footprint)
        mn_y = min(y for _,y in footprint)-1
        mx_y = max(y for _,y in footprint)
        for px in range(mn_x, mx_x+1):
            for py in range(mn_y, mx_y+1):
                cells = _card_cells(px, py)
                if any(c in footprint for c in cells):
                    positions.append((px,py))
                    added = [c for c in cells if c not in footprint]
                    footprint.update(added)
                    dfs(d+1)
                    positions.pop()
                    for c in added:
                        footprint.discard(c)
    dfs(1)
    return all_paths

def get_paths(cache_path=None):
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'paths.npy')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        p = np.load(cache_path)
        print(f"  Loaded {len(p)} paths from cache")
        return p
    print("  Computing position paths (one-time)...")
    raw = _enum_paths()
    p = np.array(raw, dtype=np.int8)
    np.save(cache_path, p)
    print(f"  Computed and cached {len(p)} paths")
    return p

# Precompute z-stacks for all 32 top/bottom combos
def _make_z_stacks():
    zs = np.zeros((32, 6), dtype=np.int8)
    for tb in range(32):
        stack = [0]
        for k in range(5):
            if tb & (1 << k):
                stack.append(k + 1)
            else:
                stack.insert(0, k + 1)
        for i, v in enumerate(stack):
            zs[tb, i] = v
    return zs

Z_STACKS = _make_z_stacks()

# ============================================================
# NUMBA JIT — OPTIMIZED FOR M1 MAX
#   v4: merged arrow detection into Phase 1, per-tb fast path,
#       direct rotation convolution, batch evaluation
# ============================================================
GRID_OFF = 7
GRID_SZ = 15
_CT1 = MAX_CT + 1  # 12
_GS = GRID_SZ * GRID_SZ


@nb.njit(fastmath=True, inline='always')
def _eval_path(pi, card_ids, any_arrows, paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr):
    """Evaluate one path for one scenario across all 32 tb configs."""
    # ---- Thread-local work arrays ----
    grid = np.full(_GS, np.int8(-1))
    gqi = np.zeros(_GS, dtype=np.int8)
    touched = np.zeros(24, dtype=np.int32)
    all_vis = np.zeros((32, 6), dtype=np.int8)
    pmf = np.zeros(_CT1, dtype=np.float64)
    new_pmf = np.zeros(_CT1, dtype=np.float64)
    grp_pmf = np.zeros(_CT1, dtype=np.float64)
    lc = np.zeros((6, 4, N_SYM), dtype=np.int8)
    # Arrow arrays (precomputed per-tb)
    all_n_cross = np.zeros(32, dtype=np.int32)
    all_cs_k = np.zeros((32, 24), dtype=np.int8)
    all_cs_r = np.zeros((32, 24), dtype=np.int8)
    all_ct_k = np.zeros((32, 24), dtype=np.int8)
    all_ct_qi = np.zeros((32, 24), dtype=np.int8)
    # Group arrays
    parent = np.zeros(6, dtype=np.int8)
    gid = np.zeros(6, dtype=np.int8)
    gsz = np.zeros(6, dtype=np.int8)
    gmap = np.zeros(6, dtype=np.int8)
    gmem = np.zeros((6, 6), dtype=np.int8)
    gfill = np.zeros(6, dtype=np.int8)

    # ---- PHASE 1: Batch compute visibility + cross-card arrows ----
    # (Arrow detection merged here to avoid rebuilding grid in Phase 2)
    for tbi in range(32):
        nt = 0
        for zi in range(6):
            k = z_stacks[tbi, zi]
            px = paths[pi, k, 0]
            py = paths[pi, k, 1]
            for qi in range(4):
                idx = (px + (qi & 1) + GRID_OFF) * GRID_SZ + (py + (qi >> 1) + GRID_OFF)
                if grid[idx] == -1:
                    touched[nt] = idx
                    nt += 1
                grid[idx] = k
                gqi[idx] = qi
        for ti in range(nt):
            idx = touched[ti]
            all_vis[tbi, grid[idx]] |= np.int8(1 << gqi[idx])

        # Search for cross-card arrows while grid is still built
        if any_arrows:
            nc = 0
            for k in range(6):
                cid = card_ids[k]
                m = all_vis[tbi, k]
                px = paths[pi, k, 0]
                py = paths[pi, k, 1]
                for r in range(4):
                    for ai in range(n_arr[cid, r]):
                        src = a_qi[cid, r, ai]
                        if not (m & (1 << src)):
                            continue
                        tgx = px + (src & 1) + GRID_OFF + a_dx[cid, r, ai]
                        tgy = py + (src >> 1) + GRID_OFF + a_dy[cid, r, ai]
                        if 0 <= tgx < GRID_SZ and 0 <= tgy < GRID_SZ:
                            tidx = tgx * GRID_SZ + tgy
                            tk = grid[tidx]
                            if tk >= 0 and tk != k and nc < 24:
                                all_cs_k[tbi, nc] = k
                                all_cs_r[tbi, nc] = r
                                all_ct_k[tbi, nc] = tk
                                all_ct_qi[tbi, nc] = gqi[tidx]
                                nc += 1
            all_n_cross[tbi] = nc

        for ti in range(nt):
            grid[touched[ti]] = -1

    # ---- PHASE 2: Evaluate each tb config (no grid needed!) ----
    path_sum = 0.0
    for tbi in range(32):
        v0 = all_vis[tbi, 0]; v1 = all_vis[tbi, 1]; v2 = all_vis[tbi, 2]
        v3 = all_vis[tbi, 3]; v4 = all_vis[tbi, 4]; v5 = all_vis[tbi, 5]

        # Precompute local card counts
        cid0 = card_ids[0]; cid1 = card_ids[1]; cid2 = card_ids[2]
        cid3 = card_ids[3]; cid4 = card_ids[4]; cid5 = card_ids[5]
        for r in range(4):
            for s in range(N_SYM):
                lc[0, r, s] = cbm[cid0, r, v0, s]
                lc[1, r, s] = cbm[cid1, r, v1, s]
                lc[2, r, s] = cbm[cid2, r, v2, s]
                lc[3, r, s] = cbm[cid3, r, v3, s]
                lc[4, r, s] = cbm[cid4, r, v4, s]
                lc[5, r, s] = cbm[cid5, r, v5, s]

        n_cross = all_n_cross[tbi] if any_arrows else 0

        if n_cross == 0:
            # ---- FAST PATH: all 6 cards independent ----
            # Direct rotation convolution: 4×12 instead of 12×12
            config_score = 0.0
            for sym in range(N_SYM):
                # Card 0: initialize PMF from 4 rotations
                for i in range(_CT1):
                    pmf[i] = 0.0
                for r in range(4):
                    v = lc[0, r, sym]
                    if v > MAX_CT:
                        v = MAX_CT
                    pmf[v] += 0.25
                # Cards 1-5: convolve directly with 4 rotation values
                for k in range(1, 6):
                    for i in range(_CT1):
                        new_pmf[i] = 0.0
                    for r in range(4):
                        c = lc[k, r, sym]
                        if c > MAX_CT:
                            c = MAX_CT
                        for prev in range(_CT1):
                            pv = pmf[prev]
                            if pv > 0.0:
                                t = prev + c
                                if t > MAX_CT:
                                    t = MAX_CT
                                new_pmf[t] += pv * 0.25
                    for i in range(_CT1):
                        pmf[i] = new_pmf[i]
                for t in range(_CT1):
                    config_score += pmf[t] * scoring[sym, t]
        else:
            # ---- SLOW PATH: union-find + grouped convolution ----
            for i in range(6):
                parent[i] = i
            for ci in range(n_cross):
                a = all_cs_k[tbi, ci]
                while parent[a] != a:
                    parent[a] = parent[parent[a]]; a = parent[a]
                b = all_ct_k[tbi, ci]
                while parent[b] != b:
                    parent[b] = parent[parent[b]]; b = parent[b]
                if a != b:
                    parent[a] = b

            n_groups = 0
            for i in range(6):
                gsz[i] = 0; gmap[i] = -1; gfill[i] = 0
            for k in range(6):
                root = k
                while parent[root] != root:
                    root = parent[root]
                if gmap[root] < 0:
                    gmap[root] = n_groups; n_groups += 1
                g = gmap[root]
                gid[k] = g
                gmem[g, gfill[g]] = k
                gfill[g] += 1; gsz[g] += 1

            config_score = 0.0
            for sym in range(N_SYM):
                first = True
                for g in range(n_groups):
                    sz = gsz[g]

                    if sz == 1:
                        # Direct rotation convolution for singleton groups
                        k = gmem[g, 0]
                        if first:
                            for i in range(_CT1):
                                pmf[i] = 0.0
                            for r in range(4):
                                v = lc[k, r, sym]
                                if v > MAX_CT:
                                    v = MAX_CT
                                pmf[v] += 0.25
                            first = False
                        else:
                            for i in range(_CT1):
                                new_pmf[i] = 0.0
                            for r in range(4):
                                c = lc[k, r, sym]
                                if c > MAX_CT:
                                    c = MAX_CT
                                for prev in range(_CT1):
                                    pv = pmf[prev]
                                    if pv > 0.0:
                                        t = prev + c
                                        if t > MAX_CT:
                                            t = MAX_CT
                                        new_pmf[t] += pv * 0.25
                            for i in range(_CT1):
                                pmf[i] = new_pmf[i]
                    else:
                        # Multi-card groups: build grp_pmf, then convolve
                        for i in range(_CT1):
                            grp_pmf[i] = 0.0

                        if sz == 2:
                            k0 = gmem[g, 0]; k1 = gmem[g, 1]
                            for r0 in range(4):
                                for r1 in range(4):
                                    tot = np.int32(lc[k0, r0, sym]) + np.int32(lc[k1, r1, sym])
                                    for ci in range(n_cross):
                                        if all_cs_k[tbi, ci] == k0 and all_cs_r[tbi, ci] == r0 and all_ct_k[tbi, ci] == k1:
                                            tot += nac[card_ids[k1], r1, all_ct_qi[tbi, ci], sym]
                                        elif all_cs_k[tbi, ci] == k1 and all_cs_r[tbi, ci] == r1 and all_ct_k[tbi, ci] == k0:
                                            tot += nac[card_ids[k0], r0, all_ct_qi[tbi, ci], sym]
                                    if tot > MAX_CT:
                                        tot = MAX_CT
                                    grp_pmf[tot] += 0.0625

                        else:
                            n_combos = 4 ** sz
                            inv = 1.0 / n_combos
                            for combo in range(n_combos):
                                c = combo
                                for gi in range(sz):
                                    parent[gmem[g, gi]] = np.int8(c & 3)
                                    c >>= 2
                                tot = np.int32(0)
                                for gi in range(sz):
                                    k = gmem[g, gi]
                                    tot += lc[k, parent[k], sym]
                                for ci in range(n_cross):
                                    if gid[all_cs_k[tbi, ci]] == g and parent[all_cs_k[tbi, ci]] == all_cs_r[tbi, ci]:
                                        tot += nac[card_ids[all_ct_k[tbi, ci]], parent[all_ct_k[tbi, ci]], all_ct_qi[tbi, ci], sym]
                                if tot > MAX_CT:
                                    tot = MAX_CT
                                grp_pmf[tot] += inv

                        if first:
                            for i in range(_CT1):
                                pmf[i] = grp_pmf[i]
                            first = False
                        else:
                            for i in range(_CT1):
                                new_pmf[i] = 0.0
                            for prev in range(_CT1):
                                pv = pmf[prev]
                                if pv > 0.0:
                                    for add in range(_CT1):
                                        gv = grp_pmf[add]
                                        if gv > 0.0:
                                            t = prev + add
                                            if t > MAX_CT:
                                                t = MAX_CT
                                            new_pmf[t] += pv * gv
                            for i in range(_CT1):
                                pmf[i] = new_pmf[i]

                for t in range(_CT1):
                    config_score += pmf[t] * scoring[sym, t]

        path_sum += config_score

    return path_sum / 32.0


@nb.njit(parallel=True, fastmath=True, cache=True)
def _eval_scenario(card_ids, paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr, card_has_arrows):
    """Single scenario, prange over paths. Best for large path counts (729K)."""
    n_paths = paths.shape[0]
    any_arrows = False
    for k in range(6):
        if card_has_arrows[card_ids[k]]:
            any_arrows = True
            break
    path_scores = np.zeros(n_paths, dtype=np.float64)
    for pi in nb.prange(n_paths):
        path_scores[pi] = _eval_path(pi, card_ids, any_arrows, paths, z_stacks,
                                      cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr)
    return np.mean(path_scores)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _eval_batch(all_card_ids, paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr, card_has_arrows):
    """Multiple scenarios, prange over scenarios. Best for small path counts.
    Each core processes complete scenarios sequentially — better M1 Max utilization."""
    n_sc = all_card_ids.shape[0]
    n_paths = paths.shape[0]
    out = np.zeros(n_sc, dtype=np.float64)
    for si in nb.prange(n_sc):
        card_ids = all_card_ids[si]
        any_arrows = False
        for k in range(6):
            if card_has_arrows[card_ids[k]]:
                any_arrows = True
                break
        s = 0.0
        for pi in range(n_paths):
            s += _eval_path(pi, card_ids, any_arrows, paths, z_stacks,
                            cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr)
        out[si] = s / n_paths
    return out


@nb.njit(fastmath=True, inline='always')
def _eval_single(card_ids, rotations, pi, tbi, paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr):
    """Evaluate ONE fully-specified placement: specific cards, rotations, path, and tb config.
    Returns a single deterministic score (no averaging)."""
    grid = np.full(_GS, np.int8(-1))
    gqi = np.zeros(_GS, dtype=np.int8)
    touched = np.zeros(24, dtype=np.int32)
    vis = np.zeros(6, dtype=np.int8)
    sym_counts = np.zeros(N_SYM, dtype=np.int32)

    # Build grid for this specific tb config
    nt = 0
    for zi in range(6):
        k = z_stacks[tbi, zi]
        px = paths[pi, k, 0]
        py = paths[pi, k, 1]
        for qi in range(4):
            idx = (px + (qi & 1) + GRID_OFF) * GRID_SZ + (py + (qi >> 1) + GRID_OFF)
            if grid[idx] == -1:
                touched[nt] = idx
                nt += 1
            grid[idx] = k
            gqi[idx] = qi

    for ti in range(nt):
        idx = touched[ti]
        vis[grid[idx]] |= np.int8(1 << gqi[idx])

    # Sum symbol counts for the specific rotations
    for s in range(N_SYM):
        sym_counts[s] = 0
    for k in range(6):
        cid = card_ids[k]
        r = rotations[k]
        v = vis[k]
        for s in range(N_SYM):
            sym_counts[s] += cbm[cid, r, v, s]

    # Cross-card arrows for the specific rotations
    any_arrows = False
    for k in range(6):
        if n_arr[card_ids[k], rotations[k]] > 0:
            any_arrows = True
            break

    if any_arrows:
        for k in range(6):
            cid = card_ids[k]
            r = rotations[k]
            m = vis[k]
            px = paths[pi, k, 0]
            py = paths[pi, k, 1]
            for ai in range(n_arr[cid, r]):
                src = a_qi[cid, r, ai]
                if not (m & (1 << src)):
                    continue
                tgx = px + (src & 1) + GRID_OFF + a_dx[cid, r, ai]
                tgy = py + (src >> 1) + GRID_OFF + a_dy[cid, r, ai]
                if 0 <= tgx < GRID_SZ and 0 <= tgy < GRID_SZ:
                    tidx = tgx * GRID_SZ + tgy
                    tk = grid[tidx]
                    if tk >= 0 and tk != k:
                        tqi = gqi[tidx]
                        tk_r = rotations[tk]
                        for s in range(N_SYM):
                            sym_counts[s] += nac[card_ids[tk], tk_r, tqi, s]

    # Score from symbol counts
    score = 0.0
    for s in range(N_SYM):
        ct = sym_counts[s]
        if ct > MAX_CT:
            ct = MAX_CT
        score += scoring[s, ct]

    return score


@nb.njit(parallel=True, fastmath=True, cache=True)
def _generate_and_eval(n, n_cards, n_paths, rng_seed,
                       paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr,
                       out_cards, out_rots, out_pi, out_tb, out_scores):
    """Generate random placements AND evaluate them in one parallel Numba call.
    Each thread uses its own deterministic RNG stream (seed + thread_index).
    Eliminates Python loop for card selection entirely."""
    for i in nb.prange(n):
        # Per-sample deterministic RNG: simple LCG (fast, good enough for Monte Carlo)
        # Seed each sample independently for reproducibility regardless of thread count
        state = np.uint64(rng_seed + i * np.uint64(6364136223846793005))

        # Fisher-Yates partial shuffle: pick 6 from n_cards
        deck = np.empty(n_cards, dtype=np.int32)
        for j in range(n_cards):
            deck[j] = j
        for j in range(6):
            # LCG step
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            idx = j + np.int32((state >> np.uint64(33)) % np.uint64(n_cards - j))
            deck[j], deck[idx] = deck[idx], deck[j]
            out_cards[i, j] = deck[j]

        # Random rotations (0-3 per card)
        for j in range(6):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            out_rots[i, j] = np.int32((state >> np.uint64(33)) & np.uint64(3))

        # Random path index
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        out_pi[i] = np.int32((state >> np.uint64(33)) % np.uint64(n_paths))

        # Random tb config (0-31)
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        out_tb[i] = np.int32((state >> np.uint64(33)) & np.uint64(31))

        # Evaluate
        out_scores[i] = _eval_single(
            out_cards[i], out_rots[i], out_pi[i], out_tb[i],
            paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _eval_placements_batch(all_card_ids, all_rotations, all_path_idx, all_tb_idx,
                           paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr):
    """Evaluate a batch of fully-specified placements in parallel."""
    n = all_card_ids.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for i in nb.prange(n):
        scores[i] = _eval_single(
            all_card_ids[i], all_rotations[i], all_path_idx[i], all_tb_idx[i],
            paths, z_stacks, cbm, scoring, nac, a_qi, a_dx, a_dy, n_arr)
    return scores


@nb.njit(parallel=True, fastmath=True, cache=True)
def _prescan_shard_counts(n, n_cards, rng_seed, n_shards, shard_counts):
    """Fast prescan: run RNG for n samples, extract only the 6 card IDs,
    compute combo_key → shard_id, and count per shard.
    No evaluation, no rotation/path/tb — just card selection + shard assignment.
    ~100x faster than full generate_and_eval.
    Uses per-thread accumulators to avoid race conditions."""
    n_threads = nb.get_num_threads()
    local_counts = np.zeros((n_threads, n_shards), dtype=np.int64)
    for i in nb.prange(n):
        tid = nb.get_thread_id()
        state = np.uint64(rng_seed + i * np.uint64(6364136223846793005))

        # Fisher-Yates partial shuffle: pick 6 cards
        deck = np.empty(n_cards, dtype=np.int32)
        for j in range(n_cards):
            deck[j] = j
        cards = np.empty(6, dtype=np.int32)
        for j in range(6):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            idx = j + np.int32((state >> np.uint64(33)) % np.uint64(n_cards - j))
            deck[j], deck[idx] = deck[idx], deck[j]
            cards[j] = deck[j]

        # Sort cards (insertion sort on 6 elements)
        for a in range(1, 6):
            key = cards[a]
            b = a - 1
            while b >= 0 and cards[b] > key:
                cards[b + 1] = cards[b]
                b -= 1
            cards[b + 1] = key

        # Pack as combo key
        combo_key = np.uint64(0)
        for j in range(6):
            combo_key |= np.uint64(cards[j]) << np.uint64(j * 8)

        # Hash mix (splitmix64 finalizer) for uniform shard distribution
        h = combo_key
        h ^= h >> np.uint64(30)
        h = h * np.uint64(0xbf58476d1ce4e5b9)
        h ^= h >> np.uint64(27)
        h = h * np.uint64(0x94d049bb133111eb)
        h ^= h >> np.uint64(31)

        shard_id = np.int64(h % np.uint64(n_shards))
        local_counts[tid, shard_id] += 1

    # Reduce per-thread counts
    for s in range(n_shards):
        total = np.int64(0)
        for t in range(n_threads):
            total += local_counts[t, s]
        shard_counts[s] += total


def evaluate_placements_batch(all_card_ids, all_rotations, all_path_idx, all_tb_idx,
                              paths, card_data):
    """Python wrapper for batch placement evaluation."""
    return _eval_placements_batch(
        all_card_ids, all_rotations, all_path_idx, all_tb_idx,
        paths, Z_STACKS,
        card_data['cbm'], card_data['scoring'], card_data['nac'],
        card_data['a_qi'], card_data['a_dx'], card_data['a_dy'],
        card_data['n_arr']
    )


def prescan_shards(n_placements, n_cards, seed, n_shards=4096, chunk_size=50_000_000):
    """Fast prescan: determine how many placements go to each shard.
    Runs RNG only (no evaluation), ~100x faster than full generation.
    Returns shard_counts array of shape (n_shards,)."""
    shard_counts = np.zeros(n_shards, dtype=np.int64)

    # Warmup JIT
    warm = np.zeros(n_shards, dtype=np.int64)
    _prescan_shard_counts(8, n_cards, np.uint64(seed), n_shards, warm)

    processed = 0
    t_start = time.time()
    while processed < n_placements:
        chunk = min(chunk_size, n_placements - processed)
        chunk_seed = np.uint64(seed) + np.uint64(processed)
        _prescan_shard_counts(chunk, n_cards, chunk_seed, n_shards, shard_counts)
        processed += chunk
        elapsed = time.time() - t_start
        rate = processed / elapsed
        eta = (n_placements - processed) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  Prescan: [{processed:,}/{n_placements:,}] "
            f"{processed/n_placements*100:.0f}% | {rate:,.0f}/s | ETA {eta:.0f}s   ")
        sys.stdout.flush()
    print(f"\n  Prescan done in {time.time()-t_start:.1f}s")
    return shard_counts


def generate_and_eval(n, n_cards, n_paths, rng_seed, paths, card_data):
    """Generate n random placements and evaluate them. Returns structured numpy array.
    All RNG + compute happens inside Numba (no Python loops)."""
    out_cards = np.empty((n, 6), dtype=np.int32)
    out_rots = np.empty((n, 6), dtype=np.int32)
    out_pi = np.empty(n, dtype=np.int32)
    out_tb = np.empty(n, dtype=np.int32)
    out_scores = np.empty(n, dtype=np.float64)

    _generate_and_eval(
        n, n_cards, n_paths, np.uint64(rng_seed),
        paths, Z_STACKS,
        card_data['cbm'], card_data['scoring'], card_data['nac'],
        card_data['a_qi'], card_data['a_dx'], card_data['a_dy'],
        card_data['n_arr'],
        out_cards, out_rots, out_pi, out_tb, out_scores)

    return out_cards, out_rots, out_pi, out_tb, out_scores


def evaluate_scenario(card_ids_list, paths, card_data):
    """Python wrapper for single-scenario evaluation."""
    cids = np.array(card_ids_list, dtype=np.int32)
    return _eval_scenario(
        cids, paths, Z_STACKS,
        card_data['cbm'], card_data['scoring'], card_data['nac'],
        card_data['a_qi'], card_data['a_dx'], card_data['a_dy'],
        card_data['n_arr'], card_data['card_has_arrows']
    )


def evaluate_batch(all_card_ids_list, paths, card_data):
    """Python wrapper for batch scenario evaluation.
    all_card_ids_list: list of tuples/lists of 6 card ids."""
    cids = np.array(all_card_ids_list, dtype=np.int32)
    return _eval_batch(
        cids, paths, Z_STACKS,
        card_data['cbm'], card_data['scoring'], card_data['nac'],
        card_data['a_qi'], card_data['a_dx'], card_data['a_dy'],
        card_data['n_arr'], card_data['card_has_arrows']
    )


# ============================================================
# PERSISTENCE (SQLite)
# ============================================================
class SimDB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS scenarios(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_ids TEXT UNIQUE, mean_score REAL, elapsed REAL,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS card_stats(
                card_id INTEGER PRIMARY KEY,
                score_sum REAL DEFAULT 0, score_sq REAL DEFAULT 0, cnt INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS placements(
                card_ids INTEGER NOT NULL,
                rotations INTEGER NOT NULL,
                path_idx INTEGER NOT NULL,
                tb_config INTEGER NOT NULL,
                score REAL NOT NULL
            );
        """)
        self.conn.commit()

    # ...existing get_config, set_config, completed methods...

    def get_config(self, k, default=None):
        r = self.conn.execute("SELECT value FROM config WHERE key=?", (k,)).fetchone()
        return r[0] if r else default

    def set_config(self, k, v):
        self.conn.execute("INSERT OR REPLACE INTO config VALUES(?,?)", (k, str(v)))
        self.conn.commit()

    def completed(self):
        return self.conn.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]

    def completed_placements(self):
        """Count total placement entries."""
        return self.conn.execute("SELECT COUNT(*) FROM placements").fetchone()[0]

    def is_done(self, cards):
        key = ','.join(str(c) for c in cards)
        return self.conn.execute("SELECT 1 FROM scenarios WHERE card_ids=?", (key,)).fetchone() is not None

    def save_batch(self, results):
        c = self.conn.cursor()
        c.execute("BEGIN")
        for cards, score, elapsed in results:
            key = ','.join(str(x) for x in cards)
            try:
                c.execute("INSERT INTO scenarios(card_ids,mean_score,elapsed) VALUES(?,?,?)",
                          (key, score, elapsed))
            except sqlite3.IntegrityError:
                continue
            for cid in cards:
                c.execute("""INSERT INTO card_stats(card_id,score_sum,score_sq,cnt)
                    VALUES(?,?,?,1) ON CONFLICT(card_id) DO UPDATE SET
                    score_sum=score_sum+excluded.score_sum,
                    score_sq=score_sq+excluded.score_sq, cnt=cnt+1""",
                          (int(cid), score, score*score))
        self.conn.commit()

    def save_placements(self, card_ids_arr, rotations_arr, path_idx_arr, tb_idx_arr, scores_arr):
        """Save a batch of fully-specified placements using packed integers + executemany."""
        # Pack card_ids: 6 values (0-119) into int64, 8 bits each
        ci = card_ids_arr.astype(np.int64)
        cards_packed = ci[:, 0] | (ci[:, 1] << 8) | (ci[:, 2] << 16) | \
                       (ci[:, 3] << 24) | (ci[:, 4] << 32) | (ci[:, 5] << 40)
        # Pack rotations: 6 values (0-3) into int16, 2 bits each
        ri = rotations_arr.astype(np.int64)
        rots_packed = ri[:, 0] | (ri[:, 1] << 2) | (ri[:, 2] << 4) | \
                      (ri[:, 3] << 6) | (ri[:, 4] << 8) | (ri[:, 5] << 10)
        # Build row tuples and bulk insert
        rows = list(zip(cards_packed.tolist(), rots_packed.tolist(),
                        path_idx_arr.tolist(), tb_idx_arr.tolist(), scores_arr.tolist()))
        self.conn.execute("BEGIN")
        self.conn.executemany(
            "INSERT INTO placements(card_ids,rotations,path_idx,tb_config,score) VALUES(?,?,?,?,?)", rows)
        self.conn.commit()

    @staticmethod
    def unpack_cards(packed):
        """Unpack int64 → list of 6 card IDs."""
        return [(packed >> (i * 8)) & 0xFF for i in range(6)]

    @staticmethod
    def unpack_rots(packed):
        """Unpack int → list of 6 rotations."""
        return [(packed >> (i * 2)) & 0x3 for i in range(6)]

    def rankings(self):
        rows = self.conn.execute(
            "SELECT card_id,score_sum,score_sq,cnt FROM card_stats ORDER BY score_sum/cnt DESC"
        ).fetchall()
        out = []
        for cid, ss, ssq, cnt in rows:
            m = ss/cnt
            v = max(0, ssq/cnt - m*m) if cnt > 1 else 0
            out.append((cid, m, v**0.5, cnt))
        return out

    def total_time(self):
        return self.conn.execute("SELECT COALESCE(SUM(elapsed),0) FROM scenarios").fetchone()[0]

    def close(self):
        self.conn.close()


# ============================================================
# MAIN SIMULATION DRIVER
# ============================================================
def run(json_path, db_path, n_scenarios, seed, max_paths, batch_size, sim_batch, n_threads):
    print("Pirate Card Simulation (Numba JIT)")
    print("=" * 50)

    # Thread control (M1 Max: use 8 perf cores, avoid 2 efficiency cores)
    if n_threads > 0:
        nb.set_num_threads(n_threads)
    print(f"  Numba threads: {nb.get_num_threads()}")

    cd = load_cards(json_path)
    print(f"  {cd['n_cards']} cards loaded")
    arrow_cards = int(cd['card_has_arrows'].sum())
    print(f"  {arrow_cards}/{cd['n_cards']} cards have arrows")

    paths = get_paths()
    if max_paths > 0 and max_paths < len(paths):
        rng_paths = np.random.default_rng(seed)
        idx = rng_paths.choice(len(paths), max_paths, replace=False)
        paths = paths[idx]
    print(f"  {len(paths)} position paths")

    # Auto-select mode: single (prange over paths) is faster for most cases
    use_batch = False
    if sim_batch == 0:
        sim_batch = 1
    if sim_batch > 1:
        use_batch = True

    mode_str = f"batch (prange over scenarios, chunk={sim_batch})" if use_batch else "per-scenario (prange over paths)"
    print(f"  Mode: {mode_str}")

    # Warm up Numba JIT (first call compiles)
    print("  Compiling JIT (first run only)...", end=' ', flush=True)
    t0 = time.time()
    _ = evaluate_scenario([0,1,2,3,4,5], paths[:2], cd)
    if use_batch:
        _ = evaluate_batch([[0,1,2,3,4,5]], paths[:2], cd)
    print(f"done in {time.time()-t0:.1f}s")

    db = SimDB(db_path)
    stored = db.get_config('seed')
    if stored is None:
        db.set_config('seed', seed)
    elif int(stored) != seed:
        seed = int(stored)
        print(f"  Using stored seed={seed}")

    done = db.completed()
    remaining = max(0, n_scenarios - done)
    print(f"  Target: {n_scenarios} | Done: {done} | Remaining: {remaining}")

    if remaining == 0:
        _show_rankings(db)
        db.close()
        return

    rng = random.Random(seed)
    todo = []
    for _ in range(n_scenarios):
        cards = tuple(rng.sample(range(cd['n_cards']), 6))
        if not db.is_done(cards) and len(todo) < remaining:
            todo.append(cards)

    print(f"  Running {len(todo)} scenarios...\n")
    t_start = time.time()
    batch = []

    if use_batch:
        # Batch mode: prange over scenarios, sequential paths per core
        for chunk_start in range(0, len(todo), sim_batch):
            chunk = todo[chunk_start:chunk_start + sim_batch]
            t0 = time.time()
            scores = evaluate_batch(list(chunk), paths, cd)
            elapsed = time.time() - t0
            per_sc = elapsed / len(chunk)
            for j, cards in enumerate(chunk):
                batch.append((cards, float(scores[j]), per_sc))

            if len(batch) >= batch_size:
                db.save_batch(batch)
                batch = []

            done_now = chunk_start + len(chunk)
            total_el = time.time() - t_start
            rate = done_now / total_el
            eta = (len(todo) - done_now) / rate if rate > 0 else 0
            pct = done_now / len(todo) * 100
            sys.stdout.write(
                f"\r  [{done_now}/{len(todo)}] {pct:.0f}% | {rate:.2f}/s | "
                f"{per_sc:.1f}s/sc | ETA {eta:.0f}s   ")
            sys.stdout.flush()
    else:
        # Per-scenario mode: prange over paths within each scenario
        for i, cards in enumerate(todo):
            t0 = time.time()
            score = evaluate_scenario(list(cards), paths, cd)
            elapsed = time.time() - t0
            batch.append((cards, score, elapsed))

            if len(batch) >= batch_size:
                db.save_batch(batch)
                batch = []

            if (i + 1) % max(1, len(todo) // 50) == 0 or i == len(todo) - 1:
                total_el = time.time() - t_start
                rate = (i + 1) / total_el
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                pct = (i + 1) / len(todo) * 100
                sys.stdout.write(
                    f"\r  [{i+1}/{len(todo)}] {pct:.0f}% | {rate:.2f}/s | "
                    f"{elapsed:.1f}s last | ETA {eta:.0f}s   ")
                sys.stdout.flush()

    if batch:
        db.save_batch(batch)

    total = time.time() - t_start
    print(f"\n\n  Done: {len(todo)} scenarios in {total:.1f}s ({total/len(todo):.2f}s avg)")
    _show_rankings(db)
    db.close()


def run_placements(json_path, store_dir, n_placements, seed, batch_size, n_threads,
                    sharded=False, n_shards=4096, n_cards_override=0):
    """Monte Carlo simulation with binary numpy storage.
    Each placement = one fully random configuration (cards, rotations, path, tb) → 1 deterministic score.
    Storage: append-only binary file (16 bytes/record). RNG state saved for exact resume.
    No duplicates: deterministic per-sample seeding from (base_seed + sample_index).
    If sharded=True, writes to hash-partitioned shard files for efficient combo grouping.
    If n_cards_override > 0, restricts card selection to first N cards."""
    mode = "SHARDED" if sharded else "FLAT"
    print(f"Pirate Card Simulation — PLACEMENTS mode (Monte Carlo, {mode})")
    print("=" * 50)

    if n_threads > 0:
        nb.set_num_threads(n_threads)
    print(f"  Numba threads: {nb.get_num_threads()}")

    cd = load_cards(json_path)
    n_cards = cd['n_cards']
    if n_cards_override > 0 and n_cards_override < n_cards:
        n_cards = n_cards_override
        print(f"  Restricting to first {n_cards} cards (from {cd['n_cards']})")
        from math import comb
        print(f"  C({n_cards},6) = {comb(n_cards,6):,} possible combos")
        print(f"  Expected placements/combo: {n_placements / comb(n_cards,6):,.0f}")
    print(f"  {n_cards} cards loaded")

    paths = get_paths()
    n_paths = len(paths)
    print(f"  {n_paths} position paths")

    # Setup binary store
    if sharded:
        store = ShardedPlacementStore(store_dir, n_shards=n_shards)
        print(f"  Storage: {store_dir}/ ({n_shards} shards, {PlacementStore.RECORD_SIZE} bytes/record)")
    else:
        store = PlacementStore(store_dir)
        print(f"  Storage: {store_dir}/  ({PlacementStore.RECORD_SIZE} bytes/record)")

    done = store.count()
    remaining = max(0, n_placements - done)
    print(f"  Target: {n_placements:,} | Done: {done:,} | Remaining: {remaining:,}")

    if remaining == 0:
        if not sharded:
            _show_placement_stats(store)
        else:
            print(f"  All {n_placements:,} placements already generated.")
        return

    # Warm up JIT (compile _generate_and_eval)
    print("  Compiling JIT (first run only)...", end=' ', flush=True)
    t0 = time.time()
    _ = generate_and_eval(8, n_cards, n_paths, seed, paths[:2], cd)
    print(f"done in {time.time()-t0:.1f}s")

    # Save metadata (flat store only)
    if not sharded:
        store.save_metadata(seed=seed, n_cards=n_cards, n_paths=n_paths,
                            json_path=json_path, target=n_placements)

    print(f"  Running {remaining:,} random placements...\n")
    t_start = time.time()
    generated = 0
    rate = 0.0

    while generated < remaining:
        chunk = min(batch_size, remaining - generated)

        # RNG seed for this chunk: base_seed offset by global sample index
        # Each sample i gets seed = (seed + done + generated + i), done in Numba
        chunk_seed = np.uint64(seed) + np.uint64(done + generated)

        # Generate + evaluate entirely in Numba (no Python loops!)
        t0 = time.time()
        out_cards, out_rots, out_pi, out_tb, out_scores = \
            generate_and_eval(chunk, n_cards, n_paths, int(chunk_seed), paths, cd)
        t_compute = time.time() - t0

        # Append to binary file
        store.append(out_cards, out_rots, out_pi, out_tb, out_scores)
        generated += chunk

        # Progress
        total_el = time.time() - t_start
        rate = generated / total_el
        eta = (remaining - generated) / rate if rate > 0 else 0
        pct = generated / remaining * 100
        sys.stdout.write(
            f"\r  [{generated:,}/{remaining:,}] {pct:.0f}% | {rate:,.0f} placements/s | "
            f"compute {chunk/t_compute:,.0f}/s | ETA {eta:.0f}s   ")
        sys.stdout.flush()

    total = time.time() - t_start
    print(f"\n\n  Done: {remaining:,} placements in {total:.1f}s ({rate:,.0f}/s)")
    est_size = (done + remaining) * PlacementStore.RECORD_SIZE
    print(f"  Total data: {est_size / 1e9:.1f} GB")
    if not sharded:
        _show_placement_stats(store)
    else:
        # Flush remaining buffered records
        result = store.flush()
        if result:
            n_flushed, n_shards_written = result
            print(f"  Final flush: {n_flushed:,} records to {n_shards_written} shards")
        final = store.count()
        print(f"  Sharded store: {final:,} total records across {n_shards} shards")


# ============================================================
# PLACEMENT BINARY STORAGE
# ============================================================
# Structured dtype: 16 bytes per record, minimal and cache-friendly
PLACEMENT_DTYPE = np.dtype([
    ('card_ids', np.int64),     # packed: 6 cards in 8 bits each (bits 0-47)
    ('rotations', np.int16),    # packed: 6 rotations in 2 bits each (bits 0-11)
    ('path_idx', np.int32),     # position path index (0 to 729528)
    ('tb_config', np.uint8),    # top/bottom config (0-31)
    ('score', np.int8),         # deterministic integer score (~-15 to ~45)
], align=False)  # 8+2+4+1+1 = 16 bytes, no padding


class PlacementStore:
    """High-performance binary storage for Monte Carlo placement results.

    Format: append-only flat binary file of 16-byte records (numpy structured array).
    - 10M placements = 160 MB, 100M = 1.6 GB, 1B = 16 GB
    - Write speed: ~50M records/s (sequential disk append)
    - Read speed: zero-copy via np.memmap()
    - Resume: count = file_size / 16 → deterministic RNG picks up exactly where it left off
    """

    RECORD_SIZE = PLACEMENT_DTYPE.itemsize  # 16 bytes

    def __init__(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        self.dir = dir_path
        self.bin_path = os.path.join(dir_path, 'placements.bin')
        self.meta_path = os.path.join(dir_path, 'metadata.json')

    def count(self):
        """Number of stored placements (derived from file size)."""
        if not os.path.exists(self.bin_path):
            return 0
        return os.path.getsize(self.bin_path) // self.RECORD_SIZE

    def save_metadata(self, **kwargs):
        meta = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                meta = json.load(f)
        meta.update(kwargs)
        meta['count'] = self.count()
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def append(self, card_ids_arr, rotations_arr, path_idx_arr, tb_idx_arr, scores_arr):
        """Append a batch of placements to the binary file."""
        n = len(scores_arr)
        batch = np.empty(n, dtype=PLACEMENT_DTYPE)

        # Pack card_ids: 6 values (0-119) into int64, 8 bits each
        ci = card_ids_arr.astype(np.int64)
        batch['card_ids'] = ci[:, 0] | (ci[:, 1] << 8) | (ci[:, 2] << 16) | \
                            (ci[:, 3] << 24) | (ci[:, 4] << 32) | (ci[:, 5] << 40)

        # Pack rotations: 6 values (0-3) into int16, 2 bits each
        ri = rotations_arr.astype(np.int16)
        batch['rotations'] = ri[:, 0] | (ri[:, 1] << 2) | (ri[:, 2] << 4) | \
                             (ri[:, 3] << 6) | (ri[:, 4] << 8) | (ri[:, 5] << 10)

        batch['path_idx'] = path_idx_arr.astype(np.int32)
        batch['tb_config'] = tb_idx_arr.astype(np.uint8)
        batch['score'] = np.round(scores_arr).astype(np.int8)

        with open(self.bin_path, 'ab') as f:
            batch.tofile(f)

    def load_mmap(self):
        """Memory-map the placements file for zero-copy reading."""
        if not os.path.exists(self.bin_path):
            return np.array([], dtype=PLACEMENT_DTYPE)
        return np.memmap(self.bin_path, dtype=PLACEMENT_DTYPE, mode='r')

    @staticmethod
    def unpack_cards_scalar(packed):
        """Unpack int64 → list of 6 card IDs (for display)."""
        return [(int(packed) >> (i * 8)) & 0xFF for i in range(6)]

    @staticmethod
    def unpack_rots_scalar(packed):
        """Unpack int16 → list of 6 rotations (for display)."""
        return [(int(packed) >> (i * 2)) & 0x3 for i in range(6)]

    @staticmethod
    def unpack_cards_array(packed_arr):
        """Unpack int64 array → (N, 6) array of card IDs (vectorized)."""
        p = packed_arr.astype(np.int64)
        return np.column_stack([(p >> (i * 8)) & 0xFF for i in range(6)])

    @staticmethod
    def unpack_rots_array(packed_arr):
        """Unpack int16 array → (N, 6) array of rotations (vectorized)."""
        p = packed_arr.astype(np.int16)
        return np.column_stack([(p >> (i * 2)) & 0x3 for i in range(6)])

    @staticmethod
    def combo_key_from_cards(card_ids_arr):
        """Compute canonical combo key from (N, 6) card IDs array.
        Sorts each row so the key is order-independent.
        Returns int64 array of length N."""
        cs = np.sort(card_ids_arr, axis=1).astype(np.int64)
        return cs[:, 0] | (cs[:, 1] << 8) | (cs[:, 2] << 16) | \
               (cs[:, 3] << 24) | (cs[:, 4] << 32) | (cs[:, 5] << 40)

    @staticmethod
    def combo_key_from_packed(packed_card_ids):
        """Compute canonical combo key from packed card_ids (int64 array).
        Unpacks, sorts, repacks. Returns int64 array."""
        unpacked = PlacementStore.unpack_cards_array(packed_card_ids)
        return PlacementStore.combo_key_from_cards(unpacked)

    @staticmethod
    def combo_key_single(cards):
        """Compute combo key for a single list/tuple of 6 card IDs."""
        cs = sorted(cards)
        key = 0
        for i, c in enumerate(cs):
            key |= (int(c) & 0xFF) << (i * 8)
        return key


class ShardedPlacementStore:
    """Hash-partitioned storage for efficient grouping by card combination.

    Records are distributed across N shard files based on:
        combo_key(sorted_cards) % n_shards

    Each shard is a flat binary file of 16-byte records (same PLACEMENT_DTYPE).
    Typical shard size: 80 GB / 4096 shards ≈ 20 MB — fits in RAM easily.

    Write buffering: records accumulate in RAM per shard, flushing to disk
    when total buffered records exceed flush_threshold. This reduces file I/O
    from thousands of open/close per batch to ~100 flushes total for 5B records.

    Usage:
        store = ShardedPlacementStore('data/placements_sharded', n_shards=4096)
        store.append(card_ids, rotations, path_idx, tb_config, scores)
        store.flush()  # flush remaining buffered records at the end
        records = store.lookup_combo([33, 29, 79, 1, 14, 109])
    """

    RECORD_SIZE = PLACEMENT_DTYPE.itemsize  # 16 bytes

    def __init__(self, dir_path, n_shards=4096, flush_threshold=50_000_000):
        self.dir = dir_path
        self.n_shards = n_shards
        self.flush_threshold = flush_threshold  # flush when this many records buffered (~800 MB)
        self._buffers = {}       # shard_id -> list of record arrays
        self._buffered = 0       # total records in buffers
        os.makedirs(dir_path, exist_ok=True)

    def _shard_path(self, shard_id):
        return os.path.join(self.dir, f'shard_{shard_id:04d}.bin')

    def shard_for_combo(self, combo_key):
        """Determine shard index for a combo key (or array of keys).
        Uses splitmix64 hash mixing for uniform distribution."""
        h = np.asarray(combo_key, dtype=np.uint64).copy()
        with np.errstate(over='ignore'):
            h ^= h >> np.uint64(30)
            h = np.multiply(h, np.uint64(0xbf58476d1ce4e5b9), casting='unsafe')
            h ^= h >> np.uint64(27)
            h = np.multiply(h, np.uint64(0x94d049bb133111eb), casting='unsafe')
            h ^= h >> np.uint64(31)
        return h % np.uint64(self.n_shards)

    def count(self):
        """Total records across all shards (on disk only, excludes buffered)."""
        total = 0
        for i in range(self.n_shards):
            p = self._shard_path(i)
            if os.path.exists(p):
                total += os.path.getsize(p) // self.RECORD_SIZE
        return total + self._buffered

    def shard_counts(self):
        """Return array of record counts per shard."""
        counts = np.zeros(self.n_shards, dtype=np.int64)
        for i in range(self.n_shards):
            p = self._shard_path(i)
            if os.path.exists(p):
                counts[i] = os.path.getsize(p) // self.RECORD_SIZE
        return counts

    def flush(self):
        """Flush all buffered records to disk. Call at end of generation."""
        if self._buffered == 0:
            return
        for sid, chunks in self._buffers.items():
            combined = np.concatenate(chunks)
            with open(self._shard_path(sid), 'ab') as f:
                combined.tofile(f)
        n_shards_written = len(self._buffers)
        n_flushed = self._buffered
        self._buffers.clear()
        self._buffered = 0
        return n_flushed, n_shards_written

    def _distribute_to_buffers(self, records, combo_keys=None):
        """Add packed records to per-shard buffers. Flush if threshold reached."""
        if combo_keys is None:
            combo_keys = PlacementStore.combo_key_from_packed(records['card_ids'])
        shard_ids = self.shard_for_combo(combo_keys).astype(np.int64)
        n = len(records)

        # Group by shard
        order = np.argsort(shard_ids)
        shard_ids_sorted = shard_ids[order]
        records_sorted = records[order]
        boundaries = np.concatenate([
            [0],
            np.where(np.diff(shard_ids_sorted) != 0)[0] + 1,
            [n]
        ])

        for b in range(len(boundaries) - 1):
            start, end = boundaries[b], boundaries[b + 1]
            sid = int(shard_ids_sorted[start])
            if sid not in self._buffers:
                self._buffers[sid] = []
            self._buffers[sid].append(records_sorted[start:end].copy())

        self._buffered += n
        if self._buffered >= self.flush_threshold:
            self.flush()

    def append(self, card_ids_arr, rotations_arr, path_idx_arr, tb_idx_arr, scores_arr):
        """Append records to buffers, distributing to shards by combo key.
        Automatically flushes to disk when buffer threshold is reached."""
        n = len(scores_arr)
        batch = np.empty(n, dtype=PLACEMENT_DTYPE)

        ci = card_ids_arr.astype(np.int64)
        batch['card_ids'] = ci[:, 0] | (ci[:, 1] << 8) | (ci[:, 2] << 16) | \
                            (ci[:, 3] << 24) | (ci[:, 4] << 32) | (ci[:, 5] << 40)

        ri = rotations_arr.astype(np.int16)
        batch['rotations'] = ri[:, 0] | (ri[:, 1] << 2) | (ri[:, 2] << 4) | \
                             (ri[:, 3] << 6) | (ri[:, 4] << 8) | (ri[:, 5] << 10)

        batch['path_idx'] = path_idx_arr.astype(np.int32)
        batch['tb_config'] = tb_idx_arr.astype(np.uint8)
        batch['score'] = np.round(scores_arr).astype(np.int8)

        combo_keys = PlacementStore.combo_key_from_cards(card_ids_arr)
        self._distribute_to_buffers(batch, combo_keys)

    def append_raw(self, records, combo_keys=None):
        """Append pre-packed PLACEMENT_DTYPE records to buffers.
        If combo_keys is None, computes from packed card_ids."""
        self._distribute_to_buffers(records, combo_keys)

    def load_shard(self, shard_id):
        """Load all records from one shard as numpy structured array."""
        p = self._shard_path(shard_id)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return np.array([], dtype=PLACEMENT_DTYPE)
        return np.fromfile(p, dtype=PLACEMENT_DTYPE)

    def mmap_shard(self, shard_id):
        """Memory-map one shard for zero-copy reading."""
        p = self._shard_path(shard_id)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return np.array([], dtype=PLACEMENT_DTYPE)
        return np.memmap(p, dtype=PLACEMENT_DTYPE, mode='r')

    def lookup_combo(self, cards):
        """Find all records matching a 6-card combination (order-independent).

        Args:
            cards: list/tuple of 6 card IDs (any order)

        Returns:
            numpy structured array of matching records (PLACEMENT_DTYPE)
        """
        key = PlacementStore.combo_key_single(cards)
        shard_id = int(self.shard_for_combo(np.uint64(key)))
        data = self.load_shard(shard_id)
        if len(data) == 0:
            return data
        # Filter: compute combo keys for all records in shard, match
        combo_keys = PlacementStore.combo_key_from_packed(data['card_ids'])
        mask = combo_keys == key
        return data[mask]

    def combo_stats(self, cards):
        """Get score statistics for a card combination.

        Returns dict with mean, std, min, max, count, scores, or None if no data.
        """
        records = self.lookup_combo(cards)
        if len(records) == 0:
            return None
        scores = records['score'].astype(np.float64)
        return {
            'cards': sorted(cards),
            'count': len(records),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': int(scores.min()),
            'max': int(scores.max()),
            'p25': float(np.percentile(scores, 25)),
            'p75': float(np.percentile(scores, 75)),
            'scores': scores,
            'records': records,
        }

    def iter_shard_combos(self, shard_id, min_count=2):
        """Iterate over all combos in a shard with at least min_count records.

        Yields (combo_key, records) tuples, sorted by combo_key.
        """
        data = self.load_shard(shard_id)
        if len(data) == 0:
            return
        combo_keys = PlacementStore.combo_key_from_packed(data['card_ids'])
        order = np.argsort(combo_keys)
        keys_sorted = combo_keys[order]
        data_sorted = data[order]
        boundaries = np.concatenate([
            [0],
            np.where(np.diff(keys_sorted) != 0)[0] + 1,
            [len(data)]
        ])
        for b in range(len(boundaries) - 1):
            start, end = boundaries[b], boundaries[b + 1]
            if end - start >= min_count:
                yield int(keys_sorted[start]), data_sorted[start:end]


def _show_placement_stats(store, top=10):
    """Show summary stats from binary placement store using numpy analytics."""
    data = store.load_mmap()
    n = len(data)
    if n == 0:
        print("  No placement results yet.")
        return

    scores = data['score'].astype(np.float64)
    file_mb = n * PlacementStore.RECORD_SIZE / 1e6

    print(f"\n{'=' * 60}")
    print(f"  Placement Stats ({n:,} placements, {file_mb:.1f} MB)")
    print(f"{'=' * 60}")
    print(f"  Score range: {scores.min():.0f} to {scores.max():.0f} (avg {scores.mean():.2f}, std {scores.std():.2f})")

    # Top placements (partial sort for speed)
    top_k = min(top, n)
    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    print(f"\n  Top {top_k} placements:")
    print(f"  {'#':<4} {'Cards':<30} {'Rots':<16} {'Path':>7} {'TB':>3} {'Score':>5}")
    print(f"  {'-'*4} {'-'*30} {'-'*16} {'-'*7} {'-'*3} {'-'*5}")
    for rank, idx in enumerate(top_idx):
        cards = store.unpack_cards_scalar(data['card_ids'][idx])
        rots = store.unpack_rots_scalar(data['rotations'][idx])
        print(f"  {rank+1:<4} {','.join(map(str,cards)):<30} {','.join(map(str,rots)):<16} "
              f"{data['path_idx'][idx]:>7} {data['tb_config'][idx]:>3} {data['score'][idx]:>5}")

    # Per-card rankings (vectorized: unpack all card_ids, compute per-card avg)
    all_cards = store.unpack_cards_array(data['card_ids'])  # (N, 6)
    max_card = int(all_cards.max()) + 1
    card_score_sum = np.zeros(max_card, dtype=np.float64)
    card_count = np.zeros(max_card, dtype=np.int64)
    for k in range(6):
        np.add.at(card_score_sum, all_cards[:, k], scores)
        np.add.at(card_count, all_cards[:, k], 1)

    mask = card_count > 0
    card_avg = np.full(max_card, -999.0)
    card_avg[mask] = card_score_sum[mask] / card_count[mask]
    ranking = np.argsort(-card_avg)

    print(f"\n  Card rankings ({n:,} placements, ~{int(card_count[mask].mean()):,} samples/card):")
    print(f"  {'#':<4} {'Card':<6} {'AvgScore':>8} {'Count':>8}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*8}")
    shown = 0
    for cid in ranking:
        if card_count[cid] == 0:
            break
        print(f"  {shown+1:<4} {cid:<6} {card_avg[cid]:>8.2f} {card_count[cid]:>8,}")
        shown += 1
        if shown >= 20:
            break


def _show_rankings(db, top=20):
    ranks = db.rankings()
    if not ranks:
        print("  No results yet.")
        return
    n = db.completed()
    tt = db.total_time()
    print(f"\n{'=' * 60}")
    print(f"  Rankings ({n} scenarios, {tt:.0f}s compute)")
    print(f"{'=' * 60}")
    print(f"  {'#':<4} {'Card':<6} {'Mean':>8} {'Std':>8} {'Count':>6}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    for i, (cid, m, sd, cnt) in enumerate(ranks[:top]):
        print(f"  {i+1:<4} {cid:<6} {m:>8.2f} {sd:>8.2f} {cnt:>6}")
    if len(ranks) > top:
        print(f"  ... ({len(ranks) - top} more)")
        for cid, m, sd, cnt in ranks[-3:]:
            r = ranks.index((cid, m, sd, cnt)) + 1
            print(f"  {r:<4} {cid:<6} {m:>8.2f} {sd:>8.2f} {cnt:>6}")


def main():
    p = argparse.ArgumentParser(description='Pirate Card Simulation')
    p.add_argument('--json', default='decks/pirate_20_25.json')
    p.add_argument('--db', default='data/simulation.db')
    p.add_argument('--scenarios', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-paths', type=int, default=0, help='0=all 729K paths')
    p.add_argument('--batch-size', type=int, default=10,
                   help='DB save batch size (for --placements, also the Numba batch size; try 10000+)')
    p.add_argument('--sim-batch', type=int, default=0,
                   help='Scenarios per Numba call. 0=auto, 1=per-path parallel')
    p.add_argument('--threads', type=int, default=0,
                   help='Numba threads. 0=all cores. Try 8 on M1 Max (skip efficiency cores)')
    p.add_argument('--rankings', action='store_true', help='Show rankings and exit')
    p.add_argument('--placements', type=int, default=0,
                   help='Monte Carlo mode: number of random placements to generate. '
                        'Stored as binary numpy file (16 bytes/record)')
    p.add_argument('--out', default='data/placements',
                   help='Output directory for placement results (default: data/placements)')
    p.add_argument('--sharded', action='store_true',
                   help='Write placements to hash-partitioned shards (4096 files) for combo grouping')
    p.add_argument('--n-shards', type=int, default=4096,
                   help='Number of shards when --sharded is used (default: 4096)')
    p.add_argument('--prescan', action='store_true',
                   help='Preview shard distribution without generating (fast RNG-only pass)')
    p.add_argument('--n-cards', type=int, default=0,
                   help='Restrict to first N cards (0=all). E.g. 20 → C(20,6)=38,760 combos')
    a = p.parse_args()
    if a.rankings:
        db = SimDB(a.db)
        _show_rankings(db)
        db.close()
        return
    if a.prescan and a.placements > 0:
        if a.threads > 0:
            nb.set_num_threads(a.threads)
        n_c = a.n_cards if a.n_cards > 0 else 120
        print(f"  Prescan: {a.placements:,} placements → {a.n_shards} shards (seed={a.seed}, {n_c} cards)")
        counts = prescan_shards(a.placements, n_c, a.seed, a.n_shards)
        non_empty = counts[counts > 0]
        print(f"  Non-empty shards: {len(non_empty):,} / {a.n_shards}")
        print(f"  Records per shard: mean={non_empty.mean():,.0f}, "
              f"min={non_empty.min():,}, max={non_empty.max():,}, "
              f"std={non_empty.std():,.0f}")
        print(f"  Total: {counts.sum():,}")
        largest = np.argsort(-counts)[:10]
        print(f"  Largest shards: {', '.join(f'{i}({counts[i]:,})' for i in largest)}")
        est_size = counts.max() * PlacementStore.RECORD_SIZE
        print(f"  Largest shard file: {est_size / 1e6:.1f} MB")
        return
    if a.placements > 0:
        run_placements(a.json, a.out, a.placements, a.seed, a.batch_size, a.threads,
                       sharded=a.sharded, n_shards=a.n_shards,
                       n_cards_override=a.n_cards if a.n_cards > 0 else 0)
    else:
        run(a.json, a.db, a.scenarios, a.seed, a.max_paths, a.batch_size, a.sim_batch, a.threads)


if __name__ == '__main__':
    main()
