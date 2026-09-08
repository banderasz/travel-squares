//! High-performance pirate card placement simulation in Rust.
//!
//! Direct port of the Python/Numba simulation with native parallelism via Rayon.
//! Reads the same JSON card data and .npy path cache, writes to the same SQLite schema.

use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rusqlite::{params, Connection};
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

// ============================================================
// CONSTANTS
// ============================================================
const N_SYM: usize = 9;
const MAX_CT: usize = 11;
const CT1: usize = MAX_CT + 1; // 12
const GRID_OFF: i8 = 7;
const GRID_SZ: usize = 15;
const GS: usize = GRID_SZ * GRID_SZ;

const ARROW_DIR_DX: [i8; 4] = [0, 0, -1, 1]; // up, down, left, right
const ARROW_DIR_DY: [i8; 4] = [-1, 1, 0, 0];

/// Scoring tables indexed by [sym][count], count 0..=11
const SCORING: [[f64; CT1]; N_SYM] = [
    // anchor
    [0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0],
    // shark
    [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -3.0, -3.0, -4.0, -4.0, -5.0, -5.0],
    // rat
    [0.0, 0.0, -1.0, -1.0, -2.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
    // kraken
    [-4.0, -3.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // map
    [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0],
    // coin
    [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.0, 0.0],
    // rum
    [-2.0, -1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    // parrot
    [0.0, 3.0, 0.0, 5.0, 0.0, 8.0, 0.0, 10.0, 0.0, 13.0, 0.0, 15.0],
    // spyglass
    [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0],
];

fn sym_id(name: &str) -> usize {
    match name {
        "anchor" => 0,
        "shark" => 1,
        "rat" => 2,
        "kraken" => 3,
        "map" => 4,
        "coin" => 5,
        "rum" => 6,
        "parrot" => 7,
        "spyglass" => 8,
        "arrow_up" => 9,
        "arrow_down" => 10,
        "arrow_left" => 11,
        "arrow_right" => 12,
        _ => panic!("Unknown symbol: {name}"),
    }
}

// ============================================================
// CARD DATA
// ============================================================
/// All precomputed card arrays, flat for cache efficiency.
struct CardData {
    n_cards: usize,
    /// [card * 4 * 16 * 9 + rot * 16 * 9 + mask * 9 + sym]
    cbm: Vec<i8>,
    /// [card * 4 * 4 * 9 + rot * 4 * 9 + qi * 9 + sym]
    nac: Vec<i8>,
    /// [card * 4 * 4 + rot * 4 + ai]
    a_qi: Vec<i8>,
    a_dx: Vec<i8>,
    a_dy: Vec<i8>,
    /// [card * 4 + rot]
    n_arr: Vec<i8>,
    card_has_arrows: Vec<bool>,
}

impl CardData {
    #[inline(always)]
    fn cbm(&self, card: usize, rot: usize, mask: usize, sym: usize) -> i8 {
        self.cbm[card * (4 * 16 * N_SYM) + rot * (16 * N_SYM) + mask * N_SYM + sym]
    }
    #[inline(always)]
    fn nac(&self, card: usize, rot: usize, qi: usize, sym: usize) -> i8 {
        self.nac[card * (4 * 4 * N_SYM) + rot * (4 * N_SYM) + qi * N_SYM + sym]
    }
    #[inline(always)]
    fn a_qi(&self, card: usize, rot: usize, ai: usize) -> i8 {
        self.a_qi[card * 16 + rot * 4 + ai]
    }
    #[inline(always)]
    fn a_dx(&self, card: usize, rot: usize, ai: usize) -> i8 {
        self.a_dx[card * 16 + rot * 4 + ai]
    }
    #[inline(always)]
    fn a_dy(&self, card: usize, rot: usize, ai: usize) -> i8 {
        self.a_dy[card * 16 + rot * 4 + ai]
    }
    #[inline(always)]
    fn n_arr(&self, card: usize, rot: usize) -> i8 {
        self.n_arr[card * 4 + rot]
    }

    // Unsafe fast accessors — skip bounds checks (indices guaranteed valid by construction)
    #[inline(always)]
    unsafe fn cbm_fast(&self, card: usize, rot: usize, mask: usize, sym: usize) -> i8 {
        *self.cbm.get_unchecked(card * (4 * 16 * N_SYM) + rot * (16 * N_SYM) + mask * N_SYM + sym)
    }
    #[inline(always)]
    unsafe fn nac_fast(&self, card: usize, rot: usize, qi: usize, sym: usize) -> i8 {
        *self.nac.get_unchecked(card * (4 * 4 * N_SYM) + rot * (4 * N_SYM) + qi * N_SYM + sym)
    }
    #[inline(always)]
    unsafe fn a_qi_fast(&self, card: usize, rot: usize, ai: usize) -> i8 {
        *self.a_qi.get_unchecked(card * 16 + rot * 4 + ai)
    }
    #[inline(always)]
    unsafe fn a_dx_fast(&self, card: usize, rot: usize, ai: usize) -> i8 {
        *self.a_dx.get_unchecked(card * 16 + rot * 4 + ai)
    }
    #[inline(always)]
    unsafe fn a_dy_fast(&self, card: usize, rot: usize, ai: usize) -> i8 {
        *self.a_dy.get_unchecked(card * 16 + rot * 4 + ai)
    }
    #[inline(always)]
    unsafe fn n_arr_fast(&self, card: usize, rot: usize) -> i8 {
        *self.n_arr.get_unchecked(card * 4 + rot)
    }
}

// ============================================================
// JSON LOADING
// ============================================================
#[derive(Deserialize)]
struct CardJson {
    card: CardInner,
}
#[derive(Deserialize)]
struct CardInner {
    quarters: Quarters,
}
#[derive(Deserialize)]
struct Quarters {
    top_left: Vec<String>,
    top_right: Vec<String>,
    bottom_left: Vec<String>,
    bottom_right: Vec<String>,
}

const ROT_MAP: [usize; 4] = [2, 0, 3, 1];
const ARROW_ROT: [(usize, usize); 4] = [(9, 12), (12, 10), (10, 11), (11, 9)];

fn rotate_sym(s: usize) -> usize {
    for &(from, to) in &ARROW_ROT {
        if s == from {
            return to;
        }
    }
    s
}

fn rotate_card(quarters: &[Vec<usize>; 4]) -> [Vec<usize>; 4] {
    [
        quarters[ROT_MAP[0]].iter().map(|&s| rotate_sym(s)).collect(),
        quarters[ROT_MAP[1]].iter().map(|&s| rotate_sym(s)).collect(),
        quarters[ROT_MAP[2]].iter().map(|&s| rotate_sym(s)).collect(),
        quarters[ROT_MAP[3]].iter().map(|&s| rotate_sym(s)).collect(),
    ]
}

fn load_cards(json_path: &str) -> CardData {
    let file = File::open(json_path).expect("Cannot open card JSON");
    let raw: Vec<CardJson> = serde_json::from_reader(file).expect("Invalid JSON");
    let n = raw.len();

    let mut nac = vec![0i8; n * 4 * 4 * N_SYM];
    let mut a_qi = vec![-1i8; n * 4 * 4];
    let mut a_dx = vec![0i8; n * 4 * 4];
    let mut a_dy = vec![0i8; n * 4 * 4];
    let mut n_arr = vec![0i8; n * 4];

    for (ci, card) in raw.iter().enumerate() {
        let q = &card.card.quarters;
        let rot0: [Vec<usize>; 4] = [
            q.top_left.iter().map(|s| sym_id(s)).collect(),
            q.top_right.iter().map(|s| sym_id(s)).collect(),
            q.bottom_left.iter().map(|s| sym_id(s)).collect(),
            q.bottom_right.iter().map(|s| sym_id(s)).collect(),
        ];
        let mut cur = rot0;
        for r in 0..4 {
            if r > 0 {
                cur = rotate_card(&cur);
            }
            let mut ai = 0usize;
            for qi in 0..4 {
                for &s in &cur[qi] {
                    if s < N_SYM {
                        nac[ci * (4 * 4 * N_SYM) + r * (4 * N_SYM) + qi * N_SYM + s] += 1;
                    } else {
                        let dir = s - 9;
                        a_qi[ci * 16 + r * 4 + ai] = qi as i8;
                        a_dx[ci * 16 + r * 4 + ai] = ARROW_DIR_DX[dir];
                        a_dy[ci * 16 + r * 4 + ai] = ARROW_DIR_DY[dir];
                        ai += 1;
                    }
                }
            }
            n_arr[ci * 4 + r] = ai as i8;
        }
    }

    // Precompute cbm[card, rot, mask, sym]
    let q_dx: [i8; 4] = [0, 1, 0, 1];
    let q_dy: [i8; 4] = [0, 0, 1, 1];
    let mut cbm = vec![0i8; n * 4 * 16 * N_SYM];

    for ci in 0..n {
        for r in 0..4 {
            for mask in 0..16u8 {
                for qi in 0..4 {
                    if mask & (1 << qi) != 0 {
                        for s in 0..N_SYM {
                            cbm[ci * (4 * 16 * N_SYM) + r * (16 * N_SYM) + (mask as usize) * N_SYM + s]
                                += nac[ci * (4 * 4 * N_SYM) + r * (4 * N_SYM) + qi * N_SYM + s];
                        }
                    }
                }
                let na = n_arr[ci * 4 + r] as usize;
                for ai in 0..na {
                    let src = a_qi[ci * 16 + r * 4 + ai];
                    if mask & (1 << src) == 0 {
                        continue;
                    }
                    let tx = q_dx[src as usize] + a_dx[ci * 16 + r * 4 + ai];
                    let ty = q_dy[src as usize] + a_dy[ci * 16 + r * 4 + ai];
                    if tx >= 0 && tx <= 1 && ty >= 0 && ty <= 1 {
                        let tq = (tx + ty * 2) as usize;
                        if mask & (1 << tq) != 0 {
                            for s in 0..N_SYM {
                                cbm[ci * (4 * 16 * N_SYM) + r * (16 * N_SYM) + (mask as usize) * N_SYM + s]
                                    += nac[ci * (4 * 4 * N_SYM) + r * (4 * N_SYM) + tq * N_SYM + s];
                            }
                        }
                    }
                }
            }
        }
    }

    let mut card_has_arrows = vec![false; n];
    for ci in 0..n {
        for r in 0..4 {
            if n_arr[ci * 4 + r] > 0 {
                card_has_arrows[ci] = true;
                break;
            }
        }
    }

    CardData {
        n_cards: n,
        cbm,
        nac,
        a_qi,
        a_dx,
        a_dy,
        n_arr,
        card_has_arrows,
    }
}

// ============================================================
// Z-STACKS
// ============================================================
fn make_z_stacks() -> [[u8; 6]; 32] {
    let mut zs = [[0u8; 6]; 32];
    for tb in 0..32u32 {
        let mut stack = vec![0u8];
        for k in 0..5u8 {
            if tb & (1 << k) != 0 {
                stack.push(k + 1);
            } else {
                stack.insert(0, k + 1);
            }
        }
        for (i, &v) in stack.iter().enumerate() {
            zs[tb as usize][i] = v;
        }
    }
    zs
}

// ============================================================
// PATH LOADING (.npy)
// ============================================================
fn load_paths_npy(path: &str) -> (Vec<i8>, usize) {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {path}"));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    assert_eq!(&buf[0..6], b"\x93NUMPY", "Not a .npy file");
    let major = buf[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    };
    let data_offset = if major == 1 { 10 } else { 12 } + header_len;

    let header_str = std::str::from_utf8(&buf[if major == 1 { 10 } else { 12 }..data_offset]).unwrap();
    let shape_start = header_str.find("'shape': (").unwrap() + 10;
    let shape_end = header_str[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();

    assert_eq!(dims.len(), 3);
    let n_paths = dims[0];

    let data: Vec<i8> = buf[data_offset..data_offset + n_paths * 12]
        .iter()
        .map(|&b| b as i8)
        .collect();

    (data, n_paths)
}

fn get_paths(npy_path: &str) -> (Vec<i8>, usize) {
    if Path::new(npy_path).exists() {
        let (data, n) = load_paths_npy(npy_path);
        eprintln!("  Loaded {n} paths from {npy_path}");
        (data, n)
    } else {
        eprintln!("  .npy cache not found at {npy_path}");
        eprintln!("  Run the Python version first to generate paths.npy, or wait for enumeration...");
        eprintln!("  Computing position paths (one-time)...");
        let raw = enum_paths();
        let n = raw.len();
        let data: Vec<i8> = raw.into_iter().flatten().collect();
        eprintln!("  Computed {n} paths");
        (data, n)
    }
}

fn card_cells(px: i32, py: i32) -> [(i32, i32); 4] {
    [(px, py), (px + 1, py), (px, py + 1), (px + 1, py + 1)]
}

fn enum_paths() -> Vec<[i8; 12]> {
    let mut all_paths = Vec::new();
    let mut positions = vec![(0i32, 0i32)];
    let mut footprint = std::collections::HashSet::new();
    for &c in &card_cells(0, 0) {
        footprint.insert(c);
    }

    fn dfs(
        d: usize,
        positions: &mut Vec<(i32, i32)>,
        footprint: &mut std::collections::HashSet<(i32, i32)>,
        all_paths: &mut Vec<[i8; 12]>,
    ) {
        if d == 6 {
            let mut p = [0i8; 12];
            for (i, &(px, py)) in positions.iter().enumerate() {
                p[i * 2] = px as i8;
                p[i * 2 + 1] = py as i8;
            }
            all_paths.push(p);
            return;
        }
        let mn_x = footprint.iter().map(|&(x, _)| x).min().unwrap() - 1;
        let mx_x = footprint.iter().map(|&(x, _)| x).max().unwrap();
        let mn_y = footprint.iter().map(|&(_, y)| y).min().unwrap() - 1;
        let mx_y = footprint.iter().map(|&(_, y)| y).max().unwrap();
        for px in mn_x..=mx_x {
            for py in mn_y..=mx_y {
                let cells = card_cells(px, py);
                if cells.iter().any(|c| footprint.contains(c)) {
                    positions.push((px, py));
                    let added: Vec<_> = cells.iter().filter(|c| !footprint.contains(c)).copied().collect();
                    for &c in &added {
                        footprint.insert(c);
                    }
                    dfs(d + 1, positions, footprint, all_paths);
                    positions.pop();
                    for &c in &added {
                        footprint.remove(&c);
                    }
                }
            }
        }
    }

    dfs(1, &mut positions, &mut footprint, &mut all_paths);
    all_paths
}

// ============================================================
// CORE EVALUATION — OPTIMIZED HOT LOOP
//   Key optimization: visibility dedup — skip redundant tb configs
//   that produce identical visibility (and thus identical scores).
//   Average savings: ~26% of Phase 2 evaluations.
// ============================================================

/// Pack 6 visibility masks into a u64 for fast comparison.
#[inline(always)]
fn vis_key(vis: &[i8; 6]) -> u64 {
    (vis[0] as u8 as u64)
        | ((vis[1] as u8 as u64) << 8)
        | ((vis[2] as u8 as u64) << 16)
        | ((vis[3] as u8 as u64) << 24)
        | ((vis[4] as u8 as u64) << 32)
        | ((vis[5] as u8 as u64) << 40)
}

#[inline(always)]
fn eval_path(
    pi: usize,
    card_ids: &[u32; 6],
    any_arrows: bool,
    paths: &[i8],
    z_stacks: &[[u8; 6]; 32],
    cd: &CardData,
) -> f64 {
    let mut grid = [-1i8; GS];
    let mut gqi = [0i8; GS];
    let mut touched = [0u32; 24];
    let mut all_vis = [[0i8; 6]; 32];
    let mut pmf = [0.0f64; CT1];
    let mut new_pmf = [0.0f64; CT1];
    let mut grp_pmf = [0.0f64; CT1];
    let mut lc = [[[0i8; N_SYM]; 4]; 6];

    let mut all_n_cross = [0u32; 32];
    let mut all_cs_k = [[0u8; 24]; 32];
    let mut all_cs_r = [[0u8; 24]; 32];
    let mut all_ct_k = [[0u8; 24]; 32];
    let mut all_ct_qi = [[0u8; 24]; 32];

    let mut parent = [0u8; 6];
    let mut gid = [0u8; 6];
    let mut gsz = [0u8; 6];
    let mut gmap = [0i8; 6];
    let mut gmem = [[0u8; 6]; 6];
    let mut gfill = [0u8; 6];

    let path_base = pi * 12;

    // ---- PHASE 1: Batch compute visibility + cross-card arrows ----
    for tbi in 0..32 {
        let mut nt = 0usize;
        for zi in 0..6 {
            let k = z_stacks[tbi][zi] as usize;
            let px = paths[path_base + k * 2] as i32;
            let py = paths[path_base + k * 2 + 1] as i32;
            for qi in 0..4u32 {
                let idx = ((px + (qi & 1) as i32 + GRID_OFF as i32) * GRID_SZ as i32
                    + (py + (qi >> 1) as i32 + GRID_OFF as i32)) as usize;
                if grid[idx] == -1 {
                    touched[nt] = idx as u32;
                    nt += 1;
                }
                grid[idx] = k as i8;
                gqi[idx] = qi as i8;
            }
        }
        for ti in 0..nt {
            let idx = touched[ti] as usize;
            all_vis[tbi][grid[idx] as usize] |= 1i8 << gqi[idx];
        }

        if any_arrows {
            let mut nc = 0usize;
            for k in 0..6 {
                let cid = card_ids[k] as usize;
                let m = all_vis[tbi][k];
                let px = paths[path_base + k * 2] as i32;
                let py = paths[path_base + k * 2 + 1] as i32;
                for r in 0..4 {
                    let na = cd.n_arr(cid, r) as usize;
                    for ai in 0..na {
                        let src = cd.a_qi(cid, r, ai);
                        if m & (1 << src) == 0 { continue; }
                        let tgx = px + (src & 1) as i32 + GRID_OFF as i32 + cd.a_dx(cid, r, ai) as i32;
                        let tgy = py + (src >> 1) as i32 + GRID_OFF as i32 + cd.a_dy(cid, r, ai) as i32;
                        if tgx >= 0 && tgx < GRID_SZ as i32 && tgy >= 0 && tgy < GRID_SZ as i32 {
                            let tidx = (tgx * GRID_SZ as i32 + tgy) as usize;
                            let tk = grid[tidx];
                            if tk >= 0 && tk as usize != k && nc < 24 {
                                all_cs_k[tbi][nc] = k as u8;
                                all_cs_r[tbi][nc] = r as u8;
                                all_ct_k[tbi][nc] = tk as u8;
                                all_ct_qi[tbi][nc] = gqi[tidx] as u8;
                                nc += 1;
                            }
                        }
                    }
                }
            }
            all_n_cross[tbi] = nc as u32;
        }

        for ti in 0..nt {
            grid[touched[ti] as usize] = -1;
        }
    }

    // ---- VISIBILITY DEDUP: find unique configs ----
    // Same visibility ⟹ same grid state ⟹ same arrows ⟹ same score.
    let mut unique_keys = [0u64; 32];
    let mut unique_tbi = [0u8; 32];
    let mut unique_weight = [0u32; 32];
    let mut n_unique = 0usize;

    for tbi in 0..32 {
        let key = vis_key(&all_vis[tbi]);
        let mut found = false;
        for u in 0..n_unique {
            if unique_keys[u] == key {
                unique_weight[u] += 1;
                found = true;
                break;
            }
        }
        if !found {
            unique_keys[n_unique] = key;
            unique_tbi[n_unique] = tbi as u8;
            unique_weight[n_unique] = 1;
            n_unique += 1;
        }
    }

    // ---- PHASE 2: Evaluate only unique configs ----
    let mut path_sum = 0.0f64;

    for ui in 0..n_unique {
        let tbi = unique_tbi[ui] as usize;
        let w = unique_weight[ui] as f64;
        let vis = all_vis[tbi];

        // Precompute local card counts
        for k in 0..6 {
            let cid = card_ids[k] as usize;
            let v = vis[k] as usize;
            for r in 0..4 {
                for s in 0..N_SYM {
                    lc[k][r][s] = cd.cbm(cid, r, v, s);
                }
            }
        }

        let n_cross = if any_arrows { all_n_cross[tbi] as usize } else { 0 };

        let config_score;
        if n_cross == 0 {
            // ---- FAST PATH: all 6 cards independent ----
            let mut cs = 0.0f64;
            for sym in 0..N_SYM {
                pmf = [0.0; CT1];
                for r in 0..4 {
                    let v = lc[0][r][sym].min(MAX_CT as i8) as usize;
                    pmf[v] += 0.25;
                }
                for k in 1..6 {
                    new_pmf = [0.0; CT1];
                    for r in 0..4 {
                        let c = lc[k][r][sym].min(MAX_CT as i8) as usize;
                        for prev in 0..CT1 {
                            let pv = pmf[prev];
                            if pv > 0.0 {
                                let t = (prev + c).min(MAX_CT);
                                new_pmf[t] += pv * 0.25;
                            }
                        }
                    }
                    pmf = new_pmf;
                }
                for t in 0..CT1 {
                    cs += pmf[t] * SCORING[sym][t];
                }
            }
            config_score = cs;
        } else {
            // ---- SLOW PATH: union-find + grouped convolution ----
            for i in 0..6u8 { parent[i as usize] = i; }
            for ci in 0..n_cross {
                let mut a = all_cs_k[tbi][ci] as usize;
                while parent[a] as usize != a { parent[a] = parent[parent[a] as usize]; a = parent[a] as usize; }
                let mut b = all_ct_k[tbi][ci] as usize;
                while parent[b] as usize != b { parent[b] = parent[parent[b] as usize]; b = parent[b] as usize; }
                if a != b { parent[a] = b as u8; }
            }

            let mut n_groups = 0usize;
            gsz = [0; 6]; gmap = [-1; 6]; gfill = [0; 6];
            for k in 0..6 {
                let mut root = k;
                while parent[root] as usize != root { root = parent[root] as usize; }
                if gmap[root] < 0 { gmap[root] = n_groups as i8; n_groups += 1; }
                let g = gmap[root] as usize;
                gid[k] = g as u8;
                gmem[g][gfill[g] as usize] = k as u8;
                gfill[g] += 1; gsz[g] += 1;
            }

            let mut cs = 0.0f64;
            for sym in 0..N_SYM {
                let mut first = true;
                for g in 0..n_groups {
                    let sz = gsz[g] as usize;

                    if sz == 1 {
                        let k = gmem[g][0] as usize;
                        if first {
                            pmf = [0.0; CT1];
                            for r in 0..4 {
                                let v = lc[k][r][sym].min(MAX_CT as i8) as usize;
                                pmf[v] += 0.25;
                            }
                            first = false;
                        } else {
                            new_pmf = [0.0; CT1];
                            for r in 0..4 {
                                let c = lc[k][r][sym].min(MAX_CT as i8) as usize;
                                for prev in 0..CT1 {
                                    let pv = pmf[prev];
                                    if pv > 0.0 {
                                        let t = (prev + c).min(MAX_CT);
                                        new_pmf[t] += pv * 0.25;
                                    }
                                }
                            }
                            pmf = new_pmf;
                        }
                    } else {
                        grp_pmf = [0.0; CT1];
                        if sz == 2 {
                            let k0 = gmem[g][0] as usize;
                            let k1 = gmem[g][1] as usize;
                            for r0 in 0..4 {
                                for r1 in 0..4 {
                                    let mut tot = lc[k0][r0][sym] as i32 + lc[k1][r1][sym] as i32;
                                    for ci in 0..n_cross {
                                        let csk = all_cs_k[tbi][ci] as usize;
                                        let csr = all_cs_r[tbi][ci] as usize;
                                        let ctk = all_ct_k[tbi][ci] as usize;
                                        let ctqi = all_ct_qi[tbi][ci] as usize;
                                        if csk == k0 && csr == r0 && ctk == k1 {
                                            tot += cd.nac(card_ids[k1] as usize, r1, ctqi, sym) as i32;
                                        } else if csk == k1 && csr == r1 && ctk == k0 {
                                            tot += cd.nac(card_ids[k0] as usize, r0, ctqi, sym) as i32;
                                        }
                                    }
                                    let t = (tot as usize).min(MAX_CT);
                                    grp_pmf[t] += 0.0625;
                                }
                            }
                        } else {
                            let n_combos = 4u32.pow(sz as u32);
                            let inv = 1.0 / n_combos as f64;
                            for combo in 0..n_combos {
                                let mut c = combo;
                                for gi in 0..sz { parent[gmem[g][gi] as usize] = (c & 3) as u8; c >>= 2; }
                                let mut tot = 0i32;
                                for gi in 0..sz { let k = gmem[g][gi] as usize; tot += lc[k][parent[k] as usize][sym] as i32; }
                                for ci in 0..n_cross {
                                    let csk = all_cs_k[tbi][ci] as usize;
                                    let csr = all_cs_r[tbi][ci] as usize;
                                    let ctk = all_ct_k[tbi][ci] as usize;
                                    let ctqi = all_ct_qi[tbi][ci] as usize;
                                    if gid[csk] as usize == g && parent[csk] as usize == csr {
                                        tot += cd.nac(card_ids[ctk] as usize, parent[ctk] as usize, ctqi, sym) as i32;
                                    }
                                }
                                let t = (tot as usize).min(MAX_CT);
                                grp_pmf[t] += inv;
                            }
                        }

                        if first {
                            pmf = grp_pmf;
                            first = false;
                        } else {
                            new_pmf = [0.0; CT1];
                            for prev in 0..CT1 {
                                let pv = pmf[prev];
                                if pv > 0.0 {
                                    for add in 0..CT1 {
                                        let gv = grp_pmf[add];
                                        if gv > 0.0 {
                                            let t = (prev + add).min(MAX_CT);
                                            new_pmf[t] += pv * gv;
                                        }
                                    }
                                }
                            }
                            pmf = new_pmf;
                        }
                    }
                }
                for t in 0..CT1 {
                    cs += pmf[t] * SCORING[sym][t];
                }
            }
            config_score = cs;
        }
        path_sum += config_score * w;
    }

    path_sum / 32.0
}

/// Evaluate a single scenario: mean score across all paths (parallel over paths).
fn evaluate_scenario(card_ids: &[u32; 6], paths: &[i8], n_paths: usize, z_stacks: &[[u8; 6]; 32], cd: &CardData) -> f64 {
    let any_arrows = card_ids.iter().any(|&c| cd.card_has_arrows[c as usize]);

    let sum: f64 = (0..n_paths)
        .into_par_iter()
        .map(|pi| eval_path(pi, card_ids, any_arrows, paths, z_stacks, cd))
        .sum();

    sum / n_paths as f64
}

// ============================================================
// SQLITE PERSISTENCE (same schema as Python)
// ============================================================
struct SimDB {
    conn: Connection,
}

impl SimDB {
    fn open(path: &str) -> Self {
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path).expect("Cannot open DB");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE IF NOT EXISTS scenarios(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 card_ids TEXT UNIQUE, mean_score REAL, elapsed REAL,
                 created_at REAL DEFAULT (strftime('%s','now'))
             );
             CREATE TABLE IF NOT EXISTS card_stats(
                 card_id INTEGER PRIMARY KEY,
                 score_sum REAL DEFAULT 0, score_sq REAL DEFAULT 0, cnt INTEGER DEFAULT 0
             );",
        )
        .unwrap();
        SimDB { conn }
    }

    fn get_config(&self, key: &str) -> Option<String> {
        self.conn
            .query_row("SELECT value FROM config WHERE key=?", params![key], |r| r.get(0))
            .ok()
    }

    fn set_config(&self, key: &str, value: &str) {
        self.conn
            .execute("INSERT OR REPLACE INTO config VALUES(?,?)", params![key, value])
            .unwrap();
    }

    fn completed(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM scenarios", [], |r| r.get::<_, i64>(0))
            .unwrap() as usize
    }

    fn is_done(&self, cards: &[u32; 6]) -> bool {
        let key = cards.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
        self.conn
            .query_row("SELECT 1 FROM scenarios WHERE card_ids=?", params![key], |_| Ok(()))
            .is_ok()
    }

    fn save_batch(&self, results: &[(Vec<u32>, f64, f64)]) {
        let tx = self.conn.unchecked_transaction().unwrap();
        for (cards, score, elapsed) in results {
            let key = cards.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
            if tx.execute(
                "INSERT OR IGNORE INTO scenarios(card_ids,mean_score,elapsed) VALUES(?,?,?)",
                params![key, score, elapsed],
            ).unwrap() > 0 {
                for &cid in cards {
                    tx.execute(
                        "INSERT INTO card_stats(card_id,score_sum,score_sq,cnt)
                         VALUES(?,?,?,1) ON CONFLICT(card_id) DO UPDATE SET
                         score_sum=score_sum+excluded.score_sum,
                         score_sq=score_sq+excluded.score_sq, cnt=cnt+1",
                        params![cid as i64, score, score * score],
                    ).unwrap();
                }
            }
        }
        tx.commit().unwrap();
    }

    fn rankings(&self) -> Vec<(i64, f64, f64, i64)> {
        let mut stmt = self.conn
            .prepare("SELECT card_id,score_sum,score_sq,cnt FROM card_stats ORDER BY score_sum/cnt DESC")
            .unwrap();
        stmt.query_map([], |r| {
            let cid: i64 = r.get(0)?;
            let ss: f64 = r.get(1)?;
            let ssq: f64 = r.get(2)?;
            let cnt: i64 = r.get(3)?;
            let m = ss / cnt as f64;
            let v = if cnt > 1 { (ssq / cnt as f64 - m * m).max(0.0) } else { 0.0 };
            Ok((cid, m, v.sqrt(), cnt))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    fn total_time(&self) -> f64 {
        self.conn
            .query_row("SELECT COALESCE(SUM(elapsed),0) FROM scenarios", [], |r| r.get(0))
            .unwrap()
    }
}

fn show_rankings(db: &SimDB, top: usize) {
    let ranks = db.rankings();
    if ranks.is_empty() {
        eprintln!("  No results yet.");
        return;
    }
    let n = db.completed();
    let tt = db.total_time();
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  Rankings ({n} scenarios, {tt:.0}s compute)");
    eprintln!("{}", "=".repeat(60));
    eprintln!("  {:<4} {:<6} {:>8} {:>8} {:>6}", "#", "Card", "Mean", "Std", "Count");
    eprintln!("  {} {} {} {} {}", "-".repeat(4), "-".repeat(6), "-".repeat(8), "-".repeat(8), "-".repeat(6));
    for (i, &(cid, m, sd, cnt)) in ranks.iter().take(top).enumerate() {
        eprintln!("  {:<4} {:<6} {:>8.2} {:>8.2} {:>6}", i + 1, cid, m, sd, cnt);
    }
    if ranks.len() > top {
        eprintln!("  ... ({} more)", ranks.len() - top);
        let last3: Vec<_> = ranks.iter().rev().take(3).copied().collect();
        for &(cid, m, sd, cnt) in last3.iter().rev() {
            let r = ranks.iter().position(|&(c, _, _, _)| c == cid).unwrap() + 1;
            eprintln!("  {:<4} {:<6} {:>8.2} {:>8.2} {:>6}", r, cid, m, sd, cnt);
        }
    }
}

// ============================================================
// CLI
// ============================================================
#[derive(Parser)]
#[command(name = "pirate_sim", about = "Pirate Card Simulation (Rust)")]
struct Cli {
    #[arg(long, default_value = "pirate_cards/pirate_20_25.json")]
    json: String,
    #[arg(long, default_value = "data/simulation.db")]
    db: String,
    #[arg(long, default_value_t = 100)]
    scenarios: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0, help = "0=all 729K paths")]
    max_paths: usize,
    #[arg(long, default_value_t = 10)]
    batch_size: usize,
    #[arg(long, default_value_t = 0, help = "Rayon threads. 0=all cores")]
    threads: usize,
    #[arg(long, default_value_t = false, help = "Show rankings and exit")]
    rankings: bool,
}

// ============================================================
// MAIN
// ============================================================
fn main() {
    let cli = Cli::parse();

    if cli.rankings {
        let db = SimDB::open(&cli.db);
        show_rankings(&db, 20);
        return;
    }

    eprintln!("Pirate Card Simulation (Rust)");
    eprintln!("{}", "=".repeat(50));

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }
    eprintln!("  Rayon threads: {}", rayon::current_num_threads());

    let cd = load_cards(&cli.json);
    eprintln!("  {} cards loaded", cd.n_cards);
    let arrow_count = cd.card_has_arrows.iter().filter(|&&b| b).count();
    eprintln!("  {}/{} cards have arrows", arrow_count, cd.n_cards);

    let npy_path = "data/paths.npy";
    let (all_paths, total_paths) = get_paths(npy_path);

    let (paths, n_paths) = if cli.max_paths > 0 && cli.max_paths < total_paths {
        let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
        let mut indices: Vec<usize> = (0..total_paths).collect();
        indices.partial_shuffle(&mut rng, cli.max_paths);
        let selected: Vec<i8> = indices[..cli.max_paths]
            .iter()
            .flat_map(|&i| &all_paths[i * 12..(i + 1) * 12])
            .copied()
            .collect();
        eprintln!("  {} position paths (randomly sampled)", cli.max_paths);
        (selected, cli.max_paths)
    } else {
        eprintln!("  {} position paths", total_paths);
        (all_paths, total_paths)
    };

    let z_stacks = make_z_stacks();

    // Warmup
    let warmup_ids: [u32; 6] = [0, 1, 2, 3, 4, 5];
    let _ = evaluate_scenario(&warmup_ids, &paths[..24], 2, &z_stacks, &cd);
    eprintln!("  Warmup done");

    let db = SimDB::open(&cli.db);
    if let Some(stored) = db.get_config("seed") {
        let stored_seed: u64 = stored.parse().unwrap_or(cli.seed);
        if stored_seed != cli.seed {
            eprintln!("  Using stored seed={stored_seed}");
        }
    } else {
        db.set_config("seed", &cli.seed.to_string());
    }

    let done = db.completed();
    let remaining = cli.scenarios.saturating_sub(done);
    eprintln!("  Target: {} | Done: {} | Remaining: {}", cli.scenarios, done, remaining);

    if remaining == 0 {
        show_rankings(&db, 20);
        return;
    }

    // Generate random scenarios
    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);
    let card_range: Vec<u32> = (0..cd.n_cards as u32).collect();
    let mut todo: Vec<[u32; 6]> = Vec::new();
    for _ in 0..cli.scenarios {
        let mut hand = [0u32; 6];
        let sample: Vec<&u32> = card_range.choose_multiple(&mut rng, 6).collect();
        for (i, &&c) in sample.iter().enumerate() {
            hand[i] = c;
        }
        if todo.len() < remaining && !db.is_done(&hand) {
            todo.push(hand);
        }
    }

    eprintln!("  Running {} scenarios...\n", todo.len());
    let t_start = Instant::now();
    let mut batch: Vec<(Vec<u32>, f64, f64)> = Vec::new();

    for (i, &cards) in todo.iter().enumerate() {
        let t0 = Instant::now();
        let score = evaluate_scenario(&cards, &paths, n_paths, &z_stacks, &cd);
        let elapsed = t0.elapsed().as_secs_f64();
        batch.push((cards.to_vec(), score, elapsed));

        if batch.len() >= cli.batch_size {
            db.save_batch(&batch);
            batch.clear();
        }

        let total_done = i + 1;
        if total_done % (todo.len() / 50).max(1) == 0 || total_done == todo.len() {
            let total_el = t_start.elapsed().as_secs_f64();
            let rate = total_done as f64 / total_el;
            let eta = (todo.len() - total_done) as f64 / rate;
            let pct = total_done as f64 / todo.len() as f64 * 100.0;
            eprint!(
                "\r  [{}/{}] {:.0}% | {:.2}/s | {:.1}s last | ETA {:.0}s   ",
                total_done, todo.len(), pct, rate, elapsed, eta
            );
        }
    }

    if !batch.is_empty() {
        db.save_batch(&batch);
    }

    let total = t_start.elapsed().as_secs_f64();
    eprintln!(
        "\n\n  Done: {} scenarios in {:.1}s ({:.2}s avg)",
        todo.len(), total, total / todo.len() as f64
    );
    show_rankings(&db, 20);
}
