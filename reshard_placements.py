#!/usr/bin/env python3
"""Re-shard existing flat placement file into hash-partitioned shards.

Reads the monolithic placements.bin in chunks, computes combo_key for each
record, and distributes to shard files based on combo_key % n_shards.

Usage:
    python reshard_placements.py [--src data/placements] [--dst data/placements_sharded]
                                 [--shards 4096] [--chunk 2000000]

Typical performance: ~15 min for 80 GB on SSD (sequential read, scatter write).
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.sim import PLACEMENT_DTYPE, PlacementStore, ShardedPlacementStore


def reshard(src_dir, dst_dir, n_shards=4096, chunk_size=2_000_000):
    src = PlacementStore(src_dir)
    n_total = src.count()

    if n_total == 0:
        print("  No records to reshard.")
        return

    dst = ShardedPlacementStore(dst_dir, n_shards=n_shards)
    existing = dst.count()

    print(f"  Source: {src.bin_path} ({n_total:,} records, "
          f"{n_total * PlacementStore.RECORD_SIZE / 1e9:.1f} GB)")
    print(f"  Destination: {dst_dir}/ ({n_shards} shards)")
    if existing > 0:
        print(f"  Warning: destination already has {existing:,} records")

    data = src.load_mmap()
    t_start = time.time()
    processed = 0

    while processed < n_total:
        chunk_end = min(processed + chunk_size, n_total)
        chunk = np.array(data[processed:chunk_end])  # copy from mmap

        # Compute combo keys and distribute to shards
        dst.append_raw(chunk)

        processed = chunk_end
        elapsed = time.time() - t_start
        rate = processed / elapsed
        eta = (n_total - processed) / rate if rate > 0 else 0
        pct = processed / n_total * 100

        sys.stdout.write(
            f"\r  [{processed:,}/{n_total:,}] {pct:.1f}% | "
            f"{rate:,.0f} rec/s | ETA {eta:.0f}s   ")
        sys.stdout.flush()

    total = time.time() - t_start
    print(f"\n\n  Done: {n_total:,} records resharded in {total:.1f}s "
          f"({n_total / total:,.0f} rec/s)")

    # Flush remaining buffered records
    result = dst.flush()
    if result:
        n_flushed, n_shards_written = result
        print(f"  Final flush: {n_flushed:,} records to {n_shards_written} shards")

    # Verify
    dst_count = dst.count()
    print(f"  Verification: {dst_count:,} records in shards "
          f"({'OK' if dst_count == n_total + existing else 'MISMATCH!'})")

    # Shard size distribution
    counts = dst.shard_counts()
    non_empty = counts[counts > 0]
    print(f"  Shards: {len(non_empty)} non-empty / {n_shards} total")
    if len(non_empty) > 0:
        sizes_mb = counts * PlacementStore.RECORD_SIZE / 1e6
        print(f"  Shard sizes: mean={sizes_mb[sizes_mb>0].mean():.1f} MB, "
              f"max={sizes_mb.max():.1f} MB, min={sizes_mb[sizes_mb>0].min():.1f} MB")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Reshard flat placement file into hash-partitioned shards')
    p.add_argument('--src', default='data/placements',
                   help='Source directory with placements.bin')
    p.add_argument('--dst', default='data/placements_sharded',
                   help='Destination directory for shard files')
    p.add_argument('--shards', type=int, default=4096,
                   help='Number of shards (default: 4096)')
    p.add_argument('--chunk', type=int, default=2_000_000,
                   help='Records per chunk (default: 2M = 32 MB)')
    a = p.parse_args()

    print("=" * 60)
    print("  RESHARD: flat placements.bin → hash-partitioned shards")
    print("=" * 60)
    reshard(a.src, a.dst, a.shards, a.chunk)
