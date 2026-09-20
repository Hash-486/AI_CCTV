# ============================================================
# dataset_groups.py -- Near-duplicate grouping for dataset splits
# ============================================================
"""
Group images that must never be separated across a train/val boundary.

The problem this solves: the original Weapon-2 split put frames of the same
video on both sides of the split, so 37% of validation images had a pixel-level
near-duplicate in training. Any metric measured that way reports memorisation.

Grouping is done on *pixels* (perceptual hash), not filenames. Filenames worked
for this dataset by luck; incoming sources in later phases carry no useful
naming convention, and the hash keeps working regardless.

Usage:
    python tools/dataset_groups.py --check datasets/weapon_clipsplit
    python tools/dataset_groups.py --check-dirs A/images B/images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Two images within this Hamming distance are treated as the same image.
# Calibrated on the Weapon-2 audit: phone-video frames of one clip sit at
# 2-21% pairwise match under this threshold, while independent scraped images
# sit at 0.02-0.18% -- a ~100x separation, so the threshold is not delicate.
NEAR_DUP_HAMMING = 6

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# popcount lookup for a single byte -- cheaper than unpacking bits
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash: compare each pixel to its right-hand neighbour."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    bits = px[:, 1:] > px[:, :-1]
    out = 0
    for b in bits.reshape(-1):
        out = (out << 1) | int(b)
    return out


def hashes_for(paths: list[Path]) -> np.ndarray:
    """dhash every path, returned as a (n, 8) uint8 array for fast comparison."""
    vals = np.array([dhash(p) for p in paths], dtype=np.uint64)
    return vals.view(np.uint8).reshape(-1, 8)


def _pair_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance between every row of a and every row of b."""
    return _POPCOUNT8[np.bitwise_xor(a[:, None, :], b[None, :, :])].sum(axis=2)


def near_duplicate_pairs(packed: np.ndarray, threshold: int = NEAR_DUP_HAMMING,
                         chunk: int = 512):
    """Yield (i, j) index pairs closer than `threshold`. Chunked to bound memory."""
    n = len(packed)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = _pair_distances(packed[start:stop], packed)
        for local_i, row in enumerate(d):
            i = start + local_i
            for j in np.nonzero(row[i + 1:] <= threshold)[0]:
                yield i, i + 1 + int(j)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(paths: list[Path], threshold: int = NEAR_DUP_HAMMING,
                 filename_key=None) -> dict[Path, int]:
    """
    Return {path: group_id}. Two images share a group if they are near-duplicates,
    transitively, or if `filename_key` maps them to the same non-None key.

    The filename channel exists because video frames sampled far apart in one clip
    are genuinely different images -- the hash will not link them -- yet they still
    share a scene, lighting and subject, so splitting them is still leakage.
    """
    if not paths:
        return {}
    uf = _UnionFind(len(paths))

    packed = hashes_for(paths)
    for i, j in near_duplicate_pairs(packed, threshold):
        uf.union(i, j)

    if filename_key is not None:
        by_key: dict[str, int] = {}
        for idx, p in enumerate(paths):
            key = filename_key(p)
            if key is None:
                continue
            if key in by_key:
                uf.union(by_key[key], idx)
            else:
                by_key[key] = idx

    roots = {}
    out = {}
    for idx, p in enumerate(paths):
        r = uf.find(idx)
        if r not in roots:
            roots[r] = len(roots)
        out[p] = roots[r]
    return out


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def cross_split_report(train_dir: Path, val_dir: Path,
                       thresholds=(4, 6, 10)) -> dict[int, int]:
    """
    Count validation images that have a near-duplicate anywhere in training.
    This is the gate: a clean split scores 0 at every threshold.
    """
    tr, va = list_images(train_dir), list_images(val_dir)
    if not tr or not va:
        raise SystemExit(f"No images found (train={len(tr)}, val={len(va)})")
    print(f"train={len(tr)} images, val={len(va)} images -- hashing...")
    tr_h, va_h = hashes_for(tr), hashes_for(va)

    best = np.full(len(va), 64, dtype=np.int16)
    for start in range(0, len(va), 512):
        stop = min(start + 512, len(va))
        d = _pair_distances(va_h[start:stop], tr_h)
        best[start:stop] = d.min(axis=1)

    results = {}
    for t in thresholds:
        hit = int((best <= t).sum())
        results[t] = hit
        flag = "OK" if hit == 0 else "LEAK"
        print(f"  [{flag:4s}] hamming<={t:2d}: {hit:4d}/{len(va)} val images "
              f"have a near-duplicate in train ({100 * hit / len(va):.1f}%)")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", type=Path,
                    help="dataset root containing train/images and val/images")
    ap.add_argument("--check-dirs", nargs=2, type=Path, metavar=("TRAIN", "VAL"),
                    help="two image directories to compare directly")
    args = ap.parse_args()

    if args.check:
        train_dir = args.check / "train" / "images"
        val_dir = args.check / "val" / "images"
        if not val_dir.exists():                      # tolerate Roboflow's "valid"
            val_dir = args.check / "valid" / "images"
    elif args.check_dirs:
        train_dir, val_dir = args.check_dirs
    else:
        ap.print_help()
        return 2

    results = cross_split_report(train_dir, val_dir)
    leaked = results.get(10, 0)
    if leaked:
        print(f"\nFAIL: {leaked} val images leak from train. The split is not usable.")
        return 1
    print("\nPASS: no cross-split near-duplicates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
