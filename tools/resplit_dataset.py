# ============================================================
# resplit_dataset.py -- Group-aware re-split of the Weapon-2 dataset
# ============================================================
"""
Rebuild the weapon dataset so no image ever sits on both sides of the split.

The Roboflow release split frame-wise, putting near-identical frames of the same
phone video in train and val. This rebuilds it group-wise: whole clips move as
units, and only genuinely independent images are split randomly.

Reads  : Weapon 2.v2i.yolov8/{train,valid}
Writes : datasets/weapon_clipsplit/{train,val}/{images,labels} + data.yaml
         datasets/weapon_clipsplit/split_manifest.json

The source directory is never modified.

Usage:
    python tools/resplit_dataset.py                 # build with defaults
    python tools/resplit_dataset.py --dry-run       # report the split, copy nothing
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_groups import build_groups, list_images  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Weapon 2.v2i.yolov8"
DEST = ROOT / "datasets" / "weapon_clipsplit"
CLASS_NAMES = ["guns", "knife"]

# Clips held out for validation.
#
# Only the gun clip is held out. The knife footage turned out to be just TWO
# distinct scenes, not three -- hashing merged `19_46_37` and `18_04_35` into one
# 558-image group, so they are the same scene shot twice. Holding either knife
# scene out would put ~1,000 highly-correlated instances from a single continuous
# shot into validation, which measures one lighting/background condition rather
# than knife detection. The independent scraped images are far more diverse per
# instance, so knife validation is drawn from those instead (see VAL_FRACTION).
VAL_CLIPS = {
    "whatsapp-video-2023-12-04-at-21_04_14_mp4",   # guns, 144 frames, 275 instances
}

# Fraction of *independent* (non-clip) instances sent to validation, per class.
# knife is higher because all knife clips stay in train, so the independent pool
# is the only source of knife validation data. guns is lower because the held-out
# gun clip already supplies 275 instances.
VAL_FRACTION = {"guns": 0.15, "knife": 0.40}

_ROBOFLOW_SUFFIX = re.compile(r"_jpg\.rf\.[0-9a-f]+$")
_CLIP = re.compile(r"(whatsapp-video-[\d-]+-at-[\d_]+?mp4)")


def clip_key(path: Path) -> str | None:
    """Source-clip identifier for video frames; None for standalone images."""
    stem = _ROBOFLOW_SUFFIX.sub("", path.stem).lower()
    m = _CLIP.match(stem)
    return m.group(1) if m else None


def label_for(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def count_instances(image_path: Path) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    lp = label_for(image_path)
    if lp.exists():
        for line in lp.read_text().splitlines():
            if line.strip():
                idx = int(line.split()[0])
                if 0 <= idx < len(CLASS_NAMES):
                    counts[CLASS_NAMES[idx]] += 1
    return counts


def choose_val_groups(independent, per_class_target, seed):
    """
    Pick which independent-image groups go to validation.

    Policy: shuffle, then admit groups until every class has met its instance
    target. Groups are admitted whole, so the final counts land near the target
    rather than exactly on it.

    Targets come from VAL_FRACTION and are deliberately NOT proportional -- see
    the note there on why knife draws more heavily from this pool. Raising a
    class's fraction tightens the error bars on that class at the cost of a
    smaller training set for it.
    """
    rng = random.Random(seed)
    order = sorted(independent.keys())
    rng.shuffle(order)

    chosen, running = set(), {name: 0 for name in CLASS_NAMES}
    for gid in order:
        if all(running[c] >= per_class_target[c] for c in CLASS_NAMES):
            break
        group = independent[gid]
        # Skip a group that would overshoot a target it has already met, so one
        # large group cannot swamp the split.
        if any(running[c] >= per_class_target[c] and group["counts"][c] > 0
               for c in CLASS_NAMES):
            continue
        chosen.add(gid)
        for c in CLASS_NAMES:
            running[c] += group["counts"][c]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=None,
                    help="override VAL_FRACTION with one value for every class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split without copying files")
    args = ap.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"Source dataset not found: {SOURCE}")

    images: list[Path] = []
    for split in ("train", "valid"):
        images.extend(list_images(SOURCE / split / "images"))
    print(f"Found {len(images)} images in {SOURCE.name}")

    print("Grouping by perceptual hash + source clip...")
    groups = build_groups(images, filename_key=clip_key)

    # Aggregate each group's contents
    agg = defaultdict(lambda: {"paths": [], "counts": {c: 0 for c in CLASS_NAMES},
                               "clips": set()})
    for path, gid in groups.items():
        g = agg[gid]
        g["paths"].append(path)
        for c, n in count_instances(path).items():
            g["counts"][c] += n
        key = clip_key(path)
        if key:
            g["clips"].add(key)
    print(f"  -> {len(agg)} groups")

    merged = [gid for gid, g in agg.items() if len(g["clips"]) > 1]
    if merged:
        print(f"  note: {len(merged)} group(s) span multiple clips (visually similar "
              f"footage merged by hash) -- they move together, which is correct")

    video_groups = {gid: g for gid, g in agg.items() if g["clips"]}
    independent = {gid: g for gid, g in agg.items() if not g["clips"]}

    fractions = ({c: args.val_frac for c in CLASS_NAMES}
                 if args.val_frac is not None else dict(VAL_FRACTION))
    ind_totals = {c: sum(g["counts"][c] for g in independent.values()) for c in CLASS_NAMES}
    target = {c: round(ind_totals[c] * fractions[c]) for c in CLASS_NAMES}
    print(f"Independent images: {sum(len(g['paths']) for g in independent.values())} "
          f"in {len(independent)} groups, instances {ind_totals}")
    print(f"Val fractions {fractions} -> target from independent images: {target}")

    val_groups = choose_val_groups(independent, target, args.seed)
    for gid, g in video_groups.items():
        if g["clips"] & VAL_CLIPS:
            val_groups.add(gid)

    assignment = {gid: ("val" if gid in val_groups else "train") for gid in agg}

    stats = {s: {"images": 0, **{c: 0 for c in CLASS_NAMES}} for s in ("train", "val")}
    for gid, g in agg.items():
        s = assignment[gid]
        stats[s]["images"] += len(g["paths"])
        for c in CLASS_NAMES:
            stats[s][c] += g["counts"][c]

    print("\nResulting split:")
    for s in ("train", "val"):
        d = stats[s]
        total = d["guns"] + d["knife"]
        print(f"  {s:5s}  images={d['images']:>5}  guns={d['guns']:>5}  "
              f"knife={d['knife']:>5}  instances={total:>5}")
    frac = stats["val"]["images"] / max(1, stats["train"]["images"] + stats["val"]["images"])
    print(f"  val is {100 * frac:.1f}% of images")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if DEST.exists():
        print(f"\nRemoving previous build at {DEST}")
        shutil.rmtree(DEST)
    for s in ("train", "val"):
        (DEST / s / "images").mkdir(parents=True, exist_ok=True)
        (DEST / s / "labels").mkdir(parents=True, exist_ok=True)

    print("Copying files...")
    copied = 0
    for gid, g in agg.items():
        s = assignment[gid]
        for path in g["paths"]:
            shutil.copy2(path, DEST / s / "images" / path.name)
            lp = label_for(path)
            dest_label = DEST / s / "labels" / (path.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, dest_label)
            else:
                dest_label.write_text("")     # background image: empty label file
            copied += 1
    print(f"  copied {copied} image/label pairs")

    (DEST / "data.yaml").write_text(
        "# Group-aware re-split of Weapon-2 v2. Built by tools/resplit_dataset.py.\n"
        "# Whole source clips are assigned to one side; no image has a near-duplicate\n"
        "# across the split. Verify with: python tools/dataset_groups.py --check <this dir>\n"
        f"path: {DEST.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    manifest = {
        "source": str(SOURCE),
        "seed": args.seed,
        "val_fractions_independent": fractions,
        "val_clips": sorted(VAL_CLIPS),
        "stats": stats,
        "groups": [
            {
                "group_id": gid,
                "split": assignment[gid],
                "images": len(g["paths"]),
                "counts": g["counts"],
                "clips": sorted(g["clips"]),
            }
            for gid, g in sorted(agg.items())
        ],
    }
    (DEST / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {DEST}")
    print("Next: python tools/dataset_groups.py --check datasets/weapon_clipsplit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
