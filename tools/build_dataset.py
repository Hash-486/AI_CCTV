# ============================================================
# build_dataset.py -- Merge, harmonise, dedup and split the stage-1 dataset
# ============================================================
"""
Build datasets/weapon_stage1 from the raw sources in datasets/_raw.

Four things happen here, in order, and the order matters:

  1. HARMONISE  every source class is mapped through tools/class_map.yaml to a
                canonical class, to __background__, or to __drop__. Mapping is
                keyed by (source, source-class-NAME) -- never by class index,
                because indices differ across sources and several sources carry
                both `Knife` and `knife` as distinct classes.
  2. GROUP      all images from all sources are hashed together and linked into
                near-duplicate groups. Doing this globally (not per source) is
                what catches cross-source duplicates from forked datasets.
  3. SPLIT      whole groups are assigned to train or val, stratified toward a
                per-class instance target. A group is never split.
  4. EMIT       images + labels copied out, plus data.yaml and a manifest.

Unmapped source classes are a hard error. That is the point of the map file.

Usage:
    python tools/build_dataset.py --dry-run     # report the merge, write nothing
    python tools/build_dataset.py               # build it
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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_groups import build_groups  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "_raw"
DEST = ROOT / "datasets" / "weapon_stage1"
MAP_FILE = Path(__file__).resolve().parent / "class_map.yaml"

BACKGROUND = "__background__"
DROP = "__drop__"

# Person-level boxes and non-object concepts. If any of these ever reaches the
# mapper it means a source was added without review -- fail rather than guess.
# A name match on "knife" in `knife_attacker` would teach the model that a whole
# human body is a knife.
FORBIDDEN = {
    "gunmen", "knife_attacker", "aggressor", "victim", "stabbing",
    "blood", "violence", "other-pose",
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_map() -> tuple[list[str], dict]:
    cfg = yaml.safe_load(MAP_FILE.read_text())
    classes = list(cfg["classes"])
    return classes, cfg["sources"]


def source_path(slug: str, spec: dict) -> Path:
    """Sources live in datasets/_raw/<slug> unless the map gives an explicit path."""
    p = spec.get("path")
    return (ROOT / p) if p else (RAW / slug)


def source_dirs() -> list[Path]:
    return sorted(d for d in RAW.iterdir() if d.is_dir()) if RAW.exists() else []


def read_names(src: Path) -> list[str] | None:
    """
    Class names in index order, from the source's own data.yaml.

    Returns None when the directory is not (yet) a usable YOLO dataset -- a
    partially-downloaded source, or one in a different annotation format that
    still needs converting. The caller warns and skips rather than aborting, so
    one incomplete source cannot block a build of everything else.
    """
    yamls = sorted(src.rglob("data.yaml"))
    if not yamls:
        return None
    cfg = yaml.safe_load(yamls[0].read_text())
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names) if names else None


_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")


def roboflow_stem_key(img: Path, src_of: dict) -> str:
    """
    Group key for Roboflow's augmented exports.

    Roboflow version exports emit N augmented copies of each source image, named
    `<original>_jpg.rf.<hash>.jpg`. `guns_mms73` has 23,130 files from only 2,488
    unique sources -- up to 18 copies each. Perceptual hashing CANNOT link those:
    a horizontally flipped copy hashes completely differently, so without this
    key the augmented siblings of one photo would land on both sides of the
    split -- a worse leak than the one this whole pipeline exists to prevent.

    Scoped by source slug on purpose: augmentation siblings only ever live inside
    one source, and generic stems like `1290_jpg` could collide by chance across
    unrelated datasets. Genuine cross-source duplicates are caught by the hash.
    """
    return f"{src_of[img]}:{_RF_SUFFIX.sub('', img.stem)}"


def label_path_for(img: Path) -> Path | None:
    """YOLO layout puts labels in a sibling 'labels' dir mirroring 'images'."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            cand = Path(*parts).with_suffix(".txt")
            return cand if cand.exists() else None
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def collect(classes: list[str], smap: dict, strict: bool):
    """Return records: {'img', 'src', 'labels': [(cls_idx, cx, cy, w, h)], 'bg': bool}."""
    idx_of = {c: i for i, c in enumerate(classes)}
    records, stats = [], defaultdict(lambda: defaultdict(int))
    errors = []

    on_disk = {d.name for d in source_dirs()}
    unlisted = on_disk - set(smap)
    for slug in sorted(unlisted):
        errors.append(f"source '{slug}' present in datasets/_raw but absent from class_map.yaml")

    for slug, spec in sorted(smap.items()):
        spec = spec or {}
        if spec.get("exclude"):
            print(f"   [excluded] {slug:<18} {spec.get('reason', '')}")
            continue
        src = source_path(slug, spec)
        if not src.exists():
            print(f"   [missing ] {slug:<18} {src} not downloaded -- skipping")
            continue
        rules = spec.get("map") or {}
        names = read_names(src)
        if names is None:
            print(f"   [skip    ] {slug:<18} no usable data.yaml "
                  f"(incomplete download, or needs format conversion)")
            continue

        unmapped = [n for n in names if n not in rules]
        if unmapped:
            errors.append(f"{slug}: unmapped classes {unmapped} (add them to class_map.yaml)")
            continue
        bad = [n for n in names if n.strip().lower() in FORBIDDEN
               and rules.get(n) not in (DROP, BACKGROUND)]
        if bad:
            errors.append(f"{slug}: person-level/non-object classes mapped to a weapon class: {bad}")
            continue

        imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
        for img in imgs:
            lp = label_path_for(img)
            kept, saw_bg, saw_drop, n_lines = [], False, False, 0
            if lp:
                for line in lp.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        ci = int(float(parts[0]))
                    except ValueError:
                        continue
                    n_lines += 1
                    if not (0 <= ci < len(names)):
                        stats[slug]["label_index_out_of_range"] += 1
                        continue
                    target = rules[names[ci]]
                    stats[slug][f"src:{names[ci]}->{target}"] += 1
                    if target == DROP:
                        saw_drop = True
                        continue
                    if target == BACKGROUND:
                        saw_bg = True
                        continue
                    kept.append((idx_of[target], *(float(x) for x in parts[1:5])))

            if kept:
                records.append({"img": img, "src": slug, "labels": kept, "bg": False})
            elif saw_drop:
                # Something was removed that we cannot vouch for -- e.g. the CCTV
                # set's ambiguous `weapon` class. The image may still contain a
                # real weapon that is now unlabelled, and using it as a background
                # would teach the model to ignore weapons. Discard it.
                stats[slug]["dropped_unsafe_after_drop"] += 1
            elif saw_bg:
                records.append({"img": img, "src": slug, "labels": [], "bg": True})
            elif lp is not None and n_lines == 0:
                # An EMPTY label file is an explicit negative -- Roboflow writes
                # these for deliberate "null examples". A MISSING label file is
                # not the same thing: it means unknown, so it is skipped below.
                stats[slug]["explicit_negative"] += 1
                records.append({"img": img, "src": slug, "labels": [], "bg": True})
            else:
                stats[slug]["dropped_no_label_file"] += 1

    if errors and strict:
        print("\nCLASS MAP ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    return records, stats


def choose_val(groups_by_id, targets, seed):
    """Admit whole groups to val until each class hits its instance target."""
    rng = random.Random(seed)
    order = sorted(groups_by_id)
    rng.shuffle(order)
    chosen, running = set(), defaultdict(int)
    bg_target = targets.get(BACKGROUND, 0)
    for gid in order:
        if all(running[c] >= t for c, t in targets.items()):
            break
        g = groups_by_id[gid]
        if any(running[c] >= targets.get(c, 0) and g["counts"].get(c, 0) > 0
               for c in targets):
            continue
        chosen.add(gid)
        for c, n in g["counts"].items():
            running[c] += n
    return chosen, running


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hamming", type=int, default=10,
                    help="near-duplicate grouping distance. Must be >= the distance "
                         "the verification gate checks, or the build will pass its own "
                         "grouping and still fail the gate (default 10)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-strict", action="store_true",
                    help="warn instead of failing on unmapped classes")
    args = ap.parse_args()

    if not MAP_FILE.exists():
        sys.exit(f"missing {MAP_FILE} -- run tools/fetch_datasets.py --report first")
    classes, smap = load_map()
    print(f"canonical classes: {classes}")

    records, stats = collect(classes, smap, strict=not args.no_strict)
    print(f"\nusable images after mapping: {len(records)}")
    per_src = defaultdict(int)
    for r in records:
        per_src[r["src"]] += 1
    for s, n in sorted(per_src.items()):
        bg = sum(1 for r in records if r["src"] == s and r["bg"])
        print(f"   {s:<20} {n:>6} images ({bg} background)")

    print("\nhashing and grouping all sources together...")
    src_of = {r["img"]: r["src"] for r in records}
    gmap = build_groups([r["img"] for r in records],
                        threshold=args.hamming,
                        filename_key=lambda p: roboflow_stem_key(p, src_of))
    groups = defaultdict(lambda: {"recs": [], "counts": defaultdict(int), "srcs": set()})
    for r in records:
        g = groups[gmap[r["img"]]]
        g["recs"].append(r)
        g["srcs"].add(r["src"])
        if r["bg"]:
            g["counts"][BACKGROUND] += 1
        for ci, *_ in r["labels"]:
            g["counts"][classes[ci]] += 1
    print(f"   {len(records)} images -> {len(groups)} groups")

    dupes = len(records) - len(groups)
    cross = [g for g in groups.values() if len(g["srcs"]) > 1]
    print(f"   near-duplicate images absorbed: {dupes} ({100*dupes/max(len(records),1):.1f}%)")
    print(f"   groups spanning >1 source (cross-source duplicates): {len(cross)}")

    totals = defaultdict(int)
    for g in groups.values():
        for c, n in g["counts"].items():
            totals[c] += n
    print("\ninstances after dedup grouping:")
    for c in classes + [BACKGROUND]:
        print(f"   {c:<16} {totals.get(c, 0):>7}")

    targets = {c: round(totals.get(c, 0) * args.val_frac) for c in classes}
    targets[BACKGROUND] = round(totals.get(BACKGROUND, 0) * args.val_frac)
    val_ids, got = choose_val(groups, targets, args.seed)
    print(f"\nval targets {dict(targets)}\n      actual {dict(got)}")

    split_of = {gid: ("val" if gid in val_ids else "train") for gid in groups}
    counts = {"train": defaultdict(int), "val": defaultdict(int)}
    imgs = {"train": 0, "val": 0}
    for gid, g in groups.items():
        s = split_of[gid]
        imgs[s] += len(g["recs"])
        for c, n in g["counts"].items():
            counts[s][c] += n
    print("\nfinal split:")
    for s in ("train", "val"):
        print(f"   {s:<6} images={imgs[s]:>6}  " +
              "  ".join(f"{c}={counts[s].get(c,0)}" for c in classes + [BACKGROUND]))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if DEST.exists():
        print(f"\nremoving previous build at {DEST}")
        shutil.rmtree(DEST)
    for s in ("train", "val"):
        (DEST / s / "images").mkdir(parents=True, exist_ok=True)
        (DEST / s / "labels").mkdir(parents=True, exist_ok=True)

    print("copying...")
    n = 0
    for gid, g in groups.items():
        s = split_of[gid]
        for r in g["recs"]:
            # prefix with source slug: filenames collide across datasets
            stem = f"{r['src']}__{r['img'].stem}"
            shutil.copy2(r["img"], DEST / s / "images" / (stem + r["img"].suffix))
            lines = [f"{ci} {a:.6f} {b:.6f} {c:.6f} {d:.6f}"
                     for ci, a, b, c, d in r["labels"]]
            (DEST / s / "labels" / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            n += 1
    print(f"   wrote {n} image/label pairs")

    (DEST / "data.yaml").write_text(
        "# Stage-1 merged weapon dataset. Built by tools/build_dataset.py.\n"
        "# Group-aware split: no image has a near-duplicate across train/val.\n"
        "# Verify: python tools/dataset_groups.py --check datasets/weapon_stage1\n"
        f"path: {DEST.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    (DEST / "build_manifest.json").write_text(json.dumps({
        "classes": classes,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "images": imgs,
        "instances": {s: dict(counts[s]) for s in counts},
        "images_per_source": dict(per_src),
        "near_duplicates_absorbed": dupes,
        "cross_source_duplicate_groups": len(cross),
        "mapping_stats": {k: dict(v) for k, v in stats.items()},
    }, indent=2))
    print(f"\nwrote {DEST}")
    print("Next: python tools/dataset_groups.py --check datasets/weapon_stage1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
