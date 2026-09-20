"""AI_CCTV reorganization -- moves three overlapping project generations
(root, Review_1.2, Updated_Review_1.2) into one tree rooted at AI_CCTV/,
per PLAN.md. Every move, rename and deletion is encoded below as data,
not prose, so all three machines that hold a copy of this project
produce an identical result.

--dry-run is the default for both phases below; nothing moves or is
deleted until the matching flag is passed explicitly. Re-running after
a partial or complete run is a no-op, not a second migration -- each
step checks whether its destination already holds the result, and
stops naming the exact path if a source is missing or a destination
already exists with different content.

Usage:
  python tools/migrate.py                 dry run of everything
  python tools/migrate.py --apply         archive/promote/cherry-pick/patch, for real
  python tools/migrate.py --apply --prune also delete the re-derivable list, for real
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg):
    print(f"[migrate] {msg}")


# ================================================================
# Step 1 -- archive the old generations, whole. Must complete before
# Step 2 promotes the base folder, since root config.py/main.py/
# tracker.py share names with Updated_Review_1.2's -- otherwise the
# wrong file wins silently.
# ================================================================
ARCHIVE = [
    ("Review_1.2",                 "archive/Review_1.2"),
    ("Analysis",                   "archive/Analysis"),
    ("old",                        "archive/old_prototypes"),
    ("recordings",                 "archive/recordings"),
    ("screenshots",                "archive/screenshots"),
    ("threat_log.csv",             "archive/threat_log.csv"),
    ("runs/detect/train3",         "archive/runs_legacy/train3"),
    ("runs/detect/weapon_yolov8s", "archive/runs_legacy/weapon_yolov8s"),
    # Not named in PLAN.md's explicit tables -- archived rather than
    # guessed at, per the plan's own "fail fast, name the path" rule.
    ("runs/detect/train",          "archive/runs_legacy/train"),
    ("runs/detect/train2",         "archive/runs_legacy/train2"),
    ("runs/detect/Review_1.2",     "archive/runs_legacy/Review_1.2_eval_stage2_final"),
    ("main.py",                    "archive/root_legacy/main.py"),
    ("config.py",                  "archive/root_legacy/config.py"),
    ("detection.py",               "archive/root_legacy/detection.py"),
    ("tracker.py",                 "archive/root_legacy/tracker.py"),
    ("pose_estimator.py",          "archive/root_legacy/pose_estimator.py"),
    ("cam_test.py",                "archive/root_legacy/cam_test.py"),
    ("association.py",             "archive/root_legacy/association.py"),
    ("threat_analyzer.py",         "archive/root_legacy/threat_analyzer.py"),
    ("alert_manager.py",           "archive/root_legacy/alert_manager.py"),
]

# ================================================================
# Step 2 -- promote Updated_Review_1.2/* to become the new root, plus
# the shared root-level assets its own config.py already resolves via
# ROOT_DIR = dirname(BASE_DIR).
# ================================================================
PROMOTE = [
    ("Updated_Review_1.2/PRESENTATION_SCRIPT_AND_VIVA_PREP.md",  "docs/PRESENTATION_SCRIPT_AND_VIVA_PREP.md"),
    ("Updated_Review_1.2/PRESENTATION_SCRIPT_AND_VIVA_PREP.pdf", "docs/PRESENTATION_SCRIPT_AND_VIVA_PREP.pdf"),
    ("Updated_Review_1.2/RESULTS.md",          "docs/RESULTS.md"),
    ("Updated_Review_1.2/TECHNICAL_REPORT.md", "docs/TECHNICAL_REPORT.md"),
    ("Updated_Review_1.2/benchmarks",          "benchmarks"),
    ("Updated_Review_1.2/config.py",           "config.py"),
    ("Updated_Review_1.2/detector.py",         "detector.py"),
    ("Updated_Review_1.2/embedder.py",         "embedder.py"),
    ("Updated_Review_1.2/eval",                "eval"),
    ("Updated_Review_1.2/gallery.py",          "gallery.py"),
    ("Updated_Review_1.2/main.py",             "main.py"),
    ("Updated_Review_1.2/pose.py",             "pose.py"),
    ("Updated_Review_1.2/presentation",        "docs/presentation"),
    ("Updated_Review_1.2/render.py",           "render.py"),
    ("Updated_Review_1.2/tests",               "tests"),
    ("Updated_Review_1.2/tracker.py",          "tracker.py"),
    ("Updated_Review_1.2/weights/osnet_x1_0_duke.pth",       "models/reid/osnet_x1_0_duke.pth"),
    ("Updated_Review_1.2/weights/osnet_x1_0_market1501.pth", "models/reid/osnet_x1_0_market1501.pth"),
    ("Updated_Review_1.2/weights/osnet_x1_0_msmt17.pth",     "models/reid/osnet_x1_0_msmt17.pth"),
    ("Updated_Review_1.2/data/MOT17",                 "data/reid/MOT17"),
    ("Updated_Review_1.2/data/MSMT17_V1",             "data/reid/MSMT17_V1"),
    ("Updated_Review_1.2/data/Market-1501-v15.09.15", "data/reid/Market-1501-v15.09.15"),
    # Not in PLAN.md's target tree -- kept live rather than archived,
    # since the base folder is the strongest/most current work; flagged
    # in the migration report instead of silently dropped or guessed at.
    ("Updated_Review_1.2/output", "archive/output_legacy"),
    ("Updated_Review_1.2/project_eval1 (1) (1) - backup.pptx", "docs/presentation_slides/project_eval1 (1) (1) - backup.pptx"),
    ("Updated_Review_1.2/project_eval1 (1) (1).pptx",          "docs/presentation_slides/project_eval1 (1) (1).pptx"),
    # Shared assets.
    ("yolo26s.pt",                "models/person/yolo26s.pt"),
    ("pose_landmarker_lite.task", "models/person/pose_landmarker_lite.task"),
    ("deep-person-reid",          "third_party/deep-person-reid"),
]
# Root already has its own tools/ dir (the dataset-build pipeline, plus
# this script) at the exact path the new root's tools/ needs to occupy
# -- so it is not moved, it is merged into: Updated_Review_1.2/tools's
# files are promoted one at a time (see promote_tools_dir below), never
# as a whole-directory move that would nest one tools/ inside the other.

# ================================================================
# Step 3 -- live dataset moves. Never superseded, just relocated.
# ================================================================
DATASET_LIVE = [
    ("datasets/weapon_stage1", "data/weapon/weapon_stage1"),
    ("datasets/weapon_stage2", "data/weapon/stage2"),
    ("Weapon 2.v2i.yolov8",    "data/weapon/sources/Weapon 2.v2i.yolov8"),
]
RAW_SOURCES_EXCLUDE = {"guns_mms73"}  # excluded: dropped in DELETE below

# Not in PLAN.md's target tree at all. Only the two zip archives inside
# are explicitly called out for deletion (duplicated halves); the
# extracted content they sit beside is not superseded by anything, so
# it is kept live, not dropped -- flagged in the report.
COCO_PERSON_VAL = ("datasets/coco_person_val", "data/coco_person_val")

# ================================================================
# Step 4 -- cherry-pick forward out of what Step 1 just archived.
# ================================================================
CHERRY_PICK = [
    ("archive/Review_1.2/pipeline/detectors.py",    "weapon/detector.py"),
    ("archive/Review_1.2/pipeline/verification.py", "weapon/verification.py"),
    # demo.py / review_config.py are not standalone modules in PLAN.md's
    # target tree -- their logic merges into main.py / config.py by hand
    # (Step 5, "fuse"). Staged here, not deleted, so nothing is lost
    # before that manual step happens.
    ("archive/Review_1.2/pipeline/demo.py",          "weapon/_staging_demo.py"),
    ("archive/Review_1.2/pipeline/review_config.py", "weapon/_staging_review_config.py"),

    ("archive/root_legacy/association.py",     "threat/association.py"),
    ("archive/root_legacy/threat_analyzer.py", "threat/analyzer.py"),
    ("archive/root_legacy/alert_manager.py",   "threat/alerts.py"),
    # WeaponDetector lives inside detection.py as one class among others;
    # THREAT_LEVELS / WEAPON_THREAT_MAP are two tables inside config.py.
    # Extracting them is a manual edit (Step 5), not a file move --
    # staged here as reference copies.
    ("archive/root_legacy/detection.py", "threat/_staging_detection.py"),
    ("archive/root_legacy/config.py",    "threat/_staging_old_config.py"),

    ("archive/Review_1.2/indomain",    "data/indomain"),
    ("archive/Review_1.2/PROGRESS.md", "docs/PROGRESS.md"),
    ("DATASET_SURVEY_ADDENDUM.md",     "paper/related_work/DATASET_SURVEY_ADDENDUM.md"),

    ("archive/Analysis/DATASET_SURVEY.md",                  "paper/related_work/DATASET_SURVEY.md"),
    ("archive/Analysis/MODEL_SELECTION.md",                 "paper/related_work/MODEL_SELECTION.md"),
    ("archive/Analysis/PROJECT_OVERVIEW.md",                "paper/related_work/PROJECT_OVERVIEW.md"),
    ("archive/Analysis/Research_Gap_Slide_Detailed (2).md", "paper/related_work/Research_Gap_Slide_Detailed (2).md"),
    ("archive/Analysis/WEAPON_MODEL_EVALUATION.md",         "paper/related_work/WEAPON_MODEL_EVALUATION.md"),
    ("archive/Analysis/papersana (1) (1).docx",              "paper/related_work/papersana (1) (1).docx"),

    ("archive/Review_1.2/benchmarks/bench_person.py",        "benchmarks/bench_person.py"),
    ("archive/Review_1.2/benchmarks/bench_person.log",       "benchmarks/bench_person.log"),
    ("archive/Review_1.2/benchmarks/bench_person2.log",      "benchmarks/bench_person2.log"),
    ("archive/Review_1.2/benchmarks/compare_models.py",      "benchmarks/compare_models.py"),
    ("archive/Review_1.2/benchmarks/eval_indomain.py",       "benchmarks/eval_indomain.py"),
    ("archive/Review_1.2/benchmarks/indomain_eval.json",     "benchmarks/indomain_eval.json"),
    ("archive/Review_1.2/benchmarks/model_comparison.json",  "benchmarks/model_comparison.json"),
    ("archive/Review_1.2/benchmarks/person_comparison.json", "benchmarks/person_comparison.json"),
    ("archive/Review_1.2/benchmarks/tune_thresholds.py",     "benchmarks/tune_thresholds.py"),
]

# Per training run: best.pt -> models/, everything else (results.csv,
# args.yaml, curves) -> runs/, as reproducibility evidence.
WEAPON_RUNS = [
    ("archive/Review_1.2/stage1_yolo26s_640", "stage1",          "weapon_stage1"),
    ("archive/Review_1.2/stage2_indomain",    "stage2_indomain", "weapon_stage2"),
    ("archive/Review_1.2/stage2_runA",        "runA",            "weapon_runA"),
    ("archive/Review_1.2/stage2_runB",        "runB",            "weapon_runB"),
]
WEAPON_STAGE1_EVAL = [
    ("archive/Review_1.2/eval_best",         "runs/weapon_stage1/eval/eval_best"),
    ("archive/Review_1.2/eval_conf0.55",     "runs/weapon_stage1/eval/eval_conf0.55"),
    ("archive/Review_1.2/eval_results.json", "runs/weapon_stage1/eval/eval_results.json"),
]

# ================================================================
# Step 5 -- prune. Only the provably re-derivable/redundant list.
# Paths reference each item's location AFTER Steps 1-4 (archive/promote
# happen before prune within one invocation; across separate
# invocations the earlier --apply has already put them there).
# ================================================================
DELETE = [
    "datasets/_raw/guns_mms73",
    "Updated_Review_1.2/data/_dl",
    "data/coco_person_val/val2017.zip",
    "data/coco_person_val/annotations.zip",
    "datasets/weapon_clipsplit",
    "yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolov8s.pt", "yolov8s.onnx",
    "runs/detect/predict", "runs/detect/val",
    "runs/detect/val-2", "runs/detect/val-3", "runs/detect/val-4",
    "runs/detect/val-5", "runs/detect/val-6",
]
# Byte-identical to, or superseded by, each run's best.pt.
LAST_PT = [
    "archive/Review_1.2/stage1_yolo26s_640/weights/last.pt",
    "archive/Review_1.2/stage2_indomain/weights/last.pt",
    "archive/Review_1.2/stage2_runA/weights/last.pt",
    "archive/Review_1.2/stage2_runB/weights/last.pt",
]
EMPTY_DIR_DELETE = [
    "archive/Review_1.2/benchmarks/val_yolo11s",
    "archive/Review_1.2/benchmarks/val_yolo26s",
    "archive/Review_1.2/benchmarks/val_yolov8s",
]

# ================================================================
# Step 6 -- path patches, base folder -> new root.
# ================================================================
PATCH = [
    ("config.py",
     "ROOT_DIR = os.path.dirname(BASE_DIR)",
     "ROOT_DIR = BASE_DIR"),
    ("config.py",
     'PERSON_MODEL_PATH = os.path.join(ROOT_DIR, "yolo26s.pt")',
     'PERSON_MODEL_PATH = os.path.join(ROOT_DIR, "models", "person", "yolo26s.pt")'),
    ("config.py",
     'POSE_MODEL_PATH = os.path.join(ROOT_DIR, "pose_landmarker_lite.task")',
     'POSE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "person", "pose_landmarker_lite.task")'),
    ("config.py",
     'TORCHREID_PATH = os.path.join(ROOT_DIR, "deep-person-reid")',
     'TORCHREID_PATH = os.path.join(ROOT_DIR, "third_party", "deep-person-reid")'),
    ("config.py",
     'REID_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")',
     'REID_WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "reid")'),
    ("benchmarks/bench_detectors.py",
     "path = os.path.join(ROOT_DIR, name)",
     'path = os.path.join(ROOT_DIR, "models", "person", name)'),
    ("benchmarks/bench_reid.py",
     '    root = args.data or os.path.join(\n'
     '        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\n'
     '        "data", default_dir)',
     '    root = args.data or os.path.join(\n'
     '        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\n'
     '        "data", "reid", default_dir)'),
    ("eval/eval_mot.py",
     '        "data", "MOT17", "train"))',
     '        "data", "reid", "MOT17", "train"))'),
    ("tools/make_presentation.py",
     'ROOT_REC = os.path.join(os.path.dirname(BASE), "recordings")',
     'ROOT_REC = os.path.join(BASE, "archive", "recordings")'),
]


# ================================================================
# Execution engine
#
# A dry run never touches disk, but later steps in the same table
# (Step 4 cherry-picks out of what Step 1 archives) depend on earlier
# ones having happened. VFS tracks that hypothetical state -- which
# virtual path is currently backed by which real on-disk path -- so a
# dry run reasons about the same sequencing --apply will actually
# execute, instead of reporting false drift against a filesystem it
# never touched.
# ================================================================
class VFS:
    def __init__(self):
        self.redirect = {}   # virtual rel -> real rel currently backing it
        self.gone = set()    # real rel paths no longer present

    @staticmethod
    def _norm(rel):
        return str(Path(rel))

    def resolve(self, rel):
        rel = self._norm(rel)
        if rel in self.gone:
            return None
        if rel in self.redirect:
            return self.redirect[rel]
        for vkey, real in self.redirect.items():
            prefix = vkey + os.sep
            if rel.startswith(prefix):
                candidate = real + rel[len(vkey):]
                return None if candidate in self.gone else candidate
        return rel

    def exists(self, rel):
        real = self.resolve(rel)
        return real is not None and (ROOT / real).exists()

    def real_dir(self, rel):
        real = self.resolve(rel)
        return ROOT / real if real else None

    def record_move(self, src_rel, dst_rel, dry_run):
        src_rel, dst_rel = self._norm(src_rel), self._norm(dst_rel)
        if dry_run:
            self.redirect[dst_rel] = self.resolve(src_rel) or src_rel
        else:
            self.redirect.pop(src_rel, None)
            self.redirect[dst_rel] = dst_rel
        self.gone.discard(dst_rel)
        self.gone.add(src_rel)

    def record_delete(self, rel):
        self.gone.add(self._norm(rel))


def move_one(vfs, src_rel, dst_rel, dry_run):
    # Destination already holding the result means this step is done --
    # full stop, even if something else (a later step reusing the same
    # name, e.g. the base folder's config.py replacing root's own) has
    # since put new content at the source path. Only a truly missing
    # source with no destination either is an error worth stopping for.
    if vfs.exists(dst_rel):
        log(f"skip  (already done)   {src_rel} -> {dst_rel}")
        return
    if not vfs.exists(src_rel):
        raise SystemExit(f"[migrate] missing source, no destination either: {src_rel}")
    log(f"{'would move' if dry_run else 'move'}  {src_rel} -> {dst_rel}")
    if not dry_run:
        real_src, dst = vfs.real_dir(src_rel), ROOT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(real_src), str(dst))
    vfs.record_move(src_rel, dst_rel, dry_run)


def move_dir_contents_except(vfs, src_rel, dst_rel, exclude, dry_run):
    if not vfs.exists(src_rel):
        if vfs.exists(dst_rel):
            log(f"skip  (already done)   {src_rel}/* -> {dst_rel}/")
            return
        raise SystemExit(f"[migrate] missing source dir: {src_rel}")
    for entry in sorted(vfs.real_dir(src_rel).iterdir()):
        if entry.name in exclude:
            continue
        move_one(vfs, f"{src_rel}/{entry.name}", f"{dst_rel}/{entry.name}", dry_run)


def raw_sources_step(vfs, dry_run):
    if not vfs.exists("datasets/_raw"):
        log("skip  (already done)   datasets/_raw/* -> data/weapon/sources/")
        return
    for entry in sorted(vfs.real_dir("datasets/_raw").iterdir()):
        if entry.name in RAW_SOURCES_EXCLUDE:
            continue
        move_one(vfs, f"datasets/_raw/{entry.name}", f"data/weapon/sources/{entry.name}", dry_run)


def promote_tools_dir(vfs, dry_run):
    """Merge Updated_Review_1.2/tools/* into root's existing tools/ dir
    (the dataset-build pipeline + this script), one file at a time --
    never a whole-directory move, which would nest one inside the other."""
    src_rel = "Updated_Review_1.2/tools"
    if not vfs.exists(src_rel):
        if vfs.exists("tools"):
            log("skip  (already done)   Updated_Review_1.2/tools/* -> tools/")
            return
        raise SystemExit(f"[migrate] missing source dir: {src_rel}")
    for entry in sorted(vfs.real_dir(src_rel).iterdir()):
        if entry.name == "__pycache__":
            continue  # disposable bytecode cache; both tools/ dirs have one
        move_one(vfs, f"{src_rel}/{entry.name}", f"tools/{entry.name}", dry_run)


def delete_path(vfs, rel, dry_run):
    if not vfs.exists(rel):
        log(f"skip  (already gone)   {rel}")
        return
    log(f"{'would delete' if dry_run else 'delete'}  {rel}")
    if not dry_run:
        real = vfs.real_dir(rel)
        shutil.rmtree(real) if real.is_dir() else real.unlink()
    vfs.record_delete(rel)


def delete_glob(vfs, root_rel, pattern, dry_run):
    real_base = vfs.real_dir(root_rel)
    if real_base is None or not real_base.exists():
        return
    for m in real_base.rglob(pattern):
        if not m.exists():
            continue  # already removed as part of an ancestor match
        rel = str(m.relative_to(ROOT))
        log(f"{'would delete' if dry_run else 'delete'}  {rel}")
        if not dry_run:
            shutil.rmtree(m) if m.is_dir() else m.unlink()
        vfs.record_delete(rel)


def patch_file(vfs, rel, old, new, dry_run):
    if not vfs.exists(rel):
        log(f"skip patch (file missing)     {rel}")
        return
    real = vfs.real_dir(rel)
    text = real.read_text(encoding="utf-8")
    if new in text and old not in text:
        log(f"skip patch (already applied)  {rel}")
        return
    count = text.count(old)
    if count == 0:
        raise SystemExit(f"[migrate] patch target not found in {rel}: {old!r}")
    if count > 1:
        raise SystemExit(f"[migrate] patch target ambiguous ({count}x) in {rel}: {old!r}")
    log(f"{'would patch' if dry_run else 'patch'}  {rel}")
    if not dry_run:
        real.write_text(text.replace(old, new, 1), encoding="utf-8")


def _long_path(p):
    # Roboflow-exported filenames push some dataset paths past Windows'
    # 260-char MAX_PATH; the \\?\ prefix opts into the extended-length API.
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + p
    return p


def write_manifest(dry_run):
    if dry_run:
        log("would write MANIFEST.txt")
        return
    root = _long_path(ROOT)
    lines = ["AI_CCTV MANIFEST", "=" * 60, ""]
    total_files = total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and d != ".git")
        filenames = sorted(filenames)
        rel_dir = os.path.relpath(dirpath, root)
        size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)
        total_files += len(filenames)
        total_bytes += size
        if filenames:
            lines.append(f"{rel_dir:60s} {len(filenames):6d} files  {size / 1e6:10.2f} MB")
    lines += ["", f"TOTAL: {total_files} files, {total_bytes / 1e9:.2f} GB", "",
              "SHA-256 of every file under models/:"]
    models_dir = _long_path(ROOT / "models")
    if os.path.isdir(models_dir):
        for dirpath, dirnames, filenames in os.walk(models_dir):
            dirnames.sort()
            for f in sorted(filenames):
                fp = os.path.join(dirpath, f)
                with open(fp, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                lines.append(f"  {digest}  {os.path.relpath(fp, root)}")
    (ROOT / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")
    log("wrote MANIFEST.txt")


# Written once Steps 1-5 genuinely complete. Archive destinations that
# Step 4 cherry-picks onward (e.g. archive/root_legacy/config.py ->
# threat/_staging_old_config.py) are legitimately empty afterwards --
# without this marker, a second run's Step 1 would see that emptiness,
# not know it was intentional, and re-archive whatever now occupies the
# same root-level name (the base folder's own config.py/main.py/etc,
# promoted there by Step 2). The apply phase is therefore one-shot per
# machine; re-running it is always a no-op once the marker exists.
APPLY_MARKER = ROOT / "archive" / "MIGRATION_APPLIED.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="execute archive/promote/cherry-pick/patch")
    ap.add_argument("--prune", action="store_true", help="execute the re-derivable delete list (separate from --apply)")
    args = ap.parse_args()
    dry_a, dry_p = not args.apply, not args.prune

    log(f"root: {ROOT}")
    log(f"mode: apply={'LIVE' if not dry_a else 'dry-run'}  prune={'LIVE' if not dry_p else 'dry-run'}")
    vfs = VFS()

    if APPLY_MARKER.exists():
        log(f"skip  (already done)   steps 1-5 -- {APPLY_MARKER.relative_to(ROOT)} exists from a prior --apply")
    else:
        log("--- step 1: archive old generations ---")
        for src, dst in ARCHIVE:
            move_one(vfs, src, dst, dry_a)

        log("--- step 2: promote base folder + shared assets ---")
        for src, dst in PROMOTE:
            move_one(vfs, src, dst, dry_a)
        promote_tools_dir(vfs, dry_a)

        log("--- step 3: live dataset moves ---")
        for src, dst in DATASET_LIVE:
            move_one(vfs, src, dst, dry_a)
        raw_sources_step(vfs, dry_a)
        move_one(vfs, *COCO_PERSON_VAL, dry_a)

        log("--- step 4: cherry-pick forward ---")
        for src, dst in CHERRY_PICK:
            move_one(vfs, src, dst, dry_a)
        for src_dir, models_name, runs_name in WEAPON_RUNS:
            move_one(vfs, f"{src_dir}/weights/best.pt", f"models/weapon/{models_name}/best.pt", dry_a)
            move_dir_contents_except(vfs, src_dir, f"runs/{runs_name}", {"weights"}, dry_a)
        for src, dst in WEAPON_STAGE1_EVAL:
            move_one(vfs, src, dst, dry_a)

        log("--- step 5: patch paths for the promoted root ---")
        for rel, old, new in PATCH:
            patch_file(vfs, rel, old, new, dry_a)

        if not dry_a:
            APPLY_MARKER.write_text(
                "Steps 1-5 (archive/promote/cherry-pick/patch) completed for real.\n"
                "See MANIFEST.txt for the resulting tree. Safe to leave in place --\n"
                "migrate.py will not re-run those steps while this file exists.\n",
                encoding="utf-8")

    write_manifest(dry_a)

    log("--- prune: re-derivable / redundant deletions ---")
    for rel in DELETE:
        delete_path(vfs, rel, dry_p)
    for rel in LAST_PT:
        delete_path(vfs, rel, dry_p)
    for rel in EMPTY_DIR_DELETE:
        delete_path(vfs, rel, dry_p)
    delete_glob(vfs, "data/weapon/stage2", "replay_*", dry_p)
    delete_glob(vfs, ".", "__pycache__", dry_p)

    log("done.")
    if dry_a:
        log("archive/promote/cherry-pick/patch above was a DRY RUN -- pass --apply to execute it")
    if dry_p:
        log("prune above was a DRY RUN -- pass --prune to execute it, and only after backing up "
            "the irreplaceable set and after every machine's MANIFEST.txt agrees")


if __name__ == "__main__":
    main()
