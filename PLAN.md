# AI_CCTV — Reorganization for Publication

## Context

**Project:** Intelligent Threat Analysis and Surveillance System Using Multi-Module AI Fusion
(Batch ECE002). **23 GB** on disk, 284 GB free — a copy-then-verify migration is affordable.

Three overlapping generations of work share one folder, and the newest one is walled off from
the rest:

- **Root `AI_CCTV/`** — the original monolithic pipeline. It holds the **only** implementation of
  the weapon-threat half of the system (`association.py`, `threat_analyzer.py`,
  `alert_manager.py`, `detection.py`'s `WeaponDetector`, and the `THREAT_LEVELS` /
  `WEAPON_THREAT_MAP` tables in `config.py`), plus all bulk assets.
- **`Review_1.2/`** — the weapon-detection line: a 28,943-image 3-class merged dataset with zero
  cross-split leakage, four trained candidates, a tracker-assisted in-domain set, and a
  detect-then-verify demo that scored 39 alerts on a weapon clip and 0 on a weapon-free clip.
- **`Updated_Review_1.2/`** — the person tracking + re-ID line, and the strongest work: public
  benchmarks reproduced within 1%, five calibrated thresholds traceable to a JSON, 22 passing
  tests, a 1,020-line technical report whose Part 12 gives a runnable command for every number.
  Its own header says *"Person tracking and identity only. No weapon detection, no threat
  analysis, no alerting."* — a grep confirms zero weapon code in it.

A conference paper and a journal paper are being written on the **full system**, so both halves
must be citable and reproducible. Weapon detection continues, rebuilt on the new base, ending in
one fused pipeline where an alert can name a tracked person.

**Outcome wanted:** one coherent project rooted at `AI_CCTV/`, built on `Updated_Review_1.2`'s
structure and conventions, every paper-citable artifact in a predictable place, dead weight gone.

## Decisions fixed

| Question | Answer |
|---|---|
| Paper scope | Full threat-detection system — re-ID **and** weapon detection |
| Weapon detection | Continues; rebuilt as a module on the new base |
| How they run together | **One fused pipeline in `main.py`** |
| Large files | Stay on this machine, organized in place |
| Root layout | `Updated_Review_1.2/` contents move **up** to become `AI_CCTV/` |
| Git contents | Code + docs + results JSON only |
| Superseded material | `archive/`, intact, then cherry-pick forward |
| Figure pipeline | Leave as-is, **document** which figures are data-derived |
| Occlusion re-run, team split, timeline | Deferred |

## Two constraints that shape everything

**1. The technical report pins the filenames.** `TECHNICAL_REPORT.md` Part 5 walks through
`detector.py`, `embedder.py`, `tracker.py`, `pose.py`, `render.py` by name; Part 12 lists exact
commands. Every script in `benchmarks/`, `eval/`, `tools/`, `tests/` bootstraps with
`sys.path.insert(0, dirname(dirname(abspath(__file__))))` and imports siblings flat. So the
re-ID modules **keep their names and sit at the new root**; weapon and threat are added
alongside as new packages. Nothing the report documents moves.

**2. `Review_1.2/pipeline/` cannot run standalone.** It imports 11 settings from the root
`config.py`, `association` from the root, and `pose_estimator` / `threat_analyzer` / `tracker`
from the root. That is not accidental coupling — `review_config.py` says it is deliberate, "so
the demo stays consistent with the production pipeline." Porting the weapon pipeline therefore
*requires* porting the threat modules with it. They move as one unit.

## Target structure

```
AI_CCTV/
├── README.md                 NEW — how to run, where things live, known caveats
├── requirements.txt          ← root's existing one (the only one in the tree)
├── .gitignore                NEW
├── config.py                 base file + new WEAPON and THREAT sections
├── main.py                   fused live pipeline
│
├── detector.py  embedder.py  gallery.py  tracker.py  pose.py  render.py
│                             person re-ID — names and paths unchanged
│
├── weapon/                   NEW ← Review_1.2/pipeline/
│   ├── detector.py               wide detect @0.25 + geometric plausibility filter
│   └── verification.py           3 gates: class-conf, temporal persistence, person association
│
├── threat/                   NEW ← root legacy; the only implementation anywhere
│   ├── association.py            3-tier: wrist ≤60px → containment ≥0.30 → IoU ≥0.20
│   ├── analyzer.py               threat state machine, posture + approach escalation, hysteresis
│   └── alerts.py                 CSV log, screenshot on escalation, recording, audio
│
├── benchmarks/               re-ID (5 scripts) + weapon (4 scripts) merged
├── eval/                     re-ID eval + clips + ground_truth + weapon eval
├── tests/                    test_gallery.py — 22 checks, plain script, not pytest
├── tools/                    capture, dataset build, figure + video generation
│
├── models/
│   ├── person/  yolo26s.pt, pose_landmarker_lite.task
│   ├── weapon/  stage1, stage2_indomain (deployed), runA, runB — best.pt only
│   └── reid/    3 × osnet_x1_0_*.pth
├── data/
│   ├── weapon/      weapon_stage1 (28,943 img), stage2, sources/
│   ├── indomain/    the 174 in-domain frames
│   ├── reid/        Market-1501, MSMT17, MOT17
│   └── clips/       5 clean camera clips + ground truth
├── runs/                     results.csv, args.yaml, curves, logs — reproducibility evidence
│
├── docs/                     TECHNICAL_REPORT, RESULTS, PROGRESS, viva prep, SCENARIOS
├── paper/
│   ├── conference/  journal/  related_work/  figures/  tables/
├── third_party/deep-person-reid/
└── archive/                  superseded, intact, gitignored
```

## Paths to update when the base folder moves up

| File | Current | Becomes |
|---|---|---|
| `config.py:21` | `ROOT_DIR = dirname(BASE_DIR)` | `ROOT_DIR = BASE_DIR` |
| `config.py:23-25` | `ROOT_DIR/{yolo26s.pt, pose_landmarker_lite.task, deep-person-reid}` | `models/person/…`, `third_party/…` |
| `config.py:91-92` | `BASE_DIR/weights` | `models/reid/` |
| `benchmarks/bench_detectors.py` | `ROOT_DIR/*.pt` | `models/person/` |
| `benchmarks/bench_reid.py` | `data/Market-1501…`, `data/MSMT17_V1` | `data/reid/…` |
| `eval/eval_mot.py` | `data/MOT17/train` | `data/reid/MOT17/train` |
| `tools/make_presentation.py:33` | `ROOT_REC = ../recordings` | `archive/recordings/` |
| every `runs/*/args.yaml` | `Documents\AI_CCTV\datasets\…` | stale — missing the `Projects\` level; rewrite or note |
| `weapon/` (ported) | 11 imports from root `config` + `association` | the new `config.py` and `threat/` |

## Answering the `.pt` question

**No — you do not need all of them.** Reviewers ask for numbers, configs and seeds. What
justifies success in a paper is small and text-based: `results.csv`, `args.yaml`, the benchmark
JSONs, and the training logs — the logs are especially good, because each one carries the full
resolved `engine\trainer:` line (~110 hyperparameters as actually applied), the exact stack
(`Ultralytics 8.4.108, Python 3.12.10, torch 2.11.0+cu128`), the exact GPU, the layer-by-layer
architecture, the dataset scan counts, and per-epoch validation.

| Keep | Why |
|---|---|
| 4 × weapon `best.pt` — stage1, stage2_indomain, runA, runB | All four appear in the selection table. **Keep runA/runB especially** — they scored 0% miss yet fire on 35% of weapon-free frames, which is the evidence for the scene-memorization finding. That is a real negative result and a genuine contribution |
| 3 × `osnet_x1_0_*.pth` | RESULTS.md reports a separate cross-domain number for each |
| `yolo26s.pt`, `pose_landmarker_lite.task` | Live inference dependencies |

| Drop | Size | Why |
|---|---:|---|
| `stage1_yolo26s_640/weights/last.pt` | 60 MB | **Byte-identical MD5 to that run's `best.pt`** — the run was killed at epoch 52 before the strip-optimizer pass, so both files are the same 60 MB checkpoint carrying optimizer state. Keep one; it can also be stripped to ~20 MB |
| 3 × stage-2 `last.pt` | ~61 MB | Final-epoch snapshots, superseded by each run's `best.pt` |
| `yolo11n/s.pt`, `yolo26n.pt`, `yolov8s.pt`, `yolov8s.onnx` | ~98 MB | Verified stock — scanning each checkpoint's strings found only `coco.yaml`, no weapon-dataset paths |

## Irreplaceable — never delete

| Asset | Size | Why |
|---|---:|---|
| `recordings/` (4 AVI, 661 frames) + `screenshots/` (5 JPEG) | 12 MB | One-off live captures of a staged armed-threat demo on four dates, overlay burnt into the pixels. **`20260411_100507.avi` is the single source of every in-domain weapon number in the project** |
| `Review_1.2/indomain/` (174 frames + `label_check.jpg`) | ~9 MB | See the correction below |
| `association.py`, `threat_analyzer.py`, `alert_manager.py`, `WeaponDetector`, threat tables | ~34 KB | Irreplaceable as authorship — no counterpart exists in the base folder |
| `DATASET_SURVEY_ADDENDUM.md` | 13 KB | Live Roboflow API verification plus a documented dead link (the ACF repo now 404s) that cannot be re-derived |
| `datasets/weapon_stage1/build_manifest.json` | 4 KB | Provenance: per-source counts, 18,480 near-duplicates absorbed, 2,299 cross-source duplicate groups. Load-bearing for the papers |
| `Analysis/WEAPON_MODEL_EVALUATION.md` | 12 KB | The leakage post-mortem — publication-grade methodology on its own |
| `Analysis/Research_Gap_Slide_Detailed (2).md` | 12 KB | 17-paper table mapped to 5 research gaps, each with the project's counter. Effectively a written Related Work + gap statement |

**Two corrections to assumptions I made earlier in this session:**

1. `recordings/` is **not** merely superseded. RESULTS.md §8 calls it unusable *for re-ID
   calibration* (annotation burnt in, separability 0.114) — and that is true. But it is the only
   footage of the armed-threat demo, and the papers now cover that half. It is evidence.
2. The in-domain set is **not hand-labelled**. `Review_1.2/tools/label_indomain.py` seeds a CSRT
   tracker from high-confidence detections and propagates boxes forward and backward through one
   clip, then a human verified the result on a contact sheet. 174 frames, **all class 0 (guns)**,
   one deliberate 0-byte negative, a single contiguous gap where frames 111–146 were rejected at
   review. It is regenerable from the recording — but the recording, the replica and the room are
   gone, which is what makes it irreplaceable.

## What gets scrapped

### Delete — re-downloadable or re-derivable

| Target | Size | Why |
|---|---:|---|
| `datasets/_raw/guns_mms73/` | 1.3 GB | `exclude: true` in `class_map.yaml`. **Zero of its 23,130 images reach any built dataset** — its export collapses rifles into `gun`, contradicting `long_gun`; also only 2,488 unique images at 9.3× augmentation |
| `Updated_Review_1.2/data/_dl/*.zip` | 2.6 GB | Download archives for Market-1501 and MSMT17, both already extracted alongside |
| `datasets/coco_person_val/` duplicated halves | ~1.0 GB | `val2017.zip` (778 MB) and `annotations.zip` (242 MB) sit beside their own extracted folders |
| `datasets/weapon_stage2/**/replay_*` | ~600 MB | 6,802 verbatim copies of stage-1 images |
| `datasets/weapon_clipsplit/` | 142 MB | 2-class re-split of Weapon-2 only, fully contained in `weapon_stage1` |
| `last.pt` files + stock weights | ~219 MB | Per the table above |
| `benchmarks/val_yolo11s/`, `val_yolo26s/`, `val_yolov8s/` | 0 B | **Completely empty** — leftover Ultralytics `save_dir` stubs. The person-comparison artifacts are the JSON and the two logs |
| `runs/detect/{predict,val,val-2..val-6}/` | 0 B | Empty directories |
| `__pycache__/` throughout | ~200 KB | |
| `old/z_test.py` | 27 B | Byte-for-byte duplicate of `old/Untitled-2.py` |

Recoverable: **~6 GB**, taking the project from 23 GB to roughly 17 GB.

### Archive intact — superseded, still evidence

- `Review_1.2/` → `archive/Review_1.2/` (then cherry-pick, below)
- `Analysis/` → `archive/Analysis/` — including all three `.pptx`. They are a genuine version
  history, not duplicates: 14 → 19 → 21 slides, newest is `project_eval1 (1).pptx` (Aug 11 12:48)
- `old/` → `archive/old_prototypes/` — all eleven superseded. The one thing that exists nowhere
  else is `cctv_proto1.py`'s **serial/Arduino alert over COM3**; no other file imports `serial`
- `runs/detect/train3/`, `runs/detect/weapon_yolov8s/` → `archive/runs_legacy/` — trained on the
  old 2-class Weapon-2 split with stale paths. `weapon_yolov8s/weights/best.pt` is still
  `WEAPON_MODEL_PATH` in the old root `config.py` and in four `old/` scripts, so it must survive
  until `weapon/` is wired
- `threat_log.csv` → `archive/` — 1,566 rows, **no header**, schema drift from 10 to 18 fields.
  1,558 rows are one uncalibrated 2026-02-27 session predating log de-duplication; only 8 rows
  pair with the irreplaceable recordings
- `recordings/`, `screenshots/` → `archive/` — irreplaceable, but not live inputs

### Live — moved forward, not archived

- `datasets/weapon_stage1/` (2.7 GB, 28,943 images) → `data/weapon/`
- `datasets/weapon_stage2/` minus `replay_*` → `data/weapon/stage2/`
- **`Weapon 2.v2i.yolov8/` (142 MB) → `data/weapon/sources/`, not `archive/`.** Its pixels are
  already inside `weapon_stage1`, so it looks redundant — but `class_map.yaml` references it by
  **local path** as `weapon2_original` and `resplit_dataset.py` reads it directly. Remove it and a
  rebuild silently drops 4,098 images down the `[missing]` branch of `build_dataset.py:148`
- Root `tools/` (dataset build pipeline) → merged into `tools/`

## Migration: move intact, then cherry-pick

**Step 1 — stage.** Create `archive/`, `models/`, `data/`, `runs/`, `docs/`, `paper/`,
`third_party/`, `weapon/`, `threat/`.

**Step 2 — archive the old generations, whole.** `Review_1.2/`, `Analysis/`, `old/`,
`recordings/`, `screenshots/`, `threat_log.csv`, `runs/`, and the root legacy `.py` files
(`main.py`, `config.py`, `detection.py`, `tracker.py`, `pose_estimator.py`, `cam_test.py`) into
`archive/root_legacy/`.

> **Ordering hazard:** the old root `config.py`, `main.py` and `tracker.py` are *different files*
> that share names with the base folder's. The archive move must complete **before** the base
> folder's contents come up, or the wrong file wins silently.

**Step 3 — promote the base folder.** Move `Updated_Review_1.2/*` up one level and apply the path
table. `yolo26s.pt` + `pose_landmarker_lite.task` → `models/person/`, `deep-person-reid/` →
`third_party/`, `weights/*.pth` → `models/reid/`.

**Step 4 — cherry-pick forward.**

| Artifact | From | To |
|---|---|---|
| Weapon pipeline (4 files, 24 KB) | `archive/Review_1.2/pipeline/` | `weapon/` |
| Threat modules (4 files, 34 KB) | `archive/root_legacy/` | `threat/` |
| 4 × `best.pt` | `archive/Review_1.2/*/weights/` | `models/weapon/` |
| `results.csv`, `args.yaml`, curves, logs | `archive/Review_1.2/` | `runs/weapon_{stage1,stage2,runA,runB}/` |
| In-domain frames + `label_check.jpg` | `archive/Review_1.2/indomain/` | `data/indomain/` |
| Weapon benchmark scripts + JSONs | `archive/Review_1.2/benchmarks/` | `benchmarks/`, `paper/tables/` |
| Stage-1 eval (`eval_best/`, `eval_conf0.55/`, `eval_results.json`) | `archive/Review_1.2/` | `runs/weapon_stage1/eval/` |
| `PROGRESS.md`, reports, viva prep | both archives + base root | `docs/` |
| The 5 survey/gap markdowns + the 17-paper docx | `archive/Analysis/` | `paper/related_work/` |
| `DATASET_SURVEY_ADDENDUM.md` | root | `paper/related_work/` |

**Step 5 — fuse.** Extend `config.py` with `WEAPON` and `THREAT` sections in the existing comment
style, and extend `main.py` so a frame passes through weapon detection and threat association
after identity resolution — so an alert names a tracked person. The re-ID path stays exactly as
documented. The port is mostly re-pointing the 11 cross-folder imports at the new `config.py` and
`threat/`.

**Step 6 — scaffolding.** `README.md` and `.gitignore` (exclude `data/`, `models/`, `runs/`,
`archive/`, `output/`, `__pycache__/`, `*.avi`, `*.mp4`), matching the code-plus-docs-plus-JSON
decision for the GitHub repo.

## Paper evidence map — what exists, what does not

**No paper draft exists.** `Analysis/papersana (1) (1).docx` is 65 pages of pasted chat transcript
producing literature-survey table rows — entries #11–27, with PDF filenames and conversational
turns left in, entry #12 duplicated, no title, abstract or section headings. There is no `.bib`,
no `.ris`, no LaTeX anywhere.

| Paper section | What already exists | Gap |
|---|---|---|
| Related Work | `Research_Gap_Slide_Detailed (2).md` — the same 17 papers, organized into 5 gaps, each with a per-paper limitation table and the project's concrete counter | Needs prose, not tables |
| Datasets | `DATASET_SURVEY.md` (26 Roboflow projects probed via API), `DATASET_SURVEY_ADDENDUM.md`, `build_manifest.json`, `TECHNICAL_REPORT.md` Part 3 | Complete |
| Method | `TECHNICAL_REPORT.md` Parts 5–6 (re-ID), `PROGRESS.md` (two-stage verification), `PROJECT_OVERVIEW.md` (architecture, legacy) | Complete |
| Model selection | `MODEL_SELECTION.md` (VRAM, throughput, latency tables), `person_comparison.json`, `detector_comparison.json` | Complete |
| Results — re-ID | RESULTS.md, `thresholds.json`, MOT17 / Market-1501 / MSMT17 JSONs, 22/22 tests | Complete |
| Results — weapon | `model_comparison.json`, `eval_results.json`, `indomain_eval.json`, per-run curves and confusion matrices | Complete |
| Negative results | `WEAPON_MODEL_EVALUATION.md` (frame-level leakage: 4 videos supply 100% of the val set), runA/runB scene memorization, four killed hypotheses in `PROGRESS.md` | Unusually strong |
| Limitations | `PROGRESS.md` (4 stated honestly), RESULTS.md §8–9 | Complete |
| Abstract, Intro, Conclusion, references | — | **Nothing exists** |

### Three model generations that must never share a results table

| Gen | Model | Classes | Data | mAP50-95 | Status |
|---|---|---|---|---|---|
| (a) | YOLOv8s | 2 | Weapon-2 v2 | 0.702 | **Author-declared untrustworthy** — Roboflow split frame-wise, so 4 phone videos leak and supply 100% of the val set |
| (b) | YOLO26s | 3 | merged 28,943, zero leakage at Hamming ≤10 | 0.654 | stage-1 |
| (c) | YOLO26s fine-tuned | 3 | + in-domain | 0.593 | **deployed** |

The 0.702 figure still appears in `PROJECT_OVERVIEW.md` and the older slides. It is the highest
number in the project and the least defensible — a reviewer who finds the leakage before you
disclose it is a rejection. The honest framing is already written in
`WEAPON_MODEL_EVALUATION.md`: lead with it as a negative result.

### Other inconsistencies to resolve before submission

- **Two stage-1 in-domain miss rates**: `indomain_eval.json` reports 74.29% @0.25 / 87.14% @0.55
  (70-frame full clip); `model_comparison.json` reports 55.6% / 85.7% (63-frame held-out last 30%).
  `PROGRESS.md` quotes the latter. Same phenomenon, different frame sets — pick one, state which.
- **`MODEL_SELECTION.md` recommends imgsz 960; everything trained and infers at 640**, and 960 at
  inference made the in-domain miss rate *worse* (56% → 68%). The recommendation was superseded by
  measurement; say so rather than leaving both documents standing.
- **Never quote a false-positive rate from `recordings/`** — `PROGRESS.md` already retracts that
  claim: those clips are the pipeline's own annotated output and contain a real weapon.
  `tune_thresholds.py` Part B is blocked on clean negatives and produced no JSON.
- **`REVIEW_RESULTS.md` was never written** — listed as the only review-blocking item, ~1–2 h, no
  GPU needed, all numbers already in `PROGRESS.md` and `benchmarks/*.json`.
- Every in-domain weapon number traces to **one clip, one replica, one room, one person**, split
  by time not scene. `PROGRESS.md` states this. The papers must too.

## Known state of the base — documented, not fixed

- **Figure provenance**: of 20 presentation figures, 1–6 and 14–15 regenerate by re-running the
  real pipeline over clips; 17 and 19 read benchmark JSON; **7–13, 18, 20 are matplotlib charts
  with the numbers hardcoded as literals**; **16 has no generator script at all**. If a number
  changes, those seven must be edited by hand.
- `eval/occlusion_results.json` was measured on the superseded `recordings/`, not `eval/clips/`.
- `benchmarks/market1501_results.json` does not match the current output template
  (`reid_market1501_results.json`); `tools/make_analysis_graphs.py:82` reads the old name.
- Ground truth is **per-event, not per-frame**; only `clip1` and `clip9` have populated `events` —
  those 8 events are the "100% (8/8)" claim. The `people` field is still placeholder text in all 5.
- Dead references: `tracker.py` imports `ADMIT_MIN_CONF` and `torch` unused; `gallery.py` imports
  `MATCH_SHORTLIST_M` unused (the two-stage shortlist is commented but never implemented);
  `config.ADMIT_REQUIRE_MATCHED` is never imported — the behaviour is hardcoded in
  `gallery.should_admit`. `ADMIT_MIN_CROP_H = 64` is flagged in `SCENARIOS.md` as a guess.
- `tests/test_gallery.py` is **not pytest** — a plain script run as `python tests/test_gallery.py`.
- **Dataset rebuild is not guaranteed reproducible**: `fetch_datasets.py` hard-exits without a
  `ROBOFLOW_API_KEY`, seven of eight sources are pinned Roboflow versions outside your control
  (the addendum already documents two Tier-1 picks that became undownloadable), and the eighth
  (`weapon2_original`) is a local folder with no remote. The build itself is deterministic given
  identical `_raw/` inputs (seed 0, val_frac 0.15, Hamming ≤10, sorted groups).
- **Housekeeping from `PROGRESS.md`**: the Roboflow API key was pasted into chat twice and should
  be rotated.

## Conventions for the new weapon and threat modules

So new code is indistinguishable from existing code:

- Library modules open with a `# ===…===` banner (60 `=`), name, one-line purpose, then a prose
  paragraph on the design decision. Runnable scripts use a `"""…"""` docstring ending in `Usage:`.
- Constants carry `# base: N` annotations with the previous value and a one-clause reason;
  measured values carry the literal marker word `CALIBRATED`.
- Scripts bootstrap with `sys.path.insert(...)` then imports tagged `# noqa: E402`.
- Scripts fail fast: `raise SystemExit(f"lowercase message: {path}")`. Library code degrades
  loudly — `print("[ClassName] WARNING: …")` and continue. No `logging` module anywhere.
- Unmeasurable values return `None` and the term is omitted from scoring, never faked.
- 4-space indent, ~79 chars, British spelling, ` -- ` for em-dashes, ~18% comment density, and
  non-obvious comments cite a measured number or a `file:line`.

## Making the categorization identical on all three machines

All three machines already hold the same 23 GB, transferred by hard drive, and the drive cannot
be sent again. So each person runs the reorganization locally — but **not by hand.** Three people
executing a prose plan manually is precisely how structures diverge: different orderings,
different judgment calls on the ambiguous items, typos in folder names.

**The reorganization is written once as a script, committed, and run identically by all three.**

### `tools/migrate.py` — the single source of truth

Every move, rename and deletion from this plan encoded as data, not prose:

```python
ARCHIVE = [          # (source, destination) — moved intact
    ("Review_1.2",            "archive/Review_1.2"),
    ("Analysis",              "archive/Analysis"),
    ("old",                   "archive/old_prototypes"),
    ...
]
PROMOTE = [...]      # Updated_Review_1.2/* -> root
CHERRY_PICK = [...]  # the table in step 4
DELETE = [...]       # only the provably re-derivable list
PATCH = [...]        # the path-rewrite table, as (file, old, new)
```

Requirements, in the base folder's own style — `raise SystemExit(f"…")` on a missing
precondition, `[migrate] ` prefixed progress lines, no `logging`:

- **`--dry-run` is the default.** Prints every action it would take and stops. Nothing moves
  until `--apply` is passed explicitly.
- **Idempotent.** Re-running after a partial or complete run is a no-op, not a second migration.
  Each step checks whether its destination already holds the result.
- **Fails fast on drift.** If a source is missing or a destination already exists with different
  content, it stops and names the path rather than guessing. Their copies may have drifted —
  stray `__pycache__`, a demo they ran, output files — so this matters.
- **Deletion is a separate flag** (`--prune`), never bundled with `--apply`.
- **Writes `MANIFEST.txt` on completion**: the full folder tree, per-folder file counts and byte
  totals, and a SHA-256 of every file under `models/`. This is the artifact the three of you
  compare — identical manifests prove identical categorization; a diff names exactly what drifted.

### Order of operations across the three machines

1. **You run it first**, `--dry-run` then `--apply`, and work through anything it flags.
2. You commit `tools/migrate.py`, `MANIFEST.txt` and this plan to the repo.
3. They pull, run `--dry-run`, then `--apply`.
4. All three compare `MANIFEST.txt`. Any difference is a real difference, located precisely.
5. Only after all three manifests agree does anyone run `--prune`.

### Serious warning about the deletion phase

**Right now those three hard-drive copies are each other's only backup.** There is no git history
here, and the drive cannot be sent again. If all three of you run `--prune` you destroy that
redundancy simultaneously, and the ~21 MB of genuinely irreplaceable material — the four
recordings, the five screenshots, the 174 in-domain frames — has no other copy anywhere.

Two protections, and I would not skip either:

- **Before any `--prune`, copy the irreplaceable set off-machine.** It is about 21 MB — it fits in
  an email attachment, a Drive folder, or Git LFS. That is a trivially cheap insurance policy
  against permanently losing the only in-domain weapon data the project has.
- **Stagger the prune.** You prune first and confirm everything still runs; they prune only after
  that. Keeping one unpruned copy for a while costs 6 GB and buys a full rollback.

### What git carries afterwards

Once migrated, `code + docs + results JSON` keeps the structure aligned going forward. Because git
does not track empty directories, the repo also carries a `.gitkeep` and a short `MANIFEST.md` in
each bulk folder (`data/`, `models/`, `runs/`, `archive/`) stating what belongs there and where it
came from — so the skeleton matches on any machine, including a fresh clone with no hard drive
behind it.

## Verification

Run from the new root, in order. Nothing is deleted until 1–8 pass.

1. **Nothing lost** — file count and total size before vs after; 23 GB in, ~17 GB out, with the
   ~6 GB accounted for line by line against the scrap table.
2. **Paths resolve** — `python -c "import config; print(config.PERSON_MODEL_PATH,
   config.TORCHREID_PATH, config.REID_MODEL_PATH)"`, then confirm each exists.
3. **Tests pass** — `python tests/test_gallery.py` prints `ALL TESTS PASSED` (22/22). No GPU,
   weights or video needed, so it is the fastest real signal.
4. **Re-ID pipeline runs** — `python main.py --video eval/clips/clip9_even_lighting.avi
   --headless --max-frames 200`.
5. **A benchmark reproduces** — `python eval/eval_reid.py --clips eval/clips --gt
   eval/ground_truth` still reports 8/8 events, matching RESULTS.md.
6. **Dataset-dependent benchmark resolves** — `python benchmarks/bench_reid.py --dataset
   market1501` gets past its `raise SystemExit` path check.
7. **Weapon demo runs** — the ported `weapon/` + `threat/` reproduce 39 confirmed alerts on the
   weapon clip and 0 on the weapon-free clip, matching `PROGRESS.md`. This is the real test that
   the cross-folder import port worked.
8. **Archive inert** — nothing in the live tree imports from `archive/`.
9. **All three machines agree** — `MANIFEST.txt` is identical on all three. This is the check
   that the categorization actually came out the same, rather than three similar-looking trees.
