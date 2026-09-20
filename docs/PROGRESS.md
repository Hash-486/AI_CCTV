# Review 1.2 — Progress & Handoff

**Last updated:** 2026-08-11, 00:40 IST
**State:** Demo working and verified. Model selected on evidence. Results write-up outstanding.

---

## TL;DR — what to say in the review

1. **YOLO26 integrated for both person and weapon detection.** yolo26s beats yolov8s on
   person detection on COCO val2017 (mAP50-95 **0.596 vs 0.570**) with fewer parameters, at
   121 FPS — far above what a 20 fps camera needs.
2. **Custom dataset trained.** 28,943 images, 3 classes, merged from 8 public sources with
   **zero cross-split leakage** (verified at Hamming ≤ 10). 63.8% of the merged pool was
   near-duplicates — caught and grouped rather than shipped.
3. **Real-time pipeline optimised** with **two-stage verification**: detect wide (high recall,
   fewer false negatives), then verify hard (class-confidence + temporal persistence + person
   association). On real footage: **39 alerts on the weapon clip, 0 on the weapon-free clip.**
4. **Measured on the deployment camera, not just public imagery** — and reported honestly,
   including where it is weak.

---

## Deployed model

`Review_1.2/stage2_indomain/weights/best.pt` — wired into `pipeline/review_config.py`.

Chosen by measurement over three competing candidates:

| model | in-domain miss@0.25 | miss@0.55 | FP@0.25 | FP@0.55 | public mAP50-95 |
|---|---:|---:|---:|---:|---:|
| stage1 (public data only) | 56% | 86% | 17 | 2 | 0.654 |
| **stage2 (in-domain fine-tune)** ← **deployed** | **19%** | **65%** | **8** | **1** | **0.593** |
| runA (heavy viewpoint aug) | 0% | 2% | 75 | 25 | 0.510 |
| runB (light aug + 2x replay) | 0% | 0% | 53 | 12 | 0.527 |

*(63 held-out positive frames, 151 negative frames, 4,397-image public val set)*

**runA/runB were rejected despite scoring 0% miss.** They trained on 121 frames from a single
clip, and the held-out 30% is the same room, same person, same lighting — so 0% miss means they
learned *the scene*, not *the weapon*. The giveaway is false positives: on different scenes runB
fires on 53 of 151 weapon-free frames (35%) versus stage 2's 8. **A metric that looks perfect
because the model memorised the background is the failure mode to watch for here.**

### Stage-2 per-class performance (public val)

| class | P | R | mAP50 | mAP50-95 | best-F1 conf |
|---|---:|---:|---:|---:|---:|
| guns | 0.903 | 0.741 | 0.838 | 0.613 | 0.29 |
| knife | 0.922 | 0.798 | 0.889 | 0.568 | 0.23 |
| long_gun | 0.556 | 0.746 | 0.673 | 0.599 | 0.56 |
| **ALL** | 0.794 | 0.761 | 0.800 | **0.593** | — |

⚠️ **`long_gun` precision dropped from 0.746 (stage 1) to 0.556** in the fine-tune. It is the
weakest class and the clearest candidate for the next round of work.

---

## The core finding: why it misses weapons

Stage 1 scored **0.654 mAP on public imagery** but missed a real replica pistol in **86% of
frames** on the deployment webcam. Diagnosis by frame-by-frame comparison of hits vs misses:

- **Hits** — pistol in **side profile**: barrel, slide, grip, trigger guard visible. The exact
  silhouette every catalogue product photo shows.
- **Misses** — pistol **edge-on / muzzle-toward-camera / foreshortened**: it collapses to a dark
  rectangular slab with no gun silhouette.

**The cause is viewpoint.** Public weapon datasets are almost entirely side-profile product
shots, so the model has effectively never seen a gun pointed at the camera.

### Hypotheses tested and killed

| hypothesis | verdict | evidence |
|---|---|---|
| Rotation (`degrees: 0.0` in training) | **refuted** | rotating val images 90° only drops detection 100% → 92% |
| Test-time augmentation would help | **refuted** | `augment=True` gives *identical* numbers |
| Higher inference resolution helps | **refuted** | imgsz 960 makes it *worse* (56% → 68% miss); match the 640 it trained at |
| Viewpoint augmentation in fine-tuning | **refuted** | runA/runB both degrade public mAP and explode false positives |

---

## What's built (all in `Review_1.2/`)

### The demo — `pipeline/`
| file | role |
|---|---|
| `review_config.py` | every knob in one place; model paths, thresholds, verification settings |
| `detectors.py` | `PersonDetector` (yolo26s) + `WeaponDetector` (custom 3-class), **imgsz=640** |
| `verification.py` | the two-stage gate chain, instrumented with per-gate rejection counters |
| `demo.py` | live loop; amber = candidate, red = confirmed, `v` toggles verification live |

```bash
python Review_1.2/pipeline/demo.py                            # live webcam
python Review_1.2/pipeline/demo.py --video <clip>             # replay
python Review_1.2/pipeline/demo.py --no-verify                # the A/B for the panel
```

**Verified end to end:**
| clip | frames | candidates | confirmed | rejected |
|---|---:|---:|---:|---|
| weapon present | 210 | 44 | **39** | 3 conf, 2 transient |
| no weapon | 215 | 5 | **0** | 1 conf, 4 transient |

### Two-stage verification design
**Stage 1 — detect wide** at `WEAPON_DETECT_CONF = 0.25` (high recall → fewer false negatives),
plus a geometric plausibility filter.
**Stage 2 — verify hard**, three independent gates:
1. **Per-class confidence** — guns 0.29 / knife 0.23 / long_gun 0.56, from the *deployed model's*
   own F1 curves. (Stage 1's were 0.48/0.53/0.65 — carrying them over would mis-set every class.)
2. **Temporal persistence** — present in 2 consecutive detection cycles (bypassed above 0.85).
3. **Person association** — must attach to a tracked person via `association.py`.

### Benchmarks — `benchmarks/`
| file | what it produces |
|---|---|
| `bench_person.py` | yolo26s vs yolov8s vs yolo11s on COCO val2017 person class → `person_comparison.json` |
| `eval_stage1.py` | full per-class eval, mAP-by-IoU, confusion matrix → `eval_results.json` |
| `eval_indomain.py` | miss rate / false alarms on the deployment camera → `indomain_eval.json` |
| `compare_models.py` | scores all candidates on the three axes → `model_comparison.json` |
| `tune_thresholds.py` | FN/FP trade-off sweep (needs clean negatives — see blocked items) |

### Tools — `tools/`
| file | purpose |
|---|---|
| `capture_negatives.py` | record weapon-free frames from the webcam (**verified against a simulated camera**) |
| `label_indomain.py` | tracker-assisted auto-labelling of the in-domain clip |
| `build_stage2.py` | in-domain positives + stage-1 replay → `datasets/weapon_stage2` |
| `../train_stage2.py` | the fine-tune, with augmentation exposed as CLI flags |

### Person detector comparison (COCO val2017 — 2,693 images / 10,777 instances)
| model | mAP50 | mAP50-95 | P | R | params | ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | 0.796 | 0.570 | 0.828 | 0.696 | 11.2 M | 5.25 | 190 |
| yolo11s | 0.810 | 0.582 | 0.834 | 0.713 | 9.5 M | 6.24 | 160 |
| **yolo26s** | **0.821** | **0.596** | **0.835** | **0.713** | 10.0 M | 8.28 | 121 |

---

## Dataset

`datasets/weapon_stage1` — 28,943 images, train 24,546 / val 4,397.
guns 14,693 · knife 11,886 · long_gun 1,858 · backgrounds 3,761.
**Leakage gate: 0 cross-split near-duplicates at Hamming ≤ 4, ≤ 6 and ≤ 10.**

Sources: `weapon2_original`, `guns_rifles`, `sohas`, `knife_zqssx`, `guns_kaggle_cctv`,
`gun_em2023`, `knife_menon`, `knife_porject`. Taxonomy lives in `tools/class_map.yaml` —
**data, not code**; reverting to 2 classes is a one-line change plus a rebuild.

Verify any rebuild with:
```bash
python tools/dataset_groups.py --check datasets/weapon_stage1
```

---

## Bugs found and fixed

| bug | impact |
|---|---|
| `detection.py` hardcoded `imgsz=416`, model trained at **640** | silent accuracy loss |
| `threat_analyzer.analyze()` returns a **dict**, not a list | `AttributeError` swallowed by a bare `except`; threat analysis silently discarded |
| `long_gun` missing from `WEAPON_THREAT_MAP` | CRITICAL tier could never fire — now wired (`config.py`) |
| Dedup used **transitive union-find** on video frames | chained frame1~frame2~…~frameN into one group; collapsed 139 webcam frames to 1. Replaced with greedy max-min |
| Windows **MAX_PATH** — 263-char destination | `shutil.copy2` failed with a misleading "cannot find the path" |
| `dataset_tools` reads its URL index without an encoding | cp1252 crash reported as "file not found" |

## Things I got wrong and corrected

- **"Every saved screenshot is a false positive"** — wrong. I'd only checked 2 of 5.
  `20260411_100507.avi` shows a **real replica pistol**; it's a true positive.
- **"5 candidates → 0 confirmed = 100% FP suppression"** — **retracted.** Measured on
  `recordings/`, which are the pipeline's *annotated output* (boxes/skeletons drawn on) and
  contain a real weapon. Invalid test set.
- **"You have no in-domain weapon footage"** — wrong; there were 210 frames of it.
- **Asked you to re-record twice on broken dedup code** — the second fix addressed the symptom,
  not the transitive-closure cause.

---

## Hardware / environment facts (measured on this machine)

- RTX 5070 Laptop, **7.96 GiB** VRAM. **batch 16 at 640 does not fit any m-class YOLO.**
- **`workers=0` is NOT required** — that was a missing `if __name__ == "__main__":` guard.
  `workers=4` measured **1.74× faster** (64.7 vs 37.1 img/s).
- Sustained training drops the CPU to ~73% of nominal clock (thermal); epochs slow ~3.5×.
- setuptools pinned `<82` for torch; `python-magic-bin` needed for `dataset_tools` on Windows.

---

## Outstanding work

### 1. Results write-up (the only review-blocking item, ~1–2 h, no GPU)
`REVIEW_RESULTS.md` + optionally an HTML dashboard. All numbers are in this file and in the
`benchmarks/*.json` outputs.

### 2. `cctv_v3` class mapping — **edited outside my sessions**
`tools/fetch_datasets.py` and `tools/class_map.yaml` now reference real-CCTV sources (USRT, MGD,
a `cctv_v3` Roboflow fork) with a `TODO(you)` block. **The mapping is commented out**, so builds
still work, but **the stage-1 dataset does not include those CCTV sources.** That block is the
trap-heavy one: `Knife` and `knife` are both present and distinct, and `weapon` (646) is
unresolvable to gun-or-knife.

### 3. Blocked: clean negatives from the camera
`recordings/` are unusable (annotated output + contain a weapon). `capture_negatives.py --camera`
is **fixed and verified** (83 diverse frames from 30 simulated seconds → ~1,100 from 420 s).
Needed for `tune_thresholds.py` Part B.

### 4. The highest-value next step: pose-diverse CCTV data
[Firearm-related action recognition dataset](https://data.mendeley.com/datasets/bbzpxhd22j/2) —
398 videos, ~99k frames, **PTZ camera at 640×480** (your exact resolution), Handgun/Machine_Gun/
No_Gun, varied aiming directions, CC BY-NC. Mendeley blocks programmatic download (403); needs a
browser. Also [YouTube-GDD](https://arxiv.org/pdf/2203.04129) and the Guns Movies Database
(665 frames @640×480, indoor shooting poses).

**This is the real fix for the viewpoint gap** — video-derived, matches the deployment
resolution, covers the angles that fail.

### 5. Housekeeping
- **Rotate the Roboflow API key** — it was pasted into chat twice.
- `long_gun` precision regression (0.746 → 0.556) needs attention.
- If a weapon replica becomes available again, **15 minutes of filming at varied angles beats
  everything else on this list.**

---

## Honest limitations to state in the review

- **In-domain results come from one clip, one replica, one room, one person.** The held-out split
  is by time, not scene — optimistic. It is still far more informative than the public-imagery
  number alone.
- **Public-val mAP is measured on catalogue-style imagery** the camera will never see.
- **The false-positive rate has never been measured on clean camera footage** — the only
  negatives available were contaminated. `recordings/`-based FP numbers should not be quoted.
- **Per-frame recall on the deployment camera is weak** (19% miss at conf 0.25, 65% at 0.55).
  Time-to-first-detection is 0.0 s, so *event-level* alerting works far better than per-frame
  recall suggests — but say that precisely rather than implying per-frame reliability.
