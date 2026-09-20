# Weapon Detector — Evaluation Report

**Run:** `runs/detect/weapon_yolov8s` · **Measured:** 2026-08-06
**Model:** YOLOv8s · 72 layers · 11,126,358 params · 28.4 GFLOPs
**Dataset:** `Weapon 2.v2i.yolov8` (Roboflow, Weapon-2 v2, CC BY 4.0) · classes `guns`, `knife`
**Environment:** Ultralytics 8.4.108 · torch 2.11.0+cu128 · RTX 5070 Laptop (8151 MiB) · imgsz 640 · batch 16

> **Headline:** overall **mAP@50-95 = 0.689** (`last.pt`) / **0.702** (`best.pt`).
> **These scores are not trustworthy.** The validation split is contaminated by frame-level
> leakage — see [The validation split is contaminated](#the-validation-split-is-contaminated).
> Treat 0.70 as an *upper bound*, not a baseline.

Interactive version of this report (charts, hover, checkpoint toggle):
<https://claude.ai/code/artifact/e52c694c-f128-48d5-93d8-fef9ba039f34>

---

## 1. Overall and per-class results

Validation split: **259 images, 481 instances** (306 `guns`, 175 `knife`).
Precision and recall below are reported at the best-F1 confidence, as Ultralytics does.

### `last.pt` — the final checkpoint

| Class | Instances | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|
| guns | 306 | 0.9176 | 0.6911 | 0.7884 | **0.7891** | **0.5837** |
| knife | 175 | 0.9519 | 0.9943 | 0.9726 | **0.9631** | **0.7945** |
| **all classes** | 481 | 0.9347 | 0.8427 | — | **0.8761** | **0.6891** |

mAP@75 (all classes): 0.7930

### `best.pt` — the checkpoint the pipeline loads

| Class | Instances | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|
| guns | 306 | 0.9320 | 0.7059 | 0.8033 | **0.8111** | **0.6079** |
| knife | 175 | 0.9207 | 0.9947 | 0.9562 | **0.9624** | **0.7962** |
| **all classes** | 481 | 0.9263 | 0.8503 | — | **0.8867** | **0.7021** |

mAP@75 (all classes): 0.8226

### Checkpoint comparison

| Metric | `last.pt` | `best.pt` | Δ (best − last) |
|---|---:|---:|---:|
| mAP@50-95, all | 0.6891 | 0.7021 | **+0.0130** |
| mAP@50, all | 0.8761 | 0.8867 | +0.0106 |
| mAP@75, all | 0.7930 | 0.8226 | +0.0296 |
| mAP@50-95, guns | 0.5837 | 0.6079 | **+0.0242** |
| mAP@50-95, knife | 0.7945 | 0.7962 | +0.0017 |

`best.pt` wins on every aggregate metric. `config.py:10` already points at it — that is correct.

### Speed

| Stage | `last.pt` | `best.pt` |
|---|---:|---:|
| Preprocess | 0.22 ms | 0.18 ms |
| **Inference** | **5.57 ms** | **5.49 ms** |
| Postprocess | 0.78 ms | 0.75 ms |

~180 FPS raw on the RTX 5070. With `WEAPON_DETECT_INTERVAL = 3` the weapon model costs roughly
1.9 ms per pipeline frame.

---

## 2. mAP at each IoU threshold

The mAP@50-95 headline is the mean of these ten columns.

**`last.pt`**

| Class | .50 | .55 | .60 | .65 | .70 | .75 | .80 | .85 | .90 | .95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guns | 0.789 | 0.771 | 0.750 | 0.726 | 0.684 | 0.639 | 0.554 | 0.460 | 0.326 | 0.138 |
| knife | 0.963 | 0.963 | 0.963 | 0.963 | 0.963 | 0.947 | 0.907 | 0.810 | 0.423 | 0.042 |
| all | 0.876 | 0.867 | 0.856 | 0.844 | 0.823 | 0.793 | 0.731 | 0.635 | 0.375 | 0.090 |

**`best.pt`**

| Class | .50 | .55 | .60 | .65 | .70 | .75 | .80 | .85 | .90 | .95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guns | 0.811 | 0.803 | 0.790 | 0.775 | 0.744 | 0.684 | 0.594 | 0.502 | 0.323 | 0.054 |
| knife | 0.962 | 0.962 | 0.962 | 0.962 | 0.962 | 0.962 | 0.903 | 0.821 | 0.433 | 0.032 |

**Read this carefully.** `knife` AP is *perfectly flat* from IoU 0.50 to 0.70 — not a single box
falls out of tolerance as the requirement tightens. Real models do not localise that cleanly.
This is the fingerprint of a model scoring frames it has effectively already seen.

---

## 3. Deployed operating point (conf = 0.55)

mAP integrates over every confidence threshold, so it says nothing about the running system.
`config.py:21` pins `WEAPON_CONF_THRESHOLD = 0.55`. These are the numbers at that point.

### Counted outcomes

| Checkpoint | Class | Correct | Missed | Wrong class | False alarm | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `last.pt` | guns | 221 | **83** | 2 | 27 | 0.891 | 0.722 |
| `last.pt` | knife | 174 | 1 | 0 | 10 | 0.935 | 0.994 |
| `best.pt` | guns | 222 | **80** | 4 | 24 | 0.902 | 0.725 |
| `best.pt` | knife | 175 | 0 | 0 | 11 | 0.921 | 1.000 |

**At the deployed threshold the model misses roughly one gun in four (26%) and no knives.**

### Confusion matrices (rows = predicted, columns = ground truth)

`best.pt` @ conf 0.55:

|  | true guns | true knife | true background |
|---|---:|---:|---:|
| **pred guns** | 222 | 0 | 24 |
| **pred knife** | 4 | 175 | 11 |
| **pred background** (missed) | 80 | 0 | 0 |

`last.pt` @ conf 0.55:

|  | true guns | true knife | true background |
|---|---:|---:|---:|
| **pred guns** | 221 | 0 | 27 |
| **pred knife** | 2 | 174 | 10 |
| **pred background** (missed) | 83 | 1 | 0 |

The two classes are almost never confused with each other; every real error is
*missed entirely* or *invented from background*.

### Is 0.55 the right threshold?

| Checkpoint | Class | Best-F1 conf | F1 there | F1 at 0.55 |
|---|---|---:|---:|---:|
| `last.pt` | guns | 0.60 | 0.7934 | 0.7856 |
| `last.pt` | knife | 0.75 | 0.9773 | 0.9618 |
| `best.pt` | guns | 0.66 | 0.8074 | 0.8030 |
| `best.pt` | knife | 0.68 | 0.9691 | 0.9569 |

0.55 sits just below both peaks — mildly recall-leaning, which is a defensible choice for an
alerting system. No change needed.

---

## 4. The validation split is contaminated

**This is the most important finding in this report.**

All **259** validation images are frames from **4 phone videos**. Those same videos also supplied
frames to the training set. Roboflow split this dataset *frame-wise*, not *clip-wise*, so the model
trained on frames sitting milliseconds either side of the frames it is scored on. Consecutive video
frames are near-duplicates.

### Frames contributed by each source video

| Source video | Train frames | Val frames | Leaks? |
|---|---:|---:|:--:|
| whatsapp 2023-11-22 19:46:37 | 295 | 84 | **yes** |
| whatsapp 2023-11-22 19:47:53 | 953 | 13 | **yes** |
| whatsapp 2023-12-04 18:04:35 | 179 | 0 | no |
| whatsapp 2023-12-04 21:04:06 | 9 | 0 | no |
| whatsapp 2023-12-04 21:04:09 | 1 | 0 | no |
| whatsapp 2023-12-04 21:04:14 | 9 | **135** | **yes** |
| whatsapp 2023-12-04 21:04:15 | 12 | 27 | **yes** |
| whatsapp 2023-12-04 21:04:18 | 96 | 0 | no |

Four videos leak, and they account for **100% of the validation set**.

### The second problem: most training data is never validated

| Split / source | Images | guns inst. | knife inst. | Total inst. |
|---|---:|---:|---:|---:|
| Train · movie stills | 2,263 | 1,849 | 639 | 2,488 |
| Train · phone-video frames | 1,576 | 217 | 1,885 | 2,102 |
| Val · phone-video frames | 259 | 306 | 175 | 481 |
| Val · movie stills | **0** | 0 | 0 | 0 |
| **All images** | **4,098** | **2,372** | **2,699** | **5,071** |

**2,263 movie-still images carrying 1,849 of the 2,066 training gun instances are never validated
at all.** (Movie stills are IMFDB-style screenshots — filenames like `Flash_Gordon_06`,
`Flatfoot_in_Africa-Pistol-2`.)

### This explains the per-class gap

The phone videos supplied **1,885 of the 2,524 training knife instances** — and *every* validation
knife. The knife score of 0.963 is largely a memory test. The gun score of 0.789 is lower precisely
because the validation guns come from those same phone clips, while the bulk of gun *training* data
sits in movie stills that are never scored.

---

## 5. Domain gap: this dataset does not look like CCTV

Bounding-box area as a percentage of image area:

| Split / class | 10th pct | Median | 90th pct | Boxes < 1% of frame |
|---|---:|---:|---:|---:|
| train · guns | 1.22% | **25.83%** | 90.84% | 7.8% |
| train · knife | 0.62% | 1.97% | 31.70% | 24.5% |
| val · guns | 1.56% | 5.07% | 13.67% | 1.0% |
| val · knife | 0.97% | 1.85% | 3.86% | 11.4% |

The median training gun box covers **more than a quarter of the frame** — these are movie close-ups.
A weapon in real overhead CCTV footage is a handful of pixels, motion-blurred, often partly occluded.
Only 7.8% of training gun boxes are under 1% of frame area; that thin tail is the only part of this
dataset resembling the deployment condition.

Even a perfectly clean split would therefore not tell you how this model performs on your cameras.

---

## 6. Training history

50 epochs, 5.34 hours, `patience: 100` (so it never early-stopped).

| Epoch | mAP@50 | mAP@50-95 |
|---:|---:|---:|
| 20 | 0.8409 | 0.6397 |
| 30 | 0.8641 | 0.6640 |
| **40** | 0.8884 | **0.7076** ← peak |
| 46 | 0.8728 | 0.6910 |
| 48 | 0.8824 | 0.6924 |
| 50 | 0.8810 | 0.6991 |

mAP@50 climbs steeply to ~0.84 by epoch 20, then gains only +0.04 across the next 30 epochs.
mAP@50-95 peaked at epoch 40 and drifted down over the final ten — which is exactly why
`last.pt` is the worse checkpoint.

> **Note on a small discrepancy:** `results.csv` records 0.6991 for epoch 50, while re-validating
> `last.pt` standalone gives 0.6891. The gap comes from validation-time settings (fused model,
> batch shapes) differing from training-time validation. Every number elsewhere in this report is
> from the standalone re-run, so they are internally consistent.

---

## 7. Verdict and recommendations

**As a piece of training, this worked.** 0.70 mAP@50-95 from 50 epochs of YOLOv8s is a healthy
result, losses fell cleanly, and at 5.6 ms it is fast enough to run every third frame alongside pose
estimation and re-ID. **As a measurement, it is unreliable — and the direction of the error is
known: the true numbers are lower than these.**

| # | Priority | Action |
|---|---|---|
| 1 | **Critical** | **Re-split by source clip, then re-measure.** Group every frame by its source video or film and assign whole groups to train or val. Expect the numbers to fall — that drop is not a regression, it is your first honest reading. |
| 2 | **Critical** | **Build a held-out set from real CCTV frames.** A few hundred labelled frames from your own cameras, never trained on, is worth more than any improvement to this dataset. It is the only number that predicts field behaviour. |
| 3 | Warning | **Keep `best.pt` loaded; drop `patience`.** `config.py:10` is already correct. Set `patience` to ~15 so future runs stop near their peak instead of drifting for ten more epochs. |
| 4 | Normal | **Feed the gun class small objects.** `guns` trails `knife` by 0.21 mAP@50-95 and misses ~26% of instances at conf 0.55. Add distant, blurred and partly occluded gun imagery, and consider raising `imgsz` above 640 — small-object recall is the specific failure, and resolution is the cheapest lever on it. |

---

## Reproducing these numbers

```python
from ultralytics import YOLO

model = YOLO("runs/detect/weapon_yolov8s/weights/best.pt")
model.val(
    data="Weapon 2.v2i.yolov8/data.yaml",
    split="val", imgsz=640, batch=16, device=0,
    workers=0,      # REQUIRED on Windows — spawn-based DataLoader workers deadlock
    plots=True,     # REQUIRED for confusion_matrix to be populated
    conf=0.55,      # only for the operating-point matrix; omit for mAP
)
```

Two gotchas worth remembering:

- **`workers=0` is mandatory on this machine.** Without it the validation job hangs at ~3s CPU and
  never progresses. The original `args.yaml` already used it.
- **`plots=True` is required to populate `confusion_matrix`**, and `val()` defaults to `conf=0.001`,
  which floods the matrix with low-confidence noise. Pass the real threshold for a meaningful matrix.
