# Model Selection — Why YOLO26s @ 960

**Decided:** 2026-08-07 · **Hardware:** RTX 5070 Laptop (7.96 GiB VRAM), 16 physical / 24 logical cores
**Stack:** Ultralytics 8.4.108 · torch 2.11.0+cu128 · CUDA 12.8

> **Verdict: train `yolo26s` at `imgsz=960`, `batch=8`, `workers=4`, `lr0=0.005`, `patience=15`.**
> Control run: `yolo26m` at `imgsz=640`, `batch=8`.

Every number below marked *measured* was benchmarked on this machine. Numbers marked
*published* come from the Ultralytics model docs. Nothing here is estimated from memory.

---

## 1. The three constraints that decide it

Model choice for this project is not a free accuracy contest. Three hard limits shape it:

1. **7.96 GiB VRAM**, of which ~6.7 GiB is usable for training once Windows takes its share.
   This eliminates the standard `batch=16` for every m-class model.
2. **The camera is 640×480** (`config.py:16`). Every weapon in every deployment frame is
   drawn from that much detail, and no model choice changes it.
3. **The weapon model does not run alone.** `main.py` runs person detection every frame,
   DeepSORT + OSNet re-ID every frame, MediaPipe pose every 4th frame, and the weapon
   model every 3rd frame (`WEAPON_DETECT_INTERVAL = 3`). The weapon model gets a slice of
   the frame budget, not the whole thing.

The project's actual failure is **small-object recall on the `guns` class** — 26% of gun
instances missed at the deployed threshold, with only 7.8% of training gun boxes exceeding
1% of frame area. That failure is what the model choice has to attack.

---

## 2. The verdict, and why

`yolo26s @ 960` wins because it is the only configuration that improves the thing that is
actually broken **without paying for it anywhere else**:

| axis | `yolo26s @ 960` | vs the obvious alternative |
|---|---|---|
| Small-object capacity | 2.25× pixel density vs 640 | `yolo26m @ 640` has 1.0× |
| Training cost | 46.6 GFLOPs/img | `yolo26m @ 640` costs **68.2** — more expensive |
| Deploy speed | **130 FPS** | *faster* than the same model at 640 (122 FPS) |
| VRAM | 5.2 GiB at batch 8 | fits with headroom |
| Architecture | STAL small-target label assignment | YOLO11 has none |

The counterintuitive result is that **going from 640 to 960 is nearly free, and going from
`s` to `m` is not.** Raising resolution costs less compute than raising model scale, and at
batch 1 inference it costs nothing at all.

### Why resolution is free at inference

Measured, fused fp16, batch 1:

| model | 640 | 960 | 1280 |
|---|---:|---:|---:|
| yolo11s | 6.23 ms / 161 FPS | **6.05 ms / 165 FPS** | 7.59 ms / 132 FPS |
| yolo26s | 8.22 ms / 122 FPS | **7.69 ms / 130 FPS** | 8.60 ms / 116 FPS |
| yolo11m | 7.87 ms / 127 FPS | 10.69 ms / 94 FPS | — |
| yolo26m | 8.90 ms / 112 FPS | 11.06 ms / 90 FPS | — |
| yolo26l | 13.01 ms / 77 FPS | — | — |
| yolov8s *(current)* | 4.73 ms / 212 FPS | — | — |

At batch 1 the small models are **kernel-launch-bound, not compute-bound** — the GPU sits
idle waiting for the CPU to dispatch work, so the extra pixels ride along in slack time. For
`yolo26s`, 960 is measurably *faster* than 640. This only holds for s-class; the m-class
models are large enough to be genuinely compute-bound, and there 960 does cost ~25%.

### Why STAL matters here specifically

YOLO26 introduces **STAL (Small-Target-Aware Label Assignment)**, which maintains positive
label coverage for small objects during training — objects that older assignment schemes
starve of positive samples. It also drops NMS (end-to-end inference) and DFL. For a project
whose measured failure is small-object recall, this is not a generic version bump; it targets
the exact defect.

---

## 3. The elimination round

### YOLOv8s — the incumbent · REJECTED
44.9 COCO mAP, the weakest of every candidate. Its only advantage is speed (212 FPS), which
is irrelevant: at `WEAPON_DETECT_INTERVAL = 3` on a 30 FPS camera, anything above ~90 FPS is
already free. Trading 3.7 mAP for headroom nobody uses is a bad deal.

### YOLO11m — the obvious upgrade · REJECTED
This was the initial candidate, and it loses to `yolo26m` on identical terms: **51.5 vs 53.1
COCO mAP at the same 20 M params and 68 GFLOPs.** There is no axis on which YOLO11m beats
YOLO26m. It also lacks STAL. Its one merit is maturity — see *When this decision changes*.

### YOLO12m — REJECTED
20.2 M params, 68.1 GFLOPs — the same class as YOLO11m and YOLO26m, but it measured the
**worst VRAM of any m-class model** (10.57 GiB at batch 16 vs 8.77 for yolo26m) and offers no
accuracy advantage over YOLO26m. Strictly dominated.

### YOLO26l — REJECTED
55.0 mAP is genuinely better, but it costs 77 FPS at 640 and only fits at batch 8. It has to
share the GPU with the person detector on *every* frame plus re-ID and pose. The accuracy gain
does not survive the frame-budget cost, and at batch 8 / 960 it needs 6.1 GiB — uncomfortably
close to the ceiling.

### YOLO26x / YOLO11x — REJECTED
55.7 M params, ~194 GFLOPs. Will not train on 8 GiB at any useful batch size.

### YOLO26n / YOLO11n — REJECTED
40.9 and 39.5 mAP — worse than the incumbent YOLOv8s. Nano exists for edge devices without a
discrete GPU. This machine has one.

### YOLO26m @ 640 — RUNNER-UP, kept as the control
53.1 mAP, 112 FPS, fits at batch 8 (4.59 GiB). The only reason it is not the primary pick is
that it spends its extra capacity on parameters rather than pixels, which is the wrong trade
for small objects — **and it costs more compute doing it** (68.2 vs 46.6 GFLOPs/img).

That reasoning is an argument, not a measurement. Train both and let the data decide; the two
runs together cost about two hours.

### YOLO26s @ 1280 — REJECTED (but reconsider if the camera is upgraded)
4.0× pixel density, still 116 FPS. Rejected because batch drops to 4 (noisy gradients, slow
wall-clock at 25.8 img/s) and because **the source is 640×480** — at 1280 the network is
looking at 4× upscaled pixels containing no additional information. The finer feature grid
still helps somewhat, but with sharply diminishing returns.

---

## 4. Measured data

### Training VRAM, peak GiB at imgsz 640

| model | params | b8 | b16 | b24 | b32 |
|---|---:|---:|---:|---:|---:|
| yolov8s | 11.2 M | 1.74 | 3.32 | 4.90 | 6.48 |
| yolo11s | 9.5 M | 2.01 | 3.83 | 5.64 | 7.46 |
| yolo26s | 10.0 M | 2.37 | 4.54 | 6.69 | 8.85 |
| yolo11m | 20.1 M | 4.06 | 7.75 ✗ | 11.45 ✗ | 15.14 ✗ |
| yolo26m | 20.4 M | 4.59 | 8.77 ✗ | 12.98 ✗ | 17.17 ✗ |
| yolo11l | 25.3 M | 4.91 | 9.79 ✗ | 14.45 ✗ | 19.12 ✗ |
| yolo26l | 24.8 M | 5.41 | 10.76 ✗ | 15.90 ✗ | 21.06 ✗ |
| yolo12m | 20.2 M | 5.46 | 10.57 ✗ | 15.67 ✗ | 20.76 ✗ |

✗ = exceeds the 7.96 GiB card. **The Ultralytics default `batch=16` at 640 OOMs on every
m-class model.** Copying a tutorial command will fail on this machine.

### Largest batch that fits, model × resolution (~6.7 GiB budget)

| model | 640 | 960 | 1280 |
|---|---|---|---|
| yolo11s | b24 (5.6 G) | b12 (6.4 G) | b4 (4.0 G) |
| yolo26s | b24 (6.6 G) | **b8 (5.2 G)** | b4 (4.8 G) |
| yolo11m | b12 (5.9 G) | b4 (4.5 G) | b2 (4.1 G) |
| yolo26m | b12 (6.6 G) | b4 (5.1 G) | b2 (4.7 G) |
| yolo26l | b8 (5.4 G) | b4 (6.1 G) | b2 (5.6 G) |

`b12` fits for m-class at 640 but lands at 6.6 of 6.7 GiB — too tight once the EMA copy,
dataloader pinning and the validation pass are added. **Use b8 for m-class.**

### Training throughput, GPU-bound (synthetic, no dataloader)

| config | img/s | GFLOPs/img |
|---|---:|---:|
| yolov8s @640 b16 | 136.7 | 28.6 |
| yolo11s @640 b16 | 130.8 | 21.5 |
| yolo26s @640 b16 | 113.3 | 20.7 |
| yolo11m @640 b8 | 59.3 | 68.0 |
| yolo11s @960 b8 | 57.0 | 48.4 |
| yolo26m @640 b8 | 52.7 | 68.2 |
| **yolo26s @960 b8** | **48.7** | **46.6** |
| yolo26l @640 b8 | 42.1 | 86.4 |
| yolo26s @1280 b4 | 25.8 | 82.8 |
| yolo26m @960 b4 | 22.7 | 153.5 |

### Published accuracy (Ultralytics docs, COCO val)

| model | mAP50-95 | params | FLOPs |
|---|---:|---:|---:|
| YOLO11n | 39.5 | 2.6 M | 6.5 B |
| YOLO26n | 40.9 | 2.4 M | 5.4 B |
| YOLOv8s | 44.9 | 11.2 M | 28.6 B |
| YOLO11s | 47.0 | 9.4 M | 21.5 B |
| **YOLO26s** | **48.6** | 9.5 M | 20.7 B |
| YOLOv8m | 50.2 | 25.9 M | 78.9 B |
| YOLO11m | 51.5 | 20.1 M | 68.0 B |
| **YOLO26m** | **53.1** | 20.4 M | 68.2 B |
| YOLO11l | 53.4 | 25.3 M | 86.9 B |
| YOLO26l | 55.0 | 24.8 M | 86.4 B |
| YOLO11x | 54.7 | 56.9 M | 194.9 B |
| YOLO26x | 57.5 | 55.7 M | 193.9 B |

**COCO mAP is a proxy, not a promise.** It ranks general 80-class detection; this project is
2-class weapon detection on a different domain. The ranking is a strong prior, not a result.

---

## 5. The setting that matters more than the model

Measured on yolov8s @640 b16, 3,839 images, one epoch:

| workers | epoch time | rate |
|---|---:|---:|
| 0 | 103.5 s | 37.1 img/s |
| **4** | **59.4 s** | **64.7 img/s (1.74×)** |
| 8 | 64.9 s | 59.1 img/s |

**`workers=0` is not required on this machine.** The original run used it, apparently after
hitting a DataLoader hang — but that hang is caused by a **missing `if __name__ == "__main__":`
guard**. Windows *spawns* worker processes, which re-import the main module; without the guard
that re-executes the whole script recursively and wedges. It is a script bug, not a platform
limit.

Every training and validation script must have the guard. Then use `workers=4`.

`workers=8` losing to `workers=4` may be an artifact of a single-epoch test failing to
amortise spawn cost across 24 logical cores — worth retesting over a longer run.

For reference, the GPU-bound ceiling for that config is 136.7 img/s, so even `workers=4` leaves
the GPU roughly half idle. Augmentation is CPU-heavy; `cache='ram'` is the next lever if the
dataset fits in memory.

---

## 6. Deployment frame budget

Per frame, at the recommended configuration:

| component | cadence | cost/frame |
|---|---|---:|
| Person detector (yolov8s @640) | every frame | 4.73 ms |
| Weapon detector (yolo26s @960) | every 3rd frame | 2.56 ms amortised |
| MediaPipe pose | every 4th frame | not measured |
| OSNet re-ID + DeepSORT | every frame | not measured |

The two YOLO models together consume ~7.3 ms/frame, a ~137 FPS ceiling. The camera runs at
640×480, and the recordings in `recordings/` are 20 FPS. **The models are nowhere near the
bottleneck at deploy time** — which is precisely why spending frame budget on accuracy is the
right trade, and why the incumbent YOLOv8s's 212 FPS buys nothing.

Pose and re-ID costs are unmeasured and could change this. Profile the full loop before
assuming headroom.

---

## 7. Caveats and when this decision changes

- **YOLO26 is new.** Export paths (ONNX, TensorRT) and third-party tooling are less
  battle-tested than YOLO11's. `onnxruntime` 1.28.0 is installed here; TensorRT and OpenVINO
  are not. **If you need to deploy to other hardware via ONNX/TensorRT, verify the export path
  early** — that is the one scenario where YOLO11m's maturity beats YOLO26m's 1.6 mAP.
- **NMS-free changes threshold semantics.** YOLO26 is end-to-end, so `iou` and
  `agnostic_nms` may become no-ops. `config.py` defines `YOLO_SOFT_NMS` and `detection.py`
  passes `iou` — re-check both. The project's own `_WEAPON_TEMPORAL_IOU` dedup in
  `detection.py` is independent and unaffected.
- **Re-tune `WEAPON_CONF_THRESHOLD`** (`config.py:21`, currently 0.55) against the new model's
  F1 curve. Do not carry the old value over.
- **If the camera is upgraded** to 1080p or higher, revisit this entire document. The
  640×480 source is the binding constraint on small-object detection, and `yolo26s @ 1280`
  becomes genuinely attractive once there is real detail to resolve.
- **If a second GPU or a desktop card becomes available**, YOLO26l and YOLO26x re-enter
  contention; they were eliminated on VRAM and frame budget, not on merit.

---

## 8. Commands

```python
# train_weapon.py -- the main guard is mandatory, see section 5
from ultralytics import YOLO

if __name__ == "__main__":
    YOLO("yolo26s.pt").train(
        data="datasets/<your-dataset>/data.yaml",
        imgsz=960,
        batch=8,
        workers=4,        # NOT 0 -- see section 5
        lr0=0.005,        # defaults assume batch 16; halve for batch 8
        epochs=150,
        patience=15,      # the old run used 100 and drifted past its peak
        seed=0,
        name="weapon_yolo26s_960",
    )
```

Control run: same file with `YOLO("yolo26m.pt")`, `imgsz=640`, `batch=8`.

Compare both on the same held-out set before committing. The argument in section 2 is
reasoning about compute and pixel density; only the two runs settle it.
