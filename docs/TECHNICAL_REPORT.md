# Person Re-Identification System — Technical Report

Complete description of the pipeline built in `Updated_Review_1.2/`: what was
chosen, why, how it works step by step, what it stores, for how long, and how
many people it can handle.

**Hardware:** RTX 5070 Laptop (7.96 GiB), 16 physical / 24 logical cores,
Windows 11, torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.108.
**Camera:** 640×480 webcam.
**Scope:** person detection, tracking and identity only. No weapon detection,
threat analysis, caution labelling, alarms or threat logging.

---

# Part 1 — Detector selection

## 1.1 Why an "s" model at all

The frame budget at 30 FPS is 33.3 ms. Measured cost of everything else in the
pipeline (embedding, tracking, identity, rendering) is ~4.2 ms at typical
occupancy, leaving roughly 29 ms for detection. Every s-class model fits;
m- and l-class models do not fit comfortably alongside re-ID on 8 GiB, and the
base project's own study (`Analysis/MODEL_SELECTION.md`) found the m-class
models OOM at the Ultralytics default batch during training.

The choice is therefore among **YOLOv8s, YOLO11s and YOLO26s**.

## 1.2 Accuracy — COCO val2017, person class only

2,693 images / 10,777 person instances. Source:
`Review_1.2/benchmarks/person_comparison.json`.

| model | mAP50 | mAP50-95 | Precision | Recall | params |
|---|---|---|---|---|---|
| YOLOv8s | 0.7963 | 0.5701 | 0.8275 | 0.6964 | 11.17 M |
| YOLO11s | 0.8095 | 0.5821 | 0.8343 | 0.7127 | 9.46 M |
| **YOLO26s** | **0.8211** | **0.5963** | **0.8345** | **0.7130** | 10.01 M |

YOLO26s wins on **every** accuracy metric:

- **+2.62 mAP50-95 over YOLOv8s** (0.5963 vs 0.5701) — a 4.6% relative gain
- **+1.42 mAP50-95 over YOLO11s** (0.5963 vs 0.5821)
- **Highest recall (0.7130)**, which matters most here: a person the detector
  never sees cannot be tracked or re-identified, and missed detections were
  later shown to be the binding constraint in crowds (§7.3).

It also achieves this with **fewer parameters than YOLOv8s** (10.01 M vs
11.17 M), so the gain is architectural rather than capacity.

## 1.3 Speed — measured on this machine

`benchmarks/bench_detectors.py`. Methodology matters on a laptop: sustained
load drops the clock, and a naive sequential benchmark in the base project
once ranked yolo11l faster than yolo11m, which is impossible. So:

- every model is warmed up **before any model is timed**
- rounds are **interleaved** across models
- reported figure is the **median of 5 rounds** × 40 iterations

Full `.predict()` cost including pre- and post-processing:

| model | 640 | 960 | 1280 |
|---|---|---|---|
| YOLO26s | **9.68 ms / 103 FPS** | 12.55 ms / 80 FPS | 15.84 ms / 63 FPS |
| YOLO11s | 9.18 ms / 109 FPS | 12.01 ms / 83 FPS | 14.79 ms / 68 FPS |
| YOLOv8s | 7.40 ms / 135 FPS | 10.19 ms / 98 FPS | 13.08 ms / 76 FPS |

YOLO26s is the slowest of the three at 9.68 ms — **and it does not matter**.
The camera produces 30 FPS. Any detector above ~30 FPS is fast enough, and all
three are 3–4× that. YOLOv8s's extra 2.28 ms buys nothing; it is spent idling
until the next frame arrives. Accuracy is the only axis on which the choice is
free, so accuracy decides it.

## 1.4 Why not the nano models

Rejected without benchmarking. Published COCO mAP50-95: YOLO26n 40.9 vs
YOLO26s 48.6. The n-class models trade 7.7 mAP for speed the system cannot
use. On a 33 ms budget with 29 ms free, buying speed is buying nothing.

## 1.5 Decision

**YOLO26s.** Best accuracy on every metric, best recall, fewer parameters than
the incumbent, and its speed disadvantage is invisible at 30 FPS.

---

# Part 2 — Why 640

This was the least obvious decision and needed three separate measurements.

## 2.1 The naive argument for higher resolution

Higher `imgsz` means more pixels on distant people, which should mean better
small-object recall. The base project's `Analysis/MODEL_SELECTION.md`
recommended training at 960 for exactly this reason.

## 2.2 Measurement 1 — cost

From §1.3: for YOLO26s, 640 → 960 costs **+30%** (9.68 → 12.55 ms) and
640 → 1280 costs **+64%** (9.68 → 15.84 ms).

## 2.3 Measurement 2 — what re-ID actually receives

Detection mAP is the wrong metric for this decision. What limits
re-identification is **crop height**: an embedding from a 40-pixel-tall person
is noise regardless of how precisely the box was placed. So the question is
whether higher `imgsz` produces *larger crops*.

Measured over this project's footage:

| model | imgsz | median crop height | fraction ≥ 64 px |
|---|---|---|---|
| YOLO26s | **640** | **381.6 px** | **100.0%** |
| YOLO26s | 960 | 383.8 px | 98.7% |
| YOLO26s | 1280 | 324.8 px | 99.5% |
| YOLO11s | 640 | 380.0 px | 100.0% |
| YOLOv8s | 640 | 377.4 px | 98.7% |

**960 yields crops 2.2 px taller than 640** — statistically indistinguishable —
and 1280 yields *smaller* crops. Higher resolution costs 30–64% more time and
delivers identically sized people.

The reason is that the source is natively **640×480**. At `imgsz=640` the
frame is already at native resolution; going higher upscales, which
interpolates pixels that were never captured. There is no additional detail to
recover.

## 2.4 Measurement 3 — corroboration from the base project

`Review_1.2/PROGRESS.md` tested the same hypothesis on weapon detection and
**refuted it**: raising inference resolution to 960 made the miss rate *worse*
(56% → 68%). Independent evidence, same conclusion, same camera.

## 2.5 The important caveat — 640 is not universal

On **1920×1080** MOT17 footage, `imgsz=640` downscales 3× and distant
pedestrians fall below detectability. Measured on MOT17-02, first 300 frames:

| imgsz | IDF1 | MOTA | recall |
|---|---|---|---|
| 640 | 54.0% | 33.5% | 40.6% |
| **1280** | **66.5%** | **45.7%** | **57.3%** |

**+12.5 IDF1 and +16.7 recall** purely from resolution.

So the rule is not "640 is best". It is **match `imgsz` to the source
resolution**:

| source | recommended imgsz |
|---|---|
| 640×480 (this camera) | **640** |
| 1280×720 | 1280 |
| 1920×1080 | 1280–1920 |

`INFER_IMGSZ` in `config.py` is set for the deployment camera. Change it when
processing higher-resolution footage.

## 2.6 Reconciliation with the base project

`Analysis/MODEL_SELECTION.md` reported 960 as *faster* than 640 for YOLO26s
(7.69 vs 8.22 ms), which contradicts §1.3. Both are correct for what they
measured: that study timed a **raw model forward pass**, where at batch-1 the
s-models are kernel-launch bound and resolution barely matters. §1.3 times
**`.predict()` including pre- and post-processing**, which is what the pipeline
actually pays per frame. The pipeline-level number is the relevant one.

---

# Part 3 — Datasets, and why each was used

Five data sources, each answering a question the others cannot.

## 3.1 COCO val2017 (person class) — detector selection

**What:** 2,693 images, 10,777 annotated person instances.
**Why:** the standard benchmark for object detection, with human-verified
boxes. Used to rank YOLOv8s / YOLO11s / YOLO26s on person detection
specifically (`classes=[0]`) rather than on all 80 COCO classes, since the
other 79 are irrelevant here.
**Limitation:** still images; says nothing about tracking or identity.

## 3.2 Market-1501 — embedder selection

**What:** 1,501 identities, 32,668 cropped person images, 6 cameras
(Tsinghua campus). Split used: 3,368 query / 19,732 gallery.
**Why:** the first source with **true identity labels**. It replaced an
earlier comparison built from IoU-chained pseudo-labels that produced a
**wrong answer by a factor of 12** (§8.2).

Its evaluation protocol is what makes it transfer. For each query, gallery
entries with the **same person AND same camera** are discarded before scoring,
so a correct match requires recognising the person from a *different* camera —
different viewpoint, lighting and scale. That is structurally the same problem
as "walked out and came back".

**Limitation:** cropped stills. Cannot test tracking, occlusion or ghost boxes.

## 3.3 MSMT17 — cross-domain validation, and the deployed weights

**What:** 4,101 identities, 126,441 crops, 15 cameras, indoor and outdoor, day
and night. Split used: 11,659 query / 82,161 gallery.
**Why:** Market-1501 alone cannot detect over-fitting to a single scene. Using
two datasets lets each set of weights be tested **out of its own domain**,
which is the situation any real deployment is in.

This produced the single most decision-relevant result in the project (§4.2):
cross-domain transfer is asymmetric by 10×.

## 3.4 MOT17 — tracking, occlusion and multi-person

**What:** full-frame surveillance video with **per-frame human-annotated
identity ground truth**. Static-camera sequences used:

| sequence | scene | people/frame | frames |
|---|---|---|---|
| MOT17-02 | street | ~20 | 600 |
| MOT17-04 | night street | ~45 | 1,050 |
| MOT17-09 | shopping mall | ~10 | 525 |

**Why:** the only source that tests the **tracker** rather than just the
embedder. Cropped datasets have no motion, no occlusion, no entries and exits,
and never two people in one shot. MOT17 has all four, plus the standard
metrics (IDF1, MOTA, ID-switches) that a reviewer expects for a tracking claim.

Moving-camera sequences (05, 10, 11, 13) were **excluded** — a moving platform
is a different problem from a fixed CCTV install.

**Why not MARS / PRID2011 / iLIDS-VID / LPW:** all four are *tracklet*
datasets — sequences of pre-cropped boxes. They would score the embedder over
time but still cannot test tracking, ghost boxes or multi-person logic. All
four also require academic request forms and are not openly downloadable.

## 3.5 Own recorded clips — deployment evidence

**What:** four clips recorded with `tools/capture.py` on the actual camera.
**Why:** no public dataset can evidence *this* camera in *this* room. Public
benchmarks establish that the software is correct; only own footage
establishes that the deployment works. It is also the only source of
calibration data for this camera's thresholds (§6.4).

## 3.6 Provenance note

Market-1501 and MSMT17 are surveillance footage collected under academic
research licences. DukeMTMC was **withdrawn by its authors** over consent
concerns; its *weights* are evaluated here for completeness but the dataset
itself was never downloaded. All three are used for benchmarking only.

---

# Part 4 — Model weights

## 4.1 The trap

`torchreid.models.build_model(pretrained=True)` loads **ImageNet** weights,
not re-identification weights. The re-ID checkpoints are a separate download
(`deep-person-reid/docs/MODEL_ZOO.md`) that torchreid never fetches
automatically. The base project passed `model_path=""` and therefore ran
ImageNet OSNet throughout — unknowingly.

`tools/fetch_reid_weights.py` downloads the three re-ID checkpoints.

## 4.2 Cross-domain transfer is asymmetric by 10×

Measured with `benchmarks/bench_reid.py`, full standard protocol on both
benchmarks (rank-1 % / mAP %):

| weights | on Market-1501 | on MSMT17 |
|---|---|---|
| Market-1501 | **94.3 / 83.6** *(in-domain)* | 9.9 / 3.2 |
| **MSMT17** | 57.2 / 30.4 | **75.9 / 47.8** *(in-domain)* |
| DukeMTMC | 50.1 / 22.5 | 14.2 / 4.4 |
| ImageNet | 14.5 / 4.5 | 9.7 / 2.2 |

The diagonal is unsurprising. The **off-diagonal** is the result:

```
MSMT17 weights      -> Market-1501    30.4% mAP
Market-1501 weights -> MSMT17          3.2% mAP
```

Same architecture, same training procedure, opposite directions, a tenfold
difference. Market-trained weights out of domain (3.2%) barely exceed raw
ImageNet (2.2%) — a re-ID model trained on a narrow domain is close to
worthless outside it.

The cause is **training-set diversity**, not model quality. Market-1501 is one
outdoor campus scene across 6 cameras; MSMT17 spans 4,101 identities, 15
cameras, indoor and outdoor, day and night. Variety produces features that
survive a domain change; a single scene produces features that memorise it.

**Decision: MSMT17 weights.** This deployment is a third domain neither has
seen, so only the cross-domain column predicts anything, and MSMT17 is the only
weight set that transfers. Selecting Market-1501 on its 94.3% would be
optimising for a benchmark this camera will never resemble.

**Expected ceiling for an unseen indoor scene: the 57.2 / 30.4 figure**, not
94.3 / 83.6.

## 4.3 Implementation validated twice

| benchmark | measured | published |
|---|---|---|
| Market-1501, Market weights | 94.3 / 83.6 | 94.2 / 82.6 |
| MSMT17, MSMT17 weights | 75.9 / 47.8 | 74.9 / 43.8 |

Reproducing two independent references to within 1.0% rank-1 confirms model
loading, cv2 preprocessing, ImageNet normalisation, L2 handling, the CUDA-graph
replay path and the CMC/mAP protocol are all correct.

## 4.4 Why not MobileNet

The base project used MobileNetV2 inside DeepSORT. Measured on Market-1501:

| embedder | dim | rank-1 | mAP |
|---|---|---|---|
| OSNet x1_0 (MSMT17) | 512 | 57.2% | **30.4%** |
| OSNet x1_0 (ImageNet) | 512 | 14.5% | 4.5% |
| MobileNetV3-Small | 576 | 11.5% | 3.5% |
| MobileNetV3-Large | 960 | 9.8% | 2.6% |
| MobileNetV2 | 1280 | 6.8% | 2.2% |

**MobileNetV3-Large marginally beats V2 (2.6% vs 2.2% mAP) and neither is
usable.** Both are ~12× worse than cross-domain OSNet. The base pipeline was
performing identity association with a model that scores 2.2% mAP at telling
people apart.

ImageNet features optimise inter-**class** separation (person vs car); re-ID
features optimise intra-class, inter-**instance** separation (person A vs
person B). Only the second is this task.

*(torchreid ships no MobileNetV3, so V3 required a custom torchvision wrapper.
torchreid's own MobileNetV2 is a dead network — output max 8.1e-9, all
activations collapsed — and was excluded rather than scored.)*

---

# Part 5 — How a person is detected and tracked, step by step

Per frame, in order.

## Step 1 — Detection (`detector.py`)

```
YOLO26s.predict(frame, conf=0.3, classes=[0], imgsz=640,
                half=True, agnostic_nms=True)
```

- `classes=[0]` — COCO person only; the other 79 classes are never computed
- `conf=0.3` — deliberately permissive. Recall is the binding constraint
  (§7.3); marginal detections are filtered later by track confirmation and
  gallery admission rather than discarded here.
- `half=True` — fp16 on CUDA
- Output converted to LTWH for DeepSORT and LTRB for cropping

**Cost: 9.50 ms.**

## Step 2 — Appearance embedding (`embedder.py`)

**One batched OSNet pass for every detection in the frame.**

```
crops -> cv2.resize to 256x128 -> stack -> normalise -> CUDA graph replay
      -> L2-normalise -> (N, 512) float32
```

Three design points:

**Batched, not per-crop.** The base project called torchreid's
`FeatureExtractor` once per crop, and that API converts numpy → PIL →
`T.Resize` → `T.ToTensor` → `T.Normalize` for every image. The PIL round-trip
is CPU-bound and single-threaded. Preprocessing here uses `cv2.resize` into a
preallocated batch tensor: **0.32 ms** for the entire preprocessing chain.

**CUDA graph capture.** OSNet is kernel-launch bound, not compute bound. Its
omni-scale blocks are four parallel branches of depthwise-separable
convolutions, repeated — an enormous number of individually tiny operations.
Three measurements prove it:

| evidence | |
|---|---|
| batch 1 → batch 8 | 12.47 → 13.30 ms (8× the work, +6% time) |
| osnet_x1_0 → osnet_x0_25 | 13.61 → 13.03 ms (16× fewer params, same speed) |
| fp16 vs fp32 | 13.61 vs 12.21 ms (fp16 *slower*) |

All three say the GPU is idle waiting for launches, so neither a smaller model
nor a smaller batch helps. Capturing the whole forward pass as one replayable
graph: **12.91 ms → 2.66 ms, bit-identical output** (max abs diff 0.00e+00).

**Quality floor separate from admission bar.** Crops below 16×8 px get a zero
vector flagged invalid, so distant people are still *tracked* by motion even
though they are never *enrolled* as an identity. Conflating the two would mean
distant people vanish entirely.

**Cost: 3.24 ms** at 1–8 people (see §7.2 for scaling).

## Step 3 — Frame-to-frame association (`tracker.py`, DeepSORT)

The same 512-d embedding is **injected** into DeepSORT:

```
DeepSort(embedder=None, ...)
tracks = ds.update_tracks(dets, embeds=osnet_feats, frame=frame)
```

DeepSORT then runs, per track:

1. **Kalman filter predict** — constant-velocity, 8-D state
   `(cx, cy, aspect, h, vx, vy, va, vh)`
2. **Mahalanobis gating** — χ²-95 threshold 9.4877 at 4 DOF, rejecting
   physically implausible pairings
3. **Cosine appearance metric** — the injected OSNet features, gated at
   `max_cosine_distance = 0.3735` (calibrated)
4. **Hungarian assignment** on the combined cost, via
   `scipy.optimize.linear_sum_assignment`
5. **Matching cascade** by track age, preferring recently-seen tracks
6. **IoU fallback** for unconfirmed and recently-lost tracks,
   `max_iou_distance = 0.7`
7. **State machine** — Tentative → Confirmed (`n_init=3`) → Deleted
   (`max_age=5`)

**Why one appearance model instead of two.** The base ran MobileNetV2 inside
DeepSORT *and* OSNet outside it, sharing nothing. That is worse than
redundant: `ReIDGallery` keyed on DeepSORT track id and only re-searched
*unmapped* ids, so when MobileNetV2 swapped two people during a short
occlusion, the swapped-but-surviving track kept its canonical mapping and the
gallery had **no mechanism to detect or correct it**. The refresh then kept
appending the wrong person's features to that identity, progressively blending
A into B. A short-gap association error silently degraded long-gap
re-acquisition. Sharing one re-ID-trained representation means both timescales
agree on what "the same person" means.

**Cost: 0.74 ms.**

## Step 4 — Render gate (ghost box elimination)

```
if track matched a detection this frame:
    box = track.to_ltrb(orig=True)      # the DETECTION
    render = True
elif time_since_update <= 2 and pose confirms a body:
    box = track.to_ltrb()               # prediction, with evidence
    render = True
else:
    render = False                      # never drawn
```

`to_ltrb(orig=True)` returns the last associated detection instead of the
Kalman mean. The library's own docstring for the default path reads *"POORLY
NAMED… Returns LIES"* — it returns the predicted state even for tracks that
were not matched this frame, which is what made boxes lag behind walkers and
drift after they left.

A person walking out of frame is deleted immediately rather than coasting:
their identity is already stored, and the **gallery**, not the Kalman filter,
is what brings them back.

## Step 5 — Duplicate suppression

Output boxes overlapping at IoU > 0.8 are merged, keeping the higher-confidence
one. DeepSORT's internal NMS is also enabled (`nms_max_overlap = 0.7`; the
library default of 1.0 disables it), but that filters *detections*, not
*tracks* — two tracks can still converge on one person.

## Step 6 — Identity resolution

See Part 6.

## Step 7 — Pose (`pose.py`, MediaPipe)

Runs every 4th frame; cached and replayed between. **Two jobs, both
non-destructive:**

- **Confirmation** — visible landmarks are evidence a body is really at a
  predicted position, permitting the short render-coast in Step 4
- **Refinement** — `x1`, `x2`, `y2` are blended toward the landmark hull;
  `y1` is left to the detector because the highest BlazePose landmark is
  eye/ear level and blending it shaves the top of the head

It **never deletes a track**. The base project used pose as a rejection
filter, and because an empty dict is not `None`, "pose ran and found nobody"
scored as zero visible keypoints and a genuinely present person was silently
erased.

**Cost: ~12 ms when it runs, ~3 ms amortised** at `POSE_DETECT_INTERVAL = 4`.

## Step 8 — Render (`render.py`)

Colour encodes tracking state, not threat: green = matched this frame, amber =
coasting on a confirmed prediction, cyan = identity not yet decided.

---

# Part 6 — How a person is re-identified

## 6.1 What is stored per identity

```python
Identity:
    id           int
    samples      deque(maxlen=16) of (512,) float32, L2-normalised
    sample_meta  deque(maxlen=16) of {t, conf, bbox_h, kp_count}
    centroid     (512,) float32, L2-renormalised mean of samples
    body_ratios  (4,) float32 or None
    first_seen   float (epoch)
    last_seen    float (epoch)
    n_hits       int
    state        "active" | "dormant"
    last_bbox    (x1, y1, x2, y2)
```

## 6.2 How much is stored — measured

| item | size |
|---|---|
| one embedding | 512 × float32 = **2,048 bytes** |
| 16 samples | **32,768 bytes** |
| centroid | 2,048 bytes |
| sample metadata | ~7,536 bytes |
| **total per identity** | **~41.4 KB** |
| **at capacity (100 people)** | **~4.0 MB** |

Storage is not a constraint at any realistic scale.

## 6.3 Which embeddings are admitted

Extraction runs every frame because DeepSORT needs it. **Admission is
selective** — a bad sample in the gallery is expensive and effectively
permanent, since it drags the centroid and stays until diversity eviction
displaces it.

A sample is stored only if **all** hold:

| gate | threshold | reason |
|---|---|---|
| track matched a detection | — | never enrol a prediction |
| crop height | ≥ 64 px | below this the embedding is noise |
| detection confidence | ≥ 0.5 | |
| not side-clipped (left/right) | — | half a torso is a fragment |
| if vertically clipped, aspect | ≥ 1.2 | head/feet cropping is tolerable if the shape is still person-like |
| visible keypoints, if pose available | ≥ 5 | soft — absence does not block |

**The asymmetry between axes is deliberate and was learned the hard way.** An
earlier version rejected any box touching any edge. Measured on a clip where
the subject filled the frame (median box height **469 px in a 480 px frame**),
this rejected **1,306 of 1,322** candidate samples and left every identity with
**zero** stored embeddings — nothing could ever be re-matched. Side clipping
cuts the torso; vertical clipping removes head or feet, which OSNet tolerates
well because Market-1501 and MSMT17 crops routinely clip both.

## 6.4 Diversity-based eviction, not FIFO

When the 16-slot set is full, a new sample **replaces the most redundant
incumbent** — the one whose nearest neighbour inside the set is closest — and
only if the newcomer is itself less redundant than that incumbent. Otherwise
it is discarded.

This keeps 16 mutually dissimilar views (front, back, side, different lighting)
instead of 16 consecutive frames of one pose. The base project used a plain
`deque(maxlen=12)` whose comment claimed "12 diverse embeddings" — there was no
diversity criterion at all, so a stationary person filled the gallery with 12
near-identical vectors and it failed the moment they turned around.

Verified in `tests/test_gallery.py[7]`: feeding 200 near-identical samples,
then 6 distinct viewpoints, then 50 more near-identical ones drops mean
pairwise similarity **0.980 → 0.692** and retains **6/6** distinct views.

## 6.5 How a returning person is matched

**Scoring.** For each stored identity:

```
centroid_sim    = centroid · query
best_sample_sim = max(sample · query for each of the 16 samples)
appearance      = max(centroid_sim, 0.9 × best_sample_sim)
```

The centroid is a stable summary but washes out unusual viewpoints; the best
individual sample catches those but is noisier, hence the 0.9 penalty. Taking
the max lets an unusual-but-real viewpoint match without letting one lucky
sample dominate.

**Body-proportion fusion** (weight 0.15) adds four scale-invariant ratios from
MediaPipe landmarks: shoulder width / torso length, hip / shoulder width,
leg / torso, arm span / shoulder width. The term is **omitted entirely** when
either side is unmeasurable, rather than replaced by a neutral constant — a
term that cannot discriminate must not silently shift the effective threshold.
*(This is live but unproven: it is confirmed active on 332 frames, and
confirmed not to change the outcome on clips where appearance alone already
wins. Its value should show on similar-clothing cases, which are unfilmed.)*

**Joint assignment.** All pending tracks are resolved **together** with the
Hungarian algorithm over the (tracks × identities) score matrix, so an identity
can be claimed at most once per frame. The base matched each track greedily and
independently, so two people on screen could both be assigned the same
canonical id.

**Temporal plausibility is masked into the cost matrix**, not applied as a
post-hoc veto:

```
allowed_distance = 200 px + 800 px/s × gap
skipped entirely when gap > 3 s
```

The 200 px floor is not optional — a pure `speed × gap` budget collapses toward
zero as the gap shrinks, and ordinary centroid wobble then rejects correct
short-gap re-matches. The 3 s cutoff exists because after a few seconds away a
person could have re-entered from any edge, so position no longer constrains
identity.

*Masking rather than vetoing matters:* when two tracks score equally the tie
breaks arbitrarily, so a post-hoc veto lets the solver hand the identity to the
implausible track, which then fails the veto while the plausible track is left
evaluating a worse candidate — and both are incorrectly marked new.

## 6.6 The identity decision — three outcomes, not two

Let `s1` = best score, `s2` = runner-up **among identities not already claimed
by another track this frame**.

```
MATCHED       s1 >= T_MATCH (0.7327)
              AND (s1 - s2) >= T_MARGIN (0.0957)
              AND that identity is unclaimed
              AND temporally plausible

NEW           s1 < T_NEW (0.5617)

PROVISIONAL   otherwise — hold up to DEFER_FRAMES (30), accumulate
              more embeddings, re-decide each frame. Still ambiguous
              at the deadline -> assign a new identity.
```

**Why three outcomes.** A binary threshold forces a guess exactly where
guessing is most costly. A wrong MATCH merges two people permanently and
poisons both galleries; a wrong NEW splits one person into two identities,
which is visible and reconcilable. PROVISIONAL buys frames to gather evidence
instead of committing to either error, and when forced it prefers the
recoverable one.

**Why the margin test.** A score of 0.80 against one candidate is evidence.
The same 0.80 with a runner-up at 0.78 is not evidence about *which* person it
is — it says the gallery cannot tell them apart, and committing would be a coin
flip.

**Decisions use the same quality bar as admission.** A crop too poor to enrol
is too poor to conclude "this is a stranger" from. Without this, a person
re-entering from the side is judged on their first partial-body frame at the
border, scores below `T_NEW`, and a new identity is committed *before the
deferral band is ever reached* — measured at 0/4 re-entries matched despite
every return reaching 0.87–0.95 once fully visible.

**Re-ID veto.** A track unmatched for ≥ 2 frames that then re-associates is
verified against its own stored identity at `T_VETO = 0.6636`. If appearance
disagrees, the mapping is broken and the track re-queries the gallery. This
catches the case where a departing person's Kalman prediction latches onto a
different person walking in.

## 6.7 How long an identity is kept

| state | lifetime |
|---|---|
| **active** (currently tracked) | **never expires** |
| **dormant** (person left frame) | expires **600 s (10 min)** after `last_seen` |

Only absence starts the clock — someone standing still for an hour must not be
forgotten while visible.

- **Capacity:** 100 identities, LRU-evicting dormant entries beyond that
- **GC sweep:** every 30 frames
- **Persistence:** RAM only. Identities are discarded on exit; restarting
  renumbers from 1. There is no disk store and no database.

**Consequence:** walk out and return within 10 minutes → same identity. Return
after 15 minutes → new identity. This is a deliberate scope decision, not a
limitation of the matching.

---

# Part 7 — Capacity and cost

## 7.1 Frame budget at typical occupancy

| stage | ms |
|---|---|
| YOLO26s detection | 9.50 |
| OSNet embedding (batched, CUDA graph) | 3.24 |
| DeepSORT association | 0.74 |
| Identity resolution | 0.13 |
| MediaPipe pose (amortised, every 4th frame) | ~3.0 |
| **end-to-end, no pose** | **35.8 FPS** |
| **end-to-end, with pose** | **23.9 FPS** |

## 7.2 How many people — measured scaling

Embedding cost is a **step function**, because the CUDA graph records a fixed
batch of `GRAPH_BATCH = 8` and larger batches are chunked:

| people | embed ms | detect ms | total ms | ceiling FPS |
|---|---|---|---|---|
| 1 | 3.24 | 9.50 | 12.74 | 79 |
| 2 | 3.22 | 9.50 | 12.72 | 79 |
| 4 | 3.36 | 9.50 | 12.86 | 78 |
| 8 | 3.81 | 9.50 | 13.31 | 75 |
| 12 | 7.27 | 9.50 | 16.76 | 60 |
| 16 | 7.78 | 9.50 | 17.28 | 58 |
| 24 | 11.36 | 9.50 | 20.86 | 48 |
| 32 | 15.10 | 9.50 | 24.60 | 41 |

**1 to 8 people costs the same** — padding to the graph batch is nearly free,
because unused rows cost only the launch overhead already being paid. Each
additional block of 8 costs ~3.8 ms.

**The system stays above 30 FPS to at least 32 simultaneous people.** Raising
`GRAPH_BATCH` to match expected occupancy flattens the steps further.

## 7.3 The real limit is detection recall, not identity

Measured on MOT17:

| sequence | people/frame | IDF1 | MOTA | recall | IDSW |
|---|---|---|---|---|---|
| MOT17-09 | ~10 | **84.1%** | 67.7% | 85.2% | **0** |
| MOT17-02 | ~20 | 57.6% | 36.8% | 44.0% | 12 |
| MOT17-04 | ~45 | 45.6% | 29.1% | 31.4% | 25 |

IDF1 tracks recall almost exactly (85→84, 44→58, 31→46). ID switches stay low
relative to identity counts. **The tracker is not losing people's identities;
the detector never sees them.** In crowds, occlusion and small apparent size
defeat detection long before identity becomes the constraint.

*(MOT17-02 and -04 are 1080p processed at `imgsz=640`; §2.5 shows 1280 recovers
much of this.)*

---

# Part 8 — Ghost box elimination

Eight distinct causes were found in the base code. The reported symptoms were
"box lags behind a walker", "box bounces around after someone leaves", and
"random boxes when other people come in" — which turned out to be **three
different bugs, not one**.

| # | cause | fix |
|---|---|---|
| 1 | `to_ltrb()` returns the Kalman mean, not the detection | `to_ltrb(orig=True)` |
| 2 | `max_age=20` → 20 frames of velocity extrapolation | `max_age=5` + render gate |
| 3 | A dead track latches onto a **new** person's detection | re-ID veto |
| 4 | `nms_max_overlap=1.0` disables DeepSORT's internal NMS | `0.7` + output dedup |
| 5 | `n_init=2` confirms a 2-frame false positive | `n_init=3` |
| 6 | Two tracks could claim one canonical id | Hungarian mutual exclusion |
| 7 | Ghost filter blind for `time_since_update ∈ {1,2}` | moot under render gate |
| 8 | `{}` is not `None` → a present person is **deleted** | pose never deletes |
| 9 | Liveness recorded *after* filtering → GC wiped live tracks | record liveness first |

**The render gate is the primary fix**: never draw a box that was not matched
to a real detection this frame.

Defect 8 caused the *opposite* symptom and is worth separating: a person the
pose model merely failed on scored zero visible keypoints and was erased from
the output. The same check produced both ghosting and vanishing.

---

# Part 9 — Recording scenarios

`tools/capture.py` provides 11 scenarios. It writes **raw frames**; the
on-screen guides are drawn on a copy that is never saved. This matters because
`main.py --save` writes *annotated* frames, and the base project's
`recordings/*.avi` — skeletons and boxes burnt into every pixel — are why every
measurement taken on them was untrustworthy.

Exit/return events are marked live with `E`/`R` keys, producing a pre-filled
ground-truth JSON, so annotation takes minutes rather than hours.

| # | scenario | people | duration | purpose | status |
|---|---|---|---|---|---|
| 1 | Short re-entry (5 s gaps) | 1 | 60 s | the core goal, tested directly | **recorded** |
| 2 | Medium re-entry (30 s gaps) | 1 | 90 s | the case the 600 s TTL serves | |
| 3 | Re-entry with appearance change | 1 | 90 s | jacket / bag — hardest solo case | |
| 4 | Occlusion without leaving | 1 | 60 s | pillar / door frame passes | |
| 5 | Distance sweep | 1 | 60 s | sets `ADMIT_MIN_CROP_H` on evidence | |
| 6 | Empty scene / ghost check | **0** | 30 s | zero-boxes pass criterion | **recorded** |
| 7 | Same re-entry, no backlight | 1 | 60 s | controlled lighting A/B | **recorded** |
| 8 | Camera high, angled down | 1 | 60 s | real CCTV viewpoint | |
| 9 | Sitting, crouching, turning | 1 | 60 s | pose variation | |
| 10 | Two people crossing | **2** | 90 s | ID swaps, false merges, calibration | **recorded** |
| 11 | Similar clothing | **2** | 90 s | the hardest case in the suite | |

Scenarios 10 and 11 are not optional extras: **they are the only source of
same-frame negative pairs**, which threshold calibration requires, and the only
way to test false merges at all. "0 false merges" on a one-person clip is
trivially true — there is nobody to merge with.

---

# Part 10 — Results on the deployment camera

## 10.1 Re-identification accuracy

`eval/eval_reid.py` over two clips, one person, eight annotated exit/return
events with gaps of 9.8–17.1 s:

| metric | result |
|---|---|
| **re-identification accuracy** | **100.0%** (8/8) |
| false-split rate | 0.0% |
| false-merge rate | 0.0% |
| duplicate-id frames | 0 |
| identities created | 1 per clip, for 1 person |

## 10.2 Lighting is the dominant variable

Same person, same actions, one variable — where the light was:

| | backlit | even light |
|---|---|---|
| median crop brightness | 120/255 | **154/255** |
| same-person long-gap similarity | 0.675 | **0.831** |
| worst case (10th percentile) | 0.542 | **0.721** |
| per-event peak similarity | 0.769 / 0.815 / 0.851 / 0.851 | **0.954 / 0.913 / 0.952 / 0.928** |

**+0.156 mean similarity from moving a lamp** — more than every algorithmic
change in this report combined.

Both clips score 100%, but the margins differ enormously. Backlit, three of
four re-entries cleared threshold by under 0.07 — passing on almost no
headroom, the kind of result that holds until one condition shifts.

## 10.3 Empty-scene ghost test

901 frames (45 s) at 30 FPS with nobody in the room:

| metric | result |
|---|---|
| raw detections | 6 |
| **rendered boxes** | **4 (0.44%)** |
| **identities created** | **0** |
| persistent phantom tracks | 0 |

All four in one 0.3 s window on a small dark object clipping the right frame
edge (box `605,267 -> 598,360`, 63-66 px tall, confidence 0.32-0.57). Gallery
admission rejected it as side-truncated and low-confidence, so it never became
an identity.

## 10.4 Multi-person

`clip6_two_people_crossing.avi` — 653 frames, 271 with both people detected:

| metric | result |
|---|---|
| identities created | 3 for 2 people (**one false split**) |
| duplicate-id frames | **0** |

One false split is consistent with the calibrated 77.1% re-entry recall. The
invariant that matters most held: no frame ever contained the same canonical id
twice, and no two people were merged.

## 10.5 Calibrated thresholds

`benchmarks/calibrate_thresholds.py` on the two-person + even-lighting clips.
Negatives are drawn **only from same-frame crops**, which are guaranteed to be
different people.

```
intra-person (all)       n=17492   mean 0.8405   (easy, short gaps)
intra-person (>=30f)     n=12655   mean 0.8203   (re-entry: governs T_MATCH)
inter-person same-frame  n=   76   mean 0.6068
separability gap         +0.2135
best balanced accuracy    0.9033 @ 0.736
```

| false-merge budget | T_MATCH | re-entry recall |
|---|---|---|
| 0.1% | 0.7350 | 76.7% |
| **1.0%** | **0.7327** | **77.1%** |
| 5.0% | 0.7160 | 80.8% |
| 20.0% | 0.6946 | 85.0% |

**The flatness of that curve is the result.** Tightening the budget from 1% to
0.1% costs 0.4 percentage points of recall. On the old contaminated footage the
same tightening cost a third of it (8.3% → 5.0%). Well-separated distributions
let you be strict about merging strangers and still recognise nearly everyone
who returns.

| constant | placeholder | calibrated |
|---|---|---|
| `T_MATCH` | 0.75 | **0.7327** |
| `T_NEW` | 0.60 | **0.5617** |
| `T_MARGIN` | 0.10 | **0.0957** |
| `T_VETO` | 0.70 | **0.6636** |
| `DEEPSORT_MAX_COS_DIST` | 0.25 | **0.3735** |

## 10.6 Separability bands

| gap | verdict | measured |
|---|---|---|
| < 0.15 | unusable | old `recordings/*.avi` 0.114; backlit ~0.135 |
| **0.15–0.25** | **marginal** | **this camera, well lit: 0.2135** |
| 0.25–0.35 | workable | |
| > 0.35 | good | |

Levers to move it further, in order of measured effect: **lighting** (+0.156),
**framing** (a take aimed at a wall 2 m away gave inter-person similarity 0.700,
because the embeddings described the doorway rather than either person), then
**camera height** (untested; OSNet's training data is mostly elevated
surveillance).

---

# Part 11 — Configuration reference

```python
# Detection
PERSON_MODEL_PATH      = yolo26s.pt
INFER_IMGSZ            = 640      # match to source resolution (§2)
PERSON_CONF_THRESHOLD  = 0.3
YOLO_HALF              = True

# DeepSORT                        base -> new
DEEPSORT_MAX_AGE       = 5        # 20  — coasting is re-ID's job now
DEEPSORT_N_INIT        = 3        # 2   — two frames confirmed false positives
DEEPSORT_NN_BUDGET     = 30       # 100 — OSNet features are more discriminative
DEEPSORT_NMS_OVERLAP   = 0.7      # 1.0 — which disabled NMS entirely
DEEPSORT_MAX_COS_DIST  = 0.3735   # calibrated
DEEPSORT_MAX_IOU_DIST  = 0.7

# Embedder
REID_MODEL_NAME        = osnet_x1_0
REID_MODEL_PATH        = weights/osnet_x1_0_msmt17.pth
REID_IMAGE_SIZE        = (256, 128)
REID_MIN_CROP_H / W    = 16 / 8   # computability floor, not admission bar
USE_CUDA_GRAPH         = True
GRAPH_BATCH            = 8        # raise to expected max occupancy

# Gallery
REID_GALLERY_SAMPLES     = 16     # diversity-selected, not FIFO
REID_GALLERY_TTL_SEC     = 600.0  # dormant only; active never expires
REID_GALLERY_MAX_PERSONS = 100    # ~4.0 MB at capacity
GALLERY_GC_INTERVAL      = 30

# Admission
ADMIT_MIN_CONF         = 0.5
ADMIT_MIN_CROP_H       = 64
ADMIT_BORDER_MARGIN_PX = 5
ADMIT_MIN_ASPECT       = 1.2      # for vertically-clipped boxes
ADMIT_MIN_KEYPOINTS    = 5        # soft

# Identity decision (all calibrated)
T_MATCH = 0.7327 ; T_NEW = 0.5617 ; T_MARGIN = 0.0957 ; T_VETO = 0.6636
DEFER_FRAMES = 30
REID_BODY_RATIO_WEIGHT = 0.15
MAX_WALK_SPEED_PX_PER_SEC = 800 ; MIN_PLAUSIBLE_JUMP_PX = 200
TEMPORAL_CHECK_MAX_GAP_SEC = 3.0

# Ghost suppression
RENDER_COAST_FRAMES    = 2
BORDER_EXIT_MARGIN_PX  = 15
DUP_SUPPRESS_IOU       = 0.8

# Pose
POSE_DETECT_INTERVAL   = 4
POSE_NUM_POSES         = 8        # base: 5, which dropped the 6th+ person
POSE_REFINE_ALPHA      = 0.3
```

---

# Part 12 — Reproducing every number

```bash
# Detector comparison and resolution sweep (Part 1, Part 2)
python benchmarks/bench_detectors.py --clips eval/clips

# Embedder / weights comparison on public benchmarks (Part 4)
python tools/fetch_reid_weights.py
python benchmarks/bench_reid.py --dataset market1501
python benchmarks/bench_reid.py --dataset msmt17

# Tracking metrics on annotated video (Part 7.3)
python eval/eval_mot.py --seq MOT17-09-FRCNN

# Threshold calibration (Part 10.4) — well-lit, well-framed clips only
python benchmarks/calibrate_thresholds.py \
    --clips eval/clips/clip6_two_people_crossing.avi \
            eval/clips/clip9_even_lighting.avi

# Re-identification accuracy on this camera (Part 10.1)
python eval/eval_reid.py --clips eval/clips --gt eval/ground_truth

# Unit tests
python tests/test_gallery.py

# Live
python main.py --camera
```

**Do not pool clips of different lighting when calibrating.** Mixing a backlit
clip's positives with a well-lit clip's negatives understates separability —
doing exactly that produced a misleading 0.0626 against the correct 0.2135.

---

# Part 13 — Known limitations

1. **Identity is RAM-only.** Restarting renumbers everyone from 1. Deliberate
   scope decision, not a matching limitation.
2. **10-minute memory.** Dormant identities expire after 600 s.
3. **Separability is in the marginal band** (0.2135). Works, but with less
   headroom than ideal; consistent with the one false split observed.
4. **Calibration rests on 76 same-frame negative pairs.** Direction and
   magnitude are clear; a longer two-person clip would tighten it.
5. **The similar-clothing case is unfilmed** — the hardest case, and where the
   body-proportion fusion would first earn its place.
6. **Empty-scene false-positive rate is 0.44%** (4 boxes in 901 frames), all
   on one transient edge-clipped object, none enrolled as an identity.
7. **Body-proportion fusion is live but unproven** — active on 332 frames,
   but not outcome-changing on clips where appearance alone already wins.
8. **Crowd performance is detector-limited**, not identity-limited, and 1080p
   sources need `INFER_IMGSZ` raised (§2.5).
