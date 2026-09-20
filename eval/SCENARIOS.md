# Recording protocol

Calibration on the existing `recordings/*.avi` produced this:

| false-merge budget | T_MATCH | re-entry recall |
|---|---|---|
| 1% | 0.8139 | **8.3%** |
| 10% | 0.6863 | 32.0% |
| 35% | 0.5633 | 77.9% |

There is no usable operating point. Same-person crops taken 30+ frames apart
score 0.6511 on average; different-person crops score 0.5370. A gap of 0.114
is not enough to separate them, and no threshold, gallery design, or margin
test can recover information the embeddings do not contain.

**This is a data problem, not a code problem.** Three things about the
existing clips cause it, and all three are fixed by recording properly:

1. **Annotation is burnt into every frame.** `recordings/*.avi` are the old
   pipeline's *annotated output* — green skeletons, boxes, ID labels and a
   status bar are drawn into the pixels. Every crop fed to the re-ID model
   contains drawn graphics, so the model partly compares line drawings.
2. **Almost no distinct identities.** Three of four clips contain one person.
   Inter-person pairs came from a single clip, and only 85 of them are
   same-frame (the only trustworthy negatives).
3. **No true re-entries.** Nobody leaves and returns, so the case the whole
   system exists to handle is never exercised.

Record raw camera frames — not pipeline output.

---

## What to record

Roughly 15 minutes of filming. Save as raw capture at 640×480, the camera's
native resolution, with **no overlay drawing of any kind**.

Use `--save-raw` on the capture tool, or point the camera at the scene and
record with any plain capture app. If you record through this pipeline, pass
`--headless` and save the *input* frames, not the rendered ones.

### Clip 1 — short re-entry (60 s)
One person. Walk out of frame, wait **5 seconds**, walk back in. Repeat 5–6
times. Vary which edge you exit and re-enter from; at least twice, exit left
and return right.

### Clip 2 — medium re-entry (90 s)
Same person. Exit, wait **30 seconds**, return. Repeat 3 times. This is the
case the 600 s gallery TTL is designed for.

### Clip 3 — long re-entry (5 min)
Same person. Exit, wait **2 minutes**, return. Twice. Change something
plausible between exits — pick up a bag, put on or remove a jacket. Note in
the ground truth which returns involved a change.

### Clip 4 — occlusion without leaving (60 s)
Walk behind a pillar, door frame, or another person, and come out the other
side. 6–8 passes. Vary how long you are hidden: roughly 0.5 s, 1 s, 2 s, 4 s.

### Clip 5 — two people crossing (90 s)
Two people walk toward each other and pass, repeatedly, 8–10 crossings. Vary
who is nearer the camera. This is what produces ID-swap failures.

### Clip 6 — similar clothing (90 s)
**The hardest case, and the most valuable clip.** Two people in similar
colour clothing (both dark tops, or both light) walking, crossing, and
swapping positions. If MobileNet vs OSNet differ anywhere, they differ here.

### Clip 7 — four or more people (2 min)
Four to six people moving naturally, entering and leaving. This is the only
clip that produces enough same-frame negative pairs for calibration to be
statistically meaningful — the current data has 85, and several thousand
are wanted.

### Clip 8 — distance sweep (60 s)
One person walking slowly from close to the camera to as far as the space
allows, and back. Establishes the crop height at which embeddings stop being
usable, which sets `ADMIT_MIN_CROP_H` on evidence instead of the current
guessed 64 px.

---

## Ground truth format

Annotation is **per event**, not per frame — a handful of lines per clip,
minutes of work rather than hours.

Write `eval/ground_truth/<clip_name>.json`:

```json
{
  "clip": "clip1_short_reentry.avi",
  "fps": 30,
  "people": {
    "A": "person in blue shirt",
    "B": "person in grey hoodie"
  },
  "events": [
    {"person": "A", "exit_frame": 120, "return_frame": 270, "note": ""},
    {"person": "A", "exit_frame": 410, "return_frame": 580,
     "note": "returned from opposite side"},
    {"person": "A", "exit_frame": 700, "return_frame": 1150,
     "note": "put on jacket"}
  ],
  "crossings": [
    {"people": ["A", "B"], "frame": 300}
  ]
}
```

Only `exit_frame` and `return_frame` are required. Scrub the video, note the
frame where the person is fully gone and the frame where they are fully back.
Approximate is fine — `eval_reid.py` searches a window around each.

---

## After recording

```bash
# 1. Recalibrate on the new footage
python benchmarks/calibrate_thresholds.py --clips eval/clips

# 2. Copy the recommended values into config.py

# 3. Re-run the embedder comparison, now with enough negative pairs
python benchmarks/bench_embedders.py --clips eval/clips

# 4. Synthetic occlusion curve (no annotation needed)
python eval/occlusion_inject.py --clip eval/clips/clip4_occlusion.avi \
    --sweep 5,10,20,30,60,120,300

# 5. Real re-identification accuracy
python eval/eval_reid.py --clips eval/clips --gt eval/ground_truth
```

Every table in `RESULTS.md` regenerates from these five commands.

---

## What good separability looks like

The single number to watch after recalibrating is the long-gap intra mean
versus the inter mean:

| gap | verdict |
|---|---|
| < 0.15 | unusable — currently **0.114** |
| 0.15–0.25 | marginal; expect frequent splitting |
| 0.25–0.35 | workable at a sane operating point |
| > 0.35 | good |

If clean footage still gives under 0.15, the problem is the camera or scene
(resolution, lighting, viewing angle) rather than the source video, and the
next lever is camera placement — people should occupy more vertical pixels.
