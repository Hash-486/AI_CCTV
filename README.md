# AI_CCTV

Real-time person tracking and weapon detection for a single camera feed
(webcam or video file). Built around YOLO26s detection, DeepSORT + OSNet
re-identification, MediaPipe pose, and a two-stage weapon-detection
verification pipeline with per-person threat-level analysis.

## What it does

- **Person tracking with persistent identity** — YOLO26s detects people,
  DeepSORT associates them frame-to-frame, and an OSNet re-identification
  gallery recognises the same person after they leave and re-enter frame.
- **Weapon detection** — a custom-trained YOLO26s model (guns / knife /
  long_gun) runs alongside person detection. Candidates pass through a
  three-gate verification stage (per-class confidence, temporal
  persistence, and association with a tracked person) before being
  treated as confirmed.
- **Threat-level analysis** — confirmed weapons are matched to the
  person holding them (wrist proximity, containment, or IoU) and drive a
  per-person threat level (SAFE / CAUTION / HIGH / CRITICAL) with
  hysteresis, so a level escalates instantly but decays gradually.

## Setup

```bash
pip install -r requirements.txt
# torch/torchvision installed separately with the CUDA index matching your GPU
```

Model weights are not tracked in this repo (see `.gitignore`) — place a
person model at `models/person/yolo26s.pt` and a weapon model at
`models/weapon/stage2_indomain/best.pt`, or point `config.py` at your own
checkpoints. Dataset build tooling lives in `tools/` (`fetch_datasets.py`,
`build_dataset.py`, `class_map.yaml`).

## Running

```bash
python main.py                       # webcam
python main.py --video path/to.avi   # replay a clip
python main.py --save out.avi        # write annotated output
python main.py --no-weapon           # person tracking only
python main.py --no-reid             # raw track IDs, no identity gallery
python main.py --no-pose             # disable MediaPipe
python main.py --headless --max-frames 500   # for benchmarking
```

`q` quits, `r` resets identities, `p` toggles pose, `t` toggles motion trails.

## Layout

| path | what |
|---|---|
| `main.py` | live pipeline entry point |
| `config.py` | every tunable threshold, one file |
| `detector.py`, `tracker.py`, `gallery.py`, `embedder.py`, `pose.py`, `render.py` | person detection, tracking, re-ID, and drawing |
| `weapon/` | weapon detection (`detector.py`) and two-stage verification (`verification.py`) |
| `threat/` | weapon-to-person association and threat-level analysis |
| `tools/` | dataset fetching, deduplication, and build scripts |
| `benchmarks/`, `eval/` | model comparisons and in-domain evaluation |
| `docs/`, `paper/related_work/` | write-ups, dataset survey notes, and results |

## Status

Deployed weapon model: `guns`/`knife`/`long_gun`, 28,943 training images
merged from 8 public sources with zero cross-split leakage, fine-tuned on
in-domain deployment-camera footage. See `docs/PROGRESS.md` for the full
evaluation history, known limitations, and outstanding work — including
the viewpoint gap (public weapon datasets are side-profile product shots;
the model still misses foreshortened/edge-on weapons) that real-CCTV
data is being pulled in to address.
