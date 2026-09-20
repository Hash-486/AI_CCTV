"""Compare person detectors and inference resolutions.

Answers two questions:
  1. YOLO26s vs YOLOv8s vs YOLO11s for person detection
  2. imgsz 640 vs 960 vs 1280

Timing methodology is taken from Review_1.2/benchmarks/bench_person.py,
which documents why it matters: this is a laptop GPU, and sustained load
drops the clock. A naive sequential benchmark once ranked yolo11l faster
than yolo11m, which is impossible. So:

  - every model is warmed up BEFORE any model is timed
  - rounds are interleaved across models
  - the reported figure is the MEDIAN of rounds, not the mean

Accuracy on a labelled set is only run when --data is given (an Ultralytics
data.yaml). Without it this reports speed and crop-quality only.

The resolution question is not settled by detection mAP alone. Higher
imgsz produces larger person crops, and crop height drives re-ID quality
far more than detection quality does -- an embedding from a 40px crop is
noise regardless of how well the box was placed. So this also reports the
crop-height distribution each configuration produces.

Usage:
  python benchmarks/bench_detectors.py --clips <dir>
  python benchmarks/bench_detectors.py --clips <dir> --data coco.yaml
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR  # noqa: E402

MODELS = ["yolo26s.pt", "yolo11s.pt", "yolov8s.pt"]
RESOLUTIONS = [640, 960, 1280]
ROUNDS = 5
ITERS = 40
WARMUP = 30


def load(name):
    path = os.path.join(ROOT_DIR, "models", "person", name)
    if not os.path.exists(path):
        return None
    m = YOLO(path)
    if torch.cuda.is_available():
        m.to("cuda")
    return m


def speed_sweep(models, frame):
    """Interleaved, warmed, median-of-rounds timing."""
    print("warming up every model/resolution before timing any of them")
    for name, m in models.items():
        for r in RESOLUTIONS:
            for _ in range(WARMUP // len(RESOLUTIONS) + 1):
                m.predict(frame, imgsz=r, classes=[0], verbose=False,
                          half=torch.cuda.is_available(), device=0)
    torch.cuda.synchronize()

    rounds = {(n, r): [] for n in models for r in RESOLUTIONS}
    for rd in range(ROUNDS):
        for name, m in models.items():
            for res in RESOLUTIONS:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(ITERS):
                    m.predict(frame, imgsz=res, classes=[0], verbose=False,
                              half=torch.cuda.is_available(), device=0)
                torch.cuda.synchronize()
                rounds[(name, res)].append(
                    (time.perf_counter() - t) / ITERS * 1000.0
                )
        print(f"  round {rd + 1}/{ROUNDS} done")
    return {k: statistics.median(v) for k, v in rounds.items()}


def crop_stats(models, clips, stride=5):
    """Distribution of detected person crop heights per configuration.

    Crop height is the variable that actually limits re-ID quality.
    """
    out = {}
    for name, m in models.items():
        for res in RESOLUTIONS:
            heights, ndet = [], 0
            for path in clips:
                cap = cv2.VideoCapture(path)
                idx = 0
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    if idx % stride == 0:
                        r = m.predict(f, imgsz=res, classes=[0], conf=0.3,
                                      verbose=False,
                                      half=torch.cuda.is_available(), device=0)
                        if r and r[0].boxes is not None and len(r[0].boxes):
                            xy = r[0].boxes.xyxy.cpu().numpy()
                            heights.extend((xy[:, 3] - xy[:, 1]).tolist())
                            ndet += len(xy)
                    idx += 1
                cap.release()
            if heights:
                h = np.array(heights)
                out[(name, res)] = {
                    "detections": ndet,
                    "median_crop_h": round(float(np.median(h)), 1),
                    "p10_crop_h": round(float(np.percentile(h, 10)), 1),
                    "frac_usable_64px": round(float((h >= 64).mean()), 4),
                }
            else:
                out[(name, res)] = {"detections": 0}
    return out


def accuracy(models, data_yaml):
    out = {}
    for name, m in models.items():
        for res in RESOLUTIONS:
            try:
                r = m.val(data=data_yaml, classes=[0], imgsz=res, batch=8,
                          device=0, workers=4, verbose=False, plots=False)
                out[(name, res)] = {
                    "mAP50": round(float(r.box.map50), 4),
                    "mAP50_95": round(float(r.box.map), 4),
                    "precision": round(float(r.box.mp), 4),
                    "recall": round(float(r.box.mr), 4),
                }
            except Exception as exc:
                out[(name, res)] = {"error": str(exc)[:120]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--data", default=None,
                    help="Ultralytics data.yaml for accuracy evaluation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = {}
    for n in MODELS:
        m = load(n)
        if m is None:
            print(f"  missing: {n}")
        else:
            models[n] = m
    if not models:
        raise SystemExit("no models found")

    paths = []
    for c in args.clips:
        if os.path.isdir(c):
            for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
                paths.extend(sorted(glob.glob(os.path.join(c, ext))))
        else:
            paths.append(c)

    frame = None
    if paths:
        cap = cv2.VideoCapture(paths[0])
        ok, frame = cap.read()
        cap.release()
    if frame is None:
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    print(f"frame {frame.shape[1]}x{frame.shape[0]}   "
          f"device {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    speeds = speed_sweep(models, frame)

    print("\nspeed (median of rounds, ms/frame and FPS)")
    header = "  " + "model".ljust(12) + "".join(f"{r:>16d}" for r in RESOLUTIONS)
    print(header)
    for name in models:
        row = "  " + name.replace(".pt", "").ljust(12)
        for res in RESOLUTIONS:
            ms = speeds[(name, res)]
            row += f"{ms:8.2f}ms/{1000 / ms:5.0f}"
        print(row)

    crops = {}
    if paths:
        print("\ncollecting crop-height statistics ...")
        crops = crop_stats(models, paths)
        print("\ncrop quality (what re-ID actually sees)")
        print("  " + "model".ljust(12) + "res".rjust(6)
              + "median_h".rjust(10) + "p10_h".rjust(8)
              + "usable>=64px".rjust(14))
        for name in models:
            for res in RESOLUTIONS:
                c = crops.get((name, res), {})
                if not c.get("detections"):
                    continue
                print(f"  {name.replace('.pt', ''):12s}{res:6d}"
                      f"{c['median_crop_h']:10.1f}{c['p10_crop_h']:8.1f}"
                      f"{c['frac_usable_64px'] * 100:13.1f}%")

    acc = {}
    if args.data:
        print("\nrunning accuracy evaluation (slow) ...")
        acc = accuracy(models, args.data)
        print("\naccuracy")
        for name in models:
            for res in RESOLUTIONS:
                a = acc.get((name, res), {})
                if "error" in a:
                    print(f"  {name:12s} {res:5d}  ERROR {a['error']}")
                else:
                    print(f"  {name:12s} {res:5d}  mAP50 {a['mAP50']:.4f}  "
                          f"mAP50-95 {a['mAP50_95']:.4f}  "
                          f"P {a['precision']:.4f}  R {a['recall']:.4f}")

    result = {
        "speed_ms": {f"{n}@{r}": round(v, 3) for (n, r), v in speeds.items()},
        "crops": {f"{n}@{r}": v for (n, r), v in crops.items()},
        "accuracy": {f"{n}@{r}": v for (n, r), v in acc.items()},
    }
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "detector_comparison.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
