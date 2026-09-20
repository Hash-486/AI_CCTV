# ============================================================
# compare_models.py -- score every candidate on the three axes that matter
# ============================================================
"""
Decide which weights go in the demo, on evidence.

A model that improves one axis by wrecking another is not progress, so every
candidate is scored on all three:

  1. IN-DOMAIN MISS RATE   held-out last 30% of the replica-pistol clip.
                           The deployment condition, and the thing being fixed.
  2. FALSE POSITIVES       the three weapon-free clips. Suppression must not
                           come back as noise.
  3. PUBLIC VAL mAP        stage-1 validation set. Catches catastrophic
                           forgetting from fine-tuning on one scene.

Promotion rule (from the plan): miss@0.25 < 19%, false positives no worse than
stage 2, public mAP50-95 >= 0.57.

Usage:
    python Review_1.2/benchmarks/compare_models.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
ROOT = REVIEW.parent

POS_CLIP = ROOT / "recordings" / "20260411_100507.avi"
NEG_CLIPS = [ROOT / "recordings" / n for n in
             ("20260228_224637.avi", "20260422_142654.avi", "20260728_191730.avi")]
DATA = ROOT / "datasets" / "weapon_stage1" / "data.yaml"
HELD_OUT = 0.70          # fine-tunes trained on the first 70% of the clip
THRESHOLDS = (0.25, 0.45, 0.55)

CANDIDATES = {
    "stage1 (public only)": REVIEW / "stage1_yolo26s_640" / "weights" / "best.pt",
    "stage2 (in-domain)":   REVIEW / "stage2_indomain" / "weights" / "best.pt",
    "runA (viewpoint aug)": REVIEW / "stage2_runA" / "weights" / "best.pt",
    "runB (aug + replay)":  REVIEW / "stage2_runB" / "weights" / "best.pt",
}


def frames_of(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def main() -> int:
    from ultralytics import YOLO

    pos = frames_of(POS_CLIP)
    held = pos[int(len(pos) * HELD_OUT):]
    neg = []
    for c in NEG_CLIPS:
        f = frames_of(c)
        neg.extend(f[::3])
    print(f"held-out positives: {len(held)} frames   negatives: {len(neg)} frames\n")

    results = {}
    for label, w in CANDIDATES.items():
        if not w.exists():
            print(f"[skip] {label}: not trained yet")
            continue
        m = YOLO(str(w))

        pos_conf = []
        for fr in held:
            r = m(fr, verbose=False, imgsz=640, conf=0.10)[0]
            pos_conf.append(float(max(r.boxes.conf).item()) if len(r.boxes) else 0.0)
        neg_conf = []
        for fr in neg:
            r = m(fr, verbose=False, imgsz=640, conf=0.10)[0]
            neg_conf.append(float(max(r.boxes.conf).item()) if len(r.boxes) else 0.0)

        v = m.val(data=str(DATA), split="val", imgsz=640, batch=8, device=0,
                  workers=4, plots=False, verbose=False)

        results[label] = {
            "weights": str(w),
            "miss": {t: round(100 * (1 - sum(1 for c in pos_conf if c >= t) / len(pos_conf)), 1)
                     for t in THRESHOLDS},
            "false_pos": {t: sum(1 for c in neg_conf if c >= t) for t in THRESHOLDS},
            "public_mAP50": round(float(v.box.map50), 4),
            "public_mAP50_95": round(float(v.box.map), 4),
        }
        print(f"[done] {label}")

    print("\n" + "=" * 92)
    print("MODEL COMPARISON")
    print("=" * 92)
    print(f"{'model':<24}" + "".join(f"{'miss@' + str(t):>10}" for t in THRESHOLDS)
          + "".join(f"{'FP@' + str(t):>9}" for t in THRESHOLDS)
          + f"{'mAP50':>9}{'mAP50-95':>10}")
    for label, r in results.items():
        print(f"{label:<24}"
              + "".join(f"{r['miss'][t]:>9.0f}%" for t in THRESHOLDS)
              + "".join(f"{r['false_pos'][t]:>9}" for t in THRESHOLDS)
              + f"{r['public_mAP50']:>9.3f}{r['public_mAP50_95']:>10.3f}")

    # apply the promotion rule
    base = results.get("stage2 (in-domain)", {})
    fp_budget = base.get("false_pos", {}).get(0.25, 10**9)
    print("\npromotion rule: miss@0.25 < 19%  |  FP@0.25 <= "
          f"{fp_budget}  |  public mAP50-95 >= 0.57")
    winner, best = None, 1e9
    for label, r in results.items():
        ok = (r["miss"][0.25] < 19.0 and r["false_pos"][0.25] <= fp_budget
              and r["public_mAP50_95"] >= 0.57)
        print(f"  {'PASS' if ok else 'fail'}  {label}")
        if ok and r["miss"][0.25] < best:
            winner, best = label, r["miss"][0.25]
    print(f"\nrecommended for the demo: {winner or 'none passed -- keep stage 2'}")

    (HERE / "model_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {HERE / 'model_comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
