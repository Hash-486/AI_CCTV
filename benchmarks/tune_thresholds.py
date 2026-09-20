# ============================================================
# tune_thresholds.py -- the false-negative / false-positive trade-off
# ============================================================
"""
Quantify the two-stage design instead of asserting it works.

Part A -- FALSE NEGATIVES. Sweep the detector's confidence over the stage-1
validation set and report per-class recall. This is what lowering the threshold
buys: every point of recall gained is a weapon that would otherwise have been
missed.

Part B -- FALSE POSITIVES. Replay real weapon-free footage from the deployment
camera through the actual detector + verifier, with verification ON and OFF, and
count alerts. This is what the verification stage costs those false positives.

Together they justify the operating point: detect wide, verify hard.

Usage:
    python Review_1.2/benchmarks/tune_thresholds.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
ROOT = REVIEW.parent
for p in (str(REVIEW / "pipeline"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

WEIGHTS = REVIEW / "stage1_yolo26s_640" / "weights" / "best.pt"
DATA = ROOT / "datasets" / "weapon_stage1" / "data.yaml"
NEG = REVIEW / "negatives" / "images"
SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def part_a_false_negatives() -> dict:
    """Per-class recall vs confidence, from the val-set P/R curves."""
    from ultralytics import YOLO

    print("\n" + "=" * 74)
    print("PART A -- FALSE NEGATIVES: recall vs detector confidence")
    print("=" * 74)
    r = YOLO(str(WEIGHTS)).val(data=str(DATA), split="val", imgsz=640, batch=8,
                               device=0, workers=4, plots=True, verbose=False,
                               project=str(HERE), name="tune_val", exist_ok=True)
    b = r.box
    names = [r.names[int(c)] for c in b.ap_class_index]
    px = np.asarray(b.px)

    out = {"conf": SWEEP, "recall": {}, "precision": {}}
    print(f"\n{'conf':>6}" + "".join(f"{n:>22}" for n in names))
    print(f"{'':>6}" + "".join(f"{'recall':>11}{'prec':>11}" for _ in names))
    for c in SWEEP:
        k = int(np.abs(px - c).argmin())
        row = f"{c:>6.2f}"
        for i, n in enumerate(names):
            rec = float(b.r_curve[i][k])
            pre = float(b.p_curve[i][k])
            out["recall"].setdefault(n, []).append(round(rec, 4))
            out["precision"].setdefault(n, []).append(round(pre, 4))
            row += f"{rec:>11.3f}{pre:>11.3f}"
        print(row)

    print("\nmissed instances at each threshold (lower = fewer false negatives):")
    inst = {"guns": 2204, "knife": 1788, "long_gun": 279}
    print(f"{'conf':>6}" + "".join(f"{n:>12}" for n in names) + f"{'TOTAL':>10}")
    for j, c in enumerate(SWEEP):
        miss = {n: int(round(inst[n] * (1 - out["recall"][n][j]))) for n in names}
        print(f"{c:>6.2f}" + "".join(f"{miss[n]:>12}" for n in names)
              + f"{sum(miss.values()):>10}")
    out["missed"] = {n: [int(round(inst[n] * (1 - v))) for v in out["recall"][n]]
                     for n in names}
    return out


def part_b_false_positives() -> dict:
    """Alerts raised on weapon-free footage, verification ON vs OFF."""
    from detectors import WeaponDetector
    from verification import TwoStageVerifier
    import review_config as rc

    print("\n" + "=" * 74)
    print("PART B -- FALSE POSITIVES: alerts on weapon-free camera footage")
    print("=" * 74)

    frames = []
    if NEG.exists():
        for p in sorted(NEG.glob("*.jpg")):
            im = cv2.imread(str(p))
            if im is not None:
                frames.append(im)
    if not frames:
        print("no negatives found -- run tools/capture_negatives.py first")
        return {}
    print(f"{len(frames)} weapon-free frames from the deployment camera\n")

    results = {"conf": [], "raw": [], "verified": []}
    print(f"{'conf':>6}{'raw detections':>17}{'after verify':>15}{'suppressed':>13}")
    for conf in SWEEP:
        det = WeaponDetector(conf=conf)
        # person boxes are unknown offline; the person gate is therefore
        # disabled here so the measured suppression comes only from the class
        # and persistence gates -- a deliberately conservative estimate.
        rc.VERIFY_REQUIRE_PERSON = False
        ver = TwoStageVerifier(enabled=True)
        raw_n = 0
        for f in frames:
            cands = det.detect(f)
            raw_n += len(cands)
            ver.verify(cands, [])
        conf_n = ver.stats.get("confirmed", 0)
        results["conf"].append(conf)
        results["raw"].append(raw_n)
        results["verified"].append(conf_n)
        supp = 100 * (raw_n - conf_n) / raw_n if raw_n else 0.0
        print(f"{conf:>6.2f}{raw_n:>17}{conf_n:>15}{supp:>12.1f}%")

    results["frames"] = len(frames)
    return results


def main() -> int:
    if not WEIGHTS.exists():
        sys.exit(f"weights not found: {WEIGHTS}")
    a = part_a_false_negatives()
    b = part_b_false_positives()

    print("\n" + "=" * 74)
    print("RECOMMENDED OPERATING POINT")
    print("=" * 74)
    print(f"  stage 1 detect conf : 0.25   (high recall -> fewer false negatives)")
    print(f"  stage 2 class gates : guns 0.48 | knife 0.53 | long_gun 0.65")
    print(f"  stage 2 persistence : 2 consecutive cycles")
    print(f"  stage 2 association : weapon must belong to a tracked person")

    (HERE / "threshold_tuning.json").write_text(
        json.dumps({"false_negatives": a, "false_positives": b}, indent=2))
    print(f"\nsaved -> {HERE / 'threshold_tuning.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
