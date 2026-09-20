# ============================================================
# eval_indomain.py -- false negatives on real footage from the deployment camera
# ============================================================
"""
Measure the miss rate on in-domain positives: a replica pistol held in front of
the actual webcam, in the actual room, at the actual 640x480.

Every other accuracy number in this project comes from public imagery. This is
the only measurement taken on the camera the system will run on, which makes it
the closest thing to a field number available before external CCTV arrives.

Ground truth is per-clip and manual, set in CLIPS below. `weapon_present=True`
means the replica is visible for essentially the whole clip, so any sampled
frame without a detection is a FALSE NEGATIVE.

CAVEAT, stated because it matters: recordings/*.avi are the pipeline's annotated
output, so each frame carries the old system's boxes, pose skeleton and labels.
A box drawn around the weapon could flatter the detector. The measurement is
therefore an optimistic bound, not a clean benchmark -- but the replica is large
and unambiguous, and the trend across thresholds is still informative.

Usage:
    python Review_1.2/benchmarks/eval_indomain.py
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
sys.path.insert(0, str(REVIEW / "pipeline"))
sys.path.insert(0, str(ROOT))

# Ground truth, established by inspecting the highest-confidence frames of each
# clip (see scratchpad/topdet.jpg -- a replica pistol held beside the head).
CLIPS = {
    "20260411_100507.avi": {"weapon_present": True,
                            "note": "replica pistol held beside the head, clearly visible"},
    "20260228_224637.avi": {"weapon_present": False,
                            "note": "person at desk, hands near face, no weapon"},
    "20260422_142654.avi": {"weapon_present": False,
                            "note": "face close to camera, hand near chin, no weapon"},
    "20260728_191730.avi": {"weapon_present": False,
                            "note": "canteen scene, several people, no weapon"},
}

SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
EVERY = 3


def main() -> int:
    from detectors import WeaponDetector

    det = WeaponDetector(conf=min(SWEEP))       # detect wide, filter afterwards

    # one decode pass; re-filter per threshold from the stored confidences
    per_clip: dict[str, list] = {}
    for name, meta in CLIPS.items():
        path = ROOT / "recordings" / name
        if not path.exists():
            print(f"[skip] {name} not found")
            continue
        cap = cv2.VideoCapture(str(path))
        i, frames = 0, []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i % EVERY:
                continue
            dets = det.detect(fr)
            frames.append(max((d["confidence"] for d in dets), default=0.0))
        cap.release()
        per_clip[name] = frames
        print(f"[scan] {name:<26} {len(frames)} sampled frames  "
              f"({'WEAPON' if meta['weapon_present'] else 'no weapon'})")

    pos = [n for n, m in CLIPS.items() if m["weapon_present"] and n in per_clip]
    neg = [n for n, m in CLIPS.items() if not m["weapon_present"] and n in per_clip]

    print("\n" + "=" * 78)
    print("IN-DOMAIN EVALUATION -- deployment webcam, 640x480")
    print("=" * 78)
    print("\nPOSITIVE clips (replica pistol present -- a miss is a FALSE NEGATIVE)")
    print(f"{'conf':>6}{'frames':>9}{'detected':>10}{'MISSED':>9}{'miss rate':>11}")
    out = {"positive": {}, "negative": {}}
    npos = sum(len(per_clip[n]) for n in pos)
    for c in SWEEP:
        hit = sum(1 for n in pos for v in per_clip[n] if v >= c)
        miss = npos - hit
        out["positive"][c] = {"frames": npos, "detected": hit, "missed": miss,
                              "miss_rate": round(miss / max(npos, 1), 4)}
        print(f"{c:>6.2f}{npos:>9}{hit:>10}{miss:>9}{100 * miss / max(npos, 1):>10.1f}%")

    print("\nNEGATIVE clips (no weapon -- any detection is a FALSE POSITIVE)")
    print(f"{'conf':>6}{'frames':>9}{'false alarms':>14}{'rate':>9}")
    nneg = sum(len(per_clip[n]) for n in neg)
    for c in SWEEP:
        fa = sum(1 for n in neg for v in per_clip[n] if v >= c)
        out["negative"][c] = {"frames": nneg, "false_alarms": fa,
                              "rate": round(fa / max(nneg, 1), 4)}
        print(f"{c:>6.2f}{nneg:>9}{fa:>14}{100 * fa / max(nneg, 1):>8.1f}%")

    print("\nper-clip false alarms at a few thresholds")
    print(f"{'clip':<28}" + "".join(f"{c:>8.2f}" for c in (0.25, 0.45, 0.55)))
    for n in neg:
        row = "".join(f"{sum(1 for v in per_clip[n] if v >= c):>8}"
                      for c in (0.25, 0.45, 0.55))
        print(f"{n:<28}{row}")

    print("\n" + "=" * 78)
    print("The trade-off this quantifies: lowering the detector threshold cuts")
    print("false negatives on the positive clip and raises false alarms on the")
    print("negative clips. Stage-2 verification is what removes the latter.")
    (HERE / "indomain_eval.json").write_text(
        json.dumps({"clips": {k: {**v, "frames": len(per_clip.get(k, []))}
                              for k, v in CLIPS.items()}, "results": out}, indent=2))
    print(f"saved -> {HERE / 'indomain_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
