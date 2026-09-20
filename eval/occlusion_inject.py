"""Measure identity retention across occlusions of controlled length.

The problem with evaluating re-identification is getting ground truth:
knowing that the person who walked back in is the SAME person who walked
out normally requires a human to say so.

This sidesteps that. Take a clip where someone is tracked continuously,
then programmatically hide them for N frames and check whether the same
canonical identity comes back. Because the person never actually left,
the correct answer is known for free, at any N, with no annotation.

    reference run    ---- person tracked continuously, identity = A
    occluded run     ---- same clip, person painted out for N frames
                          identity before the gap = A
                          identity after  the gap = A   -> retained
                                                 != A   -> lost

What this measures well:
    the identity-retention-vs-occlusion-duration curve, which is the
    headline claim of a re-ID pipeline.

What it does NOT measure:
    realism. A painted rectangle is not a pillar, and the person does not
    change pose, lighting or viewpoint while hidden -- all of which make
    real re-identification harder. Treat the resulting curve as an UPPER
    BOUND on real-world retention, and pair it with recorded scenarios
    (see SCENARIOS.md) for the honest number.

Usage:
    python eval/occlusion_inject.py --clip <video> --sweep 5,10,20,30,60,120
    python eval/occlusion_inject.py --clip <video> --mode inpaint --out r.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector  # noqa: E402
from tracker import PersonTracker  # noqa: E402

DEFAULT_SWEEP = (5, 10, 20, 30, 60, 120, 300)


def read_frames(path, limit=0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if limit and len(frames) >= limit:
            break
    cap.release()
    return frames


def run_pipeline(frames, detector, seed_background=None):
    """Track through a frame list. Returns per-frame [(canonical_id, bbox)]."""
    tracker = PersonTracker(use_reid=True)
    timeline = []
    for f in frames:
        dets, boxes, confs = detector.detect(f)
        res = tracker.update(f, dets, boxes, confs)
        timeline.append([
            (r.get("canonical_id"), r["bbox"]) for r in res
        ])
    return timeline, tracker


def dominant_identity(timeline, lo, hi):
    """Most common canonical id in [lo, hi), ignoring undecided tracks."""
    c = Counter()
    for i in range(max(0, lo), min(len(timeline), hi)):
        for cid, _b in timeline[i]:
            if cid is not None:
                c[cid] += 1
    return c.most_common(1)[0][0] if c else None


def find_continuous_subject(timeline, min_len):
    """Pick the identity present for the longest unbroken stretch.

    Returns (canonical_id, start, end) or None.
    """
    spans = defaultdict(list)
    for i, entries in enumerate(timeline):
        for cid, _b in entries:
            if cid is not None:
                spans[cid].append(i)
    best = None
    for cid, idxs in spans.items():
        if not idxs:
            continue
        run_start = prev = idxs[0]
        for i in idxs[1:] + [None]:
            if i is not None and i == prev + 1:
                prev = i
                continue
            length = prev - run_start + 1
            if length >= min_len and (best is None or length > best[3]):
                best = (cid, run_start, prev, length)
            if i is not None:
                run_start = prev = i
    if best is None:
        return None
    return best[0], best[1], best[2]


def box_at(timeline, idx, cid):
    for c, b in timeline[idx]:
        if c == cid:
            return b
    return None


def occlude(frame, bbox, mode="fill", pad=8):
    """Hide a person. `fill` paints a flat patch; `inpaint` blurs heavily.

    Flat fill is the cleaner experiment -- it removes the person entirely
    with no residual texture for the detector to latch onto. Inpaint is
    less artificial but can leave a ghost the detector still fires on,
    which would make the occlusion incomplete and the result optimistic.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    x1 = max(0, int(bbox[0]) - pad); y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad); y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return out
    if mode == "inpaint":
        patch = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (0, 0), 25)
    else:
        # Median of the frame border, so the patch reads as background
        # rather than as a black rectangle the detector treats as an object.
        border = np.concatenate([
            frame[0:4].reshape(-1, 3), frame[-4:].reshape(-1, 3),
            frame[:, 0:4].reshape(-1, 3), frame[:, -4:].reshape(-1, 3),
        ])
        out[y1:y2, x1:x2] = np.median(border, axis=0).astype(np.uint8)
    return out


def trial(frames, detector, ref_timeline, cid, start, end, n_frames, mode):
    """Occlude `cid` for n_frames in the middle of its span, then re-track."""
    span = end - start + 1
    if span < n_frames + 20:
        return None

    gap_start = start + (span - n_frames) // 2
    gap_end = gap_start + n_frames

    modified = []
    for i, f in enumerate(frames):
        if gap_start <= i < gap_end:
            b = box_at(ref_timeline, i, cid)
            if b is None:
                # Interpolate from the nearest known box so the person is
                # still hidden on frames where the reference lost them.
                prev_b = next((box_at(ref_timeline, j, cid)
                               for j in range(i, gap_start - 1, -1)
                               if box_at(ref_timeline, j, cid)), None)
                b = prev_b
            modified.append(occlude(f, b, mode) if b is not None else f)
        else:
            modified.append(f)

    timeline, tracker = run_pipeline(modified, detector)

    before = dominant_identity(timeline, max(start, gap_start - 15), gap_start)
    after = dominant_identity(timeline, gap_end + 2, gap_end + 25)

    detected_during = sum(
        1 for i in range(gap_start, min(gap_end, len(timeline)))
        if timeline[i]
    )

    return {
        "n_frames": n_frames,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "identity_before": before,
        "identity_after": after,
        "retained": (before is not None and after is not None
                     and before == after),
        "reacquired": after is not None,
        "leak_frames": detected_during,
        "identities_total": len(tracker.gallery),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--sweep", default=",".join(str(x) for x in DEFAULT_SWEEP))
    ap.add_argument("--mode", choices=["fill", "inpaint"], default="fill")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
    frames = read_frames(args.clip, args.max_frames)
    print(f"{os.path.basename(args.clip)}: {len(frames)} frames")

    detector = PersonDetector()
    print("reference run (no occlusion) ...")
    ref_timeline, ref_tracker = run_pipeline(frames, detector)

    subject = find_continuous_subject(ref_timeline, min_len=min(sweep) + 20)
    if subject is None:
        raise SystemExit(
            "no continuously-tracked person long enough to occlude. "
            "Use a clip where one person stays visible for a while."
        )
    cid, start, end = subject
    print(f"subject: identity {cid}, frames {start}-{end} "
          f"({end - start + 1} frames continuous)")
    print(f"reference identities in clip: {len(ref_tracker.gallery)}\n")

    results = []
    fps = 30.0
    for n in sweep:
        r = trial(frames, detector, ref_timeline, cid, start, end,
                  n, args.mode)
        if r is None:
            print(f"  N={n:4d}  SKIPPED (span too short)")
            continue
        results.append(r)
        status = "RETAINED" if r["retained"] else (
            "NEW ID" if r["reacquired"] else "LOST")
        print(f"  N={n:4d} ({n / fps:5.2f}s)  {status:9s}  "
              f"before={r['identity_before']} after={r['identity_after']}  "
              f"leak={r['leak_frames']}")

    if results:
        retained = sum(r["retained"] for r in results)
        print(f"\nidentity retained in {retained}/{len(results)} trials")
        print("\nretention curve")
        print(f"  {'occlusion':>12s}  {'seconds':>8s}  {'retained':>9s}")
        for r in results:
            print(f"  {r['n_frames']:9d} fr  {r['n_frames'] / fps:8.2f}  "
                  f"{'yes' if r['retained'] else 'no':>9s}")

        leaks = [r for r in results if r["leak_frames"] > 0]
        if leaks:
            print(
                f"\n  NOTE: {len(leaks)} trial(s) still detected people during\n"
                "  the occlusion window. That is expected when other people\n"
                "  are in frame, but if it happens in a single-person clip the\n"
                "  occlusion is incomplete and the result is optimistic."
            )

    print(
        "\n  This curve is an UPPER BOUND. The subject does not change pose,\n"
        "  lighting or viewpoint while hidden, and a painted patch is not a\n"
        "  real occluder. Pair with recorded scenarios for the honest number."
    )

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "occlusion_results.json")
    with open(out, "w") as fh:
        json.dump({
            "clip": os.path.basename(args.clip),
            "frames": len(frames),
            "subject_identity": cid,
            "subject_span": [start, end],
            "mode": args.mode,
            "trials": results,
        }, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
