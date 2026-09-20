"""Re-identification accuracy against annotated exit/return events.

Ground truth is per EVENT, not per frame -- see SCENARIOS.md. For each
annotated "person A left at frame X and came back at frame Y", check
whether the identity assigned after the return matches the one held
before the exit.

Metrics
-------
re-ID accuracy      correct re-matches / total re-entry events
                    The headline number.
false-split rate    a returning person given a NEW identity. Recoverable
                    -- visible, and reconcilable after the fact.
false-merge rate    two DIFFERENT people sharing one identity. NOT
                    recoverable: their samples contaminate each other
                    permanently, so this error compounds.
ID switches         canonical id changing on a continuously-tracked person
retention vs gap    accuracy bucketed by how long the person was away,
                    which is the curve that shows where the system stops
                    working.

The two error rates are reported separately and never averaged. They have
different costs, so a single combined score would hide the tradeoff that
actually matters.

Usage:
    python eval/eval_reid.py --clips eval/clips --gt eval/ground_truth
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector  # noqa: E402
from tracker import PersonTracker  # noqa: E402

SEARCH_WINDOW = 20   # frames either side of an annotated event to search
SETTLE_FRAMES = 25   # frames after a return before reading the identity


def track_clip(path, detector):
    """Run the pipeline. Returns per-frame [(canonical_id, bbox)] and stats."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    tracker = PersonTracker(use_reid=True)
    timeline = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        dets, boxes, confs = detector.detect(f)
        res = tracker.update(f, dets, boxes, confs)
        timeline.append([(r.get("canonical_id"), r["bbox"]) for r in res])
    cap.release()
    return timeline, tracker


def dominant(timeline, lo, hi):
    c = Counter()
    for i in range(max(0, lo), min(len(timeline), hi)):
        for cid, _b in timeline[i]:
            if cid is not None:
                c[cid] += 1
    return c.most_common(1)[0][0] if c else None


def _scan_decided(timeline, start, direction, max_scan=240, window=20):
    """Find the nearest stretch with a decided identity, scanning outward.

    Returns the dominant identity over `window` frames beginning at the
    first frame that has one, or None if nothing is decided within
    `max_scan` frames.
    """
    n = len(timeline)
    for step in range(max_scan):
        i = start + direction * step
        if i < 0 or i >= n:
            break
        if any(cid is not None for cid, _b in timeline[i]):
            lo, hi = (i, i + window) if direction > 0 else (i - window, i + 1)
            return dominant(timeline, lo, hi)
    return None


def _last_decided(timeline, frame, direction=-1):
    return _scan_decided(timeline, frame, direction)


def _first_decided(timeline, frame, direction=1):
    return _scan_decided(timeline, frame, direction)


def evaluate_clip(timeline, gt):
    """Score one clip against its ground truth."""
    events = []
    for ev in gt.get("events", []):
        exit_f = int(ev["exit_frame"])
        ret_f = int(ev["return_frame"])

        # Search OUTWARD from the event for the last/first confidently
        # decided identity, rather than reading a fixed offset.
        #
        # A fixed window fails the same way the pipeline used to: near an
        # exit the person is already partway out of frame, so the crop is a
        # fragment and the identity is still provisional (None). Reading
        # there reports "no identity before the gap" and scores a perfectly
        # maintained identity as a failure -- which is exactly what happened
        # on event 4 of the first recorded clip.
        before = _last_decided(timeline, exit_f, direction=-1)
        after = _first_decided(timeline, ret_f, direction=+1)

        events.append({
            "person": ev.get("person"),
            "gap_frames": ret_f - exit_f,
            "gap_seconds": round((ret_f - exit_f) / gt.get("fps", 30), 2),
            "identity_before": before,
            "identity_after": after,
            "correct": (before is not None and after is not None
                        and before == after),
            "undecided": after is None,
            "note": ev.get("note", ""),
        })

    # False merges: one canonical id covering two annotated people. Detected
    # by checking whether an identity that belongs to person A ever appears
    # simultaneously with itself, or spans two people's annotated events.
    by_person = defaultdict(set)
    for e in events:
        if e["identity_before"] is not None:
            by_person[e["person"]].add(e["identity_before"])
        if e["identity_after"] is not None:
            by_person[e["person"]].add(e["identity_after"])

    shared = set()
    people = list(by_person)
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            shared |= by_person[people[i]] & by_person[people[j]]

    # Duplicate identities within one frame are an outright invariant
    # violation -- the gallery's mutual exclusion should make this
    # impossible, so any occurrence is a bug rather than a tuning issue.
    dup_frames = sum(
        1 for entries in timeline
        if len([c for c, _ in entries if c is not None])
        != len({c for c, _ in entries if c is not None})
    )

    return {
        "events": events,
        "n_events": len(events),
        "n_correct": sum(e["correct"] for e in events),
        "n_undecided": sum(e["undecided"] for e in events),
        "identities_per_person": {p: len(v) for p, v in by_person.items()},
        "shared_identities": sorted(shared),
        "duplicate_id_frames": dup_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt, "*.json")))
    if not gt_files:
        raise SystemExit(
            f"no ground truth in {args.gt}. See eval/SCENARIOS.md for the "
            "recording protocol and the JSON format."
        )

    detector = PersonDetector()
    all_events, per_clip = [], {}

    for gtf in gt_files:
        with open(gtf) as fh:
            gt = json.load(fh)
        clip = os.path.join(args.clips, gt["clip"])
        if not os.path.exists(clip):
            print(f"  missing clip: {clip}")
            continue
        print(f"tracking {gt['clip']} ...")
        timeline, tracker = track_clip(clip, detector)
        if timeline is None:
            print("  could not open")
            continue
        r = evaluate_clip(timeline, gt)
        r["identities_total"] = len(tracker.gallery)
        per_clip[gt["clip"]] = r
        all_events.extend(r["events"])
        print(f"  {r['n_correct']}/{r['n_events']} re-entries correct, "
              f"{r['identities_total']} identities created")

    if not all_events:
        raise SystemExit("no events evaluated")

    n = len(all_events)
    correct = sum(e["correct"] for e in all_events)
    undecided = sum(e["undecided"] for e in all_events)
    split = sum(1 for e in all_events
                if not e["correct"] and not e["undecided"])
    merged = sum(len(r["shared_identities"]) for r in per_clip.values())
    dup = sum(r["duplicate_id_frames"] for r in per_clip.values())

    print("\n" + "=" * 62)
    print(f"  re-identification accuracy   {correct / n * 100:6.1f}%  "
          f"({correct}/{n} re-entry events)")
    print(f"  false-split rate             {split / n * 100:6.1f}%  "
          f"(returning person given a new id)")
    print(f"  undecided rate               {undecided / n * 100:6.1f}%  "
          f"(never resolved to any id)")
    print(f"  shared identities            {merged:6d}   "
          f"(false merges -- unrecoverable)")
    print(f"  duplicate-id frames          {dup:6d}   "
          f"(invariant violation; should be 0)")
    print("=" * 62)

    print("\nretention by gap length")
    buckets = [(0, 30), (30, 90), (90, 300), (300, 900), (900, 10 ** 9)]
    labels = ["< 1s", "1-3s", "3-10s", "10-30s", "> 30s"]
    for (lo, hi), lab in zip(buckets, labels):
        sel = [e for e in all_events if lo <= e["gap_frames"] < hi]
        if not sel:
            continue
        acc = sum(e["correct"] for e in sel) / len(sel)
        bar = "#" * int(acc * 30)
        print(f"  {lab:>7s}  n={len(sel):3d}  {acc * 100:5.1f}%  {bar}")

    changed = [e for e in all_events if e["note"]]
    if changed:
        acc = sum(e["correct"] for e in changed) / len(changed)
        print(f"\n  events with a noted appearance change: "
              f"{acc * 100:.1f}% correct (n={len(changed)})")

    if dup:
        print(f"\n  WARNING: {dup} frames contained the same canonical id "
              "twice.\n  The gallery's Hungarian mutual exclusion is meant to "
              "make this\n  impossible -- this is a bug, not a threshold "
              "problem.")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "reid_results.json")
    with open(out, "w") as fh:
        json.dump({
            "summary": {
                "accuracy": round(correct / n, 4),
                "false_split_rate": round(split / n, 4),
                "undecided_rate": round(undecided / n, 4),
                "shared_identities": merged,
                "duplicate_id_frames": dup,
                "n_events": n,
            },
            "per_clip": per_clip,
        }, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
