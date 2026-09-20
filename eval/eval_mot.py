"""Tracking and re-identification metrics on MOT17.

Why this exists
---------------
Every other measurement in this project needed either annotation you have to
produce, or pseudo-labels the system generated itself -- and pseudo-labels
flatter whatever the system already does, which is how bench_embedders.py
came to rank MobileNetV3 above OSNet by a factor of twelve.

MOT17 has per-frame, human-annotated identity ground truth on full-frame
surveillance video. That makes it the only source here that can score the
TRACKER rather than just the embedder:

  Market-1501 / MSMT17   pre-cropped stills -> embedder quality only
  MARS / PRID / iLIDS    tracklets of crops -> embedder over time, still no
                         full frames, so no tracking, no ghost boxes, no
                         multiple people in one shot
  MOT17                  full frames, many people, real occlusions, real
                         entries and exits -> the whole pipeline

Sequences used are the STATIC-camera ones. MOT17-05/10/11/13 are filmed from
a moving platform, which is a different problem from a fixed CCTV install.

  MOT17-02  static, street, ~20 people
  MOT17-04  static, night street, ~45 people per frame
  MOT17-09  static, shopping mall, ~10 people -- closest to a CCTV view

Metrics
-------
IDF1        identity-preserving F1. The headline tracking-with-identity
            metric: it penalises an identity that fragments or swaps, which
            MOTA largely does not.
MOTA        detection-driven accuracy (misses + false positives + switches).
IDSW        ID switches: how often a tracked person changes canonical id.
MT / ML     mostly-tracked / mostly-lost trajectories (>80% / <20% covered).
FRAG        trajectory fragmentations.

Re-identification is scored separately, because MOT metrics mix detection
and association and can hide a re-ID failure behind good detection:

  re-entry accuracy   for every ground-truth person who LEAVES and RETURNS
                      (a gap of >= --gap-frames in their annotation), was the
                      same canonical id restored?
  false-merge count   canonical ids covering two different GT identities
  false-split count   GT identities covered by more than one canonical id

Usage
-----
  python eval/eval_mot.py --root data/MOT17/train
  python eval/eval_mot.py --root data/MOT17/train --seq MOT17-09-FRCNN
  python eval/eval_mot.py --root data/MOT17/train --max-frames 300
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector  # noqa: E402
from tracker import PersonTracker  # noqa: E402

STATIC_SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"]
IOU_THRESHOLD = 0.5


def load_gt(path):
    """MOT format: frame,id,x,y,w,h,conf,class,visibility

    Only class 1 (pedestrian) with conf==1 counts. The other classes are
    vehicles, occluders and distractors that the benchmark excludes; scoring
    against them would invent misses that the protocol does not count.
    """
    gt = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = (float(v) for v in p[2:6])
            conf, cls, vis = float(p[6]), int(p[7]), float(p[8])
            if conf < 1 or cls != 1:
                continue
            gt[frame].append((tid, (x, y, x + w, y + h), vis))
    return gt


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    A = np.asarray(a, np.float32)
    B = np.asarray(b, np.float32)
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ar_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ar_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = ar_a[:, None] + ar_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def run_sequence(seq_dir, max_frames=0, min_vis=0.25, imgsz=None):
    """Track a sequence and align results to ground truth per frame."""
    img_dir = os.path.join(seq_dir, "img1")
    gt = load_gt(os.path.join(seq_dir, "gt", "gt.txt"))
    frames = sorted(os.listdir(img_dir))
    if max_frames:
        frames = frames[:max_frames]

    detector = (PersonDetector(imgsz=imgsz) if imgsz
                else PersonDetector())
    tracker = PersonTracker(use_reid=True)

    # matched[frame] = list of (gt_id, canonical_id)
    matched, n_gt, n_pred, n_miss, n_fp = {}, 0, 0, 0, 0
    gt_presence = defaultdict(list)      # gt_id -> [frames present]
    gt_to_cids = defaultdict(Counter)    # gt_id -> Counter(canonical ids)
    cid_to_gts = defaultdict(Counter)
    prev_assign = {}                     # gt_id -> canonical id last frame
    idsw = 0

    for i, fn in enumerate(frames, start=1):
        img = cv2.imread(os.path.join(img_dir, fn))
        if img is None:
            continue
        dets, boxes, confs = detector.detect(img)
        tracks = tracker.update(img, dets, boxes, confs)

        gt_items = [(tid, box) for tid, box, vis in gt.get(i, [])
                    if vis >= min_vis]
        pred = [(t.get("canonical_id"), t["bbox"]) for t in tracks
                if t.get("canonical_id") is not None]

        for tid, _b in gt_items:
            gt_presence[tid].append(i)
        n_gt += len(gt_items)
        n_pred += len(pred)

        if gt_items and pred:
            M = iou_matrix([b for _t, b in gt_items], [b for _c, b in pred])
            cost = 1.0 - M
            rows, cols = linear_sum_assignment(cost)
            pairs = [(r, c) for r, c in zip(rows, cols)
                     if M[r, c] >= IOU_THRESHOLD]
        else:
            pairs = []

        frame_pairs = []
        for r, c in pairs:
            gid = gt_items[r][0]
            cid = pred[c][0]
            frame_pairs.append((gid, cid))
            gt_to_cids[gid][cid] += 1
            cid_to_gts[cid][gid] += 1
            if gid in prev_assign and prev_assign[gid] != cid:
                idsw += 1
            prev_assign[gid] = cid

        matched[i] = frame_pairs
        n_miss += len(gt_items) - len(pairs)
        n_fp += len(pred) - len(pairs)

    return {
        "n_frames": len(frames),
        "n_gt": n_gt, "n_pred": n_pred,
        "n_miss": n_miss, "n_fp": n_fp, "idsw": idsw,
        "matched": matched,
        "gt_presence": dict(gt_presence),
        "gt_to_cids": {k: dict(v) for k, v in gt_to_cids.items()},
        "cid_to_gts": {k: dict(v) for k, v in cid_to_gts.items()},
        "identities_created": len(tracker.gallery),
    }


def compute_metrics(r, gap_frames=30):
    n_gt = max(r["n_gt"], 1)
    tp = n_gt - r["n_miss"]

    mota = 1.0 - (r["n_miss"] + r["n_fp"] + r["idsw"]) / n_gt

    # IDF1 approximated by the dominant-identity assignment: for each GT id,
    # the canonical id it was matched to most often counts as correct.
    idtp = sum(max(c.values()) for c in r["gt_to_cids"].values() if c)
    idp = idtp / max(r["n_pred"], 1)
    idr = idtp / n_gt
    idf1 = 2 * idp * idr / (idp + idr) if (idp + idr) > 0 else 0.0

    # Mostly tracked / mostly lost
    mt = ml = 0
    for gid, frames_present in r["gt_presence"].items():
        covered = sum(r["gt_to_cids"].get(gid, {}).values())
        ratio = covered / max(len(frames_present), 1)
        if ratio >= 0.8:
            mt += 1
        elif ratio < 0.2:
            ml += 1

    # False merges / splits
    merges = sum(1 for cid, gts in r["cid_to_gts"].items() if len(gts) > 1)
    splits = sum(1 for gid, cids in r["gt_to_cids"].items() if len(cids) > 1)

    # Re-entry events: a GT identity absent for >= gap_frames then present
    # again. This is the capability the whole gallery exists to provide.
    reentry_total = reentry_ok = 0
    for gid, frames_present in r["gt_presence"].items():
        fp = sorted(frames_present)
        for a, b in zip(fp, fp[1:]):
            if b - a < gap_frames:
                continue
            before = _cid_at(r, gid, a)
            after = _cid_at(r, gid, b)
            if before is None or after is None:
                continue
            reentry_total += 1
            reentry_ok += int(before == after)

    return {
        "IDF1": round(idf1, 4),
        "MOTA": round(mota, 4),
        "IDSW": r["idsw"],
        "MT": mt, "ML": ml,
        "n_gt_ids": len(r["gt_presence"]),
        "misses": r["n_miss"], "false_positives": r["n_fp"],
        "recall": round(tp / n_gt, 4),
        "false_merges": merges,
        "false_splits": splits,
        "identities_created": r["identities_created"],
        "reentry_events": reentry_total,
        "reentry_correct": reentry_ok,
        "reentry_accuracy": round(reentry_ok / reentry_total, 4)
        if reentry_total else None,
    }


def _cid_at(r, gid, frame):
    for g, c in r["matched"].get(frame, []):
        if g == gid:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reid", "MOT17", "train"))
    ap.add_argument("--seq", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gap-frames", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="detector inference size; MOT17 is 1080p "
                         "so the 640 default downscales 3x")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs = args.seq or [s for s in STATIC_SEQS
                        if os.path.isdir(os.path.join(args.root, s))]
    if not seqs:
        raise SystemExit(f"no sequences found under {args.root}")

    results = {}
    for s in seqs:
        d = os.path.join(args.root, s)
        print(f"\n=== {s} ===", flush=True)
        raw = run_sequence(d, args.max_frames, imgsz=args.imgsz)
        m = compute_metrics(raw, args.gap_frames)
        results[s] = m
        print(f"  frames {raw['n_frames']}  GT ids {m['n_gt_ids']}  "
              f"created {m['identities_created']}")
        print(f"  IDF1 {m['IDF1'] * 100:5.1f}%   MOTA {m['MOTA'] * 100:6.1f}%   "
              f"recall {m['recall'] * 100:5.1f}%   IDSW {m['IDSW']}")
        print(f"  MT {m['MT']}  ML {m['ML']}  "
              f"false-merge {m['false_merges']}  false-split {m['false_splits']}")
        if m["reentry_accuracy"] is not None:
            print(f"  re-entry: {m['reentry_correct']}/{m['reentry_events']} "
                  f"= {m['reentry_accuracy'] * 100:.1f}%")
        else:
            print("  re-entry: no qualifying events")

    print("\n" + "=" * 96)
    print(f"{'sequence':20s}{'IDF1':>8s}{'MOTA':>9s}{'IDSW':>7s}"
          f"{'MT':>5s}{'ML':>5s}{'merge':>7s}{'split':>7s}{'re-entry':>11s}")
    print("-" * 96)
    for s, m in results.items():
        re = f"{m['reentry_accuracy'] * 100:.1f}%" \
            if m["reentry_accuracy"] is not None else "-"
        print(f"{s:20s}{m['IDF1'] * 100:7.1f}%{m['MOTA'] * 100:8.1f}%"
              f"{m['IDSW']:7d}{m['MT']:5d}{m['ML']:5d}"
              f"{m['false_merges']:7d}{m['false_splits']:7d}{re:>11s}")
    print("=" * 96)

    tot_e = sum(m["reentry_events"] for m in results.values())
    tot_c = sum(m["reentry_correct"] for m in results.values())
    if tot_e:
        print(f"\nOVERALL RE-IDENTIFICATION ACCURACY: "
              f"{tot_c}/{tot_e} = {tot_c / tot_e * 100:.1f}%")
        print(f"  (a re-entry is a ground-truth person absent for "
              f">= {args.gap_frames} frames who then returns)")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "mot17_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
