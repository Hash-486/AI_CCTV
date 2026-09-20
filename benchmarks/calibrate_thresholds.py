"""Derive identity thresholds from measured cosine distributions.

Every threshold in config.py marked CALIBRATED is a placeholder. Guessing
them is how you get the two failure modes this pipeline exists to avoid:
set them too high and one person becomes many identities (false split);
set them too low and two people merge into one (false merge, which also
poisons the gallery permanently).

Method
------
Build two distributions from real footage:

  intra-person : embeddings of the SAME person at different times
  inter-person : embeddings of DIFFERENT people

Same-person pairs come from IoU-chained detections across consecutive
sampled frames. Different-person pairs come from detections in the SAME
frame, which are guaranteed distinct people.

Then choose thresholds at a stated operating point rather than by eye:

  T_MATCH  the similarity above which a match is accepted. Set where the
           false-merge rate hits TARGET_FALSE_MERGE.
  T_NEW    below this, definitely a stranger. Set at the low tail of the
           intra distribution, so anything that could plausibly be a known
           person is at least considered rather than immediately forked.
  T_MARGIN required gap to the runner-up. Set from how close inter-person
           scores cluster: if two identities routinely score within X of
           each other, a lead smaller than X is not evidence.
  T_VETO   used to reject a suspicious re-association. Looser than T_MATCH
           because it only has to catch a clearly-wrong body, and being
           strict here breaks legitimate tracking through partial occlusion.
  DEEPSORT_MAX_COS_DIST  DeepSORT gates on cosine DISTANCE (1 - similarity).

Usage
-----
  python benchmarks/calibrate_thresholds.py --clips <dir-or-file> [...]
  python benchmarks/calibrate_thresholds.py --clips a.avi b.avi --out thresholds.json

CAVEAT: run this on CLEAN footage. Video written by an annotating pipeline
has boxes and skeletons burnt into every crop, and you will be calibrating
partly on drawn graphics.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector  # noqa: E402
from embedder import OSNetEmbedder  # noqa: E402

TARGET_FALSE_MERGE = 0.01   # accept 1% of different-person pairs above T_MATCH
INTRA_LOW_TAIL = 0.02       # T_NEW at the 2nd percentile of same-person pairs


def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def collect(clips, stride=3, min_h=64, min_conf=0.5, min_track=3):
    """Return {clip: {pseudo_track_id: [(frame_index, embedding), ...]}}."""
    det = PersonDetector()
    emb = OSNetEmbedder()
    per_clip = {}

    for path in clips:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  skip (cannot open): {path}")
            continue

        tracks, next_id, prev = {}, 0, []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                _dets, boxes, confs = det.detect(frame)
                keep = [
                    (b, c) for b, c in zip(boxes, confs)
                    if c >= min_conf and (b[3] - b[1]) >= min_h
                ]
                if not keep:
                    # Reset the chain. Leaving `prev` populated lets a
                    # different person appearing later near the same spot
                    # be chained onto an older track, merging two people
                    # into one "intra" identity.
                    prev = []
                if keep:
                    kb = np.array([k[0] for k in keep], np.float32)
                    feats, valid = emb.embed(frame, kb)
                    cur = []
                    for (b, _c), f, v in zip(keep, feats, valid):
                        if not v:
                            continue
                        best, best_iou = None, 0.0
                        for pb, pid in prev:
                            iou = box_iou(b, pb)
                            if iou > best_iou:
                                best_iou, best = iou, pid
                        if best_iou < 0.5:
                            best = next_id
                            next_id += 1
                        tracks.setdefault(best, []).append((idx, f))
                        cur.append((b, best))
                    prev = cur
            idx += 1
        cap.release()

        tracks = {k: v for k, v in tracks.items() if len(v) >= min_track}
        per_clip[os.path.basename(path)] = tracks
        print(f"  {os.path.basename(path):32s} pseudo-tracks={len(tracks):3d} "
              f"samples={sum(len(v) for v in tracks.values()):4d}")
    return per_clip


def distributions(per_clip, long_gap_frames=30):
    """Build the score distributions.

    intra is split by TEMPORAL GAP, and this split is the whole point.

    Consecutive frames of one unbroken track are the easy case: same pose,
    same lighting, same viewpoint, scoring ~0.85+. Re-entry after an
    occlusion is the hard case, scoring far lower. Since IoU chaining
    produces overwhelmingly short-gap pairs, calibrating on all of intra
    measures the easy case and then applies the resulting threshold to the
    hard one -- which sets T_MATCH above the range real re-entries reach,
    so every returning person is rejected and given a new identity.

    Measured on this project's footage: short-gap pairs scored ~0.85 while
    an actual post-occlusion re-entry scored 0.61-0.72. A threshold derived
    from the former (0.81) rejected 100% of the latter.

    T_MATCH must therefore come from intra_long, not from intra.
    """
    intra, intra_long, inter, runner_gap = [], [], [], []
    for tracks in per_clip.values():
        ids = list(tracks)
        for tid in ids:
            items = tracks[tid]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    s = float(items[i][1] @ items[j][1])
                    intra.append(s)
                    if abs(items[j][0] - items[i][0]) >= long_gap_frames:
                        intra_long.append(s)
        # Negatives must come from crops in the SAME FRAME, which are
        # guaranteed to be different people.
        #
        # Pairing every pseudo-track against every other does NOT do that.
        # IoU chaining splits one person into a new pseudo-track every time
        # the chain breaks -- which is exactly what an exit and re-entry
        # causes -- so their cross-pairs are same-person pairs mislabelled
        # as negatives. Since T_MATCH is taken from the upper tail of this
        # distribution, those high-scoring impostors push it up and cause
        # real re-entries to be rejected: the precise failure this script
        # exists to prevent.
        #
        # It is worst on the clips most likely to be recorded first. A
        # scenario-1 clip is filmed ALONE, so every cross-track pair is the
        # same person and the entire "inter" distribution would be
        # fabricated.
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for fa, va in tracks[ids[a]]:
                    for fb, vb in tracks[ids[b]]:
                        if fa == fb:          # same frame => different people
                            inter.append(float(va @ vb))

        # How close do the top two DIFFERENT identities score for one query?
        # This is what T_MARGIN must exceed to be meaningful.
        for tid in ids:
            if len(ids) < 3:
                continue
            for _, q in tracks[tid][:5]:
                scores = []
                for other in ids:
                    if other == tid:
                        continue
                    scores.append(max(float(q @ v) for _, v in tracks[other]))
                scores.sort(reverse=True)
                if len(scores) >= 2:
                    runner_gap.append(scores[0] - scores[1])

    return (np.array(intra), np.array(intra_long),
            np.array(inter), np.array(runner_gap))


def choose(intra, intra_long, inter, runner_gap):
    # The hard positives govern T_MATCH. Fall back to all of intra only if
    # the clip contains no long-gap pairs at all, and say so.
    hard = intra_long if intra_long.size >= 30 else intra

    # Two candidates, and we take the LOWER:
    #   - the false-merge budget says "no lower than this"
    #   - the hard-positive distribution says "no higher than this, or real
    #     re-entries are rejected"
    # When they conflict, the data cannot support the requested false-merge
    # rate, and the diagnostics below make that explicit rather than
    # silently shipping a threshold that rejects everything.
    from_negatives = float(np.quantile(inter, 1.0 - TARGET_FALSE_MERGE))
    from_hard_positives = float(np.quantile(hard, 0.25))
    t_match = min(from_negatives, from_hard_positives)

    t_new = float(np.quantile(hard, INTRA_LOW_TAIL))
    t_new = min(t_new, t_match - 0.05)
    t_margin = float(np.quantile(runner_gap, 0.75)) if len(runner_gap) else 0.10
    t_veto = float(np.quantile(hard, 0.10))
    # DeepSORT gates on distance and only has to bridge max_age frames, so
    # it can use the easy short-gap distribution.
    max_cos_dist = float(1.0 - np.quantile(intra, 0.05))

    recall = float((hard >= t_match).mean())
    false_merge = float((inter >= t_match).mean())
    best_acc, best_t = 0.0, 0.0
    for t in np.linspace(0.0, 1.0, 501):
        acc = ((intra >= t).mean() + (inter < t).mean()) / 2.0
        if acc > best_acc:
            best_acc, best_t = acc, float(t)

    return {
        "T_MATCH": round(t_match, 4),
        "T_NEW": round(t_new, 4),
        "T_MARGIN": round(t_margin, 4),
        "T_VETO": round(t_veto, 4),
        "DEEPSORT_MAX_COS_DIST": round(max_cos_dist, 4),
        "_diagnostics": {
            "intra_n": int(intra.size),
            "intra_long_n": int(intra_long.size),
            "inter_n": int(inter.size),
            "used_hard_positives": bool(intra_long.size >= 30),
            "intra_mean": round(float(intra.mean()), 4),
            "intra_std": round(float(intra.std()), 4),
            "intra_long_mean": round(float(intra_long.mean()), 4)
            if intra_long.size else None,
            "inter_mean": round(float(inter.mean()), 4),
            "inter_std": round(float(inter.std()), 4),
            "separability_gap": round(float(hard.mean() - inter.mean()), 4),
            "t_match_from_negatives": round(from_negatives, 4),
            "t_match_from_hard_positives": round(from_hard_positives, 4),
            "recall_at_T_MATCH": round(recall, 4),
            "false_merge_at_T_MATCH": round(false_merge, 4),
            "best_balanced_accuracy": round(best_acc, 4),
            "best_balanced_threshold": round(best_t, 4),
        },
    }


def main():
    global TARGET_FALSE_MERGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True,
                    help="video files or directories of videos")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--false-merge", type=float, default=TARGET_FALSE_MERGE,
                    help="tolerated fraction of different-person pairs above "
                         "T_MATCH (default 0.01)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    TARGET_FALSE_MERGE = args.false_merge

    paths = []
    for c in args.clips:
        if os.path.isdir(c):
            for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
                paths.extend(sorted(glob.glob(os.path.join(c, ext))))
        else:
            paths.append(c)
    if not paths:
        raise SystemExit("no clips found")

    print(f"collecting embeddings from {len(paths)} clip(s)")
    per_clip = collect(paths, stride=args.stride)
    intra, intra_long, inter, gap = distributions(per_clip)

    if intra.size < 50 or inter.size < 50:
        raise SystemExit(
            f"not enough pairs (intra={intra.size}, inter={inter.size}). "
            "Need clips with several people and continuous tracks."
        )

    result = choose(intra, intra_long, inter, gap)
    d = result["_diagnostics"]

    print("\ndistributions")
    print(f"  intra-person (all)   n={d['intra_n']:6d}  "
          f"mean {d['intra_mean']:.4f} +- {d['intra_std']:.4f}   (easy case)")
    if d["intra_long_mean"] is not None:
        print(f"  intra-person (>=30f) n={d['intra_long_n']:6d}  "
              f"mean {d['intra_long_mean']:.4f}"
              f"{'   <-- governs T_MATCH' if d['used_hard_positives'] else ''}")
    print(f"  inter-person         n={d['inter_n']:6d}  "
          f"mean {d['inter_mean']:.4f} +- {d['inter_std']:.4f}")
    print(f"  separability gap {d['separability_gap']:+.4f}   "
          f"best balanced acc {d['best_balanced_accuracy']:.4f} "
          f"@ {d['best_balanced_threshold']:.3f}")

    print("\nT_MATCH candidates")
    print(f"  {d['t_match_from_negatives']:.4f}  from the false-merge budget "
          f"({TARGET_FALSE_MERGE * 100:.1f}%)")
    print(f"  {d['t_match_from_hard_positives']:.4f}  from the hard-positive "
          f"(long-gap) tail")
    print(f"  -> taking the lower: {result['T_MATCH']:.4f}")
    if d["t_match_from_hard_positives"] < d["t_match_from_negatives"]:
        print("     the hard positives bind, so the requested false-merge "
              "rate is not\n     achievable on this data without rejecting "
              "real re-entries")

    # The operating point is a policy choice, not a fact. A false MERGE is
    # unrecoverable -- two people share an identity and permanently corrupt
    # each other's stored samples. A false SPLIT is recoverable -- one person
    # holds two identities, which is visible and can be reconciled later.
    # So the curve is asymmetric, and you should pick a point on it rather
    # than accept whatever a single default produces.
    hard = intra_long if intra_long.size >= 30 else intra
    print("\noperating points")
    print(f"  {'false-merge':>11s}  {'T_MATCH':>8s}  {'easy recall':>12s}"
          f"  {'RE-ENTRY recall':>16s}")
    for fm in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35):
        t = float(np.quantile(inter, 1.0 - fm))
        print(f"  {fm * 100:10.1f}%  {t:8.4f}  {(intra >= t).mean() * 100:11.1f}%"
              f"  {(hard >= t).mean() * 100:15.1f}%")
    print("  'easy recall' is short-gap pairs; 'RE-ENTRY recall' is the "
          "long-gap pairs\n  that actually determine whether someone is "
          "recognised on return.")

    print("\nrecommended config.py values")
    for k in ("T_MATCH", "T_NEW", "T_MARGIN", "T_VETO", "DEEPSORT_MAX_COS_DIST"):
        print(f"  {k:24s} = {result[k]}")
    print(f"\n  at T_MATCH: {d['recall_at_T_MATCH'] * 100:.1f}% of true re-matches "
          f"accepted, {d['false_merge_at_T_MATCH'] * 100:.2f}% false merges")

    if d["separability_gap"] < 0.25 or d["best_balanced_accuracy"] < 0.85:
        print(
            "\n  WARNING: weak separability on this footage. Re-identification\n"
            "  cannot be reliable when intra- and inter-person distributions\n"
            "  overlap this much, regardless of threshold choice. Likely\n"
            "  causes: low-resolution crops, few distinct identities,\n"
            "  annotation overlays burnt into the source video, or pseudo-\n"
            "  tracks contaminated by IoU chaining through a crossing.\n"
            "  Recalibrate on clean multi-person footage before trusting\n"
            "  these numbers."
        )

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "thresholds.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
