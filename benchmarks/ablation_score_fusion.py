"""Ablation over the identity scoring function.

gallery.Identity.similarity() currently computes

    max(centroid . q,  MATCH_SAMPLE_WEIGHT * max_i(sample_i . q))

and the result is optionally fused with body-proportion similarity at
REID_BODY_RATIO_WEIGHT. Each of those choices was reasoned about but never
measured. This measures them.

Method
------
Collect embeddings with pseudo-track labels from the recorded clips, then
score every query against every OTHER track's stored sample set under each
variant. Positives are same-track pairs separated by >= 30 frames (the
re-entry case); negatives are same-frame pairs, which are guaranteed to be
different people.

Reported per variant:
    separability   intra mean - inter mean
    balanced acc   best achievable (TPR + TNR) / 2 over all thresholds
    recall @ 1% FM recall at the threshold giving 1% false merges

Accuracy on the recorded clips saturates at 100% (8/8), so it cannot rank
these variants. Separability is continuous and does.

Usage:
    python benchmarks/ablation_score_fusion.py
"""
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector          # noqa: E402
from embedder import OSNetEmbedder           # noqa: E402
from pose import PoseEstimator, compute_body_ratios  # noqa: E402
import config                                # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = [
    "clip9_even_lighting.avi",
    "clip6_two_people_crossing.avi",
    "clip12_two_people_reentry.avi",
]
STRIDE = 3
K = config.REID_GALLERY_SAMPLES


def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def collect():
    """Return samples: list of dicts {clip, track, frame, feat, ratios}."""
    det = PersonDetector()
    emb = OSNetEmbedder()
    pose = PoseEstimator()
    out = []
    for clip in CLIPS:
        path = os.path.join(BASE, "eval", "clips", clip)
        if not os.path.exists(path):
            print(f"  skip missing {clip}")
            continue
        cap = cv2.VideoCapture(path)
        nxt, prev, idx = 0, [], 0
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % STRIDE == 0:
                _d, boxes, confs = det.detect(frame)
                keep = [(b, c) for b, c in zip(boxes, confs)
                        if c >= 0.5 and (b[3] - b[1]) >= 64]
                if not keep:
                    prev = []
                else:
                    kb = np.array([k[0] for k in keep], np.float32)
                    feats, valid = emb.embed(frame, kb)
                    stub = [{"track_id": j, "bbox": tuple(kb[j])}
                            for j in range(len(kb))]
                    pmap = pose.update(frame, stub)
                    cur = []
                    for j, ((b, _c), f, v) in enumerate(zip(keep, feats, valid)):
                        if not v:
                            continue
                        best, biou = None, 0.0
                        for pb, pid in prev:
                            iou = box_iou(b, pb)
                            if iou > biou:
                                biou, best = iou, pid
                        if biou < 0.5:
                            best = nxt
                            nxt += 1
                        out.append({
                            "clip": clip, "track": f"{clip}:{best}",
                            "frame": idx, "feat": f,
                            "ratios": compute_body_ratios(pmap.get(j)),
                        })
                        cur.append((b, best))
                        n += 1
                    prev = cur
            idx += 1
        cap.release()
        print(f"  {clip:34s} {n:5d} samples")
    pose.close()
    return out


# ------------------------------------------------------------- score variants

def score_centroid(q, gal_feats, _gr, _qr):
    c = gal_feats.mean(axis=0)
    nc = np.linalg.norm(c)
    return float((c / nc) @ q) if nc > 1e-8 else 0.0


def score_bestsample(q, gal_feats, _gr, _qr):
    return float((gal_feats @ q).max())


def score_hybrid(q, gal_feats, _gr, _qr, w=config.MATCH_SAMPLE_WEIGHT):
    c = gal_feats.mean(axis=0)
    nc = np.linalg.norm(c)
    cs = float((c / nc) @ q) if nc > 1e-8 else 0.0
    bs = float((gal_feats @ q).max())
    return max(cs, w * bs)


def score_hybrid_ratios(q, gal_feats, gr, qr,
                        w=config.REID_BODY_RATIO_WEIGHT):
    base = score_hybrid(q, gal_feats, gr, qr)
    if gr is None or qr is None:
        return base
    na, nb = np.linalg.norm(qr), np.linalg.norm(gr)
    if na < 1e-8 or nb < 1e-8:
        return base
    rs = float(np.dot(qr, gr) / (na * nb))
    return (1.0 - w) * base + w * rs


VARIANTS = [
    ("centroid only", score_centroid),
    ("best sample only", score_bestsample),
    (f"hybrid max(c, {config.MATCH_SAMPLE_WEIGHT}*b)", score_hybrid),
    (f"hybrid + body ratios ({config.REID_BODY_RATIO_WEIGHT})",
     score_hybrid_ratios),
]


def build_galleries(samples):
    """Per track: up to K diverse samples + mean body ratios."""
    by = defaultdict(list)
    for s in samples:
        by[s["track"]].append(s)
    gal = {}
    for t, items in by.items():
        if len(items) < 4:
            continue
        feats = np.stack([i["feat"] for i in items])
        frames = np.array([i["frame"] for i in items])
        if len(feats) > K:
            sel = [0]
            while len(sel) < K:
                d = (feats[sel] @ feats.T).max(axis=0)
                d[sel] = 2.0
                sel.append(int(np.argmin(d)))
            feats = feats[sel]
            frames = frames[sel]
        rs = [i["ratios"] for i in items if i["ratios"] is not None]
        gal[t] = (feats, np.mean(rs, axis=0) if rs else None, frames)
    return gal, by


def evaluate(samples, gal, by, fn, min_gap=30):
    """Score each variant, excluding temporally-adjacent gallery samples.

    Without that exclusion the comparison is rigged toward "best sample":
    the gallery is drawn from the whole track, so a query at frame 100 can
    match a stored sample from frame 99 -- near-identical, and nothing to do
    with re-identification after an absence. Any variant that takes a MAX
    over samples exploits that; centroid-based variants cannot.

    So each query is scored only against gallery samples at least `min_gap`
    frames away from it, which is the situation the real system faces when
    somebody returns.
    """
    intra, inter = [], []
    for t, (feats, gr, gframes) in gal.items():
        items = by[t]
        # positives: same track, scored against temporally distant samples
        for it in items:
            far = np.abs(gframes - it["frame"]) >= min_gap
            if far.sum() < 3:
                continue
            intra.append(fn(it["feat"], feats[far], gr, it["ratios"]))
        # negatives: same-frame, different track
        for other, o_items in by.items():
            if other == t or other.split(":")[0] != t.split(":")[0]:
                continue
            fr = {i["frame"] for i in items}
            for it in o_items:
                if it["frame"] in fr:
                    inter.append(fn(it["feat"], feats, gr, it["ratios"]))
    intra, inter = np.array(intra), np.array(inter)
    if intra.size < 20 or inter.size < 10:
        return None
    ths = np.linspace(0, 1, 501)
    acc = max(((intra >= t).mean() + (inter < t).mean()) / 2 for t in ths)
    t1 = float(np.quantile(inter, 0.99))
    return {
        "n_intra": int(intra.size), "n_inter": int(inter.size),
        "intra": round(float(intra.mean()), 4),
        "inter": round(float(inter.mean()), 4),
        "separability": round(float(intra.mean() - inter.mean()), 4),
        "balanced_acc": round(float(acc), 4),
        "recall_at_1pct_fm": round(float((intra >= t1).mean()), 4),
    }


def main():
    print("collecting embeddings ...")
    samples = collect()
    print(f"  total {len(samples)} samples\n")
    gal, by = build_galleries(samples)
    print(f"galleries built: {len(gal)} pseudo-tracks\n")

    results = {}
    print(f"{'variant':40s}{'intra':>8s}{'inter':>8s}{'sep':>8s}"
          f"{'bal.acc':>9s}{'rec@1%':>9s}")
    print("-" * 82)
    for name, fn in VARIANTS:
        m = evaluate(samples, gal, by, fn)
        if m is None:
            print(f"{name:40s}   insufficient pairs")
            continue
        results[name] = m
        print(f"{name:40s}{m['intra']:8.3f}{m['inter']:8.3f}"
              f"{m['separability']:8.3f}{m['balanced_acc']:9.3f}"
              f"{m['recall_at_1pct_fm']:9.3f}")
    print("-" * 82)
    any_m = next(iter(results.values()))
    print(f"pairs: {any_m['n_intra']} positives (long-gap), "
          f"{any_m['n_inter']} negatives (same-frame)")

    out = os.path.join(BASE, "benchmarks", "ablation_score_fusion.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
