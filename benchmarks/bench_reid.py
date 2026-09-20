"""Embedder comparison on Market-1501 with the standard evaluation protocol.

This replaces the pseudo-labelled comparison in bench_embedders.py, which
rested on 85 negative pairs from footage with annotation burnt into every
crop and could not distinguish the candidates from noise.

Here the identity labels are real, there are 750 query identities, and the
protocol is the one the re-ID literature reports, so the numbers are both
trustworthy and comparable to published results.

Protocol (Zheng et al. 2015, single-query)
------------------------------------------
Filenames encode identity and camera: 0001_c1s1_001051_00.jpg
  -> person 0001, camera 1, sequence 1, frame 001051

For each query image, rank the whole gallery by cosine similarity, then
discard two kinds of gallery entry before scoring:

  same person AND same camera   the trivial case. Two crops of one person
                                from one camera seconds apart are nearly
                                identical; counting them would measure
                                image similarity, not re-identification.
  person id -1                  distractors (false detections).

A correct match therefore requires recognising the person from a DIFFERENT
camera -- different viewpoint, lighting and scale. That is exactly the
"walked out and came back" problem, which is why this benchmark transfers.

Metrics
-------
rank-1 / rank-5 / rank-10   CMC: is a correct match in the top k?
mAP                          mean average precision over all correct matches,
                             which unlike rank-1 rewards finding ALL of them
                             and is the more honest summary.

Diagnostic value
----------------
Comparing a model's score here against its separability on your own footage
separates two situations that currently look identical:

  strong here, weak on your clips  -> the model is fine; your scene, camera
                                      or source video is the constraint
  weak here too                    -> the model choice is wrong

Usage:
  python benchmarks/bench_reid_market.py
  python benchmarks/bench_reid_market.py --data data/Market-1501-v15.09.15
  python benchmarks/bench_reid_market.py --limit-query 500   (quick pass)
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import REID_WEIGHTS_DIR  # noqa: E402
from embedder import OSNetEmbedder  # noqa: E402
from bench_embedders import TorchvisionEmbedder, TorchreidEmbedder  # noqa: E402

PATTERN = re.compile(r"([-\d]+)_c(\d)")


def parse(path):
    """Return (person_id, camera_id) from a Market-1501 filename."""
    m = PATTERN.search(os.path.basename(path))
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _subsample(items, limit, seed=0):
    """Sample RANDOMLY, never the first N.

    Market-1501 filenames sort with the "-1_*" distractors first, so a head
    slice of the gallery is entirely distractors and every query becomes
    unscoreable -- which is exactly the silent failure this replaced.
    """
    if not limit or limit >= len(items):
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=limit, replace=False)
    return [items[i] for i in sorted(idx)]


def load_market(root, split, limit=0):
    folder = os.path.join(
        root, "query" if split == "query" else "bounding_box_test"
    )
    items = []
    for f in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        pid, cam = parse(f)
        if pid is not None:
            items.append((f, pid, cam))
    return _subsample(items, limit)


def load_msmt17(root, split, limit=0):
    """MSMT17 ships explicit list files; camera id is the 3rd '_' field."""
    list_path = os.path.join(
        root, "list_query.txt" if split == "query" else "list_gallery.txt"
    )
    img_dir = os.path.join(root, "test")
    items = []
    with open(list_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rel, pid = line.rsplit(" ", 1)
            cam = int(os.path.basename(rel).split("_")[2]) - 1
            items.append((os.path.join(img_dir, rel), int(pid), cam))
    return _subsample(items, limit)


DATASETS = {
    "market1501": (load_market, "Market-1501-v15.09.15"),
    "msmt17": (load_msmt17, "MSMT17_V1"),
}


def embed_all(embedder, items, batch=128, workers=8):
    """Embed a list of pre-cropped images, batched.

    These images ARE the crops, so embed_images() is used rather than the
    frame+boxes path -- the latter would run one forward pass per image and
    discard the batching that makes this tractable.
    """
    feats = np.zeros((len(items), embedder.dim), np.float32)
    paths = [it[0] for it in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            imgs = list(pool.map(cv2.imread, chunk))
            ok = [(k, im) for k, im in enumerate(imgs) if im is not None]
            if not ok:
                continue
            out = embedder.embed_images([im for _k, im in ok])
            for row, (k, _im) in enumerate(ok):
                feats[start + k] = out[row]
    return feats


def evaluate(qf, q_pids, q_cams, gf, g_pids, g_cams, topk=(1, 5, 10)):
    """Standard Market-1501 single-query CMC and mAP."""
    n_q = len(qf)
    sim = qf @ gf.T                      # cosine, both L2-normalised
    indices = np.argsort(-sim, axis=1)

    all_cmc, all_ap, valid = [], [], 0
    max_rank = max(topk)

    for i in range(n_q):
        order = indices[i]
        # Discard the trivial same-person-same-camera matches and distractors.
        remove = ((g_pids[order] == q_pids[i]) & (g_cams[order] == q_cams[i])) \
            | (g_pids[order] == -1)
        keep = ~remove

        matches = (g_pids[order] == q_pids[i])[keep].astype(np.int32)
        if not matches.any():
            # No cross-camera instance of this person exists in the gallery.
            continue
        valid += 1

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        n_rel = matches.sum()
        tmp = matches.cumsum()
        precision = [tmp[k] / (k + 1.0) for k in range(len(tmp)) if matches[k]]
        all_ap.append(np.mean(precision))

    if not valid:
        return None
    cmc = np.stack(all_cmc).mean(axis=0)
    return {
        **{f"rank{k}": round(float(cmc[k - 1]), 4) for k in topk},
        "mAP": round(float(np.mean(all_ap)), 4),
        "n_query_valid": valid,
    }


def build_candidates():
    cands = []

    def add(fn, label):
        try:
            cands.append((label, fn()))
        except Exception as exc:
            print(f"  skip {label}: {type(exc).__name__}: {str(exc)[:90]}")

    for tag in ("msmt17", "market1501", "duke"):
        p = os.path.join(REID_WEIGHTS_DIR, f"osnet_x1_0_{tag}.pth")
        if os.path.isfile(p):
            add(lambda p=p: TorchreidEmbedder("osnet_x1_0", p, (256, 128),
                                              use_cuda_graph=False),
                f"OSNet x1_0 (re-ID: {tag})")
    add(lambda: TorchreidEmbedder("osnet_x1_0", "", (256, 128),
                                  use_cuda_graph=False),
        "OSNet x1_0 (ImageNet)")
    add(lambda: TorchvisionEmbedder("mobilenet_v2", (224, 224)),
        "MobileNetV2 (ImageNet)")
    add(lambda: TorchvisionEmbedder("mobilenet_v3_large", (224, 224)),
        "MobileNetV3-L (ImageNet)")
    add(lambda: TorchvisionEmbedder("mobilenet_v3_small", (224, 224)),
        "MobileNetV3-S (ImageNet)")
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="market1501")
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit-query", type=int, default=0)
    ap.add_argument("--limit-gallery", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loader, default_dir = DATASETS[args.dataset]
    root = args.data or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reid", default_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"{args.dataset} not found at {root}")

    print(f"dataset: {args.dataset}  ({root})")
    query = loader(root, "query", args.limit_query)
    gallery = loader(root, "gallery", args.limit_gallery)
    q_pids = np.array([p for _, p, _ in query])
    q_cams = np.array([c for _, _, c in query])
    g_pids = np.array([p for _, p, _ in gallery])
    g_cams = np.array([c for _, _, c in gallery])

    print(f"query   {len(query):6d} images, {len(set(q_pids.tolist())):4d} identities")
    print(f"gallery {len(gallery):6d} images, {len(set(g_pids.tolist())):4d} identities "
          f"({int((g_pids == -1).sum())} distractors)\n")

    results = {}
    for label, emb in build_candidates():
        print(f"evaluating {label} ...", flush=True)
        t0 = time.perf_counter()
        qf = embed_all(emb, query)
        gf = embed_all(emb, gallery)
        t_embed = time.perf_counter() - t0

        m = evaluate(qf, q_pids, q_cams, gf, g_pids, g_cams)
        if m is None:
            print("   no valid queries")
            continue
        m["dim"] = int(qf.shape[1])
        m["embed_seconds"] = round(t_embed, 1)
        m["ms_per_image"] = round(t_embed / (len(query) + len(gallery)) * 1000, 3)
        results[label] = m
        print(f"   rank-1 {m['rank1'] * 100:5.1f}%   mAP {m['mAP'] * 100:5.1f}%   "
              f"({t_embed:.0f}s)")

        del emb
        torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print(f"{'embedder':32s}{'dim':>5s}{'rank-1':>9s}{'rank-5':>9s}"
          f"{'rank-10':>9s}{'mAP':>9s}{'ms/img':>9s}")
    print("-" * 92)
    for label, m in sorted(results.items(), key=lambda kv: -kv[1]["mAP"]):
        print(f"{label:32s}{m['dim']:5d}{m['rank1'] * 100:8.1f}%"
              f"{m['rank5'] * 100:8.1f}%{m['rank10'] * 100:8.1f}%"
              f"{m['mAP'] * 100:8.1f}%{m['ms_per_image']:9.2f}")
    print("=" * 92)
    ref = {"market1501": "OSNet x1_0 (Market-1501 weights): rank-1 94.2 / mAP 82.6",
           "msmt17": "OSNet x1_0 (MSMT17 weights): rank-1 74.9 / mAP 43.8"}
    print("Published reference (deep-person-reid MODEL_ZOO):")
    print(f"  {ref[args.dataset]}")
    print("\nRows whose weights match this dataset are IN-DOMAIN and therefore\n"
          "flattered. The cross-domain rows are what predict behaviour on an\n"
          "unseen scene, which is the situation this pipeline is actually in.")

    if results:
        best = max(results.values(), key=lambda m: m["mAP"])
        print(
            "\nDIAGNOSTIC\n"
            f"  Best mAP here: {best['mAP'] * 100:.1f}%.\n"
            "  Separability on your own footage: 0.114 (unusable).\n"
            "  If a model scores well here and badly there, the embedding\n"
            "  model is sound and the constraint is your scene, camera or\n"
            "  source video -- so recording clean footage will help.\n"
            "  If it scores badly in BOTH places, the model choice is wrong\n"
            "  and more footage will not fix it."
        )

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"reid_{args.dataset}_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
