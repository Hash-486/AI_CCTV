"""Compare appearance embedders for person re-identification.

Covers the MobileNetV2 vs MobileNetV3 question, and situates both against
OSNet, which is what the pipeline actually uses.

What is being compared, and why it is not an apples-to-apples list:

  MobileNetV2 (deep_sort_realtime)  ImageNet classification features from
      the 1280-d bottleneck, 224x224. This is what the base project ran
      inside DeepSORT.
  MobileNetV3-Large (torchvision)   ImageNet again, 960-d penultimate
      features. torchreid ships NO MobileNetV3, so this is a custom
      wrapper -- which is itself part of the answer to "should we use V3".
  MobileNetV2 (torchreid)           the same architecture trained with a
      re-ID objective at 256x128. Isolates the effect of TRAINING from
      the effect of ARCHITECTURE.
  OSNet x1_0 / x0_75 / x0_5         omni-scale re-ID architecture.

The MobileNetV2-ImageNet vs MobileNetV2-torchreid pair is the informative
one. Same architecture, same capacity, different training objective. Any
gap between them is attributable to the objective alone:

  ImageNet optimises inter-CLASS separation   -- person vs car
  re-ID optimises intra-class, inter-INSTANCE -- person A vs person B

Only the second is this task.

Metrics
-------
separability  intra-person minus inter-person mean cosine. The headline
              number: a large gap means a threshold exists that works.
overlap       fraction of different-person pairs scoring above the 5th
              percentile of same-person pairs. This IS the irreducible
              error floor -- no threshold choice can beat it.
rank-1        for each query, is the nearest neighbour the same person?
ms/crop       latency at batch 1, 4 and 8.

Usage:
  python benchmarks/bench_embedders.py --clips <dir>
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TORCHREID_PATH, REID_WEIGHTS_DIR  # noqa: E402
from detector import PersonDetector  # noqa: E402
from embedder import OSNetEmbedder  # noqa: E402

if TORCHREID_PATH not in sys.path and os.path.isdir(TORCHREID_PATH):
    sys.path.insert(0, TORCHREID_PATH)


# ---------------------------------------------------------------- wrappers

class TorchvisionEmbedder:
    """MobileNetV2 / V3 from torchvision with the classifier removed."""

    def __init__(self, arch="mobilenet_v3_large", size=(224, 224), device=None):
        import torchvision.models as tvm

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.h, self.w = size
        weights = "DEFAULT"
        net = getattr(tvm, arch)(weights=weights)
        # Keep the feature extractor and pooling, drop the classifier head.
        self.backbone = nn.Sequential(
            net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()
        ).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            self.dim = int(self.backbone(
                torch.zeros(1, 3, self.h, self.w, device=self.device)
            ).shape[1])
        self.name = f"{arch}(imagenet)"

    def embed(self, frame, boxes):
        n = len(boxes)
        feats = np.zeros((n, self.dim), np.float32)
        valid = np.zeros((n,), bool)
        crops, idx = [], []
        fh, fw = frame.shape[:2]
        for i, b in enumerate(boxes):
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(fw, int(b[2])), min(fh, int(b[3]))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            crops.append(cv2.resize(frame[y1:y2, x1:x2], (self.w, self.h)))
            idx.append(i)
        if not crops:
            return feats, valid
        out = self._embed_resized(crops)
        for r, i in enumerate(idx):
            feats[i] = out[r]
            valid[i] = True
        return feats, valid

    def _embed_resized(self, crops):
        batch = np.stack(crops)[:, :, :, ::-1]
        t = torch.from_numpy(
            np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        ).to(self.device).float().div_(255.0).sub_(self.mean).div_(self.std)
        with torch.no_grad():
            out = F.normalize(self.backbone(t).float(), p=2, dim=1)
        return out.cpu().numpy().astype(np.float32)

    def embed_images(self, images):
        """Embed whole pre-cropped images in one batched pass."""
        if not images:
            return np.zeros((0, self.dim), np.float32)
        crops = [cv2.resize(im, (self.w, self.h)) for im in images]
        return self._embed_resized(crops)


class TorchreidEmbedder(OSNetEmbedder):
    """Any torchreid architecture, reusing OSNetEmbedder's fast path."""

    def __init__(self, model_name, model_path="", image_size=(256, 128), **kw):
        super().__init__(model_name=model_name, model_path=model_path,
                         image_size=image_size, **kw)
        tag = os.path.basename(model_path).replace(".pth", "") \
            if model_path else "imagenet"
        self.name = f"{model_name}({tag})"


def build_candidates():
    cands = []

    def add(fn, label):
        try:
            e = fn()
            cands.append((label, e))
        except Exception as exc:
            print(f"  skip {label}: {type(exc).__name__}: {str(exc)[:90]}")

    add(lambda: TorchvisionEmbedder("mobilenet_v2", (224, 224)),
        "MobileNetV2 (ImageNet, 224)")
    add(lambda: TorchvisionEmbedder("mobilenet_v3_large", (224, 224)),
        "MobileNetV3-L (ImageNet, 224)")
    add(lambda: TorchvisionEmbedder("mobilenet_v3_small", (224, 224)),
        "MobileNetV3-S (ImageNet, 224)")
    add(lambda: TorchreidEmbedder("mobilenetv2_x1_0", "", (256, 128),
                                  use_cuda_graph=False),
        "MobileNetV2 (torchreid, 256x128)")
    add(lambda: TorchreidEmbedder("osnet_x1_0", "", (256, 128),
                                  use_cuda_graph=False),
        "OSNet x1_0 (ImageNet)")
    for tag in ("msmt17", "market1501", "duke"):
        p = os.path.join(REID_WEIGHTS_DIR, f"osnet_x1_0_{tag}.pth")
        if os.path.isfile(p):
            add(lambda p=p: TorchreidEmbedder("osnet_x1_0", p, (256, 128),
                                              use_cuda_graph=False),
                f"OSNet x1_0 (re-ID: {tag})")
    for sub in ("osnet_x0_75", "osnet_x0_5"):
        add(lambda s=sub: TorchreidEmbedder(s, "", (256, 128),
                                            use_cuda_graph=False),
            f"{sub} (ImageNet)")
    return cands


# ---------------------------------------------------------------- data

def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def collect_crops(clips, stride=3, min_h=64, min_conf=0.5, min_len=3):
    """Pseudo-labelled crops: {pseudo_id: [(frame, box, frame_key), ...]}.

    frame_key identifies the exact source frame, and it is load-bearing.

    Pseudo-identities come from IoU chaining, which FRAGMENTS one person
    into several ids whenever the chain breaks (occlusion, a missed
    detection, someone crossing). Treating every cross-id pair as
    "different people" therefore mislabels every same-person fragment pair
    as inter-person -- and a model that correctly scores those pairs high
    gets punished for being right, systematically favouring the WORSE
    embedder.

    frame_key lets evaluate() restrict inter-person pairs to crops from
    the same frame, which are genuinely different people. That is the only
    negative label this pipeline can produce without manual annotation.
    """
    det = PersonDetector()
    dataset, offset = {}, 0
    for clip_i, path in enumerate(clips):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        tracks, nxt, prev, idx = {}, 0, [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                _d, boxes, confs = det.detect(f)
                keep = [(b, c) for b, c in zip(boxes, confs)
                        if c >= min_conf and (b[3] - b[1]) >= min_h]
                cur = []
                for b, _c in keep:
                    best, best_iou = None, 0.0
                    for pb, pid in prev:
                        v = box_iou(b, pb)
                        if v > best_iou:
                            best_iou, best = v, pid
                    if best_iou < 0.5:
                        best = nxt
                        nxt += 1
                    tracks.setdefault(best, []).append(
                        (f.copy(), b, (clip_i, idx))
                    )
                    cur.append((b, best))
                prev = cur
            idx += 1
        cap.release()
        for k, v in tracks.items():
            if len(v) >= min_len:
                dataset[offset + k] = v
        offset += nxt
    return dataset


# ---------------------------------------------------------------- metrics

def evaluate(embedder, dataset):
    feats, labels, frame_keys = [], [], []
    for pid, items in dataset.items():
        for frame, box, fkey in items:
            f, v = embedder.embed(frame, np.array([box], np.float32))
            if v[0]:
                feats.append(f[0])
                labels.append(pid)
                frame_keys.append(fkey)
    if len(feats) < 10:
        return None

    X = np.stack(feats)
    norms = np.linalg.norm(X, axis=1)
    if float(np.median(norms)) < 0.5:
        # A collapsed network. torchreid's mobilenetv2_x1_0 does this: its
        # activations die to ~1e-9, so every cosine is meaningless. Report
        # it as broken rather than as a legitimately poor score.
        return {"broken": True,
                "reason": f"degenerate embeddings (median norm "
                          f"{float(np.median(norms)):.2e})"}

    y = np.array(labels)
    n = len(X)
    S = X @ X.T
    iu = np.triu_indices(n, k=1)
    same_id = y[iu[0]] == y[iu[1]]

    # Same frame => genuinely different people. This is the ONLY reliable
    # negative label available without manual annotation, because IoU
    # chaining fragments one person across several pseudo-ids.
    fk = np.array([hash(k) for k in frame_keys])
    same_frame = fk[iu[0]] == fk[iu[1]]

    intra = S[iu][same_id]
    inter = S[iu][(~same_id) & same_frame]
    if intra.size < 5 or inter.size < 5:
        return None

    # Rank-1 is deliberately NOT reported here, because no valid version of
    # it exists on this data:
    #   - unrestricted, the nearest neighbour of any crop is the adjacent
    #     frame of its own track, so every model scores ~92% and the metric
    #     cannot discriminate;
    #   - restricted to the same frame, each pseudo-track contributes
    #     exactly one crop per frame, so the nearest same-frame neighbour is
    #     ALWAYS a different identity and every model scores ~0.
    # A meaningful rank-1 needs queries and gallery entries of the same
    # person at different times, which requires real identity labels.
    # eval/eval_reid.py computes it from annotated clips.

    p5 = float(np.percentile(intra, 5))
    best_acc = max(
        ((intra >= t).mean() + (inter < t).mean()) / 2.0
        for t in np.linspace(0, 1, 401)
    )
    return {
        "broken": False,
        "dim": int(X.shape[1]),
        "n_samples": n,
        "n_identities": int(len(set(y.tolist()))),
        "n_intra_pairs": int(intra.size),
        "n_inter_pairs_same_frame": int(inter.size),
        "intra_mean": round(float(intra.mean()), 4),
        "inter_mean": round(float(inter.mean()), 4),
        "separability": round(float(intra.mean() - inter.mean()), 4),
        "overlap": round(float((inter >= p5).mean()), 4),
        "best_balanced_acc": round(float(best_acc), 4),
    }


def latency(embedder, sizes=(1, 4, 8), iters=30, warm=10):
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    out = {}
    for b in sizes:
        boxes = np.array([[40 + i * 5, 30, 140 + i * 5, 430]
                          for i in range(b)], np.float32)
        for _ in range(warm):
            embedder.embed(frame, boxes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            embedder.embed(frame, boxes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out[f"batch{b}_ms"] = round((time.perf_counter() - t) / iters * 1000, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = []
    for c in args.clips:
        if os.path.isdir(c):
            for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
                paths.extend(sorted(glob.glob(os.path.join(c, ext))))
        else:
            paths.append(c)

    print("collecting pseudo-labelled crops ...")
    dataset = collect_crops(paths, stride=args.stride)
    print(f"  {len(dataset)} pseudo-identities, "
          f"{sum(len(v) for v in dataset.values())} crops\n")
    if len(dataset) < 3:
        raise SystemExit("need at least 3 distinct pseudo-identities")

    print("building candidate embedders ...")
    results = {}
    for label, emb in build_candidates():
        print(f"  evaluating {label} ...")
        m = evaluate(emb, dataset)
        if m is None:
            print("    (insufficient valid embeddings)")
            continue
        if m.get("broken"):
            print(f"    EXCLUDED: {m['reason']}")
            results[label] = m
            del emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        m.update(latency(emb))
        results[label] = m
        del emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    good = {k: v for k, v in results.items() if not v.get("broken")}
    broken = {k: v for k, v in results.items() if v.get("broken")}

    if good:
        any_m = next(iter(good.values()))
        print("\n" + "=" * 112)
        print(f"{'embedder':34s}{'dim':>5s}{'intra':>8s}{'inter':>8s}"
              f"{'sep':>8s}{'overlap':>9s}"
              f"{'b1 ms':>9s}{'b4 ms':>8s}{'b8 ms':>8s}")
        print("-" * 100)
        for label, m in sorted(good.items(),
                               key=lambda kv: -kv[1]["separability"]):
            print(f"{label:34s}{m['dim']:5d}{m['intra_mean']:8.3f}"
                  f"{m['inter_mean']:8.3f}{m['separability']:8.3f}"
                  f"{m['overlap'] * 100:8.1f}%"
                  f"{m['batch1_ms']:9.2f}{m['batch4_ms']:8.2f}"
                  f"{m['batch8_ms']:8.2f}")
        print("=" * 100)
        print(f"pairs: {any_m['n_intra_pairs']} intra, "
              f"{any_m['n_inter_pairs_same_frame']} inter (same-frame only)")
        print("sep      = intra - inter mean cosine (higher is better)")
        print("overlap  = different-person pairs above the 5th percentile of "
              "same-person pairs")
        n_neg = any_m["n_inter_pairs_same_frame"]
        if n_neg < 500:
            print(
                f"\n  WARNING: only {n_neg} negative pairs. That is far too "
                f"few to rank these\n"
                "  embedders -- differences of this size are not "
                "distinguishable from noise.\n"
                "  Treat the ordering as UNDECIDED until this is run on "
                "footage with more\n  simultaneous people."
            )

    for label, m in broken.items():
        print(f"\nEXCLUDED  {label}: {m['reason']}")

    print(
        "\nREAD THIS BEFORE QUOTING THESE NUMBERS\n"
        "  Identities here are PSEUDO-labels from IoU chaining, so one\n"
        "  person becomes several ids whenever the chain breaks. Negative\n"
        "  pairs are therefore restricted to same-frame crops, which is\n"
        "  correct but scarce -- and positive pairs still span only within\n"
        "  a single unbroken chain, so long-gap re-identification, the\n"
        "  case that actually matters, is under-represented.\n"
        "  Run eval/eval_reid.py on clips with real identity ground truth\n"
        "  for a defensible comparison."
    )

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "embedder_comparison.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
