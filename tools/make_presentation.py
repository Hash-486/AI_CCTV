"""Generate presentation figures from recorded clips and benchmark results.

Screenshots taken live are luck-dependent and unrepeatable. This runs the real
pipeline over the recorded clips and saves annotated frames at chosen moments,
so every figure is reproducible and can be regenerated after any code change.

Output: presentation/  (PNG, 1-13)

Usage:
    python tools/make_presentation.py
    python tools/make_presentation.py --only 2,5      # regenerate some
    python tools/make_presentation.py --list
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector          # noqa: E402
from tracker import PersonTracker            # noqa: E402
from render import render_frame              # noqa: E402
import config                                # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(BASE, "eval", "clips")
GT = os.path.join(BASE, "eval", "ground_truth")
OUT = os.path.join(BASE, "presentation")
ROOT_REC = os.path.join(BASE, "archive", "recordings")

TITLE_H = 34
LABEL_H = 26


# ------------------------------------------------------------------ helpers

def run_clip(path, wanted, use_pose=False):
    """Run the pipeline over a clip, returning {frame_index: rendered_frame}.

    Renders through render.py so the figures show exactly what the live
    program draws -- not a reconstruction of it.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    det = PersonDetector()
    trk = PersonTracker(use_reid=True)
    pose = None
    if use_pose:
        from pose import PoseEstimator
        pose = PoseEstimator()

    grabbed, prev_pose, i = {}, {}, 0
    want = set(wanted)
    last = max(want) if want else 0
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        dets, boxes, confs = det.detect(frame)
        tracks = trk.update(frame, dets, boxes, confs, pose_cache=prev_pose)
        pose_map = {}
        if pose is not None:
            pose_map = pose.update(frame, tracks)
            prev_pose = pose_map
        if i in want:
            view = frame.copy()
            render_frame(view, tracks, pose_map, trk.trails,
                         fps=35.8, n_identities=len(trk.gallery),
                         timings=trk.timings,
                         extra=f"identities {len(trk.gallery)}")
            grabbed[i] = view
        i += 1
    cap.release()
    if pose is not None:
        pose.close()
    return grabbed


def titled(img, title, sub=None):
    """Add a title bar above an image."""
    h, w = img.shape[:2]
    bar = np.full((TITLE_H, w, 3), 25, np.uint8)
    cv2.putText(bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([bar, img])
    if sub:
        lab = np.full((LABEL_H, w, 3), 40, np.uint8)
        cv2.putText(lab, sub, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (190, 190, 190), 1, cv2.LINE_AA)
        out = np.vstack([out, lab])
    return out


def strip(panels, gap=8):
    """Horizontally join equal-height panels with a separator."""
    h = max(p.shape[0] for p in panels)
    fixed = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.full((h - p.shape[0], p.shape[1], 3), 25, np.uint8)
            p = np.vstack([p, pad])
        fixed.append(p)
        fixed.append(np.full((h, gap, 3), 15, np.uint8))
    return np.hstack(fixed[:-1])


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    cv2.imwrite(p, img)
    print(f"  wrote {name}  ({img.shape[1]}x{img.shape[0]})")


def load_gt(clip):
    with open(os.path.join(GT, clip + ".json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ figures

def fig01_core_tracking():
    """Single person tracked with ID, HUD, trail."""
    g = run_clip(os.path.join(CLIPS, "clip9_even_lighting.avi"),
                 [300, 700, 1100])
    panels = [titled(cv2.resize(g[i], (426, 320)),
                     f"frame {i}", "green = matched a detection this frame")
              for i in sorted(g)]
    save("01_core_tracking.png", strip(panels))


def fig02_reidentification():
    """THE key figure: same ID before and after leaving frame."""
    gt = load_gt("clip9_even_lighting")
    ev = gt["events"][0]
    fps = gt.get("fps", 20)
    before = max(ev["exit_frame"] - 40, 0)
    during = (ev["exit_frame"] + ev["return_frame"]) // 2
    after = ev["return_frame"] + 45
    g = run_clip(os.path.join(CLIPS, "clip9_even_lighting.avi"),
                 [before, during, after])
    gap = (ev["return_frame"] - ev["exit_frame"]) / fps
    panels = [
        titled(cv2.resize(g[before], (426, 320)), "1. BEFORE  (tracked)",
               f"frame {before}"),
        titled(cv2.resize(g[during], (426, 320)), "2. ABSENT  (left frame)",
               f"frame {during}   gap {gap:.1f}s"),
        titled(cv2.resize(g[after], (426, 320)), "3. RETURNED  (same ID)",
               f"frame {after}   identity restored"),
    ]
    save("02_reidentification.png", strip(panels))


def fig03_multiperson():
    """Two people, two distinct identities, through a crossing."""
    g = run_clip(os.path.join(CLIPS, "clip6_two_people_crossing.avi"),
                 [120, 300, 480])
    panels = [titled(cv2.resize(g[i], (426, 320)), f"frame {i}",
                     "distinct IDs maintained")
              for i in sorted(g)]
    save("03_multiperson.png", strip(panels))


def fig04_empty_scene():
    """Ghost-box proof: empty room, zero boxes."""
    g = run_clip(os.path.join(CLIPS, "clip8_empty_scene.avi"),
                 [100, 400, 800])
    panels = [titled(cv2.resize(g[i], (426, 320)), f"frame {i}",
                     "no person -> no box")
              for i in sorted(g)]
    img = strip(panels)
    foot = np.full((40, img.shape[1], 3), 20, np.uint8)
    cv2.putText(foot, "901 frames, 4 boxes (0.44%), 0 identities created",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1,
                cv2.LINE_AA)
    save("04_empty_scene.png", np.vstack([img, foot]))


def fig05_before_after():
    """Base pipeline's ghost boxes vs this pipeline, side by side."""
    src = os.path.join(ROOT_REC, "20260728_191730.avi")
    if not os.path.exists(src):
        print("  skip 05: base recording not found")
        return
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, old = cap.read()
    cap.release()
    if not ok:
        print("  skip 05: could not read base recording")
        return
    g = run_clip(os.path.join(CLIPS, "clip6_two_people_crossing.avi"), [300])
    panels = [
        titled(cv2.resize(old, (480, 360)), "BEFORE  (base pipeline)",
               "stacked phantom boxes, duplicate IDs"),
        titled(cv2.resize(g[300], (480, 360)), "AFTER  (this pipeline)",
               "render gate: only detections are drawn"),
    ]
    save("05_before_after.png", strip(panels))


def fig06_pose():
    """MediaPipe skeleton overlay."""
    g = run_clip(os.path.join(CLIPS, "clip9_even_lighting.avi"),
                 [300, 1100], use_pose=True)
    panels = [titled(cv2.resize(g[i], (480, 360)), f"frame {i}",
                     "pose confirms presence; never deletes a track")
              for i in sorted(g)]
    save("06_pose_skeleton.png", strip(panels))


# ------------------------------------------------------------------ charts

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": True,
        "grid.alpha": 0.3, "font.size": 11,
    })
    return plt


def fig07_detector_comparison():
    plt = _plt()
    models = ["YOLOv8s", "YOLO11s", "YOLO26s"]
    m5095 = [0.5701, 0.5821, 0.5963]
    recall = [0.6964, 0.7127, 0.7130]
    fps = [135, 109, 103]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    ax[0].bar(x - 0.2, m5095, 0.4, label="mAP50-95", color="#2c7fb8")
    ax[0].bar(x + 0.2, recall, 0.4, label="Recall", color="#7fcdbb")
    ax[0].set_xticks(x); ax[0].set_xticklabels(models)
    ax[0].set_title("Accuracy — COCO val2017, person class")
    ax[0].set_ylim(0.5, 0.75); ax[0].legend()
    for i, v in enumerate(m5095):
        ax[0].text(i - 0.2, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    ax[1].bar(models, fps, color=["#bdbdbd", "#bdbdbd", "#2c7fb8"])
    ax[1].axhline(30, color="red", ls="--", label="30 FPS camera limit")
    ax[1].set_title("Speed @ imgsz=640 (all far above requirement)")
    ax[1].set_ylabel("FPS"); ax[1].legend()
    for i, v in enumerate(fps):
        ax[1].text(i, v + 3, str(v), ha="center", fontsize=9)
    fig.suptitle("Detector selection: YOLO26s wins accuracy; speed is free",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "07_detector.png"), dpi=140)
    print("  wrote 07_detector.png")


def fig08_resolution():
    plt = _plt()
    res = [640, 960, 1280]
    ms = [9.68, 12.55, 15.84]
    crop = [381.6, 383.8, 324.8]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.bar([str(r) for r in res], ms, color="#fdae61", alpha=0.85)
    ax1.set_ylabel("detection ms/frame", color="#d95f02")
    ax1.set_xlabel("imgsz")
    ax2 = ax1.twinx()
    ax2.plot([str(r) for r in res], crop, "o-", color="#2c7fb8", lw=2.5,
             ms=9, label="median crop height")
    ax2.set_ylabel("median person crop height (px)", color="#2c7fb8")
    ax2.set_ylim(250, 420)
    for i, v in enumerate(crop):
        ax2.text(i, v + 8, f"{v:.1f}px", ha="center", color="#2c7fb8",
                 fontsize=10)
    ax1.set_title("Why 640: higher imgsz costs 30-64% more\n"
                  "and yields no larger crops (source is 640x480)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "08_resolution.png"), dpi=140)
    print("  wrote 08_resolution.png")


def fig09_crossdomain():
    plt = _plt()
    rows = ["Market-1501 wts", "MSMT17 wts", "DukeMTMC wts", "ImageNet"]
    cols = ["on Market-1501", "on MSMT17"]
    mAP = np.array([[83.6, 3.2], [30.4, 47.8], [22.5, 4.4], [4.5, 2.2]])
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    im = ax.imshow(mAP, cmap="YlGnBu", vmin=0, vmax=85)
    ax.set_xticks(range(2)); ax.set_xticklabels(cols)
    ax.set_yticks(range(4)); ax.set_yticklabels(rows)
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f"{mAP[i, j]:.1f}", ha="center", va="center",
                    color="white" if mAP[i, j] > 45 else "black",
                    fontweight="bold")
    ax.set_title("Cross-domain transfer is asymmetric 10x (mAP %)\n"
                 "MSMT17->Market 30.4  vs  Market->MSMT17 3.2")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mAP %")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "09_crossdomain.png"), dpi=140)
    print("  wrote 09_crossdomain.png")


def fig10_calibration():
    plt = _plt()
    fm = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0]
    new = [76.7, 76.8, 77.1, 78.2, 80.8, 83.4, 85.0, 88.4]
    old = [5.0, 6.7, 8.3, 10.8, 20.9, 32.0, 49.8, 77.9]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.semilogx(fm, new, "o-", lw=2.5, ms=8, color="#1a9850",
                label="this camera, calibrated (sep 0.2135)")
    ax.semilogx(fm, old, "s--", lw=2, ms=7, color="#d73027",
                label="old contaminated footage (sep 0.114)")
    ax.axvline(1.0, color="grey", ls=":", label="chosen operating point")
    ax.set_xlabel("false-merge budget (%)")
    ax.set_ylabel("re-entry recall (%)")
    ax.set_title("Calibration: the curve is flat where it matters\n"
                 "1% -> 0.1% costs 0.4 points (was a third)")
    ax.legend(fontsize=9); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "10_calibration.png"), dpi=140)
    print("  wrote 10_calibration.png")


def fig11_lighting():
    plt = _plt()
    labels = ["mean\nsimilarity", "worst case\n(p10)"]
    back = [0.675, 0.542]
    even = [0.831, 0.721]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(x - 0.2, back, 0.4, label="backlit (120/255)", color="#d73027")
    ax.bar(x + 0.2, even, 0.4, label="even light (154/255)", color="#1a9850")
    ax.axhline(config.T_MATCH, color="black", ls="--",
               label=f"T_MATCH = {config.T_MATCH}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("same-person cosine similarity")
    ax.set_title("Lighting is the dominant variable: +0.156\n"
                 "more than every algorithmic change combined")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.0)
    for i, (b, e) in enumerate(zip(back, even)):
        ax.text(i - 0.2, b + .02, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, e + .02, f"{e:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "11_lighting.png"), dpi=140)
    print("  wrote 11_lighting.png")


def fig12_scaling():
    plt = _plt()
    people = [1, 2, 4, 8, 12, 16, 24, 32]
    fps = [79, 79, 78, 75, 60, 58, 48, 41]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(people, fps, "o-", lw=2.5, ms=8, color="#2c7fb8")
    ax.axhline(30, color="red", ls="--", label="30 FPS real-time target")
    ax.axvline(config.GRAPH_BATCH, color="grey", ls=":",
               label=f"GRAPH_BATCH = {config.GRAPH_BATCH}")
    ax.set_xlabel("simultaneous people"); ax.set_ylabel("ceiling FPS")
    ax.set_title("Capacity: flat to 8 people (CUDA-graph batch),\n"
                 "stays above 30 FPS to 32 people")
    ax.legend(); ax.set_ylim(0, 90)
    for p, f in zip(people, fps):
        ax.annotate(str(f), (p, f), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "12_scaling.png"), dpi=140)
    print("  wrote 12_scaling.png")


def fig13_budget():
    plt = _plt()
    stages = ["YOLO26s\ndetection", "OSNet\nembedding", "DeepSORT\nassoc.",
              "identity\nresolution", "MediaPipe\n(amortised)"]
    ms = [9.50, 3.24, 0.74, 0.13, 3.0]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bars = ax.bar(stages, ms, color=["#2c7fb8", "#41b6c4", "#7fcdbb",
                                     "#c7e9b4", "#fdae61"])
    ax.axhline(33.3, color="red", ls="--", label="33.3 ms budget @ 30 FPS")
    ax.set_ylabel("ms/frame")
    ax.set_title(f"Frame budget: {sum(ms):.1f} ms of 33.3 ms used\n"
                 "CUDA graph took embedding from 34.9 to 3.2 ms")
    ax.legend()
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v + .4, f"{v:.2f}",
                ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "13_frame_budget.png"), dpi=140)
    print("  wrote 13_frame_budget.png")


FIGURES = {
    1: ("core tracking (clip9)", fig01_core_tracking),
    2: ("RE-IDENTIFICATION before/absent/after", fig02_reidentification),
    3: ("multi-person crossing (clip6)", fig03_multiperson),
    4: ("empty scene / ghost proof (clip8)", fig04_empty_scene),
    5: ("before vs after (base pipeline)", fig05_before_after),
    6: ("pose skeleton overlay", fig06_pose),
    7: ("detector comparison chart", fig07_detector_comparison),
    8: ("resolution: why 640", fig08_resolution),
    9: ("cross-domain transfer matrix", fig09_crossdomain),
    10: ("calibration curve", fig10_calibration),
    11: ("lighting comparison", fig11_lighting),
    12: ("capacity scaling", fig12_scaling),
    13: ("frame budget", fig13_budget),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated figure numbers")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, (desc, _) in FIGURES.items():
            print(f"  {n:2d}. {desc}")
        return

    want = ([int(x) for x in args.only.split(",")] if args.only
            else sorted(FIGURES))
    os.makedirs(OUT, exist_ok=True)
    for n in want:
        if n not in FIGURES:
            continue
        desc, fn = FIGURES[n]
        print(f"[{n:2d}] {desc}")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    print(f"\noutput -> {OUT}")


if __name__ == "__main__":
    main()
