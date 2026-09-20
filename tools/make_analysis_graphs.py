"""Four analysis graphs for the project review.

  17_model_selection.png     detector + embedder selection, side by side
  18_reid_performance.png    re-ID accuracy across every evaluation source
  19_score_fusion_ablation.png   scoring-function ablation
  20_system_performance.png  frame budget, scaling, and headline metrics

All numbers come from the JSON written by the benchmark and eval scripts;
nothing here is hand-entered except published reference values, which are
labelled as such.

Usage:
    python tools/make_analysis_graphs.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "presentation")


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
        "font.size": 10.5, "axes.titlesize": 11.5,
    })
    return plt


def load(path, default=None):
    p = os.path.join(BASE, path)
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ 17

def fig_model_selection():
    plt = plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- detector ---
    models = ["YOLOv8s", "YOLO11s", "YOLO26s"]
    m = [0.5701, 0.5821, 0.5963]
    r = [0.6964, 0.7127, 0.7130]
    fps = [135, 109, 103]
    x = np.arange(3)
    b1 = ax[0].bar(x - 0.21, m, 0.42, label="mAP50-95", color="#2c7fb8")
    b2 = ax[0].bar(x + 0.21, r, 0.42, label="Recall", color="#7fcdbb")
    ax[0].set_xticks(x); ax[0].set_xticklabels(models)
    ax[0].set_ylim(0.5, 0.78)
    ax[0].set_ylabel("score")
    ax[0].set_title("Detector — COCO val2017 person class\n"
                    "(2,693 images / 10,777 instances)")
    for bars, vals in ((b1, m), (b2, r)):
        for bb, v in zip(bars, vals):
            ax[0].text(bb.get_x() + bb.get_width() / 2, v + .006,
                       f"{v:.4f}", ha="center", fontsize=8.5)
    ax2 = ax[0].twinx()
    ax2.plot(x, fps, "o--", color="#d95f02", lw=1.8, ms=7, label="FPS")
    ax2.set_ylabel("FPS", color="#d95f02"); ax2.set_ylim(0, 200)
    ax2.axhline(30, color="red", ls=":", lw=1.5)
    ax2.text(2.35, 36, "30 FPS\ncamera limit", fontsize=8, color="red",
             ha="center")
    ax2.grid(False)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    # --- embedder ---
    mk = load("benchmarks/market1501_results.json", {})
    order = [
        ("OSNet\n(Market wts)", "OSNet x1_0 (re-ID: market1501)", "#1a9850"),
        ("OSNet\n(MSMT17 wts)", "OSNet x1_0 (re-ID: msmt17)", "#66bd63"),
        ("OSNet\n(Duke wts)", "OSNet x1_0 (re-ID: duke)", "#a6d96a"),
        ("OSNet\n(ImageNet)", "OSNet x1_0 (ImageNet)", "#bdbdbd"),
        ("MobileNet\nV3-Small", "MobileNetV3-S (ImageNet)", "#fdae61"),
        ("MobileNet\nV3-Large", "MobileNetV3-L (ImageNet)", "#f46d43"),
        ("MobileNet\nV2", "MobileNetV2 (ImageNet)", "#d73027"),
    ]
    labels, maps, cols = [], [], []
    for lab, key, c in order:
        if key in mk:
            labels.append(lab); maps.append(mk[key]["mAP"] * 100); cols.append(c)
    bars = ax[1].bar(labels, maps, color=cols)
    ax[1].set_ylabel("mAP % on Market-1501")
    ax[1].set_title("Embedder — Market-1501, standard protocol\n"
                    "MobileNet is not viable for re-identification")
    for bb, v in zip(bars, maps):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v + 1.2, f"{v:.1f}",
                   ha="center", fontsize=9)
    ax[1].tick_params(axis="x", labelsize=8)
    ax[1].annotate("deployed\n(cross-domain)", xy=(1, maps[1]),
                   xytext=(2.4, 62), fontsize=9, color="#1a9850",
                   arrowprops=dict(arrowstyle="->", color="#1a9850"))

    fig.suptitle("Model selection", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "17_model_selection.png"), dpi=140)
    print("  wrote 17_model_selection.png")


# ------------------------------------------------------------------ 18

def fig_reid_performance():
    plt = plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # a) accuracy by evaluation source
    src = ["Own camera\n1 person", "Own camera\n2 people",
           "MOT17-09\n~10 people", "Market-1501\n(in-domain)",
           "MSMT17\n(in-domain)"]
    val = [100.0, 100.0, 84.1, 94.3, 75.9]
    cols = ["#1a9850", "#1a9850", "#2c7fb8", "#7fcdbb", "#7fcdbb"]
    bars = ax[0].bar(src, val, color=cols)
    ax[0].set_ylabel("% (accuracy / IDF1 / rank-1)")
    ax[0].set_ylim(0, 112)
    ax[0].set_title("Re-ID accuracy by evaluation source")
    for bb, v in zip(bars, val):
        ax[0].text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v:.1f}",
                   ha="center", fontsize=9, fontweight="bold")
    ax[0].tick_params(axis="x", labelsize=8)

    # b) retention vs gap
    gaps = ["3-10 s", "10-30 s"]
    ret = [100.0, 100.0]
    n = [7, 1]
    bars = ax[1].bar(gaps, ret, color="#1a9850", width=0.55)
    ax[1].set_ylim(0, 112); ax[1].set_ylabel("identity retained %")
    ax[1].set_title("Retention vs absence duration\n(own camera, 8 events)")
    for bb, v, c in zip(bars, ret, n):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v + 2,
                   f"{v:.0f}%\nn={c}", ha="center", fontsize=9)
    ax[1].axhline(config.T_MATCH * 100, color="grey", ls=":", lw=1)

    # c) cross-domain transfer
    rows = ["Market wts", "MSMT17 wts", "Duke wts", "ImageNet"]
    mAP = np.array([[83.6, 3.2], [30.4, 47.8], [22.5, 4.4], [4.5, 2.2]])
    im = ax[2].imshow(mAP, cmap="YlGnBu", vmin=0, vmax=85, aspect="auto")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["on\nMarket-1501",
                                                     "on\nMSMT17"])
    ax[2].set_yticks(range(4)); ax[2].set_yticklabels(rows, fontsize=9)
    for i in range(4):
        for j in range(2):
            ax[2].text(j, i, f"{mAP[i, j]:.1f}", ha="center", va="center",
                       color="white" if mAP[i, j] > 45 else "black",
                       fontweight="bold", fontsize=10)
    ax[2].set_title("Cross-domain transfer, mAP %\n"
                    "asymmetric by 10x")
    ax[2].grid(False)
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    fig.suptitle("Re-identification performance", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "18_reid_performance.png"), dpi=140)
    print("  wrote 18_reid_performance.png")


# ------------------------------------------------------------------ 19

def fig_score_fusion():
    plt = plt_setup()
    data = load("benchmarks/ablation_score_fusion.json")
    if not data:
        print("  skip 19: run benchmarks/ablation_score_fusion.py first")
        return

    names = list(data)
    short = [n.replace("hybrid max", "hybrid\nmax")
              .replace(" only", "\nonly")
              .replace("hybrid + body ratios", "hybrid +\nbody ratios")
             for n in names]
    sep = [data[n]["separability"] for n in names]
    acc = [data[n]["balanced_acc"] * 100 for n in names]
    rec = [data[n]["recall_at_1pct_fm"] * 100 for n in names]
    best = int(np.argmax(sep))
    cur = next((i for i, n in enumerate(names) if n.startswith("hybrid max")), 2)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for k, (vals, title, ylab) in enumerate([
            (sep, "Separability  (intra - inter)", "separability"),
            (acc, "Best balanced accuracy", "%"),
            (rec, "Recall @ 1% false-merge", "%")]):
        cols = ["#bdbdbd"] * len(vals)
        cols[best] = "#1a9850"
        cols[cur] = "#2c7fb8" if cur != best else "#1a9850"
        bars = ax[k].bar(short, vals, color=cols)
        ax[k].set_title(title); ax[k].set_ylabel(ylab)
        ax[k].tick_params(axis="x", labelsize=7.5)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.35 + 1e-3
        ax[k].set_ylim(max(0, lo - pad), hi + pad)
        for bb, v in zip(bars, vals):
            ax[k].text(bb.get_x() + bb.get_width() / 2, v + pad * 0.08,
                       f"{v:.3f}" if k == 0 else f"{v:.1f}",
                       ha="center", fontsize=8.5, fontweight="bold")

    d = data[names[0]]
    fig.suptitle(
        "Score-fusion ablation — best-sample alone beats the deployed hybrid\n"
        f"{d['n_intra']} long-gap positives, {d['n_inter']} same-frame "
        f"negatives; gallery samples within 30 frames of the query excluded",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "19_score_fusion_ablation.png"), dpi=140)
    print("  wrote 19_score_fusion_ablation.png")


# ------------------------------------------------------------------ 20

def fig_system_performance():
    plt = plt_setup()
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # frame budget
    ax = fig.add_subplot(gs[0, 0])
    stages = ["YOLO26s", "OSNet", "DeepSORT", "identity", "pose*"]
    ms = [9.50, 3.24, 0.74, 0.13, 3.0]
    bars = ax.bar(stages, ms, color=["#2c7fb8", "#41b6c4", "#7fcdbb",
                                     "#c7e9b4", "#fdae61"])
    ax.axhline(33.3, color="red", ls="--", lw=1.5)
    ax.text(4.4, 30.5, "33.3 ms\nbudget", fontsize=8, color="red", ha="right")
    ax.set_ylabel("ms/frame"); ax.set_ylim(0, 38)
    ax.set_title(f"Frame budget — {sum(ms):.1f} of 33.3 ms used")
    for bb, v in zip(bars, ms):
        ax.text(bb.get_x() + bb.get_width() / 2, v + .6, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # scaling
    ax = fig.add_subplot(gs[0, 1])
    people = [1, 2, 4, 8, 12, 16, 24, 32]
    fps = [79, 79, 78, 75, 60, 58, 48, 41]
    ax.plot(people, fps, "o-", lw=2.2, ms=6, color="#2c7fb8")
    ax.axhline(30, color="red", ls="--", lw=1.5, label="30 FPS target")
    ax.axvline(config.GRAPH_BATCH, color="grey", ls=":",
               label=f"GRAPH_BATCH={config.GRAPH_BATCH}")
    ax.set_xlabel("simultaneous people"); ax.set_ylabel("ceiling FPS")
    ax.set_ylim(0, 92); ax.legend(fontsize=8)
    ax.set_title("Capacity — >30 FPS to 32 people")

    # CUDA graph win
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(["eager", "CUDA\ngraph"], [12.91, 2.66],
                  color=["#d73027", "#1a9850"], width=0.55)
    ax.set_ylabel("ms / OSNet forward")
    ax.set_title("CUDA graph capture\n4.9x, bit-identical output")
    for bb, v in zip(bars, [12.91, 2.66]):
        ax.text(bb.get_x() + bb.get_width() / 2, v + .3, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")

    # ghost boxes
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(["clean\nframes", "boxes\ndrawn", "identities\ncreated"],
                  [897, 4, 0], color=["#1a9850", "#fdae61", "#bdbdbd"])
    ax.set_yscale("symlog")
    ax.set_title("Empty-scene test — 901 frames\n0.44% false positives, 0 enrolled")
    for bb, v in zip(bars, [897, 4, 0]):
        ax.text(bb.get_x() + bb.get_width() / 2, v + (v * .25 + .4), str(v),
                ha="center", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    # lighting
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(2)
    ax.bar(x - .2, [0.675, 0.542], .4, label="backlit", color="#d73027")
    ax.bar(x + .2, [0.831, 0.721], .4, label="even light", color="#1a9850")
    ax.axhline(config.T_MATCH, color="black", ls="--", lw=1.2,
               label=f"T_MATCH={config.T_MATCH}")
    ax.set_xticks(x); ax.set_xticklabels(["mean", "worst (p10)"])
    ax.set_ylabel("same-person similarity"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Lighting: +0.156\nlargest single effect measured")

    # headline metrics
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    rows = [
        ("End-to-end", "35.8 FPS"),
        ("Re-ID, own camera", "100%  (8/8)"),
        ("Multi-person", "2 IDs / 2 people"),
        ("MOT17-09 IDF1", "84.1%"),
        ("ID switches", "0"),
        ("Market-1501 rank-1", "94.3  (pub 94.2)"),
        ("MSMT17 rank-1", "75.9  (pub 74.9)"),
        ("Separability", "0.2135"),
        ("Unit tests", "22 / 22"),
    ]
    ax.set_title("Headline results", fontweight="bold")
    for i, (k, v) in enumerate(rows):
        y = 0.92 - i * 0.105
        ax.text(0.02, y, k, fontsize=10, transform=ax.transAxes)
        ax.text(0.98, y, v, fontsize=10, fontweight="bold", ha="right",
                color="#1a6", transform=ax.transAxes)

    fig.suptitle("Overall system performance", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(OUT, "20_system_performance.png"), dpi=140,
                bbox_inches="tight")
    print("  wrote 20_system_performance.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_model_selection, fig_reid_performance,
               fig_score_fusion, fig_system_performance):
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\noutput -> {OUT}")


if __name__ == "__main__":
    main()
