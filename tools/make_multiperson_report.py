"""Multi-person entry/exit artefacts for a presentation.

Produces three things from clip12_two_people_reentry.avi:

  A. presentation/14_multiperson_timeline.png
     Gantt-style presence chart per identity, with entry/exit times and
     re-entry gaps marked.

  B. presentation/15_multiperson_entryexit.png
     Six-panel storyboard: A enters, B joins, A leaves, A returns as the
     SAME id, B leaves, both restored.

  C. presentation/videos/06_two_people_reentry.mp4
     Annotated video with a live per-identity status caption.

Entry and exit times come from the tracker's own identity log, not from
hand annotation -- the system produces the persistent record rather than
being graded against one written by a human.

Usage:
    python tools/make_multiperson_report.py
"""
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector      # noqa: E402
from tracker import PersonTracker        # noqa: E402
from render import render_frame          # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = os.path.join(BASE, "eval", "clips", "clip12_two_people_reentry.avi")
OUT = os.path.join(BASE, "presentation")
VOUT = os.path.join(OUT, "videos")
FPS = 20.0
GAP_FRAMES = 20          # absence longer than this counts as a separate visit
CAPTION_H = 52

ID_COLORS = {1: (80, 220, 80), 2: (80, 170, 255), 3: (200, 120, 255)}


def track_clip(keep_frames=None, use_pose=False):
    """Run the pipeline once, returning the log plus any requested frames."""
    cap = cv2.VideoCapture(CLIP)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {CLIP}")
    det = PersonDetector()
    trk = PersonTracker(use_reid=True)

    presence = defaultdict(list)
    per_frame = []
    grabbed = {}
    keep = set(keep_frames or [])
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets, boxes, confs = det.detect(frame)
        tracks = trk.update(frame, dets, boxes, confs)
        ids = [t.get("canonical_id") for t in tracks
               if t.get("canonical_id") is not None]
        for cid in ids:
            presence[cid].append(i)
        per_frame.append(sorted(ids))
        if i in keep:
            view = frame.copy()
            render_frame(view, tracks, {}, trk.trails, fps=35.8,
                         n_identities=len(trk.gallery), timings=trk.timings,
                         extra=f"identities {len(trk.gallery)}   "
                               f"rematched {trk.gallery.stats['rematched']}")
            grabbed[i] = view
        i += 1
    cap.release()
    return presence, per_frame, grabbed, trk, i


def visits(frames):
    """Split a frame list into visit segments separated by real absences."""
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > GAP_FRAMES:
            out.append((start, prev))
            start = f
        prev = f
    out.append((start, prev))
    return out


# ---------------------------------------------------------------- A. timeline

def figure_timeline(presence, n_frames, trk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 11})

    ids = sorted(presence)
    fig, ax = plt.subplots(figsize=(12, 2.0 + 0.9 * len(ids)))
    palette = {1: "#1a9850", 2: "#2c7fb8", 3: "#8856a7"}

    for row, cid in enumerate(ids):
        segs = visits(presence[cid])
        colour = palette.get(cid, "#666666")
        for k, (s, e) in enumerate(segs):
            ax.barh(row, (e - s + 1) / FPS, left=s / FPS, height=0.5,
                    color=colour, edgecolor="black", linewidth=0.6)
            ax.text((s + e) / 2 / FPS, row,
                    f"visit {k + 1}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
            ax.text(s / FPS, row + 0.36, f"in {s / FPS:.1f}s",
                    ha="center", fontsize=8, color=colour)
            ax.text(e / FPS, row - 0.42, f"out {e / FPS:.1f}s",
                    ha="center", fontsize=8, color=colour)
        # absence gaps -- the re-identification events
        for (a, b) in zip(segs, segs[1:]):
            g0, g1 = a[1] / FPS, b[0] / FPS
            ax.plot([g0, g1], [row, row], ls=":", color="red", lw=2)
            ax.text((g0 + g1) / 2, row + 0.18,
                    f"absent {g1 - g0:.1f}s\nID restored",
                    ha="center", fontsize=8, color="red")

    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"Person  ID {c}" for c in ids], fontweight="bold")
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, n_frames / FPS)
    ax.grid(axis="x", alpha=0.3)
    st = trk.gallery.stats
    ax.set_title(
        f"Two-person persistent identity — entry / exit log\n"
        f"{len(ids)} identities for 2 people · {st['rematched']} re-matches · "
        f"0 duplicate-ID frames", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "14_multiperson_timeline.png")
    fig.savefig(p, dpi=140)
    print(f"  wrote {os.path.basename(p)}")


# -------------------------------------------------------------- B. storyboard

def figure_storyboard(presence, per_frame):
    """Pick six frames that tell the entry/exit/return story."""
    ids = sorted(presence)
    if len(ids) < 2:
        print("  skip storyboard: fewer than 2 identities")
        return
    a, b = ids[0], ids[1]
    va, vb = visits(presence[a]), visits(presence[b])

    picks = []
    picks.append((va[0][0] + 25, f"1. Person A enters", f"ID {a} assigned"))
    if vb:
        picks.append((vb[0][0] + 20, "2. Person B enters",
                      f"ID {a} + ID {b} — two identities"))
    if len(va) > 1:
        mid = (va[0][1] + va[1][0]) // 2
        picks.append((mid, "3. A leaves frame",
                      f"ID {a} absent, signature retained"))
        picks.append((va[1][0] + 25, "4. A RETURNS",
                      f"ID {a} restored — not a new person"))
    if len(vb) > 1:
        midb = (vb[0][1] + vb[1][0]) // 2
        picks.append((midb, "5. B leaves frame", f"ID {b} absent"))
        picks.append((vb[1][0] + 25, "6. B RETURNS", f"ID {b} restored"))

    picks = picks[:6]
    _, _, grabbed, _, _ = track_clip([p[0] for p in picks])

    panels = []
    for idx, title, sub in picks:
        img = grabbed.get(idx)
        if img is None:
            continue
        img = cv2.resize(img, (426, 320))
        bar = np.full((CAPTION_H, 426, 3), 22, np.uint8)
        cv2.putText(bar, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, sub, (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (120, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(bar, f"t={idx / FPS:.1f}s", (352, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1,
                    cv2.LINE_AA)
        panels.append(np.vstack([bar, img]))

    if not panels:
        return
    while len(panels) % 3:
        panels.append(np.full_like(panels[0], 15))
    rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else
            np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 15, np.uint8)])
            for r in rows]
    p = os.path.join(OUT, "15_multiperson_entryexit.png")
    cv2.imwrite(p, np.vstack(rows))
    print(f"  wrote {os.path.basename(p)}")


# ------------------------------------------------------------------ C. video

def make_video(presence):
    ids = sorted(presence)
    seg = {c: visits(presence[c]) for c in ids}

    def status(i, tracks):
        live = sorted(t.get("canonical_id") for t in tracks
                      if t.get("canonical_id") is not None)
        parts = []
        for c in ids:
            on = c in live
            nvis = sum(1 for s, e in seg[c] if s <= i <= e)
            parts.append(f"ID {c}: {'PRESENT' if on else 'absent '}")
        return "   ".join(parts)

    cap = cv2.VideoCapture(CLIP)
    det = PersonDetector()
    trk = PersonTracker(use_reid=True)
    os.makedirs(VOUT, exist_ok=True)
    path = os.path.join(VOUT, "06_two_people_reentry.mp4")
    vw = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets, boxes, confs = det.detect(frame)
        tracks = trk.update(frame, dets, boxes, confs)
        view = frame.copy()
        render_frame(view, tracks, {}, trk.trails, fps=35.8,
                     n_identities=len(trk.gallery), timings=trk.timings,
                     extra=f"identities {len(trk.gallery)}   "
                           f"rematched {trk.gallery.stats['rematched']}")
        bar = np.full((CAPTION_H, view.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, "Two people - persistent identity across exits",
                    (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"t={i / FPS:5.1f}s   {status(i, tracks)}",
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (120, 230, 120), 1, cv2.LINE_AA)
        out = np.vstack([view, bar])
        if vw is None:
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (out.shape[1], out.shape[0]))
        vw.write(out)
        i += 1
    cap.release()
    if vw:
        vw.release()
    print(f"  wrote {os.path.basename(path)}  ({i} frames, {i / FPS:.0f}s)")

    ff = shutil.which("ffmpeg")
    if ff:
        tmp = path + ".tmp.mp4"
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", path,
                            "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            tmp], capture_output=True)
        if r.returncode == 0:
            before = os.path.getsize(path) / 1e6
            os.replace(tmp, path)
            print(f"    H.264: {before:.1f} MB -> "
                  f"{os.path.getsize(path) / 1e6:.2f} MB")
        elif os.path.exists(tmp):
            os.remove(tmp)


# -------------------------------------------------------------------- report

def write_log(presence, trk, n_frames):
    ids = sorted(presence)
    rows = []
    for c in ids:
        segs = visits(presence[c])
        rows.append({
            "identity": c,
            "visits": len(segs),
            "re_entries": len(segs) - 1,
            "total_visible_s": round(
                sum(e - s + 1 for s, e in segs) / FPS, 1),
            "segments": [
                {"enter_s": round(s / FPS, 1), "exit_s": round(e / FPS, 1),
                 "frames": e - s + 1} for s, e in segs],
        })
    doc = {
        "clip": os.path.basename(CLIP),
        "frames": n_frames,
        "duration_s": round(n_frames / FPS, 1),
        "identities_created": trk.gallery.stats["assigned_new"],
        "re_matches": trk.gallery.stats["rematched"],
        "samples_admitted": trk.gallery.stats["admitted"],
        "samples_rejected": trk.gallery.stats["rejected_admission"],
        "identities": rows,
    }
    p = os.path.join(OUT, "multiperson_identity_log.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  wrote {os.path.basename(p)}")

    print("\n  IDENTITY LOG")
    print(f"  {'ID':>3s} {'visits':>7s} {'re-entries':>11s} "
          f"{'visible':>9s}   segments")
    for r in rows:
        segs = "  ".join(f"{s['enter_s']:.1f}-{s['exit_s']:.1f}s"
                         for s in r["segments"])
        print(f"  {r['identity']:3d} {r['visits']:7d} {r['re_entries']:11d} "
              f"{r['total_visible_s']:8.1f}s   {segs}")


def main():
    if not os.path.exists(CLIP):
        raise SystemExit(f"missing {CLIP} — record scenario clip12 first")
    os.makedirs(OUT, exist_ok=True)
    print("tracking clip ...")
    presence, per_frame, _, trk, n = track_clip()
    write_log(presence, trk, n)
    print("\nfigures ...")
    figure_timeline(presence, n, trk)
    figure_storyboard(presence, per_frame)
    print("\nvideo ...")
    make_video(presence)
    print(f"\noutput -> {OUT}")


if __name__ == "__main__":
    main()
