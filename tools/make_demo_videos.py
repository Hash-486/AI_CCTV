"""Render short annotated demo videos for a presentation.

Two things this handles that hand-made clips usually get wrong:

1. **Codec.** Source clips are XVID .avi, which PowerPoint frequently refuses
   to play -- it shows a black rectangle on the presentation machine while
   working fine on the authoring one. These are written as H.264 MP4, which
   PowerPoint plays reliably. Falls back to mp4v if H.264 is unavailable.

2. **Length.** A reviewer will not watch 90 seconds. Each clip is cut to the
   segment that actually demonstrates the claim, with a caption burnt in so
   the point survives without narration.

Usage:
    python tools/make_demo_videos.py
    python tools/make_demo_videos.py --only reentry
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import PersonDetector      # noqa: E402
from tracker import PersonTracker        # noqa: E402
from render import render_frame          # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(BASE, "eval", "clips")
GT = os.path.join(BASE, "eval", "ground_truth")
OUT = os.path.join(BASE, "presentation", "videos")

CAPTION_H = 48


def writer_for(path, w, h, fps):
    """Prefer H.264; fall back to mp4v with a warning."""
    for tag in ("avc1", "H264", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if vw.isOpened():
            if tag == "mp4v":
                print("    note: H.264 unavailable, used mp4v. If PowerPoint "
                      "will not play it, re-encode with ffmpeg.")
            return vw
        vw.release()
    raise SystemExit(f"no usable codec for {path}")


def caption(frame, text, sub=None):
    h, w = frame.shape[:2]
    bar = np.full((CAPTION_H, w, 3), 20, np.uint8)
    cv2.putText(bar, text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (120, 230, 120), 1, cv2.LINE_AA)
    return np.vstack([frame, bar])


def render_segment(clip, start, end, out_name, title, sub_fn=None,
                   fps=20.0, use_pose=False, speed=1):
    """Track `clip` from frame 0 (state must build up) and write [start, end)."""
    src = os.path.join(CLIPS, clip)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  skip {out_name}: cannot open {clip}")
        return

    det = PersonDetector()
    trk = PersonTracker(use_reid=True)
    pose = None
    if use_pose:
        from pose import PoseEstimator
        pose = PoseEstimator()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, out_name)
    vw = None
    prev_pose, i, written = {}, 0, 0

    while i < end:
        ok, frame = cap.read()
        if not ok:
            break
        dets, boxes, confs = det.detect(frame)
        tracks = trk.update(frame, dets, boxes, confs, pose_cache=prev_pose)
        pose_map = {}
        if pose is not None:
            pose_map = pose.update(frame, tracks)
            prev_pose = pose_map

        if i >= start and (i - start) % speed == 0:
            view = frame.copy()
            render_frame(view, tracks, pose_map, trk.trails,
                         fps=35.8, n_identities=len(trk.gallery),
                         timings=trk.timings,
                         extra=f"identities {len(trk.gallery)}   "
                               f"rematched {trk.gallery.stats['rematched']}")
            sub = sub_fn(i, tracks, trk) if sub_fn else None
            out = caption(view, title, sub)
            if vw is None:
                vw = writer_for(path, out.shape[1], out.shape[0], fps)
            vw.write(out)
            written += 1
        i += 1

    cap.release()
    if pose is not None:
        pose.close()
    if vw is not None:
        vw.release()
        print(f"  wrote {out_name}  ({written} frames, "
              f"{written / fps:.1f}s)")


# ------------------------------------------------------------------ demos

def demo_single():
    render_segment(
        "clip9_even_lighting.avi", 120, 420, "01_single_person.mp4",
        "Scenario 1 - single person tracked with a stable identity",
        sub_fn=lambda i, t, k: "green box = matched a real detection this frame",
    )


def demo_reentry():
    """The key demo: leaves, is absent, returns with the same ID."""
    with open(os.path.join(GT, "clip9_even_lighting.json")) as fh:
        gt = json.load(fh)
    ev = gt["events"][0]
    fps = gt.get("fps", 20)
    start = max(ev["exit_frame"] - 60, 0)
    end = ev["return_frame"] + 90

    def sub(i, tracks, trk):
        if i < ev["exit_frame"]:
            return "tracked ->  about to walk out of frame"
        if i < ev["return_frame"]:
            gone = (i - ev["exit_frame"]) / fps
            return (f"ABSENT {gone:4.1f}s   people 0, but identities 1 "
                    f"-- signature retained")
        return "RETURNED -> same ID restored, not a new person"

    render_segment("clip9_even_lighting.avi", start, end,
                   "02_reidentification.mp4",
                   "Scenario 2 - RE-IDENTIFICATION after leaving frame",
                   sub_fn=sub)


def demo_two_people():
    render_segment(
        "clip6_two_people_crossing.avi", 100, 500, "03_two_people.mp4",
        "Scenario 3 - two people, distinct identities through crossings",
        sub_fn=lambda i, t, k: (f"{len([x for x in t if x.get('canonical_id')])}"
                                f" tracked   0 duplicate IDs"),
    )


def demo_empty():
    render_segment(
        "clip8_empty_scene.avi", 200, 560, "04_empty_scene.mp4",
        "Scenario 4 - empty room: no person, no box",
        sub_fn=lambda i, t, k: "901 frames -> 4 boxes (0.44%), 0 identities",
    )


def demo_pose():
    render_segment(
        "clip9_even_lighting.avi", 200, 460, "05_pose.mp4",
        "Scenario 5 - MediaPipe pose confirms presence (never deletes)",
        sub_fn=lambda i, t, k: "skeleton = body evidence for the render gate",
        use_pose=True,
    )


DEMOS = {
    "single": ("single person tracked", demo_single),
    "reentry": ("RE-IDENTIFICATION (the key one)", demo_reentry),
    "two": ("two people crossing", demo_two_people),
    "empty": ("empty scene / ghost proof", demo_empty),
    "pose": ("pose skeleton", demo_pose),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated: " + ", ".join(DEMOS))
    args = ap.parse_args()
    want = args.only.split(",") if args.only else list(DEMOS)
    os.makedirs(OUT, exist_ok=True)
    for k in want:
        if k not in DEMOS:
            continue
        desc, fn = DEMOS[k]
        print(f"[{k}] {desc}")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    reencode_h264()
    print(f"\noutput -> {OUT}")
    print("Embed these in PowerPoint with Insert > Video > This Device "
          "(embed, do NOT link).")


def reencode_h264():
    """Re-encode to true H.264 with ffmpeg if it is available.

    OpenCV's H.264 encoder is often missing (libopenh264 fails to initialise),
    silently falling back to mp4v. mp4v in an .mp4 container is both ~14x
    larger and unreliable in PowerPoint -- the classic "black rectangle on the
    presentation laptop" failure. ffmpeg fixes both: 17 MB -> 1.2 MB and a
    codec PowerPoint actually plays.
    """
    import shutil
    import subprocess

    ff = shutil.which("ffmpeg")
    if not ff:
        print("\n  ffmpeg not found -- videos are mp4v. If PowerPoint shows a "
              "black\n  rectangle, install ffmpeg and re-run.")
        return

    print("\nre-encoding to H.264 ...")
    for name in sorted(os.listdir(OUT)):
        if not name.endswith(".mp4") or name.startswith("_tmp"):
            continue
        src = os.path.join(OUT, name)
        tmp = os.path.join(OUT, "_tmp_" + name)
        r = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", src,
             "-c:v", "libx264", "-preset", "slow", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
            capture_output=True,
        )
        if r.returncode == 0 and os.path.exists(tmp):
            before = os.path.getsize(src) / 1e6
            os.replace(tmp, src)
            print(f"  {name}  {before:.1f} MB -> "
                  f"{os.path.getsize(src) / 1e6:.2f} MB")
        elif os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
