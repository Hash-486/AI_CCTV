# ============================================================
# main.py -- Live person re-ID + weapon detection pipeline
#
# Per frame:
#   1. YOLO26s detects people
#   2. OSNet embeds every detection in one batched pass
#   3. DeepSORT associates using those embeddings
#   4. The gallery resolves canonical identity
#   5. MediaPipe confirms bodies and refines boxes
#   6. Every WEAPON_DETECT_INTERVAL frames: weapon detection, then
#      two-stage verification (class confidence, persistence, person
#      association)
#   7. Confirmed weapons are matched to tracked people and fed into
#      per-person threat-level analysis (run every frame, so a level
#      steps back down through its hysteresis even between weapon
#      detection cycles)
#   8. Render
#
# Pose runs on the tracks produced by the CURRENT frame and its
# result is fed back on the NEXT frame. That one-frame lag is
# deliberate: pose needs boxes to attach landmarks to, and boxes
# need pose only as corroboration, so ordering them this way costs
# one frame of latency and avoids running the landmarker twice.
# ============================================================
import argparse
import os
import time

import cv2

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    OUTPUT_DIR,
    POSE_ENABLED,
    POSE_REFINE_ALPHA,
    FPS_SMOOTHING,
    WEAPON_DETECT_INTERVAL,
)
from detector import PersonDetector
from tracker import PersonTracker
from render import render_frame
from weapon.detector import WeaponDetector
from weapon.verification import TwoStageVerifier
from threat.association import associate_weapons_to_persons
from threat.analyzer import ThreatAnalyzer


def build_parser():
    p = argparse.ArgumentParser(description="Person re-identification + weapon detection pipeline")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", action="store_true", help="use the webcam (default)")
    src.add_argument("--video", type=str, help="path to a video file")
    p.add_argument("--save", type=str, default=None,
                   help="write annotated output to this path")
    p.add_argument("--no-reid", action="store_true",
                   help="disable the identity gallery; show raw track ids")
    p.add_argument("--no-pose", action="store_true",
                   help="disable MediaPipe entirely")
    p.add_argument("--no-weapon", action="store_true",
                   help="disable weapon detection, verification, and threat analysis")
    p.add_argument("--refine", action="store_true",
                   help="apply pose-based bounding box refinement")
    p.add_argument("--headless", action="store_true",
                   help="do not open a window (for benchmarking)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = no limit)")
    return p


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"could not open video: {args.video}")
        return cap, False
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {CAMERA_INDEX}")
    return cap, True


def main():
    args = build_parser().parse_args()

    detector = PersonDetector()
    tracker = PersonTracker(use_reid=not args.no_reid)

    pose = None
    if POSE_ENABLED and not args.no_pose:
        from pose import PoseEstimator
        pose = PoseEstimator()

    refine = None
    if args.refine and pose is not None:
        from pose import refine_bbox
        refine = refine_bbox

    weapon_det = verifier = analyzer = None
    if not args.no_weapon:
        weapon_det = WeaponDetector()
        verifier = TwoStageVerifier()
        analyzer = ThreatAnalyzer()

    cap, is_live = open_source(args)
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)

    prev_pose = {}
    fps = 0.0
    frames = 0
    t_start = time.perf_counter()
    confirmed_weapons, rejected_weapons = [], []

    print("running -- press q to quit, r to reset identities, "
          "p to toggle pose, t to toggle trails")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()

            dets, boxes, confs = detector.detect(frame)
            tracks = tracker.update(frame, dets, boxes, confs,
                                    pose_cache=prev_pose)

            pose_map = {}
            if pose is not None:
                pose_map = pose.update(frame, tracks)
                if refine is not None:
                    for tr in tracks:
                        kps = pose_map.get(tr["track_id"])
                        if kps:
                            tr["bbox"] = refine(tr["bbox"], kps,
                                                POSE_REFINE_ALPHA)
            prev_pose = pose_map

            threat_levels = {}
            if weapon_det is not None:
                # association.py / analyzer.py index person state by
                # canonical_id and require 'bbox_ltrb'; a track whose
                # identity isn't resolved yet has no stable id to key
                # threat state on, so it sits out of weapon association
                # for this frame rather than corrupting a shared bucket.
                person_view = [
                    {**tr, "bbox_ltrb": tr["bbox"]}
                    for tr in tracks if tr.get("canonical_id") is not None
                ]

                if frames % WEAPON_DETECT_INTERVAL == 0:
                    candidates = weapon_det.detect(frame)
                    confirmed_weapons, rejected_weapons = verifier.verify(
                        candidates, person_view
                    )

                associations, _unmatched = associate_weapons_to_persons(
                    person_view, confirmed_weapons
                )
                for pid, result in analyzer.analyze(person_view, associations).items():
                    threat_levels[pid] = result["level"]

            dt = time.perf_counter() - t0
            inst = 1.0 / max(dt, 1e-6)
            fps = inst if frames == 0 else (
                FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * inst
            )

            n_ids = len(tracker.gallery) if tracker.use_reid else 0
            stats = tracker.stats
            extra = (f"new {stats.get('assigned_new', 0)}  "
                     f"rematched {stats.get('rematched', 0)}  "
                     f"deferred {stats.get('deferred', 0)}  "
                     f"admitted {stats.get('admitted', 0)}")

            out = render_frame(frame, tracks, pose_map, tracker.trails,
                               fps, n_ids, tracker.timings, extra,
                               threat_levels=threat_levels,
                               confirmed_weapons=confirmed_weapons,
                               rejected_weapons=rejected_weapons)

            if writer is None and args.save:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (w, h)
                )
            if writer is not None:
                writer.write(out)

            if not args.headless:
                cv2.imshow("person re-identification", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and tracker.use_reid:
                    tracker.gallery.identities.clear()
                    tracker.track_to_canonical.clear()
                    print("identities reset")
                if key == ord("p") and pose is not None:
                    prev_pose = {}

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if pose is not None:
            pose.close()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\n{frames} frames in {elapsed:.2f}s -- {frames / max(elapsed, 1e-6):.1f} FPS")
    print("tracker stats:", tracker.stats)


if __name__ == "__main__":
    main()
