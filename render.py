# ============================================================
# render.py -- Drawing
#
# Person tracking visuals, plus weapon boxes and per-person threat
# level once a track has one (see threat/analyzer.py).
#
# Colour encodes TRACKING STATE by default, which is what you need to
# see while debugging identity and ghost behaviour:
#   green  -- matched a real detection this frame
#   amber  -- coasting on a prediction with pose confirmation
#   cyan   -- identity not yet decided (provisional)
# A person's threat level (CAUTION/HIGH/CRITICAL), when present,
# overrides the tracking-state colour -- what's holding a weapon
# matters more here than whether the box is a coasted prediction.
# ============================================================
import cv2

from config import (
    COLOR_TRACKED,
    COLOR_COASTING,
    COLOR_PROVISIONAL,
    COLOR_SKELETON,
    COLOR_TEXT,
    SHOW_SKELETON,
    SHOW_TRAIL,
    POSE_MIN_VISIBILITY,
    THREAT_LEVELS,
)
from pose import POSE_CONNECTIONS


def state_color(track):
    if track.get("provisional"):
        return COLOR_PROVISIONAL
    if track.get("coasting"):
        return COLOR_COASTING
    return COLOR_TRACKED


def draw_person(frame, track, show_conf=True, threat_level=0):
    x1, y1, x2, y2 = (int(v) for v in track["bbox"])
    color = THREAT_LEVELS[threat_level]["color"] if threat_level else state_color(track)
    thickness = 3 if threat_level >= 3 else (2 if not track.get("coasting") else 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cid = track.get("canonical_id")
    label = f"ID {cid}" if cid is not None else "ID ?"
    if threat_level:
        label += f" [{THREAT_LEVELS[threat_level]['name']}]"
    if track.get("coasting"):
        label += " (predicted)"
    if show_conf and track.get("conf"):
        label += f"  {track['conf']:.2f}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_weapon(frame, det, confirmed=True):
    """Draw a weapon detection. Confirmed = passed all verification gates;
    unconfirmed = stage-1 candidate rejected by verification (shown thin/dim
    so you can see what verification is filtering out)."""
    x1, y1, x2, y2 = (int(v) for v in det["bbox"])
    color = (0, 0, 255) if confirmed else (0, 200, 255)
    thickness = 3 if confirmed else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"{det['label'].upper()} {det['confidence']:.2f}"
    if not confirmed:
        label = f"? {label} [{det.get('reason', '')}]"
    cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2 if confirmed else 1, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, color=COLOR_SKELETON):
    if not SHOW_SKELETON or not keypoints:
        return
    for a, b in POSE_CONNECTIONS:
        pa, pb = keypoints.get(a), keypoints.get(b)
        if not pa or not pb:
            continue
        if pa[2] < POSE_MIN_VISIBILITY or pb[2] < POSE_MIN_VISIBILITY:
            continue
        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]), color, 1, cv2.LINE_AA)
    for v in keypoints.values():
        if len(v) >= 3 and v[2] >= POSE_MIN_VISIBILITY:
            cv2.circle(frame, (v[0], v[1]), 2, color, -1, cv2.LINE_AA)


def draw_trail(frame, points, color):
    """Fading motion trail, oldest point most transparent."""
    if not SHOW_TRAIL or len(points) < 2:
        return
    pts = list(points)
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame,
                 (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])),
                 c, 2, cv2.LINE_AA)


def draw_hud(frame, fps, n_people, n_identities, timings=None, extra=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 26), (30, 30, 30), -1)
    left = f"{fps:5.1f} FPS   people {n_people}   identities {n_identities}"
    cv2.putText(frame, left, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    if timings:
        right = "  ".join(f"{k.replace('_ms', '')} {v:.1f}ms"
                          for k, v in timings.items())
        (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, right, (w - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 180, 180), 1, cv2.LINE_AA)
    if extra:
        cv2.rectangle(frame, (0, h - 22), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, extra, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)


def render_frame(frame, tracks, pose_map, trails, fps, n_identities,
                 timings=None, extra=None, threat_levels=None,
                 confirmed_weapons=None, rejected_weapons=None):
    threat_levels = threat_levels or {}
    for tr in tracks:
        cid = tr.get("canonical_id")
        if cid is not None and cid in trails:
            draw_trail(frame, trails[cid], state_color(tr))
    for tr in tracks:
        draw_skeleton(frame, (pose_map or {}).get(tr["track_id"]))
        level = threat_levels.get(tr.get("canonical_id"), 0)
        draw_person(frame, tr, threat_level=level)
    for det in (rejected_weapons or []):
        draw_weapon(frame, det, confirmed=False)
    for det in (confirmed_weapons or []):
        draw_weapon(frame, det, confirmed=True)
    draw_hud(frame, fps, len(tracks), n_identities, timings, extra)
    return frame
