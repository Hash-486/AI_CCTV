# ============================================================
# alert_manager.py -- Visual Overlay, CSV Logger, Audio Alarm
# ============================================================
import csv
import os
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from config import (
    THREAT_LOG_PATH,
    THREAT_LEVELS,
    ALARM_COOLDOWN_SEC,
    ALARM_TONES,
    POSE_MIN_VISIBILITY,
    SCREENSHOT_DIR,
    RECORDING_DIR,
    LOG_DEDUP_SECONDS,
    RECORDING_SAFE_TAIL_SEC,
)
from pose_estimator import POSE_CONNECTIONS


# -------------------- Audio (Windows) --------------------
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


def _play_beep(freq: int, duration: int):
    """Play a beep in a background thread so it doesn't block the main loop."""
    if _HAS_WINSOUND:
        threading.Thread(
            target=winsound.Beep, args=(freq, duration), daemon=True
        ).start()


# -------------------- Motion Trail --------------------
_TRAIL_LEN = 30   # number of past bbox centers to draw


class AlertManager:
    """
    Handles:
      - Visual annotations (bounding boxes, skeleton wireframe, motion trails)
      - De-duplicated CSV threat logging
      - Screenshot capture on threat escalation
      - Video recording during active threat
      - Audio alarms
    """

    def __init__(self, log_path: str = THREAT_LOG_PATH):
        self.log_path = log_path
        self._last_alarm_time = 0.0
        self._flash_counter = 0

        # De-duplicated logging: (person_id, level) -> last_log_time
        self._last_log: dict = {}

        # Previous threat levels for escalation detection
        self._prev_levels: dict = {}

        # Motion trails: canonical_id -> deque of (cx, cy)
        self._trails: dict = {}

        # Video recording state
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        os.makedirs(RECORDING_DIR, exist_ok=True)
        self._video_writer: cv2.VideoWriter | None = None
        self._recording_active = False
        self._safe_since: float | None = None

        # Ensure CSV header exists
        if not os.path.isfile(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "person_id", "threat_level", "threat_name",
                    "weapons", "confidence", "bbox", "arm_raised", "approaching",
                ])

        print(f"[INIT] AlertManager ready  (log: {self.log_path})")

    # -------------------- Skeleton Wireframe --------------------
    def draw_skeleton(self, frame: np.ndarray, keypoints: dict, color: tuple):
        """Draw the 33-point pose skeleton on the frame."""
        if not keypoints:
            return

        # Draw limb connections first (lines behind dots)
        for idx_a, idx_b in POSE_CONNECTIONS:
            kp_a = keypoints.get(idx_a)
            kp_b = keypoints.get(idx_b)
            if kp_a is None or kp_b is None:
                continue
            ax, ay, av = kp_a
            bx, by, bv = kp_b
            if av < POSE_MIN_VISIBILITY or bv < POSE_MIN_VISIBILITY:
                continue
            cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

        # Draw joint dots on top
        for idx, (px, py, vis) in keypoints.items():
            if vis < POSE_MIN_VISIBILITY:
                continue
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

    # -------------------- Motion Trails --------------------
    def _update_trail(self, person_id: int, bbox_ltrb: tuple):
        l, t, r, b = bbox_ltrb
        cx = (l + r) // 2
        cy = (t + b) // 2
        if person_id not in self._trails:
            self._trails[person_id] = deque(maxlen=_TRAIL_LEN)
        self._trails[person_id].append((cx, cy))

    def draw_trail(self, frame: np.ndarray, person_id: int, color: tuple):
        """Draw a fading polyline of the person's recent positions."""
        trail = self._trails.get(person_id)
        if not trail or len(trail) < 2:
            return
        pts = list(trail)
        n = len(pts)
        for i in range(1, n):
            alpha = i / n  # 0=oldest, 1=newest
            faded = tuple(int(c * alpha) for c in color)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)

    # -------------------- Visual Overlay --------------------
    def draw_person_box(self, frame, person_id: int, bbox_ltrb: tuple, threat_info: dict):
        l, t, r, b = bbox_ltrb
        color = threat_info["color"]
        level = threat_info["level"]
        name = threat_info["name"]
        weapons = threat_info.get("weapons", [])
        arm_raised = threat_info.get("arm_raised", False)
        approaching = threat_info.get("approaching", False)

        thickness = 2 if level < 3 else 3
        cv2.rectangle(frame, (l, t), (r, b), color, thickness)

        # Build label
        badges = []
        if arm_raised:
            badges.append("ARM-UP")
        if approaching:
            badges.append("APPROACH")
        badge_str = " ".join(badges)
        label = f"ID:{person_id} [{name}]" + (f" {badge_str}" if badges else "")

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (l, t - th - 10), (l + tw + 6, t), color, -1)
        cv2.putText(frame, label, (l + 3, t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if weapons:
            weapon_str = ", ".join(weapons)
            cv2.putText(frame, weapon_str, (l, b + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_weapon_box(self, frame, weapon_det: dict, associated: bool = False):
        x1, y1, x2, y2 = weapon_det["bbox"]
        label = weapon_det["label"]
        conf = weapon_det["confidence"]
        method = weapon_det.get("method", "")
        if not associated:
            color = (0, 0, 255)
        elif method == "wrist":
            color = (0, 255, 128)   # bright green = confirmed held
        else:
            color = (0, 140, 255)   # orange = bbox match
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_flashing_border(self, frame):
        self._flash_counter += 1
        if self._flash_counter % 6 < 3:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

    def draw_fps(self, frame, fps: float):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def draw_status_bar(self, frame, person_count: int, max_threat: int):
        h, w = frame.shape[:2]
        info = THREAT_LEVELS[max_threat]
        bar_color = info["color"]
        cv2.rectangle(frame, (0, h - 30), (w, h), bar_color, -1)
        status_text = f"Persons: {person_count} | Threat: {info['name']}"
        cv2.putText(frame, status_text, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # -------------------- De-duplicated CSV Logging --------------------
    def log_threat(self, person_id: int, threat_info: dict, associations: list):
        """Append a row to the CSV log, de-duplicating by (person_id, level) + time."""
        weapons_for_person = [a for a in associations if a["person_id"] == person_id]
        if not weapons_for_person:
            return

        level = threat_info["level"]
        key = (person_id, level)
        now = time.time()
        last = self._last_log.get(key, 0.0)
        if (now - last) < LOG_DEDUP_SECONDS:
            return
        self._last_log[key] = now

        weapon_labels = ", ".join(a["weapon_label"] for a in weapons_for_person)
        max_conf = max(a["weapon_confidence"] for a in weapons_for_person)
        bboxes = "; ".join(str(a["weapon_bbox"]) for a in weapons_for_person)

        try:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    person_id,
                    level,
                    threat_info["name"],
                    weapon_labels,
                    f"{max_conf:.3f}",
                    bboxes,
                    threat_info.get("arm_raised", False),
                    threat_info.get("approaching", False),
                ])
        except Exception as e:
            print(f"[WARN] CSV log write failed: {e}")

    # -------------------- Screenshot on Escalation --------------------
    def _maybe_screenshot(self, frame: np.ndarray, person_id: int, threat_info: dict):
        """Save a screenshot when threat level rises to HIGH or CRITICAL."""
        level = threat_info["level"]
        prev = self._prev_levels.get(person_id, 0)
        if level >= 2 and level > prev:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SCREENSHOT_DIR, f"{ts}_id{person_id}_L{level}.jpg")
            cv2.imwrite(fname, frame)
        self._prev_levels[person_id] = level

    # -------------------- Video Recording --------------------
    def _update_recording(self, frame: np.ndarray, max_threat: int):
        """Start/stop video recording based on threat level."""
        h, w = frame.shape[:2]

        if max_threat >= 2:
            self._safe_since = None
            if not self._recording_active:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(RECORDING_DIR, f"{ts}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self._video_writer = cv2.VideoWriter(fname, fourcc, 20.0, (w, h))
                self._recording_active = True
                print(f"[REC] Recording started: {fname}")
        else:
            if self._recording_active:
                if self._safe_since is None:
                    self._safe_since = time.time()
                elif (time.time() - self._safe_since) >= RECORDING_SAFE_TAIL_SEC:
                    self._video_writer.release()
                    self._video_writer = None
                    self._recording_active = False
                    self._safe_since = None
                    print("[REC] Recording stopped.")

        if self._recording_active and self._video_writer is not None:
            self._video_writer.write(frame)

    # -------------------- Audio Alarm --------------------
    def trigger_alarm(self, max_threat_level: int):
        if max_threat_level < 2:
            return
        now = time.time()
        if (now - self._last_alarm_time) < ALARM_COOLDOWN_SEC:
            return
        self._last_alarm_time = now
        tone = ALARM_TONES.get(max_threat_level, ALARM_TONES[2])
        _play_beep(tone["freq"], tone["duration"])

    # -------------------- Composite --------------------
    def process_frame(
        self,
        frame: np.ndarray,
        person_tracks: list,
        weapon_detections: list,
        associations: list,
        unmatched_weapons: list,
        threat_results: dict,
        fps: float,
    ):
        """
        Draw all overlays, log threats, capture screenshots/video, trigger alarms.
        Modifies `frame` in-place.
        """
        max_threat = 0

        for pt in person_tracks:
            pid = pt["canonical_id"]
            bbox = pt["bbox_ltrb"]
            keypoints = pt.get("keypoints", {})
            t_info = threat_results.get(pid, {
                "level": 0, "name": "SAFE",
                "color": THREAT_LEVELS[0]["color"],
                "weapons": [], "arm_raised": False, "approaching": False,
            })

            # Motion trail
            self._update_trail(pid, bbox)
            self.draw_trail(frame, pid, t_info["color"])

            # Skeleton wireframe
            self.draw_skeleton(frame, keypoints, t_info["color"])

            # Bounding box + badge
            self.draw_person_box(frame, pid, bbox, t_info)

            if t_info["level"] > max_threat:
                max_threat = t_info["level"]

            # Screenshot on escalation
            self._maybe_screenshot(frame, pid, t_info)

            # CSV log
            if t_info["level"] >= 1:
                self.log_threat(pid, t_info, associations)

        # Clean up trail entries for persons no longer tracked
        active_pids = {pt["canonical_id"] for pt in person_tracks}
        stale_trails = [pid for pid in self._trails if pid not in active_pids]
        for pid in stale_trails:
            del self._trails[pid]
        stale_prev = [pid for pid in self._prev_levels if pid not in active_pids]
        for pid in stale_prev:
            del self._prev_levels[pid]

        # Build a method lookup for weapon drawing
        method_map = {tuple(a["weapon_bbox"]): a.get("method", "iou") for a in associations}

        # Draw associated weapon boxes
        for assoc in associations:
            wb = tuple(assoc["weapon_bbox"])
            for wd in weapon_detections:
                if tuple(wd["bbox"]) == wb:
                    wd_with_method = dict(wd)
                    wd_with_method["method"] = method_map.get(wb, "iou")
                    self.draw_weapon_box(frame, wd_with_method, associated=True)
                    break

        # Draw unmatched weapon boxes
        for wd in unmatched_weapons:
            self.draw_weapon_box(frame, wd, associated=False)

        # Flashing border for CRITICAL
        if max_threat >= 3:
            self.draw_flashing_border(frame)

        self.draw_status_bar(frame, len(person_tracks), max_threat)
        self.draw_fps(frame, fps)

        # Video recording
        self._update_recording(frame, max_threat)

        # Audio alarm
        self.trigger_alarm(max_threat)

    def close(self):
        """Release video writer on shutdown."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
