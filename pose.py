# ============================================================
# pose.py -- MediaPipe pose as confirmation and refinement
#
# Two jobs, both NON-DESTRUCTIVE:
#
#   1. Confirmation. Visible landmarks are evidence that a body is
#      really at a predicted position, which lets a track coast for
#      a frame or two through a detector miss without flickering.
#
#   2. Refinement. When landmarks are available the drawn box can be
#      tightened toward the actual body rather than the detector's
#      estimate.
#
# What it must NEVER do is delete a track. The base project used
# pose as a rejection filter: a track whose keypoints were missing
# was dropped from the output. Because an empty dict is not None,
# "pose ran and found nobody" was scored as zero visible keypoints
# and a genuinely present person was silently erased -- someone with
# their back turned, heavily occluded, or simply the sixth person in
# a scene where num_poses was 5. Absence of evidence is not evidence
# of absence, and here it produced the disappearing-box symptom that
# is the mirror image of ghosting.
# ============================================================
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    POSE_MODEL_PATH,
    POSE_DETECT_INTERVAL,
    POSE_MIN_VISIBILITY,
    POSE_NUM_POSES,
    POSE_MATCH_MAX_DIST_PX,
    POSE_REFINE_ALPHA,
)

# BlazePose 33-landmark topology.
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_WRIST, R_WRIST = 15, 16
L_ANKLE, R_ANKLE = 27, 28

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]

TORSO_LANDMARKS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def compute_body_ratios(keypoints):
    """Four scale-invariant body proportions, or None if unmeasurable.

    Ratios rather than lengths, because a person twice as close to the
    camera has twice the pixel dimensions but the same proportions.

    Returns None when the required landmarks are not visible. The base
    project returned a vector with 1.0 substituted for every unmeasurable
    ratio, which is indistinguishable from a genuine measurement of 1.0
    and quietly pulls unrelated people toward each other.
    """
    if not keypoints:
        return None

    def pt(i):
        v = keypoints.get(i)
        if v is None or len(v) < 3 or v[2] < POSE_MIN_VISIBILITY:
            return None
        return (v[0], v[1])

    ls, rs = pt(L_SHOULDER), pt(R_SHOULDER)
    lh, rh = pt(L_HIP), pt(R_HIP)
    la, ra = pt(L_ANKLE), pt(R_ANKLE)
    lw, rw = pt(L_WRIST), pt(R_WRIST)

    if not (ls and rs and lh and rh):
        return None

    shoulder_w = _dist(ls, rs)
    hip_w = _dist(lh, rh)
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    torso_len = _dist(shoulder_mid, hip_mid)

    if shoulder_w < 1e-3 or torso_len < 1e-3:
        return None

    ratios = [shoulder_w / torso_len, hip_w / shoulder_w]

    leg_lengths = [_dist(h, a) for h, a in ((lh, la), (rh, ra)) if h and a]
    ratios.append(
        (sum(leg_lengths) / len(leg_lengths)) / torso_len if leg_lengths else np.nan
    )
    ratios.append(_dist(lw, rw) / shoulder_w if (lw and rw) else np.nan)

    arr = np.array(ratios, np.float32)
    if np.isnan(arr).all():
        return None
    # Unmeasured ratios are filled with the mean of the measured ones so the
    # vector stays comparable, rather than with a fabricated constant.
    arr[np.isnan(arr)] = np.nanmean(arr)
    return arr


class PoseEstimator:
    """MediaPipe pose landmarker, matched to tracks."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=POSE_NUM_POSES,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._cache = {}     # deepsort track id -> keypoints
        self._stale = set()  # ids whose cached pose is from an earlier frame
        self._frame = 0

    # -------------------- inference --------------------

    def _run(self, frame):
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        return self.landmarker.detect(image)

    @staticmethod
    def _to_keypoints(landmarks, h, w):
        return {
            i: (int(lm.x * w), int(lm.y * h), float(lm.visibility))
            for i, lm in enumerate(landmarks)
        }

    @staticmethod
    def _torso_centre(kps):
        pts = [
            (kps[i][0], kps[i][1]) for i in TORSO_LANDMARKS
            if i in kps and kps[i][2] >= POSE_MIN_VISIBILITY
        ]
        if not pts:
            return None
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )

    # -------------------- per-frame update --------------------

    def update(self, frame, tracks):
        """Attach keypoints to tracks.

        Args:
            frame:  BGR image
            tracks: list of track dicts with track_id and bbox

        Returns:
            {deepsort_track_id: keypoints dict}

        Keyed on DEEPSORT TRACK ID, not canonical identity. The base
        project keyed this cache on canonical id, which broke twice over:
        a re-identification flip made the lookup miss, and building the
        dict by comprehension silently collapsed two tracks that had been
        given the same canonical id into one entry.
        """
        self._frame += 1
        h, w = frame.shape[:2]

        if not tracks:
            self._cache.clear()
            self._stale.clear()
            return {}

        if (self._frame % POSE_DETECT_INTERVAL) != 1 and self._cache:
            # Between inference frames, replay the cache. Marked stale so
            # consumers can weigh it accordingly.
            self._stale = set(self._cache)
            return {t["track_id"]: self._cache.get(t["track_id"], {})
                    for t in tracks}

        result = self._run(frame)
        poses = []
        for lms in (result.pose_landmarks or []):
            kps = self._to_keypoints(lms, h, w)
            centre = self._torso_centre(kps)
            if centre is not None:
                poses.append((centre, kps))

        assigned = self._match(poses, tracks)
        self._cache = assigned
        self._stale.clear()
        return assigned

    @staticmethod
    def _match(poses, tracks):
        """Pair poses to tracks by torso centroid, solved jointly.

        Hungarian rather than first-fit. The base project walked the track
        list and gave each pose to the first box containing its centroid,
        so when two people overlapped -- exactly when the assignment matters
        -- the pose went to whichever track happened to be earlier in the
        list rather than the one it belonged to.
        """
        if not poses or not tracks:
            return {}

        cost = np.full((len(poses), len(tracks)), 1e6, np.float32)
        for i, (centre, _kps) in enumerate(poses):
            for j, tr in enumerate(tracks):
                x1, y1, x2, y2 = tr["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                d = float(np.hypot(centre[0] - cx, centre[1] - cy))
                inside = (x1 <= centre[0] <= x2) and (y1 <= centre[1] <= y2)
                if inside or d <= POSE_MATCH_MAX_DIST_PX:
                    # Prefer containment, then proximity.
                    cost[i, j] = d if inside else d + POSE_MATCH_MAX_DIST_PX
        rows, cols = linear_sum_assignment(cost)

        out = {}
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            out[tracks[j]["track_id"]] = poses[i][1]
        return out

    def is_stale(self, track_id):
        return track_id in self._stale

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


def keypoint_hull(keypoints, min_visibility=POSE_MIN_VISIBILITY):
    """Axis-aligned bounding box of the confidently-visible landmarks."""
    pts = [
        (v[0], v[1]) for v in keypoints.values()
        if len(v) >= 3 and v[2] >= min_visibility
    ]
    if len(pts) < 4:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def refine_bbox(det_box, keypoints, alpha=POSE_REFINE_ALPHA):
    """Tighten the detector box using landmarks, on three edges only.

    Which edges the landmarks can actually inform follows directly from
    the BlazePose topology:

        0-10    nose, eyes, ears, mouth
        11-22   shoulders, elbows, wrists, hands
        23-28   hips, knees, ankles
        29-30   heels
        31-32   foot index (toes)

    So:

      x1, x2  informed. Shoulders, elbows, wrists and hands bound the body
              horizontally, and YOLO boxes are routinely loose here --
              they include background beside the torso. The hull is
              genuinely tighter and more accurate.

      y2      informed. Landmarks 31/32 are the toes, not the ankles, so
              the hull reaches the bottom of the body.

      y1      NOT informed, and left to the detector. The highest landmark
              is eye/ear level, so the hull top sits at roughly the
              forehead. Blending y1 toward it shaves off the crown of the
              head every frame. There is no landmark up there to blend
              toward, so the detector's estimate is the better one.

    An earlier version blended all four edges, which produced exactly that
    head-truncation. A "shrink-only" clamp does not fix it either: shrinking
    is precisely the direction the truncation happens in, so clamping to
    shrink-only permits the bug rather than preventing it.

    Args:
        det_box:   (x1, y1, x2, y2) from the detector
        keypoints: {landmark_index: (x, y, visibility)} or None/{}
        alpha:     blend weight toward the hull; 0 keeps det_box unchanged

    Returns:
        (x1, y1, x2, y2)
    """
    if not keypoints or alpha <= 0.0:
        return det_box
    hull = keypoint_hull(keypoints)
    if hull is None:
        return det_box

    dx1, dy1, dx2, dy2 = det_box
    hx1, _hy1, hx2, hy2 = hull
    beta = 1.0 - alpha

    x1 = beta * dx1 + alpha * hx1
    x2 = beta * dx2 + alpha * hx2
    y2 = beta * dy2 + alpha * hy2
    y1 = dy1                      # no landmark above the eyes

    # A flickering landmark can invert an edge; keep the box well-formed.
    if x2 - x1 < 4 or y2 - y1 < 4:
        return det_box
    return (x1, y1, x2, y2)
