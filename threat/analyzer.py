# ============================================================
# threat_analyzer.py -- Threat Level Engine
# ============================================================
from config import (
    WEAPON_THREAT_MAP,
    THREAT_LEVELS,
    THREAT_DOWNGRADE_FRAMES,
    POSE_MIN_VISIBILITY,
)

# MediaPipe landmark indices used for posture analysis
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16

# Approach speed: frames of history to keep per person for bbox area tracking
_AREA_HISTORY_LEN = 10
# Fraction of area increase over history window to flag as "approaching fast"
_APPROACH_AREA_RATIO = 1.5


def _is_arm_raised(keypoints: dict) -> bool:
    """
    Return True if either wrist is above the corresponding shoulder
    (in image coordinates, y increases downward, so raised = lower y value).
    Requires both wrist and shoulder to be visible.
    """
    for shoulder_idx, wrist_idx in ((_LEFT_SHOULDER, _LEFT_WRIST), (_RIGHT_SHOULDER, _RIGHT_WRIST)):
        s = keypoints.get(shoulder_idx)
        w = keypoints.get(wrist_idx)
        if s is None or w is None:
            continue
        s_x, s_y, s_vis = s
        w_x, w_y, w_vis = w
        if s_vis < POSE_MIN_VISIBILITY or w_vis < POSE_MIN_VISIBILITY:
            continue
        if w_y < s_y:  # wrist above shoulder in image coords
            return True
    return False


def _bbox_area(bbox_ltrb) -> int:
    l, t, r, b = bbox_ltrb
    return max(1, (r - l) * (b - t))


class ThreatAnalyzer:
    """
    Determines per-person threat level using:
      - Associated weapon type and count
      - Association method (wrist > containment > iou)
      - Arm posture (raised arm escalates threat)
      - Approach speed (rapid bbox growth escalates threat)
      - Temporal hysteresis (gradual step-down, not instant reset)
    """

    def __init__(self):
        # person_id -> state dict
        self._state: dict = {}
        print("[INIT] ThreatAnalyzer ready (posture-aware)")

    def _get_or_create(self, person_id: int) -> dict:
        if person_id not in self._state:
            self._state[person_id] = {
                "current_level":      0,
                "safe_streak":        THREAT_DOWNGRADE_FRAMES,
                "weapons_this_frame": [],
                "assoc_methods":      [],   # association method per weapon
                "area_history":       [],   # recent bbox areas
            }
        return self._state[person_id]

    def analyze(self, person_tracks: list, associations: list):
        """
        Parameters
        ----------
        person_tracks : list of {
            'canonical_id': int,
            'bbox_ltrb': tuple,
            'keypoints': dict,   <- from PoseEstimator
        }
        associations  : list of {
            'person_id': int,
            'weapon_label': str,
            'weapon_confidence': float,
            'weapon_bbox': list,
            'method': str,       <- 'wrist' | 'containment' | 'iou'
        }

        Returns
        -------
        threat_results : dict  {person_id -> {
            'level': int, 'name': str, 'color': tuple,
            'weapons': list, 'arm_raised': bool, 'approaching': bool,
        }}
        """
        # Build lookup: canonical_id -> track dict
        track_map = {pt["canonical_id"]: pt for pt in person_tracks}
        active_ids = set(track_map.keys())

        # Reset per-frame weapon lists
        for pid in active_ids:
            state = self._get_or_create(pid)
            state["weapons_this_frame"] = []
            state["assoc_methods"] = []

        # Populate weapons per person
        for assoc in associations:
            pid = assoc["person_id"]
            if pid in self._state:
                self._state[pid]["weapons_this_frame"].append(assoc["weapon_label"])
                self._state[pid]["assoc_methods"].append(assoc.get("method", "iou"))

        # Determine threat level per person
        threat_results = {}
        for pid in active_ids:
            state = self._state[pid]
            pt = track_map[pid]
            keypoints = pt.get("keypoints", {})
            bbox = pt["bbox_ltrb"]
            weapons = state["weapons_this_frame"]
            methods = state["assoc_methods"]

            # ---- Approach speed ----
            area = _bbox_area(bbox)
            state["area_history"].append(area)
            if len(state["area_history"]) > _AREA_HISTORY_LEN:
                state["area_history"].pop(0)

            approaching = False
            if len(state["area_history"]) >= _AREA_HISTORY_LEN:
                oldest = state["area_history"][0]
                newest = state["area_history"][-1]
                if newest >= oldest * _APPROACH_AREA_RATIO:
                    approaching = True

            # ---- Posture ----
            arm_raised = _is_arm_raised(keypoints) if keypoints else False

            # ---- Desired threat level ----
            if len(weapons) == 0:
                state["safe_streak"] += 1
                if state["safe_streak"] >= THREAT_DOWNGRADE_FRAMES:
                    desired_level = 0
                else:
                    desired_level = state["current_level"]
            else:
                state["safe_streak"] = 0

                levels = []
                for w, method in zip(weapons, methods):
                    wl = w.lower()
                    base = WEAPON_THREAT_MAP.get(wl, 1)

                    # Wrist match = full confidence; bbox-only = reduce by 1
                    if method == "wrist":
                        lvl = base
                    elif method == "containment":
                        lvl = base
                    else:  # iou only
                        lvl = max(0, base - 1)

                    # Arm raised with weapon = escalate
                    if arm_raised and method in ("wrist", "containment"):
                        lvl = min(lvl + 1, 3)

                    levels.append(lvl)

                max_level = max(levels)

                # CRITICAL for multiple distinct weapons or gun + knife
                unique_weapons = set(w.lower() for w in weapons)
                if len(unique_weapons) > 1 and "guns" in unique_weapons:
                    desired_level = 3
                elif len(weapons) > 1 and max_level >= 2:
                    desired_level = 3
                else:
                    desired_level = max_level

                # Approaching fast + weapon = escalate one more level
                if approaching and desired_level >= 1:
                    desired_level = min(desired_level + 1, 3)

            # ---- Apply hysteresis (instant up, gradual step-down) ----
            if desired_level > state["current_level"]:
                state["current_level"] = desired_level
            elif desired_level < state["current_level"]:
                if state["safe_streak"] >= THREAT_DOWNGRADE_FRAMES:
                    state["current_level"] = state["current_level"] - 1
                    state["safe_streak"] = 0

            level = state["current_level"]
            info = THREAT_LEVELS[level]
            threat_results[pid] = {
                "level":      level,
                "name":       info["name"],
                "color":      info["color"],
                "weapons":    weapons,
                "arm_raised": arm_raised,
                "approaching": approaching,
            }

        # Prune stale persons
        stale = [pid for pid in self._state if pid not in active_ids]
        for pid in stale:
            del self._state[pid]

        return threat_results
