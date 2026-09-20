# ============================================================
# association.py -- Weapon-to-Person Spatial Matching
# ============================================================
from config import (
    ASSOCIATION_IOU_THRESHOLD,
    ASSOCIATION_CONTAINMENT_THRESHOLD,
    WRIST_PROXIMITY_RADIUS,
    POSE_MIN_VISIBILITY,
)

# MediaPipe wrist landmark indices
_LEFT_WRIST  = 15
_RIGHT_WRIST = 16


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in (l, t, r, b) / (x1, y1, x2, y2) format."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter_area / float(area_a + area_b - inter_area)


def compute_containment(weapon_box, person_box):
    """
    What fraction of the weapon box is contained inside the person box?
    Useful because a held weapon is typically fully inside the person bbox.
    """
    x_a = max(weapon_box[0], person_box[0])
    y_a = max(weapon_box[1], person_box[1])
    x_b = min(weapon_box[2], person_box[2])
    y_b = min(weapon_box[3], person_box[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    weapon_area = max(1, (weapon_box[2] - weapon_box[0]) * (weapon_box[3] - weapon_box[1]))
    return inter_area / float(weapon_area)


def _weapon_center(weapon_box):
    """Return (cx, cy) center of a weapon bounding box."""
    return (
        (weapon_box[0] + weapon_box[2]) / 2.0,
        (weapon_box[1] + weapon_box[3]) / 2.0,
    )


def is_weapon_near_wrist(weapon_box: list, keypoints: dict, radius: float = WRIST_PROXIMITY_RADIUS) -> bool:
    """
    Return True if the weapon bbox center is within `radius` pixels of
    either wrist keypoint (with sufficient visibility).
    """
    if not keypoints:
        return False

    wcx, wcy = _weapon_center(weapon_box)
    for wrist_idx in (_LEFT_WRIST, _RIGHT_WRIST):
        kp = keypoints.get(wrist_idx)
        if kp is None:
            continue
        kx, ky, vis = kp
        if vis < POSE_MIN_VISIBILITY:
            continue
        dist = ((wcx - kx) ** 2 + (wcy - ky) ** 2) ** 0.5
        if dist <= radius:
            return True
    return False


def associate_weapons_to_persons(person_tracks: list, weapon_detections: list):
    """
    Match each weapon detection to the closest tracked person using a
    three-tier priority system:

      Tier 1 -- Wrist proximity  (method = "wrist")      : weapon center near a wrist keypoint
      Tier 2 -- Containment      (method = "containment") : weapon box mostly inside person box
      Tier 3 -- IoU              (method = "iou")         : bounding box overlap

    Parameters
    ----------
    person_tracks     : list of dicts  {'canonical_id', 'bbox_ltrb', 'keypoints'}
    weapon_detections : list of dicts  {'bbox', 'confidence', 'label'}

    Returns
    -------
    associations    : list of dicts
        {
          'person_id'         : int,
          'weapon_label'      : str,
          'weapon_confidence' : float,
          'weapon_bbox'       : [x1,y1,x2,y2],
          'method'            : str  ('wrist' | 'containment' | 'iou'),
        }
    unmatched_weapons : list of weapon detection dicts not matched to any person
    """
    associations = []
    unmatched_weapons = []

    for w_det in weapon_detections:
        w_box = w_det["bbox"]

        # ---- Tier 1: wrist proximity ----
        wrist_match_id = None
        for p_track in person_tracks:
            kps = p_track.get("keypoints", {})
            if is_weapon_near_wrist(w_box, kps):
                wrist_match_id = p_track["canonical_id"]
                break  # take the first (and typically only) wrist match

        if wrist_match_id is not None:
            associations.append({
                "person_id":         wrist_match_id,
                "weapon_label":      w_det["label"],
                "weapon_confidence": w_det["confidence"],
                "weapon_bbox":       w_box,
                "method":            "wrist",
            })
            continue

        # ---- Tier 2 & 3: containment / IoU ----
        best_person_id = None
        best_score = 0.0
        best_method = "iou"

        for p_track in person_tracks:
            p_box = p_track["bbox_ltrb"]
            containment = compute_containment(w_box, p_box)
            iou = compute_iou(w_box, p_box)

            if containment >= ASSOCIATION_CONTAINMENT_THRESHOLD:
                score = containment
                method = "containment"
            elif iou >= ASSOCIATION_IOU_THRESHOLD:
                score = iou
                method = "iou"
            else:
                continue

            if score > best_score:
                best_score = score
                best_person_id = p_track["canonical_id"]
                best_method = method

        if best_person_id is not None:
            associations.append({
                "person_id":         best_person_id,
                "weapon_label":      w_det["label"],
                "weapon_confidence": w_det["confidence"],
                "weapon_bbox":       w_box,
                "method":            best_method,
            })
        else:
            unmatched_weapons.append(w_det)

    return associations, unmatched_weapons
