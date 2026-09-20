# ============================================================
# verification.py -- Stage 2: turn detections into alerts
# ============================================================
"""
Stage 2 of the two-stage design: verify with high precision.

The problem this solves is concrete. Every threat screenshot the system has ever
saved is a false positive -- a canteen scene flagged CRITICAL with no weapon in
it, a hand near a face flagged HIGH. Raising the detector's confidence would fix
that by refusing to look, at the cost of missing real weapons.

Instead the detector runs wide open (conf 0.25, high recall, few false
negatives) and every candidate must clear three independent gates before it can
raise an alert:

    1. CLASS CONFIDENCE   per-class threshold from the measured F1 curves
    2. PERSISTENCE        present across N consecutive detection cycles
    3. PERSON ASSOCIATION attached to a tracked person

The gates are independent, so a false positive has to survive all three. A
texture that briefly reads as a knife fails persistence; a knife-shaped shadow
on a table fails association; a low-confidence flicker fails the class gate.

Every rejection is counted, so the demo can show exactly how much work each gate
is doing rather than asserting that it helps.
"""
from __future__ import annotations

from collections import defaultdict

from weapon.detector import box_iou
from config import (
    VERIFY_HIGH_CONF_BYPASS, VERIFY_PERSISTENCE, VERIFY_REQUIRE_PERSON,
    VERIFY_TEMPORAL_IOU, WEAPON_CLASS_CONF,
)


class TwoStageVerifier:
    """Holds the temporal state and the rejection statistics."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._tracks: list[dict] = []        # candidates carried between cycles
        self.stats = defaultdict(int)

    def reset_stats(self) -> None:
        self.stats = defaultdict(int)

    # ---------------- gate 1 ----------------
    def _passes_class_conf(self, det: dict) -> bool:
        need = WEAPON_CLASS_CONF.get(det["label"], 0.5)
        return det["confidence"] >= need

    # ---------------- gate 2 ----------------
    def _update_persistence(self, dets: list[dict]) -> None:
        """Match this cycle's detections to the previous ones and age the rest."""
        matched = set()
        for t in self._tracks:
            hit = None
            for i, d in enumerate(dets):
                if i in matched or d["label"] != t["label"]:
                    continue
                if box_iou(d["bbox"], t["bbox"]) >= VERIFY_TEMPORAL_IOU:
                    hit = i
                    break
            if hit is not None:
                matched.add(hit)
                t["bbox"] = dets[hit]["bbox"]
                t["seen"] += 1
                t["missed"] = 0
            else:
                t["missed"] += 1
        for i, d in enumerate(dets):
            if i not in matched:
                self._tracks.append({"label": d["label"], "bbox": list(d["bbox"]),
                                     "seen": 1, "missed": 0})
        # forget anything absent for two cycles
        self._tracks = [t for t in self._tracks if t["missed"] < 2]

    def _persistence_of(self, det: dict) -> int:
        for t in self._tracks:
            if t["label"] == det["label"] and \
                    box_iou(det["bbox"], t["bbox"]) >= VERIFY_TEMPORAL_IOU:
                return t["seen"]
        return 0

    # ---------------- gate 3 ----------------
    @staticmethod
    def _attached_to_person(det: dict, person_tracks: list) -> bool:
        """Weapon box overlapping / mostly inside a tracked person's box."""
        from threat.association import compute_containment, compute_iou
        from config import (ASSOCIATION_CONTAINMENT_THRESHOLD,
                            ASSOCIATION_IOU_THRESHOLD)
        for p in person_tracks:
            # tracker.PersonTracker.update() returns {'canonical_id', 'bbox_ltrb'}
            pbox = p.get("bbox_ltrb") or p.get("bbox")
            if not pbox:
                continue
            if compute_iou(det["bbox"], pbox) >= ASSOCIATION_IOU_THRESHOLD:
                return True
            if compute_containment(det["bbox"], pbox) >= ASSOCIATION_CONTAINMENT_THRESHOLD:
                return True
        return False

    # ---------------- the gate chain ----------------
    def verify(self, candidates: list[dict], person_tracks: list) -> tuple[list, list]:
        """
        Returns (confirmed, rejected). Each rejected item carries a 'reason'
        so the overlay can show which gate stopped it.
        """
        self.stats["candidates"] += len(candidates)
        if not self.enabled:                       # A/B mode for the benchmark
            self.stats["confirmed"] += len(candidates)
            return list(candidates), []

        self._update_persistence(candidates)

        confirmed, rejected = [], []
        for det in candidates:
            if not self._passes_class_conf(det):
                self.stats["rejected_class_conf"] += 1
                rejected.append({**det, "reason": "conf"})
                continue

            seen = self._persistence_of(det)
            if seen < VERIFY_PERSISTENCE and det["confidence"] < VERIFY_HIGH_CONF_BYPASS:
                self.stats["rejected_persistence"] += 1
                rejected.append({**det, "reason": "transient"})
                continue

            if VERIFY_REQUIRE_PERSON and person_tracks is not None:
                if not self._attached_to_person(det, person_tracks):
                    self.stats["rejected_no_person"] += 1
                    rejected.append({**det, "reason": "no person"})
                    continue

            self.stats["confirmed"] += 1
            confirmed.append(det)

        return confirmed, rejected

    def summary(self) -> str:
        s = self.stats
        c = s.get("candidates", 0)
        ok = s.get("confirmed", 0)
        return (f"candidates={c}  confirmed={ok}  "
                f"rejected: conf={s.get('rejected_class_conf', 0)} "
                f"transient={s.get('rejected_persistence', 0)} "
                f"no-person={s.get('rejected_no_person', 0)}")
