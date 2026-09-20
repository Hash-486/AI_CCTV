# ============================================================
# detectors.py -- YOLO26 person & weapon detection
# ============================================================
"""
Stage 1 of the two-stage design: detect with high recall.

Both detectors run YOLO26s. The weapon model is the custom 3-class model trained
for this project (guns / knife / long_gun, 28,943 images, mAP50-95 0.654); the
person model is the pretrained COCO checkpoint.

Deliberately runs at a LOW confidence threshold. Everything expensive about
precision happens in verification.py -- the detector's job here is not to miss
anything. The geometric filter is the one exception: a 5-pixel "pistol" is not a
recall opportunity, it is noise, and dropping it costs nothing.
"""
from __future__ import annotations

import warnings

import torch
from ultralytics import YOLO

# Ultralytics deprecated `half=` in favour of `quantize=`; the argument still
# works and FP16 is what we want. Silenced so the live demo output stays clean.
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")

from config import (
    INFER_IMGSZ, KNIFE_MIN_ASPECT, MIN_BOX_PX,
    PERSON_CONF_THRESHOLD, PERSON_MODEL_PATH,
    WEAPON_DETECT_CONF, WEAPON_MODEL_PATH,
)


def box_iou(a, b) -> float:
    """IoU for two [x1,y1,x2,y2] boxes."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


class PersonDetector:
    """YOLO26s, COCO class 0 only."""

    def __init__(self, model_path: str = PERSON_MODEL_PATH,
                 conf: float = PERSON_CONF_THRESHOLD):
        self.model = YOLO(model_path)
        self.conf = conf
        self._half = torch.cuda.is_available()
        print(f"[INIT] Person model : {model_path}")
        print(f"[INIT]   imgsz={INFER_IMGSZ}  conf={conf}")

    def detect(self, frame) -> list:
        """-> [[x1,y1,x2,y2], confidence, 'person'] for DeepSORT."""
        r = self.model(frame, verbose=False, half=self._half,
                       imgsz=INFER_IMGSZ, classes=[0], conf=self.conf)[0]
        out = []
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            out.append([box.tolist(), conf.item(), "person"])
        return out


class WeaponDetector:
    """
    Custom 3-class YOLO26s. Emits *candidates*, not alerts.

    Returns dicts: {'bbox', 'confidence', 'label'} -- consumed by
    verification.TwoStageVerifier, which decides what becomes an alert.
    """

    def __init__(self, model_path: str = WEAPON_MODEL_PATH,
                 conf: float = WEAPON_DETECT_CONF):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names
        self._half = torch.cuda.is_available()
        print(f"[INIT] Weapon model : {model_path}")
        print(f"[INIT]   classes={list(self.class_names.values())}")
        print(f"[INIT]   imgsz={INFER_IMGSZ} (matches training)  "
              f"stage-1 conf={conf}")

    @staticmethod
    def _plausible(label: str, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Reject boxes too small or the wrong shape to be the claimed weapon."""
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            return False
        if label == "knife":
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect < KNIFE_MIN_ASPECT:      # knives are elongated
                return False
        return True

    def detect(self, frame) -> list[dict]:
        r = self.model(frame, verbose=False, half=self._half,
                       imgsz=INFER_IMGSZ, conf=self.conf)[0]
        candidates = []
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            label = self.class_names[int(cls)]
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            if not self._plausible(label, x1, y1, x2, y2):
                continue
            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(conf), 3),
                "label": label,
            })
        return candidates
