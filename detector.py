# ============================================================
# detector.py -- YOLO person detection
#
# Wraps Ultralytics YOLO, filtered to COCO class 0 (person).
# ============================================================
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    PERSON_MODEL_PATH,
    PERSON_CONF_THRESHOLD,
    INFER_IMGSZ,
    YOLO_HALF,
    YOLO_AGNOSTIC_NMS,
)

PERSON_CLASS_ID = 0  # COCO


class PersonDetector:
    """Detects people in a frame.

    Returns detections in the format deep_sort_realtime expects:
        [([left, top, width, height], confidence, "person"), ...]

    The LTWH format is not incidental -- update_tracks() converts LTRB to
    LTWH internally, so producing LTWH directly avoids a lossy round trip
    through integer coordinates.
    """

    def __init__(
        self,
        model_path=PERSON_MODEL_PATH,
        conf=PERSON_CONF_THRESHOLD,
        imgsz=INFER_IMGSZ,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = YOLO_HALF and self.device == "cuda"

        if self.device == "cuda":
            self.model.to(self.device)

    def detect(self, frame):
        """Run detection on a BGR frame.

        Returns:
            dets:  list of ([l, t, w, h], conf, "person")
            boxes: (N, 4) float32 array of [x1, y1, x2, y2] -- the same
                   detections in LTRB, for cropping and embedding.
            confs: (N,) float32 array
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            imgsz=self.imgsz,
            half=self.half,
            agnostic_nms=YOLO_AGNOSTIC_NMS,
            verbose=False,
            device=self.device,
        )

        dets = []
        boxes = []
        confs = []

        if not results:
            return dets, np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return dets, np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf_arr = r.boxes.conf.cpu().numpy().astype(np.float32)

        for (x1, y1, x2, y2), c in zip(xyxy, conf_arr):
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            dets.append(([float(x1), float(y1), float(w), float(h)], float(c), "person"))
            boxes.append([x1, y1, x2, y2])
            confs.append(c)

        boxes = (
            np.asarray(boxes, np.float32)
            if boxes
            else np.zeros((0, 4), np.float32)
        )
        confs = (
            np.asarray(confs, np.float32)
            if confs
            else np.zeros((0,), np.float32)
        )
        return dets, boxes, confs
