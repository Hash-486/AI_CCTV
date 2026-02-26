import cv2
import datetime
import pandas as pd
import serial
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch

# --------------------- CUDA Check ---------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

# --------------------- Setup Model & Tracker ---------------------
model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# --------------------- Serial Setup (optional) ---------------------
try:
    arduino = serial.Serial('COM3', 115200, timeout=0.1)
    print("[INIT] Serial port opened")
except Exception:
    arduino = None
    print("[WARN] Serial port not available, continuing without alerts")

# --------------------- Webcam Setup (DirectShow) ---------------------
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

cap = open_camera(0) or open_camera(1)
if not cap:
    raise RuntimeError("Could not open webcam on index 0 or 1 with DirectShow")

print("[INIT] Webcam opened")

# --------------------- Logging & Parameters ---------------------
log_data     = []
frame_id     = 0
process_n    = 2     # process every Nth frame
min_conf     = 0.3   # confidence threshold
prev_time    = time.time()
weapon_cls   = [0]   # your weapon class IDs

# --------------------- Main Loop ---------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame, trying reconnect...")
        cap.release()
        cap = open_camera(0) or open_camera(1)
        if not cap:
            break
        continue

    frame_id += 1
    weapon_detected = False

    if frame_id % process_n == 0:
        # Inference
        results = model(frame, verbose=False)[0]
        boxes     = results.boxes.xyxy
        confs     = results.boxes.conf
        class_ids = results.boxes.cls

        # Filter detections
        idxs = [i for i, c in enumerate(class_ids)
                if int(c) in weapon_cls and confs[i] >= min_conf]
        weapon_boxes   = [boxes[i].tolist() for i in idxs]
        weapon_confs   = [confs[i].item() for i in idxs]
        weapon_classes = [int(class_ids[i].item()) for i in idxs]

        if weapon_boxes:
            weapon_detected = True
            detections = [[box, conf, cls]
                          for box, conf, cls in zip(weapon_boxes, weapon_confs, weapon_classes)]

            # Tracking
            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed(): continue
                l, t, r, b = map(int, track.to_ltrb())
                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                cv2.putText(frame, f"ID {track.track_id}", (l, t-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Serial alert
        if arduino:
            try:
                arduino.write(b'R' if weapon_detected else b'G')
            except:
                pass

        # Logging
        log_data.append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'threat':    'WeaponDetected' if weapon_detected else 'NoThreat',
            'frame':     frame_id
        })

        # FPS overlay
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Display
    cv2.imshow("Threat Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# --------------------- Cleanup ---------------------
cap.release()
cv2.destroyAllWindows()
pd.DataFrame(log_data).to_csv("threat_log.csv", index=False)
print("[INFO] Log saved to threat_log.csv")
