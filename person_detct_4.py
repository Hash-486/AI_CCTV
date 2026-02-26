import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# -------------------- Init Info --------------------
print("[INIT] Starting Person Detection with Tracking")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()))

# -------------------- Load Model --------------------
model = YOLO("yolov8s.pt")
print("[INIT] YOLOv8s loaded")

# -------------------- Init DeepSORT with ReID Enhancements --------------------
tracker = DeepSort(
    max_age=60,
    n_init=2,
    nn_budget=100,
    embedder="mobilenet",
    half=False,
    bgr=True,
    embedder_gpu=torch.cuda.is_available()
)

print("[INIT] DeepSORT initialized with ReID model")

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("[ERROR] Webcam could not be opened")
    exit()
print("[INIT] Webcam ready")

# -------------------- Helper: Per-ID Color --------------------
def get_color(id):
    import random
    random.seed(id)
    return tuple(random.randint(0, 255) for _ in range(3))

# -------------------- Main Loop --------------------
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed")
        break

    results = model(frame)[0]

    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls) == 0 and conf > 0.3:  # Class 0 = person
            xyxy = box.tolist()
            confidence = conf.item()
            detections.append([xyxy, confidence, 'person'])

    tracks = tracker.update_tracks(detections, frame=frame.copy())


    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        color = get_color(track_id)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Closed cleanly.")
