import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------- Init --------------------
print("[INIT] Starting Weapon Detection + Tracking")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()))

# -------------------- Load Trained YOLO Weapon Model --------------------
model = YOLO("runs/detect/weapon_yolov8s/weights/best.pt")  # adjust path if needed
print("[INIT] Weapon YOLO model loaded")

# -------------------- Init DeepSORT --------------------
tracker = DeepSort(
    max_age=30,
    n_init=2,
    nn_budget=100,
    embedder="mobilenet",
    half=False,
    bgr=True,
    embedder_gpu=torch.cuda.is_available()
)
print("[INIT] DeepSORT initialized")

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("[ERROR] Webcam could not be opened")
    exit()
print("[INIT] Webcam ready")

# -------------------- Helper --------------------
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
    
    # YOLO detections to DeepSORT format
    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        label = model.names[int(cls)]
        if conf > 0.4:  # Adjust threshold to reduce false positives
            xyxy = box.tolist()
            confidence = conf.item()
            detections.append([xyxy, confidence, label])

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame.copy())

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        label = track.get_det_class()
        color = get_color(track_id)

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"{label} ID:{track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS display
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Weapon Detection + Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Closed cleanly.")
