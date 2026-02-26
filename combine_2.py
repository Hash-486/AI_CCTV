import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# -------------------- Init Info --------------------
print("[INIT] Starting Person + Weapon Detection")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()))

# -------------------- Load Models --------------------
person_model = YOLO("yolov8s.pt")  # Pretrained model for person detection
weapon_model = YOLO("runs/detect/weapon_yolov8s/weights/best.pt")  # Your trained model

print("[INIT] Models loaded")

# -------------------- Init DeepSORT Tracker --------------------
tracker = DeepSort(
    max_age=60,
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

# -------------------- Helper: Color by Track ID --------------------
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

    # ---------- Person Detection ----------
    person_results = person_model(frame)[0]
    detections = []
    for box, conf, cls in zip(person_results.boxes.xyxy, person_results.boxes.conf, person_results.boxes.cls):
        if int(cls) == 0 and conf > 0.3:  # class 0 = person
            xyxy = box.tolist()
            confidence = conf.item()
            detections.append([xyxy, confidence, 'person'])

    # ---------- Person Tracking ----------
    tracks = tracker.update_tracks(detections, frame=frame.copy())

    person_boxes = []  # Save tracked person boxes for potential IoU matching
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        person_boxes.append([l, t, r, b])
        color = get_color(track_id)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- Weapon Detection ----------
    weapon_results = weapon_model(frame)[0]
    for box, conf, cls in zip(weapon_results.boxes.xyxy, weapon_results.boxes.conf, weapon_results.boxes.cls):
        label = weapon_model.names[int(cls)]
        if label.lower() in ['knife', 'gun'] and conf > 0.3:
            x1, y1, x2, y2 = map(int, box)

            # OPTIONAL: check IoU with person box and only draw if overlaps (comment out if not needed)
            # weapon_box = [x1, y1, x2, y2]
            # def compute_iou(boxA, boxB):
            #     xA = max(boxA[0], boxB[0])
            #     yA = max(boxA[1], boxB[1])
            #     xB = min(boxA[2], boxB[2])
            #     yB = min(boxA[3], boxB[3])
            #     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            #     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            #     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            #     iou = interArea / float(boxAArea + boxBArea - interArea)
            #     return iou
            # if any(compute_iou(weapon_box, person) > 0.5 for person in person_boxes):

            # Draw red box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---------- FPS ----------
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ---------- Show ----------
    cv2.imshow("Person + Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Closed cleanly.")
