import cv2
from ultralytics import YOLO
import torch

print("[INIT] Starting person detection test")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Explicitly use DSHOW backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Webcam not accessible via DSHOW")
    exit()

cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLO("yolov8s.pt")
print("[INIT] YOLOv8s model loaded")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    results = model(frame)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls.item()) == 0:  # Class 0 = person
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Cam Test - Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
