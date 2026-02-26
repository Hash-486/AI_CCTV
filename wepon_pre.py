import cv2
import torch
from ultralytics import YOLO
import time

# -------------------- Init --------------------
print("[INIT] Starting Weapon Detection")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()))

# -------------------- Load Trained Weapon Model --------------------
model = YOLO("runs/detect/weapon_yolov8s/weights/best.pt")  # <-- update this if needed
print("[INIT] Weapon YOLO model loaded")

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("[ERROR] Webcam could not be opened")
    exit()
print("[INIT] Webcam ready")

# -------------------- Main Loop --------------------
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed")
        break

    # Run inference
    results = model(frame)[0]

    # Draw detections
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        label = model.names[int(cls)]
        if conf > 0.3:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show FPS
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show output
    cv2.imshow("Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Closed cleanly.")
