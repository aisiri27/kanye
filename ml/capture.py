import cv2
import os
from datetime import datetime

# 10 gesture classes
CLASSES = [
    "R_no_touch",
    "R_index_touch",
    "R_middle_touch",
    "R_ring_touch",
    "R_pinky_touch",
    "L_no_touch",
    "L_index_touch",
    "L_middle_touch",
    "L_ring_touch",
    "L_pinky_touch",
]

DATASET_DIR = "dataset"

# Create folders if they don't exist
for cls in CLASSES:
    os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

cap = cv2.VideoCapture(0)
current_class = 0

print("\n📸 DATASET CAPTURE TOOL")
print("----------------------")
print("Keys:")
print("1–5  → Right hand classes")
print("6–0  → Left hand classes")
print("s    → Save image")
print("q    → Quit\n")

print("Class mapping:")
for i, cls in enumerate(CLASSES):
    print(f"{i+1}: {cls}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show current class
    cv2.putText(
        frame,
        f"Class: {CLASSES[current_class]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Gesture Dataset Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        path = os.path.join(DATASET_DIR, CLASSES[current_class], filename)
        cv2.imwrite(path, frame)
        print(f"Saved → {path}")

    if key in [ord(str(i)) for i in range(10)]:
        idx = key - ord("0")
        if idx == 0:
            current_class = 9
        else:
            current_class = idx - 1
        print(f"Switched to {CLASSES[current_class]}")

cap.release()
cv2.destroyAllWindows()
