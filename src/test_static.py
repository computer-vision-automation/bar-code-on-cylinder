import cv2
import sys
from pipeline import TestTubeBarcodePipeline

# Calibrated ROI parameters locked from calibrate_roi.py
LEFT_MIRROR_ROI  = (275, 137, 227, 1008)
CENTER_TUBE_ROI  = (626, 137, 226, 1008)
RIGHT_MIRROR_ROI = (1332, 137, 187, 1008)
TUBE_RADIUS_PX   = 160

pipeline = TestTubeBarcodePipeline(tube_radius_px=TUBE_RADIUS_PX, blur_threshold=50.0)

# 0 is usually default laptop camera, 1 or 2 will be the external Arducam
cap = cv2.VideoCapture(0)

# Set Arducam high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    sys.exit(1)

print("🎥 Live Arducam Pipeline Started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    result = pipeline.process_frame(
        frame, 
        LEFT_MIRROR_ROI, 
        CENTER_TUBE_ROI, 
        RIGHT_MIRROR_ROI
    )

    # Draw ROI boxes
    preview = frame.copy()
    for (x, y, rw, rh), color in zip(
        [LEFT_MIRROR_ROI, CENTER_TUBE_ROI, RIGHT_MIRROR_ROI], 
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    ):
        cv2.rectangle(preview, (x, y), (x + rw, y + rh), color, 2)

    if result["barcodes"]:
        for idx, bc in enumerate(result["barcodes"]):
            msg = f"✅ [{bc['engine']}]: {bc['text']}"
            cv2.putText(preview, msg, (30, 60 + (idx * 40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Live Feed - Camera View", preview)

    if result["unwrapped_strip"] is not None:
        cv2.imshow("Live Unwrapped 360 Strip", result["unwrapped_strip"])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()