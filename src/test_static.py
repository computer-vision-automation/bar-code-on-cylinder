import cv2
import sys
from pipeline import TestTubeBarcodePipeline

LEFT_MIRROR_ROI  = (275, 162, 171, 627)
CENTER_TUBE_ROI  = (640, 180, 176, 816)
RIGHT_MIRROR_ROI = (1364, 108, 148, 624)

LEFT_RADIUS_PX   = 53
CENTER_RADIUS_PX = 85
RIGHT_RADIUS_PX  = 110

pipeline = TestTubeBarcodePipeline(blur_threshold=50.0)

cap = cv2.VideoCapture(0)

# Must match the resolution calibrate_roi.py's sample image was captured at
# (1600x1200) - these ROI pixel coordinates were measured against that frame
# size. If your Arducam can't do 1600x1200, recalibrate against a live frame
# at whatever resolution you set here instead of reusing the static photo.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

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
        RIGHT_MIRROR_ROI,
        left_radius=LEFT_RADIUS_PX,
        center_radius=CENTER_RADIUS_PX,
        right_radius=RIGHT_RADIUS_PX,
    )

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