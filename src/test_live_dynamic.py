import cv2
import time
from collections import Counter, deque
from pipeline import TestTubeBarcodePipeline

CAM_W = 1600
CAM_H = 1200

# Starting geometry from the supplied calibration. The important difference
# is that these are now only the INITIAL ROIs; they are moved every frame by
# detect_dynamic_rois().
INITIAL_ROIS = [
    (275, 162, 171, 627),
    (640, 180, 176, 816),
    (1364, 108, 148, 624),
]
RADII = (53, 85, 110)

pipeline = TestTubeBarcodePipeline(blur_threshold=25.0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

rois = INITIAL_ROIS
last_detect_time = 0.0
code_history = deque(maxlen=8)

cv2.namedWindow("LIVE - moving ROI + barcode", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LIVE - moving ROI + barcode", 1100, 800)
cv2.namedWindow("LIVE - FLATTENED", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LIVE - FLATTENED", 1000, 500)

print("Live dynamic pipeline started.")
print("Blue caps are used to move the three ROIs with the tube.")
print("Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera frame failed.")
        break

    # Re-detect cap positions every frame. No manual slider movement is
    # required after the initial calibration.
    rois, detected = pipeline.detect_dynamic_rois(frame, rois)

    result = pipeline.process_frame(
        frame,
        rois[0], rois[1], rois[2],
        left_radius=RADII[0],
        center_radius=RADII[1],
        right_radius=RADII[2],
    )

    preview = frame.copy()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    names = ["LEFT", "CENTER", "RIGHT"]
    for roi, color, name, found in zip(rois, colors, names, detected):
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            preview,
            f"{name} {'TRACKED' if found else 'HOLD'}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    status = result["status"]
    cv2.putText(
        preview,
        f"{status} | sharpness={result['variance']:.1f}",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2,
    )

    # Temporal confirmation prevents one bad frame from changing the answer.
    texts = [b["text"] for b in result["barcodes"]]
    if texts:
        code_history.append(texts[0])

    stable_code = None
    if code_history:
        stable_code, count = Counter(code_history).most_common(1)[0]
        if count >= 3:
            cv2.putText(
                preview,
                f"BARCODE: {stable_code}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3,
            )

    cv2.imshow("LIVE - moving ROI + barcode", preview)

    if result["unwrapped_strip"] is not None:
        cv2.imshow("LIVE - FLATTENED", result["unwrapped_strip"])

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
