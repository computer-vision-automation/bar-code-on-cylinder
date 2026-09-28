import cv2
import time
from collections import Counter, deque
from pipeline import TestTubeBarcodePipeline

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================
CAM_W = 1600
CAM_H = 1200

# Re-detect the blue caps only every N frames.
# The ROI is held between detections, so the boxes still look live.
TRACK_EVERY = 5

# Barcode decoding is expensive because it tries multiple preprocessings
# and rotations. Do NOT run it on every camera frame.
DECODE_EVERY = 8

# Display the flattened image every frame, but use the most recent result.
# Increase to 2 if the PC is still slow.
DISPLAY_FLAT_EVERY = 1

INITIAL_ROIS = [
    (275, 162, 171, 627),
    (640, 180, 176, 816),
    (1364, 108, 148, 624),
]
RADII = (53, 85, 110)

pipeline = TestTubeBarcodePipeline(blur_threshold=15.0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# MJPG is usually much faster for USB Arducam webcams because it reduces
# USB bandwidth compared with raw BGR frames.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

rois = INITIAL_ROIS
detected = [False, False, False]

frame_no = 0
last_result = None
last_flat = None
code_history = deque(maxlen=6)

# FPS measurement
fps_time = time.perf_counter()
fps_frames = 0
fps = 0.0

cv2.namedWindow("LIVE - moving ROI + barcode", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LIVE - moving ROI + barcode", 1000, 750)

cv2.namedWindow("LIVE - FLATTENED", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LIVE - FLATTENED", 900, 450)

print("Fast live pipeline started.")
print(f"Tracking every {TRACK_EVERY} frames.")
print(f"Barcode decoding every {DECODE_EVERY} frames.")
print("Press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera frame failed.")
        break

    frame_no += 1

    # --------------------------------------------------------
    # 1. Track ROIs only periodically.
    # --------------------------------------------------------
    if frame_no == 1 or frame_no % TRACK_EVERY == 0:
        rois, detected = pipeline.detect_dynamic_rois(frame, rois)

    # --------------------------------------------------------
    # 2. Process the image.
    # --------------------------------------------------------
    # The expensive barcode decoder is deliberately skipped on most frames.
    # Flattening still updates continuously.
    if frame_no == 1 or frame_no % DECODE_EVERY == 0:
        result = pipeline.process_frame(
            frame,
            rois[0], rois[1], rois[2],
            left_radius=RADII[0],
            center_radius=RADII[1],
            right_radius=RADII[2],
        )
        last_result = result

        if result["unwrapped_strip"] is not None:
            last_flat = result["unwrapped_strip"]

        texts = [b["text"] for b in result["barcodes"]]
        if texts:
            code_history.append(texts[0])

    # --------------------------------------------------------
    # 3. Draw lightweight live preview.
    # --------------------------------------------------------
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
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    if last_result is not None:
        cv2.putText(
            preview,
            f"{last_result['status']} | sharpness={last_result['variance']:.1f}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 255),
            2,
        )

    # Require repeated reads before displaying a stable barcode.
    if code_history:
        stable_code, count = Counter(code_history).most_common(1)[0]
        if count >= 2:
            cv2.putText(
                preview,
                f"BARCODE: {stable_code}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                3,
            )

    # FPS counter
    fps_frames += 1
    now = time.perf_counter()
    elapsed = now - fps_time
    if elapsed >= 1.0:
        fps = fps_frames / elapsed
        fps_frames = 0
        fps_time = now

    cv2.putText(
        preview,
        f"FPS: {fps:.1f}",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("LIVE - moving ROI + barcode", preview)

    if last_flat is not None:
        cv2.imshow("LIVE - FLATTENED", last_flat)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
