import cv2
import sys
from pipeline import TestTubeBarcodePipeline

init_lx, init_ly, init_lw, init_lh = 275, 162, 171, 627
init_cx, init_cy, init_cw, init_ch = 640, 180, 176, 816
init_rx, init_ry, init_rw, init_rh = 1364, 108, 148, 624
init_l_radius, init_c_radius, init_r_radius = 53, 85, 110

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    sys.exit(1)

cv2.namedWindow("Live ROI Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live ROI Controls", 600, 900)

cv2.createTrackbar("Left X", "Live ROI Controls", init_lx, 1600, nothing)
cv2.createTrackbar("Left Y", "Live ROI Controls", init_ly, 1200, nothing)
cv2.createTrackbar("Left W", "Live ROI Controls", init_lw, 1000, nothing)
cv2.createTrackbar("Left H", "Live ROI Controls", init_lh, 1200, nothing)
cv2.createTrackbar("Left Radius", "Live ROI Controls", init_l_radius, 400, nothing)

cv2.createTrackbar("Center X", "Live ROI Controls", init_cx, 1600, nothing)
cv2.createTrackbar("Center Y", "Live ROI Controls", init_cy, 1200, nothing)
cv2.createTrackbar("Center W", "Live ROI Controls", init_cw, 1000, nothing)
cv2.createTrackbar("Center H", "Live ROI Controls", init_ch, 1200, nothing)
cv2.createTrackbar("Center Radius", "Live ROI Controls", init_c_radius, 400, nothing)

cv2.createTrackbar("Right X", "Live ROI Controls", init_rx, 1600, nothing)
cv2.createTrackbar("Right Y", "Live ROI Controls", init_ry, 1200, nothing)
cv2.createTrackbar("Right W", "Live ROI Controls", init_rw, 1000, nothing)
cv2.createTrackbar("Right H", "Live ROI Controls", init_rh, 1200, nothing)
cv2.createTrackbar("Right Radius", "Live ROI Controls", init_r_radius, 400, nothing)

print("🎥 Interactive Live Pipeline Started. Adjust sliders in real time.")
print("Press 'q' to quit.")

pipeline = TestTubeBarcodePipeline(blur_threshold=30.0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    lx = cv2.getTrackbarPos("Left X", "Live ROI Controls")
    ly = cv2.getTrackbarPos("Left Y", "Live ROI Controls")
    lw = max(1, cv2.getTrackbarPos("Left W", "Live ROI Controls"))
    lh = max(1, cv2.getTrackbarPos("Left H", "Live ROI Controls"))
    l_radius = cv2.getTrackbarPos("Left Radius", "Live ROI Controls")

    cx = cv2.getTrackbarPos("Center X", "Live ROI Controls")
    cy = cv2.getTrackbarPos("Center Y", "Live ROI Controls")
    cw = max(1, cv2.getTrackbarPos("Center W", "Live ROI Controls"))
    ch = max(1, cv2.getTrackbarPos("Center H", "Live ROI Controls"))
    c_radius = cv2.getTrackbarPos("Center Radius", "Live ROI Controls")

    rx = cv2.getTrackbarPos("Right X", "Live ROI Controls")
    ry = cv2.getTrackbarPos("Right Y", "Live ROI Controls")
    rw = max(1, cv2.getTrackbarPos("Right W", "Live ROI Controls"))
    rh = max(1, cv2.getTrackbarPos("Right H", "Live ROI Controls"))
    r_radius = cv2.getTrackbarPos("Right Radius", "Live ROI Controls")

    left_roi = (lx, ly, lw, lh)
    center_roi = (cx, cy, cw, ch)
    right_roi = (rx, ry, rw, rh)

    result = pipeline.process_frame(
        frame, left_roi, center_roi, right_roi,
        left_radius=l_radius or None,
        center_radius=c_radius or None,
        right_radius=r_radius or None,
    )

    preview = frame.copy()
    cv2.rectangle(preview, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
    cv2.rectangle(preview, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

    cv2.putText(preview, f"status: {result['status']} (var={result['variance']:.1f})",
                (30, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    if result["barcodes"]:
        for idx, bc in enumerate(result["barcodes"]):
            msg = f"✅ [{bc['engine']}]: {bc['text']}"
            cv2.putText(preview, msg, (30, 60 + (idx * 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Live Feed - Camera View", preview)
    if result["unwrapped_strip"] is not None:
        cv2.imshow("Live Unwrapped 360 Strip", result["unwrapped_strip"])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nFinal Calibrated Parameters:")
        print(f"LEFT_MIRROR_ROI  = {left_roi}   (radius={l_radius or 'auto'})")
        print(f"CENTER_TUBE_ROI  = {center_roi}   (radius={c_radius or 'auto'})")
        print(f"RIGHT_MIRROR_ROI = {right_roi}   (radius={r_radius or 'auto'})")
        break

cap.release()
cv2.destroyAllWindows()