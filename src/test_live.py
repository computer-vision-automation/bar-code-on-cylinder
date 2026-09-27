import cv2
import sys
from pipeline import TestTubeBarcodePipeline

# Initial default ROI values (x, y, w, h) and Tube Radius
init_lx, init_ly, init_lw, init_lh = 275, 137, 227, 1008
init_cx, init_cy, init_cw, init_ch = 626, 137, 226, 1008
init_rx, init_ry, init_rw, init_rh = 1332, 137, 187, 1008
init_r = 160

def nothing(x):
    pass

# Open camera stream (0 for default, 1 or 2 for external Arducam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    sys.exit(1)

# Create setup control window with trackbars
cv2.namedWindow("Live ROI Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live ROI Controls", 600, 700)

# Trackbars for Left Mirror ROI
cv2.createTrackbar("Left X", "Live ROI Controls", init_lx, 1920, nothing)
cv2.createTrackbar("Left Y", "Live ROI Controls", init_ly, 1080, nothing)
cv2.createTrackbar("Left W", "Live ROI Controls", init_lw, 1000, nothing)
cv2.createTrackbar("Left H", "Live ROI Controls", init_lh, 1080, nothing)

# Trackbars for Center Tube ROI
cv2.createTrackbar("Center X", "Live ROI Controls", init_cx, 1920, nothing)
cv2.createTrackbar("Center Y", "Live ROI Controls", init_cy, 1080, nothing)
cv2.createTrackbar("Center W", "Live ROI Controls", init_cw, 1000, nothing)
cv2.createTrackbar("Center H", "Live ROI Controls", init_ch, 1080, nothing)

# Trackbars for Right Mirror ROI
cv2.createTrackbar("Right X", "Live ROI Controls", init_rx, 1920, nothing)
cv2.createTrackbar("Right Y", "Live ROI Controls", init_ry, 1080, nothing)
cv2.createTrackbar("Right W", "Live ROI Controls", init_rw, 1000, nothing)
cv2.createTrackbar("Right H", "Live ROI Controls", init_rh, 1080, nothing)

# Trackbar for Cylinder Radius R
cv2.createTrackbar("Radius R", "Live ROI Controls", init_r, 500, nothing)

print("🎥 Interactive Live Pipeline Started. Adjust sliders in real time.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Read active slider values
    lx = cv2.getTrackbarPos("Left X", "Live ROI Controls")
    ly = cv2.getTrackbarPos("Left Y", "Live ROI Controls")
    lw = cv2.getTrackbarPos("Left W", "Live ROI Controls")
    lh = cv2.getTrackbarPos("Left H", "Live ROI Controls")

    cx = cv2.getTrackbarPos("Center X", "Live ROI Controls")
    cy = cv2.getTrackbarPos("Center Y", "Live ROI Controls")
    cw = cv2.getTrackbarPos("Center W", "Live ROI Controls")
    ch = cv2.getTrackbarPos("Center H", "Live ROI Controls")

    rx = cv2.getTrackbarPos("Right X", "Live ROI Controls")
    ry = cv2.getTrackbarPos("Right Y", "Live ROI Controls")
    rw = cv2.getTrackbarPos("Right W", "Live ROI Controls")
    rh = cv2.getTrackbarPos("Right H", "Live ROI Controls")

    radius = max(1, cv2.getTrackbarPos("Radius R", "Live ROI Controls"))

    left_roi = (lx, ly, max(1, lw), max(1, lh))
    center_roi = (cx, cy, max(1, cw), max(1, ch))
    right_roi = (rx, ry, max(1, rw), max(1, rh))

    # Initialize dynamic pipeline with live radius
    pipeline = TestTubeBarcodePipeline(tube_radius_px=radius, blur_threshold=30.0)

    # Process active frame through unrolling and decoding
    result = pipeline.process_frame(frame, left_roi, center_roi, right_roi)

    # Annotate bounding boxes on live camera view
    preview = frame.copy()
    cv2.rectangle(preview, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)  # Blue = Left Mirror
    cv2.rectangle(preview, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)  # Green = Center Tube
    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)  # Red = Right Mirror

    if result["barcodes"]:
        for idx, bc in enumerate(result["barcodes"]):
            msg = f"✅ [{bc['engine']}]: {bc['text']}"
            cv2.putText(preview, msg, (30, 60 + (idx * 40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Display live feed and live unwrapped 360-degree strip
    cv2.imshow("Live Feed - Camera View", preview)
    if result["unwrapped_strip"] is not None:
        cv2.imshow("Live Unwrapped 360 Strip", result["unwrapped_strip"])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nFinal Calibrated Parameters:")
        print(f"LEFT_MIRROR_ROI  = {left_roi}")
        print(f"CENTER_TUBE_ROI  = {center_roi}")
        print(f"RIGHT_MIRROR_ROI = {right_roi}")
        print(f"TUBE_RADIUS_PX   = {radius}")
        break

cap.release()
cv2.destroyAllWindows()