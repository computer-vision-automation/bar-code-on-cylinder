import cv2
import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from pipeline import TestTubeBarcodePipeline

CANDIDATE_NAMES = ["sample_tube_jpg.jpeg", "sample_tube.jpg.jpeg", "sample_tube.jpeg"]
IMAGE_PATH = None
for name in CANDIDATE_NAMES:
    p = os.path.join(CURRENT_DIR, name)
    if os.path.exists(p):
        IMAGE_PATH = p
        break
if IMAGE_PATH is None:
    matches = glob.glob(os.path.join(CURRENT_DIR, "*.jpg")) + glob.glob(os.path.join(CURRENT_DIR, "*.jpeg"))
    if matches:
        IMAGE_PATH = matches[0]

frame = cv2.imread(IMAGE_PATH) if IMAGE_PATH else None

if frame is None:
    print(f"❌ Could not find/load a sample image next to this script (tried {CANDIDATE_NAMES}).")
    sys.exit(1)

print(f"Loaded calibration image: {IMAGE_PATH}")

h, w = frame.shape[:2]

WINDOW_NAME = "ROI & Radius Tuner"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1100, 750)

def nothing(x):
    pass

cv2.createTrackbar("L_X", WINDOW_NAME, int(w * 0.15), w, nothing)
cv2.createTrackbar("L_Y", WINDOW_NAME, int(h * 0.12), h, nothing)
cv2.createTrackbar("L_W", WINDOW_NAME, int(w * 0.10), w, nothing)
cv2.createTrackbar("L_H", WINDOW_NAME, int(h * 0.55), h, nothing)
cv2.createTrackbar("L_Radius", WINDOW_NAME, 80, 400, nothing)

cv2.createTrackbar("C_X", WINDOW_NAME, int(w * 0.40), w, nothing)
cv2.createTrackbar("C_Y", WINDOW_NAME, int(h * 0.15), h, nothing)
cv2.createTrackbar("C_W", WINDOW_NAME, int(w * 0.11), w, nothing)
cv2.createTrackbar("C_H", WINDOW_NAME, int(h * 0.68), h, nothing)
cv2.createTrackbar("C_Radius", WINDOW_NAME, 85, 400, nothing)

cv2.createTrackbar("R_X", WINDOW_NAME, int(w * 0.65), w, nothing)
cv2.createTrackbar("R_Y", WINDOW_NAME, int(h * 0.06), h, nothing)
cv2.createTrackbar("R_W", WINDOW_NAME, int(w * 0.14), w, nothing)
cv2.createTrackbar("R_H", WINDOW_NAME, int(h * 0.52), h, nothing)
cv2.createTrackbar("R_Radius", WINDOW_NAME, 110, 400, nothing)

pipeline = TestTubeBarcodePipeline(blur_threshold=1.0)

print("-------------------------------------------------------")
print("🎛️  ROI CALIBRATION TOOL STARTED")
print("1. Adjust sliders until each box tightly frames its own tube view.")
print("   Left/Center/Right now have INDEPENDENT X/Y/W/H - they no longer")
print("   have to share the same vertical position or size.")
print("2. Set each *_Radius slider to 0 to auto-estimate from crop width,")
print("   or dial in a manual value if you know the true optical radius.")
print("3. Check the 'Unwrapped 360-Degree Flat Strip' window.")
print("4. Press 's' to PRINT the exact coordinates for test_static.py.")
print("5. Press 'q' or ESC to quit.")
print("-------------------------------------------------------")

for _ in range(5):
    cv2.waitKey(1)

while True:

    lx = cv2.getTrackbarPos("L_X", WINDOW_NAME)
    ly = cv2.getTrackbarPos("L_Y", WINDOW_NAME)
    lw = max(1, cv2.getTrackbarPos("L_W", WINDOW_NAME))
    lh = max(1, cv2.getTrackbarPos("L_H", WINDOW_NAME))
    l_radius = cv2.getTrackbarPos("L_Radius", WINDOW_NAME)

    cx = cv2.getTrackbarPos("C_X", WINDOW_NAME)
    cy = cv2.getTrackbarPos("C_Y", WINDOW_NAME)
    cw = max(1, cv2.getTrackbarPos("C_W", WINDOW_NAME))
    ch = max(1, cv2.getTrackbarPos("C_H", WINDOW_NAME))
    c_radius = cv2.getTrackbarPos("C_Radius", WINDOW_NAME)

    rx = cv2.getTrackbarPos("R_X", WINDOW_NAME)
    ry = cv2.getTrackbarPos("R_Y", WINDOW_NAME)
    rw = max(1, cv2.getTrackbarPos("R_W", WINDOW_NAME))
    rh = max(1, cv2.getTrackbarPos("R_H", WINDOW_NAME))
    r_radius = cv2.getTrackbarPos("R_Radius", WINDOW_NAME)

    l_roi = (lx, ly, lw, lh)
    c_roi = (cx, cy, cw, ch)
    r_roi = (rx, ry, rw, rh)

    result = pipeline.process_frame(
        frame, l_roi, c_roi, r_roi,
        left_radius=l_radius or None,
        center_radius=c_radius or None,
        right_radius=r_radius or None,
    )

    preview = frame.copy()
    cv2.rectangle(preview, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
    cv2.rectangle(preview, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

    cv2.putText(preview, f"status: {result['status']} (var={result['variance']:.1f})",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    if result["barcodes"]:
        for idx, bc in enumerate(result["barcodes"]):
            text = f"FOUND [{bc['engine']}]: {bc['text']}"
            cv2.putText(preview, text, (20, 40 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, preview)

    if result["unwrapped_strip"] is not None:
        cv2.imshow("Unwrapped 360-Degree Flat Strip", result["unwrapped_strip"])

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        print("\n✅ SAVED ROI COORDINATES:")
        print(f"LEFT_MIRROR_ROI  = {l_roi}   (radius={l_radius or 'auto'})")
        print(f"CENTER_TUBE_ROI  = {c_roi}   (radius={c_radius or 'auto'})")
        print(f"RIGHT_MIRROR_ROI = {r_roi}   (radius={r_radius or 'auto'})\n")
    elif key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()