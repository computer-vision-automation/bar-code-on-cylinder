"""
test1phase1.py - perspective-correct cylinder unwrapping for labels / tubes

pip install opencv-python numpy
python test1phase1.py curved_paper1.jpg.jpeg

1. Select ROI: LEFT/RIGHT edges must touch the paper/tube silhouette.
2. Tune sliders until text is straight and evenly sized.
Keys: S save | R reselect ROI | Q/Esc quit
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

WIN = "Flattening tuner"


def unwrap(roi, center_shift, width_pct, k, half_angle_deg):
    h, w = roi.shape[:2]
    cx = w / 2.0 + center_shift
    W = (w / 2.0) * width_pct / 100.0       # half-width of silhouette (px)
    k = max(1.05, k)                         # camera distance / radius
    half = np.deg2rad(half_angle_deg)

    # Horizontal: perspective cylinder model
    px_per_rad = W * np.sqrt((k + 1) / (k - 1))
    out_w = max(32, int(round(2 * half * px_per_rad)))
    theta = np.linspace(-half, half, out_w, dtype=np.float32)
    x_src = cx + W * np.sqrt(k * k - 1) * np.sin(theta) / (k - np.cos(theta))

    # Vertical: edges are farther away, so they look smaller -> enlarge them
    scale = (k - 1) / (k - np.cos(theta))
    cy = h / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    map_y = (cy + (ys - cy) * scale[None, :]).astype(np.float32)
    map_x = np.tile(x_src[None, :], (h, 1)).astype(np.float32)

    return cv2.remap(roi, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def rotate(img, deg):
    if abs(deg) < 0.05:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.35, blur, -0.35, 0)


def pick_roi(image):
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    x, y, w, h = map(int, cv2.selectROI("ROI", image, False, False))
    cv2.destroyWindow("ROI")
    if w <= 0 or h <= 0:
        raise SystemExit("No ROI selected.")
    return image[y:y + h, x:x + w].copy(), (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=r"D:\tech\github2\curved_paper1.jpg.jpeg")
    ap.add_argument("--out", default="flattened_result")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    roi, origin = pick_roi(image)
    rh, rw = roi.shape[:2]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    nop = lambda _: None
    cv2.createTrackbar("tilt (deg)", WIN, 300, 600, nop)             # (v-300)/10
    cv2.createTrackbar("center_shift", WIN, rw // 4, rw // 2, nop)   # v - rw/4
    cv2.createTrackbar("width_pct", WIN, 100, 130, nop)
    cv2.createTrackbar("k x10", WIN, 35, 150, nop)                   # k = 3.5
    cv2.createTrackbar("half_angle", WIN, 60, 85, nop)
    cv2.setTrackbarMin("width_pct", WIN, 70)
    cv2.setTrackbarMin("k x10", WIN, 11)
    cv2.setTrackbarMin("half_angle", WIN, 10)

    while True:
        tilt = (cv2.getTrackbarPos("tilt (deg)", WIN) - 300) / 10.0
        shift = cv2.getTrackbarPos("center_shift", WIN) - rw // 4
        wp = cv2.getTrackbarPos("width_pct", WIN)
        k = cv2.getTrackbarPos("k x10", WIN) / 10.0
        ha = cv2.getTrackbarPos("half_angle", WIN)

        flat = unwrap(rotate(roi, tilt), shift, wp, k, ha)

        s = min(1100 / flat.shape[1], 650 / flat.shape[0], 1.0)
        prev = cv2.resize(flat, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else flat
        cv2.imshow(WIN, prev)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("s"), ord("S")):
            p = Path(args.out)
            cv2.imwrite(str(p.with_name(p.name + "_flat.png")), flat)
            cv2.imwrite(str(p.with_name(p.name + "_enhanced.png")), enhance(flat))
            print(f"Saved. tilt={tilt}, shift={shift}, width_pct={wp}, k={k}, half_angle={ha}")
            print(f"ROI origin={origin}, size=({rw},{rh})")
            break
        elif key in (ord("r"), ord("R")):
            roi, origin = pick_roi(image)
            rh, rw = roi.shape[:2]
            cv2.setTrackbarMax("center_shift", WIN, rw // 2)
        elif key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()