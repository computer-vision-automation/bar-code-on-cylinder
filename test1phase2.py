
import argparse
from pathlib import Path

import cv2
import numpy as np


WINDOW = "Mirror Flattening Tuner"


def unwrap(roi, center_shift, width_pct, k, half_angle_deg):
    """Flatten a curved cylindrical label using the perspective model."""
    h, w = roi.shape[:2]

    cx = w / 2.0 + center_shift
    W = (w / 2.0) * width_pct / 100.0
    k = max(1.05, k)

    half = np.deg2rad(half_angle_deg)

    px_per_rad = W * np.sqrt((k + 1) / (k - 1))
    out_w = max(32, int(round(2 * half * px_per_rad)))

    theta = np.linspace(
        -half, half, out_w, dtype=np.float32
    )

    x_src = (
        cx
        + W * np.sqrt(k * k - 1)
        * np.sin(theta) / (k - np.cos(theta))
    )

    # Correct vertical scaling caused by perspective.
    scale = (k - 1) / (k - np.cos(theta))
    cy = h / 2.0

    ys = np.arange(h, dtype=np.float32)[:, None]

    map_y = (
        cy + (ys - cy) * scale[None, :]
    ).astype(np.float32)

    map_x = np.tile(x_src[None, :], (h, 1))

    return cv2.remap(
        roi,
        map_x.astype(np.float32),
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def rotate(img, degrees):
    if abs(degrees) < 0.05:
        return img

    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2), degrees, 1.0
    )

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def enhance(img):
    gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    ) if img.ndim == 3 else img

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)

    return cv2.addWeighted(
        gray, 1.35, blurred, -0.35, 0
    )


def select_roi(image, title):
    print(f"\nSelect the tube in the {title}.")
    print("Keep the ROI edges aligned with the tube silhouette.")

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)

    roi_rect = cv2.selectROI(
        "Select ROI", image, False, False
    )

    cv2.destroyWindow("Select ROI")

    x, y, w, h = map(int, roi_rect)

    if w <= 0 or h <= 0:
        raise SystemExit("ROI selection cancelled.")

    return image[y:y + h, x:x + w].copy()


def tune_and_save(roi, name, output_dir):
    h, w = roi.shape[:2]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def nothing(_):
        pass

    cv2.createTrackbar("Tilt x10", WINDOW, 300, 600, nothing)
    cv2.createTrackbar("Center shift", WINDOW, w // 2, w, nothing)
    cv2.createTrackbar("Width %", WINDOW, 100, 130, nothing)
    cv2.createTrackbar("k x10", WINDOW, 35, 150, nothing)
    cv2.createTrackbar("Half angle", WINDOW, 60, 85, nothing)

    cv2.setTrackbarMin("Width %", WINDOW, 70)
    cv2.setTrackbarMin("k x10", WINDOW, 11)
    cv2.setTrackbarMin("Half angle", WINDOW, 10)

    print(f"\nTuning: {name}")
    print("S = save this view | Q = quit")

    while True:
        tilt = (
            cv2.getTrackbarPos("Tilt x10", WINDOW) - 300
        ) / 10.0

        shift = (
            cv2.getTrackbarPos("Center shift", WINDOW)
            - w // 2
        )

        width_pct = cv2.getTrackbarPos("Width %", WINDOW)
        k = cv2.getTrackbarPos("k x10", WINDOW) / 10.0
        half_angle = cv2.getTrackbarPos("Half angle", WINDOW)

        corrected = rotate(roi, tilt)

        flat = unwrap(
            corrected, shift, width_pct, k, half_angle
        )

        scale = min(
            1100 / flat.shape[1],
            650 / flat.shape[0],
            1.0
        )

        preview = (
            cv2.resize(
                flat, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )
            if scale < 1.0 else flat
        )

        cv2.imshow(WINDOW, preview)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord("s"), ord("S")):
            output_dir.mkdir(parents=True, exist_ok=True)

            flat_path = output_dir / f"{name}_flat.png"
            enhanced_path = output_dir / f"{name}_enhanced.png"

            cv2.imwrite(str(flat_path), flat)
            cv2.imwrite(str(enhanced_path), enhance(flat))

            print(f"Saved: {flat_path}")
            print(f"Saved: {enhanced_path}")
            print(
                f"Parameters: tilt={tilt}, shift={shift}, "
                f"width={width_pct}, k={k}, angle={half_angle}"
            )
            break

        elif key in (ord("q"), ord("Q"), 27):
            print(f"Skipped saving {name}.")
            break

    cv2.destroyWindow(WINDOW)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--out", default="phase2_output",
        help="Output directory"
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    output_dir = Path(args.out)

    # The order here determines the processing order.
    views = [
        ("left_mirror", True),
        ("middle_direct", False),
        ("right_mirror", True),
    ]

    for name, flip_horizontal in views:
        roi = select_roi(image, name)

        if flip_horizontal:
            # Correct the left-right reversal in a plane-mirror view.
            roi = cv2.flip(roi, 1)

        tune_and_save(roi, name, output_dir)

    print("\nPhase 2 processing finished.")
    print(f"Check the output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()