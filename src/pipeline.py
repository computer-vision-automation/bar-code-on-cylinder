import cv2
import numpy as np
from pyzbar import pyzbar

# Robust import check for ZXing
try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    try:
        import zxing
        HAS_ZXING_LEGACY = True
        HAS_ZXING = False
    except ImportError:
        HAS_ZXING = False
        HAS_ZXING_LEGACY = False

class TestTubeBarcodePipeline:
    def __init__(self, tube_radius_px=160, blur_threshold=100.0):
        """
        :param tube_radius_px: Estimated radius of test tube in camera frame (pixels)
        :param blur_threshold: Minimum Laplacian variance to accept frame
        """
        self.R = tube_radius_px
        self.blur_threshold = blur_threshold

    def is_sharp(self, frame):
        """Quality Gate: Checks for motion blur using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= self.blur_threshold, variance

    def preprocess_for_decoding(self, image):
        """Enhance contrast and reduce glare for glossy test tubes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Sharpening filter
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened

    def unwrap_cylinder_roi(self, roi):
        """
        Applies cylindrical inverse mapping x' = R * arcsin(x / R)
        to flatten the curved surface of the test tube frame.
        """
        h, w = roi.shape[:2]
        center_x = w / 2.0

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            map_y[y, :] = y

        for x in range(w):
            dx = (x - center_x) / self.R
            if abs(dx) <= 1.0:
                x_source = center_x + self.R * np.arcsin(dx)
                map_x[:, x] = x_source
            else:
                map_x[:, x] = x

        return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    def blend_strips(self, strip1, strip2, blend_width=15):
        """Blends two image strips horizontally over a transition zone using alpha gradients."""
        h, w1 = strip1.shape[:2]
        w2 = strip2.shape[1]
        
        blend_width = min(blend_width, w1, w2)
        
        # Create 1D alpha ramp: 1.0 down to 0.0 across overlap columns
        alpha = np.linspace(1.0, 0.0, blend_width).astype(np.float32)
        
        # Expand dimensions for matrix broadcast depending on color channels
        if len(strip1.shape) == 3:
            alpha = alpha[np.newaxis, :, np.newaxis]  # Shape: (1, blend_width, 1)
        else:
            alpha = alpha[np.newaxis, :]               # Shape: (1, blend_width)
            
        beta = 1.0 - alpha
        
        # Construct output canvas
        blended_w = w1 + w2 - blend_width
        blended_shape = (h, blended_w, 3) if len(strip1.shape) == 3 else (h, blended_w)
        blended = np.zeros(blended_shape, dtype=strip1.dtype)
        
        # Copy non-overlapping left region
        blended[:, :w1 - blend_width] = strip1[:, :-blend_width]
        
        # Vectorized alpha blend across overlap region
        overlap1 = strip1[:, w1 - blend_width:].astype(np.float32)
        overlap2 = strip2[:, :blend_width].astype(np.float32)
        blended_overlap = overlap1 * alpha + overlap2 * beta
        blended[:, w1 - blend_width:w1] = np.clip(blended_overlap, 0, 255).astype(np.uint8)
        
        # Copy non-overlapping right region
        blended[:, w1:] = strip2[:, blend_width:]
        
        return blended

    def stitch_views(self, left_reflection, center_view, right_reflection):
        """
        Flips mirror views, applies cylindrical unwrapping, and seamlessly
        blends the 3 viewpoints into a single flat 360-degree strip.
        """
        left_flipped = cv2.flip(left_reflection, 1)
        right_flipped = cv2.flip(right_reflection, 1)

        u_left = self.unwrap_cylinder_roi(left_flipped)
        u_center = self.unwrap_cylinder_roi(center_view)
        u_right = self.unwrap_cylinder_roi(right_flipped)

        # Blend Left + Center, then blend the combined strip with Right
        left_center_blended = self.blend_strips(u_left, u_center, blend_width=15)
        full_stitched = self.blend_strips(left_center_blended, u_right, blend_width=15)

        return full_stitched

    def decode_barcode(self, image):
        """Multi-engine decoder with contrast enhancement."""
        processed_gray = self.preprocess_for_decoding(image)

        # 1. ZXing-CPP Engine
        if HAS_ZXING:
            zx_results = zxingcpp.read_barcodes(processed_gray)
            if zx_results:
                return [{"engine": "zxing-cpp", "text": r.text, "format": str(r.format)} for r in zx_results]

        # 2. PyZBar Engine
        pyz_results = pyzbar.decode(processed_gray)
        if pyz_results:
            return [{"engine": "pyzbar", "text": r.data.decode('utf-8'), "format": r.type} for r in pyz_results]

        # Fallback pass on original non-preprocessed grayscale if contrast tuning over-sharpened
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if HAS_ZXING:
            zx_results = zxingcpp.read_barcodes(gray)
            if zx_results:
                return [{"engine": "zxing-cpp", "text": r.text, "format": str(r.format)} for r in zx_results]
        
        pyz_results = pyzbar.decode(gray)
        if pyz_results:
            return [{"engine": "pyzbar", "text": r.data.decode('utf-8'), "format": r.type} for r in pyz_results]

        return []

    def process_frame(self, frame, left_roi_box, center_roi_box, right_roi_box):
        """
        Full Pipeline Execution Entry Point.
        ROI Format: (x, y, w, h)
        """
        sharp, var = self.is_sharp(frame)
        if not sharp:
            return {"status": "REJECTED_BLUR", "variance": var, "barcodes": [], "unwrapped_strip": None}

        lx, ly, lw, lh = left_roi_box
        cx, cy, cw, ch = center_roi_box
        rx, ry, rw, rh = right_roi_box

        left_crop = frame[ly:ly+lh, lx:lx+lw]
        center_crop = frame[cy:cy+ch, cx:cx+cw]
        right_crop = frame[ry:ry+rh, rx:rx+rw]

        flat_strip = self.stitch_views(left_crop, center_crop, right_crop)
        barcodes = self.decode_barcode(flat_strip)

        return {
            "status": "SUCCESS" if barcodes else "DECODE_FAILED",
            "variance": var,
            "barcodes": barcodes,
            "unwrapped_strip": flat_strip
        }