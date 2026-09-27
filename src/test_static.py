import cv2
import os
from pipeline import TestTubeBarcodePipeline

# Initialize pipeline
pipeline = TestTubeBarcodePipeline(tube_radius_px=150, blur_threshold=50.0)

# Path to your sample image
IMAGE_PATH = "test_images/sample_tube.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image file '{IMAGE_PATH}' not found!")
    print("Please place a test image of a test tube in the 'test_images' folder.")
    exit()

# Load test image
frame = cv2.imread(IMAGE_PATH)
h, w, _ = frame.shape
print(f"Loaded image: {w}x{h} px")

# Set estimated ROI bounding boxes: (x, y, width, height)
# Adjust these relative to your image dimensions!
LEFT_MIRROR_ROI  = (int(w * 0.1),  int(h * 0.2), int(w * 0.25), int(h * 0.6))
CENTER_TUBE_ROI  = (int(w * 0.38), int(h * 0.2), int(w * 0.25), int(h * 0.6))
RIGHT_MIRROR_ROI = (int(w * 0.65), int(h * 0.2), int(w * 0.25), int(h * 0.6))

# Execute processing pipeline
result = pipeline.process_frame(
    frame, 
    LEFT_MIRROR_ROI, 
    CENTER_TUBE_ROI, 
    RIGHT_MIRROR_ROI
)

print(f"Status: {result['status']} | Sharpness Variance: {result['variance']:.2f}")

# Display Barcode Results
if result["barcodes"]:
    for bc in result["barcodes"]:
        print(f"✅ DECODED [{bc['engine']}]: {bc['text']} ({bc['format']})")
else:
    print("❌ No barcode decoded. Check ROI boxes or image lighting.")

# Show original with drawn ROI boxes and the unwrapped flat strip
preview = frame.copy()
for (x, y, rw, rh), color in zip(
    [LEFT_MIRROR_ROI, CENTER_TUBE_ROI, RIGHT_MIRROR_ROI], 
    [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
):
    cv2.rectangle(preview, (x, y), (x + rw, y + rh), color, 2)

cv2.imshow("Input Frame with ROIs", preview)

if result["unwrapped_strip"] is not None:
    cv2.imshow("Unwrapped 360-Degree Flat Strip", result["unwrapped_strip"])

print("Press any key on the image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()