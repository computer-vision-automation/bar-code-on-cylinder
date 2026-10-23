# 🧪 360° Test Tube Barcode Unwrapper & Reader

A real-time computer vision pipeline designed to capture, unwrap (unroll), and decode cylindrical barcodes on test tubes using an Arducam and a dual-mirror optic setup. 

By applying geometric surface transformation to multi-view mirror reflections, this system eliminates edge-curvature distortion and reads $360^\circ$ barcode labels in **sub-20ms** without requiring mechanical tube rotation or heavy AI inference.

---

## ⚡ Key Features

* **360° Zero-Motion Capture:** Uses a 2-mirror V-configuration to capture direct front and side-reflected views in a single frame.
* **Cylindrical Surface Unwrapping:** Applies an inverse cylindrical mapping transformation ($x' = R \cdot \arcsin(x/R)$) in OpenCV to flatten curved label edges.
* **Sub-20ms Pipeline Latency:** Designed for high-throughput edge deployment running at 60+ FPS on CPU hardware.
* **Open-Source & Commercially Compliant:** Built using **Apache 2.0 / MIT** licensed libraries (`OpenCV`, `zxing-cpp`, `pyzbar`).
# Barcode on Cylinder — Curved Label Flattening & Mirror Correction

A computer vision project for processing barcode labels attached to cylindrical objects, such as laboratory test tubes. The goal is to transform curved label images into flatter, more readable images that can be used for downstream barcode detection and decoding.

## Overview

Barcodes printed on cylindrical surfaces can appear curved, distorted, or difficult to read because of surface curvature, perspective, and mirror reflections.

This project explores a two-phase image-processing pipeline:

1. **Phase 1 — Curved Label Flattening:** Corrects the apparent curvature of a label using a perspective-based cylindrical unwrapping model.
2. **Phase 2 — Mirror Correction & Flattening:** Corrects horizontally reversed mirror views, preserves the orientation of the direct middle view, and flattens each selected view independently.

The project uses OpenCV and NumPy for image transformation and enhancement.

## Features

* Interactive region-of-interest (ROI) selection.
* Perspective-based cylindrical label unwrapping.
* Adjustable parameters for tilt, horizontal alignment, width, perspective, and viewing angle.
* Horizontal flipping to correct reversed mirror views.
* Independent processing of left mirror, middle direct, and right mirror images.
* Contrast enhancement using CLAHE and sharpening.
* Saving of flattened and enhanced output images.
* A foundation for future barcode detection and decoding integration.

## Processing Pipeline

```text
Input Image
    |
    v
Select Label / Tube ROI
    |
    v
Identify View
    |
    +-----------------------+
    |                       |
Mirror View             Direct View
    |                       |
Horizontal Flip          Keep Orientation
    |                       |
    +-----------+-----------+
                |
                v
     Cylindrical Unwrapping
                |
                v
       Flattened Label Image
                |
                v
       Contrast Enhancement
                |
                v
       Save Output Images
```

## Project Files

| File             | Description                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `test2phase1.py` | Phase 1: Curved label flattening using a perspective-based cylindrical unwrapping model.                               |
| `test1phase2.py` | Phase 2: Mirror-image orientation correction and flattening of the left mirror, middle direct, and right mirror views. |

## Requirements

* Python 3.9 or later recommended
* OpenCV
* NumPy

Install the dependencies:

```bash
pip install opencv-python numpy
```

## Usage

Run the relevant script from your terminal.

### Phase 1 — Curved Label Flattening

```bash
python test2phase1.py "path/to/input_image.jpg"
```

Select the region containing the curved label. Adjust the available parameters until the label appears as straight and evenly scaled as possible.

### Phase 2 — Mirror Correction and Flattening

```bash
python test1phase2.py "path/to/input_image.jpg"
```

The intended workflow processes three views:

* **Left mirror:** Horizontally flip the reflected image, then flatten it.
* **Middle direct view:** Preserve its original orientation, then flatten it.
* **Right mirror:** Horizontally flip the reflected image, then flatten it.

Select the appropriate ROI for each view and tune its parameters independently.

> Note: The commands above assume that the scripts accept an input-image argument. Check each script's argument parser for the exact command syntax if it differs.

## Interactive Controls

The flattening tuner uses OpenCV trackbars to adjust the transformation.

Typical controls include:

* **Tilt:** Corrects the label's angular alignment.
* **Center shift:** Adjusts horizontal alignment.
* **Width:** Adjusts the estimated cylindrical silhouette width.
* **Perspective parameter (`k`):** Controls the perspective model.
* **Half angle:** Controls the angular range used for unwrapping.

Use the controls to obtain the best visual result. Press `S` to save when supported by the script, and use the other displayed keyboard controls to quit or reselect the ROI.

## Output

Depending on the script and its configuration, the pipeline produces:

* Flattened label images.
* Contrast-enhanced grayscale images.
* Separate outputs for the left mirror, middle direct view, and right mirror.

Enhanced images may improve readability, but enhancement cannot recover information that is absent because of blur, glare, occlusion, or insufficient resolution.

## Technical Approach

### 1. Region of Interest

The relevant tube or label is selected from the input image to isolate the area to be transformed.

### 2. Mirror Orientation Correction

A horizontal flip is applied to reflected views when required to correct reversed text or barcode orientation. The direct middle view is not flipped.

### 3. Cylindrical Unwrapping

A perspective-based geometric model maps output pixels to corresponding locations in the original image. The mapping approximates the cylindrical surface and compensates for the apparent horizontal curvature and perspective-related vertical scaling.

OpenCV's `remap()` function applies this mapping to generate the flattened image.

### 4. Image Enhancement

CLAHE improves local contrast, while a sharpening operation emphasizes edges. This can help make barcode bars and text more distinguishable.

## Current Scope and Limitations

The current work focuses on **still-image processing**, not real-time camera tracking.

* The flattening model is an approximation and may require manual calibration.
* Each view may need different transformation parameters.
* A horizontal flip corrects common plane-mirror reversals, but the correct orientation should be verified for the actual optical setup.
* The three flattened views are processed separately; automatic alignment and stitching into a single composite barcode image are future work.
* Barcode detection and decoding should be evaluated separately on the resulting images.
* Image quality depends on focus, illumination, reflections, resolution, and the visible portion of the barcode.

## Future Improvements

* Automatic tube and barcode ROI detection.
* Automatic estimation of cylindrical geometry and flattening parameters.
* Alignment and stitching of overlapping mirror views.
* Barcode detection and decoding integration.
* Quantitative evaluation of flattening quality and decoding success.
* Batch processing and an optional live-camera workflow.

## Applications

Potential applications include laboratory test-tube identification, cylindrical product-label processing, and barcode image preprocessing for machine-vision systems.

## License

Add the license appropriate to this repository before redistributing or using the project under a specific open-source license.

---

**Project focus:** Computer vision, geometric image transformation, mirror-view correction, and barcode image preprocessing.
