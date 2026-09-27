# 🧪 360° Test Tube Barcode Unwrapper & Reader

A real-time computer vision pipeline designed to capture, unwrap (unroll), and decode cylindrical barcodes on test tubes using an Arducam and a dual-mirror optic setup. 

By applying geometric surface transformation to multi-view mirror reflections, this system eliminates edge-curvature distortion and reads $360^\circ$ barcode labels in **sub-20ms** without requiring mechanical tube rotation or heavy AI inference.

---

## ⚡ Key Features

* **360° Zero-Motion Capture:** Uses a 2-mirror V-configuration to capture direct front and side-reflected views in a single frame.
* **Cylindrical Surface Unwrapping:** Applies an inverse cylindrical mapping transformation ($x' = R \cdot \arcsin(x/R)$) in OpenCV to flatten curved label edges.
* **Sub-20ms Pipeline Latency:** Designed for high-throughput edge deployment running at 60+ FPS on CPU hardware.
* **Open-Source & Commercially Compliant:** Built using **Apache 2.0 / MIT** licensed libraries (`OpenCV`, `zxing-cpp`, `pyzbar`).
