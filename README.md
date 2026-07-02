# Edge 3D Perception — Jetson Orin Nano + Orbbec Astra Pro Plus

Real-time object detection with metric depth estimation, running fully on-device on an NVIDIA Jetson Orin Nano. No cloud, no offloading — CUDA-accelerated YOLOv8n inference fused with a depth-aligned RGB-D stream to report live distance-to-object in the viewport.

![Demo](assets/demo.gif.gif)

---

## Hardware & Environment

| Component | Detail |
|---|---|
| Compute | NVIDIA Jetson Orin Nano — JetPack 6.2 (Ubuntu 22.04 LTS, CUDA 12.6) |
| Sensor | Orbbec Astra Pro Plus (legacy OpenNI protocol) — Depth PID `0x060f`, RGB PID `0x050f`, VID `2bc5` |
| Model | YOLOv8n (Ultralytics), CUDA-accelerated inference |
| SDK | `pyorbbecsdk`, compiled from the `main` (v1-compatible) branch |
| Python env | Isolated venv at `~/pyorbbecsdk/venv` |

### Key dependency notes
- **PyTorch + Torchvision**: installed via Jetson-optimized `cu126` wheels (not pip-default builds, which aren't built for the Orin's Tegra architecture).
- **`nvidia-cudss-cu12`**: installed with `--no-deps` to supply `libcudss.so.0` without clobbering the system's existing cuBLAS stack — a naive install pulls in a cuBLAS version mismatch that breaks CUDA inference silently.
- **`pyorbbecsdk`**: had to be built from the `main` branch specifically. The default v2 branch drops support for the Astra's OpenNI protocol entirely, which would have made the sensor unusable.

---

## Architecture

### GPU-accelerated inference + depth fusion
Per frame (640x480 RGBD):
1. RGB frame → CUDA-accelerated YOLOv8n (Ultralytics) → 2D bounding boxes.
2. Bounding box center `(cx, cy)` computed per detection.
3. **Boundary clamping** applied to `(cx, cy)` before any array indexing — objects that are detected while partially or fully off the visible frame edge (fast motion, occlusion, edge-of-frame exit) would otherwise generate an out-of-bounds index into the depth array and crash the inference loop. Clamping guarantees every lookup stays inside the allocated depth buffer.
4. The 16-bit depth value at the clamped `(cx, cy)` is sampled and converted from millimeters to meters.
5. Distance is overlaid on the live viewport next to each detection.

---

## Repository Contents

| File | Description |
|---|---|
| `edge_3d_perception.py` | Main pipeline: sensor init, alignment, inference, depth fusion, overlay rendering |
| `demo.mp4` | Raw ~10–15s recorded demo of the pipeline running live on the Jetson |
| `demo.gif` | Compressed version of the above, embedded at the top of this README |

---

## Running It

```bash
# Activate the isolated environment
source ~/pyorbbecsdk/venv/bin/activate

# Run the pipeline
python edge_3d_perception.py
```

Requires JetPack 6.2, a connected Astra Pro Plus, and the dependencies noted above already installed in the venv.

---