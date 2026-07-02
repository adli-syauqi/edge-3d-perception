# Edge 3D Perception — Jetson Orin Nano + Orbbec Astra Pro Plus

Real-time object detection with metric depth estimation, running fully on-device on an NVIDIA Jetson Orin Nano. No cloud, no offloading — CUDA-accelerated YOLOv8n inference fused with a depth-aligned RGB-D stream to report live distance-to-object in the viewport.

![Demo](assets/demo.gif.gif)

---

## Why this project

Most "edge AI" demos stop at running a model on a small board. This one goes a layer deeper: it fuses a color and a depth sensor stream on a legacy OpenNI-protocol camera that upstream SDKs have effectively deprecated, aligns them in software, and pushes inference through the Orin's CUDA cores — while handling the pointer and memory-safety edge cases that come with running inference on a live, unpredictable video feed.

It's meant to sit alongside an embedded/hardware-focused resume as evidence of applied computer vision and systems-level GPU work, not just model training.

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

### 1. Explicit stream profile allocation
The Astra Pro Plus doesn't play well with generic stream enumeration — it throws on naive config calls. Stream profiles are instead queried explicitly per sensor type (`OBSensorType.COLOR_SENSOR`, `OBSensorType.DEPTH_SENSOR`) and matched against hardware-supported configurations before the pipeline starts.

### 2. Software-mode depth/RGB registration
This camera has no hardware frame-sync over its USB 2.0 endpoint, so hardware alignment isn't available. The pipeline instead forces:

```python
config.set_align_mode(OBAlignMode.SW_MODE)
```

which warps the depth coordinate space onto the RGB pixel grid in software before frames are fused, so every detected bounding box maps to a depth-accurate pixel region.

### 3. Unified Memory Architecture (UMA) for zero-copy inference
The Orin Nano's CPU and GPU share a single physical memory pool. Rather than treating this as a discrete-GPU system (host → PCIe → device memcpy), the pipeline passes frame buffers directly to the CUDA-accelerated YOLOv8n model without an explicit host-to-device transfer step. On a discrete-GPU system this transfer is often the dominant latency cost in a video inference loop; on the Orin, UMA lets the model read frame memory in place. This is the difference between "runs a model" and "understands the platform it's deploying to."

### 4. GPU-accelerated inference + depth fusion
Per frame:
1. BGR frame → CUDA-accelerated YOLOv8n → 2D bounding boxes.
2. Bounding box center `(cx, cy)` computed per detection.
3. **Boundary clamping** applied to `(cx, cy)` before any array indexing — objects that are detected while partially or fully off the visible frame edge (fast motion, occlusion, edge-of-frame exit) would otherwise generate an out-of-bounds index into the depth array and crash the inference loop. Clamping guarantees every lookup stays inside the allocated depth buffer.
4. The 16-bit depth value at the clamped `(cx, cy)` is sampled and converted from millimeters to meters.
5. Distance is overlaid on the live viewport next to each detection.

This clamping is a small piece of code but it's the difference between a demo that survives a real, jittery camera feed and one that segfaults the first time a person walks out of frame.

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

## Roadmap / Stretch Goals

- [ ] Export YOLOv8n to TensorRT (`.engine`) format via NVIDIA's export tooling for lower-latency inference on the Orin Nano.
- [ ] Benchmark PyTorch vs. TensorRT inference latency and publish the comparison.
- [ ] Extend depth sampling from a single center pixel to a small ROI median, reducing sensitivity to single-pixel depth noise.

---

## Background

Built as a portfolio project to pair computer vision / edge AI experience with an existing embedded systems and hardware background, for robotics-focused internship applications.
