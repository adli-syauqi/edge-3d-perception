# Edge 3D Perception: Real-Time Spatial Object Detection

A lightweight, real-time spatial computer vision pipeline designed for edge computing environments. This project integrates a 2D deep learning object detector (**YOLOv8**) with an RGB-D structured-light camera (**Orbbec Astra Pro Plus**) on the **NVIDIA Jetson Orin Nano** platform to achieve accelerated real-time 3D bounding boxes and center-point spatial depth localization.

![System Demo](assets/demo.gif)

---

## 🛠️ Hardware Stack
* **Compute:** NVIDIA Jetson Orin Nano (8GB Developer Kit)
    * **OS/Environment:** JetPack 6.2 (Ubuntu 22.04 LTS, CUDA 12.6)
* **Sensor:** Orbbec Astra Pro Plus (Structured Light RGB-D)
    * **Protocol:** Legacy OpenNI 

---

## 🏗️ System Architecture & Implementation

### 1. Unified Memory Optimization (UMA)
Traditional desktop vision pipelines suffer from latency bottlenecks caused by copying high-resolution image arrays from Host System RAM over a slow PCIe bus into Discrete GPU VRAM. 

Because the Jetson Orin Nano utilizes a **Unified Memory Architecture**, the CPU and Maxwell/Ampere GPU share the same physical memory pool. This pipeline leverages Zero-Copy memory pointers; raw video frames ingested by the Orbbec SDK are transformed directly into PyTorch/CUDA tensors without array duplication, resulting in low inference latency directly on the edge.

### 2. Software Depth-to-Color Alignment (Registration)
Because the structured-light depth projector and the RGB sensor are physically offset on the camera chassis, their image planes do not inherently align. Due to the legacy USB 2.0 endpoints of the Astra Pro Plus, hardware-level synchronization is unavailable. 

This project solves this by employing the Orbbec SDK's software registration layer (`OBAlignMode.SW_MODE`). The system queries factory calibration matrices (intrinsics and extrinsics) to warp the depth matrix frame-by-frame, mapping every 16-bit depth pixel precisely to its corresponding 8-bit RGB coordinate before running inference.

---

## 🚀 Installation & Setup

### 1. System Dependencies & SDK Configuration
Build the legacy Orbbec SDK C++ and Python bindings (v1 branch) on the Jetson:
```bash
# Clone the v1-compatible branch explicitly
git clone -b main [https://github.com/orbbec/pyorbbecsdk.git](https://github.com/orbbec/pyorbbecsdk.git)
cd pyorbbecsdk

# Setup environment and compile pybind11 wrappers
python3 -m venv venv
source venv/bin/activate
pip install pybind11
mkdir build && cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j$(nproc) && make install