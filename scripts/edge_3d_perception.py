import cv2
import numpy as np
import torch
import sys
import os
from ultralytics import YOLO

# Import Orbbec bindings and the working helper utility
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, FrameSet
from utils import frame_to_bgr_image

def main():
    # 1. Initialize CUDA GPU Inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚡ Running inference on: {device.upper()}")
    
    # Load the lightweight YOLOv8 nano model
    model = YOLO("yolov8n.pt").to(device)
    
    # 2. Configure the Pipeline using Explicit Profiles
    pipeline = Pipeline()
    config = Config()
    
    try:
        # Resolve and enable the specific Color Sensor Profile
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profile_list is not None:
            color_profile = color_profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            print(f"✔ Color profile resolved: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps")
        
        # Resolve and enable the specific Depth Sensor Profile
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is not None:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            print(f"✔ Depth profile resolved: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps")
            
    except Exception as e:
        print(f"❌ Failed to resolve camera stream profiles: {e}")
        return

    # Force Software Alignment Mode for legacy Astra OpenNI cameras
    config.set_align_mode(OBAlignMode.SW_MODE)
    print("✔ Set alignment mode to Software (SW_MODE) for Astra Pro Plus consistency.")

    # Try enabling frame sync safely (it will print an expected warning but shouldn't crash us)
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(f"ℹ Note: Hardware frame sync skipped (expected for this device): {e}")

    # Start the active pipeline
    try:
        pipeline.start(config)
        print("🎥 Camera streams started successfully! Press 'q' on the visual window to exit.")
    except Exception as e:
        print(f"❌ Pipeline failed to start: {e}")
        return

    try:
        while True:
            # Grab synchronized software-aligned frame structures
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
                
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame is None or depth_frame is None:
                continue
                
            # 3. Process Frames into OpenCV Compatible Arrays
            # Use the SDK's built-in conversion utility for the color frame
            color_img = frame_to_bgr_image(color_frame)
            if color_img is None:
                continue
            
            # Unpack the 16-bit raw depth data channel
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_img = np.resize(depth_data, (depth_frame.get_height(), depth_frame.get_width()))
            
            # 4. Run Accelerated YOLOv8 Inference
            results = model(color_img, device=device, verbose=False)
            
            # 5. Extract Detections and Compute Depth Map Overlay
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Bounding box limits
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = model.names[cls_id]
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Target center coordinate of the bounding box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Prevent array clamping/out-of-bounds evaluation
                    h, w = depth_img.shape
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))
                    
                    # Extract distance data (Astra native format is millimeters)
                    depth_mm = depth_img[cy, cx]
                    depth_m = depth_mm / 1000.0
                    
                    # Render graphical bounding boxes and tracking dots
                    cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_img, (cx, cy), 4, (0, 0, 255), -1)
                    
                    # Create context label text
                    if depth_mm > 0:
                        text = f"{label} {confidence:.2f} | {depth_m:.2f}m"
                    else:
                        text = f"{label} {confidence:.2f} | Out of Range"
                        
                    cv2.putText(color_img, text, (x1, max(y1 - 10, 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 6. Stream Live Image Matrix
            cv2.imshow("Edge AI 3D Perception Dashboard", color_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down stream cleanly...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
