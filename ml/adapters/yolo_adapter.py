"""
YOLOv11 Adapter
---------------
Encapsulates Ultralytics YOLOv11 logic with hardware-adaptive exports.

Optimization:
Instead of forcing ONNX Runtime for all platforms, this adapter utilizes 
native OpenVINO exports for Intel hardware, providing superior GPU 
engagement and bypassing library dependency collisions.
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO, settings
from adapters.base import ModelAdapter
import base_utils

class YoloAdapter(ModelAdapter):
    def __init__(self):
        self.model = None
        self.base_dir = "/model_cache/yolov11"
        self.pt_path = os.path.join(self.base_dir, "best.pt")
        
        # We track two potential optimized paths
        self.onnx_path = os.path.join(self.base_dir, "best.onnx")
        self.ov_path = os.path.join(self.base_dir, "best_openvino_model")
        
        self.url = "https://huggingface.co/ApatheticWithoutTheA/3D-Print-Failure-Detector/resolve/main/yolov11-3d-print-failure-detection"

    def load(self, hardware_provider: str, status_callback) -> str:
        """
        Synchronizes weights and initializes the most efficient runtime 
        available for the detected hardware.
        """
        settings.update({'sync': False})

        # Sync PyTorch model weights
        changed = base_utils.setup_model_directory(
            self.base_dir, self.url, "best.pt", status_callback
        )

        # Determine and Prepare Optimized Format
        # If on Intel, we use the 'openvino' format. Otherwise, we use 'onnx'.
        is_intel = (hardware_provider == 'OpenVINOExecutionProvider')
        target_format = 'openvino' if is_intel else 'onnx'
        target_path = self.ov_path if is_intel else self.onnx_path

        if changed or not os.path.exists(target_path):
            status_callback(f"Exporting YOLO model to {target_format} format...")
            tmp_model = YOLO(self.pt_path)
            # dynamic=True allows the model to handle varying input frame resolutions.
            tmp_model.export(format=target_format, dynamic=True)

        status_callback(f"Loading YOLO session with {hardware_provider}...")
        
        # Initialize Inference Session
        # Loading the native OpenVINO format (folder containing xml/bin) 
        # allows Ultralytics to engage the GPU directly via the OpenVINO library.
        self.model = YOLO(target_path, task='detect')
        
        if is_intel:
            return f"iGPU ({hardware_provider} Native)"
        
        return f"CPU ({hardware_provider})"

    def infer(self, img_bytes: bytes) -> list:
        """
        Performs inference using the high-level Ultralytics API.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Payload could not be decoded into an image.")

        results = self.model(img, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                detections.append([
                    results[0].names[int(box.cls[0])],
                    float(box.conf[0]),
                    # xyxy provides absolute pixel coordinates: [x1, y1, x2, y2]
                    box.xyxy[0].tolist() 
                ])
                
        return detections