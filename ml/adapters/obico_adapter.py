"""
Obico Adapter
-------------
Preserves the legacy post-processing logic for the Obico ONNX model.
"""

import os
import requests
import numpy as np
import cv2
import onnxruntime as ort
from adapters.base import ModelAdapter
import base_utils

class ObicoAdapter(ModelAdapter):
    def __init__(self):
        self.session = None
        self.base_dir = "/model_cache/obico"
        self.onnx_path = os.path.join(self.base_dir, "obico.onnx")
        self.pointer_url = "https://raw.githubusercontent.com/TheSpaghettiDetective/obico-server/release/ml_api/model/model-weights.onnx.url"
        # Metadata required for tensor preprocessing
        self.input_name = None
        self.input_shape = None

    def load(self, hardware_provider: str, status_callback) -> str:
        """
        Synchronizes weights and initializes the ONNX Runtime session.
        """
        status_callback("Resolving Obico model pointer...")
        try:
            upstream_url = requests.get(self.pointer_url, timeout=10).text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to resolve Obico upstream URL: {e}")
        
        base_utils.setup_model_directory(
            self.base_dir, upstream_url, "obico.onnx", status_callback
        )

        status_callback(f"Initializing ONNX session ({hardware_provider})...")
        
        provider_options = {}
        if hardware_provider == 'OpenVINOExecutionProvider':
            # Rely on the OPENVINO_DEVICE=GPU env var set in engine.py
            # while still providing the cache directory for compiled kernels.
            cache_dir = os.environ.get('OV_CACHE_DIR')
            provider_options = {'cache_dir': cache_dir}

        providers = [(hardware_provider, provider_options)]
        if hardware_provider != 'CPUExecutionProvider':
            providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(
            self.onnx_path, 
            providers=providers
        )
        
        # PERSISTENCE REPAIR: Save input metadata to the instance for the infer() method.
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape # [batch, channels, height, width]
        
        # Report the actual provider used (if GPU failed, this returns CPUExecutionProvider)
        active = self.session.get_providers()[0]
        return f"iGPU ({active})" if active == 'OpenVINOExecutionProvider' else active

    def infer(self, img_bytes: bytes) -> list:
        """
        Performs inference and applies custom Obico post-processing.
        Matches the reference implementation.
        """
        if not self.session:
            raise RuntimeError("Obico session not initialized.")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Payload could not be decoded into an image.")
            
        orig_h, orig_w = img.shape[:2]
        img_h, img_w = self.input_shape[2], self.input_shape[3]
        
        # --- Preprocessing ---
        resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        
        outputs = self.session.run(None, {self.input_name: img_in})
        box_array, confs = outputs[0], outputs[1]

        if box_array.ndim == 4:
            box_array = box_array[:, :, 0]
            
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        CONF_THRESH, NMS_THRESH = 0.08, 0.45
        detections = []
        
        argwhere = max_conf[0] > CONF_THRESH
        l_box_array = box_array[0, argwhere, :]
        l_max_conf = max_conf[0, argwhere]
        l_max_id = max_id[0, argwhere]

        for j in range(confs.shape[2]):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            
            if len(ll_box_array) == 0: continue

            boxes_for_nms = []
            for k in range(len(ll_box_array)):
                x1, y1, x2, y2 = ll_box_array[k]
                boxes_for_nms.append([
                    int(x1 * orig_w), int(y1 * orig_h), 
                    int((x2 - x1) * orig_w), int((y2 - y1) * orig_h)
                ])

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, ll_max_conf.tolist(), CONF_THRESH, NMS_THRESH)

            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, x2, y2 = ll_box_array[idx]
                    detections.append([
                        "failure", 
                        float(ll_max_conf[idx]), 
                        [float(x1 * orig_w), float(y1 * orig_h), 
                         float(x2 * orig_w), float(y2 * orig_h)]
                    ])
                
        return detections