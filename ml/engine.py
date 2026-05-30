"""
AntiPasta ML Engine Orchestrator
-------------------------------
Manages hardware initialization, model synchronization, and adapter 
selection in a non-blocking background thread.

This orchestrator uses Late-Importing to ensure that environment variables 
are set before model-specific libraries (like Ultralytics) are loaded, 
and to ensure that unused AI libraries do not consume system RAM.
"""

import os
import threading
import logging
import onnxruntime as ort
from server import start_ml_server

logger = logging.getLogger("AntiPasta.MLEngine")

class AntiPastaEngine:
    def __init__(self):
        self.ready = False
        self.status = "Initializing Hardware..."
        self.active_provider = "Unknown"
        self.active_device = "Unknown"
        
        # Environment Steering
        # Establish model-specific paths before any adapter logic is loaded.
        self.engine_type = os.environ.get('ML_ENGINE', 'yolov11').lower()
        self.base_dir = f"/model_cache/{self.engine_type}"
        
        os.environ['YOLO_CONFIG_DIR'] = os.path.join(self.base_dir, ".config")
        os.environ['OV_CACHE_DIR'] = os.path.join(self.base_dir, ".ov_cache")
        
        # Ensure persistent directories exist immediately.
        os.makedirs(os.path.join(os.environ['YOLO_CONFIG_DIR'], 'Ultralytics'), exist_ok=True)
        os.makedirs(os.environ['OV_CACHE_DIR'], exist_ok=True)

        # Late-imports for dynamic adapter selection.
        if self.engine_type == "obico":
            from adapters.obico_adapter import ObicoAdapter
            self.adapter = ObicoAdapter()
        else:
            from adapters.yolo_adapter import YoloAdapter
            self.adapter = YoloAdapter()
            
    def initialize(self):
        """Perform hardware probing and model setup in the background."""
        thread = threading.Thread(target=self._run_init, daemon=True)
        thread.start()

    def _run_init(self):
        """Background initialization task."""
        try:
            # Hardware Discovery
            available = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available}")
            
            if 'OpenVINOExecutionProvider' in available:
                self.active_provider = 'OpenVINOExecutionProvider'
                # Set the global hint to prioritize GPU for all 
                # OpenVINO-based sessions (both YOLO and Obico).
                os.environ["OPENVINO_DEVICE"] = "GPU"
            else:
                self.active_provider = 'CPUExecutionProvider'
            
            # Model Setup via Adapter
            self.active_device = self.adapter.load(
                self.active_provider, 
                self._update_status
            )
            
            self.adapter.warmup()
            
            self.status = "Ready"
            self.ready = True
            logger.info(f"AI Engine Ready. Mode: {self.active_provider} on {self.active_device}")
            
        except Exception as e:
            self.status = f"Initialization Failed: {str(e)}"
            logger.error(self.status, exc_info=True)

    def _update_status(self, msg):
        self.status = msg

    def infer(self, img_bytes: bytes):
        """Delegates inference to the loaded adapter."""
        if not self.ready:
            raise RuntimeError("Engine not ready")
        return self.adapter.infer(img_bytes)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    engine = AntiPastaEngine()
    engine.initialize()
    
    start_ml_server(engine)