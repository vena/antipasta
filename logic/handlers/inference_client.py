"""
AI Inference Client
-------------------
Handles communication with the Machine Learning API via HTTP POST.
Acts as a lifecycle gatekeeper, checking for engine readiness 
before attempting heavy data transfers.
"""

import requests
import logging
import config
from core.models import HardwareStats

logger = logging.getLogger("AntiPasta.InferenceClient")

# Global session to leverage connection pooling
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {config.ML_API_TOKEN}",
    "User-Agent": f"AntiPasta/{config.VERSION}"
})

def get_ml_hardware_state() -> HardwareStats:
    """
    Queries the ML container's /info/ endpoint to retrieve its hardware status.
    Uses a short timeout to ensure the Logic API remains responsive if the backend is down.
    """
    try:
        resp = session.get(config.ML_INFO_ENDPOINT, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return HardwareStats(
                active=data.get("ready", False),
                mode=data.get("hardware", {}).get("provider", "unknown"),
                device=data.get("hardware", {}).get("device", "unknown"),
                note=data.get("status", "Ready")
            )
    except Exception:
        pass
    
    return HardwareStats(
        active=False,
        mode="disconnected",
        device="unknown",
        note="ML Backend Unreachable or Starting..."
    )

def perform_inference(image_bytes: bytes):
    """
    Sends image bytes to the ML container for analysis.
    
    Returns: (detections, inference_time, status_message)
    - detections: List of [class, confidence, bbox]
    - inference_time: Seconds taken for the request
    - status_message: Human-readable string for errors or initialization states
    """
    try:
        # READINESS CHECK
        # Query the /info/ endpoint first to prevent timing out
        # while the engine is still cold-starting.
        try:
            info_resp = session.get(config.ML_INFO_ENDPOINT, timeout=2)
            if info_resp.status_code == 200:
                info_data = info_resp.json()
                if not info_data.get("ready", False):
                    return None, 0.0, f"AI Engine Initializing: {info_data.get('status', 'Loading...')}"
            elif info_resp.status_code == 503:
                return None, 0.0, "AI Engine Busy or Loading"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # This explicitly catches the "Max retries exceeded" / "Connection refused"
            # errors that occur during the very early seconds of container boot.
            return None, 0.0, "Waiting for AI Container to start..."

        # PERFORM INFERENCE
        files = {'image': ('frame.jpg', image_bytes, 'image/jpeg')}
        
        # 15s timeout accounts for potential iGPU spin-up or large models.
        r = session.post(config.ML_API_ENDPOINT, files=files, timeout=15)
        inference_time = r.elapsed.total_seconds()
        
        if r.status_code == 200:
            detections = r.json().get("detections", [])
            return detections, inference_time, None
        elif r.status_code == 503:
            msg = r.json().get("status", "Engine Initialization in progress")
            return None, 0.0, f"AI Engine Busy: {msg}"
        else:
            return None, 0.0, f"ML API returned {r.status_code}: {r.text}"
            
    except Exception as e:
        # Map raw network exceptions to a clean status message
        err_str = str(e)
        if "Max retries exceeded" in err_str or "Failed to establish a new connection" in err_str:
            return None, 0.0, "AI Backend is currently unreachable (Starting up...)"
            
        logger.debug(f"Inference communication fault: {e}")
        return None, 0.0, f"AI Communication Error: {err_str}"