"""
API Routes
----------
Flask Blueprint containing all HTTP handlers for AntiPasta.
Maintains the public schema for system diagnostics and frame serving.
"""

import os
import io
import time
import logging
from dataclasses import asdict
from flask import Blueprint, send_file, jsonify, request, current_app

import config
from core.state import APP_STATE
from core.models import Detection
from handlers.inference_client import perform_inference, get_ml_hardware_state
from handlers.remote_image_handler import RemoteImageHandler
from handlers.annotation_handler import draw_detections
from handlers.local_storage_handler import save_latest_frame
from api.rate_limit import rate_limit

logger = logging.getLogger("AntiPasta.API")
api_bp = Blueprint('api', __name__)

@api_bp.route('/stats/')
def get_stats():
    """
    Returns a unified snapshot of system state and configuration.
    
    Merges state from across the application and containers and provides a safe, public view of the system.
    """
    # Real-time update of ML status via the inference client
    ml_hw = get_ml_hardware_state()
    APP_STATE.update_hardware("ml", **asdict(ml_hw))

    output = APP_STATE.to_dict()
    mqtt = current_app.config['MQTT_CLIENT']
    
    output["system"] = {
        "printer_state": mqtt.printer_state,
        "public_port": config.PUBLIC_PORT,
        "external_url": config.CONTROLLER_EXTERNAL_URL,
    }
    
    output["config"] = {
        "ml_engine": config.ML_ENGINE,
        "device_id": config.MQTT_DEVICE_ID,
        "discovery_prefix": config.MQTT_DISCOVERY_PREFIX,
        "public_port": config.PUBLIC_PORT,
        "controller_external_url": config.CONTROLLER_EXTERNAL_URL,
        "printer": {
            "printer_ip": config.PRINTER_IP,
            "printer_serial": config.PRINTER_SERIAL,
            "stream_type": config.STREAM_TYPE,
            "stream_persistent": config.STREAM_PERSISTENT,
            "strict_tls": config.STRICT_TLS,
            "custom_ca_cert_path": config.CUSTOM_CA_CERT_PATH,
            "frame_interval": config.FRAME_INTERVAL,
            "exclusion_zones": config.EXCLUSION_ZONES,
        },
        "thresholds": {
            "confidence": config.CONFIDENCE_THRESHOLD,
            "warning": config.WARNING_THRESHOLD,
            "pause": config.PAUSE_THRESHOLD,
            "concern": config.CONCERN_THRESHOLD,
            "critical_classes": config.CRITICAL_FAILURES
        },
        "retention": {
            "days": config.FAILURE_RETENTION_DAYS,
            "max_count": config.FAILURE_RETENTION_COUNT
        },
        "paths": {
            "failures_dir": config.FAILURES_DIR,
            "model_cache_dir": config.MODEL_CACHE_DIR
        },
        "mqtt": {
            "printer_status_topic": config.PRINTER_STATUS_TOPIC,
            "availability_topic": config.CONTROLLER_STATUS_TOPIC,
            "confidence_topic": config.TOPIC_CONFIDENCE,
            "warning_topic": config.TOPIC_WARNING,
            "pause_topic": config.TOPIC_PAUSE,
            "concern_topic": config.TOPIC_CONCERN
        },
        "unique_ids": {
            "confidence": f"{config.MQTT_DEVICE_ID}_confidence",
            "warning": f"{config.MQTT_DEVICE_ID}_warning",
            "pause": f"{config.MQTT_DEVICE_ID}_pause",
            "concern": f"{config.MQTT_DEVICE_ID}_concern",
            "status": f"{config.MQTT_DEVICE_ID}_status"
        }
    }
    
    stats = output["session"]
    if stats["frames_analyzed"] > 0:
        stats["failure_rate"] = round(
            stats["fail_frames_detected"] / stats["frames_analyzed"], 4
        )
        
    return jsonify(output)

@api_bp.route('/frame.jpg')
@rate_limit(rate=3, per_seconds=1)
def serve_frame():
    """Captures and returns a raw frame from the current camera adapter."""
    coord = current_app.config['COORDINATOR']
    camera = current_app.config['CAMERA']

    # --- Persistent Stream Mode ---
    if config.STREAM_PERSISTENT:
        # Bypass the disk cache entirely and serve the blazing-fast RAM buffer.
        # We explicitly omit save_latest_frame() here to prevent aggressive dashboard 
        # polling from causing heavy disk thrashing. The background Coordinator loop 
        # will safely handle periodic disk archiving at the defined FRAME_INTERVAL.
        frame = camera.get_frame()
        if frame:
            return send_file(io.BytesIO(frame), mimetype='image/jpeg')
        return "Camera Busy or Offline", 503

    # --- Polling Stream Mode ---
    # DEFENSIVE CACHING: Prevent external API polling from monopolizing the camera.
    # If the background loop recently saved a frame (within the interval window),
    # we serve it directly from the disk. This protects the hardware connection pool 
    # and ensures AntiPasta doesn't inadvertently DDoS the printer when a user leaves 
    # a Home Assistant dashboard open.
    if os.path.exists(config.LAST_FRAME_PATH):
        age = time.time() - os.path.getmtime(config.LAST_FRAME_PATH)
        if age < config.FRAME_INTERVAL:
            return send_file(config.LAST_FRAME_PATH, mimetype='image/jpeg')

    # Only fallback to a live, blocking capture if the background loop is suspended 
    # (e.g., the printer is idle) and the cached image is stale.
    with coord.camera_lock:
        frame = camera.get_frame()
        if frame:
            save_latest_frame(frame)
            return send_file(io.BytesIO(frame), mimetype='image/jpeg')
            
    if os.path.exists(config.LAST_FRAME_PATH):
        return send_file(config.LAST_FRAME_PATH, mimetype='image/jpeg')
        
    return "Camera Busy or Offline", 503

@api_bp.route('/failure_frame.jpg')
@rate_limit(rate=3, per_seconds=1)
def serve_failure_frame():
    """Serves annotated failure frames strictly by their event UUID."""
    req_id = request.args.get('event')
    
    # Enforce stateless architecture. Falling back to a session property
    # if the client dropped the `event` parameter could result in serving the wrong 
    # image across different print jobs or alert contexts.
    if not req_id:
        return jsonify({"error": "Missing required parameter: 'event'"}), 400
        
    safe_id = "".join(x for x in req_id if x.isalnum() or x == "-")
    target = os.path.join(config.FAILURES_DIR, f"{safe_id}.jpg")

    if os.path.exists(target):
        return send_file(target, mimetype='image/jpeg')
        
    return "Failure record not found", 404

@api_bp.route('/test/', defaults={'url': None, 'output_format': 'img'})
@api_bp.route('/test/json/', defaults={'url': None, 'output_format': 'json'})
@api_bp.route('/test/<path:url>', defaults={'output_format': 'img'})
@api_bp.route('/test/json/<path:url>', defaults={'output_format': 'json'})
@rate_limit(rate=2, per_seconds=1)
def test_router(url, output_format):
    """
    Unified AI testing endpoint. 
    Provides specific JSON feedback if the AI engine is still initializing.
    Secured behind the ALLOW_TEST_API environment flag to prevent SSRF and exposure.
    """
    # Enforce the user-level opt-in security flag
    if not config.ALLOW_TEST_API:
        return jsonify({
            "error": "Test API is disabled for security. See documentation to learn why, and how to enable this feature."
        }), 403

    coord = current_app.config['COORDINATOR']

    if url:
        target_url = url
        if request.query_string:
            target_url += "?" + request.query_string.decode('utf-8')
            
        remote_camera = RemoteImageHandler(target_url)
        try:
            img_bytes = remote_camera.get_frame()
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
            
        if not img_bytes:
            return jsonify({"error": "Fetch failed or payload empty."}), 502
    else:
        camera = current_app.config['CAMERA']
        with coord.camera_lock:
            img_bytes = camera.get_frame()
            
        if not img_bytes:
            return jsonify({"error": "Capture failed"}), 503

    # Check for readiness and perform inference
    raw_results, inf_time, status_msg = perform_inference(img_bytes)
    
    if status_msg:
        return jsonify({
            "error": "AI Backend Not Ready",
            "status": status_msg
        }), 503

    detections = [Detection(d[0], d[1], d[2]) for d in raw_results]
    filtered = coord.filter.apply(detections, img_bytes)
    
    if output_format == 'json':
        return jsonify({
            "detections": [d.to_list() for d in filtered], 
            "inference_ms": int(inf_time * 1000)
        })

    img = draw_detections(img_bytes, filtered, inf_time, config.EXCLUSION_ZONES)
    if not img: return jsonify({"error": "Image processing failed"}), 500
    return send_file(img, mimetype='image/jpeg')

@api_bp.route('/hc/')
def health():
    """Simple health check endpoint."""
    return "OK", 200