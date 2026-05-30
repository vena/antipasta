"""
AntiPasta Bootstrap
-------------------
Entry point for the application. Initializes the hardware discovery,
semantic adapters, and starts the core engine thread.
"""

import logging
from threading import Thread
from dataclasses import asdict
from flask import Flask
from waitress import serve

import config
from core.state import APP_STATE
from core.hardware import get_logic_hardware_report
from core.coordinator import AntiPastaCoordinator
from core.filter import DetectionFilter
from core.heuristics import HeuristicService
from core.session import SessionManager
from mqtt.client import AntiPastaMQTT
from api.routes import api_bp

# Adapters are neatly grouped in the handlers package
from handlers.chamber_image_handler import ChamberImageHandler
from handlers.rtsp_handler import RTSPHandler

# Global Logging Setup
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AntiPasta.Main")

def create_app():
    """Factory to initialize the Flask application and semantic instances."""
    
    # Hardware Discovery (Phase 1)
    # We perform the logic-side hardware probe before initializing handlers.
    # This allows the handlers to eventually use the 'effective_mode' for FFmpeg.
    logger.debug(f"Performing hardware discovery (Mode: {config.HW_ACCEL_MODE})...")
    logic_hw_report = get_logic_hardware_report(config.HW_ACCEL_MODE)
    APP_STATE.update_hardware("logic", **asdict(logic_hw_report))
    logger.info(f"Logic hardware initialized: {logic_hw_report.note}")

    app = Flask(__name__)
    
    # Initialize Adapters
    # The MQTT client handles HA discovery and printer state subscriptions.
    mqtt = AntiPastaMQTT(
        config.MQTT_HOST, config.MQTT_USER, config.MQTT_PASSWORD,
        config.PRINTER_STATUS_TOPIC, f"antipasta_logic_{config.PRINTER_SERIAL}"
    )

    # Choose camera driver based on the environment configuration.
    # We inject the persistent stream toggle directly into the handlers, allowing 
    # them to immediately spin up their background daemon threads if enabled.
    if config.STREAM_TYPE == "chamber_image":
        camera = ChamberImageHandler(
            config.PRINTER_IP, 
            config.PRINTER_ACCESS_CODE, 
            persistent=config.STREAM_PERSISTENT
        )
    else:
        camera = RTSPHandler(
            config.RTSP_URL, 
            persistent=config.STREAM_PERSISTENT
        )

    # Initialize Core Services (Composition Root)
    session_mgr = SessionManager(APP_STATE)
    detection_filter = DetectionFilter(config.EXCLUSION_ZONES)
    
    # The heuristic service requires the newly defined dual-track thresholds
    # to differentiate between critical failures and cosmetic artifacts.
    heuristic_svc = HeuristicService(
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        warning_streak=config.WARNING_THRESHOLD,
        pause_streak=config.PAUSE_THRESHOLD,
        concern_streak=config.CONCERN_THRESHOLD,
        critical_classes=config.CRITICAL_FAILURES
    )

    # The Coordinator orchestrates the interaction between camera, AI, and MQTT
    # using the injected service implementations.
    coord = AntiPastaCoordinator(mqtt, camera, session_mgr, detection_filter, heuristic_svc)

    # Store instances in app config for Blueprint access.
    app.config.update({
        'MQTT_CLIENT': mqtt,
        'CAMERA': camera,
        'COORDINATOR': coord
    })

    # Register the decoupled API route package.
    app.register_blueprint(api_bp)
    
    return app, coord, mqtt

if __name__ == "__main__":
    app_instance, coordinator, mqtt_client = create_app()
    
    # Start I/O connections (MQTT retry loop handles startup delays).
    mqtt_client.connect(config.MQTT_HOST)
    
    # Start the Background Detection Engine.
    engine_thread = Thread(target=coordinator.run, daemon=True)
    engine_thread.start()
    
    # Silence third-party logger noise to respect the user's LOG_LEVEL.
    logging.getLogger('waitress').setLevel(config.LOG_LEVEL)
    
    logger.info(f"AntiPasta logic controller active on internal port {config.INTERNAL_PORT}")
    
    # Serve using Waitress for production-grade stability and concurrency.
    serve(app_instance, host='0.0.0.0', port=config.INTERNAL_PORT, threads=4)