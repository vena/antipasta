"""
MQTT Client
-----------
Semantic wrapper for Paho MQTT. Hides topic strings and JSON 
serialization from the core engine. Now enforces strict type-checking 
using the AlertPayload dataclass.
"""

import paho.mqtt.client as mqtt
import logging
import json
import time
import socket
import sys
from dataclasses import asdict
from typing import Dict, Any, Optional

import config
from core.state import APP_STATE
from mqtt.discovery import get_discovery_payloads
from core.models import AlertPayload

logger = logging.getLogger("AntiPasta.MQTT.Client")

class AntiPastaMQTT:
    def __init__(self, host, user, password, printer_status_topic, client_id):
        # Version 2 callback API is required for modern Paho versions (2.0+).
        self.client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        
        if user:
            self.client.username_pw_set(user, password)
        
        self.printer_status_topic = printer_status_topic
        self.availability_topic = config.CONTROLLER_STATUS_TOPIC
        self.printer_state = "unknown"
        
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # LWT: Notifies HA immediately if the AntiPasta container crashes
        self.client.will_set(self.availability_topic, payload="offline", qos=1, retain=True)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """
        Triggered when connection is established. Re-announces discovery 
        to ensure HA entities are registered if the broker was restarted.
        """
        logger.info(f"Connected to MQTT Broker (Result: {rc})")
        client.subscribe(self.printer_status_topic)
        
        self.announce_discovery()
        self.publish_availability(True)

    def _on_message(self, client, userdata, msg):
        self.printer_state = msg.payload.decode().lower()
        logger.debug(f"Printer status update: {self.printer_state}")

    def announce_discovery(self):
        """Publishes all HA discovery payloads defined in discovery.py."""
        payloads = get_discovery_payloads(self.availability_topic)
        for topic, payload in payloads:
            self.publish_raw(topic, json.dumps(payload))
        logger.info("Home Assistant Discovery announced.")

    def publish_availability(self, online: bool):
        payload = "online" if online else "offline"
        self.publish_raw(self.availability_topic, payload)

    def publish_warning(self, payload: AlertPayload):
        """Publishes a structured alert payload for the Warning (Critical) sensor."""
        self.publish_raw(config.TOPIC_WARNING, json.dumps(asdict(payload)))

    def publish_pause(self, payload: AlertPayload):
        """Publishes a structured alert payload for the Pause (Critical) sensor."""
        self.publish_raw(config.TOPIC_PAUSE, json.dumps(asdict(payload)))
        
    def publish_concern(self, payload: AlertPayload):
        """Publishes a structured alert payload for the Concern (Artifact) sensor."""
        self.publish_raw(config.TOPIC_CONCERN, json.dumps(asdict(payload)))

    def publish_confidence(self, confidence: float, inference_ms: int, metadata: Dict[str, Any]):
        """Publishes the telemetry frame (Confidence and Inference speed)."""
        topic = config.TOPIC_CONFIDENCE
        full_payload = {
            "confidence": confidence,
            "inference_ms": inference_ms,
            **metadata
        }
        self.publish_raw(topic, json.dumps(full_payload))

    def publish_raw(self, topic: str, payload: str, retain: bool = True):
        self.client.publish(topic, payload, qos=1, retain=retain)

    def connect(self, host: str, timeout_seconds: int = 60):
        """
        Attempts to connect to the broker with a strict deadline.
        Failing fast ensures Docker can restart the container if the broker is missing,
        rather than allowing the application to idle without core functionality.
        """
        logger.info(f"Connecting to MQTT Broker at {host} (Timeout: {timeout_seconds}s)...")
        start_time = time.monotonic()
        
        while True:
            try:
                self.client.connect(host, 1883, 60)
                self.client.loop_start()
                break
            except OSError as e:
                # If the deadline has passed, log the failure and cleanly terminate.
                if time.monotonic() - start_time > timeout_seconds:
                    logger.critical(f"Failed to connect to MQTT broker after {timeout_seconds}s: {e}")
                    sys.exit(1)
                    
                # We sleep silently here to prevent spamming the logs every few 
                # seconds while waiting for the broker to finish booting.
                time.sleep(2)