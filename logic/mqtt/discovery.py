"""
MQTT Discovery Generator
------------------------
Generates Home Assistant MQTT Discovery payloads (2026.5+ compatible).
Provisions the distinct semantic tracks: Warning (Critical), Pause (Critical), and Concern (Artifact).
"""

import config
from typing import Dict, List, Any

def get_discovery_payloads(availability_topic: str) -> List[Dict[str, Any]]:
    """
    Returns a list of (topic, payload) tuples for HA discovery.
    Pulls static identifiers and topic strings directly from config.py.
    """
    prefix = config.MQTT_DISCOVERY_PREFIX
    dev_id = config.MQTT_DEVICE_ID
    
    device_info = {
        "identifiers": [dev_id],
        "name": f"AntiPasta {config.PRINTER_SERIAL}",
        "model": "AntiPasta Controller",
        "manufacturer": "AntiPasta",
        "sw_version": config.VERSION
    }

    # Modern HA Availability Schema (List-based)
    availability = [{
        "topic": availability_topic,
        "payload_available": "online",
        "payload_not_available": "offline"
    }]

    configs = []

    # Connectivity (Binary Sensor)
    configs.append((
        f"{prefix}/binary_sensor/{dev_id}_status/config",
        {
            "name": "Connectivity",
            "unique_id": f"{dev_id}_status",
            "state_topic": availability_topic,
            "device_class": "connectivity",
            "payload_on": "online",
            "payload_off": "offline",
            "device": device_info,
            "entity_category": "diagnostic"
        }
    ))

    # Failure Confidence (Sensor)
    configs.append((
        f"{prefix}/sensor/{dev_id}_confidence/config",
        {
            "name": "Failure Confidence",
            "unique_id": f"{dev_id}_confidence",
            "state_topic": config.TOPIC_CONFIDENCE,
            "value_template": "{{ value_json.confidence | default(0.0) }}",
            "unit_of_measurement": "%",
            "icon": "mdi:gauge",
            "availability": availability,
            "json_attributes_topic": config.TOPIC_CONFIDENCE,
            "device": device_info
        }
    ))

    # Failure Warning - Pre-Pause Critical (Binary Sensor)
    configs.append((
        f"{prefix}/binary_sensor/{dev_id}_warning/config",
        {
            "name": "Failure Warning",
            "unique_id": f"{dev_id}_warning",
            "state_topic": config.TOPIC_WARNING,
            "value_template": "{{ value_json.state | default('OFF') }}",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device_class": "safety",
            "availability": availability,
            "json_attributes_topic": config.TOPIC_WARNING,
            "device": device_info
        }
    ))

    # Failure Pause - Confirmed Critical (Binary Sensor)
    configs.append((
        f"{prefix}/binary_sensor/{dev_id}_pause/config",
        {
            "name": "Failure Pause",
            "unique_id": f"{dev_id}_pause",
            "state_topic": config.TOPIC_PAUSE,
            "value_template": "{{ value_json.state | default('OFF') }}",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device_class": "safety",
            "availability": availability,
            "json_attributes_topic": config.TOPIC_PAUSE,
            "device": device_info
        }
    ))
    
    # Artifact Concern - Non-Critical (Binary Sensor)
    configs.append((
        f"{prefix}/binary_sensor/{dev_id}_concern/config",
        {
            "name": "Artifact Concern",
            "unique_id": f"{dev_id}_concern",
            "state_topic": config.TOPIC_CONCERN,
            "value_template": "{{ value_json.state | default('OFF') }}",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device_class": "problem",
            "availability": availability,
            "json_attributes_topic": config.TOPIC_CONCERN,
            "device": device_info
        }
    ))

    # Inference Time (Diagnostic Sensor)
    configs.append((
        f"{prefix}/sensor/{dev_id}_inference_time/config",
        {
            "name": "Inference Time",
            "unique_id": f"{dev_id}_inference_time",
            "state_topic": config.TOPIC_CONFIDENCE,
            "value_template": "{{ value_json.inference_ms | default(0) }}",
            "unit_of_measurement": "ms",
            "icon": "mdi:timer-outline",
            "entity_category": "diagnostic",
            "availability": availability,
            "device": device_info
        }
    ))

    return configs