"""
Configuration Module
--------------------
Centralizes environment variable loading and URL construction.
Determines hardware acceleration preferences and network resolution.
Compiles regex patterns at startup to ensure safe state evaluations.
"""

import os
import json
import logging
import re
import ssl
import sys
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("AntiPasta.Config")

# --- Release Version ---
VERSION = "1.0.0"

def get_env_expanded(key, default=None):
    """
    Retrieves env var and performs variable substitution for ${PRINTER_SERIAL}.
    This allows users to define generic MQTT topics that automatically specialize 
    to their hardware instance.
    """
    val = os.getenv(key, default)
    serial = os.getenv("PRINTER_SERIAL", "")
    if val and "${PRINTER_SERIAL}" in val:
        val = val.replace("${PRINTER_SERIAL}", serial)
    return val

# --- Networking ---
INTERNAL_PORT = 8080

# Logic to handle empty or missing CONTROLLER_PORT env var safely.
public_port_env = os.getenv("CONTROLLER_PORT", "8080").strip()
PUBLIC_PORT = int(public_port_env) if public_port_env else 8080

# The base URL used by Home Assistant/Mobile App to reach the controller externally.
CONTROLLER_EXTERNAL_URL = os.getenv("CONTROLLER_EXTERNAL_URL", f"http://localhost:{PUBLIC_PORT}").rstrip('/')

# --- Hardware Acceleration ---
# HW_ACCEL_MODE options: auto (default), vaapi, v4l2, cpu
# TODO: add support for separate modes and devices for each container
HW_ACCEL_MODE = os.getenv("HW_ACCEL_MODE", "auto").lower()
HW_ACCEL_DEVICE = os.getenv("HW_ACCEL_DEVICE", "/dev/dri/renderD128")

# --- Security & TLS ---
STRICT_TLS = os.getenv("STRICT_TLS", "False").lower() == "true"
CUSTOM_CA_CERT_PATH = os.getenv("CUSTOM_CA_CERT_PATH")

# Controls access to the /test/ API routes. This prevents unauthenticated users 
# from using the application as a proxy to scan the local network (SSRF protection).
ALLOW_TEST_API = os.getenv("ALLOW_TEST_API", "False").lower() == "true"
if ALLOW_TEST_API:
    logger.warning("The /test/ API endpoint is enabled. It is intended for local development and testing only. Do not expose it to the internet!")

# Strict Validation for the Custom CA Certificate.
# We "fail closed" here. If the user provides a custom CA cert but it is invalid,
# we immediately crash the container. This prevents the application from silently
# falling back to an unverified state or spamming connection errors later.
if CUSTOM_CA_CERT_PATH:
    if not os.path.isfile(CUSTOM_CA_CERT_PATH):
        logger.critical(f"CUSTOM_CA_CERT_PATH '{CUSTOM_CA_CERT_PATH}' does not exist. Halting application.")
        sys.exit(1)
    
    try:
        # Test the certificate by attempting to load it into an ephemeral SSL context.
        # This forces the underlying OpenSSL C-library to parse the file and validate 
        # that it is a structurally sound x509 certificate.
        test_context = ssl.create_default_context()
        test_context.load_verify_locations(cafile=CUSTOM_CA_CERT_PATH)
        logger.info(f"Custom CA certificate loaded successfully from {CUSTOM_CA_CERT_PATH}")
    except ssl.SSLError as e:
        logger.critical(f"Cryptographic validation failed for CUSTOM_CA_CERT_PATH ({CUSTOM_CA_CERT_PATH}): {e}. Halting application.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed to read CUSTOM_CA_CERT_PATH ({CUSTOM_CA_CERT_PATH}): {e}. Halting application.")
        sys.exit(1)

# --- Printer Credentials ---
PRINTER_IP = os.getenv("PRINTER_IP")
PRINTER_ACCESS_CODE = os.getenv("PRINTER_ACCESS_CODE")
PRINTER_SERIAL = os.getenv("PRINTER_SERIAL", "unknown")

# --- Stream Configuration ---
STREAM_TYPE = os.getenv("STREAM_TYPE", "chamber_image")
RTSP_URL = get_env_expanded("RTSP_URL")
STREAM_PERSISTENT = os.getenv("STREAM_PERSISTENT", "False").lower() == "true"

# A strict 10MB limit prevents Memory exhaustion (OOM) vulnerabilities in the 
# event a camera stream corrupts or network packet loss destroys a framing boundary.
MAX_FRAME_SIZE_BYTES = 10485760 

# --- MQTT Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_DISCOVERY_PREFIX = os.getenv("MQTT_DISCOVERY_PREFIX", "homeassistant")

# Unique identifier for this specific printer instance to prevent collisions in HA
MQTT_DEVICE_ID = f"antipasta_{PRINTER_SERIAL}"

# --- MQTT Topics (Normalized to lowercase) ---
PRINTER_STATUS_TOPIC = (get_env_expanded("MQTT_PRINTER_STATUS_TOPIC") or "").lower()
CONTROLLER_STATUS_TOPIC = f"antipasta/{PRINTER_SERIAL}/status".lower()
TOPIC_CONFIDENCE = f"antipasta/{PRINTER_SERIAL}/confidence".lower()
TOPIC_WARNING = f"antipasta/{PRINTER_SERIAL}/warning".lower()
TOPIC_PAUSE = f"antipasta/{PRINTER_SERIAL}/pause".lower()
TOPIC_CONCERN = f"antipasta/{PRINTER_SERIAL}/concern".lower()

# --- Printer Status Matching ---
# Treating inputs as regular expressions enables flexible matching across 
# different printer firmware states. re.IGNORECASE ensures case-insensitivity.
raw_printing = os.getenv("MQTT_PRINTER_STATE_PRINTING", "printing")
raw_paused = os.getenv("MQTT_PRINTER_STATE_PAUSED", "pause.*")

try:
    STATUS_PRINTING_REGEX = re.compile(raw_printing, re.IGNORECASE)
except re.error as e:
    logger.error(f"Invalid regex for MQTT_PRINTER_STATE_PRINTING ('{raw_printing}'): {e}. Falling back to 'printing'.")
    STATUS_PRINTING_REGEX = re.compile("printing", re.IGNORECASE)

try:
    STATUS_PAUSED_REGEX = re.compile(raw_paused, re.IGNORECASE)
except re.error as e:
    logger.error(f"Invalid regex for MQTT_PRINTER_STATE_PAUSED ('{raw_paused}'): {e}. Falling back to 'pause.*'.")
    STATUS_PAUSED_REGEX = re.compile("pause.*", re.IGNORECASE)

# --- AI & Inference Settings ---
# URL NORMALIZATION: 
# We take the provided endpoint, strip trailing paths, and rebuild explicitly.
# This ensures that /p/ and /info/ routes always resolve correctly regardless
# of how the user formatted the MODEL_ENDPOINT variable.
raw_endpoint = os.getenv("MODEL_ENDPOINT", "http://antipasta-ml:3333/p/")
ml_base = raw_endpoint.split('/p/')[0].rstrip('/')

ML_API_ENDPOINT = f"{ml_base}/p/"
ML_INFO_ENDPOINT = f"{ml_base}/info/"

ML_API_TOKEN = os.getenv("ML_API_TOKEN", "internal_secret_token")
ML_ENGINE = os.getenv("ML_ENGINE", "yolov11")

# --- Exclusion Zone Parsing & Normalization ---
try:
    raw_zones = json.loads(os.getenv("EXCLUSION_ZONES", "[]"))
    
    if isinstance(raw_zones, list) and len(raw_zones) == 4 and all(isinstance(i, (int, float)) for i in raw_zones):
        EXCLUSION_ZONES = [raw_zones]
    elif isinstance(raw_zones, list):
        EXCLUSION_ZONES = raw_zones
    else:
        EXCLUSION_ZONES = []
except Exception as e:
    logger.error(f"Failed to parse EXCLUSION_ZONES: {e}")
    EXCLUSION_ZONES = []

# --- Thresholds ---
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# Determines which AI classes contribute to the critical (Warning/Pause) streak
CRITICAL_FAILURES = os.getenv("CRITICAL_FAILURES", "spaghetti failure").lower().split()

WARNING_THRESHOLD = int(os.getenv("WARNING_THRESHOLD", 3))
PAUSE_THRESHOLD = int(os.getenv("PAUSE_THRESHOLD", 6))
CONCERN_THRESHOLD = int(os.getenv("CONCERN_THRESHOLD", 3))

# --- Storage & Retention ---
LAST_FRAME_PATH = "/tmp/latest.jpg"
FAILURES_DIR = os.getenv("FAILURES_DIR", "/failures")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./tmp/model_cache")
FAILURE_RETENTION_DAYS = int(os.getenv("FAILURE_RETENTION_DAYS", 7))
FAILURE_RETENTION_COUNT = int(os.getenv("FAILURE_RETENTION_COUNT", 20))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()