# AntiPasta: Local 3D Print Failure Detection for Intel and ARM

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue?logo=docker)](https://www.docker.com/)
[![Home Assistant 2026 Ready](https://img.shields.io/badge/Home_Assistant-2026_Ready-41BDF5?logo=home-assistant)](https://www.home-assistant.io/)

AntiPasta provides a self-hosted AI failure detection system for a single 3D printer. It processes camera frames, evaluates failure confidence, and publishes alerts via MQTT.

The system is platform-aware and hardware-optimized. It utilizes hardware acceleration (Intel VAAPI or ARM V4L2 for video decoding; Intel OpenVINO or ARM NEON/XNNPACK for model inference) for extremely low resource usage. It supports standard RTSP/RTSPS streams and the proprietary Bambu Lab protocol used by the P1 and A1 series printers for camera access.

**IMPORTANT:**

1. This system **does not** control your printer. It provides MQTT messages (with discovery-enabled entities) with which _you_ can determine how to handle failures using Home Assistant or any other means.
2. I only own a P1S, so that's the main target here.
3. Believe the AI at your own risk.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Setup and Installation](#setup-and-installation)
4. [Home Assistant Integration](#home-assistant-integration)
5. [HTTP API](#http-api)
6. [Storage & Retention](#storage--retention)
7. [Acknowledgements](#acknowledgements)

---

## System Architecture

The project consists of two linked Docker containers communicating over a private bridge network.

1. **`antipasta-ml` (Unified AI Backend):** A consolidated inference engine that adapts to your hardware. It supports two interchangeable architectures:
   - **YOLOv11 (Recommended):** A high-accuracy, multi-class model ([`ApatheticWithoutTheA/3D-Print-Failure-Detector`](https://huggingface.co/ApatheticWithoutTheA/3D-Print-Failure-Detector)) that can detect multiple types of issues, including spaghetti, stringing, and zits.
   - **Obico:** Runs the [Obico Server](https://github.com/TheSpaghettiDetective/obico-server) offline ONNX model.
2. **`antipasta-logic` (Controller):** The central orchestrator. It handles video hardware discovery, camera polling, exclusion masking, heuristic thresholding, and MQTT communication.

Together, they use about 2GB of disk space with model weights, and under 1GB of RAM.

### Camera Connection

By default, AntiPasta polls the printer camera to prevent monopolizing the limited network connection pool of many 3D printers and IP cameras. When polling, it establishes a brief connection, captures a single frame to disk, and immediately disconnects, ensuring you can still view the stream in your slicer or mobile app.

You can optionally enable a persistent background connection. Persistent connection eliminates handshake latency and uses an in-memory buffer, providing much faster frame updates for dashboards and the AI engine, but permanently consumes one of your printer's limited video connection slots.

AntiPasta supports TLS cert verification for camera streams, but it is disabled by default as many printers and IP cameras use self-signed certificates. If you want to use `STRICT_TLS` with a Bambu Lab printer, for example, you must also provide the Bambu Lab CA cert, which we will not provide.

All of the above is supported for both RTSP/RTSPS and the older "throw JPEGs down the pipe forever" protocol used by the Bambu Lab A1 and P1S.

To ensure system stability in either mode, AntiPasta enforces a strict 10MB memory safety limit per frame. If network packet loss or stream corruption causes a frame buffer to exceed this size, the system will safely drop the buffer and resynchronize to prevent memory exhaustion.

### Heuristic Evaluation

Failure detection uses a dual-track **streak heuristic** to minimize false positives and distinguish between catastrophic failures and cosmetic artifacts:

- **Critical Failures (e.g., Spaghetti):** Evaluated against a `WARNING_THRESHOLD` (to alert you that a failure is possibly occurring) and a `PAUSE_THRESHOLD` (confirmed failure).
- **Artifacts (e.g., Stringing, Zits):** Non-critical cosmetic issues are evaluated on a separate track against a `CONCERN_THRESHOLD`.

A single suspect frame will not trigger an alert; the system requires a configurable number of consecutive detections before issuing any of the above states.

What constitutes a _Critical Failure_ is [configurable](#setup-and-installation). Artifact detection is only available with the `yolov11` engine. The `obico` engine cannot distinguish between types and reports all issues as critical `failure`.

---

## Prerequisites

- **Docker & Docker Compose**
- **Hardware Acceleration:**
  - An Intel CPU with an integrated GPU (iGPU) or discrete Intel GPU supporting VAAPI and OpenVINO.
  - **OR** an ARM64 device supporting NEON/XNNPACK for AI inference (such as a Raspberry Pi 4) and, optionally, V4L2 for video decoding (Rasperry Pi 4 or 5).
- **MQTT Broker:** Required to receive printer status and to send detection alerts.

---

## Setup and Installation

### Step 1: Hardware Permissions (Linux/Intel only)

To utilize Intel hardware acceleration on Intel GPUs, the containers need access to the host's `/dev/dri` device. Identify the Group ID (GID) of the `render` group on your host:

```bash
getent group render | cut -d: -f3
```

Ensure the `group_add` value in `docker-compose.yml` matches this output (the default is `992`).

_Note: For ARM/Raspberry Pi users, the system will automatically utilize optimized CPU instructions (NEON/XNNPACK) and standard V4L2 nodes._

### Step 2: Configuration

Configuration is managed with environment variables. See `example.env` for all options. For a quick start, copy `example.env` to `.env` and edit as needed. Some notable configuration variables include:

| Variable                      | Description                                                                                                                                                                  |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ML_ENGINE`                   | `yolov11` (Recommended) or `obico`.                                                                                                                                          |
| `HW_ACCEL_MODE`               | `auto` (Default), `vaapi`, `v4l2`, or `cpu`. Determines the video decoding strategy.                                                                                         |
| `PRINTER_IP`                  | The local IP address of your 3D printer.                                                                                                                                     |
| `PRINTER_ACCESS_CODE`         | The LAN/Developer mode access code.                                                                                                                                          |
| `PRINTER_SERIAL`              | The serial number of the printer (used for unique MQTT IDs).                                                                                                                 |
| `STREAM_TYPE`                 | `chamber_image` for Bambu P1/A1 printers or `rtsp` for IP cameras and X1 series.                                                                                             |
| `CRITICAL_FAILURES`           | Space-separated list of AI classes considered print-ruining (e.g., `"spaghetti failure"`). Everything else is treated as a non-critical artifact.                            |
| `WARNING_THRESHOLD`           | Consecutive critical frames required to trigger a Warning alert (approaching pause).                                                                                         |
| `PAUSE_THRESHOLD`             | Consecutive critical frames required to trigger a Pause alert.                                                                                                               |
| `CONCERN_THRESHOLD`           | Consecutive artifact frames (e.g., stringing) required to trigger a Concern alert.                                                                                           |
| `EXCLUSION_ZONES`             | Exclude areas of the frame from detections. A JSON list of coordinates: `[[x1, y1, x2, y2], ...]`. (0.0 to 1.0 scale).                                                       |
| `MQTT_PRINTER_STATUS_TOPIC`   | The MQTT topic we will subscribe to for printer status.                                                                                                                      |
| `MQTT_PRINTER_STATE_PRINTING` | The state published to `MQTT_PRINTER_STATUS_TOPIC` that says the printer is currently printing (default, `"printing"`)                                                       |
| `MQTT_PRINTER_STATE_PAUSED`   | The status published to `MQTT_PRINTER_STATUS_TOPIC` that says the printer is currently paused (default, `"paused.*"`)                                                        |
| `ALLOW_TEST_API`              | Enables the `/test/` API endpoints. Default is `False` for security. Set to `True` **only** on secured local networks, see [HTTP Test Api](#http-test-api) for more details. |

The `MQTT_PRINTER_STATUS_PRINTING` and `MQTT_PRINTER_STATUS_PAUSED` variables support Python regular expressions to match a variety of status messages.

**I cannot stress this enough: _DO NOT_ enable ALLOW_TEST_API on public or insecure networks.**

### Step 3: Build and Deployment

Use the management script to build and start the containers. The build process is multi-stage to make rebuilding faster for config changes.

```bash
chmod +x manage.sh
./manage.sh build
```

The ML container will begin an initialization phase (downloading weights and optimizing the model). You can monitor progress via the `/stats/` endpoint or Docker logs.

---

## Home Assistant Integration

AntiPasta implements **MQTT Auto-Discovery**. With the [MQTT integration](https://www.home-assistant.io/integrations/mqtt/) installed and configured, it will automatically create a new device named `AntiPasta <Serial>` providing several entities:

- **Connectivity**: (Diagnostic) Tracks if the controller is online.
- **Failure Confidence:** (Sensor) The current maximum failure probability (0-100%).
- **Failure Warning:** (Binary Sensor) Turns `on` when a critical failure streak starts (approaching a pause).
- **Failure Pause:** (Binary Sensor) Turns `on` when a critical failure is confirmed.
- **Artifact Concern:** (Binary Sensor) Turns `on` when non-critical cosmetic artifacts (like stringing) are detected.
- **Inference Time:** (Diagnostic Sensor) Monitors AI backend performance in milliseconds.

### Synchronizing the Printer State

AntiPasta only analyzes frames when the printer is actively printing. You must configure Home Assistant to pass your printer's state to the topic defined in `MQTT_PRINTER_STATUS_TOPIC`.

```yaml
alias: "AntiPasta: Sync Printer Stage"
trigger:
  - platform: state
    entity_id: sensor.p1s_current_stage
action:
  - service: mqtt.publish
    data:
      topic: "homeassistant/sensor/p1s_12345_current_stage/state"
      payload: "{{ states('sensor.p1s_current_stage') | lower }}"
      retain: true
```

Be sure to update all entities to match your configuration, and ensure the published topic matches `MQTT_PRINTER_STATUS_TOPIC`.

---

## HTTP API

The Logic Controller provides an HTTP API on `CONTROLLER_PORT` (Default: 8080) for diagnostics.

Rate Limiting: To prevent resource exhaustion and protect printer camera connection pools, the API enforces rate limits per IP address. Exceeding these limits returns an `HTTP 429 Too Many Requests` error.

| Endpoint                          | Description                                                                                                                   |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| `/stats/`                         | Returns a snapshot of system metrics, MQTT configuration, and hardware acceleration support status.                           |
| `/frame.jpg`                      | During printing, returns the most recently captured frame. Otherwise, returns the live current image from the printer camera. |
| `/failure_frame.jpg?event=<UUID>` | Returns the archived, annotated frame from the most recent pause, warning, or concern event.                                  |

### HTTP Test API

The Test API endpoints allow you to see what the failure detection engine sees from your printer's live camera or by providing a remote test image. These endpoints are _disabled by default_ to prevent local service discovery, DoS, and other nastiness.

| Endpoint                  | Description                                                                              |
| :------------------------ | :--------------------------------------------------------------------------------------- |
| `/test/`                  | Captures a frame from the printer camera and returns an annoted failure detection image. |
| `/test/<REMOTE_URL>`      | Retrieves an image from a remote URL and returns an annoted failure detection image.     |
| `/test/json/`             | Same as above, but returns raw detection coordinates and confidence as JSON.             |
| `/test/json/<REMOTE_URL>` | Same as above, but runs detection on a remote image URL.                                 |

**IMPORTANT:** To use any of the `/test/` endpoints, you must set `ALLOW_TEST_API=True` in your environment configuration. This flag is default `false`, and not present in the provided `example.env` for **good reason**. _Enabling this option always emits a warning log. Do not enable this feature on public or other insecure networks._

---

## Storage & Retention

Failure frames are archived to the directory defined by `FAILURES_DIR` (Default: `/failures`). The system automatically cleans this directory at the start of every new print session based on the `FAILURE_RETENTION_DAYS` and `FAILURE_RETENTION_COUNT` variables in your `.env`.

---

## Acknowledgements

- **[Obico (The Spaghetti Detective)](https://github.com/TheSpaghettiDetective/obico-server):** For their battle-tested failure detection model and inspiration.
- **[ApatheticWithoutTheA](https://huggingface.co/ApatheticWithoutTheA):** For their high-accuracy YOLOv11 model.
- **[Intel OpenVINO](https://docs.openvino.ai/):** For the hardware acceleration toolkit.
