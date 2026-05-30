"""
Annotation Handler
------------------
Responsible strictly for the visual representation of AI data.
Draws bounding boxes, telemetry overlays, and translucent exclusion masks 
onto image byte arrays.
"""

import cv2
import numpy as np
import io
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import config
from core.models import Detection

logger = logging.getLogger("AntiPasta.AnnotationHandler")

COLOR_ZONE_BASE = (40, 40, 40)    # Dark Charcoal
COLOR_ZONE_HATCH = (90, 90, 90)   # Muted Gray
COLOR_TELEMETRY_BG = (35, 35, 35) # Near Black
HATCH_SPACING = 12                
HATCH_THICKNESS = 2
OPACITY_OVERLAY = 0.4             

def _get_color_for_confidence(conf: float, threshold: float) -> Tuple[int, int, int]:
    """Calculates a BGR color tuple using an anchored green-to-red gradient."""
    if conf < threshold:
        ratio = conf / threshold if threshold > 0 else 0
        return (0, 255, int(255 * ratio))
    else:
        range_span = 1.0 - threshold
        ratio = (conf - threshold) / range_span if range_span > 0 else 1
        return (0, int(165 * (1.0 - ratio)), 255)

def _get_text_color(bgr_bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Returns accessible Black or White text based on background luminance."""
    b, g, r = bgr_bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 127 else (255, 255, 255)

def _draw_hatched_zone(img: np.ndarray, zone: List[float]) -> None:
    """Draws a hazard pattern over an exclusion zone."""
    img_h, img_w = img.shape[:2]
    zx1, zy1, zx2, zy2 = zone
    x1, y1 = int(zx1 * img_w), int(zy1 * img_h)
    x2, y2 = int(zx2 * img_w), int(zy2 * img_h)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_ZONE_BASE, -1)
    
    roi = img[y1:y2, x1:x2]
    rw, rh = roi.shape[1], roi.shape[0]
    
    if rw > 0 and rh > 0:
        for i in range(-rh, rw, HATCH_SPACING):
            cv2.line(roi, (i, rh), (i + rh, 0), COLOR_ZONE_HATCH, HATCH_THICKNESS, cv2.LINE_AA)

def draw_exclusion_zones(img: np.ndarray, exclusion_zones: List[List[float]]) -> np.ndarray:
    """Overlays the hazard masks on the image before AI detections are drawn."""
    if not exclusion_zones:
        return img
        
    overlay = img.copy()
    for zone in exclusion_zones:
        try:
            _draw_hatched_zone(overlay, zone)
        except Exception as e:
            logger.debug(f"Hatching failure: {e}")

    return cv2.addWeighted(overlay, OPACITY_OVERLAY, img, 1.0 - OPACITY_OVERLAY, 0)

def draw_detections(
    image_bytes: bytes, 
    detections: List[Detection], 
    inference_time: float = 0.0, 
    exclusion_zones: Optional[List] = None
) -> Optional[io.BytesIO]:
    """Orchestrates the annotation process and returns a valid JPEG payload."""
    if not image_bytes:
        return None
        
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    if exclusion_zones:
        img = draw_exclusion_zones(img, exclusion_zones)

    # VISUAL PRIORITY SORTING:
    # We sort by confidence so that high-confidence detections are drawn last.
    # This ensures that in overlapping scenarios, the detection the AI is most 
    # certain about appears at the top of the 'Z-stack' for maximum visual clarity.
    sorted_dets = sorted(detections, key=lambda x: x.confidence)
    
    max_conf = sorted_dets[-1].confidence if sorted_dets else 0.0
    primary_type = sorted_dets[-1].class_name if sorted_dets else None

    for det in sorted_dets:
        # Bounding boxes are natively formatted for drawing
        x1, y1, x2, y2 = map(int, det.bbox)
        
        color = _get_color_for_confidence(det.confidence, config.CONFIDENCE_THRESHOLD)
        thickness = 3 if det.confidence >= config.CONFIDENCE_THRESHOLD else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        conf_pct = det.confidence * 100
        label = f"{det.class_name.capitalize()}: {conf_pct:.1f}%" if det.class_name != "failure" else f"{conf_pct:.1f}%"
        
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), font, scale, _get_text_color(color), thick, cv2.LINE_AA)

    # Telemetry Panel
    def draw_tag(text, y_pos, bg_color):
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad = 6
        cv2.rectangle(img, (10, y_pos - th - pad), (10 + tw + (pad*2), y_pos + pad), bg_color, -1)
        cv2.putText(img, text, (10 + pad, y_pos), font, scale, _get_text_color(bg_color), thick, cv2.LINE_AA)
        return y_pos + th + (pad * 2) + 5
        
    curr_y = 30
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    curr_y = draw_tag(timestamp, curr_y, COLOR_TELEMETRY_BG)
    curr_y = draw_tag(f"Inference: {int(inference_time * 1000)}ms", curr_y, COLOR_TELEMETRY_BG)
    
    max_label = f" ({primary_type.capitalize()})" if primary_type and primary_type != "failure" else ""
    draw_tag(f"Max Conf: {max_conf * 100:.1f}%{max_label}", curr_y, _get_color_for_confidence(max_conf, config.CONFIDENCE_THRESHOLD))

    _, buffer = cv2.imencode('.jpg', img)
    # Using .tobytes() guarantees a standardized byte string payload, 
    # preventing truncated responses from Werkzeug dynamic length calculation.
    return io.BytesIO(buffer.tobytes())