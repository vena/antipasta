"""
Detection Filter
----------------
Filters AI detections based on user-defined exclusion zones.
Uses Intersection over Area (IoA) to determine if a detection is suppressed.
"""

import cv2
import numpy as np
import logging
from typing import List
from core.models import Detection

logger = logging.getLogger("AntiPasta.Core.Filter")

class DetectionFilter:
    def __init__(self, exclusion_zones: List[List[float]]):
        self.zones = exclusion_zones

    def apply(self, detections: List[Detection], image_bytes: bytes) -> List[Detection]:
        """
        Suppresses detections where >50% of the detection area is within an exclusion zone.
        
        Logic Choice: Intersection over Area (IoA)
        We use IoA rather than the standard Intersection over Union (IoU) to ensure that 
        failures originating outside a zone are not suppressed even if they extend into it. 
        Only detections that are primarily contained within a "dead zone" are ignored.
        """
        if not detections or not self.zones:
            return detections

        # Decode image to get actual dimensions for coordinate translation
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fail safely: If the frame is corrupted and cannot be evaluated against 
            # the exclusion zones, we must drop the unverified detections to prevent 
            # false positives from escaping containment.
            logger.warning("Filtering failed: Could not decode image. Failing safely by dropping unverified detections.")
            return []
            
        img_h, img_w = img.shape[:2]
        filtered = []

        for det in detections:
            # Models now output absolute corners directly, skipping translation math
            dx1, dy1, dx2, dy2 = det.bbox
            
            is_excluded = False
            for zone in self.zones:
                try:
                    # zone is [x1_norm, y1_norm, x2_norm, y2_norm]
                    zx1, zy1 = zone[0] * img_w, zone[1] * img_h
                    zx2, zy2 = zone[2] * img_w, zone[3] * img_h
                    
                    # Overlap rectangle calculation
                    x_left = max(dx1, zx1)
                    y_top = max(dy1, zy1)
                    x_right = min(dx2, zx2)
                    y_bottom = min(dy2, zy2)
                    
                    if x_right > x_left and y_bottom > y_top:
                        overlap_area = (x_right - x_left) * (y_bottom - y_top)
                        det_area = (dx2 - dx1) * (dy2 - dy1)
                        
                        # The 0.5 threshold ensures that if the majority of the failure 
                        # is in the exclusion zone, it's considered a false positive.
                        if det_area > 0 and (overlap_area / det_area) > 0.5:
                            is_excluded = True
                            break
                except (IndexError, TypeError):
                    continue

            if not is_excluded:
                filtered.append(det)

        return filtered