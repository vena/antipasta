"""
Model Adapter Base
------------------
Defines the standard interface for model-specific inference logic.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple

class ModelAdapter(ABC):
    @abstractmethod
    def load(self, hardware_provider: str, status_callback) -> str:
        """
        Loads the model and performs any required format exports.
        Returns the name of the hardware device utilized.
        """
        pass

    @abstractmethod
    def infer(self, img_bytes: bytes) -> List[List]:
        """
        Performs inference and returns results in standard AntiPasta format:
        [[class_name, confidence, [x1, y1, x2, y2]], ...]
        """
        pass
    
    def warmup(self):
        """
        Sends a blank image to the model for warmup and discards the result.
        """
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded_image = cv2.imencode('.png', blank_image)
        if not success:
            raise ValueError("Failed to encode warmup image.")
        blank_image_bytes = encoded_image.tobytes()
        self.infer(blank_image_bytes)
        pass