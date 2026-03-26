"""
Shared helper functions used across the pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, List


def crop_and_convert(
    frame: np.ndarray,
    bbox: Optional[List[int]] = None,
    min_size: int = 5,
) -> Optional[Image.Image]:
    """
    Crop a region from a BGR frame and return a PIL RGB Image.
    If bbox is None the full frame is returned.
    Returns None if the crop is too small.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        frame = frame[y1:y2, x1:x2]
        if frame.size == 0:
            return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
