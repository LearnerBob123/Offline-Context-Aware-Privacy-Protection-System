"""
Temporal smoothing module.
Buffers the last N frames of decisions and smooths roles / scores
to avoid flickering in the output video.
Includes optional IoU-based cross-frame object tracking.
"""

from collections import deque
from typing import List, Dict, Optional
import numpy as np


def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TemporalSmoother:
    """Keeps a sliding window of frame outputs for smoothing."""

    def __init__(self, buffer_size: int = 5, iou_threshold: float = 0.4):
        self.buffer: deque = deque(maxlen=buffer_size)
        self.iou_threshold = iou_threshold

    def update(self, frame_output: dict) -> dict:
        """
        Accepts the current frame_output, smooths roles based on
        recent history, and returns the updated frame_output.
        """
        self.buffer.append(frame_output)
        if len(self.buffer) < 2:
            return frame_output

        current_dets = frame_output["detections"]
        for det in current_dets:
            history_roles = self._find_history(det)
            if history_roles:
                det["role"] = self._majority_vote(history_roles + [det["role"]])
        return frame_output

    def _find_history(self, det: dict) -> List[str]:
        """Look back through the buffer for the same object (by IoU)."""
        roles = []
        for past in list(self.buffer)[:-1]:  # all except current
            for pd in past["detections"]:
                if pd["label"] == det["label"] and _iou(pd["bbox"], det["bbox"]) > self.iou_threshold:
                    roles.append(pd["role"])
                    break
        return roles

    @staticmethod
    def _majority_vote(roles: List[str]) -> str:
        counts: Dict[str, int] = {}
        for r in roles:
            counts[r] = counts.get(r, 0) + 1
        return max(counts, key=counts.get)
