"""
Robust video reader / writer with FPS fallback and proper resource cleanup.
"""

import cv2
import numpy as np
from config.config import PipelineConfig


class VideoReader:
    """Wraps cv2.VideoCapture with metadata helpers."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

    @property
    def fps(self) -> float:
        f = self.cap.get(cv2.CAP_PROP_FPS)
        return f if f > 0 else 25.0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """Robust writer: handles FPS=0 fallback, ensures uint8 frames."""

    def __init__(self, path: str, fps: float, width: int, height: int, config: PipelineConfig):
        self.path = path
        fps = fps if fps > 0 else config.FALLBACK_FPS
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise IOError(f"Cannot open video writer: {path}")

    def write(self, frame: np.ndarray):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
