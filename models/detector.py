"""
YOLO-based detection module for objects, faces, and license plates.
Outputs a standardised list of detections per category.
"""

from typing import Dict, List
from core.base_module import BaseModule
from config.config import PipelineConfig

# Labels that count as "screen" objects
_SCREEN_LABELS = {"laptop", "tv", "monitor", "cell phone"}


class Detector(BaseModule):
    """Runs three YOLO models and returns unified detections."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.yolo_obj = None
        self.yolo_face = None
        self.yolo_plate = None

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        from ultralytics import YOLO
        self.yolo_obj = YOLO(self.config.YOLO_OBJ_WEIGHTS)
        self.yolo_face = YOLO(self.config.YOLO_FACE_WEIGHTS)
        try:
            self.yolo_plate = YOLO(self.config.YOLO_PLATE_WEIGHTS)
        except Exception as e:
            print(f"[Detector] License-plate model unavailable: {e}")
            self.yolo_plate = None

    def release(self) -> None:
        del self.yolo_obj, self.yolo_face, self.yolo_plate
        self.yolo_obj = self.yolo_face = self.yolo_plate = None

    # ------------------------------------------------------------------
    def detect(self, frame) -> Dict[str, List[dict]]:
        """
        Returns:
            {
              "persons":  [{"bbox": [x1,y1,x2,y2], "label": "person",  "confidence": float}, …],
              "faces":    [...],
              "plates":   [...],
              "objects":  [...]   # screens / devices
            }
        """
        detections: Dict[str, List[dict]] = {
            "persons": [], "faces": [], "plates": [], "objects": []
        }

        # --- General object detection ---
        obj_res = self.yolo_obj(
            frame, imgsz=self.config.IMG_SIZE,
            conf=self.config.OBJ_CONF, verbose=False
        )[0]
        for box in obj_res.boxes:
            label = self.yolo_obj.names[int(box.cls[0])]
            item = {
                "bbox": list(map(int, box.xyxy[0])),
                "label": label,
                "confidence": round(float(box.conf[0]), 4),
            }
            if label == "person":
                detections["persons"].append(item)
            elif label in _SCREEN_LABELS:
                detections["objects"].append(item)

        # --- Face detection ---
        face_res = self.yolo_face(
            frame, imgsz=self.config.IMG_SIZE,
            conf=self.config.FACE_CONF, verbose=False
        )[0]
        for box in face_res.boxes:
            detections["faces"].append({
                "bbox": list(map(int, box.xyxy[0])),
                "label": "face",
                "confidence": round(float(box.conf[0]), 4),
            })

        # --- License plate detection ---
        if self.yolo_plate is not None:
            plate_res = self.yolo_plate(
                frame, imgsz=self.config.IMG_SIZE,
                conf=self.config.PLATE_CONF, verbose=False
            )[0]
            for box in plate_res.boxes:
                detections["plates"].append({
                    "bbox": list(map(int, box.xyxy[0])),
                    "label": "license_plate",
                    "confidence": round(float(box.conf[0]), 4),
                })

        return detections
