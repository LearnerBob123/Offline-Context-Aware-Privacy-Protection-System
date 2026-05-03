"""
YOLO-based detection module for objects, faces, and license plates.
Outputs a standardised list of detections per category.
"""

from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # Private inference helpers (each safe to call from a worker thread)
    # ------------------------------------------------------------------
    def _infer_obj(self, frame):
        return self.yolo_obj(
            frame, imgsz=self.config.IMG_SIZE,
            conf=self.config.OBJ_CONF, verbose=False
        )[0]

    def _infer_face(self, frame):
        return self.yolo_face(
            frame, imgsz=self.config.IMG_SIZE,
            conf=self.config.FACE_CONF, verbose=False
        )[0]

    def _infer_plate(self, frame):
        if self.yolo_plate is None:
            return None
        return self.yolo_plate(
            frame, imgsz=self.config.IMG_SIZE,
            conf=self.config.PLATE_CONF, verbose=False
        )[0]

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

        if self.config.PARALLEL_YOLO:
            # Issue all three model calls in parallel.
            # PyTorch releases the GIL during CUDA kernels so threads overlap.
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_obj   = pool.submit(self._infer_obj,   frame)
                fut_face  = pool.submit(self._infer_face,  frame)
                fut_plate = pool.submit(self._infer_plate, frame)
                obj_res   = fut_obj.result()
                face_res  = fut_face.result()
                plate_res = fut_plate.result()
        else:
            obj_res   = self._infer_obj(frame)
            face_res  = self._infer_face(frame)
            plate_res = self._infer_plate(frame)

        # --- Parse general object results ---
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

        # --- Parse face results ---
        for box in face_res.boxes:
            detections["faces"].append({
                "bbox": list(map(int, box.xyxy[0])),
                "label": "face",
                "confidence": round(float(box.conf[0]), 4),
            })

        # --- Parse license plate results ---
        if plate_res is not None:
            for box in plate_res.boxes:
                detections["plates"].append({
                    "bbox": list(map(int, box.xyxy[0])),
                    "label": "license_plate",
                    "confidence": round(float(box.conf[0]), 4),
                })

        return detections
