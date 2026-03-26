"""
Feature engine – enriches raw detections with spatial features,
importance scores, and embeddings into a unified per-frame output.
"""

import numpy as np
from typing import Dict, List, Optional


def _box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


class FeatureEngine:
    """
    Merges detections from all categories into a flat enriched list
    with spatial metadata and optional embeddings.
    """

    def process(
        self,
        detections: Dict[str, List[dict]],
        frame_shape: tuple,
        frame_id: int,
        frame_embedding: Optional[list] = None,
        clip_embedding: Optional[list] = None,
        context: Optional[dict] = None,
        saliency_map: Optional[np.ndarray] = None,
        saliency_threshold: float = 0.5,
        overlap_threshold: float = 0.3,
    ) -> dict:
        h, w = frame_shape[:2]
        frame_area = h * w
        enriched: List[dict] = []
        obj_id = 0

        # Normalise saliency map to [0, 1] once
        sal_norm = None
        if saliency_map is not None:
            sal_norm = saliency_map.astype(np.float32) / 255.0

        for category, items in detections.items():
            for item in items:
                bbox = item["bbox"]
                area = _box_area(bbox)
                center = _box_center(bbox)
                rel_size = area / frame_area if frame_area else 0.0

                entry = {
                    "id": obj_id,
                    "bbox": bbox,
                    "label": item.get("label", category.rstrip("s")),
                    "confidence": item["confidence"],
                    "area": area,
                    "center": [round(center[0], 2), round(center[1], 2)],
                    "relative_size": round(rel_size, 6),
                    "importance_score": 0.0,   # filled by DecisionEngine
                    "role": "background",       # filled by DecisionEngine
                }

                # --- Per-object saliency scoring ---
                if sal_norm is not None:
                    x1, y1, x2, y2 = bbox
                    roi = sal_norm[y1:y2, x1:x2]
                    if roi.size > 0:
                        mean_sal = float(np.mean(roi))
                        salient_px = int(np.sum(roi > saliency_threshold))
                        sal_overlap = salient_px / roi.size
                        entry["saliency_score"] = round(mean_sal, 4)
                        entry["saliency_overlap"] = round(sal_overlap, 4)
                        entry["is_salient"] = sal_overlap > overlap_threshold
                    else:
                        entry["saliency_score"] = 0.0
                        entry["saliency_overlap"] = 0.0
                        entry["is_salient"] = False

                # Carry over optional embeddings
                if "visual_embedding" in item:
                    entry["visual_embedding"] = item["visual_embedding"]
                if "clip_embedding" in item:
                    entry["clip_embedding"] = item["clip_embedding"]

                enriched.append(entry)
                obj_id += 1

        return {
            "frame_id": frame_id,
            "frame_shape": [h, w],
            "num_objects": len(enriched),
            "detections": enriched,
            "frame_embedding": frame_embedding,
            "clip_embedding": clip_embedding,
            "context": context,
        }
