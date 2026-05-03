"""
Visualization / rendering utilities.
Draws bounding boxes, labels, blur regions, HUD overlays,
and optional saliency heatmap overlay.
"""

import cv2
import numpy as np
from config.config import PipelineConfig

# Colour palette (BGR)
_COLORS = {
    "person": (0, 255, 0),
    "laptop": (255, 0, 0),
    "tv": (255, 0, 0),
    "monitor": (255, 0, 0),
    "cell phone": (255, 0, 0),
    "face": (0, 255, 255),
    "license_plate": (0, 165, 255),
}
_DEFAULT_COLOR = (0, 255, 255)
_SALIENT_COLOR = (0, 255, 0)      # Green – salient
_NON_SALIENT_COLOR = (0, 0, 255)  # Red   – non-salient


class Renderer:
    """Draws annotations, blur regions, and saliency overlays."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.blur_k = config.BLUR_KERNEL_SIZE

    # ------------------------------------------------------------------
    # Main annotated output
    # ------------------------------------------------------------------
    def render_blur_only(self, frame: np.ndarray, frame_output: dict) -> np.ndarray:
        """Return a clean copy of the frame with blur regions applied and nothing else.

        No bounding boxes, no labels, no HUD — suitable for live preview.
        """
        canvas = frame.copy()
        for det in frame_output.get("detections", []):
            if det["role"] == "blur":
                self._apply_blur(canvas, det["bbox"])
        return canvas

    def render(self, frame: np.ndarray, frame_output: dict) -> np.ndarray:
        canvas = frame.copy()
        h, w = frame_output["frame_shape"]
        has_saliency = any("is_salient" in d for d in frame_output.get("detections", []))

        # --- Apply blur first (so labels draw on top) ---
        for det in frame_output.get("detections", []):
            if det["role"] == "blur":
                self._apply_blur(canvas, det["bbox"])

        # --- HUD ---
        hud = [
            f"Frame: {frame_output['frame_id']}",
            f"Objects: {frame_output['num_objects']}",
        ]
        ctx = frame_output.get("context")
        if ctx:
            hud.append(f"Scene: {ctx['label']} ({ctx['confidence']:.2f})")
        # Show active strategy if present
        first_trace = next(
            (d.get("decision_trace") for d in frame_output.get("detections", [])
             if d.get("decision_trace")), None,
        )
        if first_trace and "strategy" in first_trace:
            hud.append(f"Strategy: {first_trace['strategy']}")
        for i, line in enumerate(hud):
            cv2.putText(canvas, line, (20, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- Bounding boxes & labels ---
        for det in frame_output.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]

            # Saliency-aware colouring when saliency data is present
            if has_saliency:
                color = _SALIENT_COLOR if det.get("is_salient") else _NON_SALIENT_COLOR
            else:
                color = _COLORS.get(det["label"], _DEFAULT_COLOR)

            thickness = 3 if det["role"] == "main" else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            tag = f"{det['label']} | {det['confidence']:.2f} | {det['role']}"
            if "saliency_score" in det:
                tag += f" | sal:{det['saliency_score']:.2f}"
            pos = (x1, max(y1 - 8, 20))
            cv2.putText(canvas, tag, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(canvas, tag, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        return canvas

    # ------------------------------------------------------------------
    # Pure saliency heatmap video (no boxes, no text)
    # ------------------------------------------------------------------
    def render_saliency_heatmap(
        self, frame: np.ndarray, saliency_map,
    ) -> np.ndarray:
        """Generate a clean saliency visualisation.

        Modes (controlled by config.SAL_RENDER_MODE):
            "overlay" – JET heatmap blended 50/50 with the original frame.
            "mask"    – pure JET heatmap only (no original frame).

        No bounding boxes, labels, or text are drawn.
        """
        if saliency_map is None:
            return frame.copy()

        # Min-max normalise to [0, 1]
        sal = saliency_map.astype(np.float32)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        heatmap = cv2.applyColorMap(
            (sal * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        if self.config.SAL_RENDER_MODE == "mask":
            return heatmap

        # Default: overlay
        return cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # ------------------------------------------------------------------
    def _apply_blur(self, frame: np.ndarray, bbox: list) -> None:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(
            roi, (self.blur_k, self.blur_k), 0
        )


# ----------------------------------------------------------------------
# Module-level helper (no Renderer instance needed)
# ----------------------------------------------------------------------
def render_detections_frame(frame: np.ndarray, detections: dict) -> np.ndarray:
    """Draw raw YOLO bounding boxes and labels on a copy of the frame.

    No blur is applied — this is the "detection-only" intermediate output.

    Parameters
    ----------
    frame      : BGR uint8 numpy array (H x W x 3).
    detections : dict returned by Detector.detect()  —
                 keys are category names, values are lists of detection dicts
                 each with "bbox", "label", and "confidence" keys.

    Returns
    -------
    Annotated BGR frame copy.
    """
    canvas = frame.copy()
    for items in detections.values():
        for det in items:
            x1, y1, x2, y2 = det["bbox"]
            color = _COLORS.get(det["label"], _DEFAULT_COLOR)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            tag = f"{det['label']} {det['confidence']:.2f}"
            pos = (x1, max(y1 - 6, 14))
            # Black outline for readability on any background
            cv2.putText(canvas, tag, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(canvas, tag, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
    return canvas
