"""
Context-aware decision engine.
Routes decisions to the appropriate strategy based on CLIP scene
context, then applies shared post-rules (plates, screens, faces).
Every detection receives an explainable ``decision_trace``.
"""

import numpy as np
from config.config import PipelineConfig
from engine.strategies import get_strategy, _inside


class DecisionEngine:
    """Thin router: strategy selection → execute → shared post-rules."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    def decide(self, frame_output: dict) -> dict:
        """
        Mutates frame_output['detections'] in place:
            - sets 'importance_score'
            - sets 'role'  →  "main" | "background" | "blur"
            - sets 'decision_trace' (explainable dict)
        Returns the updated frame_output.
        """
        dets = frame_output["detections"]
        if not dets:
            return frame_output

        # --- Select strategy from CLIP context ---
        context_label = (frame_output.get("context") or {}).get("label", "")
        strategy = get_strategy(context_label)
        strategy_name = type(strategy).__name__

        # --- Execute strategy (scores, roles, traces) ---
        strategy.decide(dets, frame_output, self.config)

        # --- Shared post-rules (override strategy if needed) ---
        self._apply_screen_rules(dets, strategy_name)
        self._apply_plate_rules(dets)
        self._apply_face_rules(dets)

        # Tag every trace with the strategy used
        for d in dets:
            trace = d.get("decision_trace")
            if trace:
                trace["strategy"] = strategy_name

        return frame_output

    # ------------------------------------------------------------------
    # Shared post-rules
    # ------------------------------------------------------------------
    def _apply_screen_rules(self, dets: list, strategy_name: str) -> None:
        """Screens: blur if configured (Meeting always blurs regardless)."""
        for d in dets:
            if d["label"] in ("laptop", "tv", "monitor", "cell phone"):
                # MeetingStrategy already blurred screens; avoid duplicate tag
                if strategy_name == "MeetingStrategy":
                    continue
                if self.config.BLUR_SCREENS:
                    d["role"] = "blur"
                    d.setdefault("decision_trace", {}).setdefault(
                        "rules_applied", []).append("screen_blurred")
                else:
                    d["role"] = "background"

    def _apply_plate_rules(self, dets: list) -> None:
        """License plates: always blur (or background if disabled)."""
        for d in dets:
            if d["label"] == "license_plate":
                d["role"] = "blur" if self.config.BLUR_PLATES else "background"
                d.setdefault("decision_trace", {}).setdefault(
                    "rules_applied", []).append(
                    "plate_blurred" if self.config.BLUR_PLATES else "plate_background"
                )

    def _apply_face_rules(self, dets: list) -> None:
        """Faces: inherit the role of the enclosing person bbox."""
        main_person = next(
            (d for d in dets if d["label"] == "person" and d["role"] == "main"),
            None,
        )
        for d in dets:
            if d["label"] == "face":
                if main_person and _inside(d["bbox"], main_person["bbox"]):
                    d["role"] = "main"
                    d.setdefault("decision_trace", {}).setdefault(
                        "rules_applied", []).append("face_inherits_main_person")
                else:
                    d["role"] = ("blur" if self.config.BLUR_BACKGROUND_PEOPLE
                                 else "background")
                    d.setdefault("decision_trace", {}).setdefault(
                        "rules_applied", []).append("face_blurred_background")
