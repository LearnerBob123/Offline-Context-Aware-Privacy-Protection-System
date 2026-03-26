"""
Context-aware decision strategies.

Each strategy assigns roles and importance scores to detections
using saliency-aware scoring and produces an explainable
decision_trace per object.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List


# ======================================================================
# Helpers
# ======================================================================

def _base_importance(det: dict, cx: float, cy: float) -> float:
    """Larger + more centred → higher score."""
    bx, by = det["center"]
    dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) + 1.0
    return det["area"] / dist


def _saliency_boost(det: dict, gain: float) -> float:
    """Positive boost for salient objects."""
    if det.get("is_salient"):
        return gain * det.get("saliency_score", 0.0) * det["area"]
    return 0.0


def _saliency_penalty(det: dict, penalty: float, min_overlap: float) -> float:
    """Negative penalty for low-saliency objects."""
    overlap = det.get("saliency_overlap", 1.0)
    if overlap < min_overlap:
        return -penalty * det["area"]
    return 0.0


def _inside(inner_bbox, outer_bbox) -> bool:
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def _init_trace(base: float, boost: float, penalty: float) -> dict:
    return {
        "base_score": round(base, 4),
        "saliency_boost": round(boost, 4),
        "penalty": round(penalty, 4),
        "final_score": round(base + boost + penalty, 4),
        "rules_applied": [],
    }


# ======================================================================
# Base class
# ======================================================================

class BaseDecisionStrategy(ABC):
    """Interface that every strategy must implement."""

    @abstractmethod
    def decide(self, detections: List[dict], frame_output: dict, config) -> None:
        """Mutate *detections* in-place: set importance_score, role, decision_trace."""


# ======================================================================
# Default Strategy
# ======================================================================

class DefaultStrategy(BaseDecisionStrategy):
    """
    Fallback: area + distance scoring, saliency as soft weighting.
    Same behaviour as the original monolithic engine.
    """

    def decide(self, detections: List[dict], frame_output: dict, config) -> None:
        h, w = frame_output["frame_shape"]
        cx, cy = w / 2.0, h / 2.0

        for d in detections:
            base = _base_importance(d, cx, cy)
            boost = _saliency_boost(d, config.SAL_GAIN)
            penalty = _saliency_penalty(d, config.SAL_PENALTY, config.MIN_SAL_OVERLAP)
            score = base + boost + penalty
            d["importance_score"] = round(score, 4)

            trace = _init_trace(base, boost, penalty)
            if boost > 0:
                trace["rules_applied"].append("boosted_due_to_high_saliency")
            if penalty < 0:
                trace["rules_applied"].append("penalised_low_saliency_overlap")
            d["decision_trace"] = trace

        # Elect main person
        persons = [d for d in detections if d["label"] == "person"]
        if persons:
            main = max(persons, key=lambda d: d["importance_score"])
            main["role"] = "main"
            main["decision_trace"]["rules_applied"].append("selected_as_main_person")
            for p in persons:
                if p is not main:
                    p["role"] = ("blur" if config.BLUR_BACKGROUND_PEOPLE
                                 else "background")
                    p["decision_trace"]["rules_applied"].append(
                        "blurred_as_background_person"
                        if config.BLUR_BACKGROUND_PEOPLE
                        else "background_person"
                    )


# ======================================================================
# Meeting Strategy
# ======================================================================

class MeetingStrategy(BaseDecisionStrategy):
    """
    Meeting context: main = most salient + central person.
    Screens are blurred. Other persons go to blur/background.
    """

    def decide(self, detections: List[dict], frame_output: dict, config) -> None:
        h, w = frame_output["frame_shape"]
        cx, cy = w / 2.0, h / 2.0

        for d in detections:
            base = _base_importance(d, cx, cy)
            # In meetings, centrality matters more – add a centrality bonus
            bx, by = d["center"]
            dist_norm = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) / (np.sqrt(cx**2 + cy**2) + 1.0)
            centrality_bonus = (1.0 - dist_norm) * d["area"] * 0.3

            boost = _saliency_boost(d, config.SAL_GAIN)
            penalty = _saliency_penalty(d, config.SAL_PENALTY, config.MIN_SAL_OVERLAP)
            score = base + centrality_bonus + boost + penalty
            d["importance_score"] = round(score, 4)

            trace = _init_trace(base, boost, penalty)
            trace["base_score"] = round(base + centrality_bonus, 4)
            trace["final_score"] = round(score, 4)
            trace["rules_applied"].append("meeting_centrality_bonus")
            if boost > 0:
                trace["rules_applied"].append("boosted_due_to_high_saliency")
            if penalty < 0:
                trace["rules_applied"].append("penalised_low_saliency_overlap")
            d["decision_trace"] = trace

        # Elect main person (most salient + central)
        persons = [d for d in detections if d["label"] == "person"]
        if persons:
            main = max(persons, key=lambda d: d["importance_score"])
            main["role"] = "main"
            main["decision_trace"]["rules_applied"].append("selected_as_main_person")
            for p in persons:
                if p is not main:
                    p["role"] = ("blur" if config.BLUR_BACKGROUND_PEOPLE
                                 else "background")
                    p["decision_trace"]["rules_applied"].append(
                        "blurred_as_background_person"
                    )

        # Meeting: always blur screens
        for d in detections:
            if d["label"] in ("laptop", "tv", "monitor", "cell phone"):
                d["role"] = "blur"
                d["decision_trace"]["rules_applied"].append(
                    "screen_blurred_in_meeting"
                )


# ======================================================================
# Outdoor Strategy
# ======================================================================

class OutdoorStrategy(BaseDecisionStrategy):
    """
    Outdoor context: main = largest + most salient.
    Less emphasis on centrality. Background objects suppressed.
    """

    def decide(self, detections: List[dict], frame_output: dict, config) -> None:
        h, w = frame_output["frame_shape"]
        cx, cy = w / 2.0, h / 2.0

        for d in detections:
            # Outdoor: area dominates, very low centrality weight
            base = d["area"]  # pure area – no distance weighting
            boost = _saliency_boost(d, config.SAL_GAIN * 1.2)  # saliency matters more outdoors
            penalty = _saliency_penalty(d, config.SAL_PENALTY, config.MIN_SAL_OVERLAP)
            score = base + boost + penalty
            d["importance_score"] = round(score, 4)

            trace = _init_trace(base, boost, penalty)
            trace["rules_applied"].append("outdoor_area_dominant")
            if boost > 0:
                trace["rules_applied"].append("boosted_due_to_high_saliency")
            if penalty < 0:
                trace["rules_applied"].append("penalised_low_saliency_overlap")
            d["decision_trace"] = trace

        # Elect main – largest + most salient among all objects (not just persons)
        candidates = [d for d in detections
                      if d["label"] in ("person", "face")]
        if not candidates:
            candidates = detections  # fallback: any object

        if candidates:
            main = max(candidates, key=lambda d: d["importance_score"])
            main["role"] = "main"
            main["decision_trace"]["rules_applied"].append(
                "selected_as_main_outdoor"
            )

        # Suppress small background objects
        for d in detections:
            if d["role"] == "background" and d.get("relative_size", 0) < 0.02:
                d["decision_trace"]["rules_applied"].append(
                    "suppressed_small_background"
                )


# ======================================================================
# Registry
# ======================================================================

_STRATEGIES = {
    "meeting": MeetingStrategy,
    "outdoor": OutdoorStrategy,
    "default": DefaultStrategy,
}


def get_strategy(context_label: str) -> BaseDecisionStrategy:
    """Select a strategy based on the CLIP scene label."""
    label = context_label.lower()
    if "meeting" in label or "presentation" in label:
        return MeetingStrategy()
    if "outdoor" in label or "street" in label or "park" in label:
        return OutdoorStrategy()
    return DefaultStrategy()
