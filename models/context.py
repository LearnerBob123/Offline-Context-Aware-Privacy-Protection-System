"""
CLIP-based scene context module.
Classifies the scene using prompt-based zero-shot classification.
"""

import torch
import numpy as np
from PIL import Image

from core.base_module import BaseModule
from config.config import PipelineConfig
from utils.helpers import crop_and_convert


class ContextModule(BaseModule):
    """Zero-shot scene classification with CLIP."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.text_features = None

    def initialize(self) -> None:
        import clip
        self.model, self.preprocess = clip.load(
            self.config.CLIP_MODEL_NAME, device=self.device
        )
        # Pre-encode text prompts once
        tokens = clip.tokenize(self.config.CONTEXT_PROMPTS).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def release(self) -> None:
        del self.model, self.text_features
        self.model = self.text_features = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def classify_scene(self, frame: np.ndarray) -> dict:
        """
        Returns:
            {"label": str, "confidence": float}
        """
        pil = crop_and_convert(frame)
        img_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        similarity = (img_feat @ self.text_features.T).squeeze(0)
        probs = similarity.softmax(dim=0).cpu().numpy()
        best = int(np.argmax(probs))

        # Strip "a " / "an " prefix for cleaner labels
        raw_label = self.config.CONTEXT_PROMPTS[best]
        label = raw_label.lstrip("a ").lstrip("n ").strip()

        return {"label": label, "confidence": round(float(probs[best]), 4)}

    @torch.no_grad()
    def embed_image(self, frame: np.ndarray) -> list | None:
        """Return a CLIP image embedding vector."""
        pil = crop_and_convert(frame)
        img_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_tensor)
        return feat.cpu().numpy().flatten().tolist()
