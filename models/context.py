"""
CLIP-based scene context module.
Classifies the scene using prompt-based zero-shot classification.

Uses HuggingFace `transformers` CLIP (already installed) so no separate
`openai/CLIP` pip package is required.  OpenAI model-name aliases such as
"ViT-B/32" are automatically mapped to their HuggingFace equivalents.
"""

import torch
import numpy as np
from PIL import Image

from core.base_module import BaseModule
from config.config import PipelineConfig
from utils.helpers import crop_and_convert

# Map OpenAI CLIP short-names → HuggingFace model IDs
_HF_MODEL_MAP = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
}


class ContextModule(BaseModule):
    """Zero-shot scene classification with CLIP (via HuggingFace transformers)."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.text_features = None

    def initialize(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        hf_name = _HF_MODEL_MAP.get(
            self.config.CLIP_MODEL_NAME, self.config.CLIP_MODEL_NAME
        )
        print(f"[ContextModule] Loading CLIP from HuggingFace: {hf_name}")
        self.model = CLIPModel.from_pretrained(
            hf_name, use_safetensors=True
        ).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(hf_name)

        # Pre-encode text prompts once
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.config.CONTEXT_PROMPTS,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            # Use text_model + projection directly (stable across transformers versions)
            text_out = self.model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            feats = self.model.text_projection(text_out.pooler_output).float()
            self.text_features = feats / feats.norm(dim=-1, keepdim=True)

    def release(self) -> None:
        del self.model, self.text_features
        self.model = self.text_features = self.processor = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def classify_scene(self, frame: np.ndarray) -> dict:
        """
        Returns:
            {"label": str, "confidence": float}
        """
        pil = crop_and_convert(frame)
        img_inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        vision_out = self.model.vision_model(pixel_values=img_inputs["pixel_values"])
        img_feat = self.model.visual_projection(vision_out.pooler_output).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

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
        img_inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        vision_out = self.model.vision_model(pixel_values=img_inputs["pixel_values"])
        feat = self.model.visual_projection(vision_out.pooler_output).float()
        return feat.cpu().numpy().flatten().tolist()
