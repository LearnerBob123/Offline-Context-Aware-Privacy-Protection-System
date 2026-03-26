"""
ViT-based embedding module.
Provides frame-level and optional object-level visual embeddings.
"""

import torch
import numpy as np
from PIL import Image

from core.base_module import BaseModule
from config.config import PipelineConfig
from utils.helpers import crop_and_convert


class EmbeddingModule(BaseModule):
    """Extracts ViT embeddings at configurable intervals."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None

    def initialize(self) -> None:
        import timm
        from torchvision import transforms
        self.model = timm.create_model(
            self.config.VIT_MODEL_NAME, pretrained=True, num_classes=0
        ).to(self.device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def release(self) -> None:
        del self.model
        self.model = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _embed(self, pil_img: Image.Image):
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.model(tensor).cpu().numpy().flatten().tolist()

    def embed_frame(self, frame: np.ndarray) -> list | None:
        """Return a ViT embedding for the full frame."""
        pil = crop_and_convert(frame)
        return self._embed(pil)

    def embed_objects(self, frame: np.ndarray, detections: dict) -> dict:
        """Attach 'visual_embedding' to each detection item in-place."""
        for cat in detections.values():
            for item in cat:
                pil = crop_and_convert(frame, item["bbox"])
                if pil is None:
                    continue
                item["visual_embedding"] = self._embed(pil)
        return detections
