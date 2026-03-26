"""
Saliency estimation module (optional).

Two backends:
  - "spectral"  : fast, no extra dependencies (OpenCV spectral residual)
  - "u2net"     : high-quality, requires cloned U-2-Net repo + checkpoint

Controlled by config.SALIENCY_BACKEND.
"""

import os
import sys

import cv2
import numpy as np

from core.base_module import BaseModule
from config.config import PipelineConfig


# =====================================================================
# Spectral-residual (lightweight, always available)
# =====================================================================
class SpectralSaliency:
    """CPU-only spectral residual saliency – no model download needed."""

    def compute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, angle = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
        log_mag = np.log(magnitude + 1e-9)
        spectral_residual = log_mag - cv2.boxFilter(log_mag, -1, (3, 3))
        x = np.cos(angle) * np.exp(spectral_residual)
        y = np.sin(angle) * np.exp(spectral_residual)
        dft[:, :, 0], dft[:, :, 1] = x, y
        sal = cv2.idft(dft)
        sal = cv2.magnitude(sal[:, :, 0], sal[:, :, 1]) ** 2
        sal = cv2.GaussianBlur(sal, (11, 11), 2.5)
        sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return sal


# =====================================================================
# U-2-Net  (high quality, needs repo + checkpoint)
# =====================================================================
class U2NetSaliency:
    """
    Uses U2NETP (lightweight variant) for saliency prediction.

    Expects:
      - Cloned repo at  config.U2NET_REPO_PATH  (default: ./U-2-Net)
      - Checkpoint at   <repo>/saved_models/u2netp.pth
    """

    def __init__(self, config: PipelineConfig):
        import torch
        from torchvision import transforms

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -- import U2NETP from the cloned repo --
        repo = os.path.abspath(config.U2NET_REPO_PATH)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from model import U2NETP  # type: ignore

        self.net = U2NETP(3, 1)
        ckpt_path = os.path.join(repo, "saved_models", "u2netp.pth")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._torch = torch
        print("[Saliency] U2NETP loaded on", self.device)

    def compute(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            d0 = self.net(tensor)[0]  # finest-scale output
        sal = d0.squeeze().cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        sal = (sal * 255).astype(np.uint8)
        sal = cv2.resize(sal, (frame.shape[1], frame.shape[0]))
        return sal

    def release(self):
        del self.net
        self.net = None
        self._torch.cuda.empty_cache()


# =====================================================================
# Public module (selects backend via config)
# =====================================================================
class SaliencyModule(BaseModule):
    """Saliency with pluggable backend ('spectral' or 'u2net')."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._backend = None

    def initialize(self) -> None:
        backend = self.config.SALIENCY_BACKEND
        if backend == "u2net":
            self._backend = U2NetSaliency(self.config)
            print("[Saliency] Backend: U-2-Net (U2NETP)")
        else:
            self._backend = SpectralSaliency()
            print("[Saliency] Backend: spectral residual")

    def release(self) -> None:
        if hasattr(self._backend, "release"):
            self._backend.release()
        self._backend = None

    def compute(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a saliency map (uint8, 0-255), or None if not ready."""
        if self._backend is None:
            return None
        return self._backend.compute(frame)
