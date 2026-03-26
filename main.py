"""
Entry point – parse CLI args and launch the pipeline.

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --output outputs/result.mp4 --max-frames 200
    python main.py --input video.mp4 --no-clip --no-render
"""

import argparse
import sys
import os

# Ensure project root is on the path so imports work when
# main.py is invoked directly from the project directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import PipelineConfig
from core.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modular video processing pipeline "
                    "(detection → embedding → context → decision → render → log)"
    )
    p.add_argument("--input", type=str, default=None,
                   help="Path to input video (default: config.INPUT_VIDEO)")
    p.add_argument("--output", type=str, default=None,
                   help="Path to output video (default: config.OUTPUT_VIDEO)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames to process (0 = all)")

    # Toggle flags (all enabled by default; use --no-* to disable)
    p.add_argument("--no-yolo", action="store_true", help="Disable YOLO detection")
    p.add_argument("--no-vit", action="store_true", help="Disable ViT embeddings")
    p.add_argument("--no-clip", action="store_true", help="Disable CLIP context")
    p.add_argument("--no-render", action="store_true", help="Disable video output")
    p.add_argument("--no-log", action="store_true", help="Disable JSONL logging")
    p.add_argument("--saliency", action="store_true", help="Enable saliency module")
    p.add_argument("--saliency-backend", type=str, choices=["spectral", "u2net"],
                   default=None, help="Saliency backend: 'spectral' (fast) or 'u2net' (U-2-Net)")
    p.add_argument("--saliency-render", action="store_true",
                   help="Enable saliency heatmap video output (implies --saliency)")
    p.add_argument("--sal-mode", type=str, choices=["overlay", "mask"],
                   default=None,
                   help="Saliency render mode: 'overlay' (blended) or 'mask' (pure heatmap)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()

    # Apply CLI overrides
    if args.input:
        cfg.INPUT_VIDEO = args.input
    if args.output:
        cfg.OUTPUT_VIDEO = args.output
    if args.max_frames:
        cfg.MAX_FRAMES = args.max_frames
    if args.no_yolo:
        cfg.USE_YOLO = False
    if args.no_vit:
        cfg.USE_VIT = False
    if args.no_clip:
        cfg.USE_CLIP = False
    if args.no_render:
        cfg.ENABLE_RENDER = False
    if args.no_log:
        cfg.SAVE_JSON = False
    if args.saliency:
        cfg.USE_SALIENCY = True
    if args.saliency_backend:
        cfg.SALIENCY_BACKEND = args.saliency_backend
        cfg.USE_SALIENCY = True  # implicitly enable
    if args.saliency_render:
        cfg.ENABLE_SALIENCY_RENDER = True
        cfg.USE_SALIENCY = True  # implicitly enable
    if args.sal_mode:
        cfg.SAL_RENDER_MODE = args.sal_mode
        cfg.ENABLE_SALIENCY_RENDER = True
        cfg.USE_SALIENCY = True

    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.OUTPUT_VIDEO) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.OUTPUT_LOG) or ".", exist_ok=True)
    if cfg.ENABLE_SALIENCY_RENDER:
        os.makedirs(os.path.dirname(cfg.OUTPUT_SALIENCY_VIDEO) or ".", exist_ok=True)

    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
