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
    p.add_argument("--no-vit",  action="store_true", help="Disable ViT embeddings (already off by default)")
    p.add_argument("--vit",     action="store_true", help="Enable ViT object/frame embeddings (stored in log only; does not affect blur quality)")
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

    # Performance / parallelism
    p.add_argument("--parallel-yolo", action="store_true",
                   help="Run all 3 YOLO models concurrently (ThreadPoolExecutor)")
    p.add_argument("--frame-buffer-size", type=int, default=None, metavar="N",
                   help="Depth of the read/display queues (default: 4)")
    p.add_argument("--object-embed-interval", type=int, default=None, metavar="N",
                   help="Run ViT object embeddings every N frames (default: 10; 1=every frame)")
    p.add_argument("--clip-interval", type=int, default=None, metavar="N",
                   help="Run CLIP scene classification every N frames (default: 10)")

    # Intermediate output
    p.add_argument("--save-det-video", action="store_true",
                   help="Save a detection-only video (raw frame + bboxes, no blur)")
    p.add_argument("--det-video", type=str, default=None, metavar="PATH",
                   help="Path for detection video (default: outputs/output_detections.mp4)")

    # Streaming preview
    p.add_argument("--preview", action="store_true",
                   help="Open a live blur-only preview window (no bounding boxes)")
    p.add_argument("--preview-fps", type=float, default=None, metavar="FPS",
                   help="Target FPS for the preview window (default: 15; use 30 for smoother)")

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
    if args.vit:
        cfg.USE_VIT = True
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

    # Parallelism / performance
    if args.parallel_yolo:
        cfg.PARALLEL_YOLO = True
    if args.frame_buffer_size:
        cfg.FRAME_BUFFER_SIZE = args.frame_buffer_size
    if args.object_embed_interval is not None:
        cfg.OBJECT_EMBED_INTERVAL = args.object_embed_interval
    if args.clip_interval is not None:
        cfg.CLIP_INTERVAL = args.clip_interval

    # Intermediate outputs
    if args.save_det_video:
        cfg.SAVE_DET_VIDEO = True
    if args.det_video:
        cfg.OUTPUT_DET_VIDEO = args.det_video
        cfg.SAVE_DET_VIDEO = True  # implicitly enable

    # Streaming preview
    if args.preview:
        cfg.PREVIEW = True
    if args.preview_fps is not None:
        cfg.PREVIEW_FPS = args.preview_fps
        cfg.PREVIEW = True  # implicitly enable

    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.OUTPUT_VIDEO) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.OUTPUT_LOG) or ".", exist_ok=True)
    if cfg.ENABLE_SALIENCY_RENDER:
        os.makedirs(os.path.dirname(cfg.OUTPUT_SALIENCY_VIDEO) or ".", exist_ok=True)
    if cfg.SAVE_DET_VIDEO:
        os.makedirs(os.path.dirname(cfg.OUTPUT_DET_VIDEO) or ".", exist_ok=True)

    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
