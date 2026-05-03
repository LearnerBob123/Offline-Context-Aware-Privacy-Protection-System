"""
Central configuration for the video processing pipeline.
All flags, paths, and hyperparameters are defined here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    # --- Model toggles ---
    USE_YOLO: bool = True
    USE_VIT: bool = False    # ViT embeddings are stored in the log but NOT used for blur decisions;
                             # disable by default for 7× speedup. Enable with --vit if needed.
    USE_CLIP: bool = True
    USE_SALIENCY: bool = False

    # --- Model weights ---
    YOLO_OBJ_WEIGHTS: str = "yolov8n.pt"
    YOLO_FACE_WEIGHTS: str = "yolov8n-face.pt"
    YOLO_PLATE_WEIGHTS: str = "license-plate-finetune-v1n.pt"

    # --- Detection thresholds ---
    OBJ_CONF: float = 0.3
    FACE_CONF: float = 0.4
    PLATE_CONF: float = 0.4
    IMG_SIZE: int = 640

    # --- Embedding settings ---
    FRAME_EMBED_INTERVAL: int = 10   # how often to compute full-frame ViT embedding
    OBJECT_EMBED_INTERVAL: int = 10  # how often to compute per-object ViT embeddings (0 = every frame)
    CLIP_INTERVAL: int = 15          # how often to run CLIP classify_scene (scene changes slowly)
    VIT_MODEL_NAME: str = "vit_base_patch16_224"
    CLIP_MODEL_NAME: str = "ViT-B/32"

    # --- CLIP context prompts ---
    CONTEXT_PROMPTS: List[str] = field(default_factory=lambda: [
        "a meeting room",
        "a presentation",
        "a home environment",
        "a public outdoor area",
    ])

    # --- Temporal smoothing ---
    TEMPORAL_BUFFER_SIZE: int = 5

    # --- Decision engine ---
    BLUR_BACKGROUND_PEOPLE: bool = True
    BLUR_SCREENS: bool = True
    BLUR_PLATES: bool = True
    BLUR_KERNEL_SIZE: int = 51

    # --- Saliency ---
    SALIENCY_INTERVAL: int = 5
    SALIENCY_BACKEND: str = "spectral"   # "spectral" or "u2net"
    U2NET_REPO_PATH: str = "U-2-Net"     # path to cloned U-2-Net repo
    SALIENCY_THRESHOLD: float = 0.5      # pixel-level threshold (on 0-1 map)
    SALIENCY_OVERLAP_THRESHOLD: float = 0.3  # fraction of bbox pixels above threshold
    SALIENCY_WEIGHT: float = 0.5         # soft weight added to importance score

    # --- Saliency decision tuning ---
    SAL_GAIN: float = 0.5               # multiplier applied when boosting salient objects
    SAL_PENALTY: float = 0.3            # multiplier applied when penalising low-saliency objects
    SAL_OVERLAP_THRESHOLD: float = 0.3  # min overlap to count as "salient" in decision
    MIN_SAL_OVERLAP: float = 0.1        # below this, object gets penalty

    # --- Rendering ---
    ENABLE_RENDER: bool = True
    ENABLE_SALIENCY_RENDER: bool = False  # separate saliency heatmap video
    SAL_RENDER_MODE: str = "overlay"     # "overlay" (blended on frame) or "mask" (pure heatmap)

    # --- Logging ---
    SAVE_JSON: bool = True

    # --- I/O ---
    INPUT_VIDEO: str = "video_d.mp4"
    OUTPUT_VIDEO: str = "outputs/output.mp4"
    OUTPUT_SALIENCY_VIDEO: str = "outputs/saliency_output.mp4"
    OUTPUT_LOG: str = "outputs/output.jsonl"
    MAX_FRAMES: int = 0  # 0 = process all

    # --- Video writer ---
    FALLBACK_FPS: float = 25.0
    VIDEO_CODEC: str = "mp4v"

    # --- Parallelism ---
    PARALLEL_YOLO: bool = False      # run 3 YOLO models concurrently (ThreadPoolExecutor)
    FRAME_BUFFER_SIZE: int = 4       # depth of raw-read and display queues

    # --- Intermediate outputs ---
    SAVE_DET_VIDEO: bool = False                             # save bbox-only detection video
    OUTPUT_DET_VIDEO: str = "outputs/output_detections.mp4" # path for detection video

    # --- Streaming preview ---
    PREVIEW: bool = False            # open cv2.imshow window (paced at native FPS)
    PREVIEW_FPS: float = 15.0        # target FPS for live preview window (15 or 30)
