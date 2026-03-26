"""
Central pipeline – orchestrates all modules in a config-driven,
streaming-safe, memory-efficient loop.
"""

from config.config import PipelineConfig
from core.video_processor import VideoReader, VideoWriter
from models.detector import Detector
from models.embedding import EmbeddingModule
from models.context import ContextModule
from models.saliency import SaliencyModule
from engine.feature_engine import FeatureEngine
from engine.decision_engine import DecisionEngine
from engine.temporal import TemporalSmoother
from utils.io import JSONLLogger
from utils.visualization import Renderer


class Pipeline:
    """
    End-to-end video processing pipeline.

    Flow per frame:
        Frame → Detect → Embed → Context → Feature → Decision → Temporal → Render → Log
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config

        # Modules (created lazily)
        self.detector: Detector | None = None
        self.embedder: EmbeddingModule | None = None
        self.context: ContextModule | None = None
        self.saliency: SaliencyModule | None = None

        # Engines (stateless / lightweight – always created)
        self.feature_engine = FeatureEngine()
        self.decision_engine = DecisionEngine(config)
        self.temporal = TemporalSmoother(buffer_size=config.TEMPORAL_BUFFER_SIZE)

        # Renderer (needed for main video or saliency heatmap)
        self.renderer = Renderer(config) if (config.ENABLE_RENDER or config.ENABLE_SALIENCY_RENDER) else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _init_modules(self) -> None:
        """Load only the models that are enabled."""
        if self.cfg.USE_YOLO:
            self.detector = Detector(self.cfg)
            self.detector.initialize()
            print("[Pipeline] YOLO detector loaded.")

        if self.cfg.USE_VIT:
            self.embedder = EmbeddingModule(self.cfg)
            self.embedder.initialize()
            print("[Pipeline] ViT embedder loaded.")

        if self.cfg.USE_CLIP:
            self.context = ContextModule(self.cfg)
            self.context.initialize()
            print("[Pipeline] CLIP context module loaded.")

        if self.cfg.USE_SALIENCY:
            self.saliency = SaliencyModule(self.cfg)
            self.saliency.initialize()
            print("[Pipeline] Saliency module loaded.")

    def _release_modules(self) -> None:
        for mod in (self.detector, self.embedder, self.context, self.saliency):
            if mod is not None:
                mod.release()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, input_path: str | None = None, output_path: str | None = None) -> None:
        input_path = input_path or self.cfg.INPUT_VIDEO
        output_path = output_path or self.cfg.OUTPUT_VIDEO

        # --- Open video ---
        reader = VideoReader(input_path)
        print(f"[Pipeline] Opened {input_path}  "
              f"({reader.width}x{reader.height} @ {reader.fps:.1f} fps, "
              f"{reader.total_frames} frames)")

        # --- Initialise models ---
        self._init_modules()

        # --- Writer / Logger ---
        writer = None
        if self.cfg.ENABLE_RENDER:
            writer = VideoWriter(output_path, reader.fps,
                                 reader.width, reader.height, self.cfg)

        sal_writer = None
        if self.cfg.ENABLE_SALIENCY_RENDER and self.cfg.USE_SALIENCY:
            sal_writer = VideoWriter(self.cfg.OUTPUT_SALIENCY_VIDEO, reader.fps,
                                     reader.width, reader.height, self.cfg)

        logger = None
        if self.cfg.SAVE_JSON:
            logger = JSONLLogger(self.cfg.OUTPUT_LOG)

        frame_id = 0
        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                if self.cfg.MAX_FRAMES and frame_id >= self.cfg.MAX_FRAMES:
                    break

                timestamp = frame_id / reader.fps

                # 1. Detection
                detections = {"persons": [], "faces": [], "plates": [], "objects": []}
                if self.detector:
                    detections = self.detector.detect(frame)

                # 2. Object-level embeddings (ViT)
                if self.embedder:
                    detections = self.embedder.embed_objects(frame, detections)

                # 3. Frame-level embedding (ViT, at interval)
                frame_emb = None
                if self.embedder and frame_id % self.cfg.FRAME_EMBED_INTERVAL == 0:
                    frame_emb = self.embedder.embed_frame(frame)

                # 4. Scene context (CLIP)
                clip_emb = None
                scene_ctx = None
                if self.context:
                    scene_ctx = self.context.classify_scene(frame)
                    if frame_id % self.cfg.FRAME_EMBED_INTERVAL == 0:
                        clip_emb = self.context.embed_image(frame)

                # 5. Saliency (optional, reuse last map between intervals)
                if self.saliency and frame_id % self.cfg.SALIENCY_INTERVAL == 0:
                    self._last_sal_map = self.saliency.compute(frame)
                sal_map = getattr(self, "_last_sal_map", None)

                # 6. Feature aggregation
                frame_output = self.feature_engine.process(
                    detections, frame.shape, frame_id,
                    frame_embedding=frame_emb,
                    clip_embedding=clip_emb,
                    context=scene_ctx,
                    saliency_map=sal_map,
                    saliency_threshold=self.cfg.SALIENCY_THRESHOLD,
                    overlap_threshold=self.cfg.SALIENCY_OVERLAP_THRESHOLD,
                )
                frame_output["timestamp"] = round(timestamp, 3)

                # 7. Decision engine
                frame_output = self.decision_engine.decide(frame_output)

                # 8. Temporal smoothing
                frame_output = self.temporal.update(frame_output)

                # 9. Render annotated frame
                if self.renderer and writer:
                    annotated = self.renderer.render(frame, frame_output)
                    writer.write(annotated)

                # 9b. Pure saliency heatmap video
                if self.renderer and sal_writer:
                    sal_frame = self.renderer.render_saliency_heatmap(
                        frame, sal_map,
                    )
                    sal_writer.write(sal_frame)

                # 10. Stream-write log (JSONL)
                if logger:
                    logger.log(frame_output)

                frame_id += 1
                if frame_id % 25 == 0:
                    print(f"  Processed {frame_id} frames …")

        finally:
            # Guaranteed cleanup
            reader.release()
            if writer:
                writer.release()
            if sal_writer:
                sal_writer.release()
            if logger:
                logger.close()
            self._release_modules()

        print(f"[Pipeline] Done – {frame_id} frames processed.")
        if self.cfg.SAVE_JSON:
            print(f"  Log  → {self.cfg.OUTPUT_LOG}")
        if self.cfg.ENABLE_RENDER:
            print(f"  Video → {output_path}")
        if sal_writer:
            print(f"  Saliency → {self.cfg.OUTPUT_SALIENCY_VIDEO}")
