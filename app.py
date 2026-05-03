"""
PrivacyGuard - Streamlit Application
=====================================
Models are loaded in a background thread so the browser WebSocket stays
alive during the (potentially long) CLIP / YOLO weight download.

Run:
    streamlit run app.py
"""

# stdlib
import io
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

# third-party
import cv2
import numpy as np
import streamlit as st

# project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import _model_cache as _mc   # survives Streamlit reruns (cached by sys.modules)

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PrivacyGuard - Real-time Video Blur",
    page_icon="lock",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
}
section[data-testid="stSidebar"] {
    background: rgba(22,27,34,0.95);
    border-right: 1px solid rgba(48,54,61,0.8);
}
.pg-card {
    background: rgba(22,27,34,0.85);
    border: 1px solid rgba(48,54,61,0.8);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.pg-card h3 { margin-top:0; font-size:1rem; color:#58a6ff; }
.pg-hero { text-align:center; padding:2rem 0 1.2rem; }
.pg-hero h1 {
    font-size:2.6rem; font-weight:800;
    background: linear-gradient(90deg,#58a6ff,#bc8cff,#f78166);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0.3rem;
}
.pg-hero p { color:#8b949e; font-size:1.05rem; margin:0; }
.pg-metric {
    display:inline-block;
    background:rgba(88,166,255,0.12);
    border:1px solid rgba(88,166,255,0.35);
    border-radius:20px; padding:0.25rem 0.75rem;
    font-size:0.85rem; color:#58a6ff; margin:0.2rem 0.15rem;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg,#58a6ff,#bc8cff);
}
img { border-radius:8px; }
div.stButton > button {
    background: linear-gradient(135deg,#238636,#2ea043);
    color:white; border:none; border-radius:8px;
    font-size:1rem; padding:0.55rem 1.5rem; font-weight:600;
}
div.stButton > button:hover { opacity:0.85; }
div.stDownloadButton > button {
    background: linear-gradient(135deg,#1f6feb,#388bfd);
    color:white; border:none; border-radius:8px;
    font-size:0.9rem; padding:0.4rem 1.2rem; font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ─── Global model store (lives in _model_cache, survives reruns) ─────────────


def _do_load_models(use_vit: bool, use_clip: bool, use_saliency: bool) -> None:
    """Runs in a daemon thread; populates _mc.store when done."""
    try:
        from config.config import PipelineConfig
        from models.detector import Detector
        from models.embedding import EmbeddingModule
        from models.context import ContextModule
        from models.saliency import SaliencyModule
        from engine.feature_engine import FeatureEngine
        from engine.decision_engine import DecisionEngine
        from utils.visualization import Renderer

        cfg = PipelineConfig(
            USE_VIT=use_vit, USE_CLIP=use_clip,
            USE_SALIENCY=use_saliency,
            ENABLE_RENDER=True, PREVIEW=False,
        )

        detector = Detector(cfg)
        detector.initialize()

        embedder = None
        if use_vit:
            embedder = EmbeddingModule(cfg)
            embedder.initialize()

        context = None
        if use_clip:
            context = ContextModule(cfg)
            context.initialize()

        saliency = None
        if use_saliency:
            saliency = SaliencyModule(cfg)
            saliency.initialize()

        with _mc.lock:
            _mc.store.update({
                "cfg": cfg,
                "detector": detector,
                "embedder": embedder,
                "context": context,
                "saliency": saliency,
                "feature_engine": FeatureEngine(),
                "decision_engine": DecisionEngine(cfg),
                "renderer": Renderer(cfg),
            })
        _mc.ready.set()

    except Exception:
        _mc.error.append(traceback.format_exc())
        _mc.ready.set()


def _ensure_models_loading(use_vit: bool, use_clip: bool, use_saliency: bool) -> None:
    """Start background load thread exactly once (guarded by _mc.started)."""
    if _mc.started.is_set():
        return
    _mc.started.set()
    t = threading.Thread(
        target=_do_load_models,
        args=(use_vit, use_clip, use_saliency),
        daemon=True, name="ModelLoader",
    )
    t.start()


# ─── Processing job ───────────────────────────────────────────────────────────

class _ProcessingJob:
    PREVIEW_W, PREVIEW_H = 1280, 720

    def __init__(self, input_path, output_path, log_path, models, cfg_overrides):
        self.input_path  = input_path
        self.output_path = output_path
        self.log_path    = log_path
        self.models      = models
        self.cfg_overrides = cfg_overrides
        self.preview_q: queue.Queue = queue.Queue(maxsize=6)
        self.total_frames = 0
        self.done_frames  = 0
        self.done  = False
        self.error = None
        self.live_info: dict = {}   # updated every frame; read by UI thread
        self._thread = threading.Thread(target=self._run, daemon=True, name="PGWorker")

    def start(self):
        self._thread.start()

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.PREVIEW_H, self.PREVIEW_W, 3), dtype=np.uint8)
        canvas[(self.PREVIEW_H-nh)//2:(self.PREVIEW_H-nh)//2+nh,
               (self.PREVIEW_W-nw)//2:(self.PREVIEW_W-nw)//2+nw] = resized
        return canvas

    def _run(self):
        try:
            from config.config import PipelineConfig
            from core.video_processor import VideoReader, VideoWriter
            from engine.temporal import TemporalSmoother
            from utils.io import JSONLLogger

            from engine.decision_engine import DecisionEngine
            from utils.visualization import Renderer

            base_cfg = self.models["cfg"]
            ov = self.cfg_overrides
            cfg = PipelineConfig(
                USE_YOLO=base_cfg.USE_YOLO,
                USE_VIT=ov.get("use_vit", base_cfg.USE_VIT),
                USE_CLIP=ov.get("use_clip", base_cfg.USE_CLIP),
                USE_SALIENCY=ov.get("use_saliency", base_cfg.USE_SALIENCY),
                BLUR_BACKGROUND_PEOPLE=ov.get("blur_people", True),
                BLUR_SCREENS=ov.get("blur_screens", True),
                BLUR_PLATES=ov.get("blur_plates", True),
                BLUR_KERNEL_SIZE=ov.get("blur_strength", 51),
                ENABLE_RENDER=True, SAVE_JSON=True, PREVIEW=False,
            )

            m = self.models
            # create per-job engine + renderer so blur/model settings are honoured
            renderer   = Renderer(cfg)
            detector   = m["detector"]
            embedder   = m.get("embedder") if cfg.USE_VIT else None
            context    = m.get("context")  if cfg.USE_CLIP else None
            saliency   = m.get("saliency") if cfg.USE_SALIENCY else None
            feat_eng   = m["feature_engine"]
            dec_eng    = DecisionEngine(cfg)   # fresh copy with this job's blur flags
            temporal   = TemporalSmoother(buffer_size=cfg.TEMPORAL_BUFFER_SIZE)

            reader = VideoReader(self.input_path)
            self.total_frames = reader.total_frames
            fps = reader.fps

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            writer = VideoWriter(self.output_path, fps, reader.width, reader.height, cfg)
            logger = JSONLLogger(self.log_path)

            _last_scene = _last_sal = None
            frame_id = 0

            while True:
                ok, frame = reader.read()
                if not ok:
                    break

                ts = frame_id / max(fps, 1e-6)
                detections = {"persons": [], "faces": [], "plates": [], "objects": []}
                if detector:
                    detections = detector.detect(frame)

                if embedder and frame_id % (cfg.OBJECT_EMBED_INTERVAL or 1) == 0:
                    detections = embedder.embed_objects(frame, detections)

                frame_emb = None
                if embedder and frame_id % cfg.FRAME_EMBED_INTERVAL == 0:
                    frame_emb = embedder.embed_frame(frame)

                clip_emb = None
                if context and frame_id % cfg.CLIP_INTERVAL == 0:
                    _last_scene = context.classify_scene(frame)
                    if frame_id % cfg.FRAME_EMBED_INTERVAL == 0:
                        clip_emb = context.embed_image(frame)

                if saliency and frame_id % cfg.SALIENCY_INTERVAL == 0:
                    _last_sal = saliency.compute(frame)

                frame_output = feat_eng.process(
                    detections, frame.shape, frame_id,
                    frame_embedding=frame_emb, clip_embedding=clip_emb,
                    context=_last_scene, saliency_map=_last_sal,
                    saliency_threshold=cfg.SALIENCY_THRESHOLD,
                    overlap_threshold=cfg.SALIENCY_OVERLAP_THRESHOLD,
                )
                frame_output["timestamp"] = round(ts, 3)
                frame_output = dec_eng.decide(frame_output)
                frame_output = temporal.update(frame_output)

                annotated = renderer.render(frame, frame_output)
                writer.write(annotated)

                # ── populate live context info for UI panel ────────────────
                dets = frame_output.get("detections", [])
                blur_labels = [d["label"] for d in dets if d.get("role") == "blur"]
                self.live_info = {
                    "scene":         _last_scene or {},
                    "blur_counts":   dict(Counter(blur_labels)),
                    "total_blurred": len(blur_labels),
                    "total_dets":    len(dets),
                    "strategy":      next(
                        (d.get("decision_trace", {}).get("strategy", "—")
                         for d in dets if d.get("decision_trace")),
                        "—"
                    ),
                }

                preview = self._letterbox(renderer.render_blur_only(frame, frame_output))
                try:
                    if self.preview_q.full():
                        self.preview_q.get_nowait()
                    self.preview_q.put_nowait(preview)
                except queue.Full:
                    pass

                logger.log(frame_output)
                frame_id += 1
                self.done_frames = frame_id

            reader.release()
            writer.release()

        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.done = True
            try:
                self.preview_q.put_nowait(None)
            except queue.Full:
                pass


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### PrivacyGuard")
    st.markdown("---")

    st.markdown("#### Model Settings")
    use_clip     = st.toggle("CLIP Scene Context",     value=True)
    use_vit      = st.toggle("ViT Object Embeddings",  value=False,
                              help="Disable for ~7x speedup")
    use_saliency = st.toggle("Saliency Analysis",      value=False)

    st.markdown("---")
    st.markdown("#### Blur Settings")
    blur_people  = st.toggle("Blur People",               value=True)
    blur_screens = st.toggle("Blur Screens & Devices",    value=True)
    blur_plates  = st.toggle("Blur License Plates",       value=True)
    blur_strength = st.select_slider(
        "Blur Strength", options=[21, 31, 41, 51, 61, 75, 91], value=51,
    )

    st.markdown("---")
    st.markdown("#### Output Directory")
    output_dir = st.text_input("Output folder path", value=str(ROOT / "outputs"),
                                label_visibility="collapsed")
    if output_dir and not Path(output_dir).exists():
        st.caption("Directory will be created")

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>YOLOv8 · CLIP · U2Net<br>CVDL Project 2026</small>",
        unsafe_allow_html=True,
    )

# ─── Start background model loading (non-blocking) ────────────────────────────
_ensure_models_loading(use_vit, use_clip, use_saliency)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-hero">
    <h1>PrivacyGuard</h1>
    <p>Real-time AI-powered video anonymisation &amp; privacy blurring</p>
</div>
""", unsafe_allow_html=True)

# ─── Model loading status ─────────────────────────────────────────────────────
if _mc.error:
    st.error("Model loading failed")
    st.code(_mc.error[0], language="python")
    st.stop()

if not _mc.ready.is_set():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="pg-card" style="text-align:center;padding:2.5rem">
            <h3 style="font-size:1.4rem;color:#58a6ff">Loading ML Models</h3>
            <p style="color:#8b949e;margin-bottom:1rem">
            YOLOv8 detectors, CLIP scene encoder and supporting modules
            are being loaded onto your GPU. This happens only once per session.</p>
        </div>
        """, unsafe_allow_html=True)
        for s in ["YOLOv8 Object Detector", "YOLOv8 Face Detector",
                  "License Plate Detector", "CLIP Scene Encoder",
                  "Feature & Decision Engines"]:
            st.markdown(f'<span class="pg-metric">&#8987; {s}</span>', unsafe_allow_html=True)
        # Poll every 1.5 s; Streamlit keeps WebSocket alive between reruns
        time.sleep(1.5)
        st.rerun()
    st.stop()

# ─── Active model chips ───────────────────────────────────────────────────────
active = ["YOLOv8 Objects+Faces+Plates"]
if use_clip:     active.append("CLIP Scene Context")
if use_vit:      active.append("ViT Embeddings")
if use_saliency: active.append("Saliency")

chips = "".join(f'<span class="pg-metric">OK {m}</span>' for m in active)
st.markdown(f"<div style='text-align:center;margin-bottom:1rem'>{chips}</div>",
            unsafe_allow_html=True)
st.markdown("---")

models = _mc.store  # safe to read after _mc.ready is set

# ─── Upload section ───────────────────────────────────────────────────────────
col_up, col_info = st.columns([2, 1], gap="large")

tmp_input_path = None

with col_up:
    st.markdown('<div class="pg-card"><h3>Upload Video</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        from core.video_processor import VideoReader
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_input_path = tmp.name
        try:
            r = VideoReader(tmp_input_path)
            dur = r.total_frames / max(r.fps, 1e-6)
            W, H, FPS, TF = r.width, r.height, r.fps, r.total_frames
            r.release()
            st.markdown(f"""
            <div class="pg-card">
            <h3>Video Info</h3>
            <span class="pg-metric">{uploaded.name}</span>
            <span class="pg-metric">{W}x{H}</span>
            <span class="pg-metric">{FPS:.1f} fps</span>
            <span class="pg-metric">{_fmt_time(dur)}</span>
            <span class="pg-metric">{TF} frames</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.warning("Could not read video metadata.")

with col_info:
    st.markdown("""
    <div class="pg-card">
    <h3>What gets blurred?</h3>
    <ul style="color:#c9d1d9;font-size:0.9rem;padding-left:1.2rem">
        <li>People in the background</li>
        <li>Faces (all persons)</li>
        <li>License plates</li>
        <li>Screens &amp; devices</li>
    </ul>
    </div>
    <div class="pg-card">
    <h3>Speed Tips</h3>
    <ul style="color:#c9d1d9;font-size:0.9rem;padding-left:1.2rem">
        <li>Disable ViT for ~7x speedup</li>
        <li>Lower blur strength = faster</li>
        <li>CLIP samples every 15 frames</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ─── Run button ───────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("Start Processing", disabled=(uploaded is None or tmp_input_path is None))

# ─── Processing + live preview ────────────────────────────────────────────────
if run_clicked and tmp_input_path is not None:
    stem = Path(uploaded.name).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{stem}_blurred.mp4")
    log_path    = str(out_dir / f"{stem}_log.jsonl")

    cfg_ov = dict(
        blur_people=blur_people, blur_screens=blur_screens,
        blur_plates=blur_plates, blur_strength=blur_strength,
        use_vit=use_vit, use_clip=use_clip, use_saliency=use_saliency,
    )

    job = _ProcessingJob(tmp_input_path, output_path, log_path, models, cfg_ov)
    job.start()

    st.markdown("---")
    st.markdown("### Live Preview  <small style='color:#8b949e'>(blur only · no bounding boxes)</small>",
                unsafe_allow_html=True)

    prev_col, stat_col = st.columns([3, 1], gap="large")
    with prev_col:
        preview_ph = st.empty()
    with stat_col:
        stat_ph = st.empty()

    prog = st.progress(0.0, text="Starting ...")
    t0   = time.perf_counter()

    while True:
        try:
            frame_bgr = job.preview_q.get(timeout=2.0)
        except queue.Empty:
            if job.done:
                break
            continue
        if frame_bgr is None:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preview_ph.image(frame_rgb, use_container_width=True)

        total   = max(job.total_frames, 1)
        done    = job.done_frames
        elapsed = time.perf_counter() - t0
        fps_l   = done / max(elapsed, 1e-6)
        eta     = (total - done) / max(fps_l, 1e-6)

        prog.progress(min(done / total, 1.0), text=f"Processing ... {done}/{total} frames")

        # ── live info panel ─────────────────────────────────────────────
        info     = job.live_info
        scene    = info.get("scene", {})
        sc_label = scene.get("label", "detecting\u2026")
        sc_conf  = scene.get("confidence", 0.0)
        strategy = info.get("strategy", "\u2014")
        blur_cnt = info.get("blur_counts", {})
        total_b  = info.get("total_blurred", 0)
        total_d  = info.get("total_dets", 0)

        blur_rows = "".join(
            f'<span class="pg-metric" style="margin:0.15rem">'
            f'{lbl.replace("_"," ").title()} &times;{cnt}</span>'
            for lbl, cnt in sorted(blur_cnt.items(), key=lambda x: -x[1])
        ) or '<span style="color:#8b949e;font-size:0.85rem">none detected yet</span>'

        stat_ph.markdown(f"""
        <div class="pg-card" style="margin-top:0">
          <h3>&#128202; Live Stats</h3>
          <span class="pg-metric">{done}/{total}</span>
          <span class="pg-metric">{fps_l:.1f}&nbsp;fps</span>
          <span class="pg-metric">ETA&nbsp;{_fmt_time(eta)}</span>
          <span class="pg-metric">&#9201;&nbsp;{_fmt_time(elapsed)}</span>
        </div>
        <div class="pg-card">
          <h3>&#127916; CLIP Scene Context</h3>
          <div style="font-size:1.1rem;color:#c9d1d9;margin-bottom:0.5rem">
            <b style="color:#58a6ff">{sc_label.title()}</b>
            &nbsp;<span style="color:#8b949e;font-size:0.85rem">{sc_conf*100:.0f}% confidence</span>
          </div>
          <span class="pg-metric" style="font-size:0.78rem">Strategy:&nbsp;{strategy.replace("Strategy","").strip() or strategy}</span>
        </div>
        <div class="pg-card">
          <h3>&#128683; Auto-Blurred Objects&nbsp;<span style="color:#f85149">{total_b}</span></h3>
          <div style="line-height:2.2">{blur_rows}</div>
          <div style="margin-top:0.5rem;color:#8b949e;font-size:0.78rem">
            {total_d} detections &middot; {total_b} blurred
          </div>
        </div>
        """, unsafe_allow_html=True)

    while not job.done:
        time.sleep(0.1)

    prog.progress(1.0, text="Done!")

    if job.error:
        st.error("Processing failed")
        st.code(job.error, language="python")
    else:
        elapsed_t = time.perf_counter() - t0
        st.success(
            f"Processed **{job.done_frames} frames** in {_fmt_time(elapsed_t)} "
            f"({job.done_frames / max(elapsed_t, 1e-6):.1f} fps avg)"
        )

        st.markdown("### Output Files")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="pg-card"><h3>Blurred Video</h3>', unsafe_allow_html=True)
            if Path(output_path).exists():
                with open(output_path, "rb") as f:
                    st.download_button("Download MP4", data=f.read(),
                                       file_name=Path(output_path).name, mime="video/mp4")
                st.caption(f"`{output_path}`  ({Path(output_path).stat().st_size/1e6:.1f} MB)")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="pg-card"><h3>Detection Log</h3>', unsafe_allow_html=True)
            if Path(log_path).exists():
                with open(log_path, "rb") as f:
                    st.download_button("Download JSONL", data=f.read(),
                                       file_name=Path(log_path).name,
                                       mime="application/jsonl")
                st.caption(f"`{log_path}`")
            st.markdown("</div>", unsafe_allow_html=True)
