# PrivacyGuard — Full System Documentation

> Context-aware, AI-driven video privacy blurring pipeline with a real-time Streamlit UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Flow](#3-data-flow)
4. [Module Reference](#4-module-reference)
   - [Config](#41-config--configconfigpy)
   - [Detector](#42-detector--modelsdetectorpy)
   - [Embedding (ViT)](#43-embedding-vit--modelsembeddingpy)
   - [Context (CLIP)](#44-context-clip--modelscontextpy)
   - [Feature Engine](#45-feature-engine--enginefeature_enginepy)
   - [Decision Engine](#46-decision-engine--enginedecision_enginepy)
   - [Strategies](#47-strategies--enginestrategiespy)
   - [Temporal Smoother](#48-temporal-smoother--enginetemporalpy)
   - [Renderer](#49-renderer--utilsvisualizationpy)
   - [Video I/O](#410-video-io--corevideo_processorpy)
   - [JSONL Logger](#411-jsonl-logger--utilsiopy)
5. [Context-Aware Blurring — How It Works](#5-context-aware-blurring--how-it-works)
6. [Streamlit Application](#6-streamlit-application)
   - [UI Layout](#61-ui-layout)
   - [Model Loading Strategy](#62-model-loading-strategy)
   - [Live Preview & Stats Panel](#63-live-preview--stats-panel)
   - [Blur & Model Toggles](#64-blur--model-toggles)
7. [Per-Frame Output Schema](#7-per-frame-output-schema)
8. [Configuration Reference](#8-configuration-reference)
9. [CLI Usage (main.py)](#9-cli-usage-mainpy)
10. [Project File Structure](#10-project-file-structure)
11. [Dependencies & Environment](#11-dependencies--environment)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. Project Overview

PrivacyGuard is a **modular computer vision pipeline** that automatically blurs privacy-sensitive content in videos — people, faces, license plates, and screens — adapting its blurring behaviour to the **scene context** detected by CLIP.

It can be run as:
- A **command-line tool** (`main.py`) for batch processing
- A **real-time Streamlit web app** (`app.py`) with live blur preview

### Key capabilities

| Capability | Technology |
|---|---|
| Object detection (people, vehicles, objects) | YOLOv8n |
| Face detection | YOLOv8n-face |
| License plate detection | YOLOv8n fine-tuned on plates |
| Scene / context classification | CLIP ViT-B/32 (HuggingFace) |
| Object visual embeddings | ViT (timm) |
| Saliency estimation | Spectral saliency / U2-Net |
| Role assignment & blur decisions | Context-aware rule engine |
| Temporal stability | IoU-based sliding-window smoothing |
| Annotated video output | OpenCV VideoWriter |
| Structured logging | JSONL (JSON Lines) |
| Web UI | Streamlit |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Video Frame                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────▼───────────────┐
              │         Detector               │
              │  YOLOv8 × 3 models in parallel │
              │  Objects · Faces · Plates       │
              └────────────────┬───────────────┘
                               │
     ┌─────────────────────────▼──────────────────────────┐
     │                 Embedding Module (ViT)              │
     │  Per-object crops → 768-dim embedding vectors      │
     │  (optional, every N frames, toggled in config)     │
     └─────────────────────────┬──────────────────────────┘
                               │
     ┌─────────────────────────▼──────────────────────────┐
     │              Context Module (CLIP)                  │
     │  Full frame → zero-shot scene label + confidence   │
     │  e.g. "meeting room (0.87)" — runs every 15 frames │
     └─────────────────────────┬──────────────────────────┘
                               │
     ┌─────────────────────────▼──────────────────────────┐
     │               Feature Engine                        │
     │  Merges all detections into flat enriched list:    │
     │  bbox · area · center · relative_size · saliency   │
     │  is_salient · saliency_score · embeddings · context│
     └─────────────────────────┬──────────────────────────┘
                               │
     ┌─────────────────────────▼──────────────────────────┐
     │              Decision Engine                        │
     │  1. Select strategy from CLIP label                │
     │  2. Run strategy → importance_score + role         │
     │  3. Apply shared post-rules (plates · screens ·    │
     │     faces)                                         │
     │  4. Tag each detection with decision_trace         │
     └─────────────────────────┬──────────────────────────┘
                               │
     ┌─────────────────────────▼──────────────────────────┐
     │             Temporal Smoother                       │
     │  Sliding window (N=5) + IoU cross-frame tracking   │
     │  Majority-vote role assignment → no flickering     │
     └──────────┬───────────────────────────┬─────────────┘
                │                           │
   ┌────────────▼──────────┐   ┌────────────▼──────────┐
   │      Renderer         │   │     JSONL Logger       │
   │  Blur regions applied │   │  One JSON line / frame │
   │  Boxes + HUD (debug)  │   │  Flushed immediately   │
   │  MP4 video output     │   │  Crash-safe            │
   └───────────────────────┘   └───────────────────────┘
```

---

## 3. Data Flow

Every frame travels through this chain and produces a `frame_output` dictionary:

```
read frame
    → detect()          → {"persons": [...], "faces": [...], "plates": [...], "objects": [...]}
    → embed_objects()   → adds "embedding" key to each detection (optional)
    → classify_scene()  → {"label": "meeting room", "confidence": 0.87}
    → feature_engine.process()
                        → flat enriched list + spatial metadata
    → decision_engine.decide()
                        → each detection gets role: "main" | "blur" | "background"
                           + decision_trace (fully explainable)
    → temporal.update() → role smoothed across last 5 frames (no flicker)
    → renderer.render() → blurred + annotated MP4 frame written
    → logger.log()      → one JSONL line written + flushed
```

---

## 4. Module Reference

### 4.1 Config — `config/config.py`

Single `@dataclass` — `PipelineConfig` — holding all tuneable parameters. Every module receives an instance of this at construction; no module hard-codes any value.

**Key flag groups:**

| Group | Flags |
|---|---|
| Model toggles | `USE_YOLO`, `USE_VIT`, `USE_CLIP`, `USE_SALIENCY` |
| YOLO weights | `YOLO_OBJ_WEIGHTS`, `YOLO_FACE_WEIGHTS`, `YOLO_PLATE_WEIGHTS` |
| Detection thresholds | `OBJ_CONF` (0.3), `FACE_CONF` (0.4), `PLATE_CONF` (0.4) |
| Sampling intervals | `CLIP_INTERVAL` (15), `FRAME_EMBED_INTERVAL` (10), `OBJECT_EMBED_INTERVAL` (10), `SALIENCY_INTERVAL` (5) |
| Blur decisions | `BLUR_BACKGROUND_PEOPLE`, `BLUR_SCREENS`, `BLUR_PLATES`, `BLUR_KERNEL_SIZE` (51) |
| Saliency tuning | `SAL_GAIN`, `SAL_PENALTY`, `SAL_OVERLAP_THRESHOLD`, `MIN_SAL_OVERLAP`, `SALIENCY_THRESHOLD` |
| Rendering | `ENABLE_RENDER`, `ENABLE_SALIENCY_RENDER` |
| Logging | `SAVE_JSON` |

---

### 4.2 Detector — `models/detector.py`

Runs **three YOLOv8 models** and returns a unified detection dict.

```python
{
  "persons":  [{"bbox": [x1,y1,x2,y2], "confidence": 0.92, "label": "person"}, ...],
  "faces":    [...],
  "plates":   [...],
  "objects":  [...]   # laptops, TVs, cars, etc.
}
```

- All three models share the same `imgsz` / conf thresholds from config.
- The plate model (`license-plate-finetune-v1n.pt`) gracefully degrades to `None` if the weights file is missing.
- Optional `ThreadPoolExecutor` runs all three inferences **in parallel** when `PARALLEL_YOLO=True`.

---

### 4.3 Embedding (ViT) — `models/embedding.py`

Uses a **timm ViT** (`vit_base_patch16_224`) to produce 768-dimensional embedding vectors.

Two modes:
- `embed_frame(frame)` — embed the full frame (used for CLIP-style similarity)
- `embed_objects(frame, detections)` — crop each detected bbox, batch-infer, attach `embedding` list to each detection dict

Runs every `OBJECT_EMBED_INTERVAL` frames (default: 10) to amortise cost. Disabled by default (`USE_VIT=False`) for ~7× speedup.

---

### 4.4 Context (CLIP) — `models/context.py`

Zero-shot scene classification using **OpenAI CLIP ViT-B/32** loaded through HuggingFace `transformers`.

**How it works:**
1. At `initialize()`, text prompts (e.g. `"a meeting room"`, `"a public outdoor area"`) are encoded once and stored as normalised text feature vectors.
2. On each call, the frame is encoded with the vision model.
3. Cosine similarity between the image embedding and each text embedding is computed → softmax → top label + confidence.

```python
classify_scene(frame) → {"label": "meeting room", "confidence": 0.87}
```

Runs every `CLIP_INTERVAL` frames (default: 15). The last result is **cached** and reused between calls, so there is no stale-data gap in the decision engine.

**Supported CLIP model aliases:**

| Alias | HuggingFace ID |
|---|---|
| `ViT-B/32` | `openai/clip-vit-base-patch32` |
| `ViT-B/16` | `openai/clip-vit-base-patch16` |
| `ViT-L/14` | `openai/clip-vit-large-patch14` |

---

### 4.5 Feature Engine — `engine/feature_engine.py`

Flattens and enriches all detection categories into a single list. For each detection it computes:

| Field | Description |
|---|---|
| `id` | Sequential integer ID within frame |
| `bbox` | `[x1, y1, x2, y2]` |
| `label` | Category string |
| `confidence` | YOLO confidence |
| `area` | Pixel area of bbox |
| `center` | `[cx, cy]` |
| `relative_size` | `area / frame_area` |
| `embedding` | ViT vector (if present) |
| `saliency_overlap` | Fraction of bbox pixels above saliency threshold |
| `saliency_score` | Mean saliency value inside bbox |
| `is_salient` | `saliency_overlap > overlap_threshold` |
| `role` | Default: `"background"` (overwritten by Decision Engine) |

The engine also attaches `context`, `frame_embedding`, and `clip_embedding` to the top-level `frame_output` dict.

---

### 4.6 Decision Engine — `engine/decision_engine.py`

The **routing hub**. It:

1. Reads the CLIP scene label from `frame_output["context"]["label"]`
2. Calls `get_strategy(label)` → selects the appropriate `BaseDecisionStrategy`
3. Calls `strategy.decide(dets, frame_output, config)` → sets `importance_score`, `role`, `decision_trace` on every detection
4. Applies **shared post-rules** that override the strategy output:
   - **Screen rules** — blur laptops/TVs/monitors/phones if `BLUR_SCREENS=True` (MeetingStrategy already handles this)
   - **Plate rules** — blur `license_plate` if `BLUR_PLATES=True`
   - **Face rules** — face inside main person's bbox → `"main"`; otherwise → `"blur"` or `"background"`
5. Tags every `decision_trace["strategy"]` with the strategy class name

Every detection ends up with one of three roles:

| Role | Meaning |
|---|---|
| `"main"` | Subject of interest — never blurred, drawn with full box |
| `"blur"` | Privacy-sensitive — Gaussian blur applied to bbox region |
| `"background"` | Ignored — no blur, no box in output |

---

### 4.7 Strategies — `engine/strategies.py`

Context-specific blurring logic. All extend `BaseDecisionStrategy`.

#### Strategy selection

```
CLIP label contains "meeting" or "presentation"  →  MeetingStrategy
CLIP label contains "outdoor", "street", "park"  →  OutdoorStrategy
anything else                                     →  DefaultStrategy
```

#### Strategy summaries

| Strategy | Main subject | Background people | Screens | Notes |
|---|---|---|---|---|
| **DefaultStrategy** | Most salient+central person | Blur if `BLUR_BACKGROUND_PEOPLE` | Blur if `BLUR_SCREENS` | General fallback |
| **MeetingStrategy** | Most salient+central person | Blur | Always blur | Adds centrality bonus to importance score |
| **OutdoorStrategy** | Largest salient person/face | Blur | Blur if configured | Area dominates (no distance weight); higher saliency weight |
| **MedicalStrategy** | _(none elected)_ | All blurred | All blurred | For hospital/clinical scenes |
| **ClassroomStrategy** | Front-most person (presenter) | Blur as students | Kept visible (educational content) | Y-position front bonus |
| **TrafficStrategy** | Vehicles | Pedestrians blurred | Blur if configured | Vehicles get `"main"`, pedestrians get `"blur"` |
| **RetailStrategy** | Largest/central person (staff) | Customers blurred | POS screens blurred | Staff heuristic by importance score |

> Note: `MedicalStrategy`, `ClassroomStrategy`, `TrafficStrategy`, and `RetailStrategy` are implemented but not yet wired into `get_strategy()`. They can be activated by extending the `get_strategy()` registry.

#### Importance score formula

Each strategy computes an importance score used to elect the "main" subject:

```
score = base_importance + saliency_boost + saliency_penalty [+ strategy_bonus]

base_importance = area / (euclidean_distance_from_centre + 1)
saliency_boost  = SAL_GAIN × saliency_score × area   (if is_salient)
saliency_penalty = −SAL_PENALTY × area               (if saliency_overlap < MIN_SAL_OVERLAP)

MeetingStrategy adds:  centrality_bonus = (1 − dist_norm) × area × 0.3
ClassroomStrategy adds: front_bonus = (bbox_y / H) × area × 0.2
```

#### Decision trace (explainability)

Every detection carries a `decision_trace` dict:

```json
{
  "base_score": 48231.5,
  "saliency_boost": 1204.3,
  "penalty": 0.0,
  "final_score": 49435.8,
  "strategy": "MeetingStrategy",
  "rules_applied": [
    "meeting_centrality_bonus",
    "boosted_due_to_high_saliency",
    "selected_as_main_person"
  ]
}
```

---

### 4.8 Temporal Smoother — `engine/temporal.py`

Prevents **role flickering** across frames by maintaining a sliding window of the last `N` frame outputs (default `N=5`).

For each detection in the current frame:
1. Search all previous frames in the window for a detection of the same label with `IoU > 0.4` (IoU-based identity matching)
2. Collect historical roles
3. Apply **majority vote** — current role is overridden only if the majority disagrees

This ensures that a person classified as `"main"` doesn't flicker to `"blur"` for a single frame due to a noisy YOLO detection.

---

### 4.9 Renderer — `utils/visualization.py`

#### `render(frame, frame_output)` — full annotated output

1. Applies Gaussian blur (`BLUR_KERNEL_SIZE × BLUR_KERNEL_SIZE`) to all `role == "blur"` bboxes
2. Draws bounding boxes with colour-coded labels:
   - Green = person, Blue = screen/laptop, Cyan = face, Orange = license plate
   - Saliency-aware colouring when saliency data is present
3. Renders HUD overlay (frame ID, object count, CLIP scene label, strategy name)

#### `render_blur_only(frame, frame_output)` — clean preview

Applies only blur regions — **no boxes, no labels, no HUD**. Used for:
- The live Streamlit preview (privacy-safe, uncluttered)
- Blur-only output video

---

### 4.10 Video I/O — `core/video_processor.py`

**`VideoReader`** wraps `cv2.VideoCapture`:
- Exposes `total_frames`, `fps`, `width`, `height`
- `fps` defaults to 25 if source reports ≤ 0 (webcam edge case)

**`VideoWriter`** wraps `cv2.VideoWriter`:
- Codec: `mp4v`
- Enforces `uint8` dtype on every frame write
- Released in `finally` block to prevent file corruption on crash

---

### 4.11 JSONL Logger — `utils/io.py`

Writes one JSON object per line, flushed immediately after each frame:

```jsonl
{"frame_id": 0, "timestamp": 0.0, "detections": [...], "context": {...}}
{"frame_id": 1, "timestamp": 0.033, "detections": [...]}
```

**Advantages over a single JSON dump:**
- Zero memory accumulation — only current frame lives in RAM
- Crash-safe — all written lines are valid even if the process dies
- Streamable — processable with `jq`, `pandas.read_json(lines=True)` without loading the full file

---

## 5. Context-Aware Blurring — How It Works

This is the core innovation of the system. Instead of applying the same blur rules to every video, the pipeline **adapts behaviour based on what CLIP understands about the scene**.

### Step-by-step for a single frame

```
Frame 60 arrives
  │
  ├── CLIP (every 15 frames): "meeting room" (confidence: 0.87)
  │
  ├── DecisionEngine selects: MeetingStrategy
  │
  ├── MeetingStrategy scores all detections:
  │     person A (centre, large, salient) → score: 52000  → role: "main"
  │     person B (edge, small)           → score: 8400   → role: "blur"
  │     laptop on desk                   → score: 14000  → role: "blur" (meeting rule)
  │     face (inside person A's bbox)    → role: "main"  (face post-rule)
  │     license_plate                    → role: "blur"  (plate post-rule, always)
  │
  └── Renderer:
        person A → green box (no blur)
        person B → Gaussian blur applied to bbox
        laptop   → Gaussian blur applied to bbox
        face     → green box (no blur, part of main)
        plate    → Gaussian blur applied to bbox
```

### Why CLIP runs every 15 frames

CLIP inference on a full frame takes ~20–40 ms on GPU. Running it every frame would halve throughput. Scene context changes slowly (seconds, not frames), so sampling every 15 frames (≈ 0.5 s at 30 fps) gives the same quality at a fraction of the cost.

---

## 6. Streamlit Application

### 6.1 UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sidebar                    │  Main content                         │
│  ─────────────────────      │  ─────────────────────────────────    │
│  Model Settings             │  PrivacyGuard  [hero]                 │
│    CLIP toggle              │                                       │
│    ViT toggle               │  [Upload video]   [What gets blurred] │
│    Saliency toggle          │                   [Speed tips]        │
│                             │                                       │
│  Blur Settings              │  [Start Processing]                   │
│    Blur People              │                                       │
│    Blur Screens             │  ── Live Preview ──────────────────── │
│    Blur Plates              │  ┌──────────────────┐ ┌────────────┐ │
│    Blur Strength slider     │  │   Blur preview   │ │ Live Stats │ │
│                             │  │   (clean, no     │ │ CLIP Scene │ │
│  Output Directory           │  │    boxes)        │ │ Blurred    │ │
│                             │  └──────────────────┘ │ Objects    │ │
│                             │                        └────────────┘ │
│                             │  [progress bar]                       │
│                             │                                       │
│                             │  Download: MP4 | JSONL               │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Model Loading Strategy

**Problem:** CLIP weights (~350 MB) take 15–20 s to load. Streamlit re-executes `app.py` on every browser interaction. If models were loaded in a module-level variable, they would be re-loaded on every rerun, causing OOM crashes.

**Solution:** `_model_cache.py` — a separate module that acts as a **process-level singleton**. Because Python caches imported modules in `sys.modules`, `_model_cache` is only imported once per process regardless of how many times Streamlit re-runs `app.py`.

```python
# _model_cache.py
import threading
store: dict = {}        # holds loaded model objects
lock  = threading.Lock()
ready = threading.Event()   # set when models are fully loaded
error: list = []            # holds traceback string on load failure
started = threading.Event() # guards against spawning multiple loader threads
```

The loader thread is started exactly once:
```python
def _ensure_models_loading(...):
    if _mc.started.is_set():   # already started → skip
        return
    _mc.started.set()
    threading.Thread(target=_do_load_models, ...).start()
```

While loading, the UI polls every 1.5 s (`time.sleep(1.5); st.rerun()`) showing a "Loading ML Models" card, keeping the WebSocket alive.

### 6.3 Live Preview & Stats Panel

During processing, the right column shows **3 live cards** updated every frame:

#### Card 1 — Live Stats
- Frames processed / total
- Processing fps
- ETA (mm:ss)
- Elapsed time

#### Card 2 — CLIP Scene Context
- **Scene label** (blue, title case) — e.g. *Meeting Room*
- **Confidence** — e.g. *87% confidence*
- **Active strategy** — e.g. *Strategy: Meeting*

#### Card 3 — Auto-Blurred Objects
- Pill chips for each blurred category × count — e.g. `Person ×3`, `Face ×2`, `Laptop ×1`
- Summary line: `N total detections · M blurred`

The `_ProcessingJob` background thread writes to `self.live_info` each frame:

```python
self.live_info = {
    "scene":         _last_scene or {},          # CLIP output
    "blur_counts":   dict(Counter(blur_labels)), # label → count
    "total_blurred": len(blur_labels),
    "total_dets":    len(dets),
    "strategy":      "MeetingStrategy",          # from decision_trace
}
```

The Streamlit UI loop reads `job.live_info` on each frame and re-renders the cards.

### 6.4 Blur & Model Toggles

All sidebar controls are passed into the processing job via `cfg_ov` (config overrides dict):

```python
cfg_ov = dict(
    blur_people=blur_people,       # BLUR_BACKGROUND_PEOPLE
    blur_screens=blur_screens,     # BLUR_SCREENS
    blur_plates=blur_plates,       # BLUR_PLATES
    blur_strength=blur_strength,   # BLUR_KERNEL_SIZE (21–91)
    use_vit=use_vit,
    use_clip=use_clip,
    use_saliency=use_saliency,
)
```

Inside `_ProcessingJob._run()`, a **fresh** `PipelineConfig`, `DecisionEngine`, and `Renderer` are created from these overrides. This is critical — reusing the shared model-cache config would ignore the user's sidebar choices.

---

## 7. Per-Frame Output Schema

Each frame logged to JSONL and passed between pipeline stages follows this schema:

```json
{
  "frame_id": 42,
  "timestamp": 1.4,
  "frame_shape": [1080, 1920],
  "num_objects": 4,
  "context": {
    "label": "meeting room",
    "confidence": 0.87
  },
  "frame_embedding": [0.12, -0.05, ...],
  "detections": [
    {
      "id": 0,
      "bbox": [120, 80, 420, 760],
      "label": "person",
      "confidence": 0.93,
      "area": 90000,
      "center": [270.0, 420.0],
      "relative_size": 0.043,
      "is_salient": true,
      "saliency_overlap": 0.72,
      "saliency_score": 0.81,
      "embedding": [0.04, -0.11, ...],
      "importance_score": 52341.4,
      "role": "main",
      "decision_trace": {
        "base_score": 50012.1,
        "saliency_boost": 2329.3,
        "penalty": 0.0,
        "final_score": 52341.4,
        "strategy": "MeetingStrategy",
        "rules_applied": [
          "meeting_centrality_bonus",
          "boosted_due_to_high_saliency",
          "selected_as_main_person"
        ]
      }
    }
  ]
}
```

---

## 8. Configuration Reference

Full list of all `PipelineConfig` fields:

| Field | Default | Description |
|---|---|---|
| `USE_YOLO` | `True` | Enable object/face/plate detection |
| `USE_VIT` | `False` | Enable ViT embeddings (slow; off by default) |
| `USE_CLIP` | `True` | Enable CLIP scene context |
| `USE_SALIENCY` | `False` | Enable saliency map computation |
| `YOLO_OBJ_WEIGHTS` | `yolov8n.pt` | Object detector weights path |
| `YOLO_FACE_WEIGHTS` | `yolov8n-face.pt` | Face detector weights path |
| `YOLO_PLATE_WEIGHTS` | `license-plate-finetune-v1n.pt` | Plate detector weights |
| `OBJ_CONF` | `0.3` | Object detection confidence threshold |
| `FACE_CONF` | `0.4` | Face detection confidence threshold |
| `PLATE_CONF` | `0.4` | Plate detection confidence threshold |
| `IMG_SIZE` | `640` | YOLO inference image size |
| `CLIP_INTERVAL` | `15` | Run CLIP every N frames |
| `FRAME_EMBED_INTERVAL` | `10` | Run full-frame ViT every N frames |
| `OBJECT_EMBED_INTERVAL` | `10` | Run per-object ViT every N frames |
| `SALIENCY_INTERVAL` | `5` | Run saliency every N frames |
| `CLIP_MODEL_NAME` | `ViT-B/32` | CLIP model alias |
| `VIT_MODEL_NAME` | `vit_base_patch16_224` | timm ViT model name |
| `CONTEXT_PROMPTS` | `[...]` | Zero-shot text prompts for CLIP |
| `TEMPORAL_BUFFER_SIZE` | `5` | Frames in temporal smoothing window |
| `BLUR_BACKGROUND_PEOPLE` | `True` | Blur non-main persons |
| `BLUR_SCREENS` | `True` | Blur laptops/TVs/phones |
| `BLUR_PLATES` | `True` | Blur license plates |
| `BLUR_KERNEL_SIZE` | `51` | Gaussian blur kernel (must be odd) |
| `SAL_GAIN` | `0.5` | Saliency boost multiplier |
| `SAL_PENALTY` | `0.3` | Low-saliency penalty multiplier |
| `MIN_SAL_OVERLAP` | `0.1` | Below this overlap → apply penalty |
| `SALIENCY_THRESHOLD` | `0.5` | Pixel threshold for saliency mask |
| `SALIENCY_OVERLAP_THRESHOLD` | `0.3` | Min overlap fraction to call object salient |
| `ENABLE_RENDER` | `True` | Write annotated output video |
| `SAVE_JSON` | `True` | Write JSONL log |
| `INPUT_VIDEO` | `video_d.mp4` | Default input path (CLI) |
| `OUTPUT_VIDEO` | `outputs/output.mp4` | Default output path (CLI) |
| `OUTPUT_LOG` | `outputs/output.jsonl` | Default log path (CLI) |

---

## 9. CLI Usage (`main.py`)

```bash
python main.py --input video.mp4
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | — | Input video path (required) |
| `--output PATH` | `outputs/output.mp4` | Output video path |
| `--max-frames N` | `0` (all) | Limit to first N frames |
| `--no-yolo` | — | Disable detection |
| `--no-vit` | — | Disable ViT embeddings |
| `--no-clip` | — | Disable CLIP context |
| `--no-render` | — | Skip annotated video |
| `--no-log` | — | Skip JSONL logging |
| `--saliency` | — | Enable saliency module |
| `--preview` | — | Show cv2 preview window |
| `--preview-fps N` | `15.0` | Preview window target FPS |
| `--vit` | — | Enable ViT (alias for `--no-vit` off) |
| `--parallel-yolo` | — | Run 3 YOLO models in parallel |
| `--frame-buffer-size N` | `4` | Preview frame buffer depth |
| `--clip-interval N` | `15` | CLIP sampling interval |
| `--object-embed-interval N` | `10` | ViT object sampling interval |
| `--save-det-video` | — | Save separate detection video |

---

## 10. Project File Structure

```
cvdl project/
│
├── app.py                      # Streamlit web application
├── _model_cache.py             # Process-level model singleton (survives reruns)
├── main.py                     # CLI entry point
│
├── config/
│   └── config.py               # PipelineConfig dataclass
│
├── core/
│   ├── base_module.py          # Abstract base: initialize() / release()
│   ├── pipeline.py             # Main frame-processing loop
│   ├── video_processor.py      # VideoReader / VideoWriter wrappers
│   └── frame_buffer.py         # 3-thread producer-consumer buffer
│
├── models/
│   ├── detector.py             # YOLOv8 × 3 (objects + faces + plates)
│   ├── embedding.py            # ViT per-object + frame embeddings
│   ├── context.py              # CLIP zero-shot scene classification
│   └── saliency.py             # Spectral / U2-Net saliency maps
│
├── engine/
│   ├── feature_engine.py       # Flatten & enrich detections
│   ├── decision_engine.py      # Strategy router + shared post-rules
│   ├── strategies.py           # 7 context-specific strategies
│   └── temporal.py             # IoU tracking + majority-vote smoothing
│
├── utils/
│   ├── visualization.py        # Renderer (blur + boxes + HUD)
│   ├── io.py                   # JSONLLogger
│   └── helpers.py              # crop_and_convert, misc
│
├── outputs/                    # Default output directory
│
├── U-2-Net/                    # U2-Net saliency model (submodule)
│
├── yolov8n.pt                  # YOLOv8n general object weights
├── yolov8n-face.pt             # YOLOv8n face detection weights
└── license-plate-finetune-v1n.pt  # Fine-tuned plate detector
```

---

## 11. Dependencies & Environment

**Conda environment:** `gpu` (`C:\Users\91930\anaconda3\envs\gpu\python.exe`)

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1+cu121 | Deep learning runtime (CUDA) |
| `ultralytics` | 8.4.46 | YOLOv8 inference |
| `transformers` | 5.7.0 | CLIP model loading |
| `timm` | 1.0.26 | ViT embeddings |
| `opencv-python` | 4.13.0 | Video I/O, drawing, blur |
| `streamlit` | 1.57.0 | Web UI |
| `Pillow` | — | Image preprocessing |
| `numpy` | — | Array operations |

**Launch command:**
```bash
python -m streamlit run app.py \
  --server.port 8501 \
  --server.fileWatcherType none \
  --server.headless true
```

- `--server.fileWatcherType none` — prevents Streamlit from watching and reloading `app.py` on file changes, which would kill the background processing thread mid-run
- `--server.headless true` — keeps the server alive even when no browser is connected (important during the 15 s model load window)

---

## 12. Design Decisions & Trade-offs

### JSONL over monolithic JSON
Writing one line per frame and flushing immediately means zero memory accumulation and crash safety. A 1-hour video at 30 fps produces ~108,000 lines — comfortably readable by any JSON Lines tool.

### CLIP every 15 frames, not every frame
Scene context changes over seconds, not frames. Sampling every 15 frames (~0.5 s at 30 fps) recovers ~40 ms GPU time per 15 frames with no perceptible quality loss.

### ViT disabled by default
Per-object ViT embeddings are written to the JSONL log for downstream analysis but are not used in the real-time blur decision loop. Disabling them (the default) gives approximately 7× speedup on the embedding stage.

### `_model_cache.py` singleton pattern
Streamlit's execution model `exec()`s `app.py` on every browser event. Any module-level state is reset. Moving model storage into a separate module that is only imported once per Python process avoids repeated model loading and OOM crashes.

### Per-job `DecisionEngine` + `Renderer`
Models are loaded once and shared. But `DecisionEngine` and `Renderer` are constructed fresh for each processing job, taking the user's current sidebar settings (blur toggles, blur strength). This is the only way to honour per-run configuration without reloading heavy GPU models.

### Temporal smoothing prevents flickering
YOLO confidence varies frame-to-frame. Without smoothing, a person could flip between `"main"` and `"blur"` every few frames, causing visible blur-unblur flicker. The 5-frame IoU-tracked majority vote eliminates this at negligible computational cost.

### Importance score for "main" election
Rather than hard-coding rules like "the largest person is main", the importance score blends area, distance from centre, and saliency into a single number. This makes the election:
- Robust to edge cases (e.g. a large background extra vs. a smaller central subject)
- Controllable via config (`SAL_GAIN`, `SAL_PENALTY`)
- Transparent via `decision_trace`
