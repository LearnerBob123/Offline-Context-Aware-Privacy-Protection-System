# Video Processing Pipeline — Modular Research Framework

A config-driven, memory-efficient, streaming-safe computer vision pipeline for video analysis, combining **object detection (YOLO)**, **visual embeddings (ViT)**, **scene understanding (CLIP)**, **rule-based decision making**, **temporal smoothing**, and **annotated video rendering**.

---

## 1. System Overview

The pipeline processes each frame through a sequence of independent, modular stages:

```
Input Video
    │
    ▼
┌─────────────┐
│  Detection   │  YOLO (objects, faces, license plates)
└─────┬───────┘
      ▼
┌─────────────┐
│  Embedding   │  ViT frame & object embeddings
└─────┬───────┘
      ▼
┌─────────────┐
│   Context    │  CLIP zero-shot scene classification
└─────┬───────┘
      ▼
┌─────────────┐
│   Feature    │  Merges spatial features, embeddings, metadata
│   Engine     │
└─────┬───────┘
      ▼
┌─────────────┐
│  Decision    │  Rule-based role assignment (main / blur / background)
│  Engine      │
└─────┬───────┘
      ▼
┌─────────────┐
│  Temporal    │  Sliding-window smoothing + IoU tracking
│  Smoother    │
└─────┬───────┘
      ▼
┌──────────┐  ┌────────────┐
│ Renderer │  │ JSONL Log  │
│ (video)  │  │ (per-frame)│
└──────────┘  └────────────┘
```

Every module can be toggled on/off via the central `PipelineConfig`.

---

## 2. Data Flow

```
Frame  →  Detection  →  Embedding  →  Context  →  Feature Engine  →  Decision Engine  →  Temporal Smoothing  →  Render  →  Log
```

Each frame produces a dictionary:

```json
{
  "frame_id": 42,
  "timestamp": 1.68,
  "frame_shape": [1080, 1920],
  "num_objects": 5,
  "context": {"label": "meeting room", "confidence": 0.87},
  "detections": [
    {
      "id": 0,
      "bbox": [100, 200, 300, 600],
      "label": "person",
      "confidence": 0.92,
      "importance_score": 48231.5,
      "role": "main"
    }
  ]
}
```

---

## 3. Logging Design — JSONL

The prototype accumulated all frame outputs in a Python list and dumped them as a single JSON file, which:

* **Crashes on long videos** — unbounded memory growth.
* **Loses all data** if the process dies before the final write.

This framework uses **JSONL (JSON Lines)** instead:

* Every frame is written as **one JSON object per line**, flushed immediately.
* **Zero in-memory accumulation** — only the current frame's data lives in RAM.
* If the process crashes at frame 5000, you still have 4999 complete records.
* Easy to process with standard tools: `wc -l`, `jq`, `pandas.read_json(lines=True)`.

Output file: `outputs/output.jsonl`

```
{"frame_id": 0, "timestamp": 0.0, "detections": [...]}
{"frame_id": 1, "timestamp": 0.04, "detections": [...]}
...
```

---

## 4. Video Output Handling

| Aspect | Implementation |
|--------|---------------|
| **Codec** | `mp4v` — widely supported, no external codec install needed |
| **FPS fallback** | If the source reports FPS ≤ 0(some webcam captures), defaults to **25 fps** |
| **Frame dtype** | Enforced `uint8` before every write to avoid OpenCV errors |
| **Resource safety** | Writer is released in a `finally` block to prevent corruption |

Output file: `outputs/output.mp4`

---

## 5. How to Run

### Basic

```bash
python main.py --input video.mp4
```

### All options

```bash
python main.py \
  --input video.mp4 \
  --output outputs/custom.mp4 \
  --max-frames 500 \
  --no-clip \
  --no-vit \
  --saliency
```

| Flag | Effect |
|------|--------|
| `--input PATH` | Input video path |
| `--output PATH` | Output video path |
| `--max-frames N` | Process only first N frames (0 = all) |
| `--no-yolo` | Disable YOLO detection |
| `--no-vit` | Disable ViT embeddings |
| `--no-clip` | Disable CLIP context |
| `--no-render` | Skip annotated video output |
| `--no-log` | Skip JSONL logging |
| `--saliency` | Enable saliency module |

### Prerequisites

```bash
pip install ultralytics opencv-python torch torchvision timm pillow numpy
pip install git+https://github.com/openai/CLIP.git
```

Model weights (`yolov8n.pt`, `yolov8n-face.pt`, `license-plate-finetune-v1n.pt`) should be in the project root or specify full paths in `config/config.py`.

---

## 6. Extending the System

### Add a new model

1. Create `models/my_model.py` inheriting from `BaseModule`.
2. Implement `initialize()`, `release()`, and your inference method.
3. Add a toggle flag in `PipelineConfig` (e.g., `USE_MY_MODEL`).
4. Wire it into `core/pipeline.py` following the existing pattern.

### Add a new decision rule

Edit `engine/decision_engine.py` → `DecisionEngine.decide()`:

```python
# Example: blur all faces unconditionally
for d in dets:
    if d["label"] == "face":
        d["role"] = "blur"
```

### Add a new feature

Edit `engine/feature_engine.py` → add the field to the enriched dict:

```python
entry["my_feature"] = compute_something(item)
```

---

## 7. Project Structure

```
project_root/
├── config/
│   └── config.py              # All flags, paths, hyperparameters
├── core/
│   ├── base_module.py          # Abstract base for all modules
│   ├── video_processor.py      # VideoReader / VideoWriter
│   └── pipeline.py             # Central orchestration loop
├── models/
│   ├── detector.py             # YOLO wrapper (objects, faces, plates)
│   ├── embedding.py            # ViT embeddings
│   ├── context.py              # CLIP scene classification
│   └── saliency.py             # Optional saliency maps
├── engine/
│   ├── feature_engine.py       # Enriches detections with spatial features
│   ├── decision_engine.py      # Rule-based role assignment
│   └── temporal.py             # Sliding-window smoothing + IoU tracking
├── utils/
│   ├── io.py                   # Streaming JSONL logger
│   ├── visualization.py        # Renderer (boxes, blur, HUD)
│   └── helpers.py              # Shared utilities (crop, convert)
├── outputs/
│   ├── output.jsonl            # Per-frame structured log
│   └── output.mp4              # Annotated output video
├── main.py                     # CLI entry point
└── README.md
```
