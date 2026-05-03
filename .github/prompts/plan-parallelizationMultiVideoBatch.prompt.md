# Plan: Single-Video Real-Time Streaming with Maximum Throughput

**TL;DR** — The pipeline is fully sequential with no buffering. We introduce three layers of parallelism (I/O threading, batched ViT inference, optional parallel YOLO) and a **producer-consumer streaming architecture** so processed frames are rendered as fast as possible and played back at exactly the video's native FPS via a paced display consumer. The pipeline races ahead filling a bounded buffer (the "delay"); the viewer thread drains it at real-time speed — achieving smooth, live-like playback from a single video source without skipping or tearing.

---

## Phase 1 — Config Extensions
1. `config/config.py`: Add four new `PipelineConfig` fields — `SAVE_DET_VIDEO: bool`, `OUTPUT_DET_VIDEO: str`, `PARALLEL_YOLO: bool`, `FRAME_BUFFER_SIZE: int = 4`

---

## Phase 2 — Intra-Frame Optimizations *(parallel with Phase 1)*
2. `models/embedding.py` → `embed_objects()`: Replace the per-crop sequential ViT loop with a single batched tensor forward pass — collect all crops, stack, run once, distribute embeddings back.
3. `models/detector.py` → `detect()`: When `PARALLEL_YOLO=True`, issue all 3 YOLO model calls simultaneously via `concurrent.futures.ThreadPoolExecutor(max_workers=3)` — PyTorch releases the GIL for CUDA ops, so this is safe.

---

## Phase 3 — Producer-Consumer Streaming Pipeline *(depends on Phase 1)*
4. **Create `core/frame_buffer.py`** — `FrameBuffer` class implementing a three-thread pipeline:

   **Thread A — Raw Reader (producer)**
   - `start_reader_thread()` → daemon thread calls `VideoReader.read()` in a tight loop, pushes `(frame_idx, raw_frame)` onto a bounded `raw_queue` (size = `FRAME_BUFFER_SIZE`).
   - Pushes sentinel `None` on EOF. Naturally back-pressures if ML thread is slower than decode speed.

   **Thread B — ML Worker (main thread)**
   - Replaces the current monolithic loop in `Pipeline.run()`.
   - Pops from `raw_queue`, runs all 10 pipeline stages (detect → embed → saliency → decide → render), pushes `(frame_idx, annotated, saliency_frame, det_frame)` onto bounded `display_queue` (size = `FRAME_BUFFER_SIZE`).
   - Bounded `display_queue` back-pressures the ML thread if the display consumer falls behind, preventing unbounded memory growth.

   **Thread C — Paced Display / Writer (consumer)**
   - `start_writer_thread(writers...)` → daemon thread pops from `display_queue`.
   - **Playback pacing**: tracks `playback_start = time.perf_counter()` on first frame. For frame `i`, computes `target_t = playback_start + i / fps`. Calls `time.sleep(max(0, target_t - time.perf_counter()))` before each write/display call.
   - Routes frames to up to 3 `VideoWriter`s (main, saliency, detection) and/or `cv2.imshow()` if `--preview` flag is set.
   - If `display_queue` is empty (ML is slower than real-time), consumer blocks until next frame arrives — no frame drops, just natural wait.

   **Result**: the pipeline races through frames as fast as GPU allows, buffering up to `FRAME_BUFFER_SIZE` frames ahead. If processing is faster than real-time the buffer fills and pacing kicks in. If processing is slower than real-time the buffer drains and the viewer waits — both cases produce smooth, tear-free playback.

5. `core/pipeline.py` → `Pipeline.run()`: Replace direct `reader.read()` / `writer.write()` loop with `FrameBuffer` start/get/put/stop. Wrap in `try/finally` to guarantee thread shutdown on exception.

---

## Phase 4 — Detection Video Intermediate Output *(depends on Phase 1)*
6. `utils/visualization.py`: Add `render_detections_frame(frame, detections) → np.ndarray` — draws raw YOLO bboxes + labels on an unblurred frame copy.
7. `core/pipeline.py`: When `SAVE_DET_VIDEO=True`, open a third `VideoWriter` (`det_writer`) at startup; call `render_detections_frame()` after the detection step, push into `FrameBuffer`'s detection channel.

---

## Phase 5 — Expose All New Flags in main.py *(depends on Phases 1–4)*
8. `main.py` → `parse_args()`: Add:
   - `--save-det-video` → sets `SAVE_DET_VIDEO=True`
   - `--parallel-yolo` → sets `PARALLEL_YOLO=True`
   - `--frame-buffer-size N` → sets `FRAME_BUFFER_SIZE=N` (default 4)
   - `--preview` → open `cv2.imshow` window in the paced consumer thread for live viewing

---

## Relevant Files

| File | Change |
|------|--------|
| `config/config.py` | Add 4 new fields + `PREVIEW: bool` |
| `models/embedding.py` | Batch ViT inference |
| `models/detector.py` | Optional ThreadPoolExecutor |
| `core/pipeline.py` | FrameBuffer integration, det_writer |
| `core/frame_buffer.py` | **NEW** — three-thread producer-consumer buffer |
| `utils/visualization.py` | Add `render_detections_frame()` |
| `main.py` | 4 new CLI flags |

---

## Output Structure (unchanged flat layout)
```
outputs/
  output.mp4               ← blurred output (always)
  output_saliency.mp4      ← saliency overlay (--saliency-render)
  output_detections.mp4    ← bbox-only raw video (--save-det-video)
  output.jsonl             ← per-frame metadata (always)
```
`main.py` path defaults remain backward-compatible.

---

## Verification Steps
1. `python main.py --input <video>` → identical output to current behavior (backward compat, no preview window)
2. `python main.py --input <video> --preview` → `cv2.imshow` window opens; playback appears smooth at native FPS; output file written simultaneously
3. `python main.py --input <video> --save-det-video` → `output_detections.mp4` written alongside `output.mp4`
4. `python main.py --input <video> --saliency --saliency-render` → `output_saliency.mp4` written
5. `python main.py --input <video> --parallel-yolo` → profile YOLO stage wall time vs. baseline; expect ~2–3× speedup on detection step
6. `python main.py --input <video> --frame-buffer-size 8` → increase buffer depth; confirm no frame drops on slower GPU by observing `display_queue` depth in debug log
7. Instrument `embed_objects()` before/after batching with 5+ detections per frame; confirm single-pass speedup

---

## Scope Boundaries
- **Included**: three-thread producer-consumer I/O pipeline, paced real-time playback consumer, batched ViT inference, optional parallel YOLO, detection video intermediate output, `--preview` live window
- **Excluded**: multi-video batch processing, multi-GPU support, process-level parallelism, dual-backend saliency per pass
- **No refactoring** of existing strategy/decision logic, temporal smoother, or renderer internals
