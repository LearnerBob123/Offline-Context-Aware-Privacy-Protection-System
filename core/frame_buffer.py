"""
Three-thread producer-consumer frame buffer for real-time streaming.

Architecture
------------
Thread A  [FrameReader]  – reads raw frames from VideoReader into raw_queue.
Thread B  [ML Worker]    – the main pipeline thread; pops from raw_queue,
                           runs all inference stages, pushes results to
                           display_queue.
Thread C  [FrameWriter]  – drains display_queue, writes to VideoWriter(s),
                           and optionally shows a paced cv2.imshow window.

The two bounded queues provide natural back-pressure:
  • raw_queue   : if ML is slower than decode, reader pauses (queue full).
  • display_queue: if writer is slower than ML (rare), ML pauses.

Pacing (--preview mode)
-----------------------
The writer thread records t_start on the first frame, then for each
frame_idx sleeps until  t_start + frame_idx / fps  before displaying.
File writes are always immediate (no pacing).
"""

import threading
import queue
import time

import cv2
import numpy as np


class FrameBuffer:
    """Manages reader and writer daemon threads around two bounded queues."""

    _SENTINEL = None  # pushed to signal end-of-stream

    def __init__(self, reader, buffer_size: int, fps: float):
        """
        Parameters
        ----------
        reader      : VideoReader  – source of raw frames.
        buffer_size : int          – max depth of each queue.
        fps         : float        – native video FPS (used for pacing).
        """
        self.reader = reader
        self.fps = fps
        self.buffer_size = buffer_size

        self._raw_q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._display_q: queue.Queue = queue.Queue(maxsize=buffer_size)

        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Thread A – raw frame reader
    # ------------------------------------------------------------------
    def start_reader_thread(self) -> None:
        """Start the background thread that decodes frames into raw_queue."""

        def _read() -> None:
            idx = 0
            while not self._stop_event.is_set():
                ret, frame = self.reader.read()
                if not ret:
                    # Signal EOF to the ML worker
                    self._raw_q.put(self._SENTINEL)
                    break
                # Block here if ML worker is busy (back-pressure)
                self._raw_q.put((idx, frame))
                idx += 1

        self._reader_thread = threading.Thread(
            target=_read, name="FrameReader", daemon=True
        )
        self._reader_thread.start()

    # ------------------------------------------------------------------
    # Thread C – paced writer / display consumer
    # ------------------------------------------------------------------
    def start_writer_thread(
        self,
        writer=None,
        sal_writer=None,
        det_writer=None,
        preview: bool = False,
        preview_fps: float = 15.0,
    ) -> None:
        """
        Start the background thread that consumes processed frames.

        Parameters
        ----------
        writer      : VideoWriter | None  – main (blurred) output writer.
        sal_writer  : VideoWriter | None  – saliency heatmap writer.
        det_writer  : VideoWriter | None  – detection-only bbox writer.
        preview     : bool                – open a cv2.imshow window.
        preview_fps : float               – target display FPS for pacing
                                           (typically 15 or 30). The pipeline
                                           races ahead filling the buffer;
                                           the viewer drains it at this rate.
        """

        PREVIEW_W, PREVIEW_H = 1280, 720   # fixed window size (HD)

        def _write() -> None:
            t_start: float | None = None
            frame_display_idx: int = 0   # counts frames actually shown (for pacing)
            window_created = False

            while not self._stop_event.is_set():
                try:
                    item = self._display_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is self._SENTINEL:
                    break

                frame_idx, annotated, sal_frame, det_frame, preview_frame = item

                # ---- File writes (always immediate, no pacing) ----
                if writer is not None and annotated is not None:
                    writer.write(annotated)
                if sal_writer is not None and sal_frame is not None:
                    sal_writer.write(sal_frame)
                if det_writer is not None and det_frame is not None:
                    det_writer.write(det_frame)

                # ---- Live preview window (paced at preview_fps) ----
                if preview and preview_frame is not None:
                    if t_start is None:
                        t_start = time.perf_counter()
                    # Sleep until this display slot's deadline
                    target_t = t_start + frame_display_idx / max(preview_fps, 1.0)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                    # Resize to fixed window size, preserving aspect ratio with black bars
                    h, w = preview_frame.shape[:2]
                    scale = min(PREVIEW_W / w, PREVIEW_H / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(preview_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    canvas = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                    y0 = (PREVIEW_H - nh) // 2
                    x0 = (PREVIEW_W - nw) // 2
                    canvas[y0:y0 + nh, x0:x0 + nw] = resized

                    if not window_created:
                        cv2.namedWindow("Live Preview  (press Q to quit)", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Live Preview  (press Q to quit)", PREVIEW_W, PREVIEW_H)
                        window_created = True

                    cv2.imshow("Live Preview  (press Q to quit)", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._stop_event.set()
                        break
                    frame_display_idx += 1

            if preview:
                cv2.destroyAllWindows()

        self._writer_thread = threading.Thread(
            target=_write, name="FrameWriter", daemon=True
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # ML worker interface (Thread B – called from main pipeline loop)
    # ------------------------------------------------------------------
    def get_frame(self):
        """
        Block until a raw frame is available.

        Returns
        -------
        (frame_idx: int, frame: np.ndarray)  or  None on end-of-stream.
        """
        return self._raw_q.get()

    def put_frame(
        self,
        frame_idx: int,
        annotated: np.ndarray | None,
        sal_frame: np.ndarray | None = None,
        det_frame: np.ndarray | None = None,
        preview_frame: np.ndarray | None = None,
    ) -> None:
        """Push a fully processed frame tuple onto the display queue."""
        self._display_q.put((frame_idx, annotated, sal_frame, det_frame, preview_frame))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def signal_done(self) -> None:
        """Tell the writer thread that no more frames are coming."""
        self._display_q.put(self._SENTINEL)

    def stop(self, timeout: float = 10.0) -> None:
        """
        Gracefully shut down both daemon threads.

        Call this in the pipeline's finally block after the ML loop ends.
        """
        self._stop_event.set()

        # Unblock the reader thread if it is waiting on a full raw_queue
        try:
            self._raw_q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=timeout)
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
