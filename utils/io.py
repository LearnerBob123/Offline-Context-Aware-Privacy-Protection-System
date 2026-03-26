"""
Streaming JSONL logger – writes one JSON line per frame.
No in-memory accumulation of results.
"""

import json
import os


class JSONLLogger:
    """Appends one JSON line per call. Memory-safe for long videos."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")

    def log(self, record: dict) -> None:
        """Write a single frame record as one JSON line."""
        # Strip large embedding vectors before writing to keep logs lean
        clean = self._strip_embeddings(record)
        self.fh.write(json.dumps(clean) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh and not self.fh.closed:
            self.fh.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_embeddings(record: dict) -> dict:
        """Remove bulky embedding arrays to keep JSONL readable."""
        out = {}
        for k, v in record.items():
            if k in ("frame_embedding", "clip_embedding"):
                out[k] = v is not None  # store flag only
                continue
            if k == "detections" and isinstance(v, list):
                cleaned = []
                for d in v:
                    d_copy = {
                        dk: dv for dk, dv in d.items()
                        if dk not in ("visual_embedding", "clip_embedding")
                    }
                    cleaned.append(d_copy)
                out[k] = cleaned
                continue
            out[k] = v
        return out
