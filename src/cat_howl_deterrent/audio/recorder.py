"""Rolling pre-roll + post-roll capture for fine-tuning data collection.

When a trigger fires, we want a clip that includes audio from BEFORE the
trigger (so the howl onset is captured) plus a few seconds AFTER. This
keeps a deque of recent hop-sized chunks at 16 kHz, then on a trigger
starts an "active capture" that collects N more hops before writing the
WAV.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cat_howl_deterrent.config import (
    HOP_SECONDS,
    HOWL_POST_SECONDS,
    HOWL_PRE_SECONDS,
    SAMPLE_RATE,
)


class HowlRecorder:
    """Captures pre/post-roll audio around triggers.

    Caller responsibilities:
      - call `feed_chunk(chunk_16k)` once per main-loop iteration
      - call `start_capture(class_name, gated)` when a trigger fires
      - the recorder writes finished captures to `out_dir/` as they finish
    """

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pre_chunks = int(np.ceil(HOWL_PRE_SECONDS / HOP_SECONDS))
        self._pre_roll: deque = deque(maxlen=pre_chunks)
        self._post_target = int(np.ceil(HOWL_POST_SECONDS / HOP_SECONDS))
        self._active: list[dict[str, Any]] = []

    def feed_chunk(self, chunk: np.ndarray) -> None:
        """Add a 16 kHz mono chunk to the pre-roll and any active captures."""
        self._pre_roll.append(chunk)
        for cap in self._active:
            cap["audio"].append(chunk)
            cap["remaining"] -= 1
        self._drain_finished()

    def start_capture(self, ts: float, trigger_class: str, gated: bool) -> None:
        """Begin a new pre+post-roll capture seeded with the current pre-roll."""
        self._active.append(
            {
                "ts": ts,
                "trigger_class": trigger_class,
                "gated": gated,
                "audio": list(self._pre_roll),
                "remaining": self._post_target,
            }
        )

    def _drain_finished(self) -> None:
        finished = [c for c in self._active if c["remaining"] <= 0]
        if not finished:
            return
        for cap in finished:
            clip = np.concatenate(cap["audio"]).astype(np.float32)
            ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(cap["ts"]))
            suffix = "_gated" if cap["gated"] else ""
            out_path = self.out_dir / f"{ts_str}_{cap['trigger_class']}{suffix}.wav"
            try:
                sf.write(out_path, clip, SAMPLE_RATE)
                print(
                    f"  ♪ saved howl recording: {out_path.name} "
                    f"({len(clip)/SAMPLE_RATE:.1f}s)"
                )
            except Exception as e:
                print(f"  ! howl save failed: {e}")
        self._active = [c for c in self._active if c["remaining"] > 0]
