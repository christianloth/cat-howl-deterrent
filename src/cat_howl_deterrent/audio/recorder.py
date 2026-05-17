"""Dynamic-length pre/post-roll capture for fine-tuning data collection.

When a trigger fires we want a clip that includes audio from BEFORE the
trigger (so the howl onset is captured) plus as much audio AFTER as the
cat continues vocalizing. Strategy:

  * Maintain a rolling pre-roll deque of recent 16 kHz hop chunks.
  * On trigger, seed a new "active capture" with the pre-roll contents.
  * For each subsequent hop, append it to the active capture and update
    a quiet-counter:
      - if the cat-class score is ≥ HOWL_EXTEND_THRESHOLD, reset quiet=0
      - else increment quiet by one hop
  * Finalize the capture when either:
      - quiet has accumulated ≥ HOWL_TAIL_SECONDS *and* total post-roll
        is already past HOWL_POST_MIN_SECONDS, OR
      - total post-roll has reached HOWL_MAX_SECONDS (safety net).

This makes short meows produce short clips and long howls produce long
clips, without padding silence onto the end or chopping vocalizations in
half.
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
    HOWL_EXTEND_THRESHOLD,
    HOWL_MAX_SECONDS,
    HOWL_POST_MIN_SECONDS,
    HOWL_PRE_SECONDS,
    HOWL_TAIL_SECONDS,
    SAMPLE_RATE,
)


def _hops_for(seconds: float) -> int:
    """Round seconds up to a whole number of hop chunks."""
    return int(np.ceil(seconds / HOP_SECONDS))


class HowlRecorder:
    """Captures pre/post-roll audio around triggers, with dynamic post-roll.

    Caller responsibilities:
      - call `feed_chunk(chunk_16k, cat_score)` once per main-loop iteration,
        passing the highest cat-class score for that frame
      - call `start_capture(ts, class_name, gated)` when a trigger fires
      - finished captures are written to `out_dir/` as they finalize
    """

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._pre_roll: deque = deque(maxlen=_hops_for(HOWL_PRE_SECONDS))
        self._min_hops = _hops_for(HOWL_POST_MIN_SECONDS)
        self._tail_hops = _hops_for(HOWL_TAIL_SECONDS)
        self._max_hops = _hops_for(HOWL_MAX_SECONDS)
        self._active: list[dict[str, Any]] = []

    def feed_chunk(self, chunk: np.ndarray, cat_score: float) -> None:
        """Add a 16 kHz mono chunk to the pre-roll and any active captures.

        `cat_score` is the max score across TRIGGER_CLASSES for this frame —
        used to decide whether to extend the post-roll.
        """
        self._pre_roll.append(chunk)
        for cap in self._active:
            cap["audio"].append(chunk)
            cap["post_hops"] += 1
            if cat_score >= HOWL_EXTEND_THRESHOLD:
                cap["quiet_hops"] = 0
            else:
                cap["quiet_hops"] += 1
            # Finalize if we've had enough silence (after min post-roll),
            # or if we've hit the hard ceiling.
            min_reached = cap["post_hops"] >= self._min_hops
            tail_done = cap["quiet_hops"] >= self._tail_hops
            hit_ceiling = cap["post_hops"] >= self._max_hops
            cap["finished"] = (min_reached and tail_done) or hit_ceiling
        self._drain_finished()

    def start_capture(self, ts: float, trigger_class: str, gated: bool) -> None:
        """Begin a new pre+post-roll capture seeded with the current pre-roll."""
        self._active.append(
            {
                "ts": ts,
                "trigger_class": trigger_class,
                "gated": gated,
                "audio": list(self._pre_roll),
                "post_hops": 0,
                "quiet_hops": 0,
                "finished": False,
            }
        )

    def _drain_finished(self) -> None:
        finished = [c for c in self._active if c["finished"]]
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
                    f"({len(clip)/SAMPLE_RATE:.1f}s, post={cap['post_hops']*HOP_SECONDS:.1f}s)"
                )
            except Exception as e:
                print(f"  ! howl save failed: {e}")
        self._active = [c for c in self._active if not c["finished"]]
