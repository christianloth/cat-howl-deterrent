"""Per-trigger event logging (WAV + JSON to ./logs/)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from cat_howl_deterrent.config import LOG_DIR, SAMPLE_RATE


@dataclass
class Trigger:
    timestamp: float
    trigger_score: float
    trigger_class: str
    trigger_scores: dict
    log_scores: dict
    context: dict | None
    top_classes: list
    audio: np.ndarray


def save_trigger(trigger: Trigger, save_clips: bool = True, log_dir: Path = LOG_DIR) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(trigger.timestamp))
    if save_clips:
        sf.write(log_dir / f"{ts}.wav", trigger.audio, SAMPLE_RATE)
    payload = {
        "timestamp": trigger.timestamp,
        "trigger_class": trigger.trigger_class,
        "trigger_score": float(trigger.trigger_score),
        "trigger_scores": trigger.trigger_scores,
        "log_scores": trigger.log_scores,
        "context": trigger.context,
        "top_classes": trigger.top_classes,
    }
    (log_dir / f"{ts}.json").write_text(json.dumps(payload, indent=2))
