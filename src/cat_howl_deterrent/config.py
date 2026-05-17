"""All configuration knobs in one place.

Values come from environment variables where appropriate; the rest are
module-level constants so they can be tweaked or imported. The shapes
match what the original monolith expected, so behavior is preserved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ── Audio / model shape ─────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # YAMNet requires 16 kHz mono
FRAME_SECONDS = 0.96          # YAMNet frame length
HOP_SECONDS = 0.48            # How often we re-classify
FRAME_SAMPLES = 15_600        # 0.96 s @ 16 kHz


# ── Trigger gating ─────────────────────────────────────────────────────────
TRIGGER_THRESHOLD = 0.5       # Max score across TRIGGER_CLASSES to consider
SUSTAIN_FRAMES = 2            # Must trigger this many frames in a row
COOLDOWN_SECONDS = 30         # Min seconds between deterrent pulses
HEAD_GATE_THRESHOLD = 0.5     # Verifier-head probability gate
REQUIRE_HEAD_IF_AVAILABLE = True

# YAMNet AudioSet class IDs we gate the deterrent on.
TRIGGER_CLASSES = {
    76: "Cat",
    78: "Meow",
    80: "Caterwaul",
}
# Extra classes recorded in logs but NOT used for gating.
LOG_CLASSES = {
    77: "Purr",
    79: "Hiss",
    81: "Growling",
    82: "Bow-wow",      # Dog — useful to spot mic confusion
    83: "Yip",
}


# ── Deterrent tone ─────────────────────────────────────────────────────────
TONE_FREQ_LOW_HZ = 22_000
TONE_FREQ_HIGH_HZ = 24_000
TONE_PULSE_MS = 150
TONE_GAP_MS = 100
TONE_VOLUME = 0.4
TONE_SAMPLE_RATE = 48_000


# ── Howl-recording (fine-tuning data capture) ──────────────────────────────
HOWL_PRE_SECONDS = 3.0
HOWL_POST_SECONDS = 5.0


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
HOWL_DIR = PROJECT_ROOT / "howl_recordings"
MODELS_DIR = PROJECT_ROOT / "models"

CONTEXT_HEAD_PATH = MODELS_DIR / "yamnet_audioset_cats_head.keras"
CONTEXT_LABELS_PATH = MODELS_DIR / "yamnet_audioset_cats_labels.json"
COREML_MODEL_PATH = PROJECT_ROOT / "yamnet.mlpackage"
YAMNET_CLASSES_PATH = PROJECT_ROOT / "yamnet_classes.json"


# ── Quiet-hours filter (set to None = always active) ───────────────────────
QUIET_HOURS_START: int | None = None
QUIET_HOURS_END: int | None = None


# ── Env-var-resolved runtime config ────────────────────────────────────────
def _bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    log_only: bool
    record_howls: bool
    mic_device: str | None
    mic_backend: str
    deterrent_seconds: float

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        return cls(
            backend=os.environ.get("BACKEND", "cpu").lower(),
            log_only=_bool("LOG_ONLY"),
            record_howls=_bool("RECORD_HOWLS"),
            mic_device=os.environ.get("MIC_DEVICE") or None,
            mic_backend=os.environ.get("MIC_BACKEND", "sd").lower(),
            deterrent_seconds=float(os.environ.get("DETERRENT_SECONDS", "3.0")),
        )
