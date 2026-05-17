"""Ultrasonic deterrent tone — randomized pulsed burst in the cat-only band.

We pulse in 22–24 kHz (above adult human hearing, well within cats').
Frequency is randomized per-pulse so the cat doesn't habituate. The whole
burst is regenerated on every trigger.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from cat_howl_deterrent.config import (
    TONE_FREQ_HIGH_HZ,
    TONE_FREQ_LOW_HZ,
    TONE_GAP_MS,
    TONE_PULSE_MS,
    TONE_SAMPLE_RATE,
    TONE_VOLUME,
)


def build_tone(duration_s: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    pulse_samples = int(TONE_SAMPLE_RATE * TONE_PULSE_MS / 1000)
    gap_samples = int(TONE_SAMPLE_RATE * TONE_GAP_MS / 1000)
    target_samples = int(TONE_SAMPLE_RATE * duration_s)
    fade = int(0.01 * TONE_SAMPLE_RATE)

    out: list[np.ndarray] = []
    n = 0
    t = np.arange(pulse_samples) / TONE_SAMPLE_RATE
    envelope = np.ones(pulse_samples, dtype=np.float32)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

    while n < target_samples:
        freq = float(rng.uniform(TONE_FREQ_LOW_HZ, TONE_FREQ_HIGH_HZ))
        pulse = (np.sin(2 * np.pi * freq * t).astype(np.float32) * envelope * TONE_VOLUME)
        out.append(pulse)
        n += pulse_samples
        if n >= target_samples:
            break
        out.append(np.zeros(gap_samples, dtype=np.float32))
        n += gap_samples

    return np.concatenate(out)[:target_samples]


def play(tone: np.ndarray) -> None:
    try:
        sd.play(tone, TONE_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"  ! tone playback failed: {e}")
