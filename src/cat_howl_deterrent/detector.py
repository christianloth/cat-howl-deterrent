"""Main detection loop — wires up audio, classifier, gating, recording, and deterrent.

Architecture:
  mic capture (native rate)
    → ring buffer (native rate)
    → resampler → 16 kHz frame
    → YAMNet (scores + optional embedding)
    → trigger threshold + sustain frames
    → verifier head gate (if available)
    → save trigger / start howl capture / play tone
"""

from __future__ import annotations

import threading
import time

import numpy as np

from cat_howl_deterrent.audio.capture import _MicBase, make_mic_capture
from cat_howl_deterrent.audio.recorder import HowlRecorder
from cat_howl_deterrent.audio.resample import make_resampler
from cat_howl_deterrent.classifier.backends import make_backend
from cat_howl_deterrent.classifier.verifier import VerifierHead, gate
from cat_howl_deterrent.config import (
    COOLDOWN_SECONDS,
    FRAME_SAMPLES,
    FRAME_SECONDS,
    HEAD_GATE_THRESHOLD,
    HOWL_DIR,
    LOG_CLASSES,
    QUIET_HOURS_END,
    QUIET_HOURS_START,
    SAMPLE_RATE,
    SUSTAIN_FRAMES,
    TONE_FREQ_HIGH_HZ,
    TONE_FREQ_LOW_HZ,
    TRIGGER_CLASSES,
    TRIGGER_THRESHOLD,
    RuntimeConfig,
)
from cat_howl_deterrent.deterrent import build_tone
from cat_howl_deterrent.deterrent import play as play_tone
from cat_howl_deterrent.events import Trigger, save_trigger


def in_quiet_hours() -> bool:
    """True when we're allowed to fire (= within active hours).

    Naming is historical: returns whether deterrent is permitted. None/None
    means always active.
    """
    if QUIET_HOURS_START is None or QUIET_HOURS_END is None:
        return True
    hour = time.localtime().tm_hour
    if QUIET_HOURS_START <= QUIET_HOURS_END:
        return QUIET_HOURS_START <= hour < QUIET_HOURS_END
    return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END


def _start_watchdog(mic: _MicBase) -> None:
    """Print diagnostics if no callbacks arrive after stream start."""

    def _run():
        deadline = time.time() + 20.0
        last = -1
        while time.time() < deadline:
            time.sleep(5)
            n = mic.callbacks_received
            if n == last and n == 0:
                print(
                    f"  ! [watchdog] no mic callbacks in "
                    f"{time.time() - mic.started_at:.0f}s — check mic permission / TCC"
                )
            elif n > 0:
                print(f"  · [watchdog] {n} callbacks received — audio is flowing")
                return
            last = n
        if mic.callbacks_received == 0:
            import os

            print(
                "  !!! [watchdog] FATAL: no audio after 20s. "
                "Exiting so the launcher can detect failure."
            )
            os._exit(2)

    threading.Thread(target=_run, daemon=True).start()


def run(cfg: RuntimeConfig) -> None:
    print(f"Backend: {cfg.backend}")
    print(
        f"Deterrent: {cfg.deterrent_seconds:.1f}s pulsed in "
        f"{TONE_FREQ_LOW_HZ/1000:g}–{TONE_FREQ_HIGH_HZ/1000:g} kHz"
    )

    backend = make_backend(cfg.backend)
    class_names = backend.class_names
    head = VerifierHead.load()
    rng = np.random.default_rng()

    mic = make_mic_capture(cfg.mic_backend, cfg.mic_device)
    print(f"Mic native rate: {mic.native_sr} Hz → decimating to {SAMPLE_RATE} Hz")
    print(f"Mic backend: {cfg.mic_backend}")

    resample = make_resampler(mic.native_sr, SAMPLE_RATE)

    frame_len_native = int(round(mic.native_sr * FRAME_SECONDS))
    ring_native = np.zeros(frame_len_native, dtype=np.float32)

    recorder = HowlRecorder(HOWL_DIR) if cfg.record_howls else None
    if recorder:
        print(f"Recording howls to {HOWL_DIR}/ (pre/post-roll)")

    sustain_count = 0
    last_trigger_ts = 0.0
    next_level_print = time.time()

    print("Listening... (Ctrl+C to stop)")
    mic.start()
    _start_watchdog(mic)
    try:
        while True:
            chunk_native = mic.queue.get()
            ring_native = np.concatenate([ring_native[len(chunk_native):], chunk_native])
            ring = resample(ring_native)
            if ring.shape[0] != FRAME_SAMPLES:
                ring = (
                    ring[-FRAME_SAMPLES:]
                    if ring.shape[0] > FRAME_SAMPLES
                    else np.pad(ring, (FRAME_SAMPLES - ring.shape[0], 0))
                )

            scores, embedding = backend.predict(ring)

            trigger_scores = {n: float(scores[i]) for i, n in TRIGGER_CLASSES.items()}
            log_scores = {n: float(scores[i]) for i, n in LOG_CLASSES.items()}
            trigger_class, trigger_score = max(trigger_scores.items(), key=lambda kv: kv[1])

            # Feed the recorder AFTER scoring so we can drive dynamic post-roll
            # extension off the current cat-class score.
            if recorder is not None:
                recorder.feed_chunk(resample(chunk_native), trigger_score)
            top5_idx = np.argsort(scores)[-5:][::-1]
            top5 = [(class_names[i], float(scores[i])) for i in top5_idx]

            if trigger_score >= TRIGGER_THRESHOLD:
                sustain_count += 1
                print(
                    f"  [{time.strftime('%H:%M:%S')}] {trigger_class}={trigger_score:.2f} "
                    f"top={top5[0][0]} sustain={sustain_count}"
                )
            else:
                sustain_count = 0

            # 30 s heartbeat / level meter.
            now_hb = time.time()
            if now_hb >= next_level_print:
                rms = float(np.sqrt(np.mean(ring.astype(np.float64) ** 2)))
                dbfs = 20 * np.log10(rms + 1e-9)
                peak = float(np.max(np.abs(ring)))
                print(
                    f"  · [{time.strftime('%H:%M:%S')}] "
                    f"mic: rms={dbfs:6.1f} dBFS  peak={peak:.3f}  "
                    f"top1={top5[0][0]} ({top5[0][1]:.2f})"
                )
                next_level_print = now_hb + 30

            now = time.time()
            ready = (
                sustain_count >= SUSTAIN_FRAMES
                and now - last_trigger_ts > COOLDOWN_SECONDS
                and in_quiet_hours()
            )
            if not ready:
                continue

            decision = gate(head, embedding)
            if decision.gated:
                print(
                    f"  ─── GATED on {trigger_class} ({decision.reason} < "
                    f"{HEAD_GATE_THRESHOLD}). Logging, not firing."
                )
            else:
                action = (
                    "LOGGING ONLY" if cfg.log_only
                    else f"playing {cfg.deterrent_seconds:.1f}s "
                         f"{TONE_FREQ_LOW_HZ/1000:g}-{TONE_FREQ_HIGH_HZ/1000:g} kHz burst"
                )
                print(f"  >>> TRIGGER on {trigger_class} ({decision.reason}) ({action}).")

            save_trigger(Trigger(
                timestamp=now,
                trigger_score=trigger_score,
                trigger_class=trigger_class,
                trigger_scores=trigger_scores,
                log_scores=log_scores,
                context=decision.context,
                top_classes=top5,
                audio=ring.copy(),
            ))
            if recorder is not None:
                recorder.start_capture(now, trigger_class, decision.gated)
            if not decision.gated and not cfg.log_only:
                tone = build_tone(cfg.deterrent_seconds, rng=rng)
                threading.Thread(target=play_tone, args=(tone,), daemon=True).start()
            last_trigger_ts = now
            sustain_count = 0
    finally:
        mic.stop()
