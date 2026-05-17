"""
Cat howl detector + near-ultrasonic deterrent.

Listens to the default mic, runs YAMNet on a rolling window, and when cat
vocalizations cross a threshold, plays a pulsed ~18 kHz tone through the
default output (inaudible to most adults, very audible to cats).

Every trigger is logged to ./logs/ as a WAV snippet + JSON of class scores,
so you can review false positives/negatives and tune the threshold.

Backends (set via BACKEND env var):
    coreml  — Apple Neural Engine via Core ML  (Mac M-series, fastest + lowest power)
    mps     — Apple GPU via tensorflow-metal     (Mac M-series, no Core ML conversion)
    cuda    — NVIDIA GPU via TensorFlow         (L4 / any CUDA GPU)
    cpu     — CPU via TensorFlow                (default, works everywhere)

Setup:
    pip install sounddevice soundfile numpy
    # then ONE of:
    pip install tensorflow tensorflow-hub                 # cpu / cuda
    pip install tensorflow-macos tensorflow-metal tensorflow-hub  # mps
    pip install coremltools                               # coreml (+ run convert_yamnet_coreml.py once)

Run:
    BACKEND=coreml python cat_howl_deterrent.py
    BACKEND=cpu     python cat_howl_deterrent.py
    BACKEND=cuda    python cat_howl_deterrent.py
    BACKEND=mps     python cat_howl_deterrent.py

    LOG_ONLY=1 BACKEND=coreml python cat_howl_deterrent.py   # detect + log, no tone

Press Ctrl+C to stop.
"""

import csv
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

BACKEND = os.environ.get("BACKEND", "cpu").lower()
LOG_ONLY = os.environ.get("LOG_ONLY", "0") not in ("0", "", "false", "False")


# ---------- CONFIG ----------
SAMPLE_RATE = 16_000           # YAMNet requires 16 kHz mono
FRAME_SECONDS = 0.96           # YAMNet frame length
HOP_SECONDS = 0.48             # How often we re-classify

# ─── V1 ARCHITECTURE ───────────────────────────────────────────────────────
# Stage 1 (TRIGGER):  YAMNet raw score in {Cat, Meow, Caterwaul}  ≥ threshold
#                     → fires deterrent. The class with the highest score is
#                       logged as the meow type.
# Stage 2 (LOG-ONLY): Optional fine-tuned head produces a verifier probability
#                     written into the log. Does NOT gate the trigger.
# ───────────────────────────────────────────────────────────────────────────

TRIGGER_THRESHOLD = 0.5        # Max score across TRIGGER_CLASSES
SUSTAIN_FRAMES = 2             # Must trigger this many frames in a row
COOLDOWN_SECONDS = 30          # Min seconds between deterrent pulses

# Secondary head — verifies that the YAMNet trigger is actually a cat vocalization
# and not a purr/false-positive. Defaults to AudioSet-trained binary head (88% acc).
CONTEXT_HEAD_PATH = "models/yamnet_audioset_cats_head.keras"
CONTEXT_LABELS_PATH = "models/yamnet_audioset_cats_labels.json"

# Verifier-head gate: when the head is loaded, the trigger fires only if
# head_p >= HEAD_GATE_THRESHOLD. Prevents 77%-of-purrs false-firing seen in sims.
HEAD_GATE_THRESHOLD = 0.5
# If the backend doesn't expose embeddings (e.g., CoreML), the gate is bypassed
# and the trigger uses YAMNet scores only. Set to True to *block* triggers when
# the head isn't available (safer; but means CoreML never fires).
REQUIRE_HEAD_IF_AVAILABLE = True

# Deterrent tone — randomized pulsing pattern in 22–24 kHz band.
# NOTE: 22–24 kHz is ABOVE the Kanto ORA's rated 22 kHz roll-off, so output
# will be attenuated. Cats still hear it; you may need to raise TONE_VOLUME.
TONE_FREQ_LOW_HZ = 22_000      # Lower bound of randomized frequency band
TONE_FREQ_HIGH_HZ = 24_000     # Upper bound
TONE_PULSE_MS = 150            # One pulse length
TONE_GAP_MS = 100              # Silence between pulses
TONE_VOLUME = 0.4              # 0.0 - 1.0. Higher than before because 22+ kHz rolls off.
TONE_SAMPLE_RATE = 48_000      # Output rate (Nyquist 24 kHz — exactly our top)
DETERRENT_SECONDS = float(os.environ.get("DETERRENT_SECONDS", "3.0"))

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SAVE_CLIPS = True              # Save 3s WAV of each trigger for review

# Quiet-hours filter — set to None to always be active
QUIET_HOURS_START = None       # e.g., 0  (midnight)
QUIET_HOURS_END = None         # e.g., 7  (7 AM)
# ----------------------------


# Classes that gate the deterrent trigger (any one crossing threshold fires it)
TRIGGER_CLASSES = {
    76: "Cat",
    78: "Meow",
    80: "Caterwaul",
}
# Extra classes recorded in logs but NOT used for gating (Purr/Hiss/Growling/etc.)
LOG_CLASSES = {
    77: "Purr",
    79: "Hiss",
    81: "Growling",
    82: "Bow-wow",      # Dog — useful to spot mic confusion
    83: "Yip",
}

FRAME_SAMPLES = 15600         # 0.96 s @ 16 kHz


def _load_class_names_via_tfhub(model):
    import tensorflow as tf
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path) as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def _load_class_names_from_disk():
    p = Path("yamnet_classes.json")
    if p.exists():
        return json.loads(p.read_text())
    return None


class TFBackend:
    """TF-based backend. device='/CPU:0', '/GPU:0' (cuda or mps via tensorflow-metal)."""
    def __init__(self, device: str):
        import tensorflow as tf
        import tensorflow_hub as hub
        self.tf = tf
        self.device = device
        if device == "/CPU:0":
            tf.config.set_visible_devices([], "GPU")
        print(f"Loading YAMNet on {device}...")
        with tf.device(device):
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.class_names = _load_class_names_from_disk() or _load_class_names_via_tfhub(self.model)

    def predict(self, waveform: np.ndarray):
        with self.tf.device(self.device):
            scores, embeddings, _ = self.model(waveform)
        return scores.numpy().mean(axis=0), embeddings.numpy().mean(axis=0)


class CoreMLBackend:
    """Apple Neural Engine via Core ML. Run convert_yamnet_coreml.py once first."""
    def __init__(self, model_path: str = "yamnet.mlpackage"):
        import coremltools as ct
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python convert_yamnet_coreml.py` first."
            )
        print(f"Loading Core ML model from {model_path}...")
        self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        self.class_names = _load_class_names_from_disk()
        if self.class_names is None:
            raise FileNotFoundError("yamnet_classes.json missing; rerun the converter.")

    def predict(self, waveform: np.ndarray):
        out = self.model.predict({"waveform": waveform.astype(np.float32)})
        scores = None
        embedding = None
        for v in out.values():
            arr = np.asarray(v).reshape(-1)
            if arr.shape[0] == 521:
                scores = arr
            elif arr.shape[0] == 1024:
                embedding = arr
        return scores, embedding


def make_backend():
    if BACKEND == "coreml":
        return CoreMLBackend()
    if BACKEND == "cuda":
        return TFBackend("/GPU:0")
    if BACKEND == "mps":
        return TFBackend("/GPU:0")   # tensorflow-metal exposes Apple GPU as /GPU:0
    if BACKEND == "cpu":
        return TFBackend("/CPU:0")
    raise ValueError(f"Unknown BACKEND={BACKEND!r}. Use coreml | mps | cuda | cpu.")


def in_quiet_hours():
    if QUIET_HOURS_START is None or QUIET_HOURS_END is None:
        return True
    hour = time.localtime().tm_hour
    if QUIET_HOURS_START <= QUIET_HOURS_END:
        return QUIET_HOURS_START <= hour < QUIET_HOURS_END
    return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END


def build_deterrent_tone(duration_s: float = None, rng: np.random.Generator = None):
    """Generate a randomized pulsed ultrasonic burst of length ~duration_s.

    Each pulse is a sine at a random frequency in [TONE_FREQ_LOW_HZ, TONE_FREQ_HIGH_HZ].
    Pulses and gaps repeat until total duration is reached. Re-randomized on every
    call so the cat can't habituate to a fixed pattern.
    """
    if duration_s is None:
        duration_s = DETERRENT_SECONDS
    if rng is None:
        rng = np.random.default_rng()

    pulse_samples = int(TONE_SAMPLE_RATE * TONE_PULSE_MS / 1000)
    gap_samples = int(TONE_SAMPLE_RATE * TONE_GAP_MS / 1000)
    target_samples = int(TONE_SAMPLE_RATE * duration_s)
    fade = int(0.01 * TONE_SAMPLE_RATE)

    out = []
    n = 0
    t = np.arange(pulse_samples) / TONE_SAMPLE_RATE
    envelope = np.ones(pulse_samples, dtype=np.float32)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

    while n < target_samples:
        freq = float(rng.uniform(TONE_FREQ_LOW_HZ, TONE_FREQ_HIGH_HZ))
        pulse = np.sin(2 * np.pi * freq * t).astype(np.float32) * envelope * TONE_VOLUME
        out.append(pulse); n += pulse_samples
        if n >= target_samples:
            break
        out.append(np.zeros(gap_samples, dtype=np.float32)); n += gap_samples

    signal = np.concatenate(out)[:target_samples]
    return signal


def play_deterrent(tone):
    try:
        sd.play(tone, TONE_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"  ! tone playback failed: {e}")


@dataclass
class Trigger:
    timestamp: float
    trigger_score: float
    trigger_class: str
    trigger_scores: dict     # {class_name: score} for TRIGGER_CLASSES
    log_scores: dict         # {class_name: score} for LOG_CLASSES (purr, hiss, etc.)
    context: dict | None     # fine-tuned head output, or None
    top_classes: list
    audio: np.ndarray


def save_trigger(trigger: Trigger):
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(trigger.timestamp))
    if SAVE_CLIPS:
        sf.write(LOG_DIR / f"{ts}.wav", trigger.audio, SAMPLE_RATE)
    with open(LOG_DIR / f"{ts}.json", "w") as f:
        json.dump({
            "timestamp": trigger.timestamp,
            "trigger_class": trigger.trigger_class,
            "trigger_score": float(trigger.trigger_score),
            "trigger_scores": trigger.trigger_scores,
            "log_scores": trigger.log_scores,
            "context": trigger.context,
            "top_classes": trigger.top_classes,
        }, f, indent=2)


def load_context_head():
    head_p, labels_p = Path(CONTEXT_HEAD_PATH), Path(CONTEXT_LABELS_PATH)
    if not head_p.exists() or not labels_p.exists():
        return None, None
    try:
        import tensorflow as tf
        head = tf.keras.models.load_model(head_p)
        labels = json.loads(labels_p.read_text())
        print(f"Loaded context head: {labels}")
        return head, labels
    except Exception as e:
        print(f"  ! context head load failed: {e}")
        return None, None


def main():
    print(f"Backend: {BACKEND}")
    print(f"Deterrent: {DETERRENT_SECONDS:.1f}s pulsed in "
          f"{TONE_FREQ_LOW_HZ/1000:g}–{TONE_FREQ_HIGH_HZ/1000:g} kHz")
    backend = make_backend()
    class_names = backend.class_names
    context_head, context_labels = load_context_head()
    rng = np.random.default_rng()

    frame_len = FRAME_SAMPLES
    hop_len = int(SAMPLE_RATE * HOP_SECONDS)
    ring = np.zeros(frame_len, dtype=np.float32)

    audio_q = queue.Queue()

    def mic_cb(indata, frames, time_info, status):
        if status:
            print(f"  ! mic status: {status}")
        audio_q.put(indata[:, 0].copy())

    sustain_count = 0
    last_trigger_ts = 0.0

    print("Listening... (Ctrl+C to stop)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=hop_len,
        callback=mic_cb,
    ):
        while True:
            chunk = audio_q.get()
            # Slide window
            ring = np.concatenate([ring[len(chunk):], chunk])

            # Classify
            scores, embedding = backend.predict(ring)

            # Trigger metric: MAX score over the trigger-eligible cat classes
            trigger_scores = {name: float(scores[i]) for i, name in TRIGGER_CLASSES.items()}
            log_scores = {name: float(scores[i]) for i, name in LOG_CLASSES.items()}
            trigger_class, trigger_score = max(trigger_scores.items(), key=lambda kv: kv[1])
            top5_idx = np.argsort(scores)[-5:][::-1]
            top5 = [(class_names[i], float(scores[i])) for i in top5_idx]

            if trigger_score >= TRIGGER_THRESHOLD:
                sustain_count += 1
                print(f"  [{time.strftime('%H:%M:%S')}] {trigger_class}={trigger_score:.2f} "
                      f"top={top5[0][0]} sustain={sustain_count}")
            else:
                sustain_count = 0

            now = time.time()
            ready = (
                sustain_count >= SUSTAIN_FRAMES
                and now - last_trigger_ts > COOLDOWN_SECONDS
                and in_quiet_hours()
            )
            if ready:
                # ── Stage 2: verifier-head gate ─────────────────────────────
                # Binary head outputs 1 logit → sigmoid → P(cat-vocal).
                # Multi-class head: take prob of the "cat-vocal" label if present,
                # else max non-"non-cat" prob.
                context = None
                head_prob = None
                if context_head is not None and embedding is not None:
                    import tensorflow as tf
                    logits = context_head.predict(embedding[None, :], verbose=0)[0]
                    if logits.shape[-1] == 1:
                        head_prob = float(tf.sigmoid(logits).numpy()[0])
                        context = {"cat-vocal": head_prob, "non-cat": 1.0 - head_prob}
                    else:
                        probs = tf.nn.softmax(logits).numpy()
                        context = {context_labels[i]: float(probs[i]) for i in range(len(probs))}
                        if "cat-vocal" in context:
                            head_prob = context["cat-vocal"]
                        else:
                            head_prob = max(v for k, v in context.items() if k != "non-cat")

                # ── Gate decision ───────────────────────────────────────────
                if context_head is not None and embedding is not None:
                    gated = head_prob < HEAD_GATE_THRESHOLD
                    gate_str = f"head_p={head_prob:.2f}"
                elif context_head is not None and embedding is None:
                    # Head loaded but backend can't supply embeddings (CoreML)
                    gated = REQUIRE_HEAD_IF_AVAILABLE
                    gate_str = "head=unavailable(backend)"
                else:
                    gated = False
                    gate_str = "no-head"

                if gated:
                    print(f"  ─── GATED on {trigger_class} ({gate_str} < "
                          f"{HEAD_GATE_THRESHOLD}). Logging, not firing.")
                else:
                    action = ("LOGGING ONLY" if LOG_ONLY
                              else f"playing {DETERRENT_SECONDS:.1f}s "
                                   f"{TONE_FREQ_LOW_HZ/1000:g}-{TONE_FREQ_HIGH_HZ/1000:g} kHz burst")
                    print(f"  >>> TRIGGER on {trigger_class} ({gate_str}) ({action}).")

                save_trigger(Trigger(
                    timestamp=now,
                    trigger_score=trigger_score,
                    trigger_class=trigger_class,
                    trigger_scores=trigger_scores,
                    log_scores=log_scores,
                    context=context,
                    top_classes=top5,
                    audio=ring.copy(),
                ))
                if not gated and not LOG_ONLY:
                    tone = build_deterrent_tone(rng=rng)
                    threading.Thread(target=play_deterrent, args=(tone,), daemon=True).start()
                last_trigger_ts = now
                sustain_count = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
