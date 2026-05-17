"""
Simulation of the v1 detection + deterrent logic against AudioSet clips.

Replicates EXACTLY the trigger rule from cat_howl_deterrent.py:
   trigger_score = max(scores[Cat], scores[Meow], scores[Caterwaul])
   triggered     = trigger_score >= TRIGGER_THRESHOLD  (on at least one 0.96 s frame)

Reports per-class trigger rates and writes one deterrent-tone WAV to disk so
you can audit what the burst sounds like.

Run:
    source .venv/bin/activate
    python simulate_detector.py
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # YAMNet on CPU, matches finetune

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# Mirror constants from cat_howl_deterrent.py
TRIGGER_CLASSES = {76: "Cat", 78: "Meow", 80: "Caterwaul"}
TRIGGER_THRESHOLD = 0.5
HEAD_GATE_THRESHOLD = 0.5    # verifier-head gate

TONE_FREQ_LOW_HZ = 22_000
TONE_FREQ_HIGH_HZ = 24_000
TONE_PULSE_MS = 150
TONE_GAP_MS = 100
TONE_VOLUME = 0.4
TONE_SAMPLE_RATE = 48_000
DETERRENT_SECONDS = float(os.environ.get("DETERRENT_SECONDS", "3.0"))

# Stage 2 head (log only)
HEAD_PATH = Path("models/yamnet_audioset_cats_head.keras")
HEAD_LABELS = Path("models/yamnet_audioset_cats_labels.json")


def build_deterrent_tone(duration_s: float, rng: np.random.Generator):
    pulse_samples = int(TONE_SAMPLE_RATE * TONE_PULSE_MS / 1000)
    gap_samples = int(TONE_SAMPLE_RATE * TONE_GAP_MS / 1000)
    target_samples = int(TONE_SAMPLE_RATE * duration_s)
    fade = int(0.01 * TONE_SAMPLE_RATE)
    envelope = np.ones(pulse_samples, dtype=np.float32)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    t = np.arange(pulse_samples) / TONE_SAMPLE_RATE

    out, n, freqs = [], 0, []
    while n < target_samples:
        f = float(rng.uniform(TONE_FREQ_LOW_HZ, TONE_FREQ_HIGH_HZ))
        freqs.append(f)
        out.append(np.sin(2 * np.pi * f * t).astype(np.float32) * envelope * TONE_VOLUME)
        n += pulse_samples
        if n >= target_samples:
            break
        out.append(np.zeros(gap_samples, dtype=np.float32)); n += gap_samples
    return np.concatenate(out)[:target_samples], freqs


def score_clip(yamnet, head, wav_path: Path):
    wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    wav = wav.astype(np.float32)
    if len(wav) < 15600:
        wav = np.pad(wav, (0, 15600 - len(wav)))
    scores, embeddings, _ = yamnet(wav)
    scores = scores.numpy()
    per_frame_trigger = np.max(scores[:, list(TRIGGER_CLASSES.keys())], axis=1)
    worst = int(per_frame_trigger.argmax())
    trigger_score = float(per_frame_trigger[worst])
    fired_cls_idx = int(np.argmax([scores[worst][i] for i in TRIGGER_CLASSES]))
    fired_cls = list(TRIGGER_CLASSES.values())[fired_cls_idx]
    head_prob = None
    if head is not None:
        emb = embeddings.numpy().mean(axis=0)[None, :]
        logits = head.predict(emb, verbose=0)[0]
        head_prob = float(tf.sigmoid(logits).numpy()[0])
    return trigger_score, fired_cls, head_prob


def main():
    print("Loading YAMNet (CPU)...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    head = None
    if HEAD_PATH.exists():
        print(f"Loading verifier head: {HEAD_PATH}")
        head = tf.keras.models.load_model(HEAD_PATH)

    # ---- Build & save one deterrent burst for audit ----
    rng = np.random.default_rng(seed=42)
    tone, freqs = build_deterrent_tone(DETERRENT_SECONDS, rng)
    burst_path = Path("logs/sample_deterrent_burst.wav")
    burst_path.parent.mkdir(exist_ok=True)
    sf.write(burst_path, tone, TONE_SAMPLE_RATE)
    print(f"\nSample deterrent burst:")
    print(f"  duration:    {len(tone)/TONE_SAMPLE_RATE:.2f}s")
    print(f"  pulses:      {len(freqs)}")
    print(f"  freq range:  {min(freqs)/1000:.2f}–{max(freqs)/1000:.2f} kHz")
    print(f"  saved to:    {burst_path}")

    # ---- Run detector against AudioSet clips ----
    AUDIOSET = Path("data/audioset_cats")
    GROUPS = ["Caterwaul", "Meow", "Hiss", "Cat", "Purr", "_negatives"]

    print("\nScoring AudioSet clips...")
    results = defaultdict(list)
    for grp in GROUPS:
        d = AUDIOSET / grp
        if not d.exists():
            continue
        paths = sorted(d.glob("*.wav"))
        # Cap negatives at 100 — we have 800 and don't need them all
        if grp == "_negatives":
            paths = paths[:100]
        for p in tqdm(paths, desc=grp):
            ts, fc, hp = score_clip(yamnet, head, p)
            results[grp].append((p.name, ts, fc, hp))

    # ---- Report (Stage1-only vs Stage1+head-gate) ----
    print("\n" + "=" * 100)
    print(f"{'Group':<14} {'N':>4} {'S1 only':>10} {'rate':>6}  "
          f"{'S1+gate':>10} {'rate':>6}  {'mean_head_p':>12}")
    print("-" * 100)
    for grp in GROUPS:
        rs = results.get(grp, [])
        if not rs:
            continue
        s1 = [r for r in rs if r[1] >= TRIGGER_THRESHOLD]
        s1_gated = [r for r in s1 if r[3] is not None and r[3] >= HEAD_GATE_THRESHOLD]
        mean_h = np.mean([r[3] for r in rs if r[3] is not None]) if head else float("nan")
        head_str = f"{mean_h:>12.3f}" if head else f"{'n/a':>12}"
        print(f"{grp:<14} {len(rs):>4} {len(s1):>10} {len(s1)/len(rs):>6.0%}  "
              f"{len(s1_gated):>10} {len(s1_gated)/len(rs):>6.0%}  {head_str}")

    # Examples
    print("\n--- A few triggered cat-vocal examples ---")
    for grp in ["Caterwaul", "Meow"]:
        for name, ts, fc, hp in results.get(grp, [])[:5]:
            mark = "TRIGGER" if ts >= TRIGGER_THRESHOLD else "miss   "
            head_str = f" head_p={hp:.2f}" if hp is not None else ""
            print(f"  [{grp:<10}] {mark} score={ts:.2f} fired={fc:<10}{head_str}  {name}")

    print("\n--- A few negatives ---")
    for name, ts, fc, hp in results.get("_negatives", [])[:8]:
        mark = "FALSE-FIRE" if ts >= TRIGGER_THRESHOLD else "ok"
        head_str = f" head_p={hp:.2f}" if hp is not None else ""
        print(f"  [neg]        {mark:<10} score={ts:.2f} fired={fc:<10}{head_str}  {name}")


if __name__ == "__main__":
    main()
