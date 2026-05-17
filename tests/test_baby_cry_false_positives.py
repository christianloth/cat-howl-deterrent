"""
Test: do baby cries (and related human sounds) false-trigger our cat detector?

For each test clip we compute the EXACT trigger metric the live script uses:
    trigger_score = max(scores[Cat], scores[Meow], scores[Caterwaul])
Then we check whether it crosses TRIGGER_THRESHOLD=0.5.

This streams a fresh batch of "Baby cry", "Crying/sobbing", "Wail/moan", and
"Baby laughter" clips from AudioSet and reports the false-trigger rate.
"""
import csv
from pathlib import Path

import librosa
import numpy as np
import tensorflow_hub as hub

# Labels (folders under data/audioset_babycry_test/) — populated by
# download_babycry_clips.py beforehand. No streaming here to avoid TF/pyarrow conflict.
TEST_LABELS = [
    "Baby cry, infant cry",
    "Crying, sobbing",
    "Wail, moan",
    "Baby laughter",
]

TRIGGER_CLASSES = {76: "Cat", 78: "Meow", 80: "Caterwaul"}
TRIGGER_THRESHOLD = 0.5
OUT = Path("data/audioset_babycry_test")


def load_clips():
    have = {}
    for label in TEST_LABELS:
        have[label] = sorted((OUT / label).glob("*.wav"))
        print(f"  {label}: {len(have[label])} clips")
    return have


def score_clip(yamnet, wav_path: Path):
    """Replicate live-script logic: split into 0.96s windows, take MAX over windows
    of max(Cat, Meow, Caterwaul). Return the worst-case trigger score."""
    wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    wav = wav.astype(np.float32)
    if len(wav) < 15600:
        wav = np.pad(wav, (0, 15600 - len(wav)))

    # YAMNet processes any length; we get per-sub-frame scores
    scores, _, _ = yamnet(wav)
    scores = scores.numpy()   # (T, 521)

    # For each sub-frame compute max-over-trigger-classes; then max over frames.
    per_frame_trigger = np.max(scores[:, list(TRIGGER_CLASSES.keys())], axis=1)
    worst_frame_score = per_frame_trigger.max()
    worst_frame_idx = per_frame_trigger.argmax()
    worst_frame = scores[worst_frame_idx]

    # What class actually fired
    fired = {name: float(worst_frame[i]) for i, name in TRIGGER_CLASSES.items()}
    fired_class = max(fired, key=fired.get)

    # Also surface the top non-cat call for context
    top_idx = int(worst_frame.argmax())
    return worst_frame_score, fired_class, fired, top_idx


def main():
    print("Loading clips from disk...")
    have = load_clips()
    print("\nLoading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    # Get class names for nicer printout
    import tensorflow as tf
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path) as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]

    print(f"\nTrigger threshold = {TRIGGER_THRESHOLD}")
    print("=" * 90)

    overall = []
    for label, paths in have.items():
        results = []
        for p in paths:
            score, fired_class, fired, top_idx = score_clip(yamnet, p)
            results.append((p, score, fired_class, fired, top_idx))

        scores = np.array([r[1] for r in results])
        false_trig = (scores >= TRIGGER_THRESHOLD).sum()
        print(f"\n[{label}]  n={len(results)}  "
              f"false-trigger-rate = {false_trig}/{len(results)} ({false_trig/len(results):.0%})")
        print(f"  trigger_score:  min={scores.min():.2f}  median={np.median(scores):.2f}  "
              f"max={scores.max():.2f}")

        # Show the worst offenders
        worst = sorted(results, key=lambda r: -r[1])[:5]
        for p, s, fc, fired, top_idx in worst:
            top_name = class_names[top_idx]
            marker = "TRIGGER" if s >= TRIGGER_THRESHOLD else "ok"
            print(f"    {marker:<8} {s:.2f}  on {fc}({fired[fc]:.2f})  "
                  f"top-overall={top_name!r}  ({p.name})")

        overall.extend(results)

    total = len(overall)
    tot_fp = sum(1 for r in overall if r[1] >= TRIGGER_THRESHOLD)
    print("\n" + "=" * 90)
    print(f"OVERALL false trigger rate on baby/cry/wail clips: {tot_fp}/{total} ({tot_fp/total:.0%})")


if __name__ == "__main__":
    main()
