"""
Score every purr clip (Purr/ + Purr_extra/) against the trigger logic and
report Stage1 trigger rate vs Stage1+gate rate.

GPU is attempted first; falls back to CPU if YAMNet/cuDNN errors.
"""
import json
import os
import sys
from pathlib import Path

# Allow opt-out: TEST_PURR_DEVICE=cpu to force CPU
TEST_PURR_DEVICE = os.environ.get("TEST_PURR_DEVICE", "gpu").lower()
if TEST_PURR_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    # cuDNN autotune is the source of the YAMNet 'status 1002' crash. Disable it.
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

TRIGGER_CLASSES = {76: "Cat", 78: "Meow", 80: "Caterwaul"}
TRIGGER_THRESHOLD = 0.5
HEAD_GATE_THRESHOLD = 0.5

PURR_DIRS = [Path("data/audioset_cats/Purr"), Path("data/audioset_cats/Purr_extra")]
HEAD_PATH = Path("models/yamnet_audioset_cats_head.keras")


def try_load_yamnet_on_device():
    """Try GPU first (with autotune off); fall back to CPU if it crashes."""
    if TEST_PURR_DEVICE == "cpu":
        print("FORCED CPU (TEST_PURR_DEVICE=cpu)")
        return hub.load("https://tfhub.dev/google/yamnet/1"), "cpu"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"Available GPUs: {len(gpus)} — attempting YAMNet on GPU...")
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        # smoke test
        _ = yamnet(np.zeros(15600, dtype=np.float32))
        print("GPU YAMNet works!")
        return yamnet, "gpu"
    except Exception as e:
        print(f"GPU YAMNet failed ({e.__class__.__name__}); falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # need to reload TF context — easiest is to just rerun the script in CPU mode
        return hub.load("https://tfhub.dev/google/yamnet/1"), "cpu-fallback"


def main():
    yamnet, device = try_load_yamnet_on_device()
    head = tf.keras.models.load_model(HEAD_PATH) if HEAD_PATH.exists() else None
    if head is None:
        print(f"ERROR: head not found at {HEAD_PATH}")
        sys.exit(1)

    paths = []
    for d in PURR_DIRS:
        paths += list(d.glob("*.wav"))
    print(f"Scoring {len(paths)} purr clips on {device}...")

    results = []
    for p in tqdm(paths):
        wav, _ = librosa.load(str(p), sr=16000, mono=True)
        wav = wav.astype(np.float32)
        if len(wav) < 15600:
            wav = np.pad(wav, (0, 15600 - len(wav)))
        scores, embeddings, _ = yamnet(wav)
        scores = scores.numpy()
        per_frame_trigger = np.max(scores[:, list(TRIGGER_CLASSES.keys())], axis=1)
        trig = float(per_frame_trigger.max())

        emb = embeddings.numpy().mean(axis=0)[None, :]
        logits = head.predict(emb, verbose=0)[0]
        head_p = float(tf.sigmoid(logits).numpy()[0])

        results.append({"path": str(p), "trigger_score": trig, "head_p": head_p})

    n = len(results)
    s1 = [r for r in results if r["trigger_score"] >= TRIGGER_THRESHOLD]
    s1_gated = [r for r in s1 if r["head_p"] >= HEAD_GATE_THRESHOLD]
    mean_score = float(np.mean([r["trigger_score"] for r in results]))
    mean_headp = float(np.mean([r["head_p"] for r in results]))

    print("\n" + "=" * 70)
    print(f"  Total purr clips:        {n}")
    print(f"  Stage1 trigger rate:     {len(s1)}/{n} = {len(s1)/n:.1%}")
    print(f"  Stage1 + gate rate:      {len(s1_gated)}/{n} = {len(s1_gated)/n:.1%}")
    print(f"  Mean YAMNet trig-score:  {mean_score:.3f}")
    print(f"  Mean verifier head_p:    {mean_headp:.3f}")
    print(f"  Device:                  {device}")

    out_json = Path("logs/purr_suppression_results.json")
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps({
        "n_clips": n,
        "stage1_rate": len(s1) / n,
        "stage1_gated_rate": len(s1_gated) / n,
        "mean_trigger_score": mean_score,
        "mean_head_p": mean_headp,
        "device": device,
        "results": results,
    }, indent=2))
    print(f"  Saved details:           {out_json}")


if __name__ == "__main__":
    main()
