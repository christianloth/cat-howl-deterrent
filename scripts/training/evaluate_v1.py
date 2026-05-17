"""
Full v1 evaluation against many datasets.

For every clip folder under data/eval/, compute:
  - Stage1 trigger rate (max YAMNet score on Cat/Meow/Caterwaul ≥ 0.5)
  - Stage1 + verifier-gate rate (also requires head_p ≥ 0.5)
  - mean YAMNet trig score
  - mean verifier head_p

Writes a CSV summary + JSON details.
"""
import csv
import json
import os
from pathlib import Path

# Force CPU for YAMNet — cuDNN issues on these L4s
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

TRIGGER_CLASSES = {76: "Cat", 78: "Meow", 80: "Caterwaul"}
TRIGGER_THRESHOLD = 0.5
HEAD_GATE_THRESHOLD = 0.5

EVAL_ROOT = Path("data/eval")
HEAD_PATH = Path("models/yamnet_audioset_cats_head.keras")
OUT_DIR = Path("logs/v1_eval")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def score_clip(yamnet, head, wav_path: Path):
    wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
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
    return trig, head_p


def expected(group: str) -> str:
    """Return 'cat-vocal', 'cat-non-vocal', or 'non-cat' — what the gate SHOULD say."""
    if group.startswith("cat__"):
        sub = group.split("__")[-1]
        return "cat-non-vocal" if sub == "Purr" else "cat-vocal"
    return "non-cat"


def main():
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    head = tf.keras.models.load_model(HEAD_PATH)

    # Discover all evaluation folders (excluding raw esc50 dump)
    folders = sorted([p for p in EVAL_ROOT.glob("*") if p.is_dir()
                      and p.name not in ("esc50_raw",)])
    # ESC-50 has nested per-class folders
    extras = []
    if (EVAL_ROOT / "esc50").exists():
        extras = sorted([p for p in (EVAL_ROOT / "esc50").glob("*") if p.is_dir()])
    folders = [f for f in folders if f.name != "esc50"] + extras

    rows = []
    all_results = {}

    for folder in folders:
        clips = sorted(folder.glob("*.wav"))
        if not clips:
            continue
        results = []
        for p in tqdm(clips, desc=folder.name):
            trig, head_p = score_clip(yamnet, head, p)
            results.append((p.name, trig, head_p))
        n = len(results)
        s1 = sum(1 for _, t, _ in results if t >= TRIGGER_THRESHOLD)
        s1_g = sum(1 for _, t, h in results
                   if t >= TRIGGER_THRESHOLD and h >= HEAD_GATE_THRESHOLD)
        mean_t = float(np.mean([r[1] for r in results]))
        mean_h = float(np.mean([r[2] for r in results]))
        rows.append({
            "group": folder.name,
            "expected": expected(folder.name),
            "n": n,
            "stage1_fire": s1,
            "stage1_rate": s1 / n,
            "gated_fire": s1_g,
            "gated_rate": s1_g / n,
            "mean_trigger_score": round(mean_t, 3),
            "mean_head_p": round(mean_h, 3),
        })
        all_results[folder.name] = results

    # Write outputs
    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json_path = OUT_DIR / "details.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    # Print summary
    print("\n" + "=" * 115)
    print(f"{'Group':<32} {'Expected':<14} {'N':>4} {'S1':>5} {'rate':>6}  "
          f"{'S1+gate':>8} {'rate':>6}  {'trig':>5} {'head':>5}")
    print("-" * 115)
    for r in sorted(rows, key=lambda r: (r["expected"], -r["gated_rate"])):
        print(f"{r['group']:<32} {r['expected']:<14} {r['n']:>4} "
              f"{r['stage1_fire']:>5} {r['stage1_rate']:>6.0%}  "
              f"{r['gated_fire']:>8} {r['gated_rate']:>6.0%}  "
              f"{r['mean_trigger_score']:>5.2f} {r['mean_head_p']:>5.2f}")

    # Aggregated metrics
    cat_vocal = [r for r in rows if r["expected"] == "cat-vocal"]
    non_cat = [r for r in rows if r["expected"] == "non-cat"]
    cat_nv = [r for r in rows if r["expected"] == "cat-non-vocal"]

    def agg(group):
        if not group:
            return None
        n = sum(r["n"] for r in group)
        gated = sum(r["gated_fire"] for r in group)
        return gated, n, gated / n

    print("\nAGGREGATES (after gate):")
    for label, grp in [("Cat-vocal (target — want HIGH)", cat_vocal),
                       ("Cat-non-vocal (Purr — want LOW)", cat_nv),
                       ("Non-cat (want LOW)", non_cat)]:
        r = agg(grp)
        if r:
            print(f"  {label:<40} {r[0]:>5}/{r[1]:<5} = {r[2]:.1%}")

    print(f"\nSaved: {csv_path}")
    print(f"       {json_path}")


if __name__ == "__main__":
    main()
