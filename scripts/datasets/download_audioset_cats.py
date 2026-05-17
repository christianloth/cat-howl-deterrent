"""
Stream AudioSet from HuggingFace and save clips that match any cat-related class.

Output:
    data/audioset_cats/<label>/<video_id>.wav   (16 kHz mono)
    data/audioset_cats/manifest.csv             (one row per clip + labels)
"""
import csv
import os
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

CAT_MIDS = {
    "/m/01yrx":   "Cat",
    "/m/02yds9":  "Purr",
    "/m/07qrkrw": "Meow",
    "/m/07rjwbb": "Hiss",
    "/m/07r81j2": "Caterwaul",
}

OUT = Path("data/audioset_cats")
OUT.mkdir(parents=True, exist_ok=True)
for label in CAT_MIDS.values():
    (OUT / label).mkdir(exist_ok=True)

MANIFEST = OUT / "manifest.csv"


def stream_and_save(config: str, split: str, max_clips: int | None = None):
    print(f"\n=== Streaming AudioSet/{config}/{split} ===")
    ds = load_dataset("agkphysics/AudioSet", config, split=split, streaming=True)

    n_seen = 0
    saved = {label: 0 for label in CAT_MIDS.values()}
    rows = []

    with tqdm(unit=" clips") as pbar:
        for sample in ds:
            n_seen += 1
            pbar.update(1)
            mids = set(sample["labels"])
            hit = mids & CAT_MIDS.keys()
            if not hit:
                continue

            primary_label = CAT_MIDS[max(hit, key=lambda m: list(CAT_MIDS).index(m) if m in CAT_MIDS else 999)]
            # Prefer Caterwaul > Meow > Hiss > Purr > Cat in priority
            for mid in ["/m/07r81j2", "/m/07qrkrw", "/m/07rjwbb", "/m/02yds9", "/m/01yrx"]:
                if mid in hit:
                    primary_label = CAT_MIDS[mid]
                    break

            vid = sample["video_id"]
            out_path = OUT / primary_label / f"{vid}.wav"
            if out_path.exists():
                continue

            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sf.write(out_path, audio.astype("float32"), 16000)
            saved[primary_label] += 1
            rows.append({
                "video_id": vid,
                "primary_label": primary_label,
                "all_labels": "|".join(sample["human_labels"]),
                "split": f"{config}/{split}",
                "path": str(out_path),
            })
            pbar.set_postfix(saved)

            if max_clips and sum(saved.values()) >= max_clips:
                break

    print(f"Scanned {n_seen} clips. Saved per class: {saved}")
    return rows


def main():
    all_rows = []
    # Just balanced for now (~22k clips, manageable). Add unbalanced later if needed.
    all_rows += stream_and_save("balanced", "train", max_clips=None)
    all_rows += stream_and_save("balanced", "test", max_clips=None)

    # Write manifest
    if all_rows:
        with open(MANIFEST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nManifest: {MANIFEST}  ({len(all_rows)} clips)")


if __name__ == "__main__":
    main()
