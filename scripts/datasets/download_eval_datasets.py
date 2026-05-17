"""
Pull a wide variety of cat-sound and non-cat datasets for v1 evaluation.
NO tensorflow — pure datasets/curl to avoid pyarrow/TF crash.

Sources:
 - AudioSet 'unbalanced' subset: pull more Cat, Meow, Caterwaul, Hiss, Purr clips
   (we only had 'balanced' before) and pull human-distress confusables
   (Baby cry, Crying, Wail, Baby laughter, Screaming, Whimpering)
 - ESC-50 GitHub: 50 classes incl. 'cat', 'dog', 'siren', 'crying baby', etc.
"""
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path("data/eval")
ROOT.mkdir(parents=True, exist_ok=True)

# ── AudioSet target classes for evaluation ─────────────────────────────────
AUDIOSET_TARGETS = {
    # cat classes
    "/m/01yrx":   ("cat__Cat",         200),
    "/m/02yds9":  ("cat__Purr",        300),
    "/m/07qrkrw": ("cat__Meow",        200),
    "/m/07rjwbb": ("cat__Hiss",        200),
    "/m/07r81j2": ("cat__Caterwaul",   200),
    # confusable human/animal sounds
    "/t/dd00002": ("hum__BabyCry",      80),
    "/m/0463cq4": ("hum__CryingSobbing", 80),
    "/m/07qw_06": ("hum__WailMoan",     80),
    "/t/dd00001": ("hum__BabyLaughter", 50),
    "/m/03qc9zr": ("hum__Screaming",    50),
    "/t/dd00136": ("hum__WhimperDog",   50),
    # other animals
    "/m/0bt9lr":  ("ani__Dog",          50),
    "/m/01h8n0":  ("ani__BowWow",       30),
}


def stream_audioset_split(config: str, split: str):
    print(f"\n=== AudioSet/{config}/{split} ===")
    ds = load_dataset("agkphysics/AudioSet", config, split=split, streaming=True)
    have = {k: list((ROOT / v[0]).glob("*.wav"))
            for k, v in AUDIOSET_TARGETS.items()}
    for k, v in AUDIOSET_TARGETS.items():
        (ROOT / v[0]).mkdir(exist_ok=True)
    pbar = tqdm(unit=" clips")
    for sample in ds:
        pbar.update(1)
        mids = set(sample["labels"])
        for mid, (folder, cap) in AUDIOSET_TARGETS.items():
            if len(have[mid]) >= cap:
                continue
            if mid not in mids:
                continue
            out_path = ROOT / folder / f"{sample['video_id']}.wav"
            if out_path.exists():
                continue
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sf.write(out_path, audio.astype("float32"), 16000)
            have[mid].append(out_path)
            pbar.set_postfix({v[0].split("__")[-1]: len(have[k])
                              for k, v in list(AUDIOSET_TARGETS.items())[:6]})
            break
        if all(len(have[k]) >= v[1] for k, v in AUDIOSET_TARGETS.items()):
            break
    pbar.close()
    print({AUDIOSET_TARGETS[k][0]: len(v) for k, v in have.items()})


def fetch_esc50():
    """Clone ESC-50 from GitHub. ~600 MB, but just metadata + 5s WAVs."""
    esc_root = ROOT / "esc50_raw"
    out_root = ROOT / "esc50"
    if any(out_root.glob("*/*.wav")):
        print(f"ESC-50 already organized at {out_root}")
        return

    if not esc_root.exists():
        print("Cloning ESC-50...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/karolpiczak/ESC-50.git", str(esc_root)],
            check=True,
        )

    # Read meta CSV, copy each file into a per-class folder
    import csv
    meta = esc_root / "meta" / "esc50.csv"
    out_root.mkdir(exist_ok=True)
    with open(meta) as f:
        for row in csv.DictReader(f):
            cat = f"esc50__{row['category']}"
            dst = out_root / cat
            dst.mkdir(exist_ok=True)
            src = esc_root / "audio" / row["filename"]
            if not src.exists():
                continue
            tgt = dst / row["filename"]
            if not tgt.exists():
                shutil.copy(src, tgt)

    # Optionally resample to 16k mono (ESC-50 is 44.1k stereo)
    print("Resampling ESC-50 to 16 kHz mono...")
    for wav in tqdm(list(out_root.glob("*/*.wav"))):
        y, sr = librosa.load(str(wav), sr=16000, mono=True)
        sf.write(wav, y.astype("float32"), 16000)
    print(f"ESC-50 organized at {out_root}")


def main():
    # Pull from balanced first (small, already have cache), then unbalanced for more
    stream_audioset_split("balanced", "train")
    stream_audioset_split("unbalanced", "train")
    fetch_esc50()
    print("\nDone.")


if __name__ == "__main__":
    main()
