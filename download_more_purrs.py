"""
Stream MANY more purr clips from AudioSet's unbalanced split (~2M clips total).
NO tensorflow — pure datasets to avoid pyarrow/TF crash.
"""
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

PURR_MID = "/m/02yds9"
PURR_DIR = Path("data/audioset_cats/Purr_extra")
PURR_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 500


def main():
    have = list(PURR_DIR.glob("*.wav"))
    if len(have) >= TARGET:
        print(f"Already have {len(have)} extra purrs.")
        return

    print(f"Streaming AudioSet unbalanced/train for {TARGET - len(have)} more purrs...")
    ds = load_dataset("agkphysics/AudioSet", "unbalanced", split="train", streaming=True)
    pbar = tqdm(unit=" clips", desc="scanning")
    for sample in ds:
        pbar.update(1)
        if PURR_MID not in sample["labels"]:
            continue
        vid = sample["video_id"]
        out_path = PURR_DIR / f"{vid}.wav"
        if out_path.exists():
            continue
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(out_path, audio.astype("float32"), 16000)
        have.append(out_path)
        pbar.set_postfix(purrs=len(have))
        if len(have) >= TARGET:
            break
    pbar.close()
    print(f"\nTotal purrs in Purr_extra/: {len(have)}")


if __name__ == "__main__":
    main()
