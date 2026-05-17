"""
Download non-cat AudioSet clips for use as 'easy' negatives in fine-tuning.
NO tensorflow — pure datasets + librosa to avoid pyarrow/TF GIL conflict.
"""
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

CAT_MIDS = {"/m/01yrx", "/m/02yds9", "/m/07qrkrw", "/m/07rjwbb", "/m/07r81j2"}
NEG_TARGET = 800
NEG_DIR = Path("data/audioset_cats/_negatives")
NEG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    have = list(NEG_DIR.glob("*.wav"))
    if len(have) >= NEG_TARGET:
        print(f"Already have {len(have)} negatives.")
        return

    print(f"Need {NEG_TARGET - len(have)} more negatives. Streaming AudioSet...")
    ds = load_dataset("agkphysics/AudioSet", "balanced", split="train", streaming=True)
    pbar = tqdm(total=NEG_TARGET - len(have))
    for sample in ds:
        if len(have) >= NEG_TARGET:
            break
        if set(sample["labels"]) & CAT_MIDS:
            continue
        vid = sample["video_id"]
        out_path = NEG_DIR / f"{vid}.wav"
        if out_path.exists():
            continue
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(out_path, audio.astype("float32"), 16000)
        have.append(out_path)
        pbar.update(1)
    pbar.close()
    print(f"Total negatives on disk: {len(have)}")


if __name__ == "__main__":
    main()
