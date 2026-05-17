"""
Download baby-cry / human-distress clips from AudioSet for false-positive testing.
NO tensorflow import — pure datasets + librosa to avoid pyarrow/TF GIL conflict.
"""
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

TEST_MIDS = {
    "/t/dd00002":  "Baby cry, infant cry",
    "/m/0463cq4":  "Crying, sobbing",
    "/m/07qw_06":  "Wail, moan",
    "/t/dd00001":  "Baby laughter",
}
CAT_MIDS = {"/m/01yrx", "/m/02yds9", "/m/07qrkrw", "/m/07rjwbb", "/m/07r81j2"}
PER_CLASS = 25
OUT = Path("data/audioset_babycry_test")

for label in TEST_MIDS.values():
    (OUT / label).mkdir(parents=True, exist_ok=True)


def main():
    have = {label: list((OUT / label).glob("*.wav")) for label in TEST_MIDS.values()}
    needed = {k: max(0, PER_CLASS - len(v)) for k, v in have.items()}
    print(f"Needed: {needed}")
    if sum(needed.values()) == 0:
        print("All cached.")
        return

    ds = load_dataset("agkphysics/AudioSet", "balanced", split="train", streaming=True)
    pbar = tqdm(unit=" clips")
    for sample in ds:
        pbar.update(1)
        mids = set(sample["labels"])
        if mids & CAT_MIDS:
            continue
        hit = mids & TEST_MIDS.keys()
        if not hit:
            continue
        label = TEST_MIDS[next(iter(hit))]
        if len(have[label]) >= PER_CLASS:
            continue
        out_path = OUT / label / f"{sample['video_id']}.wav"
        if out_path.exists():
            continue
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(out_path, audio.astype("float32"), 16000)
        have[label].append(out_path)
        pbar.set_postfix({k: len(v) for k, v in have.items()})
        if all(len(have[k]) >= PER_CLASS for k in TEST_MIDS.values()):
            break
    pbar.close()
    print({k: len(v) for k, v in have.items()})


if __name__ == "__main__":
    main()
