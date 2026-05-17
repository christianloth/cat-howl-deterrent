# cat-howl-deterrent

24/7 cat-howl detector with a near-ultrasonic deterrent. Listens to the mic,
runs YAMNet for general cat-sound detection, gates on a fine-tuned verifier
head to suppress purr false-positives, and plays a randomized 22–24 kHz
pulsed burst through the speakers when an actual howl is detected.

## Architecture

```
        mic ──► YAMNet ──┬──► 521 class scores ──► max(Cat,Meow,Caterwaul) ≥ 0.5
                         │                                  │ Stage 1
                         │                                  ▼
                         └──► 1024-d embedding ──► head ──► P(cat-vocal) ≥ 0.5
                                                              │ Stage 2 (gate)
                                                              ▼
                                                  randomized 22–24 kHz, 3 s burst
```

Stage 1 (Google's YAMNet) catches anything cat-shaped — including purrs.
Stage 2 (custom 1024 → 128 → 1 head) vetoes purrs and other false positives
based on the embedding.

## Setup (Mac)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt           # see below

# Copy env template
cp .env.example .env                      # then edit if you like

# One-time: convert YAMNet to a CoreML model that runs on the Neural Engine
python convert_yamnet_coreml.py
```

`requirements.txt` is intentionally not committed yet; install:

```bash
pip install tensorflow tensorflow-hub sounddevice soundfile numpy librosa coremltools tqdm scikit-learn
```

For Mac M-series GPU instead of Neural Engine:
```bash
pip install tensorflow-macos tensorflow-metal tensorflow-hub sounddevice soundfile numpy
```

## Running

```bash
# Dry run — detect and log, no tone (recommended first night)
LOG_ONLY=1 BACKEND=coreml python cat_howl_deterrent.py

# Live
BACKEND=coreml python cat_howl_deterrent.py
```

Triggers are written to `./logs/<timestamp>.{json,wav}` for offline review
regardless of `LOG_ONLY`.

## Environment variables

See [`.env.example`](.env.example).

| Var | Default | Meaning |
|---|---|---|
| `BACKEND` | `cpu` | `coreml` / `mps` / `cuda` / `cpu` |
| `LOG_ONLY` | `0` | If `1`, no tone is emitted |
| `DETERRENT_SECONDS` | `3.0` | Length of the randomized pulsed burst |

## Files

| File | Role |
|---|---|
| `cat_howl_deterrent.py` | Live detector + deterrent |
| `convert_yamnet_coreml.py` | One-time YAMNet → CoreML conversion (Mac) |
| `download_audioset_cats.py` | Pulls cat clips from AudioSet for training |
| `download_audioset_negatives.py` | Pulls non-cat clips as negatives |
| `download_babycry_clips.py` | Pulls human-distress clips for false-positive testing |
| `download_more_purrs.py` | Adds more purr clips for further evaluation |
| `download_eval_datasets.py` | Pulls a broad evaluation set + ESC-50 |
| `finetune_catmeows.py` | Trains a 3-class context head on the CatMeows dataset |
| `finetune_audioset_cats.py` | Trains the binary cat-vocal verifier head |
| `simulate_detector.py` | Offline simulation of the trigger + gate logic |
| `evaluate_v1.py` | Full evaluation against multiple sound datasets |
| `test_baby_cry_false_positives.py` | Specific baby-cry false-positive test |
| `test_purr_suppression.py` | Specific purr-suppression test |
| `test_yamnet_gpu.py` | Diagnostic for YAMNet GPU inference |

## Notes on cat-safety

The deterrent operates in 22–24 kHz, inaudible to most adults and well within
a cat's hearing range. It's pulsed (anti-habituation) and gated by a verifier
head so purring, hissing, meowing for food, etc. don't trigger it. Volume
should be set low; the goal is to gracefully startle, never to harm.
