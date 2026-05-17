# cat-howl-deterrent

24/7 cat-howl detector with an optional near-ultrasonic deterrent. Listens to
the mic, runs YAMNet for general cat-sound detection, gates on a fine-tuned
verifier head to suppress purr false-positives, and plays a randomized 22–24
kHz pulsed burst when an actual howl is detected. Optionally records
pre/post-roll audio clips of every detection for fine-tuning data.

## Quick start

```bash
# Install
uv sync                          # creates .venv from pyproject.toml
uv pip install -e '.[macos]'     # on macOS, for pyobjc mic permission helper

# Run (with sensible defaults: CPU, Yeti, log-only, recording on)
./fix_audio_and_start.sh         # macOS — restarts coreaudiod and launches
# or, manually:
BACKEND=cpu LOG_ONLY=1 RECORD_HOWLS=1 MIC_DEVICE=Yeti cat-howl-deterrent
```

The detector logs triggers to `./logs/<timestamp>.{wav,json}` and (with
`RECORD_HOWLS=1`) saves `~8 s` clips to `./howl_recordings/<timestamp>_<class>.wav`.

## Project layout

```
src/cat_howl_deterrent/
├── __init__.py
├── __main__.py              # python -m cat_howl_deterrent
├── cli.py                   # console entry point
├── config.py                # constants + RuntimeConfig (env vars)
├── detector.py              # main run loop
├── deterrent.py             # ultrasonic tone generation + playback
├── events.py                # trigger save (WAV + JSON)
├── audio/
│   ├── capture.py           # SoundDeviceMic + FFmpegMic backends
│   ├── recorder.py          # pre/post-roll howl capture
│   └── resample.py          # native → 16 kHz resampler
├── classifier/
│   ├── backends.py          # TFBackend (CPU/GPU) + CoreMLBackend
│   └── verifier.py          # context head + gate decision
└── macos/
    └── mic_permission.py    # pyobjc/AVFoundation TCC helper

scripts/                     # one-off scripts (run via uv run ...)
├── convert_yamnet_coreml.py # YAMNet → .mlpackage converter
├── simulate_detector.py     # offline simulation
├── datasets/                # dataset fetchers
└── training/                # fine-tuning + evaluation

tests/                       # standalone smoke tests (not pytest)

fix_audio_and_start.sh       # canonical macOS launcher (restarts coreaudiod)
run_overnight.sh             # lighter launcher (no coreaudiod restart)
models/                      # trained head weights + labels
logs/                        # per-trigger WAV + JSON
howl_recordings/             # pre/post-roll fine-tuning clips
```

## Runtime configuration (env vars)

| Var               | Default | Meaning                                                |
| ----------------- | ------- | ------------------------------------------------------ |
| `BACKEND`         | `cpu`   | YAMNet backend: `cpu`, `mps`, `cuda`, `coreml`         |
| `LOG_ONLY`        | `0`     | If `1`, never plays the deterrent tone                 |
| `RECORD_HOWLS`    | `0`     | If `1`, save 3 s pre + 5 s post WAV per detection      |
| `MIC_DEVICE`      | _none_  | int index or substring of input device name           |
| `MIC_BACKEND`     | `sd`    | `sd` (sounddevice/PortAudio) or `ffmpeg` (AVFoundation)|
| `DETERRENT_SECONDS`| `3.0`  | Length of the ultrasonic burst                        |

## Architecture

```
mic (native rate)
  ↓ resample
16 kHz frame  →  YAMNet  →  scores + 1024-d embedding
                              ↓ (if Cat/Meow/Caterwaul ≥ 0.5 for 2 frames)
                            verifier head (cat-vocal probability)
                              ↓ (if ≥ 0.5)
                            TRIGGER  →  log + (optional) ultrasonic burst
                                     →  (optional) save 8 s clip for fine-tuning
```

## Why CPU?

YAMNet warm inference is **~2 ms** on an M-series P-core vs the 0.48 s hop
budget. GPU/ANE only help with batched inference; for single-frame realtime
they lose to CPU due to per-call dispatch overhead. The detector uses about
0.4% of one core.

CoreML conversion is supported (see `scripts/convert_yamnet_coreml.py`) but
**blocked at the framework level**: YAMNet's mel-spectrogram uses `RFFT` +
`ComplexAbs`, which coremltools does not implement. A split-frontend
approach is possible but not worth the engineering for this workload.

## Microphone permission on macOS

The detector needs the python interpreter to have TCC microphone access.
The first launch from a UI-attached terminal will pop a permission dialog;
click Allow. After that, the permission is sticky for that python binary.

If audio silently hangs (callbacks never fire), `coreaudiod` is likely
wedged — usually by Rogue Amoeba's Loopback AudioServerPlugin. Run
`./fix_audio_and_start.sh` which kills/respawns coreaudiod before
launching.

## Models

The verifier head (binary cat-vocal classifier on YAMNet embeddings) lives
at `models/yamnet_audioset_cats_head.keras` along with its labels file.
Fine-tuning scripts in `scripts/training/`.
