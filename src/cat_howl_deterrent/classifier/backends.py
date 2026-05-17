"""YAMNet inference backends.

Two implementations:

  * TFBackend — TensorFlow-Hub YAMNet on CPU or GPU (via tensorflow-metal
    or CUDA). Provides both scores and embeddings.

  * CoreMLBackend — Apple Neural Engine via a pre-converted .mlpackage.
    Only emits scores (no embeddings), so the verifier-head gate becomes
    unavailable.

For the typical M-series Mac use case, plain CPU is the right answer:
YAMNet warm inference is ~2 ms vs the 0.48 s hop budget. GPU/ANE only
help when batching many frames.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cat_howl_deterrent.config import COREML_MODEL_PATH, YAMNET_CLASSES_PATH


# ── Class-name loading ────────────────────────────────────────────────────
def _load_class_names_from_disk() -> list[str] | None:
    if YAMNET_CLASSES_PATH.exists():
        return json.loads(YAMNET_CLASSES_PATH.read_text())
    return None


def _load_class_names_via_tfhub(model) -> list[str]:
    import tensorflow as tf

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path) as f:
        return [row["display_name"] for row in csv.DictReader(f)]


# ── Backends ─────────────────────────────────────────────────────────────
class TFBackend:
    """TensorFlow YAMNet. device='/CPU:0' or '/GPU:0'."""

    def __init__(self, device: str):
        import tensorflow as tf
        import tensorflow_hub as hub

        self.tf = tf
        self.device = device
        if device == "/CPU:0":
            tf.config.set_visible_devices([], "GPU")
        print(f"Loading YAMNet on {device}...")
        with tf.device(device):
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.class_names = (
            _load_class_names_from_disk() or _load_class_names_via_tfhub(self.model)
        )

    def predict(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        with self.tf.device(self.device):
            scores, embeddings, _ = self.model(waveform)
        return scores.numpy().mean(axis=0), embeddings.numpy().mean(axis=0)


class CoreMLBackend:
    """Apple Neural Engine via Core ML. No embeddings → verifier head off."""

    def __init__(self, model_path: Path = COREML_MODEL_PATH):
        import coremltools as ct

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python scripts/convert_yamnet_coreml.py` first."
            )
        print(f"Loading Core ML model from {model_path}...")
        self.model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
        names = _load_class_names_from_disk()
        if names is None:
            raise FileNotFoundError("yamnet_classes.json missing; rerun the converter.")
        self.class_names = names

    def predict(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        out = self.model.predict({"waveform": waveform.astype(np.float32)})
        scores = None
        embedding = None
        for v in out.values():
            arr = np.asarray(v).reshape(-1)
            if arr.shape[0] == 521:
                scores = arr
            elif arr.shape[0] == 1024:
                embedding = arr
        return scores, embedding


def make_backend(name: str):
    name = name.lower()
    if name == "coreml":
        return CoreMLBackend()
    if name == "cuda":
        return TFBackend("/GPU:0")
    if name == "mps":
        return TFBackend("/GPU:0")  # tensorflow-metal exposes Apple GPU as /GPU:0
    if name == "cpu":
        return TFBackend("/CPU:0")
    raise ValueError(f"Unknown BACKEND={name!r}. Use coreml | mps | cuda | cpu.")
