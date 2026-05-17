"""
Convert YAMNet (TF Hub) → Core ML .mlpackage for Apple Neural Engine.

Run once on a Mac:
    pip install tensorflow tensorflow-hub coremltools numpy
    python convert_yamnet_coreml.py

Outputs:
    yamnet.mlpackage       — Core ML model
    yamnet_classes.json    — 521 class names (used by both backends)
"""

import csv
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

FRAME_SAMPLES = 15600   # 0.96 s @ 16 kHz — the YAMNet window we feed at runtime


def main():
    print("Loading YAMNet from TF Hub...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    # Dump class names alongside the model so the runtime never needs TF Hub
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path) as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]
    Path("yamnet_classes.json").write_text(json.dumps(class_names))
    print(f"Wrote yamnet_classes.json ({len(class_names)} classes)")

    # Output BOTH scores and embeddings so the verifier-head gate works on M4.
    @tf.function(input_signature=[tf.TensorSpec(shape=[FRAME_SAMPLES], dtype=tf.float32)])
    def predict(waveform):
        scores, embeddings, _ = yamnet(waveform)
        return (
            tf.reduce_mean(scores, axis=0),       # (521,)
            tf.reduce_mean(embeddings, axis=0),   # (1024,)
        )

    print("Tracing and converting to Core ML (~1-2 min)...")
    mlmodel = ct.convert(
        predict.get_concrete_function(),
        source="tensorflow",
        inputs=[ct.TensorType(name="waveform", shape=(FRAME_SAMPLES,))],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,   # CPU + GPU + Neural Engine
    )
    mlmodel.save("yamnet.mlpackage")
    print("Saved yamnet.mlpackage")

    # Quick smoke test
    print("Smoke testing...")
    dummy = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    out = mlmodel.predict({"waveform": dummy})
    for k, v in out.items():
        print(f"  Output '{k}': shape={np.asarray(v).shape}")
    # The 521-d one is scores
    scores = next(v for v in out.values() if np.asarray(v).reshape(-1).shape[0] == 521)
    print(f"  Top class on silence: {class_names[int(np.argmax(scores))]}")


if __name__ == "__main__":
    main()
