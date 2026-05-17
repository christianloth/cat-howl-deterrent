"""Stage-2 verifier head: gates the trigger using a fine-tuned binary classifier.

The YAMNet scores get us close, but raw "Cat" probability false-fires on
purrs, vacuum cleaners, and assorted noise. The verifier head — a small
Keras model trained on AudioSet cat-vocal vs non-cat — runs on YAMNet's
1024-d embedding and produces P(cat-vocal). The trigger only fires if
that probability beats `HEAD_GATE_THRESHOLD`.

Returns None when the head can't be loaded (missing files, or backend
doesn't expose embeddings — i.e. CoreML).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cat_howl_deterrent.config import (
    CONTEXT_HEAD_PATH,
    CONTEXT_LABELS_PATH,
    HEAD_GATE_THRESHOLD,
    REQUIRE_HEAD_IF_AVAILABLE,
)


@dataclass
class GateDecision:
    gated: bool
    head_prob: float | None
    context: dict | None
    reason: str


class VerifierHead:
    def __init__(self, model, labels: list[str]):
        self.model = model
        self.labels = labels

    @classmethod
    def load(cls) -> VerifierHead | None:
        head_p = Path(CONTEXT_HEAD_PATH)
        labels_p = Path(CONTEXT_LABELS_PATH)
        if not head_p.exists() or not labels_p.exists():
            return None
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(head_p)
            labels = json.loads(labels_p.read_text())
            print(f"Loaded context head: {labels}")
            return cls(model, labels)
        except Exception as e:
            print(f"  ! context head load failed: {e}")
            return None

    def evaluate(self, embedding: np.ndarray) -> tuple[float, dict]:
        """Run the head on a 1024-d embedding. Returns (p_cat_vocal, context)."""
        import tensorflow as tf

        logits = self.model.predict(embedding[None, :], verbose=0)[0]
        if logits.shape[-1] == 1:
            head_prob = float(tf.sigmoid(logits).numpy()[0])
            context = {"cat-vocal": head_prob, "non-cat": 1.0 - head_prob}
        else:
            probs = tf.nn.softmax(logits).numpy()
            context = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
            if "cat-vocal" in context:
                head_prob = context["cat-vocal"]
            else:
                head_prob = max(v for k, v in context.items() if k != "non-cat")
        return head_prob, context


def gate(
    head: VerifierHead | None,
    embedding: np.ndarray | None,
) -> GateDecision:
    """Compute the gate decision for one frame."""
    if head is not None and embedding is not None:
        head_prob, context = head.evaluate(embedding)
        return GateDecision(
            gated=head_prob < HEAD_GATE_THRESHOLD,
            head_prob=head_prob,
            context=context,
            reason=f"head_p={head_prob:.2f}",
        )
    if head is not None and embedding is None:
        # CoreML can't supply embeddings → respect REQUIRE_HEAD_IF_AVAILABLE.
        return GateDecision(
            gated=REQUIRE_HEAD_IF_AVAILABLE,
            head_prob=None,
            context=None,
            reason="head=unavailable(backend)",
        )
    return GateDecision(gated=False, head_prob=None, context=None, reason="no-head")
