"""More aggressive workarounds for YAMNet cuDNN-1002 on L4."""
import os

# Env vars MUST be set before importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_CUDNN_USE_FRONTEND"] = "0"            # use legacy cuDNN API
os.environ["TF_DISABLE_CUDNN_TENSOR_OP_MATH"] = "1"   # disable TF32 / tensor cores
os.environ["TF_XLA_FLAGS"] = ""                       # disable XLA auto-cluster
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Older fallback for graph API
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
print(f"TF {tf.__version__}")

# Memory growth
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

# Disable XLA JIT for default scope
tf.config.optimizer.set_jit(False)

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")

wav = np.random.randn(15600).astype(np.float32)
try:
    scores, embeddings, _ = yamnet(wav)
    print(f"✓ Inference OK: scores={scores.shape}, emb={embeddings.shape}")
except Exception as e:
    print(f"✗ Still failing: {type(e).__name__}: {str(e)[:200]}")
