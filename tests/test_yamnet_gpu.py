"""Diagnose YAMNet on GPU. Apply common cuDNN-1002 workarounds one at a time."""
import os
import sys
import time

# Env vars must be set BEFORE importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
# Force only one GPU — multi-GPU + variable-input-len model has been a known pain point
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print(f"TF {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible: {len(gpus)} — {gpus}")

# Belt-and-suspenders: memory growth on every visible GPU
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
        print(f"  set_memory_growth({g.name})")
    except Exception as e:
        print(f"  set_memory_growth failed on {g}: {e}")

print("Loading YAMNet...")
t0 = time.time()
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print(f"  loaded in {time.time()-t0:.1f}s")

print("Smoke test: random 0.96s window...")
wav = np.random.randn(15600).astype(np.float32)
try:
    t0 = time.time()
    scores, embeddings, spectrogram = yamnet(wav)
    dt = time.time() - t0
    print(f"  ✓ inference OK in {dt*1000:.1f}ms — scores={scores.shape}, emb={embeddings.shape}")
except Exception as e:
    print(f"  ✗ inference failed: {type(e).__name__}: {e}")
    sys.exit(1)

print("Throughput test (100 inferences)...")
t0 = time.time()
for _ in range(100):
    yamnet(wav)
dt = time.time() - t0
print(f"  100 inferences in {dt:.2f}s → {100/dt:.1f} clips/sec")

print("\nALL GOOD on GPU.")
