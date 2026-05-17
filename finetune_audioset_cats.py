"""
Fine-tune a YAMNet head as a binary "cat-vocalization or not" detector.

Uses AudioSet clips we pulled into data/audioset_cats/{Cat,Meow,Caterwaul,Hiss,Purr}/.

Positives:  Cat / Meow / Caterwaul / Hiss  (the trigger classes — incl. Hiss as
             "distressed cat sound")
Negatives:  Purr (cat but NOT alarming)  +  random non-cat AudioSet clips.

We grab the negatives by streaming a small slice of AudioSet/balanced/train
and skipping any clip whose labels intersect our cat MIDs.

Run:
    python finetune_audioset_cats.py
"""
import json
import os
from pathlib import Path

# Force CPU for YAMNet inference — cuDNN occasionally chokes on YAMNet's variable-
# length conv, and CPU is plenty fast for ~1.5k clips. Training (the head) is
# also tiny, so GPU buys us nothing here.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

POS_DIRS = ["Cat", "Meow", "Caterwaul", "Hiss"]
NEG_DIRS = ["Purr"]

ROOT = Path("data/audioset_cats")
NEG_DIR = ROOT / "_negatives"      # populated by download_audioset_negatives.py
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
CACHE = Path("data/audioset_embeddings.npz")


def embed_clip(yamnet, wav_path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(wav) < 15600:
        wav = np.pad(wav, (0, 15600 - len(wav)))
    _, embeddings, _ = yamnet(wav.astype(np.float32))
    return embeddings.numpy().mean(axis=0)


def extract():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X"], d["y"]

    print("Loading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    # Positives
    pos_paths = []
    for d in POS_DIRS:
        pos_paths += list((ROOT / d).glob("*.wav"))
    # "Hard" negatives: purr (real cat sound, but not alarming)
    hard_neg = []
    for d in NEG_DIRS:
        hard_neg += list((ROOT / d).glob("*.wav"))
    # "Easy" negatives: pre-downloaded non-cat clips
    easy_neg = list(NEG_DIR.glob("*.wav"))
    if not easy_neg:
        print(f"ERROR: no negatives in {NEG_DIR}. Run download_audioset_negatives.py first.")
        raise SystemExit(1)

    print(f"Positives: {len(pos_paths)}   Hard-negs (Purr): {len(hard_neg)}   "
          f"Easy-negs (non-cat): {len(easy_neg)}")

    X, y = [], []
    for p in tqdm(pos_paths, desc="pos"):
        X.append(embed_clip(yamnet, p)); y.append(1)
    for p in tqdm(hard_neg + easy_neg, desc="neg"):
        X.append(embed_clip(yamnet, p)); y.append(0)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    np.savez_compressed(CACHE, X=X, y=y)
    print(f"Saved: {CACHE}  X={X.shape}  pos={y.sum()}  neg={len(y)-y.sum()}")
    return X, y


def main():
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    X, y = extract()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=60,
        batch_size=64,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)],
    )

    pred = (tf.sigmoid(model.predict(Xte, verbose=0))[:, 0] > 0.5).numpy().astype(int)
    print("\n=== Test Set ===")
    print(classification_report(yte, pred, target_names=["non-cat", "cat-vocal"], digits=3))
    print(confusion_matrix(yte, pred))

    model.save(OUT_DIR / "yamnet_audioset_cats_head.keras")
    with open(OUT_DIR / "yamnet_audioset_cats_labels.json", "w") as f:
        json.dump(["non-cat", "cat-vocal"], f)
    print(f"\nSaved: {OUT_DIR / 'yamnet_audioset_cats_head.keras'}")


if __name__ == "__main__":
    main()
