"""
Fine-tune a custom head on YAMNet embeddings using the CatMeows dataset.

Pipeline:
  WAV → resample to 16 kHz → YAMNet (frozen) → 1024-d embedding (mean-pooled)
       → custom head (1024 -> 64 -> 3) → softmax over {Brushing, Food, Isolation}

Split is GROUPED BY CAT so test cats are unseen — otherwise the model just
memorizes individual cat voices and we'd massively overestimate accuracy.

Run:
    source .venv/bin/activate
    python finetune_catmeows.py
"""

import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

DATA_DIR = Path("data/catmeows/dataset")
CACHE = Path("data/yamnet_embeddings.npz")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

LABELS = {"B": 0, "F": 1, "I": 2}
LABEL_NAMES = ["Brushing", "Food", "Isolation"]


def parse_filename(p: Path):
    """B_TIG01_EU_FN_GIU01_201.wav -> (label_idx=0, cat_id='TIG01')"""
    parts = p.stem.split("_")
    return LABELS[parts[0]], parts[1]


def extract_embeddings():
    if CACHE.exists():
        print(f"Loading cached embeddings from {CACHE}")
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["groups"]

    print("Loading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    wavs = sorted(DATA_DIR.glob("*.wav"))
    print(f"Found {len(wavs)} clips. Extracting embeddings...")

    X, y, groups = [], [], []
    for p in tqdm(wavs):
        label, cat = parse_filename(p)
        wav, _ = librosa.load(str(p), sr=16000, mono=True)
        wav = wav.astype(np.float32)
        # YAMNet requires at least one full frame (15600 samples)
        if len(wav) < 15600:
            wav = np.pad(wav, (0, 15600 - len(wav)))
        _, embeddings, _ = yamnet(wav)
        X.append(embeddings.numpy().mean(axis=0))   # mean over sub-frames -> (1024,)
        y.append(label)
        groups.append(cat)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)
    np.savez_compressed(CACHE, X=X, y=y, groups=groups)
    print(f"Cached: {CACHE}  X={X.shape}  classes={np.bincount(y)}  cats={len(set(groups))}")
    return X, y, groups


def build_head():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(LABEL_NAMES)),   # logits
    ])


def main():
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    X, y, groups = extract_embeddings()

    # Group split: cats in train must not appear in test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    print(f"Train: {len(Xtr)} clips, {len(set(groups[train_idx]))} cats")
    print(f"Test:  {len(Xte)} clips, {len(set(groups[test_idx]))} cats (unseen)")

    model = build_head()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=40,
        batch_size=32,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )

    # Final eval
    pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
    print("\n=== Test Set (unseen cats) ===")
    print(classification_report(yte, pred, target_names=LABEL_NAMES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, pred))

    # Save
    model.save(OUT_DIR / "yamnet_catmeows_head.keras")
    with open(OUT_DIR / "yamnet_catmeows_labels.json", "w") as f:
        json.dump(LABEL_NAMES, f)
    print(f"\nSaved: {OUT_DIR / 'yamnet_catmeows_head.keras'}")


if __name__ == "__main__":
    main()
