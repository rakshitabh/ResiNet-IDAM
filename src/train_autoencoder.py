"""Train the reproducible HAI dense Autoencoder baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hai-train-scaled.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "autoencoder.keras"
THRESHOLD_PATH = MODEL_DIR / "anomaly_threshold.json"
CONFIG_PATH = MODEL_DIR / "training_config.json"

SEED = 42
EPOCHS = 20
BATCH_SIZE = 512
VALIDATION_FRACTION = 0.20
THRESHOLD_PERCENTILE = 99


def build_model(feature_count: int) -> keras.Model:
    inputs = keras.Input(shape=(feature_count,))
    encoded = layers.Dense(32, activation="relu")(inputs)
    encoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(32, activation="relu")(encoded)
    outputs = layers.Dense(feature_count, activation="linear")(decoded)
    model = keras.Model(inputs, outputs, name="dense_autoencoder_baseline")
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    tf.keras.utils.set_random_seed(SEED)
    frame = pd.read_csv(DATA_PATH)
    feature_columns = [
        column for column in frame.columns if column not in {"timestamp", "label"}
    ]
    if not feature_columns:
        raise ValueError("No model features found in the training dataset.")

    values = frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Training features contain NaN or infinite values.")

    split_index = int(len(values) * (1 - VALIDATION_FRACTION))
    X_train = values[:split_index]
    X_validation = values[split_index:]
    model = build_model(len(feature_columns))
    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_validation, X_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    reconstructed = model.predict(
        X_validation, batch_size=BATCH_SIZE, verbose=0
    )
    errors = np.mean(np.square(X_validation - reconstructed), axis=1)
    threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    THRESHOLD_PATH.write_text(
        json.dumps(
            {
                "model_version": "dense-autoencoder-baseline-v1",
                "threshold": threshold,
                "threshold_percentile": THRESHOLD_PERCENTILE,
                "feature_columns": feature_columns,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                "features": len(feature_columns),
                "threshold_selection": (
                    "99th percentile of chronological normal validation errors"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "architecture": [32, 16, 32, len(feature_columns)],
                "split": "chronological 80/20",
                "shuffle": False,
                "optimizer": "adam",
                "loss": "mse",
                "final_epoch_count": len(history.history["loss"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Model saved: {MODEL_PATH}")
    print(f"Threshold saved: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
