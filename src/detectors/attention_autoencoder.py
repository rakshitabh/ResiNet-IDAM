"""Attention Autoencoder for Feature-Weighted Anomaly Detection in ResiNet-IDAM v2.0.

Provides a self-attention gating layer over input features prior to bottleneck encoding,
serving as an experimental ablation to evaluate whether attention improves held-out detection.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except (ImportError, Exception):
    keras = None
    layers = None
    HAS_TF = False


class NumpyAttentionAutoencoder:
    """Portable forward-pass engine for the attention-weighted autoencoder."""

    def __init__(self, weights_path: str | Path) -> None:
        data = np.load(weights_path)
        self.w_attn, self.b_attn = data["w_attn"], data["b_attn"]
        self.w1, self.b1 = data["w1"], data["b1"]
        self.wb, self.bb = data["wb"], data["bb"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w_out, self.b_out = data["w_out"], data["b_out"]

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        # Sigmoid attention gating
        attn_raw = X @ self.w_attn + self.b_attn
        attn = 1.0 / (1.0 + np.exp(-np.clip(attn_raw, -15.0, 15.0)))
        x_attended = X * attn

        # Bottleneck autoencoder
        h1 = np.maximum(0, x_attended @ self.w1 + self.b1)
        hb = np.maximum(0, h1 @ self.wb + self.bb)
        h2 = np.maximum(0, hb @ self.w2 + self.b2)
        recon = h2 @ self.w_out + self.b_out
        return recon


def build_keras_attention_autoencoder(feature_count: int) -> keras.Model:
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required to train the attention autoencoder.")

    inputs = keras.Input(shape=(feature_count,), name="input_features")
    attn_weights = layers.Dense(feature_count, activation="sigmoid", name="attn_gate")(inputs)
    attended_inputs = layers.multiply([inputs, attn_weights], name="attended_features")

    e1 = layers.Dense(32, activation="relu", name="enc_1")(attended_inputs)
    b = layers.Dense(16, activation="relu", name="bottleneck")(e1)
    d1 = layers.Dense(32, activation="relu", name="dec_1")(b)
    outputs = layers.Dense(feature_count, activation="linear", name="reconstruction")(d1)

    model = keras.Model(inputs, outputs, name=f"attention_autoencoder_{feature_count}")
    model.compile(optimizer="adam", loss="mse")
    return model


def export_attention_weights_to_npz(model: keras.Model, output_path: str | Path) -> None:
    attn = model.get_layer("attn_gate").get_weights()
    enc_1 = model.get_layer("enc_1").get_weights()
    b = model.get_layer("bottleneck").get_weights()
    dec_1 = model.get_layer("dec_1").get_weights()
    out = model.get_layer("reconstruction").get_weights()

    weights = {
        "w_attn": attn[0], "b_attn": attn[1],
        "w1": enc_1[0], "b1": enc_1[1],
        "wb": b[0], "bb": b[1],
        "w2": dec_1[0], "b2": dec_1[1],
        "w_out": out[0], "b_out": out[1],
    }
    np.savez_compressed(output_path, **weights)

