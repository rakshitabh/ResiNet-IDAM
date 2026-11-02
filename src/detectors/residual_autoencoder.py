"""Residual Autoencoder for IT and OT Anomaly Detection in ResiNet-IDAM v2.0.

Provides residual skip connections between corresponding encoder and decoder layers:
  Encoder: Input -> E1 -> E2 -> Bottleneck
  Decoder: Bottleneck -> D1 (+ E2) -> D2 (+ E1) -> Reconstruction Output

Includes both a Keras trainer and a portable NumPy forward-pass inference engine.
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


class NumpyResidualAutoencoder:
    """Portable NumPy forward-pass inference engine for the residual autoencoder."""

    def __init__(self, weights_path: str | Path) -> None:
        data = np.load(weights_path)
        self.w_e1, self.b_e1 = data["w_e1"], data["b_e1"]
        self.w_e2, self.b_e2 = data["w_e2"], data["b_e2"]
        self.w_b, self.b_b = data["w_b"], data["b_b"]
        self.w_d1, self.b_d1 = data["w_d1"], data["b_d1"]
        self.w_d2, self.b_d2 = data["w_d2"], data["b_d2"]
        self.w_out, self.b_out = data["w_out"], data["b_out"]

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        # Encoder
        e1 = np.maximum(0, X @ self.w_e1 + self.b_e1)
        e2 = np.maximum(0, e1 @ self.w_e2 + self.b_e2)
        b = np.maximum(0, e2 @ self.w_b + self.b_b)

        # Decoder with residual skip connections
        d1 = np.maximum(0, b @ self.w_d1 + self.b_d1) + e2
        d2 = np.maximum(0, d1 @ self.w_d2 + self.b_d2) + e1
        recon = d2 @ self.w_out + self.b_out
        return recon


def build_keras_residual_autoencoder(feature_count: int) -> keras.Model:
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required to train the residual autoencoder.")

    # Size scaled proportionally to input dimension
    e1_dim = 64 if feature_count >= 70 else 48
    e2_dim = 32 if feature_count >= 70 else 24
    b_dim = 16 if feature_count >= 70 else 12

    inputs = keras.Input(shape=(feature_count,), name="input_features")
    e1 = layers.Dense(e1_dim, activation="relu", name="enc_1")(inputs)
    e2 = layers.Dense(e2_dim, activation="relu", name="enc_2")(e1)
    bottleneck = layers.Dense(b_dim, activation="relu", name="bottleneck")(e2)

    d1 = layers.Dense(e2_dim, activation="relu", name="dec_1")(bottleneck)
    d1_res = layers.add([d1, e2], name="res_skip_1")

    d2 = layers.Dense(e1_dim, activation="relu", name="dec_2")(d1_res)
    d2_res = layers.add([d2, e1], name="res_skip_2")

    outputs = layers.Dense(feature_count, activation="linear", name="reconstruction")(d2_res)

    model = keras.Model(inputs, outputs, name=f"residual_autoencoder_{feature_count}")
    model.compile(optimizer="adam", loss="mse")
    return model


def export_weights_to_npz(model: keras.Model, output_path: str | Path) -> None:
    enc_1 = model.get_layer("enc_1").get_weights()
    enc_2 = model.get_layer("enc_2").get_weights()
    b = model.get_layer("bottleneck").get_weights()
    dec_1 = model.get_layer("dec_1").get_weights()
    dec_2 = model.get_layer("dec_2").get_weights()
    out = model.get_layer("reconstruction").get_weights()

    weights = {
        "w_e1": enc_1[0], "b_e1": enc_1[1],
        "w_e2": enc_2[0], "b_e2": enc_2[1],
        "w_b": b[0], "b_b": b[1],
        "w_d1": dec_1[0], "b_d1": dec_1[1],
        "w_d2": dec_2[0], "b_d2": dec_2[1],
        "w_out": out[0], "b_out": out[1],
    }
    np.savez_compressed(output_path, **weights)

