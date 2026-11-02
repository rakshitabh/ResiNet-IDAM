"""Unit tests for Phase 2 Residual Autoencoder and Sequence Window modules."""

import numpy as np
import pytest

from src.detectors.residual_autoencoder import NumpyResidualAutoencoder
from src.detectors.sequence_window_detector import (
    CausalSequenceWindowScorer,
    compute_detection_delay,
)


def test_causal_sequence_window_scorer_mean():
    scorer = CausalSequenceWindowScorer(window_size=3, aggregation="mean")
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = scorer.score_sequence(scores)

    # Expected: [1.0, 1.5, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(out, [1.0, 1.5, 2.0, 3.0, 4.0])


def test_causal_sequence_window_scorer_max():
    scorer = CausalSequenceWindowScorer(window_size=3, aggregation="max")
    scores = [1.0, 5.0, 2.0, 1.0, 0.5]
    out = scorer.score_sequence(scores)

    # Expected: [1.0, 5.0, 5.0, 5.0, 2.0]
    np.testing.assert_allclose(out, [1.0, 5.0, 5.0, 5.0, 2.0])


def test_causal_window_strictly_backward_looking():
    """Verify that future scores do not affect earlier window outputs."""
    scorer = CausalSequenceWindowScorer(window_size=3, aggregation="mean")
    seq1 = [1.0, 2.0, 3.0]
    out1 = scorer.score_sequence(seq1)

    scorer.reset()
    seq2 = [1.0, 2.0, 3.0, 999.0, 999.0]
    out2 = scorer.score_sequence(seq2)

    # First 3 outputs must be identical regardless of future 999.0 values
    np.testing.assert_allclose(out1, out2[:3])


def test_detection_delay_calculation():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0])

    delay_info = compute_detection_delay(y_true, y_pred)
    assert delay_info["total_attack_bursts"] == 2
    assert delay_info["detected_attack_bursts"] == 2
    assert delay_info["burst_detection_rate"] == 1.0
    # Burst 1 starts at idx 2, detected at idx 3 -> delay 1
    # Burst 2 starts at idx 7, detected at idx 8 -> delay 1
    assert delay_info["median_detection_delay_samples"] == 1.0


def test_numpy_residual_autoencoder_inference(tmp_path):
    # Create synthetic weights
    weights = {
        "w_e1": np.eye(4, dtype=np.float32), "b_e1": np.zeros(4, dtype=np.float32),
        "w_e2": np.eye(4, dtype=np.float32), "b_e2": np.zeros(4, dtype=np.float32),
        "w_b": np.eye(4, dtype=np.float32), "b_b": np.zeros(4, dtype=np.float32),
        "w_d1": np.eye(4, dtype=np.float32), "b_d1": np.zeros(4, dtype=np.float32),
        "w_d2": np.eye(4, dtype=np.float32), "b_d2": np.zeros(4, dtype=np.float32),
        "w_out": np.eye(4, dtype=np.float32), "b_out": np.zeros(4, dtype=np.float32),
    }
    p = tmp_path / "test_res.npz"
    np.savez_compressed(p, **weights)

    model = NumpyResidualAutoencoder(p)
    x = np.ones((2, 4), dtype=np.float32)
    out = model.predict(x)
    assert out.shape == (2, 4)
    assert np.isfinite(out).all()

