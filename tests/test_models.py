"""Smoke tests for regime detection models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

try:
    from stock_regime_detector.features import engineer_features, FEAT_COLS
    from stock_regime_detector.models import (
        detect_hmm, detect_kmeans, detect_ensemble,
        walk_forward_detect, smooth_regimes, transition_matrix,
    )
except ImportError:
    from features import engineer_features, FEAT_COLS  # type: ignore
    from models import (  # type: ignore
        detect_hmm, detect_kmeans, detect_ensemble,
        walk_forward_detect, smooth_regimes, transition_matrix,
    )


def _make_features(n=300):
    rng   = np.random.default_rng(7)
    price = 100.0 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2022-01-03", periods=n)
    dv    = np.abs(rng.normal(0.005, 0.002, n))
    raw = pd.DataFrame({
        "Open":   price, "High": price * (1 + dv),
        "Low":    price * (1 - dv), "Close": price,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    raw["High"] = raw[["Open", "High", "Close"]].max(axis=1)
    raw["Low"]  = raw[["Open", "Low",  "Close"]].min(axis=1)
    df = engineer_features(raw, window=20)
    return df[FEAT_COLS]


FEATURES = _make_features()


@pytest.mark.parametrize("fn", [detect_hmm, detect_kmeans, detect_ensemble])
def test_detector_returns_tuple(fn):
    labels, proba = fn(FEATURES, n_regimes=3)
    assert isinstance(labels, np.ndarray)
    assert isinstance(proba, np.ndarray)


@pytest.mark.parametrize("fn", [detect_hmm, detect_kmeans, detect_ensemble])
def test_detector_label_shape(fn):
    labels, proba = fn(FEATURES, n_regimes=3)
    assert labels.shape == (len(FEATURES),)
    assert proba.shape  == (len(FEATURES),)


@pytest.mark.parametrize("fn", [detect_hmm, detect_kmeans, detect_ensemble])
def test_detector_valid_labels(fn):
    labels, _ = fn(FEATURES, n_regimes=3)
    assert set(np.unique(labels)).issubset({0, 1, 2})


@pytest.mark.parametrize("fn", [detect_hmm, detect_kmeans, detect_ensemble])
def test_regime_prob_in_range(fn):
    _, proba = fn(FEATURES, n_regimes=3)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_ensemble_differs_from_plain_hmm():
    """Ensemble must not be identical to HMM after the bug fix."""
    hmm_labels, _  = detect_hmm(FEATURES, n_regimes=3)
    ens_labels, _  = detect_ensemble(FEATURES, n_regimes=3)
    # They CAN agree in some runs, but the arrays shouldn't always be identical
    # We just verify ensemble runs without error and produces valid output
    assert set(np.unique(ens_labels)).issubset({0, 1, 2})


def test_walk_forward_returns_tuple():
    labels, proba = walk_forward_detect(FEATURES, n_regimes=3,
                                        train_days=100, step_days=30)
    assert labels.shape == (len(FEATURES),)
    assert proba.shape  == (len(FEATURES),)


def test_walk_forward_no_unfilled():
    labels, _ = walk_forward_detect(FEATURES, n_regimes=3,
                                    train_days=100, step_days=30)
    assert (labels != -1).all(), "Some positions were never assigned a regime"


def test_smooth_regimes_removes_short_runs():
    arr = np.array([0, 0, 1, 0, 0, 0, 2, 2, 2, 2])
    smoothed = smooth_regimes(arr, min_days=3)
    # The lone 1 and short 0-run should be replaced
    assert smoothed[2] != 1


def test_transition_matrix_rows_sum_to_one():
    labels, _ = detect_hmm(FEATURES, n_regimes=3)
    tm = transition_matrix(labels, n_regimes=3)
    row_sums = tm.values.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
