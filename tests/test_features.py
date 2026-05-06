"""Smoke tests for feature engineering."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

# Import via package (works after pip install -e . or from parent dir)
try:
    from stock_regime_detector.features import engineer_features, FEAT_COLS
except ImportError:
    # Fallback: direct import when repo dir is on sys.path
    from features import engineer_features, FEAT_COLS  # type: ignore


def _make_ohlcv(n=300):
    rng   = np.random.default_rng(0)
    price = 100.0 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2022-01-03", periods=n)
    dv    = np.abs(rng.normal(0.005, 0.002, n))
    df = pd.DataFrame({
        "Open":   price * (1 + rng.uniform(-0.002, 0.002, n)),
        "High":   price * (1 + dv),
        "Low":    price * (1 - dv),
        "Close":  price,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"]  = df[["Open", "Low",  "Close"]].min(axis=1)
    return df


def test_output_has_all_feature_cols():
    df = engineer_features(_make_ohlcv())
    for col in FEAT_COLS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_no_nan_after_engineering():
    df = engineer_features(_make_ohlcv())
    nan_counts = df[FEAT_COLS].isnull().sum()
    assert nan_counts.sum() == 0, f"NaNs found in features:\n{nan_counts[nan_counts > 0]}"


def test_row_count_reduces_by_warmup():
    raw = _make_ohlcv(300)
    out = engineer_features(raw, window=20)
    # dropna removes at least window rows
    assert len(out) < len(raw)
    assert len(out) > 0


def test_volatility_is_positive():
    df = engineer_features(_make_ohlcv())
    assert (df["volatility"] >= 0).all()


def test_rsi_in_range():
    df = engineer_features(_make_ohlcv())
    assert df["rsi"].between(0, 100).all()


def test_custom_window():
    df10 = engineer_features(_make_ohlcv(), window=10)
    df30 = engineer_features(_make_ohlcv(), window=30)
    # Larger window loses more rows to warm-up
    assert len(df10) > len(df30)
