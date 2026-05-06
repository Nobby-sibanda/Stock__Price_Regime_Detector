"""Smoke tests for signal generation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

try:
    from stock_regime_detector.features import engineer_features, FEAT_COLS
    from stock_regime_detector.models import detect_hmm
    from stock_regime_detector.strategy import generate_signals
    from stock_regime_detector.config import MAX_POSITION_SIZE
except ImportError:
    from features import engineer_features, FEAT_COLS  # type: ignore
    from models import detect_hmm  # type: ignore
    from strategy import generate_signals  # type: ignore
    from config import MAX_POSITION_SIZE  # type: ignore


def _make_df(n=300):
    rng   = np.random.default_rng(13)
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
    return engineer_features(raw, window=20)


DF      = _make_df()
LABELS, PROBA = detect_hmm(DF[FEAT_COLS], n_regimes=3)


def test_generate_signals_returns_dataframe():
    out = generate_signals(DF, LABELS, regime_prob=PROBA)
    assert isinstance(out, pd.DataFrame)


def test_output_has_required_columns():
    out = generate_signals(DF, LABELS, regime_prob=PROBA)
    for col in ("regime", "signal", "strat_ret", "regime_prob"):
        assert col in out.columns, f"Missing column: {col}"


def test_regime_prob_stored_when_provided():
    out = generate_signals(DF, LABELS, regime_prob=PROBA)
    assert "regime_prob" in out.columns
    np.testing.assert_array_equal(out["regime_prob"].values, PROBA)


def test_no_regime_prob_column_when_omitted():
    out = generate_signals(DF, LABELS)
    assert "regime_prob" not in out.columns


def test_signal_bounded_by_max_position_size():
    out = generate_signals(DF, LABELS, regime_prob=PROBA)
    assert (out["signal"].abs() <= MAX_POSITION_SIZE + 1e-9).all()


def test_strat_ret_no_nan_after_first_bar():
    out = generate_signals(DF, LABELS, regime_prob=PROBA)
    assert out["strat_ret"].iloc[1:].isnull().sum() == 0


def test_strategy_thresholds_from_config():
    """Changing target_vol should scale position sizes proportionally."""
    out_low  = generate_signals(DF, LABELS, target_vol=0.10)
    out_high = generate_signals(DF, LABELS, target_vol=0.20)
    # Higher vol target -> larger average absolute signal (before cap)
    assert out_high["signal"].abs().mean() >= out_low["signal"].abs().mean()
