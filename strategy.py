"""
Signal generation with:
  - Vectorised regime-specific logic (no Python loop)
  - Volatility-targeting: size proportional to target_vol / realised_vol
    (capped at MAX_POSITION_SIZE)
  - Transaction cost deducted on every signal change
  - Optional regime_prob stored as a column in the output DataFrame
"""

import numpy as np
import pandas as pd

from .config import (
    MR_ZSCORE_THRESHOLD,
    VOL_RSI_LOW, VOL_RSI_HIGH, VOL_SIGNAL_SIZE,
    MAX_POSITION_SIZE,
)


def generate_signals(
    df: pd.DataFrame,
    regimes: np.ndarray,
    regime_prob: np.ndarray | None = None,
    target_vol: float = 0.15,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    df = df.copy()

    slope  = df["trend_slope"].values
    zscore = df["zscore"].values
    rsi    = df["rsi"].values
    r      = np.asarray(regimes)

    # ── Regime-specific raw signals (vectorised) ──────────────────────────
    trending_sig = np.sign(slope)
    mr_sig       = np.where(zscore >  MR_ZSCORE_THRESHOLD, -1.0,
                   np.where(zscore < -MR_ZSCORE_THRESHOLD,  1.0, 0.0))
    vol_sig      = np.where(rsi < VOL_RSI_LOW,   VOL_SIGNAL_SIZE,
                   np.where(rsi > VOL_RSI_HIGH,  -VOL_SIGNAL_SIZE, 0.0))

    raw_sig = np.where(r == 0, trending_sig,
              np.where(r == 1, mr_sig, vol_sig))

    # ── Volatility-targeting position size ────────────────────────────────
    ann_vol   = df["volatility"].values
    size      = np.where(ann_vol > 1e-6, target_vol / ann_vol, 1.0)
    size      = np.clip(size, 0.0, MAX_POSITION_SIZE)
    sized_sig = raw_sig * size

    # ── Transaction cost on every signal change ───────────────────────────
    sig_series = pd.Series(sized_sig, index=df.index)
    changed    = sig_series.diff().abs() > 1e-9
    strat_ret  = sig_series.shift(1) * df["returns"] - changed * transaction_cost

    df["regime"]    = regimes
    df["signal"]    = sized_sig
    df["strat_ret"] = strat_ret
    if regime_prob is not None:
        df["regime_prob"] = np.asarray(regime_prob)
    return df
