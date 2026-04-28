import numpy as np
import pandas as pd

FEAT_COLS = [
    "returns",      # 0
    "volatility",   # 1  ← used by semantic_labels (vi=1)
    "trend_slope",  # 2  ← used by semantic_labels (si=2)
    "zscore",       # 3
    "rsi",          # 4
    "atr",          # 5
    "bb_width",     # 6
    "vol_ratio",    # 7
    "ret_autocorr", # 8
]

# Indices used downstream for regime labelling
VOL_IDX   = FEAT_COLS.index("volatility")
SLOPE_IDX = FEAT_COLS.index("trend_slope")


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised rolling OLS slope via stride tricks — replaces the O(n*w)
    Python loop in the original.  ~100x faster on long series.
    """
    windows = np.lib.stride_tricks.sliding_window_view(arr.astype(float), window)
    x  = np.arange(window, dtype=float)
    x -= x.mean()
    ssx = (x ** 2).sum()
    # subtract row means, then dot with x
    y   = windows - windows.mean(axis=1, keepdims=True)
    slp = (y * x).sum(axis=1) / ssx
    result = np.full(len(arr), np.nan)
    result[window - 1:] = slp
    return result


def engineer_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    df    = df.copy()

    df["returns"]     = close.pct_change()
    df["log_ret"]     = np.log(close / close.shift(1))
    df["volatility"]  = df["log_ret"].rolling(window).std() * np.sqrt(252)
    df["trend_slope"] = _rolling_slope(close.values, window)

    roll_mean      = close.rolling(window).mean()
    roll_std       = close.rolling(window).std()
    df["zscore"]   = (close - roll_mean) / (roll_std + 1e-9)

    delta          = close.diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss           = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - (100 / (1 + gain / (loss + 1e-9)))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"]      = tr.rolling(14).mean()

    upper          = roll_mean + 2 * roll_std
    lower_b        = roll_mean - 2 * roll_std
    df["bb_width"] = (upper - lower_b) / (roll_mean + 1e-9)
    df["vol_ratio"] = vol / (vol.rolling(window).mean() + 1e-9)

    # Vectorised lag-1 autocorrelation — replaces the slow rolling lambda
    ret = df["returns"]
    df["ret_autocorr"] = ret.rolling(window).corr(ret.shift(1))

    df.dropna(inplace=True)
    return df
