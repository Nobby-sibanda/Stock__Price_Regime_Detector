import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEAT_COLS = [
    "returns",      # 0
    "volatility",   # 1  <- used by semantic_labels (VOL_IDX=1)
    "trend_slope",  # 2  <- used by semantic_labels (SLOPE_IDX=2)
    "zscore",       # 3
    "rsi",          # 4
    "atr",          # 5
    "bb_width",     # 6
    "vol_ratio",    # 7
    "ret_autocorr", # 8
]

VOL_IDX   = FEAT_COLS.index("volatility")
SLOPE_IDX = FEAT_COLS.index("trend_slope")


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised rolling OLS slope via stride tricks.
    ~100x faster than a Python loop on long series.
    """
    windows = np.lib.stride_tricks.sliding_window_view(arr.astype(float), window)
    x  = np.arange(window, dtype=float)
    x -= x.mean()
    ssx = (x ** 2).sum()
    y   = windows - windows.mean(axis=1, keepdims=True)
    slp = (y * x).sum(axis=1) / ssx
    result = np.full(len(arr), np.nan)
    result[window - 1:] = slp
    return result


def _rolling_autocorr(s: pd.Series, window: int) -> pd.Series:
    """
    Vectorised lag-1 rolling autocorrelation via stride tricks.
    Avoids the O(n*w) pandas rolling corr overhead.
    """
    arr     = s.values.astype(float)
    arr_lag = np.roll(arr, 1)
    arr_lag[0] = np.nan

    wins     = np.lib.stride_tricks.sliding_window_view(arr,     window)
    wins_lag = np.lib.stride_tricks.sliding_window_view(arr_lag, window)

    ym  = wins     - wins.mean(axis=1,     keepdims=True)
    ylm = wins_lag - wins_lag.mean(axis=1, keepdims=True)

    num  = (ym * ylm).sum(axis=1)
    den  = np.sqrt((ym ** 2).sum(axis=1) * (ylm ** 2).sum(axis=1)) + 1e-9
    corr = num / den

    result = np.full(len(arr), np.nan)
    result[window - 1:] = corr
    return pd.Series(result, index=s.index)


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

    df["ret_autocorr"] = _rolling_autocorr(df["returns"], window)

    df.dropna(inplace=True)
    log.debug("Engineered features: %d rows x %d feature cols", len(df), len(FEAT_COLS))
    return df
