import numpy as np
import pandas as pd


def simulate_market_data(ticker, n_days=756):
    """
    Synthetic OHLCV data with alternating market regimes.
    Replace with load_live_data() for real prices.
    """
    seeds  = {"SPY": 42,    "QQQ": 7,     "AAPL": 13}
    starts = {"SPY": 420.0, "QQQ": 340.0, "AAPL": 160.0}
    # Hash unknown tickers so each gets distinct but reproducible synthetic data
    seed = seeds.get(ticker, abs(hash(ticker)) % (2 ** 31))
    rng  = np.random.default_rng(seed)

    reg_seq = [0, 1, 2] * 4
    lengths = [int(rng.integers(60, 120)) for _ in reg_seq]
    regime_labels = np.concatenate(
        [np.full(l, r) for r, l in zip(reg_seq, lengths)]
    )[:n_days]

    regime_params = {
        0: (0.0009, 0.008),
        1: (0.0001, 0.005),
        2: (0.0000, 0.022),
    }

    start_price = starts.get(ticker, 100.0)
    price = [start_price]
    for i in range(1, n_days):
        drift, vol = regime_params[regime_labels[i]]
        price.append(price[-1] * (1 + drift + vol * rng.standard_normal()))

    price = np.array(price)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    dv    = np.abs(rng.normal(0.005, 0.003, n_days))

    df = pd.DataFrame({
        "Open":   price * (1 + rng.uniform(-0.003, 0.003, n_days)),
        "High":   price * (1 + dv),
        "Low":    price * (1 - dv),
        "Close":  price,
        "Volume": rng.integers(40_000_000, 160_000_000, n_days).astype(float),
    }, index=dates)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"]  = df[["Open", "Low",  "Close"]].min(axis=1)
    return df


def load_live_data(ticker, period="3y"):
    """Fetch OHLCV from Yahoo Finance. Requires: pip install yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        )

    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df[df["Volume"] > 0].copy()
    return df
