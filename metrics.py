import numpy as np
import pandas as pd

from .config import REGIME_NAMES


def _sharpe(r: pd.Series) -> float:
    return float((r.mean() / (r.std() + 1e-9)) * np.sqrt(252))


def _max_dd(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    return float(((cum - cum.cummax()) / (cum.cummax() + 1e-9)).min())


def _cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def _calmar(r: pd.Series) -> float:
    dd = _max_dd(r)
    return _cagr(r) / (abs(dd) + 1e-9)


def performance_metrics(df: pd.DataFrame) -> dict:
    ret = df["strat_ret"].dropna()
    bnh = df["returns"].dropna()

    m = {
        "Strategy Sharpe" : round(_sharpe(ret), 3),
        "Buy&Hold Sharpe" : round(_sharpe(bnh), 3),
        "Strategy CAGR"   : f"{_cagr(ret) * 100:.2f}%",
        "Buy&Hold CAGR"   : f"{_cagr(bnh) * 100:.2f}%",
        "Strategy Max DD" : f"{_max_dd(ret) * 100:.2f}%",
        "Buy&Hold Max DD" : f"{_max_dd(bnh) * 100:.2f}%",
        "Strategy Calmar" : round(_calmar(ret), 3),
        "Total Days"      : len(df),
    }

    for rid, name in REGIME_NAMES.items():
        sub = df[df["regime"] == rid]["strat_ret"].dropna()
        m[f"{name} Days"]   = len(sub)
        m[f"{name} Sharpe"] = round(_sharpe(sub), 3) if len(sub) > 5 else "N/A"

    return m
