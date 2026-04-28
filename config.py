import argparse

REGIME_NAMES  = {0: "TRENDING", 1: "MEAN-REV", 2: "VOLATILE"}
REGIME_COLORS = {0: "#00d4aa",  1: "#f5a623",  2: "#e74c3c"}
REGIME_ICONS  = {0: "[UP]",     1: "[MR]",      2: "[VOL]"}

STRATEGY_DESC = {
    0: ("MOMENTUM - Follow trend direction via slope sign.\n"
        "        Use EMA 20/50 crossovers for confirmation.\n"
        "        Trail stop-loss at 2xATR below entry.\n"
        "        Take profit at 3-5xATR extension."),
    1: ("MEAN REVERSION - Fade Z-score extremes (+/-1.5 sigma).\n"
        "        Short overbought (Z>+1.5), long oversold (Z<-1.5).\n"
        "        Target: revert to rolling 20-day mean.\n"
        "        Use Bollinger Band squeeze as entry filter."),
    2: ("DEFENSIVE - Reduce position size to 25% of normal.\n"
        "        Stay flat unless RSI hits extreme (<25 or >75).\n"
        "        Prefer cash or short-vol options hedges.\n"
        "        Wait for ATR compression before re-entering."),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Stock Regime Detector & Adaptive Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL"],
                   help="Ticker symbols to analyse")
    p.add_argument("--n-regimes", type=int, default=None,
                   help="Number of regimes (omit for auto-select 2-6 via silhouette)")
    p.add_argument("--window", type=int, default=20,
                   help="Rolling window for feature calculation")
    p.add_argument("--live", action="store_true",
                   help="Fetch real data via yfinance (requires: pip install yfinance)")
    p.add_argument("--period", default="3y",
                   help="yfinance download period when --live is set (e.g. 1y, 3y, 5y)")
    p.add_argument("--walk-forward", action="store_true",
                   help="Walk-forward re-fitting: retrain HMM every step-days on a rolling window")
    p.add_argument("--train-days", type=int, default=252,
                   help="Training window length for walk-forward mode")
    p.add_argument("--step-days", type=int, default=63,
                   help="Re-fit interval (days) for walk-forward mode")
    p.add_argument("--smooth", type=int, default=3,
                   help="Minimum consecutive days to lock a regime (0 = off)")
    p.add_argument("--target-vol", type=float, default=0.15,
                   help="Annualised volatility target for position sizing")
    p.add_argument("--transaction-cost", type=float, default=0.0005,
                   help="One-way cost per trade as a fraction (default = 5 bps)")
    p.add_argument("--outdir", default="outputs",
                   help="Directory for saved chart PNGs")
    return p.parse_args()
