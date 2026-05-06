import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Stock Regime Detector & Adaptive Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL"],
                   help="Ticker symbols to analyse")
    p.add_argument("--n-regimes", type=int, default=None,
                   help="Number of regimes (omit for auto-select via BIC/silhouette)")
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
    p.add_argument("--export-metrics", default=None, metavar="FILE",
                   help="Export all metrics to a CSV file (e.g. results.csv)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging output")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without fitting any model")
    return p.parse_args()
