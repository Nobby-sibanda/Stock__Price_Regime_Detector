"""
Entry point — run with:
    python -m stock_regime_detector [options]
    python stock_regime_detector.py [options]
"""

import warnings
warnings.filterwarnings("ignore")

import os

from .config import parse_args, REGIME_NAMES, REGIME_ICONS, STRATEGY_DESC
from .data import simulate_market_data, load_live_data
from .features import engineer_features, FEAT_COLS
from .models import (
    detect_hmm, detect_kmeans, detect_ensemble,
    walk_forward_detect, smooth_regimes, transition_matrix,
)
from .strategy import generate_signals
from .metrics import performance_metrics
from .plot import plot_dashboard


def _build_methods(args) -> list[tuple[str, callable]]:
    n = args.n_regimes
    methods = [
        ("HMM",      lambda f, n=n: detect_hmm(f, n)),
        ("KMeans",   lambda f, n=n: detect_kmeans(f, n)),
        ("Ensemble", lambda f, n=n: detect_ensemble(f, n)),
    ]
    if args.walk_forward:
        methods.append((
            "WalkFwd",
            lambda f, n=n: walk_forward_detect(
                f, n, train_days=args.train_days, step_days=args.step_days
            ),
        ))
    return methods


def run_pipeline(ticker: str, args) -> dict:
    print(f"\n{'='*60}")
    print(f"  TICKER: {ticker}")
    print(f"{'='*60}")

    if args.live:
        print(f"  Fetching live data via yfinance (period={args.period})...")
        df = load_live_data(ticker, period=args.period)
    else:
        df = simulate_market_data(ticker)
    print(f"  Loaded {len(df)} trading days of OHLCV data")

    df       = engineer_features(df, window=args.window)
    features = df[FEAT_COLS]

    results = {}
    for method, fn in _build_methods(args):
        print(f"\n  -- {method} Detection --")
        regimes = fn(features)

        if args.smooth > 1:
            regimes = smooth_regimes(regimes, min_days=args.smooth)
            print(f"    Smoothed (min_days={args.smooth})")

        df_sig  = generate_signals(df, regimes,
                                   target_vol=args.target_vol,
                                   transaction_cost=args.transaction_cost)
        metrics = performance_metrics(df_sig)
        trans   = transition_matrix(regimes)

        print("\n  Performance Metrics:")
        for k, v in metrics.items():
            print(f"    {k:<30} {v}")

        print("\n  Regime Transition Matrix:")
        print(trans.round(3).to_string())

        cur = int(df_sig["regime"].iloc[-1])
        print(f"\n  Current Regime  : {REGIME_ICONS[cur]} {REGIME_NAMES[cur]}")
        print(f"  Recommended     : {STRATEGY_DESC[cur]}")

        plot_dashboard(df_sig, ticker, method, outdir=args.outdir)
        results[method] = {"df": df_sig, "metrics": metrics, "trans": trans}

    return results


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  STOCK REGIME DETECTOR  --  Starting...")
    print("=" * 60)
    print(f"  Tickers    : {args.tickers}")
    print(f"  Window     : {args.window}d")
    print(f"  Regimes    : {'auto' if args.n_regimes is None else args.n_regimes}")
    print(f"  Data       : {'live (yfinance)' if args.live else 'simulated'}")
    print(f"  Smooth     : {args.smooth}d min run")
    print(f"  Target vol : {args.target_vol:.0%}")
    print(f"  Txn cost   : {args.transaction_cost * 10_000:.1f} bps")
    if args.walk_forward:
        print(f"  Walk-fwd   : train={args.train_days}d  step={args.step_days}d")

    all_results: dict[str, dict] = {}
    for ticker in args.tickers:
        all_results[ticker] = run_pipeline(ticker, args)

    # ── Cross-ticker summary (Ensemble or first available method) ─────────
    print("\n\n" + "=" * 60)
    print("  CROSS-TICKER SUMMARY  (Ensemble method)")
    print("=" * 60)
    print(f"  {'Ticker':<8} {'Regime':<16} {'Sharpe':>8} {'CAGR':>10} {'Max DD':>10}")
    print(f"  {'-'*8} {'-'*16} {'-'*8} {'-'*10} {'-'*10}")

    for ticker, res in all_results.items():
        key    = "Ensemble" if "Ensemble" in res else next(iter(res))
        df_sig = res[key]["df"]
        m      = res[key]["metrics"]
        rid    = int(df_sig["regime"].iloc[-1])
        label  = f"{REGIME_ICONS[rid]} {REGIME_NAMES[rid]}"
        print(f"  {ticker:<8} {label:<16} {str(m['Strategy Sharpe']):>8} "
              f"{m['Strategy CAGR']:>10} {m['Strategy Max DD']:>10}")

    print(f"\n  Charts saved to '{args.outdir}/'")
    print("  Done.")


if __name__ == "__main__":
    main()
