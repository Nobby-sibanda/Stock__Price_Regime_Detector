"""
Entry point -- run with:
    python -m stock_regime_detector [options]
    python stock_regime_detector.py [options]
"""

import logging
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from joblib import Parallel, delayed

from .cli import parse_args
from .config import REGIME_NAMES, REGIME_ICONS, STRATEGY_DESC
from .data import simulate_market_data, load_live_data
from .features import engineer_features, FEAT_COLS
from .models import (
    detect_hmm, detect_kmeans, detect_ensemble,
    walk_forward_detect, smooth_regimes, transition_matrix,
)
from .strategy import generate_signals
from .metrics import performance_metrics
from .plot import plot_dashboard


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="  %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


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
    log = logging.getLogger(__name__)
    log.info("")
    log.info("=" * 58)
    log.info("TICKER: %s", ticker)
    log.info("=" * 58)

    if args.live:
        log.info("Fetching live data via yfinance (period=%s)...", args.period)
        df = load_live_data(ticker, period=args.period)
    else:
        df = simulate_market_data(ticker)
    log.info("Loaded %d trading days of OHLCV data", len(df))

    df       = engineer_features(df, window=args.window)
    features = df[FEAT_COLS]

    results = {}
    for method, fn in _build_methods(args):
        log.info("")
        log.info("-- %s Detection --", method)
        regimes, regime_prob = fn(features)

        if args.smooth > 1:
            regimes = smooth_regimes(regimes, min_days=args.smooth)
            log.info("Smoothed (min_days=%d)", args.smooth)

        df_sig  = generate_signals(
            df, regimes, regime_prob=regime_prob,
            target_vol=args.target_vol,
            transaction_cost=args.transaction_cost,
        )
        metrics = performance_metrics(df_sig)
        trans   = transition_matrix(regimes)

        log.info("")
        log.info("Performance Metrics:")
        for k, v in metrics.items():
            log.info("  %-30s %s", k, v)

        log.info("")
        log.info("Regime Transition Matrix:")
        for line in trans.round(3).to_string().split("\n"):
            log.info("  %s", line)

        cur      = int(df_sig["regime"].iloc[-1])
        cur_prob = float(df_sig["regime_prob"].iloc[-1]) if "regime_prob" in df_sig else 0.0
        log.info("")
        log.info("Current Regime  : %s %s  (confidence %.0f%%)",
                 REGIME_ICONS[cur], REGIME_NAMES[cur], cur_prob * 100)
        log.info("Recommended     : %s", STRATEGY_DESC[cur])

        plot_dashboard(df_sig, ticker, method, outdir=args.outdir)
        results[method] = {"df": df_sig, "metrics": metrics, "trans": trans}

    return results


def _export_metrics(all_results: dict, path: str) -> None:
    rows = []
    for ticker, res in all_results.items():
        for method, data in res.items():
            row = {"ticker": ticker, "method": method}
            row.update(data["metrics"])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    logging.getLogger(__name__).info("Metrics exported to %s", path)


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=" * 58)
    log.info("STOCK REGIME DETECTOR  --  Starting...")
    log.info("=" * 58)
    log.info("Tickers    : %s", args.tickers)
    log.info("Window     : %dd", args.window)
    log.info("Regimes    : %s", "auto" if args.n_regimes is None else args.n_regimes)
    log.info("Data       : %s", "live (yfinance)" if args.live else "simulated")
    log.info("Smooth     : %dd min run", args.smooth)
    log.info("Target vol : %.0f%%", args.target_vol * 100)
    log.info("Txn cost   : %.1f bps", args.transaction_cost * 10_000)
    if args.walk_forward:
        log.info("Walk-fwd   : train=%dd  step=%dd", args.train_days, args.step_days)

    if args.dry_run:
        log.info("[dry-run] Config printed. Exiting without fitting.")
        return

    n_jobs = min(len(args.tickers), os.cpu_count() or 1)
    if n_jobs > 1:
        log.info("Parallelising %d tickers across %d workers", len(args.tickers), n_jobs)
        ticker_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_pipeline)(t, args) for t in args.tickers
        )
    else:
        ticker_results = [run_pipeline(t, args) for t in args.tickers]

    all_results: dict[str, dict] = dict(zip(args.tickers, ticker_results))

    # ── Cross-ticker summary ───────────────────────────────────────────────────
    log.info("")
    log.info("=" * 58)
    log.info("CROSS-TICKER SUMMARY  (Ensemble method)")
    log.info("=" * 58)
    log.info("  %-8s %-18s %8s %10s %10s",
             "Ticker", "Regime", "Sharpe", "CAGR", "Max DD")
    log.info("  %s %s %s %s %s",
             "-"*8, "-"*18, "-"*8, "-"*10, "-"*10)

    for ticker, res in all_results.items():
        key    = "Ensemble" if "Ensemble" in res else next(iter(res))
        df_sig = res[key]["df"]
        m      = res[key]["metrics"]
        rid    = int(df_sig["regime"].iloc[-1])
        label  = f"{REGIME_ICONS[rid]} {REGIME_NAMES[rid]}"
        log.info("  %-8s %-18s %8s %10s %10s",
                 ticker, label, m["Strategy Sharpe"],
                 m["Strategy CAGR"], m["Strategy Max DD"])

    if args.export_metrics:
        _export_metrics(all_results, args.export_metrics)

    log.info("")
    log.info("Charts saved to '%s/'", args.outdir)
    log.info("Done.")


if __name__ == "__main__":
    main()
