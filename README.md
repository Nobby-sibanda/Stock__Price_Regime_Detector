# Stock Price Regime Detector

An unsupervised machine-learning toolkit that identifies hidden market regimes in stock price data and generates regime-adaptive trading signals with volatility targeting.

---

## Overview

Financial markets cycle through structurally distinct phases — trending, mean-reverting, and volatile. This project detects those phases automatically using **Gaussian Hidden Markov Models (HMM)** and **KMeans clustering**, then adapts a trading strategy to each detected regime in real time.

```
OHLCV Data  ──►  Feature Engineering  ──►  Regime Detection  ──►  Signal Generation  ──►  Dashboard
              (9 technical indicators)   (HMM / KMeans /        (Volatility-targeted     (8-panel PNG)
                                          Ensemble / Walk-fwd)   position sizing)
```

---

## Detected Regimes

| Label | ID | Characteristics | Adaptive Strategy |
|-------|----|-----------------|-------------------|
| **TRENDING** | 0 | High directional slope, moderate volatility | Momentum — follow trend via slope sign; EMA 20/50 crossover confirmation; 2×ATR trail stop |
| **MEAN-REV** | 1 | Low slope, tight volatility band | Mean-reversion — fade z-score extremes (±1.5σ); target reversion to 20-day mean |
| **VOLATILE** | 2 | Elevated volatility, directionless | Defensive — reduce position to 25%; stay flat unless RSI < 25 or > 75 |

---

## Project Structure

```
Stock__Price_Regime_Detector/
├── __main__.py          # Entry point — orchestrates the full pipeline
├── __init__.py          # Package init
├── config.py            # CLI argument parsing, regime names/colors/strategies
├── data.py              # Simulated & live (yfinance) OHLCV data loading
├── features.py          # Feature engineering (9 technical indicators)
├── models.py            # Regime detection: HMM, KMeans, Ensemble, Walk-forward
├── strategy.py          # Signal generation with vol-targeting & transaction costs
├── metrics.py           # Performance metrics (Sharpe, CAGR, Max DD, Calmar)
├── plot.py              # 8-panel matplotlib dashboard
└── Stock Regime Detector — launcher script   # Top-level run script
```

---

## Features (9 Technical Indicators)

| Feature | Description |
|---------|-------------|
| `returns` | Daily price returns |
| `volatility` | Annualised rolling log-return standard deviation |
| `trend_slope` | Vectorised rolling OLS slope on closing prices |
| `zscore` | Price deviation from rolling 20-day mean (in standard deviations) |
| `rsi` | 14-day Relative Strength Index |
| `atr` | 14-day Average True Range |
| `bb_width` | Bollinger Band width (upper − lower) / mid |
| `vol_ratio` | Daily volume relative to rolling average |
| `ret_autocorr` | Lag-1 return autocorrelation (momentum vs mean-reversion signal) |

---

## Detection Methods

### HMM (`--` default)
Gaussian HMM with full covariance matrices (`hmmlearn`). Trained end-to-end, then cluster centroids are mapped semantically: highest-volatility cluster → VOLATILE, highest |slope| among the rest → TRENDING, remainder → MEAN-REV.

### KMeans
KMeans with **automatic k selection** (silhouette score across k ∈ {2…6}) when `--n-regimes` is omitted. The same semantic centroid mapping is applied.

### Ensemble
Majority-vote combination of HMM and KMeans predictions using the same auto-selected k. Ties resolve in favour of the HMM (the probabilistic model).

### Walk-Forward (`--walk-forward`)
Rolling-window HMM re-fit: trains on the last `--train-days` trading days, predicts the next `--step-days`, then slides forward. Eliminates look-ahead bias and keeps the model calibrated to recent market structure.

---

## Signal Generation & Position Sizing

| Regime | Raw Signal Logic |
|--------|-----------------|
| TRENDING | `sign(trend_slope)` — long if slope > 0, short if slope < 0 |
| MEAN-REV | Short if z-score > +1.5, long if z-score < −1.5, flat otherwise |
| VOLATILE | ±0.25 if RSI hits extremes (< 25 or > 75), flat otherwise |

**Volatility-targeting** scales every raw signal by `target_vol / realized_vol` (capped at 3×), so position size contracts automatically in high-volatility regimes.

**Transaction costs** (default 5 bps per trade, `--transaction-cost`) are deducted whenever the signal changes.

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Annualised risk-adjusted return (strategy & buy-and-hold) |
| CAGR | Compound Annual Growth Rate |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar Ratio | CAGR / |Max Drawdown| |
| Per-regime Sharpe | Sharpe breakdown for each of the three regimes |

---

## Dashboard Output

Each run saves an 8-panel PNG to `outputs/` (configurable with `--outdir`):

| Panel | Content |
|-------|---------|
| 1 | Price chart with colour-coded regime shading |
| 2 | Regime label timeline |
| 3 | Cumulative returns — strategy vs buy-and-hold |
| 4 | Sized position signal (includes vol-targeting) |
| 5 | Days-per-regime bar chart |
| 6 | RSI (14) with overbought/oversold bands |
| 7 | Return distribution KDE by regime |
| 8 | Z-score with ±1.5σ signal bands |

---

## Installation

```bash
# Clone
git clone https://github.com/Nobby-sibanda/Stock__Price_Regime_Detector.git
cd Stock__Price_Regime_Detector

# Install dependencies
pip install numpy pandas scikit-learn hmmlearn matplotlib

# Optional — only needed for live data (--live flag)
pip install yfinance
```

---

## Usage

```bash
# Run on simulated data (default tickers: SPY, QQQ, AAPL)
python stock_regime_detector.py

# Run on live data for specific tickers
python stock_regime_detector.py --live --tickers MSFT TSLA NVDA

# Fix 3 regimes, apply walk-forward refit
python stock_regime_detector.py --live --n-regimes 3 --walk-forward

# Full option reference
python stock_regime_detector.py --help
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | `SPY QQQ AAPL` | Space-separated ticker symbols |
| `--live` | off | Fetch real data via yfinance |
| `--period` | `3y` | yfinance download period (`1y`, `3y`, `5y`) |
| `--n-regimes` | auto | Fix number of regimes; omit for silhouette auto-select (2–6) |
| `--window` | `20` | Rolling window (days) for feature calculation |
| `--walk-forward` | off | Enable rolling-window HMM re-fitting |
| `--train-days` | `252` | Training window length for walk-forward mode |
| `--step-days` | `63` | Re-fit interval (days) for walk-forward mode |
| `--smooth` | `3` | Minimum consecutive days to lock a regime label |
| `--target-vol` | `0.15` | Annualised volatility target for position sizing |
| `--transaction-cost` | `0.0005` | One-way cost per trade (5 bps) |
| `--outdir` | `outputs` | Directory for saved chart PNGs |

---

## Example Output

```
============================================================
  STOCK REGIME DETECTOR  --  Starting...
============================================================
  Tickers    : ['SPY', 'QQQ', 'AAPL']
  Window     : 20d
  Regimes    : auto
  Data       : live (yfinance)

============================================================
  TICKER: SPY
============================================================
  Loaded 756 trading days of OHLCV data

  -- Ensemble Detection --
    Auto-selected k=3  (silhouette=0.4821)
    Ensemble silhouette score: 0.4714

  Performance Metrics:
    Strategy Sharpe                1.342
    Buy&Hold Sharpe                0.891
    Strategy CAGR                  18.45%
    Strategy Max DD                -8.21%
    ...

  Current Regime  : [UP] TRENDING
  Recommended     : MOMENTUM - Follow trend direction via slope sign.
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Time-series data handling |
| `scikit-learn` | KMeans, silhouette scoring, feature scaling |
| `hmmlearn` | Gaussian Hidden Markov Model |
| `matplotlib` | Dashboard visualisation |
| `yfinance` *(optional)* | Live market data download |

---

## Author

**Nobby Sibanda** — [GitHub](https://github.com/Nobby-sibanda)
