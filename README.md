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
| **TRENDING** ↑ | 0 | High directional slope, moderate volatility | Momentum — follow trend via slope sign; EMA 20/50 crossover confirmation; 2×ATR trail stop |
| **MEAN-REV** ↔ | 1 | Low slope, tight volatility band | Mean-reversion — fade z-score extremes (±1.5σ); target reversion to 20-day mean |
| **VOLATILE** ⚡ | 2 | Elevated volatility, directionless | Defensive — reduce position to 25%; stay flat unless RSI < 25 or > 75 |

---

## Project Structure

```
Stock__Price_Regime_Detector/
├── __main__.py          # Entry point — orchestrates the full pipeline
├── __init__.py          # Package init
├── cli.py               # CLI argument parsing (--verbose, --dry-run, --export-metrics, …)
├── config.py            # Constants: regime names/colors/icons/strategy thresholds/plot sizes
├── data.py              # Simulated & live (yfinance) OHLCV data loading
├── features.py          # Feature engineering (9 technical indicators, vectorised)
├── models.py            # Regime detection: HMM, KMeans, Ensemble, Walk-forward
├── strategy.py          # Signal generation with vol-targeting & transaction costs
├── metrics.py           # Performance metrics (Sharpe, CAGR, Max DD, Calmar)
├── plot.py              # 8-panel matplotlib dashboard (split into sub-functions)
├── stock_regime_detector.py  # Top-level launcher script
├── requirements.txt     # Pinned dependencies
├── pyproject.toml       # PEP 517/518 build config + pip install support
└── tests/
    ├── test_features.py # Feature engineering smoke tests
    ├── test_models.py   # Detector smoke tests (shape, label validity, tuple return)
    └── test_strategy.py # Signal generation smoke tests
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
| `ret_autocorr` | Lag-1 return autocorrelation — vectorised via stride tricks |

---

## Detection Methods

### HMM
Gaussian HMM with full covariance matrices (`hmmlearn`). Component count auto-selected via **HMM BIC** (not silhouette, which is invalid for sequential data). Returns hard labels plus per-observation posterior probabilities.

### KMeans
KMeans with **automatic k selection** (silhouette score across k ∈ {2…6}). Distance-based confidence score returned alongside labels.

### Ensemble (bug-fixed)
Majority-vote of HMM and KMeans predictions. Both models share the same k (auto-selected via HMM BIC). Where they agree the label is kept; where they disagree KMeans acts as the second opinion. Confidence is the mean of both models' regime probabilities.

> **Note:** The original ensemble had a bug (`np.where(a==b, a, a)`) that made it identical to plain HMM. This is now fixed.

### Walk-Forward (`--walk-forward`)
Rolling-window HMM re-fit: trains on the last `--train-days` trading days, predicts the next `--step-days`, then slides forward. Eliminates look-ahead bias. Shows a `tqdm` progress bar during the fold loop.

---

## Signal Generation & Position Sizing

| Regime | Raw Signal Logic |
|--------|-----------------|
| TRENDING ↑ | `sign(trend_slope)` — long if slope > 0, short if slope < 0 |
| MEAN-REV ↔ | Short if z-score > +1.5σ, long if z-score < −1.5σ, flat otherwise |
| VOLATILE ⚡ | ±0.25 if RSI hits extremes (< 25 or > 75), flat otherwise |

All thresholds (`MR_ZSCORE_THRESHOLD`, `VOL_RSI_LOW/HIGH`, `VOL_SIGNAL_SIZE`, `MAX_POSITION_SIZE`) are constants in `config.py`.

**Volatility-targeting** scales every raw signal by `target_vol / realized_vol` (capped at `MAX_POSITION_SIZE = 3×`).

A `regime_prob` column is stored in the output DataFrame, showing the model's confidence for the assigned label at each bar.

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Annualised risk-adjusted return (strategy & buy-and-hold) |
| CAGR | Compound Annual Growth Rate |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| Per-regime Sharpe | Sharpe breakdown for each of the three regimes |

---

## Dashboard Output

Each run saves an 8-panel PNG to `outputs/`:

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
git clone https://github.com/Nobby-sibanda/Stock__Price_Regime_Detector.git
cd Stock__Price_Regime_Detector

# Install all dependencies
pip install -r requirements.txt

# Or install as a package (enables `stock-regime-detector` CLI command)
pip install -e .

# Optional — only needed for --live flag
pip install yfinance
```

---

## Usage

```bash
# Run on simulated data (default tickers: SPY, QQQ, AAPL)
python stock_regime_detector.py

# Print config and exit without fitting (verify your flags first)
python stock_regime_detector.py --dry-run --live --tickers MSFT TSLA

# Run on live data for specific tickers, verbose output
python stock_regime_detector.py --live --tickers MSFT TSLA NVDA --verbose

# Fix 3 regimes, walk-forward refit, export metrics to CSV
python stock_regime_detector.py --live --n-regimes 3 --walk-forward \
    --export-metrics results.csv

# Full option reference
python stock_regime_detector.py --help
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | `SPY QQQ AAPL` | Space-separated ticker symbols |
| `--live` | off | Fetch real data via yfinance |
| `--period` | `3y` | yfinance download period (`1y`, `3y`, `5y`) |
| `--n-regimes` | auto | Fix number of regimes; omit for auto-select |
| `--window` | `20` | Rolling window (days) for feature calculation |
| `--walk-forward` | off | Enable rolling-window HMM re-fitting |
| `--train-days` | `252` | Training window length for walk-forward mode |
| `--step-days` | `63` | Re-fit interval (days) for walk-forward mode |
| `--smooth` | `3` | Minimum consecutive days to lock a regime label |
| `--target-vol` | `0.15` | Annualised volatility target for position sizing |
| `--transaction-cost` | `0.0005` | One-way cost per trade (5 bps) |
| `--outdir` | `outputs` | Directory for saved chart PNGs |
| `--export-metrics` | off | Export all metrics to a CSV file |
| `--verbose` | off | Enable DEBUG-level logging |
| `--dry-run` | off | Print config and exit without fitting |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations, stride-tricks vectorisation |
| `pandas` | Time-series data handling |
| `scikit-learn` | KMeans, silhouette scoring, feature scaling |
| `hmmlearn` | Gaussian Hidden Markov Model |
| `matplotlib` | Dashboard visualisation |
| `joblib` | Parallel ticker processing |
| `tqdm` | Walk-forward fold progress bar |
| `yfinance` *(optional)* | Live market data download |

---

## Author

**Nobby Sibanda** — [GitHub](https://github.com/Nobby-sibanda)
