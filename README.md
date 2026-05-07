# Stock Price Regime Detector

An unsupervised machine-learning toolkit that identifies hidden market regimes in stock price data, generates regime-adaptive trading signals with volatility targeting, and persists every analysis to a database for future ML research.

---

## Overview

Financial markets cycle through structurally distinct phases — trending, mean-reverting, and volatile. This project detects those phases automatically using **Gaussian Hidden Markov Models (HMM)** and **KMeans clustering**, then adapts a trading strategy to each detected regime in real time.

```
OHLCV Data  ──►  Feature Engineering  ──►  Regime Detection  ──►  Signal Generation  ──►  Dashboard
              (9 technical indicators)   (HMM / KMeans /        (Volatility-targeted     (8-panel PNG)
                                          Ensemble / Walk-fwd)   position sizing)
                                                    │
                                                    ▼
                                          SQLite Database
                                    (users · analyses · timeseries)
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
├── app.py               # Flask web UI — routes, auth integration, DB persistence
├── auth.py              # Authentication Blueprint (register / login / logout)
├── db.py                # SQLite persistence layer (users, analyses, timeseries)
├── conftest.py          # Pytest root conftest — fixes package import path
├── run.py               # CLI launcher: python run.py [options]
│
├── __main__.py          # Package entry point — orchestrates the full pipeline
├── __init__.py          # Package init
├── cli.py               # CLI argument parsing (--verbose, --dry-run, --export-metrics, …)
├── config.py            # Constants: regime names/colors/icons/strategy thresholds/plot sizes
├── data.py              # Simulated & live (yfinance) OHLCV data loading
├── features.py          # Feature engineering (9 technical indicators, vectorised)
├── models.py            # Regime detection: HMM, KMeans, Ensemble, Walk-forward
├── strategy.py          # Signal generation with vol-targeting & transaction costs
├── metrics.py           # Performance metrics (Sharpe, CAGR, Max DD, Calmar)
├── plot.py              # 8-panel matplotlib dashboard
│
├── templates/
│   ├── login.html       # Login page
│   ├── register.html    # Account creation page
│   ├── index.html       # Analysis configuration form
│   ├── results.html     # Live-polling results page
│   └── history.html     # Past analyses table + export links
│
├── requirements.txt     # Pinned dependencies
├── pyproject.toml       # PEP 517/518 build config + pip install support
└── tests/
    ├── test_features.py # Feature engineering smoke tests
    ├── test_models.py   # Detector smoke tests (shape, label validity, tuple return)
    └── test_strategy.py # Signal generation smoke tests
```

---

## Authentication

Every route in the web UI requires a logged-in account. On first visit users are redirected to `/login`.

| Page | Description |
|------|-------------|
| `/register` | Create an account — username (≥3 chars) and password (≥6 chars) |
| `/login` | Sign in with optional "remember me" |
| `/logout` | End the session |

Passwords are hashed with **werkzeug's PBKDF2-SHA256** implementation — plain-text passwords are never stored.

---

## Database

Every completed analysis is automatically saved to `regime_history.db` (SQLite, no external server required).

### Tables

| Table | Contents |
|-------|----------|
| `users` | Registered accounts — username, hashed password, created timestamp |
| `analyses` | One row per run — run ID, user, tickers, full config JSON, status, timestamps |
| `analysis_results` | Per-ticker per-method summary — metrics, current regime, confidence, transition matrix, chart URL |
| `regime_timeseries` | Full daily time series for every run — OHLCV + 9 features + regime label + signal + strategy return |

The `regime_timeseries` table is the **raw ML dataset**: 18 columns of structured, labelled daily market data that can be used directly to train classifiers, regressors, or clustering models on real regime behaviour.

### Web Routes

| Route | Description |
|-------|-------------|
| `/history` | Table of all your past analyses — status, tickers, duration, and links back to results |
| `/export/csv` | Download all stored timeseries as a flat CSV |
| `/export/json` | Same data as a JSON array |

Past runs survive server restarts — revisiting `/results/<run_id>` for an old run rebuilds the results payload from the database.

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

### Ensemble
Majority-vote of HMM and KMeans predictions. Both models share the same k (auto-selected via HMM BIC). Where they agree the label is kept; where they disagree KMeans acts as the second opinion. Confidence is the mean of both models' regime probabilities.

### Walk-Forward
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

# Install all dependencies (includes Flask + Flask-Login)
pip install -r requirements.txt

# Or install as an editable package (enables `stock-regime-detector` CLI)
pip install -e .

# Dev dependencies (pytest, coverage)
pip install -e ".[dev]"
```

---

## Web UI Usage

```bash
python app.py
# Open http://localhost:5000
```

1. **Register** an account at `/register`
2. **Configure** tickers, data source, model settings, and strategy parameters on the home page
3. **Run** the analysis — a spinner shows progress while HMM, KMeans, and Ensemble models fit
4. **View results** — regime dashboard charts, metrics, transition matrix, and current regime recommendation
5. **History** — revisit any past run from `/history`; export the full timeseries dataset as CSV or JSON for your own ML work

### Web Pages

| Page | Description |
|------|-------------|
| `/` | Analysis configuration form |
| `/results/{id}` | Auto-refreshing results — charts, metrics, transition matrix, current regime |
| `/history` | All past analyses with export links |
| `/export/csv` | Full timeseries dataset as CSV |
| `/export/json` | Full timeseries dataset as JSON |

---

## CLI Usage

```bash
# Run on simulated data (default tickers: SPY, QQQ, AAPL)
python run.py

# Print config and exit without fitting
python run.py --dry-run --live --tickers MSFT TSLA

# Run on live data for specific tickers, verbose output
python run.py --live --tickers MSFT TSLA NVDA --verbose

# Fix 3 regimes, walk-forward refit, export metrics to CSV
python run.py --live --n-regimes 3 --walk-forward --export-metrics results.csv

# Full option reference
python run.py --help
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
# 30 tests, all passing
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
| `flask` | Web UI framework |
| `flask-login` | Session-based authentication |
| `yfinance` *(optional)* | Live market data download |

---

## Running with Docker

```bash
git clone https://github.com/Nobby-sibanda/Stock__Price_Regime_Detector.git
cd Stock__Price_Regime_Detector
docker build -t stock-regime-detector .

# Start the container
docker run -p 5000:5000 stock-regime-detector

# Persist the database and chart outputs between runs
docker run -p 5000:5000 \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/regime_history.db:/workspace/regime_history.db \
  stock-regime-detector
```

Open `http://localhost:5000`, register an account, and run your first analysis.

---

## Author

**Nobby Sibanda** — [GitHub](https://github.com/Nobby-sibanda)
