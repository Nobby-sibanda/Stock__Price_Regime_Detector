"""
Flask web UI for the Stock Regime Detector.
Run:  python app.py
Open: http://localhost:5000
"""

import os
import sys
import json
import uuid
import threading
import traceback
import logging

from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, send_from_directory,
)

# ── Package path (works both locally and inside the Docker layout) ────────────
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
for _p in [_parent, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stock_regime_detector.data     import simulate_market_data, load_live_data
from stock_regime_detector.features import engineer_features, FEAT_COLS
from stock_regime_detector.models   import (
    detect_hmm, detect_kmeans, detect_ensemble,
    walk_forward_detect, smooth_regimes, transition_matrix,
)
from stock_regime_detector.strategy import generate_signals
from stock_regime_detector.metrics  import performance_metrics
from stock_regime_detector.plot     import plot_dashboard
from stock_regime_detector.config   import REGIME_NAMES, REGIME_ICONS, STRATEGY_DESC

# ── Flask app ─────────────────────────────────────────────────────────────────
app      = Flask(__name__)
RUNS     = {}          # keyed by run_id
OUT_ROOT = os.path.join(_here, "outputs")
os.makedirs(OUT_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)


# ── Analysis worker ───────────────────────────────────────────────────────────

def _run_analysis(run_id, cfg):
    RUNS[run_id]["status"] = "running"
    try:
        outdir = os.path.join(OUT_ROOT, run_id)
        os.makedirs(outdir, exist_ok=True)

        tickers       = cfg["tickers"]
        n_regimes     = cfg["n_regimes"]
        window        = cfg["window"]
        live          = cfg["live"]
        period        = cfg["period"]
        walk_forward  = cfg["walk_forward"]
        train_days    = cfg["train_days"]
        step_days     = cfg["step_days"]
        smooth        = cfg["smooth"]
        target_vol    = cfg["target_vol"]
        txn_cost      = cfg["txn_cost"]

        all_results = {}

        for ticker in tickers:
            log.info("Processing %s ...", ticker)
            df = load_live_data(ticker, period=period) if live else simulate_market_data(ticker)
            df = engineer_features(df, window=window)
            features = df[FEAT_COLS]

            methods = [
                ("HMM",      lambda f: detect_hmm(f, n_regimes)),
                ("KMeans",   lambda f: detect_kmeans(f, n_regimes)),
                ("Ensemble", lambda f: detect_ensemble(f, n_regimes)),
            ]
            if walk_forward:
                methods.append(("WalkFwd",
                    lambda f: walk_forward_detect(f, n_regimes, train_days, step_days)))

            ticker_res = {}
            for method, fn in methods:
                regimes, regime_prob = fn(features)
                if smooth > 1:
                    regimes = smooth_regimes(regimes, min_days=smooth)

                df_sig  = generate_signals(df, regimes, regime_prob=regime_prob,
                                           target_vol=target_vol,
                                           transaction_cost=txn_cost)
                metrics = performance_metrics(df_sig)
                trans   = transition_matrix(regimes)
                plot_dashboard(df_sig, ticker, method, outdir=outdir)

                cur      = int(df_sig["regime"].iloc[-1])
                cur_prob = float(df_sig["regime_prob"].iloc[-1]) \
                           if "regime_prob" in df_sig else 0.0

                ticker_res[method] = {
                    "metrics":       metrics,
                    "transition":    trans.round(3).to_dict("split"),
                    "regime_id":     cur,
                    "regime_name":   REGIME_NAMES[cur],
                    "regime_icon":   REGIME_ICONS[cur],
                    "regime_prob":   round(cur_prob * 100, 1),
                    "strategy_desc": STRATEGY_DESC[cur],
                    "chart_url":     f"/outputs/{run_id}/{ticker}_{method}_dashboard.png",
                }
                log.info("  %s/%s done", ticker, method)

            all_results[ticker] = ticker_res

        RUNS[run_id].update(status="done", results=all_results)

    except Exception as exc:
        RUNS[run_id].update(status="error",
                            error=str(exc),
                            traceback=traceback.format_exc())
        log.exception("Run %s failed", run_id)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    f   = request.form
    cfg = dict(
        tickers      = [t.strip().upper() for t in f.get("tickers", "SPY").split()],
        n_regimes    = int(f["n_regimes"]) if f.get("n_regimes") else None,
        window       = int(f.get("window", 20)),
        live         = f.get("live") == "on",
        period       = f.get("period", "3y"),
        walk_forward = f.get("walk_forward") == "on",
        train_days   = int(f.get("train_days", 252)),
        step_days    = int(f.get("step_days", 63)),
        smooth       = int(f.get("smooth", 3)),
        target_vol   = float(f.get("target_vol", 0.15)),
        txn_cost     = float(f.get("txn_cost", 0.0005)),
    )
    run_id = str(uuid.uuid4())[:8]
    RUNS[run_id] = {"status": "pending", "cfg": cfg}
    threading.Thread(target=_run_analysis, args=(run_id, cfg), daemon=True).start()
    return redirect(url_for("results", run_id=run_id))


@app.route("/results/<run_id>")
def results(run_id):
    if run_id not in RUNS:
        return "Run not found", 404
    return render_template("results.html", run_id=run_id)


@app.route("/api/status/<run_id>")
def api_status(run_id):
    run = RUNS.get(run_id)
    if not run:
        return jsonify(status="not_found"), 404
    payload = {"status": run["status"]}
    if run["status"] == "done":
        payload["results"] = run["results"]
    elif run["status"] == "error":
        payload["error"]     = run.get("error")
        payload["traceback"] = run.get("traceback")
    return jsonify(payload)


@app.route("/outputs/<run_id>/<filename>")
def serve_chart(run_id, filename):
    return send_from_directory(os.path.join(OUT_ROOT, run_id), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
