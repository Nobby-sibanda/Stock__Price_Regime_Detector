"""
Flask web UI for the Stock Regime Detector.
Run:  python app.py
Open: http://localhost:5000
"""

import os
import sys

# ── Import path setup ─────────────────────────────────────────────────────────
# stock_regime_detector.py (the CLI launcher) shadows the package when the
# project directory is on sys.path.  Strip it; add only the parent so Python
# resolves 'stock_regime_detector' to the package directory.

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)

while _here in sys.path:
    sys.path.remove(_here)

if _parent not in sys.path:
    sys.path.insert(0, _parent)

import io
import csv
import json
import uuid
import threading
import traceback
import logging
from datetime import datetime, timezone

from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, send_from_directory, Response,
)
from flask_login import login_required, current_user

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

import stock_regime_detector.db   as _db
from stock_regime_detector.auth  import auth_bp, init_login_manager

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=os.path.join(_here, "templates"),
)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# Auth
app.register_blueprint(auth_bp)
init_login_manager(app)

# Init DB
_db.init_db()

RUNS     = {}          # in-memory status cache (run_id -> dict)
OUT_ROOT = os.path.join(_here, "outputs")
os.makedirs(OUT_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)


# ── Analysis worker ───────────────────────────────────────────────────────────

def _run_analysis(run_id, cfg):
    RUNS[run_id]["status"] = "running"
    _db.update_run_status(run_id, "running")
    try:
        outdir = os.path.join(OUT_ROOT, run_id)
        os.makedirs(outdir, exist_ok=True)

        tickers      = cfg["tickers"]
        n_regimes    = cfg["n_regimes"]
        window       = cfg["window"]
        live         = cfg["live"]
        period       = cfg["period"]
        walk_forward = cfg["walk_forward"]
        train_days   = cfg["train_days"]
        step_days    = cfg["step_days"]
        smooth       = cfg["smooth"]
        target_vol   = cfg["target_vol"]
        txn_cost     = cfg["txn_cost"]

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

                result_data = {
                    "metrics":       metrics,
                    "transition":    trans.round(3).to_dict("split"),
                    "regime_id":     cur,
                    "regime_name":   REGIME_NAMES[cur],
                    "regime_icon":   REGIME_ICONS[cur],
                    "regime_prob":   round(cur_prob * 100, 1),
                    "strategy_desc": STRATEGY_DESC[cur],
                    "chart_url":     f"/outputs/{run_id}/{ticker}_{method}_dashboard.png",
                }
                ticker_res[method] = result_data

                # ── Persist to database ──────────────────────────────────────
                _db.insert_result(run_id, ticker, method, result_data)
                _db.insert_timeseries(run_id, ticker, method, df_sig)

                log.info("  %s/%s done", ticker, method)

            all_results[ticker] = ticker_res

        RUNS[run_id].update(status="done", results=all_results)
        _db.update_run_status(run_id, "done")

    except Exception as exc:
        err_msg = str(exc)
        tb      = traceback.format_exc()
        RUNS[run_id].update(status="error", error=err_msg, traceback=tb)
        _db.update_run_status(run_id, "error", error=err_msg)
        log.exception("Run %s failed", run_id)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
@login_required
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

    # Persist the run record immediately so history shows it
    _db.insert_run(run_id, current_user.id, cfg)

    threading.Thread(target=_run_analysis, args=(run_id, cfg), daemon=True).start()
    return redirect(url_for("results", run_id=run_id))


@app.route("/results/<run_id>")
@login_required
def results(run_id):
    # Accept runs from in-memory (current session) or DB (past sessions)
    if run_id not in RUNS:
        row = _db.get_run(run_id)
        if not row:
            return "Run not found", 404
        # Reconstruct a minimal status dict from DB for polling
        RUNS[run_id] = {"status": row["status"], "cfg": json.loads(row["config"])}
        if row["status"] == "done":
            # Rebuild results payload from DB so the status API works
            RUNS[run_id]["results"] = _rebuild_results_from_db(run_id)
        elif row["status"] == "error":
            RUNS[run_id]["error"]     = row["error"] or "unknown error"
            RUNS[run_id]["traceback"] = ""
    return render_template("results.html", run_id=run_id)


def _rebuild_results_from_db(run_id: str) -> dict:
    rows = _db.get_results_for_run(run_id)
    out  = {}
    for row in rows:
        t, m = row["ticker"], row["method"]
        out.setdefault(t, {})[m] = {
            "metrics":       json.loads(row["metrics"]),
            "transition":    json.loads(row["transition_json"]),
            "regime_id":     row["regime_id"],
            "regime_name":   row["regime_name"],
            "regime_icon":   REGIME_ICONS.get(row["regime_id"], ""),
            "regime_prob":   row["regime_prob"],
            "strategy_desc": row["strategy_desc"],
            "chart_url":     row["chart_url"],
        }
    return out


@app.route("/api/status/<run_id>")
@login_required
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
@login_required
def serve_chart(run_id, filename):
    return send_from_directory(os.path.join(OUT_ROOT, run_id), filename)


# ── History & export routes ───────────────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    raw_runs = _db.list_runs(user_id=current_user.id)
    runs     = _enrich_runs(raw_runs)
    return render_template("history.html", runs=runs)


@app.route("/export/csv")
@login_required
def export_csv():
    rows = _db.export_all_timeseries(user_id=current_user.id)
    if not rows:
        return "No data to export.", 204

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=regime_timeseries.csv"},
    )


@app.route("/export/json")
@login_required
def export_json():
    rows = _db.export_all_timeseries(user_id=current_user.id)
    return Response(
        json.dumps(rows, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=regime_timeseries.json"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich_runs(raw_runs: list[dict]) -> list[dict]:
    for r in raw_runs:
        r["tickers_list"] = json.loads(r["tickers"])
        r["config_dict"]  = json.loads(r["config"])
        if r["created_at"] and r["completed_at"]:
            try:
                fmt = "%Y-%m-%d %H:%M:%S"
                start = datetime.strptime(r["created_at"],   fmt)
                end   = datetime.strptime(r["completed_at"], fmt)
                secs  = int((end - start).total_seconds())
                r["duration"] = f"{secs}s"
            except Exception:
                r["duration"] = "—"
        else:
            r["duration"] = "—"
    return raw_runs


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
