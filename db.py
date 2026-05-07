"""
SQLite persistence layer.

Tables
------
users            – registered accounts (hashed passwords)
analyses         – one row per submitted run (config + status)
analysis_results – per-ticker per-method summary (metrics, current regime, chart)
regime_timeseries – full daily time series stored for every completed run
                    (the raw material for future ML work)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regime_history.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT    UNIQUE NOT NULL,
    password_hash TEXT   NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    run_id       TEXT PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id),
    tickers      TEXT NOT NULL,   -- JSON list
    config       TEXT NOT NULL,   -- JSON dict of all params
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   TEXT NOT NULL,
    completed_at TEXT,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES analyses(run_id),
    ticker          TEXT NOT NULL,
    method          TEXT NOT NULL,
    metrics         TEXT NOT NULL,   -- JSON
    regime_id       INTEGER,
    regime_name     TEXT,
    regime_prob     REAL,
    chart_url       TEXT,
    transition_json TEXT,            -- JSON (split-orient)
    strategy_desc   TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regime_timeseries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    ticker       TEXT    NOT NULL,
    method       TEXT    NOT NULL,
    date         TEXT    NOT NULL,
    close        REAL,
    returns      REAL,
    volatility   REAL,
    trend_slope  REAL,
    zscore       REAL,
    rsi          REAL,
    atr          REAL,
    bb_width     REAL,
    vol_ratio    REAL,
    ret_autocorr REAL,
    regime       INTEGER,
    regime_prob  REAL,
    signal       REAL,
    strat_ret    REAL
);

CREATE INDEX IF NOT EXISTS idx_ts_run_ticker_method
    ON regime_timeseries (run_id, ticker, method);
CREATE INDEX IF NOT EXISTS idx_results_run ON analysis_results (run_id);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(username: str, password_hash: str) -> int:
    now = _now()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username, password_hash, now),
        )
        return cur.lastrowid


def get_user_by_username(username: str):
    with _conn() as con:
        return con.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()


def get_user_by_id(user_id: int):
    with _conn() as con:
        return con.execute(
            "SELECT * FROM users WHERE id=?", (user_id,)
        ).fetchone()


# ── Analyses ──────────────────────────────────────────────────────────────────

def insert_run(run_id: str, user_id: int, cfg: dict) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO analyses
               (run_id, user_id, tickers, config, status, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                run_id,
                user_id,
                json.dumps(cfg["tickers"]),
                json.dumps(cfg),
                "pending",
                _now(),
            ),
        )


def update_run_status(run_id: str, status: str, error: str | None = None) -> None:
    completed = _now() if status in ("done", "error") else None
    with _conn() as con:
        con.execute(
            """UPDATE analyses
               SET status=?, completed_at=?, error=?
               WHERE run_id=?""",
            (status, completed, error, run_id),
        )


def get_run(run_id: str):
    with _conn() as con:
        return con.execute(
            "SELECT * FROM analyses WHERE run_id=?", (run_id,)
        ).fetchone()


def list_runs(user_id: int | None = None, limit: int = 200):
    with _conn() as con:
        if user_id is not None:
            rows = con.execute(
                """SELECT a.*, u.username
                   FROM analyses a LEFT JOIN users u ON a.user_id=u.id
                   WHERE a.user_id=?
                   ORDER BY a.created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT a.*, u.username
                   FROM analyses a LEFT JOIN users u ON a.user_id=u.id
                   ORDER BY a.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ── Per-ticker results ────────────────────────────────────────────────────────

def insert_result(run_id: str, ticker: str, method: str, data: dict) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO analysis_results
               (run_id, ticker, method, metrics, regime_id, regime_name,
                regime_prob, chart_url, transition_json, strategy_desc, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                ticker,
                method,
                json.dumps(data["metrics"]),
                data["regime_id"],
                data["regime_name"],
                data["regime_prob"],
                data["chart_url"],
                json.dumps(data["transition"]),
                data["strategy_desc"],
                _now(),
            ),
        )


def get_results_for_run(run_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM analysis_results WHERE run_id=? ORDER BY ticker, method",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Time series ───────────────────────────────────────────────────────────────

def insert_timeseries(run_id: str, ticker: str, method: str, df) -> None:
    """
    Bulk-insert the daily time series DataFrame into regime_timeseries.
    df must have columns matching the table schema (index = date).
    """
    cols = [
        "close", "returns", "volatility", "trend_slope", "zscore",
        "rsi", "atr", "bb_width", "vol_ratio", "ret_autocorr",
        "regime", "regime_prob", "signal", "strat_ret",
    ]
    col_map = {
        "close":        "Close",
        "returns":      "returns",
        "volatility":   "volatility",
        "trend_slope":  "trend_slope",
        "zscore":       "zscore",
        "rsi":          "rsi",
        "atr":          "atr",
        "bb_width":     "bb_width",
        "vol_ratio":    "vol_ratio",
        "ret_autocorr": "ret_autocorr",
        "regime":       "regime",
        "regime_prob":  "regime_prob",
        "signal":       "signal",
        "strat_ret":    "strat_ret",
    }
    rows = []
    for date, row in df.iterrows():
        r = [run_id, ticker, method, str(date.date())]
        for col in cols:
            src = col_map[col]
            val = row.get(src, None)
            r.append(float(val) if val is not None and val == val else None)
        rows.append(tuple(r))

    placeholders = ",".join(["?"] * (4 + len(cols)))
    with _conn() as con:
        con.executemany(
            f"""INSERT INTO regime_timeseries
                (run_id, ticker, method, date,
                 close, returns, volatility, trend_slope, zscore,
                 rsi, atr, bb_width, vol_ratio, ret_autocorr,
                 regime, regime_prob, signal, strat_ret)
                VALUES ({placeholders})""",
            rows,
        )


def get_timeseries(run_id: str, ticker: str, method: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM regime_timeseries
               WHERE run_id=? AND ticker=? AND method=?
               ORDER BY date""",
            (run_id, ticker, method),
        ).fetchall()
        return [dict(r) for r in rows]


def export_all_timeseries(user_id: int | None = None) -> list[dict]:
    """Return all stored time series rows (optionally filtered by user)."""
    with _conn() as con:
        if user_id is not None:
            rows = con.execute(
                """SELECT ts.*, a.user_id
                   FROM regime_timeseries ts
                   JOIN analyses a ON ts.run_id = a.run_id
                   WHERE a.user_id = ?
                   ORDER BY ts.run_id, ts.ticker, ts.method, ts.date""",
                (user_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT ts.* FROM regime_timeseries ts
                   ORDER BY ts.run_id, ts.ticker, ts.method, ts.date"""
            ).fetchall()
        return [dict(r) for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
