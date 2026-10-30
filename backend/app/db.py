"""SQLite storage: schema, connections and first-start loading of the demo transactions."""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger("stdetector.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  alert_id       INTEGER PRIMARY KEY,
  transaction_id TEXT UNIQUE NOT NULL,
  customer_id    TEXT NOT NULL,
  score          REAL NOT NULL,
  band           TEXT NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('NEW','NOTIFIED','CONFIRMED_NOT_ME','CONFIRMED_LEGIT')),
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

CREATE TABLE IF NOT EXISTS notifications (
  id          INTEGER PRIMARY KEY,
  alert_id    INTEGER NOT NULL REFERENCES alerts(alert_id),
  token_hash  TEXT UNIQUE NOT NULL,
  expires_at  TEXT NOT NULL,
  used_at     TEXT,
  sent_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS otps (
  id          INTEGER PRIMARY KEY,
  customer_id TEXT NOT NULL,
  purpose     TEXT NOT NULL CHECK (purpose IN ('confirm_legit','unfreeze')),
  alert_id    INTEGER,
  code_hash   TEXT NOT NULL,
  expires_at  TEXT NOT NULL,
  attempts    INTEGER NOT NULL DEFAULT 0,
  used_at     TEXT,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
  customer_id   TEXT PRIMARY KEY,
  status        TEXT NOT NULL CHECK (status IN ('active','frozen')),
  frozen_at     TEXT,
  frozen_reason TEXT,
  updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
  case_id           INTEGER PRIMARY KEY,
  alert_id          INTEGER UNIQUE NOT NULL REFERENCES alerts(alert_id),
  customer_id       TEXT NOT NULL,
  transaction_id    TEXT NOT NULL,
  status            TEXT NOT NULL CHECK (status IN
                      ('OPEN','INVESTIGATING','SHADOW_CREDITED','RESOLVED_REFUNDED','RESOLVED_REJECTED')),
  notified_at       TEXT NOT NULL,
  reported_at       TEXT NOT NULL,
  zero_liability    INTEGER NOT NULL,
  shadow_credit_due TEXT NOT NULL,
  resolution_due    TEXT NOT NULL,
  resolution        TEXT,
  updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);

CREATE TABLE IF NOT EXISTS case_actions (
  id         INTEGER PRIMARY KEY,
  case_id    INTEGER NOT NULL REFERENCES cases(case_id),
  action     TEXT NOT NULL,
  actor      TEXT NOT NULL CHECK (actor IN ('system','customer','bank_ops')),
  note       TEXT,
  created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS case_actions_no_update BEFORE UPDATE ON case_actions
  BEGIN SELECT RAISE(ABORT, 'case_actions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS case_actions_no_delete BEFORE DELETE ON case_actions
  BEGIN SELECT RAISE(ABORT, 'case_actions is append-only'); END;

CREATE TABLE IF NOT EXISTS feedback (
  transaction_id TEXT PRIMARY KEY,
  customer_id    TEXT NOT NULL,
  response       TEXT NOT NULL CHECK (response IN ('not_me','was_me')),
  created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trusted_patterns (
  id              INTEGER PRIMARY KEY,
  customer_id     TEXT NOT NULL,
  pattern_key     TEXT NOT NULL,
  description     TEXT NOT NULL,
  source_alert_id INTEGER NOT NULL,
  created_at      TEXT NOT NULL,
  expires_at      TEXT NOT NULL,
  revoked_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_trusted_customer ON trusted_patterns(customer_id);

CREATE TABLE IF NOT EXISTS outbox (
  id         INTEGER PRIMARY KEY,
  channel    TEXT NOT NULL CHECK (channel IN ('email','sms')),
  recipient  TEXT NOT NULL,
  customer_id TEXT,
  subject    TEXT,
  body       TEXT NOT NULL,
  delivery   TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency (
  key         TEXT NOT NULL,
  endpoint    TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response    TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  PRIMARY KEY (key, endpoint)
);

CREATE TABLE IF NOT EXISTS nemotron_generations (
  id              INTEGER PRIMARY KEY,
  transaction_id  TEXT NOT NULL,
  generation_type TEXT NOT NULL CHECK (generation_type IN ('customer_message', 'case_memo')),
  source          TEXT NOT NULL CHECK (source IN ('nemotron', 'template')),
  model           TEXT NOT NULL,
  output_text     TEXT NOT NULL,
  latency_ms      REAL NOT NULL,
  tokens          INTEGER,
  verified        INTEGER NOT NULL,
  leakage_passed  INTEGER NOT NULL,
  fallback_reason TEXT,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nemotron_txn ON nemotron_generations(transaction_id);
"""

_local = threading.local()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """One atomic write transaction. BEGIN IMMEDIATE takes the write lock up front, so two
        concurrent clicks on the same link are serialised instead of racing."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.read() as conn:
            conn.executescript(SCHEMA)

    def transactions_loaded(self) -> bool:
        with self.read() as conn:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'").fetchone()
            if not row:
                return False
            return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] > 0

    def load_transactions(self, csv_path: Path, max_rows: int = 0) -> int:
        df = pd.read_csv(csv_path)
        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df.sort_values(["ts", "transaction_id"], kind="mergesort").reset_index(drop=True)
        if max_rows and len(df) > max_rows:
            idx = np.linspace(0, len(df) - 1, max_rows).round().astype(int)
            df = df.iloc[np.unique(idx)].reset_index(drop=True)
        if "anomaly_pct" in df:
            df["anomaly_pct"] = df["anomaly_pct"].clip(upper=100.0)   # percentile scale is 0-100
        df["source"] = "dataset"
        df["status"] = "completed"
        with self.read() as conn:
            conn.execute("DROP TABLE IF EXISTS transactions")
            df.head(0).to_sql("transactions", conn, index=False)
            df.to_sql("transactions", conn, index=False, if_exists="append", chunksize=20_000)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_txn_id ON transactions(transaction_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_customer_ts ON transactions(customer_id, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_ts ON transactions(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_alert_score ON transactions(alert, score DESC)")
        log.info("loaded %d demo transactions into %s", len(df), self.path)
        return len(df)


def rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def clean(v):
    """JSON-safe scalar (NaN -> None, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v
