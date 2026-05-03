# database.py
# ============================================================
# Central SQLite database layer for the Stock Dashboard.
#
# SQLite is a file-based database built into Python — no server
# needed. Every table lives in a single file: dashboard.db.
# All pages and scripts import from here so the schema is defined
# exactly once and stays consistent everywhere.
#
# Usage from any file:
#   from database import get_connection, seed_universe, get_universe
# ============================================================

import sqlite3
import json
import os
from datetime import datetime

# ------------------------------------------------------------
# DATABASE FILE PATH
# The database file lives next to this script.
# os.path.dirname(__file__) means "the folder this file is in".
# ------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "dashboard.db")

# Default universe of tickers to seed on first run.
DEFAULT_UNIVERSE = [
    "SPY", "VOO", "AAPL", "MSFT", "GOOGL",
    "AMZN", "NVDA", "META", "TSLA", "WMT", "CHRW",
]


def get_connection() -> sqlite3.Connection:
    """
    Open and return a connection to the SQLite database.

    SQLite connections are not thread-safe by default, so we use
    check_same_thread=False which is safe for Streamlit's single-user model.

    Returns:
        sqlite3.Connection: An open connection to dashboard.db.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # row_factory makes rows behave like dictionaries so you can do
    # row["ticker"] instead of row[0].
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """
    Create all required tables if they do not already exist.

    This is safe to call on every app startup — CREATE TABLE IF NOT EXISTS
    means it only creates the table the very first time and skips it after.
    Call this once at the top of every page.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── universe table ──────────────────────────────────────
    # Stores the list of tickers the user is tracking.
    # source is either "manual" (user typed it) or "finviz" (screener added it).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS universe (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT UNIQUE NOT NULL,
            added_date  TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'manual'
        )
    """)

    # ── signal_presets table ────────────────────────────────
    # Stores named sets of signal conditions the user has saved.
    # conditions_json is a JSON string that encodes a Python dictionary
    # of condition names and their on/off state plus numeric values.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_presets (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT UNIQUE NOT NULL,
            conditions_json  TEXT NOT NULL,
            created_date     TEXT NOT NULL
        )
    """)

    # ── alert_log table ─────────────────────────────────────
    # Every time alerts.py fires a LEAP signal, it writes a row here.
    # This lets the Command Center page show a history of alerts.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_log (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker               TEXT NOT NULL,
            conditions_triggered TEXT NOT NULL,
            price_at_alert       REAL NOT NULL,
            timestamp            TEXT NOT NULL
        )
    """)

    # ── paper_trades table ──────────────────────────────────
    # Records every paper trade the user logs through the UI.
    # status is either "open" (position still held) or "closed".
    # strike_price and expiration_date are only filled for LEAP trades.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker             TEXT NOT NULL,
            trade_type         TEXT NOT NULL,
            entry_price        REAL NOT NULL,
            shares_or_contracts INTEGER NOT NULL,
            strike_price       REAL,
            expiration_date    TEXT,
            entry_date         TEXT NOT NULL,
            notes              TEXT,
            status             TEXT NOT NULL DEFAULT 'open',
            exit_price         REAL,
            exit_date          TEXT
        )
    """)

    # ── iq_theses table ─────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iq_theses (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            title              TEXT NOT NULL,
            logical_chain      TEXT NOT NULL,
            affected_industries TEXT NOT NULL,
            stock_names        TEXT NOT NULL,
            confidence         INTEGER NOT NULL DEFAULT 50,
            status             TEXT NOT NULL DEFAULT 'active',
            timeframe          TEXT NOT NULL DEFAULT '3-6 months',
            created_date       TEXT NOT NULL,
            last_updated       TEXT NOT NULL,
            run_count          INTEGER NOT NULL DEFAULT 1,
            postmortem         TEXT
        )
    """)

    # ── watchlist_pins table ─────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_pins (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT UNIQUE NOT NULL,
            reason      TEXT NOT NULL DEFAULT 'manual',
            added_date  TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    # Seed universe with defaults on first run
    seed_universe()

    # Seed the built-in LEAP Setup preset on first run
    seed_leap_preset()


def seed_universe() -> None:
    """
    Insert the default tickers into the universe table if the table is empty.

    This only runs once — if any tickers already exist, it does nothing.
    """
    conn = get_connection()
    cursor = conn.cursor()

    existing = cursor.execute("SELECT COUNT(*) as cnt FROM universe").fetchone()["cnt"]
    if existing == 0:
        today = datetime.now().strftime("%Y-%m-%d")
        for tkr in DEFAULT_UNIVERSE:
            cursor.execute(
                "INSERT OR IGNORE INTO universe (ticker, added_date, source) VALUES (?, ?, ?)",
                (tkr.upper(), today, "default"),
            )
        conn.commit()
    conn.close()


def seed_leap_preset() -> None:
    """
    Insert the built-in LEAP Setup preset if it does not already exist.

    The LEAP Setup looks for stocks that have pulled back significantly
    from their highs and are sitting near long-term support (200-day MA),
    with oversold momentum (RSI below 35) and above-average volume.
    """
    leap_conditions = {
        "rsi_below":         {"enabled": True,  "value": 35},
        "rsi_above":         {"enabled": False, "value": 70},
        "near_200ma":        {"enabled": True,  "value": 2.0},
        "near_50ma":         {"enabled": False, "value": 2.0},
        "down_from_52w":     {"enabled": True,  "value": 10.0},
        "vol_multiplier":    {"enabled": True,  "value": 1.5},
        "bb_lower_touch":    {"enabled": False},
        "macd_bullish":      {"enabled": False},
    }
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO signal_presets (name, conditions_json, created_date) VALUES (?, ?, ?)",
        ("LEAP Setup", json.dumps(leap_conditions), datetime.now().strftime("%Y-%m-%d")),
    )
    conn.commit()
    conn.close()


# ============================================================
# UNIVERSE CRUD
# ============================================================

def get_universe() -> list[str]:
    """
    Return the current list of ticker symbols from the universe table.

    Returns:
        list[str]: Ticker symbols in the order they were added, e.g. ["SPY", "AAPL", ...].
    """
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM universe ORDER BY id").fetchall()
    conn.close()
    return [r["ticker"] for r in rows]


def add_ticker(ticker: str, source: str = "manual") -> bool:
    """
    Add a new ticker to the universe. Silently ignores duplicates.

    Args:
        ticker (str): The stock ticker symbol, e.g. "AAPL".
        source (str): Where this ticker came from — "manual" or "finviz".

    Returns:
        bool: True if the ticker was newly added, False if it already existed.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO universe (ticker, added_date, source) VALUES (?, ?, ?)",
            (ticker.upper(), datetime.now().strftime("%Y-%m-%d"), source),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # UNIQUE constraint failed — ticker already exists
        return False
    finally:
        conn.close()


def remove_ticker(ticker: str) -> None:
    """
    Remove a ticker from the universe table.

    Args:
        ticker (str): The ticker to remove, e.g. "AAPL".
    """
    conn = get_connection()
    conn.execute("DELETE FROM universe WHERE ticker = ?", (ticker.upper(),))
    conn.commit()
    conn.close()


# ============================================================
# SIGNAL PRESETS CRUD
# ============================================================

def get_presets() -> list[dict]:
    """
    Return all saved signal presets as a list of dictionaries.

    Each dict has keys: id, name, conditions (already parsed from JSON),
    created_date.

    Returns:
        list[dict]: All presets, newest first.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, conditions_json, created_date FROM signal_presets ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "conditions": json.loads(r["conditions_json"]),
            "created_date": r["created_date"],
        }
        for r in rows
    ]


def save_preset(name: str, conditions: dict) -> None:
    """
    Save or overwrite a named signal preset.

    Args:
        name (str): The human-readable name, e.g. "Dip Buyer".
        conditions (dict): The condition configuration dictionary.
    """
    conn = get_connection()
    conn.execute(
        """INSERT INTO signal_presets (name, conditions_json, created_date)
           VALUES (?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET conditions_json=excluded.conditions_json""",
        (name, json.dumps(conditions), datetime.now().strftime("%Y-%m-%d")),
    )
    conn.commit()
    conn.close()


# ============================================================
# ALERT LOG
# ============================================================

def log_alert(ticker: str, conditions: str, price: float) -> None:
    """
    Write a fired alert to the alert_log table.

    Args:
        ticker (str): The ticker that triggered the alert, e.g. "NVDA".
        conditions (str): Human-readable description of which conditions fired.
        price (float): The closing price at the time of the alert.
    """
    conn = get_connection()
    conn.execute(
        "INSERT INTO alert_log (ticker, conditions_triggered, price_at_alert, timestamp) VALUES (?,?,?,?)",
        (ticker, conditions, price, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_recent_alerts(limit: int = 20) -> list[dict]:
    """
    Return the most recent N alerts from the alert_log table.

    Args:
        limit (int): How many alerts to return. Defaults to 20.

    Returns:
        list[dict]: Alert rows, newest first.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM alert_log ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def alert_fired_recently(ticker: str, hours: int = 24) -> bool:
    """
    Check whether an alert for this ticker already fired within the last N hours.

    This prevents the alert engine from sending duplicate emails for the same
    setup on the same day.

    Args:
        ticker (str): The ticker to check.
        hours (int): How many hours back to look. Default 24.

    Returns:
        bool: True if an alert was already sent recently, False otherwise.
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM alert_log WHERE ticker=? AND timestamp >= ?",
        (ticker, cutoff),
    ).fetchone()
    conn.close()
    return row["cnt"] > 0


# ============================================================
# PAPER TRADES CRUD
# ============================================================

def add_paper_trade(
    ticker: str,
    trade_type: str,
    entry_price: float,
    shares_or_contracts: int,
    entry_date: str,
    notes: str = "",
    strike_price: float = None,
    expiration_date: str = None,
) -> int:
    """
    Insert a new paper trade into the database.

    Args:
        ticker (str): Stock ticker symbol.
        trade_type (str): "Stock" or "LEAP".
        entry_price (float): Price paid per share or per contract.
        shares_or_contracts (int): Number of shares or option contracts.
        entry_date (str): Trade date as YYYY-MM-DD string.
        notes (str): Optional free-text notes about the trade rationale.
        strike_price (float): For LEAP trades only — the option strike price.
        expiration_date (str): For LEAP trades only — option expiration as YYYY-MM-DD.

    Returns:
        int: The newly inserted row's database ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO paper_trades
           (ticker, trade_type, entry_price, shares_or_contracts, entry_date,
            notes, status, strike_price, expiration_date)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            ticker.upper(), trade_type, entry_price, shares_or_contracts,
            entry_date, notes, "open", strike_price, expiration_date,
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_open_trades() -> list[dict]:
    """
    Return all paper trades with status "open".

    Returns:
        list[dict]: Open trades as dictionaries, oldest entry first.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM paper_trades WHERE status='open' ORDER BY entry_date ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closed_trades() -> list[dict]:
    """
    Return all paper trades with status "closed".

    Returns:
        list[dict]: Closed trades as dictionaries, newest exit first.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM paper_trades WHERE status='closed' ORDER BY exit_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def close_paper_trade(trade_id: int, exit_price: float, exit_date: str) -> None:
    """
    Mark an open trade as closed and record the exit price and date.

    Args:
        trade_id (int): The database ID of the trade to close.
        exit_price (float): Price at which the position was exited.
        exit_date (str): Exit date as YYYY-MM-DD string.
    """
    conn = get_connection()
    conn.execute(
        "UPDATE paper_trades SET status='closed', exit_price=?, exit_date=? WHERE id=?",
        (exit_price, exit_date, trade_id),
    )
    conn.commit()
    conn.close()


def get_recent_trades_for_digest(days: int = 7) -> list[dict]:
    """
    Return all trades opened or closed in the last N days, for the weekly digest.

    Args:
        days (int): How many days back to look. Default 7.

    Returns:
        list[dict]: Matching trades as dictionaries.
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM paper_trades
           WHERE entry_date >= ? OR (exit_date IS NOT NULL AND exit_date >= ?)
           ORDER BY entry_date DESC""",
        (cutoff, cutoff),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# IQ THESES CRUD
# ============================================================

def save_iq_thesis(
    title: str,
    logical_chain: str,
    affected_industries: str,
    stock_names: str,
    confidence: int,
    timeframe: str,
) -> int:
    """
    Insert a new IQ thesis or increment run_count if same title exists.

    Returns:
        int: Row ID of the inserted or updated thesis.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing = cursor.execute(
        "SELECT id, run_count FROM iq_theses WHERE title=? AND status='active'",
        (title,)
    ).fetchone()
    if existing:
        cursor.execute(
            "UPDATE iq_theses SET run_count=?, last_updated=?, logical_chain=?, "
            "affected_industries=?, stock_names=?, confidence=?, timeframe=? WHERE id=?",
            (existing["run_count"] + 1, now, logical_chain, affected_industries,
             stock_names, confidence, timeframe, existing["id"]),
        )
        row_id = existing["id"]
    else:
        cursor.execute(
            """INSERT INTO iq_theses
               (title, logical_chain, affected_industries, stock_names,
                confidence, status, timeframe, created_date, last_updated, run_count)
               VALUES (?,?,?,?,?,?,?,?,?,1)""",
            (title, logical_chain, affected_industries, stock_names,
             confidence, "active", timeframe, now, now),
        )
        row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_iq_theses(status: str = "active") -> list[dict]:
    """Return all theses with the given status, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM iq_theses WHERE status=? ORDER BY last_updated DESC",
        (status,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_iq_thesis(thesis_id: int, postmortem: str) -> None:
    """Mark a thesis as resolved and record a postmortem note."""
    conn = get_connection()
    conn.execute(
        "UPDATE iq_theses SET status='resolved', postmortem=?, last_updated=? WHERE id=?",
        (postmortem, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), thesis_id),
    )
    conn.commit()
    conn.close()


def get_all_iq_stock_names() -> list[str]:
    """Return a flat deduplicated list of tickers mentioned in active theses."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT stock_names FROM iq_theses WHERE status='active'"
    ).fetchall()
    conn.close()
    tickers = []
    for r in rows:
        for t in r["stock_names"].split(","):
            t = t.strip().upper()
            if t and t not in tickers:
                tickers.append(t)
    return tickers


# ============================================================
# WATCHLIST PINS CRUD
# ============================================================

def get_watchlist_pins() -> list[dict]:
    """Return all manually pinned watchlist entries."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM watchlist_pins ORDER BY added_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_watchlist_pin(ticker: str, reason: str = "manual") -> bool:
    """Pin a ticker to the watchlist. Returns True if newly added."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO watchlist_pins (ticker, reason, added_date) VALUES (?,?,?)",
            (ticker.upper(), reason, datetime.now().strftime("%Y-%m-%d")),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def remove_watchlist_pin(ticker: str) -> None:
    """Remove a ticker from manual watchlist pins."""
    conn = get_connection()
    conn.execute("DELETE FROM watchlist_pins WHERE ticker=?", (ticker.upper(),))
    conn.commit()
    conn.close()


def get_recent_alerts_for_digest(days: int = 7) -> list[dict]:
    """
    Return alert_log rows from the last N days, for the weekly digest.

    Args:
        days (int): How many days back to look. Default 7.

    Returns:
        list[dict]: Alert rows as dictionaries.
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM alert_log WHERE timestamp >= ? ORDER BY timestamp DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
