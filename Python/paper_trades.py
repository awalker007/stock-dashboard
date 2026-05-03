# paper_trades.py
# ============================================================
# Paper trading helper functions.
#
# This module is imported by pages/1_Command_Center.py to render
# the paper trading tracker section and generate the weekly digest.
#
# It does NOT import Streamlit, so it can also be called from
# scripts or tests without starting a Streamlit session.
# ============================================================

import os
from datetime import datetime, timedelta
import anthropic as _anthropic
from database import (
    get_open_trades, get_closed_trades,
    get_recent_alerts_for_digest, get_recent_trades_for_digest,
)


def generate_weekly_digest(api_key: str) -> str:
    """
    Ask Claude to write a weekly performance review of the trading activity.

    Gathers all alerts and trades from the past 7 days, plus current open
    positions, then passes everything to Claude for a plain-english summary.

    The digest covers:
    - Which signals fired and whether they were good setups in hindsight
    - How open positions are currently performing
    - What the model would have returned on paper vs what was actually logged
    - What to watch in the coming week from the current universe

    Args:
        api_key (str): Anthropic API key loaded from environment or sidebar.

    Returns:
        str: The weekly digest as a formatted string (HTML-safe plain text).
    """
    # ── Gather data from the last 7 days ─────────────────────
    recent_alerts = get_recent_alerts_for_digest(days=7)
    recent_trades = get_recent_trades_for_digest(days=7)
    open_positions = get_open_trades()

    # ── Build the context block passed to Claude ──────────────
    # We format everything as a readable text block so Claude can
    # understand the trading activity without needing structured parsing.

    alerts_text = "None" if not recent_alerts else "\n".join(
        f"- {a['ticker']} on {a['timestamp'][:10]}: {a['conditions_triggered']} at ${a['price_at_alert']:.2f}"
        for a in recent_alerts
    )

    trades_text = "None" if not recent_trades else "\n".join(
        f"- {t['ticker']} {t['trade_type']} {t['status']}: entered {t['entry_date']} @ ${t['entry_price']:.2f}"
        + (f", exited {t['exit_date']} @ ${t['exit_price']:.2f}" if t['exit_date'] else " (still open)")
        for t in recent_trades
    )

    # For open positions we want to show the unrealised P&L context
    open_text = "None" if not open_positions else "\n".join(
        f"- {t['ticker']} {t['trade_type']} entered {t['entry_date']} @ ${t['entry_price']:.2f} x {t['shares_or_contracts']}"
        + (f" (LEAP strike ${t['strike_price']:.2f} exp {t['expiration_date']})" if t['trade_type'] == "LEAP" else "")
        for t in open_positions
    )

    today_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""Today is {today_str}.  You are reviewing the past week of trading activity.

ALERTS THAT FIRED THIS WEEK (from LEAP Setup signal scanner):
{alerts_text}

TRADES OPENED OR CLOSED THIS WEEK:
{trades_text}

CURRENT OPEN POSITIONS:
{open_text}

Write a weekly performance review in plain english covering:
1. Which signals fired this week and whether they look like good setups in hindsight based on price action.
2. How the current open positions are performing relative to the entry thesis.
3. What the model would have returned on paper if every signal alert had been traded.
4. Two or three things to watch in the coming week based on what happened this week.

Be direct and specific. Reference actual tickers and prices. No filler sentences."""

    client = _anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
