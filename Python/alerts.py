# alerts.py
# ============================================================
# Standalone LEAP Setup alert engine.
#
# Run this script on a schedule — every 30 minutes during market
# hours — to scan the universe for LEAP Setup signals and send
# email alerts when conditions are met.
#
# This file does NOT import Streamlit.  It runs purely as a
# Python script and communicates only via SQLite and email.
#
# HOW TO RUN:
#   python alerts.py
#
# HOW TO SCHEDULE (macOS/Linux cron):
#   */30 9-16 * * 1-5 /path/to/venv/bin/python /path/to/alerts.py
#   (every 30 mins between 9am-4pm, Monday-Friday)
#
# HOW TO SCHEDULE (cloud — see .env.example for required vars):
#   Add to your server's crontab or use a cloud scheduler service.
# ============================================================

import os
import sys
import smtplib
import yfinance as yf
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, timedelta, datetime

# Load .env file so environment variables are available even when running
# the script directly from the terminal (not through Streamlit).
from dotenv import load_dotenv
load_dotenv()

# Import shared modules — these live in the parent directory.
# We add the parent directory to sys.path so Python can find them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    initialize_database, get_universe, get_presets,
    log_alert, alert_fired_recently,
)
from indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    evaluate_leap_conditions,
)

# ── Environment variable loading ─────────────────────────────
# All sensitive values come from the .env file.
# Never hardcode email addresses or passwords in source code.
ALERT_EMAIL     = os.environ.get("ALERT_EMAIL", "")
ALERT_PASSWORD  = os.environ.get("ALERT_PASSWORD", "")
ALERT_RECIPIENT = os.environ.get("ALERT_RECIPIENT", "")


def download_data(ticker: str) -> pd.DataFrame:
    """
    Download 1 year of daily OHLCV data for a ticker from Yahoo Finance.

    1 year is enough for the signal conditions we check:
    - RSI needs ~14 days warm-up
    - 200-day MA needs 200 days
    - 52-week high needs 252 days

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        pd.DataFrame: OHLCV data indexed by date, or empty DataFrame on error.
    """
    try:
        today = date.today()
        start = today - timedelta(days=400)   # 400 days to ensure full 200-day MA
        df = yf.download(ticker, start=str(start), end=str(today),
                         auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0)
        df.index   = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"  [ERROR] Could not download {ticker}: {e}")
        return pd.DataFrame()


def build_confidence_summary(ticker: str, signal_mask: pd.Series, close: pd.Series) -> str:
    """
    Calculate a quick historical confidence statistic for the LEAP Setup.

    Looks at every historical signal hit and checks what the stock did
    120 trading days later.  Returns a one-line summary for the email body.

    Args:
        ticker (str): Ticker symbol for context.
        signal_mask (pd.Series): Boolean Series — True on signal hit dates.
        close (pd.Series): Full closing price series.

    Returns:
        str: A plain-english confidence summary sentence.
    """
    import numpy as np
    hits        = close.index[signal_mask]
    returns_120 = []

    for hit_date in hits:
        pos = close.index.get_loc(hit_date)
        if pos + 120 < len(close):
            ret = (float(close.iloc[pos + 120]) / float(close.iloc[pos]) - 1) * 100
            returns_120.append(ret)

    if not returns_120:
        return "No historical 120-day outcome data available yet."

    import numpy as np
    avg      = np.mean(returns_120)
    win_rate = sum(1 for r in returns_120 if r > 0) / len(returns_120) * 100
    return (
        f"This setup has fired {len(hits)} times on {ticker} over the available history. "
        f"Average 120-day return: {avg:+.1f}%. Win rate: {win_rate:.0f}%."
    )


def send_email_alert(
    ticker: str,
    price: float,
    conditions_text: str,
    confidence: str,
) -> bool:
    """
    Send an email alert via Gmail SMTP.

    Gmail requires an App Password (not your regular Gmail password).
    Generate one at myaccount.google.com/apppasswords with 2-Step Verification enabled.

    SMTP (Simple Mail Transfer Protocol) is the standard protocol for sending email.
    We connect to Gmail's SMTP server on port 587, start TLS encryption, log in
    with the app password, and send the message.

    Args:
        ticker (str): The ticker that triggered the alert.
        price (float): Current closing price.
        conditions_text (str): Human-readable list of conditions that fired.
        confidence (str): Historical confidence summary sentence.

    Returns:
        bool: True if sent successfully, False otherwise.
    """
    if not all([ALERT_EMAIL, ALERT_PASSWORD, ALERT_RECIPIENT]):
        print("  [SKIP] Email credentials not configured in .env")
        return False

    subject = f"LEAP Signal Alert — {ticker}"

    # Build a clean HTML email body
    body_html = f"""
    <html><body style="font-family: Arial, sans-serif; color: #0f2044; max-width: 600px;">
    <h2 style="color: #2596be;">⚡ LEAP Setup Signal: {ticker}</h2>
    <table style="width: 100%; border-collapse: collapse;">
      <tr>
        <td style="padding: 8px; background: #f0f4f8; font-weight: bold;">Ticker</td>
        <td style="padding: 8px;">{ticker}</td>
      </tr>
      <tr>
        <td style="padding: 8px; background: #f0f4f8; font-weight: bold;">Price at Alert</td>
        <td style="padding: 8px;">${price:,.2f}</td>
      </tr>
      <tr>
        <td style="padding: 8px; background: #f0f4f8; font-weight: bold;">Conditions Triggered</td>
        <td style="padding: 8px;">{conditions_text}</td>
      </tr>
      <tr>
        <td style="padding: 8px; background: #f0f4f8; font-weight: bold;">Historical Confidence</td>
        <td style="padding: 8px;">{confidence}</td>
      </tr>
      <tr>
        <td style="padding: 8px; background: #f0f4f8; font-weight: bold;">Timestamp</td>
        <td style="padding: 8px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
      </tr>
    </table>
    <p style="color: #94a3b8; font-size: 12px; margin-top: 20px;">
      Stock Dashboard · LEAP Alert Engine · This is not financial advice.
    </p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = ALERT_EMAIL
    msg["To"]      = ALERT_RECIPIENT
    msg.attach(MIMEText(body_html, "html"))

    try:
        # Port 587 is the standard Gmail SMTP submission port.
        # starttls() upgrades the plain-text connection to an encrypted one.
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(ALERT_EMAIL, ALERT_PASSWORD)
            server.sendmail(ALERT_EMAIL, ALERT_RECIPIENT, msg.as_string())
        print(f"  [EMAIL SENT] Alert for {ticker} sent to {ALERT_RECIPIENT}")
        return True
    except Exception as e:
        print(f"  [EMAIL ERROR] Could not send alert for {ticker}: {e}")
        return False


def run_alerts() -> None:
    """
    Main function: scan the universe and fire LEAP Setup alerts.

    Steps:
    1. Connect to the database and read the universe.
    2. Load the built-in LEAP Setup conditions from the signal_presets table.
    3. For each ticker: download data, calculate indicators, evaluate conditions.
    4. If all conditions fire on the most recent trading day:
       a. Check whether we already sent an alert for this ticker in the last 24 hours.
       b. If not, log the alert and send an email.

    This function is designed to be called every 30 minutes during market hours.
    """
    initialize_database()

    print(f"\n{'='*55}")
    print(f"  LEAP Alert Engine — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")

    universe = get_universe()
    print(f"  Universe: {', '.join(universe)}")

    # Load the LEAP Setup preset from the database.
    # If for some reason it was deleted, use the hardcoded defaults below.
    presets = get_presets()
    leap_preset = next((p for p in presets if p["name"] == "LEAP Setup"), None)

    if leap_preset:
        conditions = leap_preset["conditions"]
        print(f"  Using preset: {leap_preset['name']}")
    else:
        # Default LEAP Setup conditions as a fallback
        conditions = {
            "rsi_below":      {"enabled": True,  "value": 35},
            "near_200ma":     {"enabled": True,  "value": 2.0},
            "down_from_52w":  {"enabled": True,  "value": 10.0},
            "vol_multiplier": {"enabled": True,  "value": 1.5},
        }
        print("  Using hardcoded LEAP Setup defaults (no preset found).")

    alerts_fired = 0

    for ticker in universe:
        print(f"\n  Scanning {ticker}…")

        # Step 1: Download data
        df = download_data(ticker)
        if df.empty or len(df) < 50:
            print(f"    Skipping — insufficient data ({len(df)} rows)")
            continue

        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()

        # Step 2: Calculate indicators
        rsi_s            = calculate_rsi(close, 14)
        macd_l, sig_l, _ = calculate_macd(close, 12, 26, 9)
        _, _, bb_lo      = calculate_bollinger_bands(close, 20, 2.0)

        # Step 3: Evaluate conditions on every row, get the full signal mask
        signal_mask = evaluate_leap_conditions(df, conditions, rsi_s, macd_l, sig_l, bb_lo)

        # Step 4: Check only the MOST RECENT trading day
        # iloc[-1] means "the last row" — the most recent date in the dataset
        if not signal_mask.iloc[-1]:
            print(f"    No signal today.")
            continue

        # All conditions fired on the most recent day!
        latest_price = float(close.iloc[-1])

        # Step 5: Build a human-readable list of what triggered
        fired = []
        if conditions.get("rsi_below", {}).get("enabled"):
            rsi_val = float(rsi_s.iloc[-1])
            fired.append(f"RSI {rsi_val:.1f} < {conditions['rsi_below']['value']}")
        if conditions.get("near_200ma", {}).get("enabled"):
            ma200    = float(close.rolling(200).mean().iloc[-1])
            pct_from = abs(latest_price - ma200) / ma200 * 100
            fired.append(f"Price {pct_from:.1f}% from 200-day MA (${ma200:,.2f})")
        if conditions.get("down_from_52w", {}).get("enabled"):
            high52  = float(close.rolling(252).max().iloc[-1])
            dd      = (high52 - latest_price) / high52 * 100
            fired.append(f"Down {dd:.1f}% from 52W high (${high52:,.2f})")
        if conditions.get("vol_multiplier", {}).get("enabled"):
            avg_vol = float(volume.rolling(20).mean().iloc[-1])
            today_vol = float(volume.iloc[-1])
            fired.append(f"Volume {today_vol/avg_vol:.1f}× above 20-day avg")

        conditions_text = " | ".join(fired)
        print(f"    ⚡ SIGNAL FIRED: {conditions_text}")
        print(f"    Price: ${latest_price:,.2f}")

        # Step 6: Check for duplicate alert in last 24 hours
        if alert_fired_recently(ticker, hours=24):
            print(f"    [SKIP] Alert already sent for {ticker} within 24 hours.")
            continue

        # Step 7: Calculate confidence summary for email
        confidence = build_confidence_summary(ticker, signal_mask, close)
        print(f"    Confidence: {confidence}")

        # Step 8: Log to database
        log_alert(ticker, conditions_text, latest_price)
        print(f"    [LOGGED] Alert written to database.")

        # Step 9: Send email
        sent = send_email_alert(ticker, latest_price, conditions_text, confidence)
        alerts_fired += 1

    print(f"\n{'='*55}")
    print(f"  Scan complete. Alerts fired: {alerts_fired}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    # This block runs only when you execute: python alerts.py
    # It will not run when alerts.py is imported by another module.
    run_alerts()
