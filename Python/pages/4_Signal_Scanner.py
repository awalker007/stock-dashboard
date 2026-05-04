# pages/4_Signal_Scanner.py
# ============================================================
# Signal Scanner — the most important analytical page in the app.
#
# This page lets you build a set of technical conditions, scan
# 5 years of historical price data to find every time those
# conditions all fired simultaneously, and then see what happened
# to the stock price after each of those historical signals.
#
# Layout (top to bottom):
#   1. Sidebar: ticker dropdown + universe manager
#   2. Preset selector (defaults to LEAP Setup)
#   3. Condition builder (8 conditions, 2 columns)
#   4. Run Signal Scan button (prominent)
#   5. Signal count summary + chart with signal markers
#   6. CONFIDENCE RATING CARD (hero output — large, prominent)
#   7. Outcome summary table (with correct forward return logic)
#   8. LEAP Optimizer panel
#
# NA FIX: Forward returns use iloc to look exactly N trading days
# ahead from each signal date. If not enough future days exist
# (signal fired too recently), a dash is shown. NaN prices are
# also caught and treated as unavailable. This eliminates the
# NA/nan display issue from the previous implementation.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import os
from dotenv import load_dotenv

load_dotenv()

from utils import inject_css, universe_manager_sidebar, card_header, section_label
from utils import ACCENT, BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG, HEADER_BLUE
from indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_historical_volatility, evaluate_leap_conditions,
)
from database import initialize_database, get_universe, get_presets, save_preset

st.set_page_config(page_title="Signal Scanner", page_icon="🔬", layout="wide")
initialize_database()
inject_css()

# ============================================================
# SIDEBAR — ticker selector + universe manager
# ============================================================

st.sidebar.markdown("## 🔬  SIGNAL SCANNER")

UNIVERSE = get_universe()
ticker = st.sidebar.selectbox("Ticker", UNIVERSE, key="scanner_ticker")

st.sidebar.markdown("---")
universe_manager_sidebar()

# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    f"<h1 style='margin-bottom:4px;'>🔬 Signal Scanner</h1>"
    f"<div style='color:{TEXT_LIGHT};font-size:0.82rem;margin-bottom:20px;'>"
    f"Historical signal analysis · {ticker} · {date.today().strftime('%A, %B %d, %Y')}</div>",
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOAD
# ============================================================


@st.cache_data(ttl=900)
def load_data(tkr: str) -> pd.DataFrame:
    """Download 5 years of daily OHLCV data — full history for accurate indicator math."""
    today = date.today()
    start = today - timedelta(days=365 * 5)
    df = yf.download(tkr, start=str(start), end=str(today), auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_localize(None)
    return df


df = load_data(ticker)

if df.empty:
    st.error(f"No data returned for {ticker}.")
    st.stop()

close  = df["Close"].squeeze()
volume = df["Volume"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
opens  = df["Open"].squeeze()

# ============================================================
# HOW TO USE — collapsible instructions (starts closed)
# ============================================================

with st.expander("ℹ️  How To Use The Signal Scanner", expanded=False):
    st.markdown(
        f"""<div style='background:#e8f4fd;border:1px solid #1d4ed8;border-radius:8px;padding:16px;'>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 8px 0;'>
<strong>What this page does:</strong> Define a set of technical conditions, run the scan, and see
every historical date when all conditions fired simultaneously on {ticker} over the last 5 years.
The Confidence Rating Card tells you whether this signal has historically been a reliable entry point.
</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 8px 0;'>
<strong>LEAP Setup preset:</strong> Fires when RSI &lt; 35 (oversold), price is within 2% of the
200-day MA (long-term support), price is down 10%+ from 52-week high (pullback), and volume is
above average (real participation). This combination identifies quality companies being sold off
by the broader market — the ideal LEAP call option entry.
</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0;'>
<strong>Outcome table:</strong> Each row is one historical signal hit. The return columns show
what the stock did over the next 5, 20, 60, and 120 trading days. A dash means not enough
future data is available yet (signal fired too recently). Green = positive, red = negative.
</p>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ============================================================
# PRESET SELECTOR — LEAP Setup is the default
# ============================================================

presets      = get_presets()
preset_names = [p["name"] for p in presets]

# Put LEAP Setup first so it is the default selection
if "LEAP Setup" in preset_names:
    preset_names = ["LEAP Setup"] + [p for p in preset_names if p != "LEAP Setup"]

sel_preset = st.selectbox(
    "Preset",
    options=preset_names if preset_names else ["LEAP Setup"],
    index=0,
    help="LEAP Setup is the flagship preset. Build your own and save it below.",
    key="preset_sel",
)

# Load preset conditions, falling back to LEAP Setup defaults
default_conditions = {
    "rsi_below":      {"enabled": True,  "value": 35},
    "rsi_above":      {"enabled": False, "value": 70},
    "near_200ma":     {"enabled": True,  "value": 2.0},
    "near_50ma":      {"enabled": False, "value": 2.0},
    "down_from_52w":  {"enabled": True,  "value": 10.0},
    "vol_multiplier": {"enabled": True,  "value": 1.5},
    "bb_lower_touch": {"enabled": False},
    "macd_bullish":   {"enabled": False},
}

if sel_preset:
    for p in presets:
        if p["name"] == sel_preset:
            default_conditions = p["conditions"]
            break

# ============================================================
# CONDITION BUILDER
# ============================================================

st.markdown(
    f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
    f"overflow:hidden;margin-bottom:12px;'>"
    f"{card_header('⚡  Condition Builder  — LEAP Setup is the flagship preset')}"
    f"<div style='padding:16px;'>",
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)

with col_a:
    c = default_conditions.get("rsi_below", {})
    rsi_below_on  = st.checkbox("RSI below threshold", value=c.get("enabled", True), key="cb_rsi_below")
    rsi_below_val = st.number_input("RSI below value", 1, 99, int(c.get("value", 35)), key="cv_rsi_below") if rsi_below_on else c.get("value", 35)

    c = default_conditions.get("near_200ma", {})
    near_200_on  = st.checkbox("Price within X% of 200-day MA", value=c.get("enabled", True), key="cb_200ma")
    near_200_val = st.number_input("% from 200-day MA", 0.1, 20.0, float(c.get("value", 2.0)), 0.5, key="cv_200ma") if near_200_on else c.get("value", 2.0)

    c = default_conditions.get("down_from_52w", {})
    down52_on  = st.checkbox("Price down X% from 52W high", value=c.get("enabled", True), key="cb_52w")
    down52_val = st.number_input("% drawdown from 52W high", 1.0, 70.0, float(c.get("value", 10.0)), 1.0, key="cv_52w") if down52_on else c.get("value", 10.0)

    c = default_conditions.get("bb_lower_touch", {})
    bb_touch_on = st.checkbox("Bollinger lower band touch", value=c.get("enabled", False), key="cb_bbtch")

with col_b:
    c = default_conditions.get("rsi_above", {})
    rsi_above_on  = st.checkbox("RSI above threshold", value=c.get("enabled", False), key="cb_rsi_above")
    rsi_above_val = st.number_input("RSI above value", 1, 99, int(c.get("value", 70)), key="cv_rsi_above") if rsi_above_on else c.get("value", 70)

    c = default_conditions.get("near_50ma", {})
    near_50_on  = st.checkbox("Price within X% of 50-day MA", value=c.get("enabled", False), key="cb_50ma")
    near_50_val = st.number_input("% from 50-day MA", 0.1, 20.0, float(c.get("value", 2.0)), 0.5, key="cv_50ma") if near_50_on else c.get("value", 2.0)

    c = default_conditions.get("vol_multiplier", {})
    vol_on  = st.checkbox("Volume X times above 20-day avg", value=c.get("enabled", True), key="cb_vol")
    vol_val = st.number_input("Volume multiplier", 0.5, 10.0, float(c.get("value", 1.5)), 0.25, key="cv_vol") if vol_on else c.get("value", 1.5)

    c = default_conditions.get("macd_bullish", {})
    macd_bull_on = st.checkbox("MACD bullish crossover", value=c.get("enabled", False), key="cb_macd_bull")

st.markdown("</div></div>", unsafe_allow_html=True)

# ── Save preset ──────────────────────────────────────────────
save_col1, save_col2 = st.columns([3, 1])
preset_name_input = save_col1.text_input("Save as preset name", placeholder="e.g. Dip Buyer", key="preset_name")
if save_col2.button("💾 Save Preset", key="save_preset_btn"):
    if preset_name_input.strip():
        conditions_to_save = {
            "rsi_below":      {"enabled": rsi_below_on,  "value": rsi_below_val},
            "rsi_above":      {"enabled": rsi_above_on,  "value": rsi_above_val},
            "near_200ma":     {"enabled": near_200_on,   "value": near_200_val},
            "near_50ma":      {"enabled": near_50_on,    "value": near_50_val},
            "down_from_52w":  {"enabled": down52_on,     "value": down52_val},
            "vol_multiplier": {"enabled": vol_on,        "value": vol_val},
            "bb_lower_touch": {"enabled": bb_touch_on},
            "macd_bullish":   {"enabled": macd_bull_on},
        }
        save_preset(preset_name_input.strip(), conditions_to_save)
        st.success(f"Preset '{preset_name_input.strip()}' saved.")
        st.rerun()

# ============================================================
# RUN SIGNAL SCAN BUTTON — prominent placement
# ============================================================

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
run_scan_col1, run_scan_col2, run_scan_col3 = st.columns([1, 2, 1])
with run_scan_col2:
    run_scan = st.button(
        "▶  Run Signal Scan",
        type="primary",
        use_container_width=True,
        key="run_scan_btn",
    )
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ============================================================
# SCAN RESULTS
# ============================================================

if run_scan or st.session_state.get("scan_ran"):

    current_conditions = {
        "rsi_below":      {"enabled": rsi_below_on,  "value": rsi_below_val},
        "rsi_above":      {"enabled": rsi_above_on,  "value": rsi_above_val},
        "near_200ma":     {"enabled": near_200_on,   "value": near_200_val},
        "near_50ma":      {"enabled": near_50_on,    "value": near_50_val},
        "down_from_52w":  {"enabled": down52_on,     "value": down52_val},
        "vol_multiplier": {"enabled": vol_on,        "value": vol_val},
        "bb_lower_touch": {"enabled": bb_touch_on},
        "macd_bullish":   {"enabled": macd_bull_on},
    }

    if run_scan:
        st.session_state["scan_ran"]        = True
        st.session_state["scan_conditions"] = current_conditions
        st.session_state["scan_ticker"]     = ticker

    # Compute indicators on full 5-year history
    rsi_full               = calculate_rsi(close, 14)
    macd_full, sig_full, _ = calculate_macd(close, 12, 26, 9)
    _, _, bb_lower_full    = calculate_bollinger_bands(close, 20, 2.0)

    signal_mask  = evaluate_leap_conditions(
        df, current_conditions,
        rsi_full, macd_full, sig_full, bb_lower_full,
    )
    signal_dates = df.index[signal_mask]

    st.markdown(
        f"<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;"
        f"padding:10px 14px;margin:10px 0;font-size:0.88rem;color:{TEXT_DARK};'>"
        f"This signal has fired <strong>{len(signal_dates)}</strong> times on "
        f"<strong>{ticker}</strong> over the last 5 years.</div>",
        unsafe_allow_html=True,
    )

    if len(signal_dates) == 0:
        st.info("No signal hits found. Try relaxing the conditions.")
        st.stop()

    # ── CHART WITH SIGNAL MARKERS ─────────────────────────────
    # Rendered directly below the scan button — first thing you see after running.
    fig_scan = go.Figure()

    # Price line (full 5-year history)
    fig_scan.add_trace(go.Scatter(
        x=close.index,
        y=close.values,
        mode="lines",
        name=ticker,
        line=dict(color=ACCENT, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:,.2f}<extra></extra>",
    ))

    # Signal markers — green upward triangles at each hit date
    sig_prices = [float(close.loc[d]) for d in signal_dates if d in close.index]
    sig_labels = [d.strftime("%Y-%m-%d") for d in signal_dates if d in close.index]
    if sig_prices:
        fig_scan.add_trace(go.Scatter(
            x=[d for d in signal_dates if d in close.index],
            y=sig_prices,
            mode="markers",
            name="Signal Hit",
            marker=dict(
                symbol="triangle-up",
                color=GREEN,
                size=12,
                line=dict(color="#1a5c38", width=1),
            ),
            hovertemplate="<b>Signal: %{x|%Y-%m-%d}</b><br>Entry: $%{y:,.2f}<extra>Signal</extra>",
        ))

    fig_scan.update_layout(
        paper_bgcolor="#f0f4f8",
        plot_bgcolor=CARD_BG,
        height=320,
        margin=dict(l=60, r=20, t=30, b=40),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(color="#334155"),
        ),
        xaxis=dict(gridcolor="#e8edf5", tickfont=dict(color="#64748b", size=10)),
        yaxis=dict(
            gridcolor="#e8edf5",
            tickfont=dict(color="#64748b", size=10),
            tickprefix="$",
            title="Price (USD)",
        ),
        font=dict(color="#334155"),
    )

    st.markdown(
        f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
        f"overflow:hidden;margin-bottom:20px;'>"
        f"{card_header(f'📈  {ticker} — 5Y Price with Signal Hits ({len(signal_dates)} signals)')}"
        f"<div style='padding:4px 0 0 0;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_scan, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── OUTCOME CALCULATION — NA FIX ─────────────────────────────────────────
    # For each signal hit date:
    #   1. Find the date's position in the price series using get_loc (integer index)
    #   2. Look forward exactly 5, 20, 60, 120 trading days using iloc
    #   3. If future_pos >= len(close): not enough future data → return None (shows as —)
    #   4. If the price at that position is NaN (rare missing data): return None
    #   5. Calculate return as (future_price - signal_price) / signal_price * 100
    # This eliminates "NA"/"nan" in the table — only None reaches ret_cell, which shows —

    results = []
    for sig_date in signal_dates:
        pos          = df.index.get_loc(sig_date)
        signal_price = float(close.iloc[pos])

        def fwd_ret(n: int):
            """Return % gain/loss n trading days after this signal, or None if unavailable."""
            future_pos = pos + n
            if future_pos >= len(close):
                return None  # Signal too recent — not enough future trading days
            future_price = float(close.iloc[future_pos])
            if np.isnan(future_price):
                return None  # Missing price data at that date
            return (future_price - signal_price) / signal_price * 100

        results.append({
            "Signal Date": sig_date.strftime("%Y-%m-%d"),
            "Entry Price": signal_price,
            "R5D":         fwd_ret(5),
            "R20D":        fwd_ret(20),
            "R60D":        fwd_ret(60),
            "R120D":       fwd_ret(120),
        })

    # ── CONFIDENCE RATING CARD — HERO OUTPUT ─────────────────
    # This is the north star of the entire app. Made large and visually prominent.
    # Computes win rate and avg return at the 120-day horizon.

    returns_120 = [r["R120D"] for r in results if r["R120D"] is not None]
    returns_60  = [r["R60D"]  for r in results if r["R60D"]  is not None]
    returns_20  = [r["R20D"]  for r in results if r["R20D"]  is not None]
    n_hits      = len(signal_dates)

    avg_ret_120 = np.mean(returns_120) if returns_120 else None
    win_rate    = (sum(1 for r in returns_120 if r > 0) / len(returns_120) * 100) if returns_120 else None
    best        = max(returns_120) if returns_120 else None
    worst       = min(returns_120) if returns_120 else None

    # Confidence label based on win rate and avg return
    if avg_ret_120 is not None and win_rate is not None:
        if win_rate >= 70 and avg_ret_120 >= 10:
            conf_grade       = "STRONG"
            conf_grade_color = GREEN
            conf_bg          = "rgba(5,150,105,0.08)"
        elif win_rate >= 50 and avg_ret_120 >= 0:
            conf_grade       = "MODERATE"
            conf_grade_color = "#d97706"
            conf_bg          = "rgba(217,119,6,0.08)"
        else:
            conf_grade       = "WEAK"
            conf_grade_color = RED
            conf_bg          = "rgba(220,38,38,0.08)"

        avg_20_str  = f"{np.mean(returns_20):+.1f}%"  if returns_20  else "—"
        avg_60_str  = f"{np.mean(returns_60):+.1f}%"  if returns_60  else "—"
        avg_120_str = f"{avg_ret_120:+.1f}%"
        wr_str      = f"{win_rate:.1f}%"
        best_str    = f"+{best:.1f}%"
        worst_str   = f"{worst:.1f}%"

        st.markdown(
            f"""<div style='background:#fff;border:2px solid #2596be;border-radius:12px;
overflow:hidden;margin:8px 0 20px 0;
box-shadow:0 6px 24px rgba(37,150,190,0.18);'>

  <!-- Header bar -->
  <div style='background:#2596be;padding:18px 24px;display:flex;
  justify-content:space-between;align-items:center;'>
    <span style='color:#fff;font-size:1.05rem;font-weight:700;
    letter-spacing:0.06em;text-transform:uppercase;'>
      🎯 Confidence Rating — {ticker}
    </span>
    <span style='background:rgba(255,255,255,0.2);color:#fff;
    font-size:0.88rem;font-weight:700;padding:4px 14px;border-radius:20px;
    letter-spacing:0.08em;'>{conf_grade}</span>
  </div>

  <!-- Metric grid -->
  <div style='display:grid;grid-template-columns:repeat(6,1fr);
  gap:0;border-bottom:1px solid {BORDER};'>

    <div style='padding:18px 16px;text-align:center;border-right:1px solid {BORDER};'>
      <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:6px;'>Signal Hits</div>
      <div style='font-size:1.8rem;font-weight:700;color:#0f2044;'>{n_hits}</div>
      <div style='font-size:0.68rem;color:#94a3b8;'>over 5 years</div>
    </div>

    <div style='padding:18px 16px;text-align:center;border-right:1px solid {BORDER};
    background:{conf_bg};'>
      <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:6px;'>Win Rate 120D</div>
      <div style='font-size:1.8rem;font-weight:700;color:{conf_grade_color};'>{wr_str}</div>
      <div style='font-size:0.68rem;color:#94a3b8;'>pct wins</div>
    </div>

    <div style='padding:18px 16px;text-align:center;border-right:1px solid {BORDER};
    background:{conf_bg};'>
      <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:6px;'>Avg Return 120D</div>
      <div style='font-size:1.8rem;font-weight:700;
      color:{"#059669" if avg_ret_120 >= 0 else "#dc2626"};'>{avg_120_str}</div>
      <div style='font-size:0.68rem;color:#94a3b8;'>avg outcome</div>
    </div>

    <div style='padding:18px 16px;text-align:center;border-right:1px solid {BORDER};'>
      <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:6px;'>Avg Return 20D</div>
      <div style='font-size:1.8rem;font-weight:700;
      color:{"#059669" if returns_20 and np.mean(returns_20) >= 0 else "#dc2626"};'>{avg_20_str}</div>
      <div style='font-size:0.68rem;color:#94a3b8;'>short term</div>
    </div>

    <div style='padding:18px 16px;text-align:center;border-right:1px solid {BORDER};'>
      <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:6px;'>Best 120D</div>
      <div style='font-size:1.8rem;font-weight:700;color:{GREEN};'>{best_str}</div>
      <div style='font-size:0.68rem;color:#94a3b8;'>top outcome</div>
    </div>

    <div style='padding:18px 16px;text-align:center;'>
      <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
      letter-spacing:0.08em;margin-bottom:6px;'>Worst 120D</div>
      <div style='font-size:1.8rem;font-weight:700;color:{RED};'>{worst_str}</div>
      <div style='font-size:0.68rem;color:#94a3b8;'>floor outcome</div>
    </div>
  </div>

  <!-- Summary text -->
  <div style='padding:16px 24px;font-size:0.88rem;color:{TEXT_DARK};line-height:1.65;'>
    This setup has fired <strong>{n_hits}</strong> times on <strong>{ticker}</strong> over the
    last 5 years. Based on {len(returns_120)} completed 120-day windows, entering at signal and
    holding 120 trading days returned an average of <strong>{avg_120_str}</strong> with a
    <strong>{wr_str}</strong> win rate. Best outcome: <strong>{best_str}</strong>.
    Worst outcome: <strong>{worst_str}</strong>.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    else:
        # Not enough future data to compute 120-day outcomes
        st.markdown(
            f"<div style='background:#fff;border:2px solid {BORDER};border-radius:12px;"
            f"overflow:hidden;margin:8px 0 20px 0;'>"
            f"<div style='background:#2596be;padding:18px 24px;'>"
            f"<span style='color:#fff;font-size:1.05rem;font-weight:700;'>"
            f"🎯 Confidence Rating — {ticker}</span></div>"
            f"<div style='padding:20px;font-size:0.88rem;color:{TEXT_MID};'>"
            f"<strong>{n_hits}</strong> signal hit(s) found. Not enough future trading data "
            f"yet to calculate 120-day outcomes. Check back after the most recent signal has "
            f"had 120 trading days of price history.</div></div>",
            unsafe_allow_html=True,
        )

    # ── OUTCOME SUMMARY TABLE ─────────────────────────────────

    def ret_cell(val) -> str:
        """Render one return cell. None or NaN → dash. Positive → green. Negative → red."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return (
                f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;"
                f"color:{TEXT_LIGHT};'>—</td>"
            )
        bg  = "rgba(5,150,105,0.08)"  if val >= 0 else "rgba(220,38,38,0.08)"
        col = GREEN if val >= 0 else RED
        return (
            f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;"
            f"background:{bg};color:{col};font-weight:600;'>{val:+.1f}%</td>"
        )

    header_style = (
        f"padding:8px 12px;font-size:0.72rem;font-weight:700;"
        f"color:#fff;text-align:right;background:{TEXT_DARK};"
    )
    th = (
        f"<tr>"
        f"<th style='{header_style}text-align:left;'>Date</th>"
        f"<th style='{header_style}'>Entry $</th>"
        f"<th style='{header_style}'>+5D%</th>"
        f"<th style='{header_style}'>+20D%</th>"
        f"<th style='{header_style}'>+60D%</th>"
        f"<th style='{header_style}'>+120D%</th>"
        f"</tr>"
    )

    td_base = f"padding:8px 12px;font-size:0.8rem;color:{TEXT_DARK};border-bottom:1px solid #e8edf5;"
    rows_html = ""
    for i, row in enumerate(results):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        rows_html += (
            f"<tr style='background:{bg};'>"
            f"<td style='{td_base}'>{row['Signal Date']}</td>"
            f"<td style='{td_base}text-align:right;font-weight:600;'>${row['Entry Price']:,.2f}</td>"
            + ret_cell(row["R5D"])
            + ret_cell(row["R20D"])
            + ret_cell(row["R60D"])
            + ret_cell(row["R120D"])
            + "</tr>"
        )

    # Summary row — average and win rate per column
    def avg_ret_cell(col_name: str) -> str:
        vals = [r[col_name] for r in results if r[col_name] is not None]
        if not vals:
            return (
                f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;"
                f"color:{TEXT_LIGHT};background:#f0f4f8;'>—</td>"
            )
        avg = np.mean(vals)
        wr  = sum(1 for r in vals if r > 0) / len(vals) * 100
        bg  = "rgba(5,150,105,0.10)" if avg >= 0 else "rgba(220,38,38,0.10)"
        col = GREEN if avg >= 0 else RED
        return (
            f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;"
            f"background:{bg};color:{col};font-weight:700;'>"
            f"{avg:+.1f}%<br><span style='font-weight:400;font-size:0.72rem;"
            f"color:{TEXT_LIGHT};'>{wr:.0f}% win</span></td>"
        )

    sum_row = (
        f"<tr style='background:#f0f4f8;'>"
        f"<td style='padding:8px 12px;font-size:0.8rem;font-weight:700;color:{TEXT_DARK};'>Average</td>"
        f"<td style='padding:8px 12px;'></td>"
        + avg_ret_cell("R5D") + avg_ret_cell("R20D")
        + avg_ret_cell("R60D") + avg_ret_cell("R120D")
        + "</tr>"
    )

    st.markdown(
        f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
        f"overflow:hidden;margin-bottom:16px;'>"
        f"{card_header(f'{ticker} — Historical Signal Outcomes')}"
        f"<div style='overflow-x:auto;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead>{th}</thead><tbody>{rows_html}{sum_row}</tbody>"
        f"</table></div></div>",
        unsafe_allow_html=True,
    )

    # ── LEAP OPTIMIZER ────────────────────────────────────────

    st.markdown(
        f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
        f"overflow:hidden;margin:8px 0;'>"
        f"{card_header('🦘  LEAP Optimizer')}"
        f"<div style='padding:16px;'>",
        unsafe_allow_html=True,
    )

    try:
        tkr_obj     = yf.Ticker(ticker)
        expirations = tkr_obj.options

        if expirations:
            target_exp = date.today() + timedelta(days=365)
            best_exp   = min(
                expirations,
                key=lambda d_str: abs((datetime.strptime(d_str, "%Y-%m-%d").date() - target_exp).days),
            )

            chain = tkr_obj.option_chain(best_exp)
            calls = chain.calls

            if not calls.empty:
                current_px = float(close.iloc[-1])

                hv = calculate_historical_volatility(close, 30)

                calls_sorted = calls.copy()
                calls_sorted["dist_atm"] = abs(calls_sorted["strike"] - current_px)
                atm_call = calls_sorted.nsmallest(1, "dist_atm").iloc[0]
                atm_iv   = float(atm_call["impliedVolatility"])

                if atm_iv > hv * 1.3:
                    iv_label = "🔴 HIGH — options are expensive; buying calls costs more premium"
                    iv_color = RED
                elif atm_iv < hv * 0.9:
                    iv_label = "🟢 LOW — favorable environment for buying calls; premium is cheap"
                    iv_color = GREEN
                else:
                    iv_label = "🟡 MODERATE — options are fairly priced"
                    iv_color = "#d97706"

                itm_target = current_px * 0.92
                calls_sorted["dist_itm"] = abs(calls_sorted["strike"] - itm_target)
                rec_call   = calls_sorted.nsmallest(1, "dist_itm").iloc[0]

                rec_strike = float(rec_call["strike"])
                rec_ask    = float(rec_call["ask"]) if float(rec_call["ask"]) > 0 else float(rec_call["lastPrice"])
                rec_label  = f"{ticker} {best_exp} ${rec_strike:.0f} Call"

                break_even      = rec_strike + rec_ask
                be_pct_above    = (break_even / current_px - 1) * 100
                approx_delta    = 0.80
                stock_move_needed = rec_ask / approx_delta
                double_target_px  = current_px + stock_move_needed
                double_pct        = (double_target_px / current_px - 1) * 100

                fired_conditions = []
                if rsi_below_on:
                    fired_conditions.append(f"RSI below {rsi_below_val}")
                if down52_on:
                    fired_conditions.append(f"price down {down52_val:.0f}%+ from 52W high")
                if near_200_on:
                    fired_conditions.append(f"price within {near_200_val:.1f}% of 200-day MA")
                if vol_on:
                    fired_conditions.append(f"volume {vol_val}× above average")
                conditions_text = " and ".join(fired_conditions) if fired_conditions else "selected conditions"

                opt_col1, opt_col2 = st.columns(2)

                with opt_col1:
                    st.markdown(f"**IV Environment**")
                    st.markdown(
                        f"<span style='color:{iv_color};font-weight:600;'>{iv_label}</span><br>"
                        f"<span style='font-size:0.78rem;color:{TEXT_LIGHT};'>"
                        f"ATM IV: {atm_iv*100:.1f}%  ·  30D HV: {hv*100:.1f}%</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                    st.markdown(f"**Recommended Contract**")
                    st.markdown(
                        f"<code style='font-size:0.9rem;color:{ACCENT};font-weight:700;'>{rec_label}</code>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span style='font-size:0.78rem;color:{TEXT_LIGHT};'>"
                        f"Ask: ${rec_ask:.2f}  ·  Current stock: ${current_px:,.2f}</span>",
                        unsafe_allow_html=True,
                    )

                with opt_col2:
                    st.metric("Break Even",          f"${break_even:,.2f}",          f"+{be_pct_above:.1f}% above current price")
                    st.metric("Double Target",        f"${double_target_px:,.2f}",    f"+{double_pct:.1f}% stock move needed")
                    st.metric("Max Loss / contract",  f"${rec_ask * 100:,.0f}",       "premium paid (100 shares)")

                if avg_ret_120 is not None and win_rate is not None:
                    st.markdown(
                        f"<div style='background:#f8fafc;border-radius:6px;padding:12px 14px;"
                        f"margin-top:12px;font-size:0.84rem;color:{TEXT_MID};line-height:1.6;'>"
                        f"<strong>Why this setup:</strong> {ticker} shows {conditions_text}. "
                        f"Historical data shows this configuration has produced an average 120-day return "
                        f"of {avg_ret_120:+.1f}% with a {win_rate:.0f}% win rate. "
                        f"The LEAP contract allows participation in a recovery while capping "
                        f"maximum risk to the ${rec_ask * 100:,.0f} premium paid per contract.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No call options data available for this ticker.")
        else:
            st.info("This ticker has no listed options.")

    except Exception as ex:
        st.warning(f"Options data unavailable: {ex}")

    st.markdown("</div></div>", unsafe_allow_html=True)
