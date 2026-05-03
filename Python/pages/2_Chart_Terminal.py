# pages/2_Chart_Terminal.py
# ============================================================
# Chart Terminal — full interactive price chart for any ticker
# in the universe, plus a Signals tab with condition builder,
# historical signal replay, outcome table, confidence rating,
# and LEAP options optimizer.
#
# Everything that was in stock_dashboard.py lives here unchanged,
# with the signals tab added directly below the chart.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta, datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file.
# python-dotenv reads the .env file in the project root and makes
# variables available via os.environ. This way API keys never appear in code.
load_dotenv()

# ── Shared helpers ───────────────────────────────────────────
from utils import inject_css, universe_manager_sidebar, card_header, section_label
from utils import ACCENT, BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG
from indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_vwap, evaluate_leap_conditions, calculate_historical_volatility,
)
from database import initialize_database, get_universe, get_presets, save_preset

# ── Bootstrap ────────────────────────────────────────────────
st.set_page_config(page_title="Chart Terminal", page_icon="📈", layout="wide")
initialize_database()
inject_css()

# Note: anthropic import removed — the news section that used it was removed
# from this page. The Signals tab is for signal analysis only; news lives on
# the Command Center page where it belongs alongside the universe feed.

# ============================================================
# SIDEBAR — controls
# ============================================================

st.sidebar.markdown("## ⚙ CONTROLS")

UNIVERSE = get_universe()
ticker = st.sidebar.selectbox("Ticker", UNIVERSE)

st.sidebar.markdown("---")
normalize = st.sidebar.toggle("Normalize to 100", value=False)
st.sidebar.markdown("---")

# ── Moving averages ──────────────────────────────────────────
st.sidebar.markdown("**Moving Averages**")
MA_DEFAULTS = [
    {"period": 20,  "color": "#f97316", "label": "MA 20"},
    {"period": 50,  "color": "#2563eb", "label": "MA 50"},
    {"period": 200, "color": "#7c3aed", "label": "MA 200"},
]
ma_configs = []
for i, d in enumerate(MA_DEFAULTS):
    checked = st.sidebar.checkbox(d["label"], value=(i == 0), key=f"ma_show{i}")
    if checked:
        c1, c2 = st.sidebar.columns([3, 1])
        period = c1.number_input("Period", min_value=2, max_value=500, value=d["period"], step=1, key=f"ma_p{i}")
        color  = c2.color_picker("Color", value=d["color"], key=f"ma_c{i}", label_visibility="collapsed")
        st.sidebar.markdown(
            f"<div style='height:3px;background:{color};border-radius:2px;"
            f"margin:-6px 0 8px 0;opacity:0.9;'></div>",
            unsafe_allow_html=True)
        ma_configs.append((int(period), color))

st.sidebar.markdown("---")
st.sidebar.markdown("**Bollinger Bands**")
show_bb = st.sidebar.toggle("Show Bollinger Bands", value=False)
if show_bb:
    bb_period = st.sidebar.number_input("BB Period",          min_value=2,   max_value=200, value=20,  step=1)
    bb_std    = st.sidebar.number_input("Std Dev Multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
else:
    bb_period, bb_std = 20, 2.0

st.sidebar.markdown("---")
st.sidebar.markdown("**VWAP**")
show_vwap = st.sidebar.toggle("Show VWAP", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Indicators**")

show_rsi = st.sidebar.toggle("Show RSI", value=True)
if show_rsi:
    rsi_period = st.sidebar.number_input("RSI Period", min_value=2, max_value=100, value=14, step=1)
    st.sidebar.markdown(
        "<div style='display:flex;gap:4px;margin:4px 0 6px 0;'>"
        "<span style='flex:1;text-align:center;background:rgba(220,38,38,0.35);border-radius:4px;"
        "font-size:0.65rem;font-weight:700;padding:3px 0;'>OB 80</span>"
        "<span style='flex:1;text-align:center;background:rgba(148,163,184,0.25);border-radius:4px;"
        "font-size:0.65rem;font-weight:700;padding:3px 0;'>MID 50</span>"
        "<span style='flex:1;text-align:center;background:rgba(5,150,105,0.35);border-radius:4px;"
        "font-size:0.65rem;font-weight:700;padding:3px 0;'>OS 30</span>"
        "</div>", unsafe_allow_html=True)
else:
    rsi_period = 14

show_macd = st.sidebar.toggle("Show MACD", value=False)
if show_macd:
    c1, c2, c3 = st.sidebar.columns(3)
    macd_fast = c1.number_input("Fast", min_value=2, max_value=50,  value=12, step=1)
    macd_slow = c2.number_input("Slow", min_value=2, max_value=200, value=26, step=1)
    macd_sig  = c3.number_input("Sig",  min_value=2, max_value=50,  value=9,  step=1)
else:
    macd_fast, macd_slow, macd_sig = 12, 26, 9

st.sidebar.markdown("---")
st.sidebar.markdown("**Signal Markers**")
show_signals = st.sidebar.toggle("RSI Crossover Signals", value=True)

st.sidebar.markdown("---")
universe_manager_sidebar()

# ============================================================
# DATA — always fetch 5 years; range selector slices the view
# ============================================================

@st.cache_data(ttl=900)
def load_data(tkr: str) -> pd.DataFrame:
    """
    Download 5 years of daily OHLCV data for a ticker from Yahoo Finance.

    We always fetch the full 5-year history so the indicator calculations
    (RSI, Bollinger Bands, 200-day MA) have enough data to be accurate.
    The range selector buttons then slice what gets *shown* on the chart
    without ever re-fetching — fast and accurate.

    Args:
        tkr (str): Ticker symbol e.g. "AAPL".

    Returns:
        pd.DataFrame: DataFrame with columns Open, High, Low, Close, Volume
                      indexed by date (timezone-naive datetime).
    """
    today = date.today()
    start = today - timedelta(days=365 * 5)
    df = yf.download(tkr, start=str(start), end=str(today), auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    # tz_localize(None) strips the timezone from the DatetimeIndex so that
    # simple date subtraction works without timezone mismatch errors.
    df.index = df.index.tz_localize(None)
    return df


df = load_data(ticker)

if df.empty:
    st.error(f"No data returned for {ticker}.")
    st.stop()

close  = df["Close"].squeeze()
opens  = df["Open"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
volume = df["Volume"].squeeze()

# ============================================================
# INDICATORS — computed on full 5-year history for accuracy
# ============================================================

rsi_series                          = calculate_rsi(close, int(rsi_period))
macd_line, macd_sig_line, macd_hist = calculate_macd(close, int(macd_fast), int(macd_slow), int(macd_sig))
bb_upper, bb_mid, bb_lower          = calculate_bollinger_bands(close, int(bb_period), float(bb_std))
vwap_series                         = calculate_vwap(high, low, close, volume)

# RSI crossover dates — used for signal marker triangles on the chart.
# A crossover UP is when RSI was below 30 yesterday and is at/above 30 today.
# A crossover DOWN is when RSI was above 80 yesterday and is at/below 80 today.
rsi_cross_up   = (rsi_series.shift(1) < 30) & (rsi_series >= 30)
rsi_cross_down = (rsi_series.shift(1) > 80) & (rsi_series <= 80)

# ============================================================
# NORMALIZATION
# ============================================================

# When normalize=True every price series is divided by the first close
# in the dataset and multiplied by 100. This lets you compare two
# very different-priced stocks (e.g. NVDA at $900 vs CHRW at $60) on
# the same chart scale — both start at 100 on day one.
base = float(close.iloc[0])


def maybe_norm(s: pd.Series, do_it: bool) -> pd.Series:
    """
    Optionally normalise a price series so it starts at 100.

    Args:
        s (pd.Series): Raw price series.
        do_it (bool): If True, normalise. If False, return unchanged.

    Returns:
        pd.Series: Normalised or unchanged series.
    """
    return (s / base) * 100 if do_it else s


plot_close    = maybe_norm(close,       normalize)
plot_high     = maybe_norm(high,        normalize)
plot_low      = maybe_norm(low,         normalize)
plot_bb_upper = maybe_norm(bb_upper,    normalize)
plot_bb_mid   = maybe_norm(bb_mid,      normalize)
plot_bb_lower = maybe_norm(bb_lower,    normalize)
plot_vwap     = maybe_norm(vwap_series, normalize)

y_label  = "Normalized (base=100)" if normalize else "Price (USD)"
y_prefix = "" if normalize else "$"

# ============================================================
# METRIC CARDS (top of page)
# ============================================================

today_dt = date.today()
latest   = float(close.iloc[-1])
prev     = float(close.iloc[-2]) if len(close) > 1 else latest
chg      = latest - prev
chg_pct  = (chg / prev) * 100
high52   = float(close.rolling(252).max().iloc[-1])
low52    = float(close.rolling(252).min().iloc[-1])
avgvol   = float(volume.rolling(20).mean().iloc[-1])
arrow    = "▲" if chg >= 0 else "▼"

st.markdown(
    f"## {ticker} &nbsp;<span style='font-size:1rem;color:#64748b'>Chart Terminal</span>",
    unsafe_allow_html=True,
)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Price",     f"${latest:,.2f}")
m2.metric("Day Change",     f"{arrow} ${abs(chg):.2f}", f"{chg_pct:+.2f}%")
m3.metric("52W High",       f"${high52:,.2f}")
m4.metric("52W Low",        f"${low52:,.2f}")
m5.metric("Avg Volume 20D", f"{avgvol/1e6:.1f}M")
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ============================================================
# RETURNS TABLE
# ============================================================

def calc_return(series: pd.Series, n_days: int):
    """
    Calculate the percentage return over the last n trading days.

    Args:
        series (pd.Series): Price series.
        n_days (int): Number of trading days to look back.

    Returns:
        float or None: Return as a percentage, or None if not enough data.
    """
    if len(series) <= n_days:
        return None
    return (float(series.iloc[-1]) / float(series.iloc[-1 - n_days]) - 1) * 100


ytd_slice = close[close.index.year == today_dt.year]
ytd_ret   = (float(close.iloc[-1]) / float(ytd_slice.iloc[0]) - 1) * 100 if len(ytd_slice) > 1 else None

return_periods = [
    ("1W", calc_return(close, 5)),   ("1M",  calc_return(close, 21)),
    ("3M", calc_return(close, 63)),  ("6M",  calc_return(close, 126)),
    ("YTD", ytd_ret),                ("1Y",  calc_return(close, 252)),
    ("3Y", calc_return(close, 756)), ("5Y",  calc_return(close, 1260)),
]


def fmt_return(val):
    """Format a return value as a coloured string tuple (text, colour)."""
    if val is None:
        return "—", TEXT_LIGHT
    return (f"+{val:.2f}%", GREEN) if val >= 0 else (f"{val:.2f}%", RED)


cells = "".join(
    f"<td style='text-align:center;padding:10px 0;border-right:1px solid #e8edf5;'>"
    f"<div style='font-size:0.68rem;color:#64748b;font-weight:600;letter-spacing:0.08em;"
    f"text-transform:uppercase;margin-bottom:5px;'>{lbl}</div>"
    f"<div style='font-size:1rem;font-weight:700;color:{fmt_return(v)[1]};'>{fmt_return(v)[0]}</div></td>"
    for lbl, v in return_periods
)

st.markdown(
    f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
    f"overflow:hidden;box-shadow:0 1px 4px rgba(15,32,68,0.08);margin-bottom:16px;'>"
    f"<table style='width:100%;border-collapse:collapse;'><tr>{cells}</tr></table></div>",
    unsafe_allow_html=True,
)

# ============================================================
# RANGE SELECTOR
# ============================================================
# The selected pill slices the DataFrame to only the rows in
# that window. We pass this sliced data directly to Plotly
# instead of manipulating axis ranges — Plotly draws exactly
# what it receives, which is simpler and always reliable.

selected_range = st.radio(
    "Range",
    options=["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"],
    index=6,
    horizontal=True,
    label_visibility="collapsed",
)

last_date = df.index.max()
range_map = {
    "1D":  last_date - timedelta(days=1),
    "5D":  last_date - timedelta(days=7),
    "1M":  last_date - timedelta(days=31),
    "3M":  last_date - timedelta(days=92),
    "6M":  last_date - timedelta(days=183),
    "YTD": pd.Timestamp(last_date.year, 1, 1),
    "1Y":  last_date - timedelta(days=365),
    "5Y":  df.index.min(),
}
cutoff = range_map.get(selected_range or "1Y", df.index.min())

# idx is the subset of dates that falls within the selected range.
# Every trace in the chart uses v(series) which returns series.loc[idx].
idx = df.index[df.index >= cutoff]


def v(series: pd.Series) -> pd.Series:
    """Slice any full-history series to the currently selected date range."""
    return series.loc[idx]


# Signal markers: only show on dates that are within the selected range
up_idx   = idx[rsi_cross_up.loc[idx]]   if len(idx) else idx
down_idx = idx[rsi_cross_down.loc[idx]] if len(idx) else idx

# Volume bar colours: green when close >= open (up day), red otherwise
vol_colors = [
    "rgba(5,150,105,0.5)" if float(c) >= float(o) else "rgba(220,38,38,0.5)"
    for c, o in zip(v(close), v(opens))
]

# Signal marker vertical offset — small nudge below/above the candle tip
slice_range   = float(v(plot_high).max() - v(plot_low).min()) if len(idx) else 1
marker_offset = slice_range * 0.015

# ============================================================
# SUBPLOT LAYOUT
# ============================================================

row_map      = {"price": 1}
row_heights  = [0.65]
panel_titles = [ticker]
next_row     = 2

if show_macd:
    row_map["macd"] = next_row
    row_heights.append(0.20)
    panel_titles.append(f"MACD  {int(macd_fast)}/{int(macd_slow)}/{int(macd_sig)}")
    next_row += 1

if show_rsi:
    row_map["rsi"] = next_row
    row_heights.append(0.20)
    panel_titles.append(f"RSI  {int(rsi_period)}")
    next_row += 1

n_rows = next_row - 1
# specs tells make_subplots that row 1 uses a secondary y-axis (for volume),
# while all other rows use the default single axis.
specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_rows - 1)

# ── Build the chart tabs ─────────────────────────────────────
chart_tab, signals_tab = st.tabs(["📈  Chart", "🔬  Signals"])

with chart_tab:
    # ── TradingView Lightweight Charts ───────────────────────────────────────
    # TradingView Lightweight Charts is an open-source (Apache 2.0) canvas-based
    # charting library built specifically for financial data. It gives us native
    # candlesticks, multi-pane layouts, crosshair sync, and smooth zoom/pan — none
    # of which Plotly handles cleanly for OHLCV data.
    #
    # We load it from unpkg CDN so there is nothing to install — one <script> tag.
    #
    # st.components.v1.html embeds a complete HTML document inside a Streamlit
    # <iframe>. It is the standard escape hatch for any browser API that Streamlit's
    # own widget system cannot provide. The iframe has no live channel back to Python
    # after the initial render, so ALL data must travel through the HTML string itself.
    #
    # Data handoff pattern:
    #   Python computes every indicator series → converts each to a list of
    #   {"time": "YYYY-MM-DD", "value": float} dicts → serialises as a JS `const`
    #   declaration → injects into the HTML f-string as a literal JS variable.
    #   The JavaScript reads those variables on page load and calls the
    #   LightweightCharts API to render. Zero round-trips; everything is in the page.
    import streamlit.components.v1 as components

    # ── Pane heights (pixels) ────────────────────────────────
    _ph  = 360                              # price pane
    _vh  = 90                               # volume pane (always shown, ~20% of price)
    _rh  = 150 if show_rsi  else 0         # RSI pane
    _mh  = 150 if show_macd else 0         # MACD pane
    _total_h = _ph + _vh + _rh + _mh

    # ── Python → JS serialisation helpers ───────────────────
    def _ser(name: str, s: pd.Series) -> str:
        """Emit `const <name> = [{time, value}, ...]` for a Series sliced to idx."""
        pts = []
        for dt, val in s.loc[idx].items():
            if pd.notna(val):
                pts.append(f'{{"time":"{dt.strftime("%Y-%m-%d")}","value":{float(val):.4f}}}')
        return f"const {name} = [{','.join(pts)}];"

    def _candles() -> tuple[str, str]:
        """Raw OHLCV for candlestick mode, plus normalised close for line mode."""
        raw, norm = [], []
        for dt in idx:
            o  = float(opens.loc[dt])
            h_ = float(high.loc[dt])
            l_ = float(low.loc[dt])
            c_ = float(close.loc[dt])
            nc = float(plot_close.loc[dt])  # already normalised if normalize=True
            raw.append(
                f'{{"time":"{dt.strftime("%Y-%m-%d")}",'
                f'"open":{o:.4f},"high":{h_:.4f},"low":{l_:.4f},"close":{c_:.4f}}}'
            )
            norm.append(f'{{"time":"{dt.strftime("%Y-%m-%d")}","value":{nc:.4f}}}')
        return (
            f"const candleData     = [{','.join(raw)}];",
            f"const normalizedData = [{','.join(norm)}];",
        )

    def _volume() -> str:
        pts = []
        for dt in idx:
            c_ = float(close.loc[dt])
            o  = float(opens.loc[dt])
            col = "rgba(5,150,105,0.5)" if c_ >= o else "rgba(220,38,38,0.5)"
            pts.append(
                f'{{"time":"{dt.strftime("%Y-%m-%d")}",'
                f'"value":{float(volume.loc[dt]):.0f},"color":"{col}"}}'
            )
        return f"const volumeData = [{','.join(pts)}];"

    def _markers() -> str:
        pts = []
        if show_signals:
            for dt in up_idx:
                if dt in idx:
                    pts.append(
                        f'{{"time":"{dt.strftime("%Y-%m-%d")}",'
                        f'"position":"belowBar","color":"{GREEN}",'
                        f'"shape":"arrowUp","text":"OS"}}'
                    )
            for dt in down_idx:
                if dt in idx:
                    pts.append(
                        f'{{"time":"{dt.strftime("%Y-%m-%d")}",'
                        f'"position":"aboveBar","color":"{RED}",'
                        f'"shape":"arrowDown","text":"OB"}}'
                    )
        return f"const markerData = [{','.join(pts)}];"

    def _mas() -> str:
        """One JS const per enabled MA, using normalised values to match price scale."""
        parts = []
        for i, (ma_p, ma_c) in enumerate(ma_configs):
            ma_vals = maybe_norm(close.rolling(ma_p).mean(), normalize)
            pts = []
            for dt in idx:
                val = ma_vals.get(dt)
                if val is not None and pd.notna(val):
                    pts.append(f'{{"time":"{dt.strftime("%Y-%m-%d")}","value":{float(val):.4f}}}')
            parts += [
                f"const maData{i}   = [{','.join(pts)}];",
                f"const maColor{i}  = '{ma_c}';",
                f"const maPeriod{i} = {ma_p};",
            ]
        return "\n".join(parts)

    def _macd_hist() -> str:
        if not show_macd:
            return "const macdHist = [];"
        pts = []
        for dt, val in macd_hist.loc[idx].items():
            if pd.notna(val):
                col = "rgba(5,150,105,0.7)" if float(val) >= 0 else "rgba(220,38,38,0.7)"
                pts.append(
                    f'{{"time":"{dt.strftime("%Y-%m-%d")}",'
                    f'"value":{float(val):.4f},"color":"{col}"}}'
                )
        return f"const macdHist = [{','.join(pts)}];"

    # Build all JS data declarations
    _cdecl, _ndecl = _candles()
    _vdecl   = _volume()
    _mkdecl  = _markers()
    _rsidecl = _ser("rsiData",  rsi_series)    if show_rsi  else "const rsiData  = [];"
    _mldecl  = _ser("macdLine", macd_line)     if show_macd else "const macdLine = [];"
    _msdecl  = _ser("macdSig",  macd_sig_line) if show_macd else "const macdSig  = [];"
    _mhdecl  = _macd_hist()
    _budecl  = _ser("bbUpper",  plot_bb_upper) if show_bb   else "const bbUpper  = [];"
    _bldecl  = _ser("bbLower",  plot_bb_lower) if show_bb   else "const bbLower  = [];"
    _bmdecl  = _ser("bbMid",    plot_bb_mid)   if show_bb   else "const bbMid    = [];"
    _vwdecl  = _ser("vwapData", plot_vwap)     if show_vwap else "const vwapData = [];"
    _madecl  = _mas()
    _nma     = len(ma_configs)

    # JS array literals for the MA loop
    _ma_data_js  = "[" + ",".join(f"maData{i}"   for i in range(_nma)) + "]"
    _ma_col_js   = "[" + ",".join(f"maColor{i}"  for i in range(_nma)) + "]"
    _ma_per_js   = "[" + ",".join(f"maPeriod{i}" for i in range(_nma)) + "]"

    tv_html = f"""<!DOCTYPE html>
<html>
<head>
<!--
  TradingView Lightweight Charts — MIT/Apache 2.0, loaded from unpkg CDN.
  https://tradingview.github.io/lightweight-charts/
-->
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; font-family: system-ui, sans-serif; overflow: hidden; }}
  .pw {{ position: relative; width: 100%; }}   /* pane wrapper */
  .pl {{                                        /* pane label */
    position: absolute; top: 5px; left: 8px;
    font-size: 10px; font-weight: 600;
    color: #94a3b8; letter-spacing: 0.05em;
    text-transform: uppercase; z-index: 10;
    pointer-events: none;
  }}
  .div {{ height: 1px; background: #e8edf5; }}
</style>
</head>
<body>
<div id="root"></div>
<script>
// ═════════════════════════════════════════════════════════════════════════
// DATA — Python → JavaScript handoff.
//
// Every series was computed in Python, converted to an array of
// {{time: "YYYY-MM-DD", value: float}} objects, and injected here as a
// const declaration.  The JS below reads them on load — no round-trips.
// ═════════════════════════════════════════════════════════════════════════
{_cdecl}
{_ndecl}
{_vdecl}
{_mkdecl}
{_rsidecl}
{_mldecl}
{_msdecl}
{_mhdecl}
{_budecl}
{_bldecl}
{_bmdecl}
{_vwdecl}
{_madecl}

// Sidebar toggle flags
const NORMALIZE = {str(normalize).lower()};
const SHOW_RSI  = {str(show_rsi).lower()};
const SHOW_MACD = {str(show_macd).lower()};
const SHOW_BB   = {str(show_bb).lower()};
const SHOW_VWAP = {str(show_vwap).lower()};

// Moving average arrays (one entry per enabled MA)
const MA_DATA    = {_ma_data_js};
const MA_COLORS  = {_ma_col_js};
const MA_PERIODS = {_ma_per_js};

// Pane heights match Python calculation above
const PRICE_H = {_ph};
const VOL_H   = {_vh};
const RSI_H   = {_rh};
const MACD_H  = {_mh};

// ── Wrap everything in setTimeout ─────────────────────────────────────
// Streamlit renders components inside an <iframe> whose width starts at 0.
// The browser sets the real iframe width asynchronously after paint.
// If we read window.innerWidth synchronously the chart gets width=0 and
// renders blank even after data is loaded.  A 150ms delay lets Streamlit
// finish sizing the iframe so we get the real pixel width on first render.
setTimeout(function() {{

  const W = window.innerWidth || 800;

  function makeOpts(h, showTime) {{
    return {{
      width:     W,
      height:    h,
      layout:    {{ background: {{ color: "#ffffff" }}, textColor: "#64748b", fontSize: 11 }},
      grid:      {{ vertLines: {{ color: "#e8edf5" }}, horzLines: {{ color: "#e8edf5" }} }},
      crosshair: {{ mode: 1 }},
      timeScale: {{
        borderColor:    "#dce4ef",
        timeVisible:    true,
        secondsVisible: false,
        visible:        !!showTime,
      }},
      rightPriceScale: {{ borderColor: "#dce4ef" }},
      handleScale:  true,
      handleScroll: true,
    }};
  }}

  // ── DOM builder ──────────────────────────────────────────────────────
  function addPane(id, label, h) {{
    const root = document.getElementById("root");
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;width:100%;height:" + h + "px;";
    const el = document.createElement("div");
    el.id = id;
    el.style.cssText = "width:" + W + "px;height:" + h + "px;";
    wrap.appendChild(el);
    if (label) {{
      const lbl = document.createElement("div");
      lbl.className = "pl";
      lbl.textContent = label;
      wrap.appendChild(lbl);
    }}
    root.appendChild(wrap);
    const div = document.createElement("div");
    div.className = "div";
    root.appendChild(div);
    return el;
  }}

  // ── Create pane elements ──────────────────────────────────────────────
  const priceEl = addPane("p-price", "{ticker}", PRICE_H);
  const volEl   = addPane("p-vol",   "VOL",      VOL_H);
  const rsiEl   = SHOW_RSI  ? addPane("p-rsi",  "RSI {int(rsi_period)}", RSI_H)  : null;
  const macdEl  = SHOW_MACD ? addPane("p-macd", "MACD",                  MACD_H) : null;

  // Determine which pane is on the bottom (only it shows the time axis)
  const bottomIsRsi  = SHOW_RSI  && !SHOW_MACD;
  const bottomIsMacd = SHOW_MACD;
  const bottomIsVol  = !SHOW_RSI && !SHOW_MACD;

  const priceChart = LightweightCharts.createChart(priceEl, makeOpts(PRICE_H, false));
  const volChart   = LightweightCharts.createChart(volEl,   makeOpts(VOL_H,   bottomIsVol));
  const rsiChart   = SHOW_RSI  ? LightweightCharts.createChart(rsiEl,  makeOpts(RSI_H,  bottomIsRsi))  : null;
  const macdChart  = SHOW_MACD ? LightweightCharts.createChart(macdEl, makeOpts(MACD_H, true))         : null;

  const allCharts = [priceChart, volChart, rsiChart, macdChart].filter(Boolean);

  // ── PRICE PANE ───────────────────────────────────────────────────────
  let priceSeries;
  if (NORMALIZE) {{
    priceSeries = priceChart.addLineSeries({{ color: "#1d4ed8", lineWidth: 2 }});
    priceSeries.setData(normalizedData);
  }} else {{
    priceSeries = priceChart.addCandlestickSeries({{
      upColor: "#059669",       downColor: "#dc2626",
      borderUpColor: "#059669", borderDownColor: "#dc2626",
      wickUpColor:   "#059669", wickDownColor:   "#dc2626",
    }});
    priceSeries.setData(candleData);
  }}

  if (markerData.length) priceSeries.setMarkers(markerData);

  if (SHOW_BB) {{
    const bbU = priceChart.addLineSeries({{ color: "#9b59b6", lineWidth: 1, lineStyle: 2, title: "BB ↑" }});
    const bbL = priceChart.addLineSeries({{ color: "#9b59b6", lineWidth: 1, lineStyle: 2, title: "BB ↓" }});
    const bbM = priceChart.addLineSeries({{ color: "#7c3aed", lineWidth: 1,               title: "BB mid" }});
    bbU.setData(bbUpper); bbL.setData(bbLower); bbM.setData(bbMid);
  }}

  MA_DATA.forEach((data, i) => {{
    priceChart.addLineSeries({{ color: MA_COLORS[i], lineWidth: 1.5, title: "MA " + MA_PERIODS[i] }}).setData(data);
  }});

  if (SHOW_VWAP) {{
    priceChart.addLineSeries({{ color: "#0f2044", lineWidth: 1.5, lineStyle: 2, title: "VWAP" }}).setData(vwapData);
  }}

  // ── VOLUME PANE ──────────────────────────────────────────────────────
  volChart.applyOptions({{ rightPriceScale: {{ visible: false }}, leftPriceScale: {{ visible: false }} }});
  volChart.addHistogramSeries({{ priceFormat: {{ type: "volume" }} }}).setData(volumeData);

  // ── RSI PANE ─────────────────────────────────────────────────────────
  if (rsiChart) {{
    rsiChart.addLineSeries({{ color: "#7c3aed", lineWidth: 1.5 }}).setData(rsiData);
    [70, 30].forEach(lvl => {{
      const s = rsiChart.addLineSeries({{
        color: lvl === 70 ? "rgba(220,38,38,0.5)" : "rgba(5,150,105,0.5)",
        lineWidth: 1, lineStyle: 2,
        lastValueVisible: false, priceLineVisible: false,
      }});
      s.setData(rsiData.map(d => ({{ time: d.time, value: lvl }})));
    }});
  }}

  // ── MACD PANE ────────────────────────────────────────────────────────
  if (macdChart) {{
    macdChart.addHistogramSeries({{ priceFormat: {{ type: "price", precision: 3 }} }}).setData(macdHist);
    macdChart.addLineSeries({{ color: "#1d4ed8", lineWidth: 1.5 }}).setData(macdLine);
    macdChart.addLineSeries({{ color: "#dc2626", lineWidth: 1.5 }}).setData(macdSig);
  }}

  // ── Sync time scales across all panes ────────────────────────────────
  allCharts.forEach(src => {{
    src.timeScale().subscribeVisibleLogicalRangeChange(range => {{
      if (!range) return;
      allCharts.forEach(dst => {{ if (dst !== src) dst.timeScale().setVisibleLogicalRange(range); }});
    }});
  }});

  // ── Resize handler ────────────────────────────────────────────────────
  window.addEventListener("resize", () => {{
    const newW = window.innerWidth;
    allCharts.forEach(c => c.applyOptions({{ width: newW }}));
  }});

}}, 150);  // end setTimeout — 150ms lets Streamlit finish sizing the iframe
</script>
</body>
</html>"""

    # Quick debug: show how many candle bars Python generated for this range.
    # If this reads 0 the problem is in Python data prep; if > 0 it is JS-side.
    st.caption(f"debug — candle bars in range: {len(idx)}  |  first date: {idx[0].date() if len(idx) else 'none'}")

    _tv_ok = False
    try:
        components.html(tv_html, height=_total_h + 8, scrolling=False)
        _tv_ok = True
    except Exception as _tv_err:
        st.warning(f"TradingView render failed ({_tv_err}) — falling back to Plotly.")

    if not _tv_ok:
        # ── Plotly candlestick fallback ───────────────────────
        # Proper go.Candlestick with shared x-axis subplots for volume, RSI, MACD.
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=row_heights,
            subplot_titles=panel_titles,
            specs=specs,
        )

        # Price pane: candlestick or normalised line
        if normalize:
            fig.add_trace(go.Scatter(
                x=idx, y=v(plot_close), mode="lines", name=ticker, showlegend=False,
                line=dict(color=ACCENT, width=2),
                fill="tozeroy", fillcolor="rgba(29,78,216,0.10)",
            ), row=1, col=1, secondary_y=False)
        else:
            fig.add_trace(go.Candlestick(
                x=idx,
                open=v(opens), high=v(high), low=v(low), close=v(close),
                name=ticker, showlegend=False,
                increasing_line_color="#059669", decreasing_line_color="#dc2626",
            ), row=1, col=1, secondary_y=False)

        if show_bb:
            fig.add_trace(go.Scatter(x=idx, y=v(plot_bb_upper), mode="lines", name="BB Upper",
                line=dict(color="#9b59b6", width=1, dash="dot")), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=idx, y=v(plot_bb_lower), mode="lines", name="BB Lower",
                line=dict(color="#9b59b6", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(155,89,182,0.06)", showlegend=False),
                row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=idx, y=v(plot_bb_mid), mode="lines", name="BB Mid",
                line=dict(color="#7c3aed", width=1)), row=1, col=1, secondary_y=False)

        for (ma_period, color) in ma_configs:
            ma_vals = maybe_norm(close.rolling(window=ma_period).mean(), normalize)
            fig.add_trace(go.Scatter(x=idx, y=v(ma_vals), mode="lines", name=f"MA {ma_period}",
                line=dict(color=color, width=1.5)), row=1, col=1, secondary_y=False)

        if show_vwap:
            fig.add_trace(go.Scatter(x=idx, y=v(plot_vwap), mode="lines", name="VWAP",
                line=dict(color="#0f2044", width=1.5, dash="dash")), row=1, col=1, secondary_y=False)

        if show_signals:
            if len(up_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=up_idx, y=v(plot_low).loc[up_idx] - marker_offset,
                    mode="markers", name="RSI Oversold Cross",
                    marker=dict(symbol="triangle-up", color=GREEN, size=10,
                                line=dict(color="#334155", width=0.5)),
                ), row=1, col=1, secondary_y=False)
            if len(down_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=down_idx, y=v(plot_high).loc[down_idx] + marker_offset,
                    mode="markers", name="RSI Overbought Cross",
                    marker=dict(symbol="triangle-down", color=RED, size=10,
                                line=dict(color="#334155", width=0.5)),
                ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Bar(
            x=idx, y=v(volume),
            marker_color=vol_colors, marker_line_width=0,
            name="Volume", showlegend=False,
        ), row=1, col=1, secondary_y=True)
        fig.update_yaxes(
            range=[0, float(v(volume).max()) * 5],
            showticklabels=False, showgrid=False, showline=False,
            row=1, col=1, secondary_y=True,
        )

        if show_macd:
            r = row_map["macd"]
            hist_colors = [GREEN if val >= 0 else RED for val in v(macd_hist).fillna(0)]
            fig.add_trace(go.Bar(x=idx, y=v(macd_hist), marker_color=hist_colors,
                marker_line_width=0, name="MACD Hist", showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=idx, y=v(macd_line), mode="lines", name="MACD",
                line=dict(color=ACCENT, width=1.5)), row=r, col=1)
            fig.add_trace(go.Scatter(x=idx, y=v(macd_sig_line), mode="lines", name="Signal",
                line=dict(color=RED, width=1.5)), row=r, col=1)

        if show_rsi:
            r = row_map["rsi"]
            fig.add_trace(go.Scatter(x=idx, y=v(rsi_series), mode="lines", name="RSI",
                line=dict(color="#7c3aed", width=1.5), showlegend=False), row=r, col=1)
            for level, color, label in [(70, RED, "70"), (30, GREEN, "30")]:
                fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1),
                              annotation_text=label, annotation_font_color=color,
                              annotation_position="right", row=r, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor=RED,   opacity=0.06, line_width=0, row=r, col=1)
            fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.06, line_width=0, row=r, col=1)

        total_height = 540 + (180 if show_macd else 0) + (180 if show_rsi else 0)
        AXIS_STYLE = dict(gridcolor="#e8edf5", tickfont=dict(color="#64748b", size=11),
                          showline=True, linecolor=BORDER, zeroline=False)
        fig.update_layout(
            paper_bgcolor="#f0f4f8", plot_bgcolor=CARD_BG,
            hovermode="x unified",
            font=dict(color="#334155", family="Inter, system-ui, sans-serif"),
            dragmode="pan", height=total_height,
            margin=dict(l=60, r=40, t=50, b=80),
            legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0,
                        bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1,
                        font=dict(color="#334155")),
            barmode="overlay",
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        fig.update_yaxes(title=y_label, tickprefix=y_prefix, row=1, col=1, secondary_y=False)
        if show_macd:
            fig.update_yaxes(title="MACD", row=row_map["macd"], col=1)
        if show_rsi:
            fig.update_yaxes(title="RSI", range=[0, 100], tickvals=[0, 30, 50, 70, 100],
                             row=row_map["rsi"], col=1)
        for ann in fig.layout.annotations:
            ann.font = dict(color="#94a3b8", size=10)

        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True, "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toImage"],
            "displaylogo": False,
        })

# ============================================================
# SIGNALS TAB
# ============================================================

with signals_tab:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── News removed from this tab ────────────────────────────
    # A news feed previously lived here. It was moved to the Command Center
    # page where it belongs alongside the universe feed and morning briefing.
    # The Signals tab is for signal analysis only: condition builder, scan
    # results, outcome table, confidence rating, and LEAP optimizer.

    # ── Instructions card ─────────────────────────────────────
    # st.expander creates a collapsible section. expanded=False means it starts
    # closed — the user sees only the header label and can click to open it.
    # Instructions are collapsed by default so they don't push the actual signal
    # tools off screen every time someone opens this tab. Power users can ignore
    # it completely; new users can open it once to understand the workflow.
    with st.expander("ℹ️  How To Use The Signals Tab", expanded=False):
        st.markdown(
            f"""<div style='background:#e8f4fd;border:1px solid #1d4ed8;border-radius:8px;padding:16px;'>

<p style='font-size:0.82rem;font-weight:700;color:{TEXT_DARK};letter-spacing:0.06em;
text-transform:uppercase;margin:0 0 6px 0;'>What This Tab Does</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 4px 0;'>
The Signals tab lets you define a set of technical conditions and scan 5 years of historical
price data to find every time those conditions all fired at the same time on the selected stock.
It then shows you what happened to the stock price after each of those historical signals —
5 days later, 20 days later, 60 days later, and 120 days later. This tells you whether your
signal logic has historically been a reliable entry point before you risk any real money.
</p>

<hr style='border:none;border-top:1px solid #bfdbfe;margin:14px 0;'>

<p style='font-size:0.82rem;font-weight:700;color:{TEXT_DARK};letter-spacing:0.06em;
text-transform:uppercase;margin:0 0 6px 0;'>How To Use It</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 4px 0;'>
Start by selecting a ticker from the dropdown in the sidebar. Then either choose the LEAP Setup
preset from the preset dropdown at the top of this tab or build your own signal by checking the
condition boxes below. Once your conditions are set press the Run Signal Scan button. Green
triangle markers will appear on the chart at every historical date where all your conditions
fired simultaneously. The Outcome Table below the chart shows what happened after each of
those signals.
</p>

<hr style='border:none;border-top:1px solid #bfdbfe;margin:14px 0;'>

<p style='font-size:0.82rem;font-weight:700;color:{TEXT_DARK};letter-spacing:0.06em;
text-transform:uppercase;margin:0 0 6px 0;'>Understanding The LEAP Setup Preset</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 4px 0;'>
The LEAP Setup is the most important preset in this app and the foundation of the entire
strategy. It fires when four things are all true at the same time: the stock is down 10 percent
or more from its 52 week high, the price is within 2 percent of the 200 day moving average,
the RSI is below 35 indicating the stock is oversold, and volume is above average confirming
real buying and selling interest at this level. This combination identifies quality companies
being sold off by the broader market rather than by any company-specific problem. That is
exactly the setup where a 12 to 24 month LEAP call option has historically had the highest
probability of returning 100 percent or more.
</p>

<hr style='border:none;border-top:1px solid #bfdbfe;margin:14px 0;'>

<p style='font-size:0.82rem;font-weight:700;color:{TEXT_DARK};letter-spacing:0.06em;
text-transform:uppercase;margin:0 0 6px 0;'>Understanding The Confidence Rating</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 4px 0;'>
The confidence rating is calculated purely from historical data in this app. It tells you how
many times this exact combination of conditions has fired on this stock over the last 5 years
and what the average outcome was at each time interval. A high win rate at the 120 day mark
combined with a strong average return means this signal has historically been a reliable entry
point. A low win rate means the conditions alone are not enough and you should look for
additional confirmation before entering a position.
</p>

<hr style='border:none;border-top:1px solid #bfdbfe;margin:14px 0;'>

<p style='font-size:0.82rem;font-weight:700;color:{TEXT_DARK};letter-spacing:0.06em;
text-transform:uppercase;margin:0 0 6px 0;'>Understanding The Outcome Table</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0 0 4px 0;'>
Each row in the outcome table represents one historical date when your signal fired. The return
columns show the percentage gain or loss if you had bought the stock on the exact day the signal
fired and held for that many trading days. Green cells mean the stock was higher at that point.
Red cells mean it was lower. The summary row at the very bottom averages all outcomes across
every signal hit so you can see the overall historical edge at a glance.
</p>

<hr style='border:none;border-top:1px solid #bfdbfe;margin:14px 0;'>

<p style='font-size:0.82rem;font-weight:700;color:{TEXT_DARK};letter-spacing:0.06em;
text-transform:uppercase;margin:0 0 6px 0;'>A Note On Using This Responsibly</p>
<p style='font-size:0.84rem;color:#475569;line-height:1.7;margin:0;'>
Past performance does not guarantee future results. A signal that worked 18 out of 23 times
historically is a strong edge but not a certainty. Always consider the current market regime,
the company's fundamental health, and your own risk tolerance before entering any position.
This tool is designed to inform your decisions, not make them for you.
</p>

</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Preset selector ──────────────────────────────────────
    presets      = get_presets()
    preset_names = [p["name"] for p in presets]

    sel_preset = st.selectbox(
        "Load Preset", ["— select —"] + preset_names,
        help="Load a saved condition configuration.",
        key="preset_sel",
    )

    # Build the default conditions (LEAP Setup values shown initially)
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

    # If user selected a preset, overlay its saved values
    if sel_preset != "— select —":
        for p in presets:
            if p["name"] == sel_preset:
                default_conditions = p["conditions"]
                break

    # ── Condition builder ────────────────────────────────────
    st.markdown(
        f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
        f"overflow:hidden;margin-bottom:12px;'>"
        f"{card_header('⚡  Condition Builder  — LEAP Setup is the flagship preset')}"
        f"<div style='padding:16px;'>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        # RSI below threshold
        c = default_conditions.get("rsi_below", {})
        rsi_below_on  = st.checkbox("RSI below threshold", value=c.get("enabled", True), key="cb_rsi_below")
        rsi_below_val = st.number_input("RSI below value", 1, 99, int(c.get("value", 35)), key="cv_rsi_below") if rsi_below_on else c.get("value", 35)

        # Price within X% of 200-day MA
        c = default_conditions.get("near_200ma", {})
        near_200_on  = st.checkbox("Price within X% of 200-day MA", value=c.get("enabled", True), key="cb_200ma")
        near_200_val = st.number_input("% from 200-day MA", 0.1, 20.0, float(c.get("value", 2.0)), 0.5, key="cv_200ma") if near_200_on else c.get("value", 2.0)

        # Price down X% from 52-week high
        c = default_conditions.get("down_from_52w", {})
        down52_on  = st.checkbox("Price down X% from 52W high", value=c.get("enabled", True), key="cb_52w")
        down52_val = st.number_input("% drawdown from 52W high", 1.0, 70.0, float(c.get("value", 10.0)), 1.0, key="cv_52w") if down52_on else c.get("value", 10.0)

        # Bollinger Band lower touch
        c = default_conditions.get("bb_lower_touch", {})
        bb_touch_on = st.checkbox("Bollinger lower band touch", value=c.get("enabled", False), key="cb_bbtch")

    with col_b:
        # RSI above threshold
        c = default_conditions.get("rsi_above", {})
        rsi_above_on  = st.checkbox("RSI above threshold", value=c.get("enabled", False), key="cb_rsi_above")
        rsi_above_val = st.number_input("RSI above value", 1, 99, int(c.get("value", 70)), key="cv_rsi_above") if rsi_above_on else c.get("value", 70)

        # Price within X% of 50-day MA
        c = default_conditions.get("near_50ma", {})
        near_50_on  = st.checkbox("Price within X% of 50-day MA", value=c.get("enabled", False), key="cb_50ma")
        near_50_val = st.number_input("% from 50-day MA", 0.1, 20.0, float(c.get("value", 2.0)), 0.5, key="cv_50ma") if near_50_on else c.get("value", 2.0)

        # Volume multiplier
        c = default_conditions.get("vol_multiplier", {})
        vol_on  = st.checkbox("Volume X times above 20-day avg", value=c.get("enabled", True), key="cb_vol")
        vol_val = st.number_input("Volume multiplier", 0.5, 10.0, float(c.get("value", 1.5)), 0.25, key="cv_vol") if vol_on else c.get("value", 1.5)

        # MACD bullish crossover
        c = default_conditions.get("macd_bullish", {})
        macd_bull_on = st.checkbox("MACD bullish crossover", value=c.get("enabled", False), key="cb_macd_bull")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Save preset ──────────────────────────────────────────
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

    # ── Run scan ─────────────────────────────────────────────
    run_scan = st.button("▶  Run Signal Scan", type="primary", key="run_scan_btn")

    if run_scan or st.session_state.get("scan_ran"):
        # Build current condition dict from UI state
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
            st.session_state["scan_ran"] = True
            st.session_state["scan_conditions"] = current_conditions
            st.session_state["scan_ticker"] = ticker

        # Recalculate indicators on full history for the scan
        rsi_full             = calculate_rsi(close, 14)
        macd_full, sig_full, _ = calculate_macd(close, 12, 26, 9)
        _, _, bb_lower_full  = calculate_bollinger_bands(close, 20, 2.0)

        # evaluate_leap_conditions returns a boolean Series — True on each
        # date where ALL active conditions were simultaneously true.
        signal_mask = evaluate_leap_conditions(
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
        else:
            # ── Outcome summary table ─────────────────────────────
            # For each signal hit, look up the price 5, 20, 60, 120 trading days later.
            # We use integer iloc positions rather than date arithmetic because
            # the market is closed on weekends and holidays, so "20 trading days"
            # is not the same as "20 calendar days".
            results = []
            for sig_date in signal_dates:
                pos = df.index.get_loc(sig_date)          # integer row index of this date
                entry_price = float(close.iloc[pos])

                def fwd(n: int):
                    """Return the closing price n trading days after the signal, or None."""
                    future_pos = pos + n
                    if future_pos < len(close):
                        return float(close.iloc[future_pos])
                    return None

                def pct(future_price):
                    """Return the percentage return, or None if future price not yet available."""
                    if future_price is None:
                        return None
                    return (future_price / entry_price - 1) * 100

                p5   = fwd(5);   p20 = fwd(20)
                p60  = fwd(60);  p120 = fwd(120)

                results.append({
                    "Signal Date":     sig_date.strftime("%Y-%m-%d"),
                    "Entry Price":     entry_price,
                    "P5D":  p5,   "R5D":  pct(p5),
                    "P20D": p20,  "R20D": pct(p20),
                    "P60D": p60,  "R60D": pct(p60),
                    "P120D":p120, "R120D":pct(p120),
                })

            outcome_df = pd.DataFrame(results)

            # ── Confidence card ───────────────────────────────────
            returns_120 = [r for r in outcome_df["R120D"].dropna()]
            n_hits       = len(signal_dates)
            avg_ret      = np.mean(returns_120) if returns_120 else None
            win_rate     = (sum(1 for r in returns_120 if r > 0) / len(returns_120) * 100) if returns_120 else None
            best         = max(returns_120) if returns_120 else None
            worst        = min(returns_120) if returns_120 else None

            if avg_ret is not None:
                conf_text = (
                    f"This setup has fired <strong>{n_hits}</strong> times on "
                    f"<strong>{ticker}</strong> over 5 years. "
                    f"Based on historical outcomes, entering at this signal and holding for "
                    f"120 days returned an average of <strong>{avg_ret:+.1f}%</strong> with a "
                    f"<strong>{win_rate:.1f}%</strong> win rate. "
                    f"The best outcome was <strong>+{best:.1f}%</strong> and the worst was "
                    f"<strong>{worst:.1f}%</strong>."
                )
            else:
                conf_text = "Not enough future data yet to calculate 120-day outcomes."

            st.markdown(
                f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
                f"overflow:hidden;margin:16px 0;'>"
                f"{card_header('🎯  Signal Confidence')}"
                f"<div style='padding:16px;font-size:0.88rem;color:{TEXT_DARK};"
                f"line-height:1.65;'>{conf_text}</div></div>",
                unsafe_allow_html=True,
            )

            # ── Outcome table ─────────────────────────────────────
            def ret_cell(val, price):
                """Return an HTML <td> cell coloured green/red based on return sign."""
                if val is None:
                    return f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;color:{TEXT_LIGHT};'>—</td>"
                bg  = "rgba(5,150,105,0.08)"  if val >= 0 else "rgba(220,38,38,0.08)"
                col = GREEN if val >= 0 else RED
                return (
                    f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;"
                    f"background:{bg};color:{col};font-weight:600;'>{val:+.1f}%</td>"
                )

            header_style = f"padding:8px 12px;font-size:0.72rem;font-weight:700;color:#fff;text-align:right;background:{TEXT_DARK};"
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
                    + ret_cell(row["R5D"],  row["P5D"])
                    + ret_cell(row["R20D"], row["P20D"])
                    + ret_cell(row["R60D"], row["P60D"])
                    + ret_cell(row["R120D"],row["P120D"])
                    + "</tr>"
                )

            # Summary/averages row at bottom
            def avg_ret_cell(col_name):
                """Return summary cell for a return column: average ± win rate."""
                vals = [r[col_name] for r in results if r[col_name] is not None]
                if not vals:
                    return f"<td style='padding:8px 12px;text-align:right;font-size:0.8rem;color:{TEXT_LIGHT};background:#f0f4f8;'>—</td>"
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

            # ── LEAP Optimizer ────────────────────────────────────
            st.markdown(
                f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
                f"overflow:hidden;margin:8px 0;'>"
                f"{card_header('🦘  LEAP Optimizer')}"
                f"<div style='padding:16px;'>",
                unsafe_allow_html=True,
            )

            try:
                tkr_obj     = yf.Ticker(ticker)
                expirations = tkr_obj.options   # list of "YYYY-MM-DD" strings

                if expirations:
                    # Find the expiration closest to 12 months from today.
                    # LEAP options typically expire in January of the next calendar year.
                    target_exp = date.today() + timedelta(days=365)
                    best_exp   = min(
                        expirations,
                        key=lambda d_str: abs((datetime.strptime(d_str, "%Y-%m-%d").date() - target_exp).days),
                    )

                    chain = tkr_obj.option_chain(best_exp)
                    calls = chain.calls

                    if not calls.empty:
                        current_px = float(close.iloc[-1])

                        # ── IV Environment ────────────────────────────
                        # Historical volatility: how much the stock has actually moved
                        hv = calculate_historical_volatility(close, 30)

                        # ATM IV: implied volatility of the call option whose strike
                        # is nearest to the current stock price.
                        calls_sorted = calls.copy()
                        calls_sorted["dist_atm"] = abs(calls_sorted["strike"] - current_px)
                        atm_call = calls_sorted.nsmallest(1, "dist_atm").iloc[0]
                        atm_iv   = float(atm_call["impliedVolatility"])

                        # IV vs HV comparison:
                        # If options are pricing in more movement than stocks have
                        # historically shown (IV > HV * 1.3), options are expensive.
                        if atm_iv > hv * 1.3:
                            iv_label = "🔴 HIGH — options are expensive; buying calls costs more premium"
                            iv_color = RED
                        elif atm_iv < hv * 0.9:
                            iv_label = "🟢 LOW — favorable environment for buying calls; premium is cheap"
                            iv_color = GREEN
                        else:
                            iv_label = "🟡 MODERATE — options are fairly priced"
                            iv_color = "#d97706"

                        # ── Recommended contract ──────────────────────
                        # Delta ~0.80 means the option moves $0.80 for every $1 move in the stock.
                        # Deep ITM (in-the-money) calls with high delta behave more like stock
                        # ownership while limiting maximum loss to the premium paid.
                        # We approximate 0.80 delta by finding calls 5-10% in the money.
                        itm_target = current_px * 0.92    # 8% below current price = approx delta 0.80
                        calls_sorted["dist_itm"] = abs(calls_sorted["strike"] - itm_target)
                        rec_call   = calls_sorted.nsmallest(1, "dist_itm").iloc[0]

                        rec_strike = float(rec_call["strike"])
                        rec_ask    = float(rec_call["ask"]) if float(rec_call["ask"]) > 0 else float(rec_call["lastPrice"])
                        rec_label  = f"{ticker} {best_exp} ${rec_strike:.0f} Call"

                        # ── Break even calculation ────────────────────
                        # You need the stock to be above (strike + premium paid) at expiration
                        # just to break even. Premium paid = ask price × 100 shares per contract.
                        break_even     = rec_strike + rec_ask
                        be_pct_above   = (break_even / current_px - 1) * 100

                        # ── Double target ─────────────────────────────
                        # Very rough approximation: for a deep ITM LEAP with delta ≈ 0.80,
                        # you need the stock to move approximately (option_premium / delta)
                        # dollars for the contract to double in value.
                        approx_delta   = 0.80
                        stock_move_needed = rec_ask / approx_delta
                        double_target_px  = current_px + stock_move_needed
                        double_pct        = (double_target_px / current_px - 1) * 100

                        # ── Build conditions summary ──────────────────
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

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**IV Environment**")
                            st.markdown(f"<span style='color:{iv_color};font-weight:600;'>{iv_label}</span><br>"
                                        f"<span style='font-size:0.78rem;color:{TEXT_LIGHT};'>ATM IV: {atm_iv*100:.1f}%  ·  30D HV: {hv*100:.1f}%</span>",
                                        unsafe_allow_html=True)
                            st.markdown("")
                            st.markdown(f"**Recommended Contract**")
                            st.markdown(f"<code style='font-size:0.9rem;color:{ACCENT};font-weight:700;'>{rec_label}</code>",
                                        unsafe_allow_html=True)
                            st.markdown(f"<span style='font-size:0.78rem;color:{TEXT_LIGHT};'>Ask: ${rec_ask:.2f}  ·  Current stock: ${current_px:,.2f}</span>",
                                        unsafe_allow_html=True)

                        with col2:
                            st.metric("Break Even", f"${break_even:,.2f}", f"+{be_pct_above:.1f}% above current price")
                            st.metric("Double Target", f"${double_target_px:,.2f}", f"+{double_pct:.1f}% stock move needed")
                            st.metric("Max Loss / contract", f"${rec_ask * 100:,.0f}", "premium paid (100 shares)")

                        st.markdown(
                            f"<div style='background:#f8fafc;border-radius:6px;padding:12px 14px;"
                            f"margin-top:12px;font-size:0.84rem;color:{TEXT_MID};line-height:1.6;'>"
                            f"<strong>Why this setup:</strong> {ticker} shows {conditions_text}. "
                            f"Historical data shows this configuration has produced an average 120-day return "
                            f"of {avg_ret:+.1f}% with a {win_rate:.0f}% win rate. "
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

# ============================================================
# FUNDAMENTALS TABLE
# ============================================================

@st.cache_data(ttl=900)
def load_info(tkr: str) -> dict:
    """
    Fetch fundamental data from Yahoo Finance for a given ticker.

    yf.Ticker(tkr).info returns a dictionary with hundreds of keys
    including price ratios, market cap, dividends, earnings dates, and more.
    We cache this for 15 minutes to avoid hammering the API on every rerun.

    Args:
        tkr (str): Ticker symbol.

    Returns:
        dict: Raw Yahoo Finance info dict, or empty dict on error.
    """
    try:
        return yf.Ticker(tkr).info
    except Exception:
        return {}


info = load_info(ticker)


def fv(key):
    """Return raw value or 'N/A'."""
    return info.get(key, "N/A")


def fp(key):
    """Return a dollar-formatted price string or 'N/A'."""
    val = info.get(key)
    return f"${val:,.2f}" if val else "N/A"


def fvol(key):
    """Return a comma-separated integer string or 'N/A'."""
    val = info.get(key)
    return f"{val:,.0f}" if val else "N/A"


def fcap(key):
    """Return a human-readable market cap string (T/B/M) or 'N/A'."""
    val = info.get(key)
    if not val:
        return "N/A"
    if val >= 1e12:
        return f"{val/1e12:.3f}T"
    if val >= 1e9:
        return f"{val/1e9:.2f}B"
    return f"{val/1e6:.2f}M"


def fts(key):
    """Convert a Unix timestamp stored in Yahoo Finance info to a human date string."""
    val = info.get(key)
    if not val:
        return "N/A"
    try:
        return pd.Timestamp(val, unit="s").strftime("%b %d, %Y")
    except Exception:
        return "N/A"


dl, dh = info.get("dayLow"), info.get("dayHigh")
wl, wh = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
bid, bsz = info.get("bid"), info.get("bidSize", "")
ask, asz = info.get("ask"), info.get("askSize", "")
dr, dy   = info.get("dividendRate"), info.get("dividendYield")

fund_rows = [
    ("Previous Close", fp("previousClose"),  "Day's Range",       f"${dl:,.2f} – ${dh:,.2f}" if dl and dh else "N/A"),
    ("Open",           fp("open"),           "52 Week Range",     f"${wl:,.2f} – ${wh:,.2f}" if wl and wh else "N/A"),
    ("Bid",            f"${bid:,.2f} × {bsz}" if bid else "N/A", "Volume",      fvol("volume")),
    ("Ask",            f"${ask:,.2f} × {asz}" if ask else "N/A", "Avg. Volume", fvol("averageVolume")),
    ("Market Cap",     fcap("marketCap"),    "Beta (5Y Monthly)", str(fv("beta"))),
    ("PE Ratio (TTM)", str(fv("trailingPE")),"EPS (TTM)",         str(fv("trailingEps"))),
    ("Earnings Date",  fts("earningsTimestamp"), "Fwd Div & Yield",
     f"{dr:.2f} ({dy*100:.2f}%)" if dr and dy else "N/A"),
    ("Ex-Div Date",    fts("exDividendDate"),"1Y Target Est",     fp("targetMeanPrice")),
]


def fund_row_html(l1, v1, l2, v2, shade):
    """Build one HTML table row for the fundamentals table."""
    bg = "#f8fafc" if shade else "#ffffff"
    td = "padding:9px 16px;font-size:0.82rem;border-bottom:1px solid #e8edf5;"
    return (
        f"<tr style='background:{bg};'>"
        f"<td style='{td}color:#64748b;width:22%;'>{l1}</td>"
        f"<td style='{td}color:{TEXT_DARK};font-weight:600;width:28%;'>{v1}</td>"
        f"<td style='{td}color:#64748b;width:22%;border-left:1px solid #e8edf5;'>{l2}</td>"
        f"<td style='{td}color:{TEXT_DARK};font-weight:600;width:28%;'>{v2}</td></tr>"
    )


table_rows_html = "".join(
    fund_row_html(l1, v1, l2, v2, i % 2 == 0)
    for i, (l1, v1, l2, v2) in enumerate(fund_rows)
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
    f"overflow:hidden;box-shadow:0 2px 8px rgba(15,32,68,0.06);'>"
    f"{card_header(f'{ticker} — Key Statistics')}"
    f"<table style='width:100%;border-collapse:collapse;'>{table_rows_html}</table></div>",
    unsafe_allow_html=True,
)
