# pages/3_Chart_Terminal.py
# ============================================================
# Chart Terminal — full interactive price chart experience.
#
# This page is the charting workbench. Everything you need to
# analyse price action for any ticker in the universe is here:
#
#   Sidebar controls:
#     - Ticker dropdown
#     - Normalize to 100 toggle
#     - Moving averages (up to 3, each with custom period + color)
#     - Bollinger Bands toggle with configurable period and std dev
#     - VWAP toggle
#     - RSI toggle with configurable period
#     - MACD toggle with configurable fast/slow/signal periods
#     - RSI crossover signal markers toggle
#     - Universe Manager
#
#   Main content (top to bottom):
#     - Metric cards: last price, day change, 52W high/low, avg vol
#     - Returns table: 1W, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y
#     - Range selector: 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y
#     - TradingView Lightweight Charts (with Plotly fallback)
#     - Key Statistics fundamentals table
#
# Signal scanning lives on page 4 (Signal Scanner).
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta, datetime
import os
from dotenv import load_dotenv

load_dotenv()

from utils import inject_css, universe_manager_sidebar, card_header, section_label, load_price_history
from utils import ACCENT, BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG
from indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_vwap, calculate_historical_volatility,
)
from database import initialize_database, get_universe

st.set_page_config(page_title="Chart Terminal", page_icon="📈", layout="wide")
initialize_database()
inject_css()

# ============================================================
# SIDEBAR — controls
# ============================================================

st.sidebar.markdown("## ⚙ CHART TERMINAL")

UNIVERSE = get_universe()
ticker = st.sidebar.selectbox("Ticker", UNIVERSE, key="chart_ticker")

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


df = load_price_history(ticker)

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

rsi_cross_up   = (rsi_series.shift(1) < 30) & (rsi_series >= 30)
rsi_cross_down = (rsi_series.shift(1) > 80) & (rsi_series <= 80)

# ============================================================
# NORMALIZATION
# ============================================================

base = float(close.iloc[0])


def maybe_norm(s: pd.Series, do_it: bool) -> pd.Series:
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
# METRIC CARDS
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
idx = df.index[df.index >= cutoff]


def v(series: pd.Series) -> pd.Series:
    return series.loc[idx]


up_idx   = idx[rsi_cross_up.loc[idx]]   if len(idx) else idx
down_idx = idx[rsi_cross_down.loc[idx]] if len(idx) else idx

vol_colors = [
    "rgba(5,150,105,0.5)" if float(c) >= float(o) else "rgba(220,38,38,0.5)"
    for c, o in zip(v(close), v(opens))
]

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
specs  = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_rows - 1)

# ============================================================
# CHART — TradingView Lightweight Charts with Plotly fallback
# ============================================================

import streamlit.components.v1 as components

_ph      = 360
_vh      = 90
_rh      = 150 if show_rsi  else 0
_mh      = 150 if show_macd else 0
_total_h = _ph + _vh + _rh + _mh


def _ser(name: str, s: pd.Series) -> str:
    pts = []
    for dt, val in s.loc[idx].items():
        if pd.notna(val):
            pts.append(f'{{"time":"{dt.strftime("%Y-%m-%d")}","value":{float(val):.4f}}}')
    return f"const {name} = [{','.join(pts)}];"


def _candles() -> tuple[str, str]:
    raw, norm = [], []
    for dt in idx:
        o  = float(opens.loc[dt])
        h_ = float(high.loc[dt])
        l_ = float(low.loc[dt])
        c_ = float(close.loc[dt])
        nc = float(plot_close.loc[dt])
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


def _macd_hist_js() -> str:
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


_cdecl, _ndecl = _candles()
_vdecl   = _volume()
_mkdecl  = _markers()
_rsidecl = _ser("rsiData",  rsi_series)    if show_rsi  else "const rsiData  = [];"
_mldecl  = _ser("macdLine", macd_line)     if show_macd else "const macdLine = [];"
_msdecl  = _ser("macdSig",  macd_sig_line) if show_macd else "const macdSig  = [];"
_mhdecl  = _macd_hist_js()
_budecl  = _ser("bbUpper",  plot_bb_upper) if show_bb   else "const bbUpper  = [];"
_bldecl  = _ser("bbLower",  plot_bb_lower) if show_bb   else "const bbLower  = [];"
_bmdecl  = _ser("bbMid",    plot_bb_mid)   if show_bb   else "const bbMid    = [];"
_vwdecl  = _ser("vwapData", plot_vwap)     if show_vwap else "const vwapData = [];"
_madecl  = _mas()
_nma     = len(ma_configs)

_ma_data_js  = "[" + ",".join(f"maData{i}"   for i in range(_nma)) + "]"
_ma_col_js   = "[" + ",".join(f"maColor{i}"  for i in range(_nma)) + "]"
_ma_per_js   = "[" + ",".join(f"maPeriod{i}" for i in range(_nma)) + "]"

tv_html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; font-family: system-ui, sans-serif; overflow: hidden; }}
  .pw {{ position: relative; width: 100%; }}
  .pl {{
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

const NORMALIZE = {str(normalize).lower()};
const SHOW_RSI  = {str(show_rsi).lower()};
const SHOW_MACD = {str(show_macd).lower()};
const SHOW_BB   = {str(show_bb).lower()};
const SHOW_VWAP = {str(show_vwap).lower()};

const MA_DATA    = {_ma_data_js};
const MA_COLORS  = {_ma_col_js};
const MA_PERIODS = {_ma_per_js};

const PRICE_H = {_ph};
const VOL_H   = {_vh};
const RSI_H   = {_rh};
const MACD_H  = {_mh};

setTimeout(function() {{

  const W = window.innerWidth || 800;

  function makeOpts(h, showTime) {{
    return {{
      width: W, height: h,
      layout:    {{ background: {{ color: "#ffffff" }}, textColor: "#64748b", fontSize: 11 }},
      grid:      {{ vertLines: {{ color: "#e8edf5" }}, horzLines: {{ color: "#e8edf5" }} }},
      crosshair: {{ mode: 1 }},
      timeScale: {{
        borderColor: "#dce4ef", timeVisible: true, secondsVisible: false, visible: !!showTime,
      }},
      rightPriceScale: {{ borderColor: "#dce4ef" }},
      handleScale: true, handleScroll: true,
    }};
  }}

  function addPane(id, label, h) {{
    const root = document.getElementById("root");
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;width:100%;height:" + h + "px;";
    const el = document.createElement("div");
    el.id = id; el.style.cssText = "width:" + W + "px;height:" + h + "px;";
    wrap.appendChild(el);
    if (label) {{
      const lbl = document.createElement("div");
      lbl.className = "pl"; lbl.textContent = label;
      wrap.appendChild(lbl);
    }}
    root.appendChild(wrap);
    const div = document.createElement("div");
    div.className = "div"; root.appendChild(div);
    return el;
  }}

  const priceEl = addPane("p-price", "{ticker}", PRICE_H);
  const volEl   = addPane("p-vol",   "VOL",      VOL_H);
  const rsiEl   = SHOW_RSI  ? addPane("p-rsi",  "RSI {int(rsi_period)}", RSI_H)  : null;
  const macdEl  = SHOW_MACD ? addPane("p-macd", "MACD",                  MACD_H) : null;

  const bottomIsRsi  = SHOW_RSI  && !SHOW_MACD;
  const bottomIsMacd = SHOW_MACD;
  const bottomIsVol  = !SHOW_RSI && !SHOW_MACD;

  const priceChart = LightweightCharts.createChart(priceEl, makeOpts(PRICE_H, false));
  const volChart   = LightweightCharts.createChart(volEl,   makeOpts(VOL_H,   bottomIsVol));
  const rsiChart   = SHOW_RSI  ? LightweightCharts.createChart(rsiEl,  makeOpts(RSI_H,  bottomIsRsi))  : null;
  const macdChart  = SHOW_MACD ? LightweightCharts.createChart(macdEl, makeOpts(MACD_H, true))         : null;

  const allCharts = [priceChart, volChart, rsiChart, macdChart].filter(Boolean);

  let priceSeries;
  if (NORMALIZE) {{
    priceSeries = priceChart.addLineSeries({{ color: "#1d4ed8", lineWidth: 2 }});
    priceSeries.setData(normalizedData);
  }} else {{
    priceSeries = priceChart.addCandlestickSeries({{
      upColor: "#059669", downColor: "#dc2626",
      borderUpColor: "#059669", borderDownColor: "#dc2626",
      wickUpColor: "#059669", wickDownColor: "#dc2626",
    }});
    priceSeries.setData(candleData);
  }}

  if (markerData.length) priceSeries.setMarkers(markerData);

  if (SHOW_BB) {{
    const bbU = priceChart.addLineSeries({{ color: "#9b59b6", lineWidth: 1, lineStyle: 2, title: "BB ↑" }});
    const bbL = priceChart.addLineSeries({{ color: "#9b59b6", lineWidth: 1, lineStyle: 2, title: "BB ↓" }});
    const bbM = priceChart.addLineSeries({{ color: "#7c3aed", lineWidth: 1, title: "BB mid" }});
    bbU.setData(bbUpper); bbL.setData(bbLower); bbM.setData(bbMid);
  }}

  MA_DATA.forEach((data, i) => {{
    priceChart.addLineSeries({{ color: MA_COLORS[i], lineWidth: 1.5, title: "MA " + MA_PERIODS[i] }}).setData(data);
  }});

  if (SHOW_VWAP) {{
    priceChart.addLineSeries({{ color: "#0f2044", lineWidth: 1.5, lineStyle: 2, title: "VWAP" }}).setData(vwapData);
  }}

  volChart.applyOptions({{ rightPriceScale: {{ visible: false }}, leftPriceScale: {{ visible: false }} }});
  volChart.addHistogramSeries({{ priceFormat: {{ type: "volume" }} }}).setData(volumeData);

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

  if (macdChart) {{
    macdChart.addHistogramSeries({{ priceFormat: {{ type: "price", precision: 3 }} }}).setData(macdHist);
    macdChart.addLineSeries({{ color: "#1d4ed8", lineWidth: 1.5 }}).setData(macdLine);
    macdChart.addLineSeries({{ color: "#dc2626", lineWidth: 1.5 }}).setData(macdSig);
  }}

  allCharts.forEach(src => {{
    src.timeScale().subscribeVisibleLogicalRangeChange(range => {{
      if (!range) return;
      allCharts.forEach(dst => {{ if (dst !== src) dst.timeScale().setVisibleLogicalRange(range); }});
    }});
  }});

  window.addEventListener("resize", () => {{
    const newW = window.innerWidth;
    allCharts.forEach(c => c.applyOptions({{ width: newW }}));
  }});

}}, 150);
</script>
</body>
</html>"""

st.caption(f"debug — candle bars in range: {len(idx)}  |  first date: {idx[0].date() if len(idx) else 'none'}")

_tv_ok = False
try:
    components.html(tv_html, height=_total_h + 8, scrolling=False)
    _tv_ok = True
except Exception as _tv_err:
    st.warning(f"TradingView render failed ({_tv_err}) — falling back to Plotly.")

if not _tv_ok:
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=panel_titles,
        specs=specs,
    )

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
# FUNDAMENTALS TABLE
# ============================================================


@st.cache_data(ttl=900)
def load_info(tkr: str) -> dict:
    try:
        return yf.Ticker(tkr).info
    except Exception:
        return {}


info = load_info(ticker)


def fv(key):
    return info.get(key, "N/A")


def fp(key):
    val = info.get(key)
    return f"${val:,.2f}" if val else "N/A"


def fvol(key):
    val = info.get(key)
    return f"{val:,.0f}" if val else "N/A"


def fcap(key):
    val = info.get(key)
    if not val:
        return "N/A"
    if val >= 1e12:
        return f"{val/1e12:.3f}T"
    if val >= 1e9:
        return f"{val/1e9:.2f}B"
    return f"{val/1e6:.2f}M"


def fts(key):
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
