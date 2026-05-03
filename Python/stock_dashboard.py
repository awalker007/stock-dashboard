# stock_dashboard.py
# ============================================================
# Magnificent 7 Stock Dashboard — Streamlit + Plotly
# Launch: streamlit run stock_dashboard.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import anthropic as _anthropic

st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ── Sidebar ─────────────────────────────────────────────────────────────
       CONTRAST FIX: The sidebar background is #2596be (medium blue). The old
       rule "[data-testid="stSidebar"] * { color: #ffffff }" set EVERY element
       to white — including input field text, which also had a near-white
       background, making it invisible. The fix uses two explicit layers instead
       of a wildcard:
         Layer 1 — static text (labels, headings, markdown) → white, readable on blue.
         Layer 2 — interactive elements (inputs, selects, number fields) → white
                   background with dark #0f2044 text so the user can read them.
    ──────────────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] { background-color: #2596be; border-right: 1px solid #1a7fa0; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }

    /* Layer 1 — all static text defaults to white on the blue sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }

    /* Widget labels sit on the blue sidebar — slightly softened white */
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stNumberInput > label,
    [data-testid="stSidebar"] .stTextInput > label,
    [data-testid="stSidebar"] .stToggle > label,
    [data-testid="stSidebar"] .stCheckbox > label,
    [data-testid="stSidebar"] .stRadio > label { color: #d0eef8 !important; font-size: 0.75rem !important; }

    /* Checkbox and radio option text */
    [data-testid="stSidebar"] .stCheckbox span,
    [data-testid="stSidebar"] .stRadio span { color: #ffffff !important; }

    /* Toggle labels */
    [data-testid="stSidebar"] .stToggle label { color: #ffffff !important; }

    /* Layer 2 — interactive elements: white bg, dark text */
    [data-testid="stSidebar"] input {
        color: #0f2044 !important;
        background-color: #ffffff !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stSidebar"] input::placeholder { color: #7bafd4 !important; }

    /* Selectbox / dropdown — white background, dark selected-value text */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div { color: #0f2044 !important; }

    /* Dropdown option list (Streamlit renders this in a portal outside the sidebar) */
    [data-baseweb="popover"] [role="option"] { color: #0f2044 !important; }
    [data-baseweb="popover"] [role="option"]:hover { background: #e0f0ff !important; }

    /* Number input wrapper */
    [data-testid="stSidebar"] [data-baseweb="input"] { background-color: #ffffff !important; }
    [data-testid="stSidebar"] [data-baseweb="input"] input { color: #0f2044 !important; }

    /* Color picker hex field */
    [data-testid="stSidebar"] [data-testid="stColorPicker"] input { color: #0f2044 !important; }

    /* MA / indicator cards */
    [data-testid="stSidebar"] .ma-card {
        background: rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 8px 10px 4px 10px;
        margin-bottom: 6px;
        border-left: 4px solid rgba(255,255,255,0.5);
    }
    /* Toggle track color */
    [data-testid="stSidebar"] [data-testid="stToggle"] > div { background-color: rgba(255,255,255,0.25) !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] { background:#fff; border:1px solid #dce4ef; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(15,32,68,0.08); }
    [data-testid="stMetricLabel"] { color:#64748b !important; font-size:0.72rem !important; letter-spacing:0.08em; text-transform:uppercase; }
    [data-testid="stMetricValue"] { color:#0f2044 !important; font-size:1.3rem !important; font-weight:700; }
    [data-testid="stMetricDelta"] svg { display:none; }

    /* ── Range radio → styled like pill buttons ── */
    div[data-testid="stRadio"] > div { display:flex; flex-wrap:nowrap; gap:5px; margin-bottom:6px; }
    div[data-testid="stRadio"] label {
        background: #ffffff;
        border: 1.5px solid #dce4ef;
        border-radius: 20px;
        padding: 4px 13px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
        transition: all 0.15s;
        white-space: nowrap;
    }
    div[data-testid="stRadio"] label:hover { border-color: #1d4ed8; color: #1d4ed8; }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #1d4ed8;
        color: #ffffff !important;
        border-color: #1d4ed8;
    }
    div[data-testid="stRadio"] label input { display:none; }

    .stPlotlyChart { border:1px solid #dce4ef; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(15,32,68,0.08); }
    h1 { color:#0f2044 !important; font-size:1.4rem !important; font-weight:700; }
    h2, h3 { color:#475569 !important; font-size:0.85rem !important; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙ CONTROLS")

TICKERS = ["SPY", "VOO", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "WMT", "CHRW"]
ticker = st.sidebar.selectbox("Ticker", TICKERS)

st.sidebar.markdown("---")
normalize = st.sidebar.toggle("Normalize to 100", value=False)
st.sidebar.markdown("---")

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
    bb_period = st.sidebar.number_input("BB Period",         min_value=2,   max_value=200, value=20,  step=1)
    bb_std    = st.sidebar.number_input("Std Dev Multiplier",min_value=0.5, max_value=5.0, value=2.0, step=0.5)
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
st.sidebar.markdown("**AI News**")
anthropic_key = st.sidebar.text_input(
    "Anthropic API Key", type="password", placeholder="sk-ant-...",
    help="Get a key at console.anthropic.com"
)

# ============================================================
# DATA — always fetch 5Y; range pills slice the view
# ============================================================

@st.cache_data
def load_data(tkr: str) -> pd.DataFrame:
    today = date.today()
    start = today - timedelta(days=365 * 5)
    df = yf.download(tkr, start=str(start), end=str(today), auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_localize(None)   # strip timezone so date arithmetic is simple
    return df

df = load_data(ticker)

if df.empty:
    st.error("No data returned.")
    st.stop()

close  = df["Close"].squeeze()
opens  = df["Open"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
volume = df["Volume"].squeeze()

# ============================================================
# INDICATORS — calculated on full 5Y for accuracy
# ============================================================

def calculate_rsi(series, period):
    delta    = series.diff()
    gains    = delta.clip(lower=0)
    losses   = (-delta).clip(lower=0)
    alpha    = 1 / period
    avg_gain = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast, slow, sig):
    ml = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

def calculate_bb(series, period, num_std):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + num_std * std, sma, sma - num_std * std

rsi_series                    = calculate_rsi(close, int(rsi_period))
macd_line, macd_sig_line, macd_hist = calculate_macd(close, int(macd_fast), int(macd_slow), int(macd_sig))
bb_upper, bb_mid, bb_lower    = calculate_bb(close, int(bb_period), bb_std)
typical_price                 = (high + low + close) / 3
vwap_series                   = (typical_price * volume).cumsum() / volume.cumsum()

rsi_cross_up   = (rsi_series.shift(1) < 30) & (rsi_series >= 30)
rsi_cross_down = (rsi_series.shift(1) > 80) & (rsi_series <= 80)

# ============================================================
# NORMALIZATION
# ============================================================

base = float(close.iloc[0])

def maybe_norm(s, do_it):
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

today  = date.today()
latest = float(close.iloc[-1])
prev   = float(close.iloc[-2]) if len(close) > 1 else latest
chg    = latest - prev
chg_pct = (chg / prev) * 100
high52  = float(close.rolling(252).max().iloc[-1])
low52   = float(close.rolling(252).min().iloc[-1])
avgvol  = float(volume.rolling(20).mean().iloc[-1])
arrow   = "▲" if chg >= 0 else "▼"

st.markdown(f"## {ticker} &nbsp;<span style='font-size:1rem;color:#64748b'>Stock Dashboard</span>", unsafe_allow_html=True)
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

def calc_return(series, n_days):
    if len(series) <= n_days:
        return None
    return (float(series.iloc[-1]) / float(series.iloc[-1 - n_days]) - 1) * 100

ytd_slice = close[close.index.year == today.year]
ytd_ret   = (float(close.iloc[-1]) / float(ytd_slice.iloc[0]) - 1) * 100 if len(ytd_slice) > 1 else None

return_periods = [
    ("1W", calc_return(close,5)), ("1M", calc_return(close,21)),
    ("3M", calc_return(close,63)), ("6M", calc_return(close,126)),
    ("YTD", ytd_ret), ("1Y", calc_return(close,252)),
    ("3Y", calc_return(close,756)), ("5Y", calc_return(close,1260)),
]

def fmt(val):
    if val is None: return "—", "#94a3b8"
    return (f"+{val:.2f}%", "#059669") if val >= 0 else (f"{val:.2f}%", "#dc2626")

cells = "".join(f"""<td style='text-align:center;padding:10px 0;border-right:1px solid #e8edf5;'>
    <div style='font-size:0.68rem;color:#64748b;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:5px;'>{lbl}</div>
    <div style='font-size:1rem;font-weight:700;color:{fmt(v)[1]};'>{fmt(v)[0]}</div></td>"""
    for lbl, v in return_periods)

st.markdown(f"""<div style='background:#fff;border:1px solid #dce4ef;border-radius:8px;
    overflow:hidden;box-shadow:0 1px 4px rgba(15,32,68,0.08);margin-bottom:16px;'>
  <table style='width:100%;border-collapse:collapse;'><tr>{cells}</tr></table></div>""",
    unsafe_allow_html=True)

# ============================================================
# RANGE SELECTOR
# ============================================================
# The selected pill slices df to only the relevant rows.
# We pass this sliced data directly to Plotly — no axis range tricks needed.
# Indicators are still calculated on the full 5Y above for accuracy.

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

# Slice every series to the selected range — this is what actually changes the chart
idx = df.index[df.index >= cutoff]

def v(series):
    """Return only the rows inside the selected date range."""
    return series.loc[idx]

# Sliced signal marker dates
up_idx   = idx[rsi_cross_up.loc[idx]]   if len(idx) else idx
down_idx = idx[rsi_cross_down.loc[idx]] if len(idx) else idx

# Volume colours for the sliced window
vol_colors = [
    "rgba(5,150,105,0.5)" if float(c) >= float(o) else "rgba(220,38,38,0.5)"
    for c, o in zip(v(close), v(opens))
]

# Marker offset relative to the sliced price range
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

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
    subplot_titles=panel_titles,
    specs=specs,
)

# ============================================================
# ROW 1 — PRICE AREA CHART + OVERLAYS
# ============================================================

fig.add_trace(go.Scatter(
    x=idx, y=v(plot_close),
    mode="lines", name=ticker, showlegend=False,
    line=dict(color="#1d4ed8", width=2),
    fill="tozeroy", fillcolor="rgba(29,78,216,0.12)",
), row=1, col=1, secondary_y=False)

if show_bb:
    fig.add_trace(go.Scatter(x=idx, y=v(plot_bb_upper), mode="lines", name="BB Upper",
        line=dict(color="#9b59b6", width=1, dash="dot")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=v(plot_bb_lower), mode="lines", name="BB Lower",
        line=dict(color="#9b59b6", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(155,89,182,0.08)", showlegend=False),
        row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=v(plot_bb_mid), mode="lines", name="BB Mid",
        line=dict(color="#7c3aed", width=1)), row=1, col=1, secondary_y=False)

for (period, color) in ma_configs:
    ma_vals = maybe_norm(close.rolling(window=period).mean(), normalize)
    fig.add_trace(go.Scatter(x=idx, y=v(ma_vals), mode="lines", name=f"MA {period}",
        line=dict(color=color, width=1.5)), row=1, col=1, secondary_y=False)

if show_vwap:
    fig.add_trace(go.Scatter(x=idx, y=v(plot_vwap), mode="lines", name="VWAP",
        line=dict(color="#0f2044", width=1.5, dash="dash")), row=1, col=1, secondary_y=False)

if show_signals:
    if len(up_idx) > 0:
        fig.add_trace(go.Scatter(
            x=up_idx, y=v(plot_low).loc[up_idx] - marker_offset,
            mode="markers", name="RSI Oversold Cross",
            marker=dict(symbol="triangle-up", color="#059669", size=10,
                        line=dict(color="#334155", width=0.5)),
        ), row=1, col=1, secondary_y=False)
    if len(down_idx) > 0:
        fig.add_trace(go.Scatter(
            x=down_idx, y=v(plot_high).loc[down_idx] + marker_offset,
            mode="markers", name="RSI Overbought Cross",
            marker=dict(symbol="triangle-down", color="#dc2626", size=10,
                        line=dict(color="#334155", width=0.5)),
        ), row=1, col=1, secondary_y=False)

# ============================================================
# VOLUME BARS (secondary y-axis, inside price panel)
# ============================================================

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

# ============================================================
# MACD PANEL
# ============================================================

if show_macd:
    r = row_map["macd"]
    hist_colors = ["#059669" if val >= 0 else "#dc2626" for val in v(macd_hist).fillna(0)]
    fig.add_trace(go.Bar(x=idx, y=v(macd_hist), marker_color=hist_colors,
        marker_line_width=0, name="Histogram", showlegend=False), row=r, col=1)
    fig.add_trace(go.Scatter(x=idx, y=v(macd_line), mode="lines", name="MACD",
        line=dict(color="#1d4ed8", width=1.5)), row=r, col=1)
    fig.add_trace(go.Scatter(x=idx, y=v(macd_sig_line), mode="lines", name="Signal",
        line=dict(color="#dc2626", width=1.5)), row=r, col=1)

# ============================================================
# RSI PANEL
# ============================================================

if show_rsi:
    r = row_map["rsi"]
    fig.add_trace(go.Scatter(x=idx, y=v(rsi_series), mode="lines", name="RSI",
        line=dict(color="#7c3aed", width=1.5), showlegend=False), row=r, col=1)

    for level, color, label in [(80,"#dc2626","80"), (50,"#94a3b8","50"), (30,"#059669","30")]:
        fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1),
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="right", row=r, col=1)

    fig.add_hrect(y0=80, y1=100, fillcolor="#dc2626", opacity=0.07, line_width=0, row=r, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#059669", opacity=0.07, line_width=0, row=r, col=1)

# ============================================================
# LAYOUT
# ============================================================

total_height = 540 + (180 if show_macd else 0) + (180 if show_rsi else 0)

AXIS_STYLE = dict(gridcolor="#e8edf5", tickfont=dict(color="#64748b", size=11),
                  showline=True, linecolor="#dce4ef", zeroline=False)

fig.update_layout(
    paper_bgcolor="#f0f4f8", plot_bgcolor="#ffffff",
    hovermode="x unified",
    font=dict(color="#334155", family="Inter, system-ui, sans-serif"),
    dragmode="pan",
    height=total_height,
    margin=dict(l=60, r=40, t=50, b=80),
    legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0,
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#dce4ef", borderwidth=1,
                font=dict(color="#334155")),
    barmode="overlay",
)

fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)
fig.update_yaxes(title=y_label, tickprefix=y_prefix, row=1, col=1, secondary_y=False)
if show_macd:
    fig.update_yaxes(title="MACD", row=row_map["macd"], col=1)
if show_rsi:
    fig.update_yaxes(title="RSI", range=[0,100], tickvals=[0,30,50,80,100],
                     row=row_map["rsi"], col=1)

for ann in fig.layout.annotations:
    ann.font = dict(color="#94a3b8", size=10)

st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True, "displayModeBar": True,
    "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d","toImage"],
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

def fv(key):        return info.get(key, "N/A")
def fp(key):
    v = info.get(key); return f"${v:,.2f}" if v else "N/A"
def fvol(key):
    v = info.get(key); return f"{v:,.0f}" if v else "N/A"
def fcap(key):
    v = info.get(key)
    if not v: return "N/A"
    if v >= 1e12: return f"{v/1e12:.3f}T"
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    return f"{v/1e6:.2f}M"
def fts(key):
    v = info.get(key)
    if not v: return "N/A"
    try: return pd.Timestamp(v, unit="s").strftime("%b %d, %Y")
    except: return "N/A"

dl, dh = info.get("dayLow"), info.get("dayHigh")
wl, wh = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
bid, bsz = info.get("bid"), info.get("bidSize","")
ask, asz = info.get("ask"), info.get("askSize","")
dr, dy   = info.get("dividendRate"), info.get("dividendYield")

rows = [
    ("Previous Close", fp("previousClose"),   "Day's Range",         f"${dl:,.2f} – ${dh:,.2f}" if dl and dh else "N/A"),
    ("Open",           fp("open"),            "52 Week Range",       f"${wl:,.2f} – ${wh:,.2f}" if wl and wh else "N/A"),
    ("Bid",            f"${bid:,.2f} × {bsz}" if bid else "N/A",    "Volume",        fvol("volume")),
    ("Ask",            f"${ask:,.2f} × {asz}" if ask else "N/A",    "Avg. Volume",   fvol("averageVolume")),
    ("Market Cap",     fcap("marketCap"),     "Beta (5Y Monthly)",   str(fv("beta"))),
    ("PE Ratio (TTM)", str(fv("trailingPE")), "EPS (TTM)",           str(fv("trailingEps"))),
    ("Earnings Date",  fts("earningsTimestamp"), "Fwd Div & Yield",  f"{dr:.2f} ({dy*100:.2f}%)" if dr and dy else "N/A"),
    ("Ex-Div Date",    fts("exDividendDate"), "1Y Target Est",       fp("targetMeanPrice")),
]

def fund_row(l1, v1, l2, v2, shade):
    bg = "#f8fafc" if shade else "#ffffff"
    td = "padding:9px 16px;font-size:0.82rem;border-bottom:1px solid #e8edf5;"
    return (f"<tr style='background:{bg};'>"
            f"<td style='{td}color:#64748b;width:22%;'>{l1}</td>"
            f"<td style='{td}color:#0f2044;font-weight:600;width:28%;'>{v1}</td>"
            f"<td style='{td}color:#64748b;width:22%;border-left:1px solid #e8edf5;'>{l2}</td>"
            f"<td style='{td}color:#0f2044;font-weight:600;width:28%;'>{v2}</td></tr>")

table_rows = "".join(fund_row(l1,v1,l2,v2, i%2==0) for i,(l1,v1,l2,v2) in enumerate(rows))

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='background:#fff;border:1px solid #dce4ef;border-radius:8px;
            overflow:hidden;box-shadow:0 2px 8px rgba(15,32,68,0.06);'>
  <div style='padding:10px 16px;background:#2596be;'>
    <span style='color:#fff;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'>
      {ticker} — Key Statistics
    </span>
  </div>
  <table style='width:100%;border-collapse:collapse;'>{table_rows}</table>
</div>""", unsafe_allow_html=True)

# ============================================================
# AI NEWS SECTION
# ============================================================

@st.cache_data(ttl=900)
def fetch_news(tkr: str):
    """Return list of (title, summary, source, url, date) tuples."""
    try:
        items = yf.Ticker(tkr).news or []
    except Exception:
        return []
    results = []
    for item in items[:8]:
        c = item.get("content", {})
        title   = c.get("title", "")
        summary = c.get("summary", "") or c.get("description", "")
        source  = c.get("provider", {}).get("displayName", "")
        url_obj = c.get("canonicalUrl", {})
        url     = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
        pub     = c.get("pubDate", "")[:10]
        if title:
            results.append((title, summary, source, url, pub))
    return results

@st.cache_data(ttl=900, show_spinner=False)
def ai_news_brief(tkr: str, headlines_blob: str, api_key: str) -> str:
    """Ask Claude to write a concise market brief from the headlines."""
    client = _anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"You are a sharp financial analyst. Here are the latest news headlines for {tkr}:\n\n"
        f"{headlines_blob}\n\n"
        "Write a concise 3-5 sentence market brief that:\n"
        "- Identifies the dominant theme or story\n"
        "- Notes any material risk or catalyst\n"
        "- Ends with a one-sentence sentiment read (bullish / neutral / bearish and why)\n"
        "Be direct. No filler. No bullet points."
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

news_items = fetch_news(ticker)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown(
        f"<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.08em;"
        f"text-transform:uppercase;color:#475569;margin-bottom:10px;'>"
        f"Latest News — {ticker}</div>",
        unsafe_allow_html=True,
    )
    if not news_items:
        st.markdown("<span style='color:#94a3b8;font-size:0.82rem;'>No recent headlines found.</span>",
                    unsafe_allow_html=True)
    else:
        for title, summary, source, url, pub in news_items:
            link = f"<a href='{url}' target='_blank' style='color:#1d4ed8;text-decoration:none;font-weight:600;font-size:0.84rem;'>{title}</a>" if url else f"<span style='color:#0f2044;font-weight:600;font-size:0.84rem;'>{title}</span>"
            meta = f"<span style='color:#94a3b8;font-size:0.72rem;'>{source}{'  ·  ' + pub if pub else ''}</span>"
            blurb = f"<div style='color:#475569;font-size:0.78rem;margin-top:2px;'>{summary[:120]}{'…' if len(summary)>120 else ''}</div>" if summary else ""
            st.markdown(
                f"<div style='background:#fff;border:1px solid #e8edf5;border-radius:7px;"
                f"padding:10px 14px;margin-bottom:7px;'>{link}<br>{meta}{blurb}</div>",
                unsafe_allow_html=True,
            )

with right_col:
    st.markdown(
        "<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#475569;margin-bottom:10px;'>"
        "AI Market Brief</div>",
        unsafe_allow_html=True,
    )
    if not anthropic_key:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8edf5;border-radius:7px;padding:16px;'>"
            "<span style='color:#94a3b8;font-size:0.82rem;'>Enter your Anthropic API key in the sidebar "
            "to generate an AI market brief.</span></div>",
            unsafe_allow_html=True,
        )
    elif not news_items:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8edf5;border-radius:7px;padding:16px;'>"
            "<span style='color:#94a3b8;font-size:0.82rem;'>No headlines to analyze.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        headlines_blob = "\n".join(
            f"{i+1}. {t} — {s[:100]}" for i, (t, s, *_) in enumerate(news_items)
        )
        with st.spinner("Analyzing headlines…"):
            try:
                brief = ai_news_brief(ticker, headlines_blob, anthropic_key)
                st.markdown(
                    f"<div style='background:#fff;border:1px solid #dce4ef;border-radius:7px;"
                    f"padding:16px;font-size:0.84rem;color:#0f2044;line-height:1.65;'>{brief}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='text-align:right;margin-top:4px;'>"
                    "<span style='font-size:0.68rem;color:#cbd5e1;'>Powered by Claude · refreshes every 15 min</span></div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.markdown(
                    f"<div style='background:#fff4f4;border:1px solid #fca5a5;border-radius:7px;padding:14px;"
                    f"font-size:0.82rem;color:#dc2626;'>Error: {e}</div>",
                    unsafe_allow_html=True,
                )
