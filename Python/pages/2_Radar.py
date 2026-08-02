# pages/2_Radar.py
# ============================================================
# Radar — "what is setting up right now" page.
#
# This page answers the question: which stocks in the universe are
# showing real momentum buildup right now, and what is the smart
# watchlist telling us to pay attention to?
#
# Contains:
#   - Momentum Radar: scans the universe for stocks meeting all 5
#     criteria simultaneously (rising lows, above-average volume,
#     strong 6-month return, RSI 50-70, price above both MAs)
#   - Smart Watchlist: aggregates radar hits + IQ thesis stocks +
#     recent alert tickers + manually pinned tickers into one feed
#     with sparklines and daily change
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from utils import inject_css, universe_manager_sidebar, section_label
from utils import BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG, ACCENT
from database import (
    initialize_database, get_universe, get_recent_alerts,
    get_all_iq_stock_names, get_watchlist_pins, add_watchlist_pin, remove_watchlist_pin,
)
from indicators import calculate_rsi

st.set_page_config(page_title="Radar", page_icon="📡", layout="wide")
initialize_database()
inject_css()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("## 📡  RADAR")
st.sidebar.markdown("---")
universe_manager_sidebar()

st.markdown(
    f"<h1 style='margin-bottom:4px;'>📡 Radar</h1>"
    f"<div style='color:{TEXT_LIGHT};font-size:0.82rem;margin-bottom:20px;'>"
    f"What is setting up right now · {date.today().strftime('%A, %B %d, %Y')}</div>",
    unsafe_allow_html=True,
)

UNIVERSE = get_universe()

# ============================================================
# SECTION 1 — MOMENTUM RADAR
# ============================================================


@st.cache_data(ttl=86400, show_spinner=False)
def run_momentum_radar(tickers: tuple) -> list[dict]:
    """
    Scan universe for stocks meeting all 5 momentum criteria.
    Cached 24h — one expensive scan per day.
    """
    flagged = []
    if not tickers:
        return flagged
    today = date.today()
    start_6m = today - timedelta(days=182)
    start_1y = today - timedelta(days=400)

    # Fetch every ticker's history in a single batched call instead of one
    # request per ticker — yfinance multithreads the batch internally.
    batch = yf.download(list(tickers), start=str(start_1y), end=str(today),
                        group_by="ticker", auto_adjust=True, progress=False)
    batch.index = batch.index.tz_localize(None)

    for tkr in tickers:
        try:
            df = batch[tkr]
            close  = df["Close"].squeeze()
            volume = df["Volume"].squeeze()
            if close.dropna().shape[0] < 130:
                continue

            # Criterion 1: 20-day lows are higher now vs 3 months ago
            recent_20d_low  = float(close.iloc[-20:].min())
            earlier_20d_low = float(close.iloc[-83:-63].min()) if len(close) > 83 else None
            if earlier_20d_low is None or recent_20d_low <= earlier_20d_low:
                continue

            # Criterion 2: 20D avg vol ≥ 120% of 3-6mo avg vol
            vol_20d   = float(volume.iloc[-20:].mean())
            vol_3_6mo = float(volume.iloc[-182:-63].mean()) if len(volume) > 182 else float(volume.mean())
            if vol_20d < vol_3_6mo * 1.20:
                continue

            # Criterion 3: Up ≥15% over 6 months
            idx_6m = df.index[df.index >= pd.Timestamp(start_6m)]
            if len(idx_6m) < 2:
                continue
            ret_6m = (float(close.iloc[-1]) / float(close.loc[idx_6m[0]]) - 1) * 100
            if ret_6m < 15.0:
                continue

            # Criterion 4: RSI 50-70
            rsi_s   = calculate_rsi(close, 14)
            rsi_now = float(rsi_s.iloc[-1])
            if not (50 <= rsi_now <= 70):
                continue

            # Criterion 5: Price above 50D and 200D MA
            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])
            cur   = float(close.iloc[-1])
            if cur <= ma50 or cur <= ma200:
                continue

            flagged.append({
                "ticker":     tkr,
                "price":      cur,
                "ret_6m":     ret_6m,
                "rsi":        rsi_now,
                "vol_ratio":  vol_20d / vol_3_6mo,
                "ma50":       ma50,
                "ma200":      ma200,
                "low_higher": (recent_20d_low - earlier_20d_low) / earlier_20d_low * 100,
            })
        except Exception:
            pass
    return flagged


section_label("Momentum Radar")
with st.spinner("Scanning universe for momentum setups (cached 24h)…"):
    radar_hits = run_momentum_radar(tuple(UNIVERSE))

if not radar_hits:
    st.markdown(
        f"<div style='font-size:0.83rem;color:{TEXT_LIGHT};margin-bottom:8px;'>"
        f"No stocks in the current universe meet all 5 momentum criteria today.</div>",
        unsafe_allow_html=True,
    )
else:
    radar_cols = st.columns(min(len(radar_hits), 3))
    for i, hit in enumerate(radar_hits):
        col = radar_cols[i % 3]
        with col:
            col.markdown(
                f"<div style='background:#fff;border:1px solid {BORDER};"
                f"border-left:4px solid {GREEN};border-radius:8px;padding:12px 14px;"
                f"margin-bottom:10px;'>"
                f"<div style='font-size:0.88rem;font-weight:700;color:{TEXT_DARK};"
                f"margin-bottom:6px;'>{hit['ticker']} "
                f"<span style='font-size:0.78rem;font-weight:400;color:{GREEN};'>"
                f"${hit['price']:,.2f}</span></div>"
                f"<div style='font-size:0.75rem;color:{TEXT_MID};line-height:1.6;'>"
                f"↑ {hit['ret_6m']:.1f}% over 6 months<br>"
                f"RSI {hit['rsi']:.0f} · Vol {hit['vol_ratio']:.1f}× avg<br>"
                f"20D lows rising {hit['low_higher']:.1f}%<br>"
                f"Above MA50 (${hit['ma50']:,.0f}) & MA200 (${hit['ma200']:,.0f})"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        # Auto-add radar hits to the watchlist so they persist across sessions
        add_watchlist_pin(hit["ticker"], reason="Momentum Radar")

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 2 — SMART WATCHLIST
# ============================================================

section_label("Smart Watchlist")

# Build watchlist: pins + IQ stocks + recent alerts + radar flags
wl_pins = {p["ticker"]: p["reason"] for p in get_watchlist_pins()}
iq_stocks = get_all_iq_stock_names()
recent_alert_tickers = list(dict.fromkeys(
    a["ticker"] for a in get_recent_alerts(limit=30)
    if datetime.strptime(a["timestamp"][:10], "%Y-%m-%d") >= datetime.now() - timedelta(days=5)
))

watchlist_all: dict[str, str] = {}
for tkr in iq_stocks:
    watchlist_all[tkr] = watchlist_all.get(tkr, "IQ Thesis")
for tkr in recent_alert_tickers:
    if tkr not in watchlist_all:
        watchlist_all[tkr] = "LEAP Alert"
for tkr in [h["ticker"] for h in radar_hits]:
    if tkr not in watchlist_all:
        watchlist_all[tkr] = "Momentum Radar"
for tkr, reason in wl_pins.items():
    if tkr not in watchlist_all:
        watchlist_all[tkr] = reason if reason != "Momentum Radar" else "Momentum Radar"

if not watchlist_all:
    st.markdown(
        f"<div style='font-size:0.83rem;color:{TEXT_LIGHT};'>"
        f"Watchlist is empty — run the IQ engine, LEAP alerts, or pin tickers manually below.</div>",
        unsafe_allow_html=True,
    )
else:
    @st.cache_data(ttl=300)
    def load_watchlist_data(tickers: tuple) -> dict:
        result = {}
        if not tickers:
            return result
        # One batched multi-ticker request instead of one download per ticker.
        batch = yf.download(list(tickers), period="35d", interval="1d",
                            group_by="ticker", auto_adjust=True, progress=False)
        for tkr in tickers:
            try:
                closes = batch[tkr]["Close"].dropna()
                if len(closes) < 2:
                    result[tkr] = {"price": None, "day_pct": None, "sparkline": []}
                    continue
                cur = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                day_pct = (cur / prev - 1) * 100
                sparkline = closes.iloc[-30:].tolist() if len(closes) >= 30 else closes.tolist()
                result[tkr] = {"price": cur, "day_pct": day_pct, "sparkline": sparkline}
            except Exception:
                result[tkr] = {"price": None, "day_pct": None, "sparkline": []}
        return result

    wl_data = load_watchlist_data(tuple(watchlist_all.keys()))

    def make_sparkline_svg(values: list) -> str:
        if len(values) < 2:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1
        w, h = 80, 28
        pts = " ".join(
            f"{int(i / (len(values) - 1) * w)},{int(h - (v - mn) / rng * h)}"
            for i, v in enumerate(values)
        )
        color = GREEN if values[-1] >= values[0] else RED
        return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
                f'style="width:{w}px;height:{h}px;">'
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>')

    wl_cols = st.columns(3)
    for i, (tkr, reason) in enumerate(watchlist_all.items()):
        col = wl_cols[i % 3]
        d = wl_data.get(tkr, {})
        price    = d.get("price")
        day_pct  = d.get("day_pct")
        sparksvg = make_sparkline_svg(d.get("sparkline", []))
        pct_str  = f"{'+' if (day_pct or 0) >= 0 else ''}{day_pct:.2f}%" if day_pct is not None else "—"
        pct_col  = GREEN if (day_pct or 0) >= 0 else RED

        reason_color_map = {
            "IQ Thesis": "#5b21b6", "LEAP Alert": ACCENT,
            "Momentum Radar": GREEN, "manual": TEXT_MID,
        }
        r_color = reason_color_map.get(reason, TEXT_MID)

        is_pinned = tkr in wl_pins
        with col:
            col.markdown(
                f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
                f"padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;"
                f"align-items:center;'>"
                f"<div>"
                f"<div style='font-size:0.88rem;font-weight:700;color:{TEXT_DARK};'>{tkr}</div>"
                f"<div style='font-size:0.72rem;margin-top:2px;'>"
                f"<span style='color:{r_color};font-weight:600;'>{reason}</span></div>"
                f"<div style='font-size:0.84rem;font-weight:700;color:{pct_col};margin-top:4px;'>"
                f"{'$' + f'{price:,.2f}' if price else '—'}  {pct_str}</div>"
                f"</div>"
                f"<div>{sparksvg}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if is_pinned:
                if col.button(f"✕ Unpin {tkr}", key=f"unpin_{tkr}"):
                    remove_watchlist_pin(tkr)
                    st.rerun()
            else:
                if col.button(f"📌 Pin {tkr}", key=f"pin_{tkr}"):
                    add_watchlist_pin(tkr, reason="manual")
                    st.rerun()

# ── Manual pin input ─────────────────────────────────────────
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
pin_c1, pin_c2 = st.columns([3, 1])
manual_pin = pin_c1.text_input(
    "Pin a ticker manually", placeholder="e.g. AMD",
    key="manual_pin_input", label_visibility="collapsed",
)
if pin_c2.button("📌 Pin", key="manual_pin_btn"):
    if manual_pin.strip():
        add_watchlist_pin(manual_pin.strip().upper(), reason="manual")
        st.rerun()
