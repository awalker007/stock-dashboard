# pages/5_Portfolio.py
# ============================================================
# Portfolio — paper trading tracker and alert log.
#
# This page is the performance accountability layer of the app.
# Everything you need to track how your paper trades and LEAP
# positions are performing lives here:
#
#   - Paper trades tracker with three tabs:
#       Open Positions: live unrealised P&L for stock trades
#       LEAP Positions: open LEAP contracts with unrealised P&L
#       Trade History: all closed trades with realised P&L
#       Log Trade: form to record a new paper trade
#   - Performance metrics: total trades, win rate, total realised P&L
#   - Equity curve: cumulative realised P&L chart over time
#   - Alert log: last 20 LEAP Setup alerts from alerts.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime

from utils import inject_css, universe_manager_sidebar, card_header, section_label
from utils import ACCENT, BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG
from database import (
    initialize_database, get_universe,
    get_open_trades, get_closed_trades,
    add_paper_trade, close_paper_trade,
    get_recent_alerts,
)

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
initialize_database()
inject_css()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("## 💼  PORTFOLIO")
st.sidebar.markdown("---")
universe_manager_sidebar()

st.markdown(
    f"<h1 style='margin-bottom:4px;'>💼 Portfolio</h1>"
    f"<div style='color:{TEXT_LIGHT};font-size:0.82rem;margin-bottom:20px;'>"
    f"Paper trades · LEAP positions · alert log · "
    f"{date.today().strftime('%A, %B %d, %Y')}</div>",
    unsafe_allow_html=True,
)

UNIVERSE = get_universe()

# ============================================================
# SECTION 1 — PAPER TRADING TRACKER
# ============================================================

section_label("Paper Trading Tracker")

open_trades   = get_open_trades()
closed_trades = get_closed_trades()

total = len(open_trades) + len(closed_trades)

# Split by type for display
open_stocks = [t for t in open_trades if t["trade_type"] == "Stock"]
open_leaps  = [t for t in open_trades if t["trade_type"] == "LEAP"]

closed_pnls = []
for t in closed_trades:
    if t["exit_price"] and t["entry_price"]:
        pnl = (t["exit_price"] - t["entry_price"]) * t["shares_or_contracts"]
        if t["trade_type"] == "LEAP":
            pnl *= 100
        closed_pnls.append(pnl)

win_rate       = (sum(1 for p in closed_pnls if p > 0) / len(closed_pnls) * 100) if closed_pnls else 0
total_realized = sum(closed_pnls)

sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
sm1.metric("Total Trades",   total)
sm2.metric("Open Stocks",    len(open_stocks))
sm3.metric("Open LEAPs",     len(open_leaps))
sm4.metric("Closed",         len(closed_trades))
sm5.metric("Win Rate",       f"{win_rate:.1f}%")
sm6.metric("Total Realized", f"${total_realized:+,.0f}")

pt_tab1, pt_tab2, pt_tab3, pt_tab4 = st.tabs([
    "📋  Open Positions",
    "🦘  LEAP Positions",
    "📜  Trade History",
    "➕  Log Trade",
])

# ── Fetch current prices for all open tickers once ───────────
@st.cache_data(ttl=120)
def fetch_current_prices(tickers: tuple) -> dict:
    result = {}
    if not tickers:
        return result
    # One batched multi-ticker request instead of one download per ticker.
    batch = yf.download(list(tickers), period="2d", interval="1d",
                        group_by="ticker", auto_adjust=True, progress=False)
    for tkr in tickers:
        try:
            closes = batch[tkr]["Close"].dropna()
            result[tkr] = float(closes.iloc[-1]) if len(closes) else None
        except Exception:
            result[tkr] = None
    return result


all_open_tickers = tuple(set(t["ticker"] for t in open_trades))
current_prices   = fetch_current_prices(all_open_tickers) if all_open_tickers else {}

# ── Tab 1: Open Stock Positions ───────────────────────────────
with pt_tab1:
    if not open_stocks:
        st.markdown(
            f"<span style='color:{TEXT_LIGHT};font-size:0.85rem;'>"
            f"No open stock positions. Use the Log Trade tab to add one.</span>",
            unsafe_allow_html=True,
        )
    else:
        th_style = f"padding:9px 14px;font-size:0.72rem;font-weight:700;color:#fff;background:{TEXT_DARK};"
        header_row = (
            f"<tr>"
            f"<th style='{th_style}text-align:left;'>Ticker</th>"
            f"<th style='{th_style}'>Entry Date</th>"
            f"<th style='{th_style}'>Entry $</th>"
            f"<th style='{th_style}'>Shares</th>"
            f"<th style='{th_style}'>Current $</th>"
            f"<th style='{th_style}'>Unrealized P&L</th>"
            f"<th style='{th_style}'>P&L %</th>"
            f"</tr>"
        )
        td = "padding:9px 14px;font-size:0.82rem;border-bottom:1px solid #e8edf5;"
        rows_html = ""
        for i, trade in enumerate(open_stocks):
            bg  = "#f8fafc" if i % 2 == 0 else "#ffffff"
            cur = current_prices.get(trade["ticker"])
            if cur:
                pnl_d   = (cur - trade["entry_price"]) * trade["shares_or_contracts"]
                pnl_pct = (cur / trade["entry_price"] - 1) * 100
                pnl_col = GREEN if pnl_d >= 0 else RED
                cur_str = f"${cur:,.2f}"
                pnl_str = f"${pnl_d:+,.0f}"
                pct_str = f"{pnl_pct:+.1f}%"
            else:
                pnl_col = TEXT_LIGHT
                cur_str = "—"; pnl_str = "—"; pct_str = "—"
            rows_html += (
                f"<tr style='background:{bg};'>"
                f"<td style='{td}font-weight:700;color:{TEXT_DARK};'>{trade['ticker']}</td>"
                f"<td style='{td}text-align:center;color:{TEXT_MID};'>{trade['entry_date']}</td>"
                f"<td style='{td}text-align:right;color:{TEXT_DARK};font-weight:600;'>${trade['entry_price']:,.2f}</td>"
                f"<td style='{td}text-align:right;color:{TEXT_MID};'>{trade['shares_or_contracts']}</td>"
                f"<td style='{td}text-align:right;color:{TEXT_DARK};'>{cur_str}</td>"
                f"<td style='{td}text-align:right;font-weight:700;color:{pnl_col};'>{pnl_str}</td>"
                f"<td style='{td}text-align:right;font-weight:700;color:{pnl_col};'>{pct_str}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
            f"overflow:hidden;margin-bottom:12px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead>{header_row}</thead><tbody>{rows_html}</tbody></table></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Close a Stock Position"):
            stock_options = {
                f"#{t['id']} {t['ticker']} @ ${t['entry_price']:.2f}": t["id"]
                for t in open_stocks
            }
            sel_label = st.selectbox("Select trade to close", list(stock_options.keys()), key="close_stock_sel")
            exit_price_val = st.number_input("Exit price ($)", min_value=0.01, value=1.0, step=0.01, key="exit_price_stock")
            exit_date_val  = st.date_input("Exit date", value=date.today(), key="exit_date_stock")
            if st.button("Confirm Close", type="primary", key="confirm_close_stock_btn"):
                close_paper_trade(stock_options[sel_label], exit_price_val, str(exit_date_val))
                st.success("Trade closed.")
                st.rerun()

# ── Tab 2: LEAP Positions ─────────────────────────────────────
with pt_tab2:
    if not open_leaps:
        st.markdown(
            f"<span style='color:{TEXT_LIGHT};font-size:0.85rem;'>"
            f"No open LEAP positions. Use the Log Trade tab to add one.</span>",
            unsafe_allow_html=True,
        )
    else:
        th_style_l = f"padding:9px 14px;font-size:0.72rem;font-weight:700;color:#fff;background:{TEXT_DARK};"
        leap_header = (
            f"<tr>"
            f"<th style='{th_style_l}text-align:left;'>Ticker</th>"
            f"<th style='{th_style_l}'>Entry Date</th>"
            f"<th style='{th_style_l}'>Strike $</th>"
            f"<th style='{th_style_l}'>Expiration</th>"
            f"<th style='{th_style_l}'>Contracts</th>"
            f"<th style='{th_style_l}'>Entry Premium</th>"
            f"<th style='{th_style_l}'>Stock Now $</th>"
            f"<th style='{th_style_l}'>Notes</th>"
            f"</tr>"
        )
        tdl = "padding:9px 14px;font-size:0.82rem;border-bottom:1px solid #e8edf5;"
        leap_rows = ""
        for i, trade in enumerate(open_leaps):
            bg  = "#f8fafc" if i % 2 == 0 else "#ffffff"
            cur = current_prices.get(trade["ticker"])
            cur_str = f"${cur:,.2f}" if cur else "—"
            exp_str = trade.get("expiration_date") or "—"
            strike  = trade.get("strike_price")
            strike_str = f"${strike:,.2f}" if strike else "—"

            # Highlight ITM vs OTM status
            itm_label = ""
            if cur and strike:
                if cur > strike:
                    itm_label = f" <span style='background:{GREEN};color:#fff;border-radius:3px;padding:1px 5px;font-size:0.65rem;'>ITM</span>"
                else:
                    itm_label = f" <span style='background:{RED};color:#fff;border-radius:3px;padding:1px 5px;font-size:0.65rem;'>OTM</span>"

            leap_rows += (
                f"<tr style='background:{bg};'>"
                f"<td style='{tdl}font-weight:700;color:{TEXT_DARK};'>{trade['ticker']}{itm_label}</td>"
                f"<td style='{tdl}text-align:center;color:{TEXT_MID};'>{trade['entry_date']}</td>"
                f"<td style='{tdl}text-align:right;color:{TEXT_DARK};font-weight:600;'>{strike_str}</td>"
                f"<td style='{tdl}text-align:center;color:{TEXT_MID};'>{exp_str}</td>"
                f"<td style='{tdl}text-align:right;color:{TEXT_MID};'>{trade['shares_or_contracts']}</td>"
                f"<td style='{tdl}text-align:right;color:{TEXT_DARK};font-weight:600;'>${trade['entry_price']:,.2f}/sh</td>"
                f"<td style='{tdl}text-align:right;color:{TEXT_DARK};'>{cur_str}</td>"
                f"<td style='{tdl}color:{TEXT_LIGHT};font-size:0.75rem;'>{(trade.get('notes') or '')[:60]}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
            f"overflow:hidden;margin-bottom:12px;'>"
            f"{card_header('🦘  Open LEAP Contracts')}"
            f"<div style='overflow-x:auto;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead>{leap_header}</thead><tbody>{leap_rows}</tbody></table></div></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Close a LEAP Position"):
            leap_options = {
                f"#{t['id']} {t['ticker']} ${t.get('strike_price','?')} exp {t.get('expiration_date','?')}": t["id"]
                for t in open_leaps
            }
            sel_leap = st.selectbox("Select LEAP to close", list(leap_options.keys()), key="close_leap_sel")
            exit_px_leap  = st.number_input("Exit premium per share ($)", min_value=0.01, value=1.0, step=0.01, key="exit_price_leap")
            exit_dt_leap  = st.date_input("Exit date", value=date.today(), key="exit_date_leap")
            if st.button("Confirm Close LEAP", type="primary", key="confirm_close_leap_btn"):
                close_paper_trade(leap_options[sel_leap], exit_px_leap, str(exit_dt_leap))
                st.success("LEAP position closed.")
                st.rerun()

# ── Tab 3: Trade History ──────────────────────────────────────
with pt_tab3:
    if not closed_trades:
        st.markdown(
            f"<span style='color:{TEXT_LIGHT};font-size:0.85rem;'>No closed trades yet.</span>",
            unsafe_allow_html=True,
        )
    else:
        th_style2 = f"padding:9px 14px;font-size:0.72rem;font-weight:700;color:#fff;background:{TEXT_DARK};"
        td2 = "padding:9px 14px;font-size:0.82rem;border-bottom:1px solid #e8edf5;"
        hist_header = (
            f"<tr>"
            f"<th style='{th_style2}text-align:left;'>Ticker</th>"
            f"<th style='{th_style2}text-align:left;'>Type</th>"
            f"<th style='{th_style2}'>Entry</th><th style='{th_style2}'>Exit</th>"
            f"<th style='{th_style2}'>Entry $</th><th style='{th_style2}'>Exit $</th>"
            f"<th style='{th_style2}'>Realized P&L</th>"
            f"</tr>"
        )
        hist_rows = ""
        for i, trade in enumerate(closed_trades):
            bg   = "#f8fafc" if i % 2 == 0 else "#ffffff"
            mult = 100 if trade["trade_type"] == "LEAP" else 1
            pnl  = (
                (trade["exit_price"] - trade["entry_price"]) * trade["shares_or_contracts"] * mult
                if trade["exit_price"] else None
            )
            pnl_col = GREEN if (pnl or 0) >= 0 else RED
            hist_rows += (
                f"<tr style='background:{bg};'>"
                f"<td style='{td2}font-weight:700;color:{TEXT_DARK};'>{trade['ticker']}</td>"
                f"<td style='{td2}color:{TEXT_MID};'>{trade['trade_type']}</td>"
                f"<td style='{td2}text-align:center;color:{TEXT_MID};'>{trade['entry_date']}</td>"
                f"<td style='{td2}text-align:center;color:{TEXT_MID};'>{trade['exit_date'] or '—'}</td>"
                f"<td style='{td2}text-align:right;font-weight:600;color:{TEXT_DARK};'>${trade['entry_price']:,.2f}</td>"
                f"<td style='{td2}text-align:right;font-weight:600;color:{TEXT_DARK};'>${trade['exit_price']:,.2f}</td>"
                f"<td style='{td2}text-align:right;font-weight:700;color:{pnl_col};'>"
                f"{'$' + f'{pnl:+,.0f}' if pnl is not None else '—'}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
            f"overflow:hidden;margin-bottom:16px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead>{hist_header}</thead><tbody>{hist_rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )

        # Equity curve — shown when there are 2+ closed trades with P&L
        if len(closed_pnls) > 1:
            equity_vals = []
            running = 0
            for trade in sorted(closed_trades, key=lambda t: t["exit_date"] or ""):
                if trade["exit_price"] and trade["exit_date"]:
                    mult = 100 if trade["trade_type"] == "LEAP" else 1
                    running += (trade["exit_price"] - trade["entry_price"]) * trade["shares_or_contracts"] * mult
                    equity_vals.append({"date": trade["exit_date"], "pnl": running})
            eq_df = pd.DataFrame(equity_vals)
            fig_eq = go.Figure(go.Scatter(
                x=eq_df["date"], y=eq_df["pnl"], mode="lines", fill="tozeroy",
                line=dict(color=ACCENT, width=2), fillcolor="rgba(29,78,216,0.10)",
            ))
            fig_eq.update_layout(
                paper_bgcolor="#f0f4f8", plot_bgcolor=CARD_BG,
                height=200, margin=dict(l=50, r=20, t=20, b=40),
                xaxis=dict(gridcolor="#e8edf5", tickfont=dict(color="#64748b", size=10)),
                yaxis=dict(
                    gridcolor="#e8edf5",
                    tickfont=dict(color="#64748b", size=10),
                    tickprefix="$",
                ),
                showlegend=False,
            )
            section_label("Cumulative Realised P&L — Equity Curve")
            st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

# ── Tab 4: Log Trade ──────────────────────────────────────────
with pt_tab4:
    with st.form("new_trade_form"):
        fc1, fc2 = st.columns(2)
        trade_ticker = fc1.selectbox("Ticker", UNIVERSE, key="nt_ticker")
        trade_type   = fc2.radio("Type", ["Stock", "LEAP"], horizontal=True, key="nt_type")
        fc3, fc4 = st.columns(2)
        entry_price = fc3.number_input("Entry price ($)", min_value=0.01, value=100.0, step=0.01, key="nt_entry")
        qty         = fc4.number_input("Shares / Contracts", min_value=1, value=1, step=1, key="nt_qty")
        strike = None; expiry = None
        if trade_type == "LEAP":
            fc5, fc6 = st.columns(2)
            strike = fc5.number_input("Strike price ($)", min_value=0.01, value=100.0, step=0.01, key="nt_strike")
            expiry = str(fc6.date_input("Expiration date", key="nt_expiry"))
        entry_date_val = st.date_input("Entry date", value=date.today(), key="nt_date")
        notes_val      = st.text_area(
            "Notes (optional)", height=60, key="nt_notes",
            placeholder="Why this trade? What signal triggered it?",
        )
        if st.form_submit_button("📌  Log Trade", type="primary"):
            add_paper_trade(
                ticker=trade_ticker, trade_type=trade_type, entry_price=entry_price,
                shares_or_contracts=qty, entry_date=str(entry_date_val),
                notes=notes_val, strike_price=strike, expiration_date=expiry,
            )
            st.success(f"Trade logged: {qty} × {trade_ticker} {trade_type} @ ${entry_price:.2f}")
            st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 2 — LEAP ALERT LOG
# ============================================================

section_label("LEAP Alert Log — last 20 alerts")
recent_alerts = get_recent_alerts(limit=20)

if not recent_alerts:
    st.markdown(
        f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
        f"padding:16px;font-size:0.84rem;color:{TEXT_LIGHT};'>"
        f"No alerts fired yet. Run <code>python alerts.py</code> on a schedule "
        f"to start receiving LEAP Setup alerts.</div>",
        unsafe_allow_html=True,
    )
else:
    al1, al2 = st.columns(2)
    for i, alert in enumerate(recent_alerts):
        col = al1 if i % 2 == 0 else al2
        col.markdown(
            f"<div style='background:#fff;border:1px solid {BORDER};border-radius:7px;"
            f"padding:10px 14px;margin-bottom:7px;'>"
            f"<span style='background:{ACCENT};color:#fff;border-radius:4px;"
            f"padding:2px 7px;font-size:0.68rem;font-weight:700;'>{alert['ticker']}</span>"
            f"&nbsp;&nbsp;<span style='font-size:0.72rem;color:{TEXT_LIGHT};'>{alert['timestamp']}</span>"
            f"<div style='font-size:0.8rem;color:{TEXT_DARK};margin-top:4px;'>"
            f"{alert['conditions_triggered']}</div>"
            f"<div style='font-size:0.78rem;color:{TEXT_MID};'>Price at alert: "
            f"<strong>${alert['price_at_alert']:,.2f}</strong></div></div>",
            unsafe_allow_html=True,
        )
