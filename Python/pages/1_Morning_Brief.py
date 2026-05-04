# pages/1_Morning_Brief.py
# ============================================================
# Morning Brief — landing page for the stock dashboard.
#
# This is the first page the user sees each morning. It contains:
#   - PULSE button: AI-generated market briefing with web search
#   - IQ button: macro intelligence thesis engine
#   - Sector performance treemap with drill-down to company cards
#   - Universe news feed filtered by ticker or sector selection
#
# Momentum Radar and Smart Watchlist live on page 2 (Radar).
# Chart and signals live on pages 3 and 4.
# Portfolio and trades live on page 5.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from utils import inject_css, universe_manager_sidebar, card_header, section_label
from utils import ACCENT, BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG, HEADER_BLUE
from database import (
    initialize_database, get_universe, get_recent_alerts,
    save_iq_thesis, get_iq_theses, resolve_iq_thesis, get_all_iq_stock_names,
)
import anthropic as _anthropic

st.set_page_config(page_title="Morning Brief", page_icon="🌅", layout="wide")
initialize_database()
inject_css()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("## 🌅  MORNING BRIEF")
anthropic_key = st.sidebar.text_input(
    "Anthropic API Key", type="password",
    value=ANTHROPIC_API_KEY,
    placeholder="sk-ant-...",
    help="Required for PULSE, IQ Engine, and Weekly Digest",
)
st.sidebar.markdown("---")
universe_manager_sidebar()

st.markdown(
    f"<h1 style='margin-bottom:4px;'>🌅 Morning Brief</h1>"
    f"<div style='color:{TEXT_LIGHT};font-size:0.82rem;margin-bottom:20px;'>"
    f"Market intelligence · {date.today().strftime('%A, %B %d, %Y')}</div>",
    unsafe_allow_html=True,
)

UNIVERSE = get_universe()

# ── Session state ─────────────────────────────────────────────
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

# ============================================================
# SECTION 1 — PULSE + IQ BUTTONS
# ============================================================

PULSE_SYSTEM = (
    "You are a sharp senior market analyst writing a morning briefing for a professional trader. "
    "Be direct. Use actionable language. No bullet points inside paragraphs. No filler sentences. "
    "Write each section header in ALL CAPS followed by a colon and then the paragraph. "
    "The five section headers you must use exactly are: "
    "MARKET PULSE, MACRO EVENTS THIS WEEK, EARNINGS ON DECK, SECTOR NARRATIVES, WATCHLIST SPOTLIGHT."
)

IQ_SYSTEM = (
    "You are a macro intelligence analyst. Your job is to identify non-obvious second-order economic "
    "and geopolitical forces that will move specific sectors and stocks over the next 3-6 months. "
    "Be specific. Name actual companies. Show the logical chain of causation. No filler."
)


@st.cache_data(ttl=900, show_spinner=False)
def run_pulse(api_key: str, today_str: str) -> tuple[str, str]:
    client = _anthropic.Anthropic(api_key=api_key)
    universe_str = ", ".join(UNIVERSE)
    user_msg = (
        f"Today is {today_str}. Generate a morning market briefing covering these five sections.\n\n"
        f"Our watchlist universe is: {universe_str}\n\n"
        "MARKET PULSE: One paragraph on major index performance today, VIX level, where SPY sits "
        "relative to its 50 and 200 day moving averages, overall market tone.\n\n"
        "MACRO EVENTS THIS WEEK: Fed meetings, CPI, PCE, jobs reports, any major economic releases "
        "this week and what they mean for the market.\n\n"
        f"EARNINGS ON DECK: Any company from this universe reporting this week — {universe_str}. "
        "Expected move, analyst consensus, what to watch.\n\n"
        "SECTOR NARRATIVES: What is driving moves in sectors relevant to the universe. Oil, rates, "
        "dollar, geopolitical. Only what actually matters to these names.\n\n"
        "WATCHLIST SPOTLIGHT: Two or three names from the universe showing interesting setups, "
        "unusual volume, catalysts approaching, or technical levels being tested right now."
    )
    messages = [{"role": "user", "content": user_msg}]
    tools    = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2500,
            system=PULSE_SYSTEM, tools=tools, messages=messages,
        )
        max_loops = 8; loop_count = 0
        while response.stop_reason == "tool_use" and loop_count < max_loops:
            loop_count += 1
            tool_results = [{"type": "tool_result", "tool_use_id": b.id, "content": "Search completed."}
                            for b in response.content if b.type == "tool_use"]
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
            response = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=2500,
                system=PULSE_SYSTEM, tools=tools, messages=messages,
            )
    except Exception:
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2500,
            system=PULSE_SYSTEM, messages=[{"role": "user", "content": user_msg}],
        )
    text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    return text, datetime.now().strftime("%I:%M %p")


def parse_pulse_sections(text: str) -> dict[str, str]:
    section_keys = [
        "MARKET PULSE", "MACRO EVENTS THIS WEEK",
        "EARNINGS ON DECK", "SECTOR NARRATIVES", "WATCHLIST SPOTLIGHT",
    ]
    result = {}
    for i, key in enumerate(section_keys):
        start = text.find(key + ":")
        if start == -1:
            start = text.find(key)
        if start == -1:
            continue
        end = len(text)
        for next_key in section_keys[i + 1:]:
            pos = text.find(next_key + ":", start + 1)
            if pos == -1:
                pos = text.find(next_key, start + 1)
            if pos != -1:
                end = min(end, pos)
        body = text[start:end].replace(key + ":", "").replace(key, "").strip()
        result[key] = body
    return result if result else {"BRIEFING": text}


# ── Hero button styles ────────────────────────────────────────
# EKG SVG for animated heartbeat on PULSE button
EKG_SVG = """<svg viewBox="0 0 80 40" xmlns="http://www.w3.org/2000/svg"
  style="width:80px;height:40px;vertical-align:middle;margin-right:7px;display:inline-block;">
  <polyline
    points="0,20 20,20 28,20 33,4 38,36 43,20 60,20 80,20"
    fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <animate attributeName="stroke-dasharray"
      from="0,220" to="220,0"
      dur="2s" repeatCount="indefinite"/>
  </polyline>
</svg>"""

BRAIN_SVG = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"
  style="width:20px;height:20px;vertical-align:middle;margin-right:6px;fill:white;display:inline-block;">
  <path d="M13 3a4 4 0 0 1 3.95 3.4A3.5 3.5 0 0 1 19.5 10a3.5 3.5 0 0 1-1 2.43V17a2 2 0 0 1-2 2h-1v1a2 2 0 0 1-4 0v-1H10a2 2 0 0 1-2-2v-4.57A3.5 3.5 0 0 1 6.5 10a3.5 3.5 0 0 1 2.55-3.36A4 4 0 0 1 13 3z"/>
</svg>"""

st.markdown(f"""
<style>
  div[data-testid="stColumn"]:nth-child(2) button,
  div[data-testid="column"]:nth-child(2) button {{
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: none !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(29,78,216,0.35) !important;
  }}
  div[data-testid="stColumn"]:nth-child(4) button,
  div[data-testid="column"]:nth-child(4) button {{
    background: #5b21b6 !important;
    color: #ffffff !important;
    border: none !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(91,33,182,0.35) !important;
  }}
</style>
""", unsafe_allow_html=True)

btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns([1.5, 1.5, 0.5, 1.5, 1.5])
with btn_c2:
    pulse_clicked = st.button(" ", use_container_width=True, key="pulse_btn")
with btn_c4:
    iq_clicked = st.button(" ", use_container_width=True, key="iq_btn")

st.markdown(
    f"<div style='display:flex;justify-content:center;gap:200px;margin-top:-50px;"
    f"margin-bottom:32px;pointer-events:none;'>"
    f"<div style='display:flex;align-items:center;'>"
    f"{EKG_SVG}"
    f"<span style='color:#ffffff;font-size:0.95rem;font-weight:700;letter-spacing:0.08em;'>PULSE</span>"
    f"</div>"
    f"<div style='display:flex;align-items:center;'>"
    f"{BRAIN_SVG}"
    f"<span style='color:#ffffff;font-size:0.95rem;font-weight:700;letter-spacing:0.08em;'>IQ</span>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── PULSE output ─────────────────────────────────────────────
if pulse_clicked or st.session_state.get("pulse_ran"):
    if not anthropic_key:
        st.error("Enter your Anthropic API key in the sidebar to use PULSE.")
    else:
        if pulse_clicked:
            st.session_state["pulse_ran"] = True
            run_pulse.clear()
        with st.spinner("Generating morning briefing with web search…"):
            try:
                briefing_text, generated_at = run_pulse(anthropic_key, str(date.today()))
                sections = parse_pulse_sections(briefing_text)
                SECTION_ICONS = {
                    "MARKET PULSE": "📊", "MACRO EVENTS THIS WEEK": "📅",
                    "EARNINGS ON DECK": "📣", "SECTOR NARRATIVES": "🏭",
                    "WATCHLIST SPOTLIGHT": "🔦", "BRIEFING": "📰",
                }
                cols = st.columns(2)
                for i, (sec_name, body) in enumerate(sections.items()):
                    icon = SECTION_ICONS.get(sec_name, "📌")
                    with cols[i % 2]:
                        st.markdown(
                            f"<div style='background:#fff;border:1px solid {BORDER};"
                            f"border-radius:8px;overflow:hidden;margin-bottom:14px;'>"
                            f"{card_header(f'{icon}  {sec_name}')}"
                            f"<div style='padding:14px 16px;font-size:0.84rem;"
                            f"color:{TEXT_DARK};line-height:1.7;'>{body}</div></div>",
                            unsafe_allow_html=True,
                        )
                st.markdown(
                    f"<div style='text-align:right;margin-top:-6px;margin-bottom:8px;'>"
                    f"<span style='font-size:0.72rem;color:{TEXT_LIGHT};'>"
                    f"Generated at {generated_at} · refreshes every 15 min</span></div>",
                    unsafe_allow_html=True,
                )
                digest_c1, digest_c2, digest_c3 = st.columns([3, 1, 3])
                with digest_c2:
                    if st.button("📋  Weekly Report", use_container_width=True):
                        from paper_trades import generate_weekly_digest
                        with st.spinner("Writing weekly performance review…"):
                            try:
                                digest = generate_weekly_digest(anthropic_key)
                                st.markdown(
                                    f"<div style='background:#fff;border:1px solid {BORDER};"
                                    f"border-radius:8px;overflow:hidden;margin:12px 0;'>"
                                    f"{card_header('📋  Weekly Performance Digest')}"
                                    f"<div style='padding:16px;font-size:0.84rem;color:{TEXT_DARK};"
                                    f"line-height:1.7;'>{digest}</div></div>",
                                    unsafe_allow_html=True,
                                )
                            except Exception as e:
                                st.error(f"Digest error: {e}")
            except Exception as e:
                st.error(f"PULSE error: {e}")

# ── IQ Engine output ─────────────────────────────────────────
today_weekday = date.today().weekday()
is_iq_auto_day = today_weekday in (0, 4)  # Mon and Fri

if iq_clicked or st.session_state.get("iq_ran"):
    if not anthropic_key:
        st.error("Enter your Anthropic API key in the sidebar to use IQ Engine.")
    else:
        if iq_clicked:
            st.session_state["iq_ran"] = True

        existing_theses = get_iq_theses(status="active")

        iq_col1, iq_col2 = st.columns([6, 1])
        with iq_col2:
            force_iq = st.button("⟳ Refresh", key="iq_force_refresh")

        run_iq_now = force_iq or (iq_clicked and (is_iq_auto_day or not existing_theses))

        if run_iq_now:
            with st.spinner("Running macro intelligence scan…"):
                try:
                    import re
                    universe_str = ", ".join(UNIVERSE)
                    existing_titles = [t["title"] for t in existing_theses]
                    existing_block = (
                        "Previously identified theses (avoid duplicating these):\n" +
                        "\n".join(f"- {t}" for t in existing_titles)
                        if existing_titles else ""
                    )
                    user_msg = (
                        f"Today is {date.today().isoformat()}. Our stock universe: {universe_str}.\n\n"
                        f"{existing_block}\n\n"
                        "Identify 3 macro or geopolitical theses that are non-obvious and will materially "
                        "affect specific industries and stocks in the next 3-6 months.\n\n"
                        "For each thesis respond in this exact JSON format inside a ```json block:\n"
                        "[\n"
                        "  {\n"
                        '    "title": "Short thesis title (max 10 words)",\n'
                        '    "logical_chain": "Full causal chain — what happens, why, who benefits or suffers (3-5 sentences)",\n'
                        '    "affected_industries": "Industry 1, Industry 2",\n'
                        '    "stock_names": "TICKER1, TICKER2, TICKER3",\n'
                        '    "confidence": 70,\n'
                        '    "timeframe": "3-6 months"\n'
                        "  }\n"
                        "]\n\n"
                        "Use web search to ground your theses in current real-world events."
                    )
                    client = _anthropic.Anthropic(api_key=anthropic_key)
                    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}]
                    messages = [{"role": "user", "content": user_msg}]
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001", max_tokens=2000,
                        system=IQ_SYSTEM, tools=tools, messages=messages,
                    )
                    max_loops = 6; loops = 0
                    while response.stop_reason == "tool_use" and loops < max_loops:
                        loops += 1
                        tool_results = [{"type": "tool_result", "tool_use_id": b.id, "content": "Search completed."}
                                        for b in response.content if b.type == "tool_use"]
                        messages = messages + [
                            {"role": "assistant", "content": response.content},
                            {"role": "user", "content": tool_results},
                        ]
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001", max_tokens=2000,
                            system=IQ_SYSTEM, tools=tools, messages=messages,
                        )
                    raw_text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
                    json_match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_text)
                    if json_match:
                        theses_data = json.loads(json_match.group(1))
                        for td in theses_data:
                            save_iq_thesis(
                                title=td.get("title", "Untitled"),
                                logical_chain=td.get("logical_chain", ""),
                                affected_industries=td.get("affected_industries", ""),
                                stock_names=td.get("stock_names", ""),
                                confidence=int(td.get("confidence", 50)),
                                timeframe=td.get("timeframe", "3-6 months"),
                            )
                        st.rerun()
                    else:
                        st.warning("Could not parse thesis JSON from IQ response.")
                except Exception as e:
                    st.error(f"IQ Engine error: {e}")

        active_theses = get_iq_theses(status="active")
        if active_theses:
            section_label("Macro Intelligence — Active Theses")
            for thesis in active_theses:
                conf = thesis["confidence"]
                conf_color = GREEN if conf >= 70 else ("#f59e0b" if conf >= 50 else RED)
                with st.container():
                    st.markdown(
                        f"<div style='background:#fff;border:1px solid {BORDER};"
                        f"border-radius:8px;overflow:hidden;margin-bottom:12px;'>"
                        f"<div style='background:#5b21b6;padding:10px 16px;display:flex;"
                        f"justify-content:space-between;align-items:center;'>"
                        f"<span style='color:#fff;font-size:0.82rem;font-weight:700;"
                        f"letter-spacing:0.06em;'>{thesis['title']}</span>"
                        f"<span style='color:#e9d5ff;font-size:0.72rem;'>"
                        f"Confidence: <strong style='color:#fff;'>{conf}%</strong> · "
                        f"{thesis['timeframe']} · run {thesis['run_count']}×</span></div>"
                        f"<div style='padding:14px 16px;'>"
                        f"<p style='font-size:0.84rem;color:{TEXT_DARK};line-height:1.65;"
                        f"margin:0 0 10px 0;'>{thesis['logical_chain']}</p>"
                        f"<div style='display:flex;gap:12px;flex-wrap:wrap;font-size:0.75rem;'>"
                        f"<span style='color:{TEXT_MID};'><strong>Sectors:</strong> "
                        f"{thesis['affected_industries']}</span>"
                        f"<span style='color:{ACCENT};'><strong>Stocks:</strong> "
                        f"{thesis['stock_names']}</span></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    resolve_col1, resolve_col2 = st.columns([5, 1])
                    with resolve_col2:
                        if st.button("✓ Resolved", key=f"resolve_{thesis['id']}"):
                            st.session_state[f"resolving_{thesis['id']}"] = True
                    if st.session_state.get(f"resolving_{thesis['id']}"):
                        postmortem = st.text_area(
                            "Postmortem note (what happened, was the thesis right?)",
                            key=f"pm_{thesis['id']}", height=80,
                        )
                        if st.button("Confirm Resolve", key=f"confirm_resolve_{thesis['id']}"):
                            resolve_iq_thesis(thesis["id"], postmortem)
                            st.session_state.pop(f"resolving_{thesis['id']}", None)
                            st.rerun()

        resolved_theses = get_iq_theses(status="resolved")
        if resolved_theses:
            with st.expander(f"🪦  Thesis Graveyard — {len(resolved_theses)} resolved", expanded=False):
                for thesis in resolved_theses:
                    st.markdown(
                        f"<div style='background:#f8fafc;border:1px solid {BORDER};"
                        f"border-radius:7px;padding:12px 14px;margin-bottom:8px;'>"
                        f"<div style='font-size:0.82rem;font-weight:700;color:{TEXT_MID};"
                        f"text-decoration:line-through;margin-bottom:4px;'>{thesis['title']}</div>"
                        f"<div style='font-size:0.78rem;color:{TEXT_MID};line-height:1.5;"
                        f"margin-bottom:6px;'>{thesis['logical_chain']}</div>"
                        f"<div style='font-size:0.72rem;color:{TEXT_LIGHT};'>"
                        f"Created: {thesis['created_date'][:10]} · "
                        f"Resolved: {thesis['last_updated'][:10]} · "
                        f"Run {thesis['run_count']}× · Stocks: {thesis['stock_names']}</div>"
                        + (f"<div style='margin-top:6px;background:#fff;border-left:3px solid #5b21b6;"
                           f"padding:6px 10px;font-size:0.78rem;color:{TEXT_DARK};'>"
                           f"<strong>Postmortem:</strong> {thesis['postmortem']}</div>"
                           if thesis.get("postmortem") else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 2 — SECTOR TREEMAP + DRILL-DOWN
# ============================================================

SECTOR_ETFS = {
    "XLK":  {"name": "Technology",               "weight": 29},
    "XLC":  {"name": "Communication Services",   "weight": 9},
    "XLV":  {"name": "Health Care",              "weight": 13},
    "XLF":  {"name": "Financials",               "weight": 13},
    "XLY":  {"name": "Consumer Discretionary",   "weight": 11},
    "XLI":  {"name": "Industrials",              "weight": 9},
    "XLP":  {"name": "Consumer Staples",         "weight": 6},
    "XLE":  {"name": "Energy",                   "weight": 4},
    "XLB":  {"name": "Materials",                "weight": 2},
    "XLRE": {"name": "Real Estate",              "weight": 2},
    "XLU":  {"name": "Utilities",                "weight": 2},
}

SECTOR_STOCKS = {
    "Technology":               ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "AMD", "ADBE", "CSCO", "QCOM", "TXN"],
    "Communication Services":   ["META", "GOOGL", "NFLX", "DIS", "T", "VZ", "CMCSA", "EA", "TTWO", "WBD"],
    "Health Care":              ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN", "GILD"],
    "Financials":               ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW"],
    "Consumer Discretionary":   ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM"],
    "Industrials":              ["GE", "RTX", "CAT", "HON", "UNP", "BA", "DE", "LMT", "NOC", "MMM"],
    "Consumer Staples":         ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS"],
    "Energy":                   ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "PXD"],
    "Materials":                ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DD", "PPG", "ALB", "CF"],
    "Real Estate":              ["AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "SPG", "VICI"],
    "Utilities":                ["NEE", "SO", "DUK", "AEP", "SRE", "EXC", "D", "ED", "PCG", "ETR"],
}


@st.cache_data(ttl=600)
def load_nested_treemap_data() -> pd.DataFrame:
    all_tickers = [tkr for stocks in SECTOR_STOCKS.values() for tkr in stocks]
    rows = []
    for tkr in all_tickers:
        try:
            info = yf.Ticker(tkr).fast_info
            mkt_cap = getattr(info, "market_cap", None) or 1e9
            last = getattr(info, "last_price", None)
            prev = getattr(info, "previous_close", None)
            if last and prev and prev > 0:
                chg = (last / prev - 1) * 100
            else:
                chg = 0.0
        except Exception:
            mkt_cap = 1e9
            chg = 0.0
        rows.append({"ticker": tkr, "market_cap": mkt_cap, "change_pct": chg})

    sector_lookup = {
        tkr: sector
        for sector, stocks in SECTOR_STOCKS.items()
        for tkr in stocks
    }
    df = pd.DataFrame(rows)
    df["sector"] = df["ticker"].map(sector_lookup)
    df["market_cap"] = df["market_cap"].clip(lower=1e8)
    return df


@st.cache_data(ttl=600)
def load_stock_changes(tickers: tuple) -> dict:
    result = {}
    for tkr in tickers:
        try:
            df = yf.download(tkr, period="5d", interval="1d", auto_adjust=True, progress=False)
            if len(df) >= 2:
                closes = df["Close"].squeeze()
                result[tkr] = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1) * 100
            else:
                result[tkr] = 0.0
        except Exception:
            result[tkr] = 0.0
    return result


section_label("Sector Performance Treemap")
with st.spinner("Loading sector treemap…"):
    treemap_df = load_nested_treemap_data()

fig_tree = px.treemap(
    treemap_df,
    path=["sector", "ticker"],
    values="market_cap",
    color="change_pct",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    custom_data=["ticker", "change_pct"],
)
fig_tree.update_traces(
    texttemplate="<b>%{label}</b><br>%{customdata[1]:.2f}%",
    hovertemplate="<b>%{customdata[0]}</b><br>Daily change: %{customdata[1]:.2f}%<extra></extra>",
    textfont=dict(size=12),
)
fig_tree.update_layout(
    paper_bgcolor="#f0f4f8", margin=dict(l=0, r=0, t=10, b=0),
    height=440, coloraxis_showscale=False,
)

treemap_result = st.plotly_chart(
    fig_tree, use_container_width=True,
    on_select="rerun", key="sector_treemap",
    config={"displayModeBar": False},
)

st.markdown(
    f"<div style='font-size:0.75rem;color:{TEXT_LIGHT};margin-top:-8px;margin-bottom:12px;'>"
    f"Block size = market cap. Color = today's daily change. "
    f"Click a stock to filter the news feed below.</div>",
    unsafe_allow_html=True,
)

# ── Treemap click → set selected_sector or selected_ticker ───
if treemap_result and hasattr(treemap_result, "selection"):
    pts = treemap_result.selection.get("points", [])
    if pts:
        clicked_label = pts[0].get("label", "")
        if clicked_label in SECTOR_STOCKS:
            # Clicked a sector outer block
            if st.session_state.selected_sector == clicked_label:
                st.session_state.selected_sector = None
            else:
                st.session_state.selected_sector = clicked_label
                st.session_state.selected_ticker = None
        else:
            # Clicked an inner stock block — check if it's a known ticker
            all_tickers = [t for stocks in SECTOR_STOCKS.values() for t in stocks]
            if clicked_label in all_tickers:
                st.session_state.selected_ticker = (
                    None if st.session_state.selected_ticker == clicked_label else clicked_label
                )
                for sector, stocks in SECTOR_STOCKS.items():
                    if clicked_label in stocks:
                        st.session_state.selected_sector = sector
                        break

# ── Company cards grid ────────────────────────────────────────
if st.session_state.selected_sector and st.session_state.selected_sector in SECTOR_STOCKS:
    stocks = SECTOR_STOCKS[st.session_state.selected_sector]
    stock_changes = load_stock_changes(tuple(stocks))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    cards_hdr_col, cards_clear_col = st.columns([6, 1])
    with cards_hdr_col:
        section_label(f"Top 10 Holdings — {st.session_state.selected_sector}")
    with cards_clear_col:
        if st.button("✕ Clear", key="clear_sector_btn"):
            st.session_state.selected_sector = None
            st.session_state.selected_ticker = None
            st.rerun()

    for row_stocks in [stocks[:5], stocks[5:]]:
        card_cols = st.columns(5)
        for col, tkr in zip(card_cols, row_stocks):
            chg       = stock_changes.get(tkr, 0.0)
            chg_color = GREEN if chg >= 0 else RED
            sign_str  = ("+" if chg >= 0 else "") + f"{chg:.1f}%"
            is_active = st.session_state.selected_ticker == tkr
            border    = f"2px solid {ACCENT}" if is_active else f"1px solid {BORDER}"
            with col:
                if st.button(" ", key=f"stock_card_{tkr}", use_container_width=True):
                    st.session_state.selected_ticker = None if is_active else tkr
                    st.rerun()
                st.markdown(
                    f"<div style='margin-top:-42px;pointer-events:none;text-align:center;"
                    f"background:{CARD_BG};border:{border};border-radius:8px;padding:8px 4px;'>"
                    f"<div style='font-size:0.80rem;font-weight:700;color:{TEXT_DARK};'>{tkr}</div>"
                    f"<div style='font-size:0.75rem;font-weight:600;color:{chg_color};'>{sign_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 3 — UNIVERSE NEWS FEED
# ============================================================


@st.cache_data(ttl=600)
def fetch_all_news(tickers: tuple) -> list:
    all_items = []
    for tkr in tickers:
        try:
            items = yf.Ticker(tkr).news or []
            for item in items[:5]:
                c       = item.get("content", {})
                title   = c.get("title", "")
                summary = c.get("summary", "") or c.get("description", "")
                source  = c.get("provider", {}).get("displayName", "")
                url_obj = c.get("canonicalUrl", {})
                url     = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
                pub     = c.get("pubDate", "")[:10]
                if title:
                    all_items.append({
                        "ticker": tkr, "title": title, "summary": summary,
                        "source": source, "url": url, "pub": pub,
                    })
        except Exception:
            pass
    all_items.sort(key=lambda x: x["pub"], reverse=True)
    return all_items


all_news = fetch_all_news(tuple(UNIVERSE))

stock_filter = st.session_state.get("selected_ticker")

news_hdr_col, news_clear_col = st.columns([5, 1])
with news_hdr_col:
    if stock_filter:
        section_label(f"Latest News — {stock_filter}")
    else:
        section_label("Universe News Feed")
with news_clear_col:
    if stock_filter and st.button("✕ Clear filter", key="news_clear_filter"):
        st.session_state.selected_ticker = None
        st.rerun()

filter_options = ["ALL"] + sorted(set(n["ticker"] for n in all_news))
radio_default  = filter_options.index(stock_filter) if stock_filter in filter_options else 0
active_filter  = st.radio(
    "Filter by ticker", filter_options,
    index=radio_default, horizontal=True,
    label_visibility="collapsed", key="news_filter",
)

filtered_news = all_news if active_filter == "ALL" else [n for n in all_news if n["ticker"] == active_filter]

news_left, news_right = st.columns(2)
for i, item in enumerate(filtered_news[:20]):
    col = news_left if i % 2 == 0 else news_right
    with col:
        tag  = (f"<span style='background:{ACCENT};color:#fff;border-radius:4px;"
                f"padding:2px 7px;font-size:0.68rem;font-weight:700;margin-right:6px;'>"
                f"{item['ticker']}</span>")
        link = (f"<a href='{item['url']}' target='_blank' style='color:{TEXT_DARK};"
                f"text-decoration:none;font-weight:600;font-size:0.82rem;'>{item['title']}</a>"
                if item["url"]
                else f"<span style='color:{TEXT_DARK};font-weight:600;font-size:0.82rem;'>{item['title']}</span>")
        meta = (f"<span style='color:{TEXT_LIGHT};font-size:0.72rem;'>"
                f"{item['source']}{'  ·  ' + item['pub'] if item['pub'] else ''}</span>")
        blurb = (f"<div style='color:{TEXT_MID};font-size:0.78rem;margin-top:2px;'>"
                 f"{item['summary'][:120]}{'…' if len(item['summary']) > 120 else ''}</div>"
                 if item["summary"] else "")
        st.markdown(
            f"<div style='background:#fff;border:1px solid #e8edf5;border-radius:7px;"
            f"padding:10px 14px;margin-bottom:7px;'>{tag}{link}<br>{meta}{blurb}</div>",
            unsafe_allow_html=True,
        )

if not filtered_news:
    st.markdown(
        f"<span style='color:{TEXT_LIGHT};font-size:0.82rem;'>"
        f"No headlines found for {active_filter}.</span>",
        unsafe_allow_html=True,
    )
