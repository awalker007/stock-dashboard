# pages/1_Command_Center.py
# ============================================================
# Command Center — morning briefing, macro IQ, sector treemap,
# momentum radar, smart watchlist, news feed, paper trading.
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from utils import inject_css, universe_manager_sidebar, card_header, section_label
from utils import ACCENT, BORDER, TEXT_DARK, TEXT_MID, TEXT_LIGHT, GREEN, RED, CARD_BG, HEADER_BLUE
from database import (
    initialize_database, get_universe,
    get_recent_alerts, get_open_trades, get_closed_trades,
    add_paper_trade, close_paper_trade,
    get_recent_alerts_for_digest, get_recent_trades_for_digest,
    save_iq_thesis, get_iq_theses, resolve_iq_thesis, get_all_iq_stock_names,
    get_watchlist_pins, add_watchlist_pin, remove_watchlist_pin,
)
from indicators import calculate_rsi
import anthropic as _anthropic

st.set_page_config(page_title="Command Center", page_icon="🎯", layout="wide")
initialize_database()
inject_css()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("## 🎯  COMMAND CENTER")
anthropic_key = st.sidebar.text_input(
    "Anthropic API Key", type="password",
    value=ANTHROPIC_API_KEY,
    placeholder="sk-ant-...",
    help="Required for PULSE, IQ Engine, and Weekly Digest",
)
st.sidebar.markdown("---")
universe_manager_sidebar()

st.markdown(
    f"<h1 style='margin-bottom:4px;'>🎯 Command Center</h1>"
    f"<div style='color:{TEXT_LIGHT};font-size:0.82rem;margin-bottom:20px;'>"
    f"Morning briefing · {date.today().strftime('%A, %B %d, %Y')}</div>",
    unsafe_allow_html=True,
)

UNIVERSE = get_universe()

# ── Session state initialization ──────────────────────────────
# st.session_state is a dictionary that survives Streamlit re-runs.
# Every time any widget is touched, Streamlit re-executes this entire
# script from top to bottom. Without session_state, any selection the
# user made would reset to None on the next re-run. We initialise both
# keys here so they always exist before any widget reads them.
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None   # name of the active sector button
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None   # ticker of the active company card

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


# ── Hero buttons row ─────────────────────────────────────────
#
# HOW THE PULSE BUTTON WORKS:
# Streamlit's st.button() only accepts plain text labels — you cannot inject
# HTML or SVG directly into it. To show an animated heartbeat icon ON the
# button we use a two-layer trick:
#   1. A real st.button() handles click detection (invisible label text).
#   2. An HTML/SVG element is rendered on top via st.markdown(), overlapping
#      the button area using CSS negative margin. It has pointer-events:none
#      so clicks fall through to the button underneath.
#
# WHAT SVG IS:
# SVG (Scalable Vector Graphics) is an XML-based image format built into
# every modern browser. Instead of pixels it uses geometric commands like
# polyline (a series of connected straight-line segments). The image scales
# perfectly to any size without blurring.
#
# HOW THE ANIMATION WORKS:
# stroke-dasharray controls the repeating pattern of visible/invisible
# segments along a line. Setting it to "0,200" makes the entire line
# invisible (0px drawn, 200px gap). Setting it to "200,0" makes the line
# fully visible. The <animate> element smoothly transitions between these
# two states over 1.8 seconds and loops forever — this creates the effect
# of the EKG line drawing itself from left to right continuously.

# EKG path: flat baseline → sharp spike up (near top) → down past baseline
# → return to center → flat baseline to the end — the classic heartbeat shape.
# The viewBox "0 0 80 40" defines the coordinate space (80 wide, 40 tall).
# The style width/height set the actual pixel size rendered on screen.
# Points trace: start at left-middle (0,20), flat to 25%, sharp spike to near
# top (33,4), plunge past baseline (38,36), recover to middle (43,20), flat to end.
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
  /* ── PULSE hero button (column 2) ──────────────────────────────────────
     Target both old testid ("column") and new Streamlit testid ("stColumn").
     Same for the button element: old="stButton", new="stBaseButton".      */
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
  /* ── IQ hero button (column 4) ── */
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
    # Label is a non-breaking space — height is preserved but text is invisible.
    # The white SVG overlay below is what the user actually sees.
    pulse_clicked = st.button("\u00a0", use_container_width=True, key="pulse_btn")
with btn_c4:
    iq_clicked = st.button("\u00a0", use_container_width=True, key="iq_btn")

# SVG overlay — rendered with negative top margin so it sits visually ON the buttons.
# pointer-events:none means clicks pass straight through to the st.button() below.
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
today_weekday = date.today().weekday()  # Monday=0, Friday=4
is_iq_auto_day = today_weekday in (0, 4)  # Mon and Fri

if iq_clicked or st.session_state.get("iq_ran"):
    if not anthropic_key:
        st.error("Enter your Anthropic API key in the sidebar to use IQ Engine.")
    else:
        if iq_clicked:
            st.session_state["iq_ran"] = True

        existing_theses = get_iq_theses(status="active")

        # Force refresh button
        iq_col1, iq_col2 = st.columns([6, 1])
        with iq_col2:
            force_iq = st.button("⟳ Refresh", key="iq_force_refresh")

        run_iq_now = force_iq or (iq_clicked and (is_iq_auto_day or not existing_theses))

        if run_iq_now:
            with st.spinner("Running macro intelligence scan…"):
                try:
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
                    # Extract JSON block
                    import re
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

        # Render active theses cards
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

        # Thesis Graveyard
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
def load_sector_data() -> pd.DataFrame:
    rows = []
    for etf, meta in SECTOR_ETFS.items():
        try:
            df = yf.download(etf, period="5d", interval="1d", auto_adjust=True, progress=False)
            if len(df) >= 2:
                closes = df["Close"].squeeze()
                chg = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1) * 100
            else:
                chg = 0.0
            rows.append({"etf": etf, "name": meta["name"], "weight": meta["weight"], "change_pct": chg})
        except Exception:
            rows.append({"etf": etf, "name": meta["name"], "weight": meta["weight"], "change_pct": 0.0})
    return pd.DataFrame(rows)


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
sector_df = load_sector_data()

fig_tree = px.treemap(
    sector_df, path=["name"], values="weight",
    color="change_pct", color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0, custom_data=["etf", "change_pct"],
)
fig_tree.update_traces(
    texttemplate="<b>%{label}</b><br>%{customdata[1]:.2f}%",
    hovertemplate="<b>%{label}</b> (%{customdata[0]})<br>Daily change: %{customdata[1]:.2f}%<extra></extra>",
    textfont=dict(size=14),
)
fig_tree.update_layout(
    paper_bgcolor="#f0f4f8", margin=dict(l=0, r=0, t=10, b=0),
    height=300, coloraxis_showscale=False,
)

treemap_result = st.plotly_chart(
    fig_tree, use_container_width=True,
    on_select="rerun", key="sector_treemap",
    config={"displayModeBar": False},
)

st.markdown(
    f"<div style='font-size:0.75rem;color:{TEXT_LIGHT};margin-top:-8px;margin-bottom:12px;'>"
    f"Block size = approximate S&P 500 sector weight. Color = today's daily performance. "
    f"Click a sector block to expand its top 10 holdings.</div>",
    unsafe_allow_html=True,
)

# ── Treemap click → set selected_sector ──────────────────────
if treemap_result and hasattr(treemap_result, "selection"):
    pts = treemap_result.selection.get("points", [])
    if pts:
        clicked_label = pts[0].get("label", "")
        if clicked_label in SECTOR_STOCKS:
            if st.session_state.selected_sector == clicked_label:
                st.session_state.selected_sector = None
            else:
                st.session_state.selected_sector = clicked_label
                st.session_state.selected_ticker = None

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
                if st.button("\u00a0", key=f"stock_card_{tkr}", use_container_width=True):
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
# SECTION 3 — MOMENTUM RADAR
# ============================================================


@st.cache_data(ttl=86400, show_spinner=False)
def run_momentum_radar(tickers: tuple) -> list[dict]:
    """
    Scan universe for stocks meeting all 5 momentum criteria.
    Cached 24h — one expensive scan per day.
    """
    flagged = []
    today = date.today()
    start_6m = today - timedelta(days=182)
    start_1y = today - timedelta(days=400)

    for tkr in tickers:
        try:
            df = yf.download(tkr, start=str(start_1y), end=str(today),
                             auto_adjust=True, progress=False)
            if len(df) < 130:
                continue
            df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)
            close  = df["Close"].squeeze()
            volume = df["Volume"].squeeze()

            # Criterion 1: 20-day lows are higher now vs 3 months ago
            recent_20d_low  = float(close.iloc[-20:].min())
            earlier_20d_low = float(close.iloc[-83:-63].min()) if len(close) > 83 else None
            if earlier_20d_low is None or recent_20d_low <= earlier_20d_low:
                continue

            # Criterion 2: 20D avg vol ≥ 120% of 3-6mo avg vol
            vol_20d = float(volume.iloc[-20:].mean())
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
            rsi_s = calculate_rsi(close, 14)
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
        # Auto-add to watchlist
        add_watchlist_pin(hit["ticker"], reason="Momentum Radar")

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 4 — SMART WATCHLIST
# ============================================================

section_label("Smart Watchlist")

# Build watchlist: pins + IQ stocks + recent alerts + radar flags
wl_pins = {p["ticker"]: p["reason"] for p in get_watchlist_pins()}
iq_stocks = get_all_iq_stock_names()
recent_alert_tickers = list(dict.fromkeys(
    a["ticker"] for a in get_recent_alerts(limit=30)
    if datetime.strptime(a["timestamp"][:10], "%Y-%m-%d") >= datetime.now() - timedelta(days=5)
))

watchlist_all: dict[str, str] = {}  # ticker → reason
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
        for tkr in tickers:
            try:
                df = yf.download(tkr, period="35d", interval="1d",
                                 auto_adjust=True, progress=False)
                if len(df) < 2:
                    continue
                closes = df["Close"].squeeze()
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

# Manual pin input
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
pin_c1, pin_c2 = st.columns([3, 1])
manual_pin = pin_c1.text_input("Pin a ticker manually", placeholder="e.g. AMD", key="manual_pin_input", label_visibility="collapsed")
if pin_c2.button("📌 Pin", key="manual_pin_btn"):
    if manual_pin.strip():
        add_watchlist_pin(manual_pin.strip().upper(), reason="manual")
        st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 5 — UNIVERSE NEWS FEED
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

# Active stock-card filter — set by clicking a company card in the sector drill-down
stock_filter = st.session_state.get("selected_ticker")

# Header row with active filter tag + clear
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

# Ticker pill filter (radio)
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
    st.markdown(f"<span style='color:{TEXT_LIGHT};font-size:0.82rem;'>No headlines found for {active_filter}.</span>", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>", unsafe_allow_html=True)

# ============================================================
# SECTION 6 — PAPER TRADING TRACKER
# ============================================================

section_label("Paper Trading Tracker")

open_trades   = get_open_trades()
closed_trades = get_closed_trades()

total = len(open_trades) + len(closed_trades)
closed_pnls = []
for t in closed_trades:
    if t["exit_price"] and t["entry_price"]:
        pnl = (t["exit_price"] - t["entry_price"]) * t["shares_or_contracts"]
        if t["trade_type"] == "LEAP":
            pnl *= 100
        closed_pnls.append(pnl)

win_rate       = (sum(1 for p in closed_pnls if p > 0) / len(closed_pnls) * 100) if closed_pnls else 0
total_realized = sum(closed_pnls)

sm1, sm2, sm3, sm4, sm5 = st.columns(5)
sm1.metric("Total Trades",   total)
sm2.metric("Open",           len(open_trades))
sm3.metric("Closed",         len(closed_trades))
sm4.metric("Win Rate",       f"{win_rate:.1f}%")
sm5.metric("Total Realized", f"${total_realized:+,.0f}")

pt_tab1, pt_tab2, pt_tab3 = st.tabs(["📋  Open Positions", "📜  Trade History", "➕  Log Trade"])

with pt_tab1:
    if not open_trades:
        st.markdown(f"<span style='color:{TEXT_LIGHT};font-size:0.85rem;'>No open trades. Use the Log Trade tab to add one.</span>", unsafe_allow_html=True)
    else:
        open_tickers = list(set(t["ticker"] for t in open_trades))
        current_prices = {}
        for tkr in open_tickers:
            try:
                hist = yf.download(tkr, period="2d", interval="1d", auto_adjust=True, progress=False)
                current_prices[tkr] = float(hist["Close"].squeeze().iloc[-1])
            except Exception:
                current_prices[tkr] = None

        th_style = f"padding:9px 14px;font-size:0.72rem;font-weight:700;color:#fff;background:{TEXT_DARK};"
        header_row = (
            f"<tr>"
            f"<th style='{th_style}text-align:left;'>Ticker</th>"
            f"<th style='{th_style}text-align:left;'>Type</th>"
            f"<th style='{th_style}'>Entry Date</th>"
            f"<th style='{th_style}'>Entry $</th>"
            f"<th style='{th_style}'>Current $</th>"
            f"<th style='{th_style}'>Unrealized P&L</th>"
            f"<th style='{th_style}'>P&L %</th>"
            f"</tr>"
        )
        td = "padding:9px 14px;font-size:0.82rem;border-bottom:1px solid #e8edf5;"
        rows_html = ""
        for i, trade in enumerate(open_trades):
            bg  = "#f8fafc" if i % 2 == 0 else "#ffffff"
            cur = current_prices.get(trade["ticker"])
            if cur:
                qty     = trade["shares_or_contracts"]
                mult    = 100 if trade["trade_type"] == "LEAP" else 1
                pnl_d   = (cur - trade["entry_price"]) * qty * mult
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
                f"<td style='{td}color:{TEXT_MID};'>{trade['trade_type']}</td>"
                f"<td style='{td}text-align:center;color:{TEXT_MID};'>{trade['entry_date']}</td>"
                f"<td style='{td}text-align:right;color:{TEXT_DARK};font-weight:600;'>${trade['entry_price']:,.2f}</td>"
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
        with st.expander("Close a Position"):
            trade_options = {f"#{t['id']} {t['ticker']} {t['trade_type']} @ ${t['entry_price']:.2f}": t["id"] for t in open_trades}
            sel_label = st.selectbox("Select trade to close", list(trade_options.keys()), key="close_trade_sel")
            exit_price_val = st.number_input("Exit price ($)", min_value=0.01, value=1.0, step=0.01, key="exit_price")
            exit_date_val  = st.date_input("Exit date", value=date.today(), key="exit_date")
            if st.button("Confirm Close", type="primary", key="confirm_close_btn"):
                close_paper_trade(trade_options[sel_label], exit_price_val, str(exit_date_val))
                st.success("Trade closed.")
                st.rerun()

with pt_tab2:
    if not closed_trades:
        st.markdown(f"<span style='color:{TEXT_LIGHT};font-size:0.85rem;'>No closed trades yet.</span>", unsafe_allow_html=True)
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
            pnl  = (trade["exit_price"] - trade["entry_price"]) * trade["shares_or_contracts"] * mult if trade["exit_price"] else None
            pnl_col = GREEN if (pnl or 0) >= 0 else RED
            hist_rows += (
                f"<tr style='background:{bg};'>"
                f"<td style='{td2}font-weight:700;color:{TEXT_DARK};'>{trade['ticker']}</td>"
                f"<td style='{td2}color:{TEXT_MID};'>{trade['trade_type']}</td>"
                f"<td style='{td2}text-align:center;color:{TEXT_MID};'>{trade['entry_date']}</td>"
                f"<td style='{td2}text-align:center;color:{TEXT_MID};'>{trade['exit_date'] or '—'}</td>"
                f"<td style='{td2}text-align:right;font-weight:600;color:{TEXT_DARK};'>${trade['entry_price']:,.2f}</td>"
                f"<td style='{td2}text-align:right;font-weight:600;color:{TEXT_DARK};'>${trade['exit_price']:,.2f}</td>"
                f"<td style='{td2}text-align:right;font-weight:700;color:{pnl_col};'>${pnl:+,.0f}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<div style='background:#fff;border:1px solid {BORDER};border-radius:8px;"
            f"overflow:hidden;margin-bottom:16px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead>{hist_header}</thead><tbody>{hist_rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )
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
                yaxis=dict(gridcolor="#e8edf5", tickfont=dict(color="#64748b", size=10), tickprefix="$"),
                showlegend=False,
            )
            section_label("Cumulative Realised P&L")
            st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

with pt_tab3:
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
        notes_val      = st.text_area("Notes (optional)", height=60, key="nt_notes",
                                       placeholder="Why this trade? What signal triggered it?")
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
# SECTION 7 — ALERT LOG
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
