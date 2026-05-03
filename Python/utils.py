# utils.py
# ============================================================
# Shared helpers imported by every page.
# Keeps CSS, theme constants, and small UI helpers in one place
# so changes propagate everywhere automatically.
# ============================================================

import streamlit as st

# ── Theme constants ─────────────────────────────────────────
# These match every color already used in the existing dashboard.
# Import them in any file so you never type a hex code twice.
PAGE_BG      = "#f0f4f8"   # light grey-blue page background
SIDEBAR_BG   = "#2596be"   # the signature blue sidebar
CARD_BG      = "#ffffff"   # white card background
BORDER       = "#dce4ef"   # card border colour
ACCENT       = "#1d4ed8"   # primary blue for buttons and links
TEXT_DARK    = "#0f2044"   # nearly black heading text
TEXT_MID     = "#475569"   # secondary body text
TEXT_LIGHT   = "#94a3b8"   # muted caption text
GREEN        = "#059669"   # positive / bullish
RED          = "#dc2626"   # negative / bearish
HEADER_BLUE  = "#2596be"   # card header bar (same as sidebar)


def inject_css() -> None:
    """
    Inject the shared CSS that styles every page consistently.

    Call this once at the top of every page file, right after
    st.set_page_config.  Streamlit re-injects it on every rerun,
    which is fine — the browser deduplicates identical style blocks.
    """
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {PAGE_BG}; }}
        .block-container {{ padding-top: 1rem !important; padding-bottom: 0rem; }}
        footer {{ visibility: hidden; }}
        #MainMenu {{ visibility: hidden; }}

        /* ── Sidebar ──────────────────────────────────────────────────────────────
           CONTRAST FIX: The sidebar background is #2596be (medium blue). Without
           explicit overrides, Streamlit renders input fields, dropdowns, and number
           inputs with white or near-white text ON a white or near-white background
           — completely invisible. The rules below fix this in two layers:
             1. All plain text (headings, markdown, labels) stays white so it reads
                against the blue background.
             2. All interactive elements (inputs, selects, number fields) get a
                white (#ffffff) background with dark (#0f2044) text so the user
                can actually see what they typed or selected.
        ──────────────────────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; border-right: 1px solid #1a7fa0; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.25) !important; }}

        /* Layer 1 — all static text and labels default to white so they show on the blue bg */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{ color: #ffffff !important; }}

        /* Widget labels sit directly on the blue sidebar — slightly softened white */
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stNumberInput > label,
        [data-testid="stSidebar"] .stTextInput > label,
        [data-testid="stSidebar"] .stToggle > label,
        [data-testid="stSidebar"] .stCheckbox > label,
        [data-testid="stSidebar"] .stRadio > label {{ color: #d0eef8 !important; font-size: 0.75rem !important; }}

        /* Checkbox and radio option text */
        [data-testid="stSidebar"] .stCheckbox span,
        [data-testid="stSidebar"] .stRadio span {{ color: #ffffff !important; }}

        /* Toggle labels */
        [data-testid="stSidebar"] .stToggle label {{ color: #ffffff !important; }}

        /* Layer 2 — interactive elements: white bg, dark text so user can read their input */
        [data-testid="stSidebar"] input {{
            color: {TEXT_DARK} !important;
            background-color: #ffffff !important;
            border-color: rgba(255,255,255,0.5) !important;
        }}
        [data-testid="stSidebar"] input::placeholder {{ color: #7bafd4 !important; }}

        /* Selectbox / dropdown — white background, dark selected-value text */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: #ffffff !important;
            border-color: rgba(255,255,255,0.5) !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div {{ color: {TEXT_DARK} !important; }}

        /* Dropdown option list (Streamlit renders this in a portal outside the sidebar element) */
        [data-baseweb="popover"] [role="option"] {{ color: {TEXT_DARK} !important; }}
        [data-baseweb="popover"] [role="option"]:hover {{ background: #e0f0ff !important; }}

        /* Number input wrapper — white background */
        [data-testid="stSidebar"] [data-baseweb="input"] {{
            background-color: #ffffff !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="input"] input {{ color: {TEXT_DARK} !important; }}

        /* Color picker hex field */
        [data-testid="stSidebar"] [data-testid="stColorPicker"] input {{ color: {TEXT_DARK} !important; }}

        /* Multiselect tags and text area — white bg, dark text */
        [data-testid="stSidebar"] [data-baseweb="tag"] {{ background: rgba(255,255,255,0.25) !important; }}
        [data-testid="stSidebar"] textarea {{
            color: {TEXT_DARK} !important;
            background-color: #ffffff !important;
        }}

        /* Expander header — slightly lighter blue pill so it reads on the sidebar */
        [data-testid="stSidebar"] .streamlit-expanderHeader {{
            color: #ffffff !important;
            background: rgba(255,255,255,0.15) !important;
            border-radius: 6px !important;
        }}

        /* Toggle track background */
        [data-testid="stSidebar"] [data-testid="stToggle"] > div {{ background-color: rgba(255,255,255,0.25) !important; }}

        /* ── Metric cards ── */
        [data-testid="stMetric"] {{ background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(15,32,68,0.08); }}
        [data-testid="stMetricLabel"] {{ color:#64748b !important; font-size:0.72rem !important; letter-spacing:0.08em; text-transform:uppercase; }}
        [data-testid="stMetricValue"] {{ color:{TEXT_DARK} !important; font-size:1.3rem !important; font-weight:700; }}
        [data-testid="stMetricDelta"] svg {{ display:none; }}

        /* ── Range radio → styled like pill buttons ── */
        div[data-testid="stRadio"] > div {{ display:flex; flex-wrap:nowrap; gap:5px; margin-bottom:6px; }}
        div[data-testid="stRadio"] label {{
            background: #ffffff;
            border: 1.5px solid {BORDER};
            border-radius: 20px;
            padding: 4px 13px;
            font-size: 0.78rem;
            font-weight: 600;
            color: {TEXT_MID};
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }}
        div[data-testid="stRadio"] label:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
        div[data-testid="stRadio"] label:has(input:checked) {{
            background: {ACCENT};
            color: #ffffff !important;
            border-color: {ACCENT};
        }}
        div[data-testid="stRadio"] label input {{ display:none; }}

        .stPlotlyChart {{ border:1px solid {BORDER}; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(15,32,68,0.08); }}
        h1 {{ color:{TEXT_DARK} !important; font-size:1.4rem !important; font-weight:700; }}
        h2, h3 {{ color:{TEXT_MID} !important; font-size:0.85rem !important; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; }}

        /* ── Tab styling ── */
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 2px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px 8px 0 0;
            border: 1.5px solid transparent;
            padding: 8px 20px;
            font-weight: 600;
            font-size: 0.83rem;
            color: {TEXT_MID};
        }}
        .stTabs [aria-selected="true"] {{
            background: {CARD_BG} !important;
            border-color: {BORDER} !important;
            border-bottom-color: {CARD_BG} !important;
            color: {TEXT_DARK} !important;
        }}
    </style>
    """, unsafe_allow_html=True)


def card_header(title: str, bg: str = HEADER_BLUE) -> str:
    """
    Return an HTML string for a styled card header bar.

    Args:
        title (str): The text to display in the header.
        bg (str): Background colour hex. Defaults to the signature blue.

    Returns:
        str: HTML fragment — pass to st.markdown(..., unsafe_allow_html=True).
    """
    return (
        f"<div style='padding:10px 16px;background:{bg};border-radius:8px 8px 0 0;'>"
        f"<span style='color:#fff;font-size:0.78rem;font-weight:700;"
        f"letter-spacing:0.08em;text-transform:uppercase;'>{title}</span></div>"
    )


def section_label(text: str) -> None:
    """
    Render a small uppercase section label in the muted secondary text colour.

    Args:
        text (str): The label text, e.g. "Latest News — AAPL".
    """
    st.markdown(
        f"<div style='font-size:0.78rem;font-weight:700;letter-spacing:0.08em;"
        f"text-transform:uppercase;color:{TEXT_MID};margin-bottom:10px;'>{text}</div>",
        unsafe_allow_html=True,
    )


def universe_manager_sidebar(label: str = "🌐 Universe Manager") -> None:
    """
    Render the Universe Manager expander in the sidebar.

    This lets the user add/remove tickers and pull suggestions from Finviz.
    It reads and writes to the SQLite universe table via database.py.
    Changes take effect on the next page rerun — no restart required.

    Args:
        label (str): The expander label shown in the sidebar.
    """
    from database import get_universe, add_ticker, remove_ticker, initialize_database

    initialize_database()

    with st.sidebar.expander(label, expanded=False):
        universe = get_universe()

        # ── Current universe tags ──────────────────────────
        tags_html = " ".join(
            f"<span style='background:rgba(255,255,255,0.2);border-radius:4px;"
            f"padding:2px 7px;font-size:0.72rem;font-weight:700;'>{t}</span>"
            for t in universe
        )
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Add ticker manually ────────────────────────────
        new_tkr = st.text_input("Add ticker", placeholder="e.g. AMD", key="um_add_input")
        if st.button("＋ Add", key="um_add_btn"):
            if new_tkr.strip():
                added = add_ticker(new_tkr.strip().upper(), source="manual")
                if added:
                    st.success(f"{new_tkr.upper()} added.")
                    st.rerun()
                else:
                    st.warning(f"{new_tkr.upper()} already in universe.")

        # ── Remove tickers ─────────────────────────────────
        st.markdown("**Remove:**")
        to_remove = st.selectbox("Select ticker to remove", ["—"] + universe, key="um_remove_sel")
        if st.button("✕ Remove", key="um_remove_btn"):
            if to_remove != "—":
                remove_ticker(to_remove)
                st.success(f"{to_remove} removed.")
                st.rerun()

        # ── Finviz screener ────────────────────────────────
        if st.button("🔍 Refresh from Finviz", key="um_finviz_btn"):
            try:
                # finviz.Screener filters:
                # cap_largeunder = Large cap and above (>$10B)
                # sh_avgvol_o1000 = Average volume over 1 million
                from finviz.screener import Screener
                filters = ["cap_largeover", "sh_avgvol_o1000"]
                screen  = Screener(filters=filters, table="Overview", order="marketcap")
                added_count = 0
                for row in list(screen)[:30]:
                    tkr = row.get("Ticker", "")
                    if tkr and add_ticker(tkr, source="finviz"):
                        added_count += 1
                st.success(f"Added {added_count} new tickers from Finviz.")
                st.rerun()
            except Exception as e:
                st.error(f"Finviz error: {e}")
