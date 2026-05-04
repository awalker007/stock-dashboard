# app.py
# ============================================================
# Entry point for the 5-page Stock Dashboard application.
#
# Page structure:
#   1. Morning Brief  — PULSE, IQ, sector treemap, news feed
#   2. Radar          — Momentum Radar, Smart Watchlist
#   3. Chart Terminal — interactive price chart, fundamentals
#   4. Signal Scanner — condition builder, historical backtesting
#   5. Portfolio      — paper trades, LEAP tracker, alert log
#
# Streamlit discovers pages/ automatically and builds the sidebar.
# This file bootstraps the database and redirects to page 1.
#
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
from database import initialize_database

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure all SQLite tables exist before any page tries to read them.
initialize_database()

# Redirect immediately to Morning Brief so the root URL is not blank.
st.switch_page("pages/1_Morning_Brief.py")
