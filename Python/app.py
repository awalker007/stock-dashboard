# app.py
# ============================================================
# Entry point for the multi-page Streamlit application.
#
# Streamlit automatically discovers pages in the pages/ folder
# and builds the sidebar navigation from them. This file just
# sets the global page config and immediately redirects the user
# to the Command Center page so the root URL is not blank.
#
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
from database import initialize_database

# ── Page configuration ───────────────────────────────────────
# layout="wide" means the content fills the full browser width.
# This must be the very first Streamlit call in the file.
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database bootstrap ───────────────────────────────────────
# Make sure all SQLite tables exist before any page tries to use them.
initialize_database()

# ── Redirect to Command Center ───────────────────────────────
# st.switch_page sends the user straight to the first real page.
# Without this the entry point would show an empty white screen.
st.switch_page("pages/1_Command_Center.py")
