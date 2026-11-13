"""SmartResume AI application entrypoint and workspace navigation."""

import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="SmartResume AI", page_icon=":material/description:", layout="wide", initial_sidebar_state="expanded")

from ui.styles import CUSTOM_CSS
from database.db_manager import db
from modules.genai_assistant import GenAIAssistant
from app_views.ai_assistant import render_ai_assistant
from app_views.analyzer import render_analyzer
from app_views.comparison import render_comparison
from app_views.dashboard import render_dashboard
from app_views.entity_linking import render_entity_linking
from app_views.evaluation import render_evaluation
from app_views.extraction import render_extraction
from app_views.ner_view import render_ner
from app_views.reports import render_reports
from app_views.sentiment import render_sentiment
from app_views.settings_view import render_settings
from app_views.topics import render_topics
from app_views.upload import render_upload

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    metrics = db.get_kpi_metrics()
    assistant = GenAIAssistant()
    llm_status = "Verified" if st.session_state.get("gemini_connection_verified") else ("Configured" if assistant.is_configured() else "Offline")
    st.markdown("""<div class="sidebar-brand"><div class="sidebar-brand-mark">SR</div><div><strong>SmartResume AI</strong><span>NLP intelligence workspace</span></div></div>""", unsafe_allow_html=True)
    st.caption(f"{metrics['total_resumes']} uploaded resume(s) · Gemini {llm_status}")

overview_page = st.Page(render_dashboard, title="Overview", icon=":material/dashboard:", default=True)
upload_page = st.Page(render_upload, title="Upload resume", icon=":material/upload_file:")
st.session_state["upload_page"] = upload_page

page = st.navigation(
    {
        "WORKSPACE": [
            overview_page,
            upload_page,
            st.Page(render_analyzer, title="Resume analyzer", icon=":material/manage_search:"),
        ],
        "NLP ANALYSIS": [
            st.Page(render_ner, title="Named entity recognition", icon=":material/category:"),
            st.Page(render_extraction, title="Information extraction", icon=":material/account_tree:"),
            st.Page(render_entity_linking, title="Entity linking", icon=":material/hub:"),
            st.Page(render_sentiment, title="Sentiment analysis", icon=":material/psychology:"),
            st.Page(render_topics, title="Topic modeling", icon=":material/donut_large:"),
        ],
        "INTELLIGENCE": [
            st.Page(render_ai_assistant, title="AI resume assistant", icon=":material/auto_awesome:"),
            st.Page(render_comparison, title="Candidate comparison", icon=":material/compare:"),
        ],
        "MANAGEMENT": [
            st.Page(render_evaluation, title="Dataset & model evaluation", icon=":material/analytics:"),
            st.Page(render_reports, title="Reports", icon=":material/description:"),
            st.Page(render_settings, title="Settings", icon=":material/settings:"),
        ],
    },
    position="sidebar",
)
page.run()
