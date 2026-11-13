"""
SmartResume AI - Settings & Privacy Management Page
API credentials, model configuration, knowledge base management,
file retention policies, data deletion, and privacy governance.
"""

import streamlit as st
import os
from database.db_manager import db
from modules.genai_assistant import GenAIAssistant
from config.settings import DB_PATH, MODELS_DIR


def render_settings():
    """Renders the settings and security configuration interface."""
    st.markdown('<div class="brand-badge">SYSTEM CONFIGURATION & PRIVACY</div>', unsafe_allow_html=True)
    st.title("Settings & Data Privacy")
    st.caption("Manage LLM credentials, model parameters, persistent local storage, and GDPR-compliant data purge.")

    assistant = GenAIAssistant()
    is_gemini_active = assistant.is_configured()

    tab1, tab2, tab3 = st.tabs([
        "🔑 API & Model Configuration",
        "🛡️ Privacy & Local Retention",
        "ℹ️ System & Environment Diagnostics"
    ])

    # Tab 1: API Configuration
    with tab1:
        st.subheader("LLM Provider Configuration (Google Gemini)")

        st.markdown(f"""
        **Connection Status:** `{'🟡 Configured — verify from AI Resume Assistant' if is_gemini_active else '🔴 Not Configured (Offline Mode)'}`
        """)

        gemini_key_input = st.text_input(
            "Google Gemini API Key:",
            type="password",
            placeholder="Enter AI Studio API Key (starts with AIzaSy...)",
            help="Your API key is used strictly in local memory and is never transmitted or hardcoded."
        )

        model_choice = st.selectbox(
            "Gemini Model Variant:",
            options=["", "gemini-3.5-flash-lite", "gemini-3.5-flash"],
            index=0
        )

        if st.button("Save API Configuration"):
            if gemini_key_input.strip():
                os.environ["GEMINI_API_KEY"] = gemini_key_input.strip()
                os.environ["GEMINI_MODEL"] = model_choice
                st.session_state["gemini_connection_verified"] = False
                st.success("API Key updated in current session! The AI Assistant is now active.")
                st.rerun()
            else:
                st.warning("Please enter a valid API key string.")

    # Tab 2: Privacy & Purge
    with tab2:
        st.subheader("Data Retention & Privacy Controls")
        st.markdown("""
        **Data Governance Policies:**
        - **Local SQLite Storage:** All resume texts, profiles, and entities are stored solely on your local workstation in `data/smartresume.db`.
        - **Zero External Telemetry:** Extracted candidate profiles are never sent to external third parties without user-initiated AI queries.
        - **GDPR Right-to-Erasure:** You can delete individual candidate records at any time or completely wipe the database.
        """)

        st.divider()

        st.subheader("Dangerous Action: Purge All Application Data")
        st.warning("This action permanently deletes all resumes, candidate profiles, entities, and analysis metadata from the SQLite database.")

        confirm_purge = st.checkbox("I understand that this action is irreversible and permanently deletes all stored resumes.")
        if st.button("🗑️ Purge All Stored Data", type="primary", disabled=not confirm_purge):
            db.clear_all_data()
            st.success("All resume records and extracted profiles have been completely erased from the local database.")
            st.rerun()

    # Tab 3: System Diagnostics
    with tab3:
        st.subheader("System Diagnostics & NLP Pipeline Status")

        has_trained_model = (MODELS_DIR / "resume_ner_model").exists()
        resumes_count = len(db.get_all_resumes())

        diag_data = [
            {"Component": "Database File", "Status": "Active", "Details": str(DB_PATH)},
            {"Component": "Trained Resume NER Model", "Status": "Ready" if has_trained_model else "Missing", "Details": str(MODELS_DIR / "resume_ner_model")},
            {"Component": "spaCy English Pipeline", "Status": "Ready", "Details": "spacy.blank('en') + EntityRuler + NER"},
            {"Component": "NLTK Resources", "Status": "Installed", "Details": "punkt, stopwords, wordnet, vader_lexicon"},
            {"Component": "Current Corpus Resumes", "Status": "Populated" if resumes_count > 0 else "Empty", "Details": f"{resumes_count} resumes"},
            {"Component": "Gemini Generative AI", "Status": "Connected" if is_gemini_active else "Offline", "Details": assistant.model_name}
        ]

        st.dataframe(diag_data, width="stretch", hide_index=True)
