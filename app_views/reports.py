"""
SmartResume AI - Reports & Downloads Page
Provides one-click individual candidate and batch exports in
Structured JSON, Tabular CSV, and Executive TXT dossier formats.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from database.db_manager import db
from modules.report_generator import ReportGenerator


def render_reports():
    """Renders the comprehensive report generation and export page."""
    st.markdown('<div class="brand-badge">EXPORT & REPORTING</div>', unsafe_allow_html=True)
    st.title("Reports & Data Export")
    st.caption("Generate structured exports of extracted candidate profiles, canonical skill linkages, and batch analytics.")

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload and analyze a resume to generate a report.")
        return

    tab1, tab2 = st.tabs(["📄 Individual Candidate Dossier", "📦 Batch Corpus Export"])

    # Tab 1: Individual
    with tab1:
        st.subheader("Individual Candidate Export")
        resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
        selected_label = st.selectbox("Select Candidate:", list(resume_options.keys()))
        selected_id = resume_options[selected_label]

        profile = db.get_candidate_profile(selected_id)
        if not profile:
            st.warning("Candidate profile not found.")
            return

        # Preview Summary Card
        st.markdown(f"""
        <div class="saas-card">
            <h3 style="margin: 0; color: #1E40AF;">{profile.get('candidate_name') or 'Candidate Profile'}</h3>
            <p style="color: #64748B; margin: 4px 0 0 0;">
                ✉️ {profile.get('email') or 'N/A'} &nbsp;|&nbsp; 
                📞 {profile.get('phone') or 'N/A'} &nbsp;|&nbsp; 
                📍 {profile.get('location') or 'N/A'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_d1, col_d2, col_d3 = st.columns(3)

        # JSON Download
        with col_d1:
            st.markdown('<div class="saas-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("#### Structured JSON")
            st.caption("Complete candidate profile with schema, entities, and skills.")
            json_payload = ReportGenerator.generate_json_export(profile, include_entities=True)
            st.download_button(
                "⬇️ Download JSON",
                data=json_payload,
                file_name=f"SmartResume_Profile_RES-{selected_id:03d}.json",
                mime="application/json",
                width="stretch"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # TXT Dossier
        with col_d2:
            st.markdown('<div class="saas-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("#### Executive TXT Dossier")
            st.caption("Human-readable formatted executive candidate brief.")
            txt_payload = ReportGenerator.generate_txt_brief(profile)
            st.download_button(
                "⬇️ Download TXT Dossier",
                data=txt_payload,
                file_name=f"SmartResume_Dossier_RES-{selected_id:03d}.txt",
                mime="text/plain",
                width="stretch"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # CSV Export
        with col_d3:
            st.markdown('<div class="saas-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("#### Single-Row CSV")
            st.caption("Tabular row for HRIS and ATS database import.")
            csv_payload = ReportGenerator.generate_csv_export([profile])
            st.download_button(
                "⬇️ Download CSV",
                data=csv_payload,
                file_name=f"SmartResume_Record_RES-{selected_id:03d}.csv",
                mime="text/csv",
                width="stretch"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("👁️ Preview Executive TXT Dossier"):
            st.text(txt_payload)

    # Tab 2: Batch Export
    with tab2:
        st.subheader("Batch Corpus Export")
        st.caption(f"Export all `{len(resumes)}` processed resumes currently in the database.")

        # Collect all profiles
        all_profiles = []
        for r in resumes:
            prof = db.get_candidate_profile(r["id"])
            if prof:
                all_profiles.append(prof)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Master Candidate Database (CSV)")
            st.caption("All extracted candidates in a single normalized CSV spreadsheet.")
            batch_csv = ReportGenerator.generate_csv_export(all_profiles)
            st.download_button(
                f"⬇️ Download Master CSV ({len(all_profiles)} Candidates)",
                data=batch_csv,
                file_name=f"SmartResume_Master_Database_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b2:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Combined JSON Archive")
            st.caption("Complete JSON collection of candidate records and telemetry.")
            batch_json = json.dumps({"candidates": all_profiles, "exported_at": datetime.now().isoformat()}, indent=2)
            st.download_button(
                f"⬇️ Download Full JSON Archive",
                data=batch_json,
                file_name=f"SmartResume_Full_Archive_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                width="stretch"
            )
            st.markdown('</div>', unsafe_allow_html=True)
