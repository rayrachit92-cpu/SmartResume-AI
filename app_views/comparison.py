"""
SmartResume AI - Candidate Comparison Page
Side-by-side comparative analysis of multiple candidate profiles:
Skill overlap matrices, unique competencies, and objective credential benchmarking.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from database.db_manager import db
from modules.candidate_comparison import CandidateComparator
from ui.styles import get_plotly_theme


def render_comparison():
    """Renders the transparent candidate comparison view."""
    theme_cfg = get_plotly_theme()
    st.markdown('<div class="brand-badge">TALENT ANALYTICS • SIDE-BY-SIDE</div>', unsafe_allow_html=True)
    st.title("Candidate Profile Comparison")
    st.caption("Objective, side-by-side credential and skill comparison across multiple candidate profiles.")

    st.info(
        "⚖️ **Ethical Evaluation Policy:** SmartResume AI displays factual skill overlaps and credential summaries. "
        "It deliberately avoids automated 'hire / no-hire' ranking or algorithmic bias."
    )

    resumes = db.get_all_resumes()
    if len(resumes) < 2:
        st.warning("Comparison requires at least 2 uploaded and analyzed resumes.")
        return

    # Multi-select candidates
    options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_labels = st.multiselect(
        "Select Candidates to Compare (2 or more):",
        options=list(options.keys()),
        default=list(options.keys())[:min(3, len(options))]
    )

    if len(selected_labels) < 2:
        st.info("Please select at least 2 candidates above.")
        return

    # Fetch candidate profile dicts
    candidates = []
    for label in selected_labels:
        r_id = options[label]
        prof = db.get_candidate_profile(r_id)
        if prof:
            prof["resume_id"] = r_id
            candidates.append(prof)

    # Run comparison
    comparison = CandidateComparator.compare_candidates(candidates)
    if comparison.get("status") != "success":
        st.warning(comparison.get("message"))
        return

    # Summary Table
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("Credential & Trajectory Overview")
    st.dataframe(comparison["summary_df"], width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Shared vs Unique Skills
    col_common, col_unique = st.columns(2)

    with col_common:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("🤝 Shared Skills (Overlap)")
        common = comparison["common_skills"]
        if common:
            pills = " ".join([
                f'<span style="background-color: #ECFDF5; color: #047857; padding: 4px 10px; '
                f'border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin: 3px; display: inline-block; border: 1px solid #A7F3D0;">'
                f'{s}</span>'
                for s in common
            ])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No common skills shared across ALL selected candidates.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_unique:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("⭐ Differentiating / Unique Skills")
        for cname, u_skills in comparison["unique_skills"].items():
            st.markdown(f"**{cname}:**")
            if u_skills:
                u_pills = " ".join([
                    f'<span style="background-color: #EFF6FF; color: #1D4ED8; padding: 3px 8px; '
                    f'border-radius: 10px; font-size: 0.78rem; margin: 2px; display: inline-block; border: 1px solid #BFDBFE;">'
                    f'{s}</span>'
                    for s in u_skills[:10]
                ])
                st.markdown(u_pills, unsafe_allow_html=True)
            else:
                st.caption("No mutually exclusive skills.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Full Skill Coverage Matrix
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("Full Competency Matrix")
    st.caption("Presence ('✓') or absence ('—') of skills across the candidate pool.")
    st.dataframe(comparison["skill_matrix_df"], width="stretch", height=360, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
