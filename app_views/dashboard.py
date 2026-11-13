"""
SmartResume AI - Overview Dashboard Page
Live analytics, KPIs, entity distributions, and recent resume activity.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from database.db_manager import db
from ui.styles import get_plotly_theme


def render_dashboard():
    """Renders the primary executive SaaS analytics dashboard."""
    theme_cfg = get_plotly_theme()

    # Top Header
    st.markdown('<div class="brand-badge">RESUME INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Overview")
    st.caption(f"Resume analysis activity · {datetime.now().strftime('%A, %B %d, %Y')}")

    all_resumes = db.get_all_resumes()
    if not all_resumes:
        st.markdown("### Start with a resume")
        st.write("Upload a PDF or DOCX resume to extract text, identify entities, build a structured profile, and generate reports from the actual document.")
        if st.button("Upload a resume", type="primary", icon=":material/upload_file:"):
            st.switch_page(st.session_state["upload_page"])
        st.divider()
        c1, c2, c3 = st.columns(3)
        for column, icon, title, detail in (
            (c1, ":material/category:", "Named entity recognition", "Detect skills, education, roles, organizations, and contact details."),
            (c2, ":material/account_tree:", "Information extraction", "Create a structured candidate profile from the uploaded document."),
            (c3, ":material/hub:", "Entity linking", "Normalize skills to the curated knowledge base."),
        ):
            with column:
                with st.container(border=True):
                    st.markdown(f"### {icon} {title}")
                    st.caption(detail)
        st.subheader("Recent analysis")
        st.info("Upload a resume to get started.")
        return

    kpi = db.get_kpi_metrics()
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-title">Resumes</span>
            <span class="kpi-value">{kpi['total_resumes']}</span>
            <span class="kpi-sub">Total documents</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-title">Candidates</span>
            <span class="kpi-value">{kpi['total_candidates']}</span>
            <span class="kpi-sub">Extracted profiles</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-title">Unique Skills</span>
            <span class="kpi-value">{kpi['unique_skills']}</span>
            <span class="kpi-sub">Canonical entities</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-title">Entities Found</span>
            <span class="kpi-value">{kpi['total_entities']}</span>
            <span class="kpi-sub">Span detections</span>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-title">Completeness</span>
            <span class="kpi-value">{kpi['avg_completeness']}%</span>
            <span class="kpi-sub">Profile density</span>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="kpi-container">
            <span class="kpi-title">Success Rate</span>
            <span class="kpi-value">{kpi['success_rate']}%</span>
            <span class="kpi-sub">Pipeline reliability</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visualizations Section
    col_chart1, col_chart2 = st.columns(2)

    # 1. Skill Category Distribution
    with col_chart1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Skill Category Distribution")

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT category, COUNT(*) as cnt
                FROM entity_links
                WHERE category != '' AND category != 'Uncategorized'
                GROUP BY category
                ORDER BY cnt DESC
            """)
            cat_rows = cur.fetchall()

        if cat_rows:
            df_cat = pd.DataFrame([dict(r) for r in cat_rows])
            fig_cat = px.pie(
                df_cat, names="category", values="cnt",
                hole=0.45,
                color_discrete_sequence=theme_cfg["layout"]["colorway"]
            )
            fig_cat.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin={"t": 10, "b": 10, "l": 10, "r": 10},
                font={"family": "Inter, sans-serif"}
            )
            st.plotly_chart(fig_cat, width="stretch")
        else:
            st.caption("No categorized skills extracted yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Entity Category Distribution
    with col_chart2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("NER Entities Identified by Type")

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT entity_label, COUNT(*) as cnt
                FROM extracted_entities
                GROUP BY entity_label
                ORDER BY cnt DESC
            """)
            ent_rows = cur.fetchall()

        if ent_rows:
            df_ent = pd.DataFrame([dict(r) for r in ent_rows])
            fig_ent = px.bar(
                df_ent, x="entity_label", y="cnt",
                color="entity_label",
                color_discrete_sequence=theme_cfg["layout"]["colorway"]
            )
            fig_ent.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin={"t": 10, "b": 10, "l": 10, "r": 10},
                xaxis_title="Entity Label",
                yaxis_title="Total Count",
                showlegend=False,
                font={"family": "Inter, sans-serif"}
            )
            st.plotly_chart(fig_ent, width="stretch")
        else:
            st.caption("No entities identified yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Most Frequently Extracted Skills
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("Top Extracted Canonical Skills Across Corpus")

    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT canonical_name, category, COUNT(*) as frequency
            FROM entity_links
            GROUP BY canonical_name
            ORDER BY frequency DESC
            LIMIT 10
        """)
        top_skills = cur.fetchall()

    if top_skills:
        df_top = pd.DataFrame([dict(r) for r in top_skills])
        fig_top = px.bar(
            df_top, x="frequency", y="canonical_name",
            orientation='h',
            color="category",
            color_discrete_sequence=theme_cfg["layout"]["colorway"]
        )
        fig_top.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Document Occurrences",
            yaxis_title="",
            font={"family": "Inter, sans-serif"}
        )
        st.plotly_chart(fig_top, width="stretch")
    else:
        st.caption("No skills to rank.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Recent Activity Table
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("Recent Analyzed Resumes")

    table_data = []
    for r in all_resumes[:10]:
        prof = db.get_candidate_profile(r["id"])
        cname = prof.get("candidate_name") if prof else ""
        cname = cname or "Candidate Profile"
        completeness = prof.get("completeness_score", 0.0) if prof else 0.0

        table_data.append({
            "Resume ID": f"RES-{r['id']:03d}",
            "Candidate Name": cname,
            "File Name": r["filename"],
            "Format": r["file_type"],
            "Pages": r["page_count"],
            "Completeness": f"{completeness}%",
            "Upload Timestamp": r["upload_timestamp"][:16].replace("T", " ")
        })

    st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
