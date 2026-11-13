"""
SmartResume AI - Named Entity Recognition (NER) Page
Interactive entity visualization, color-coded highlights, category filtering,
and span-level confidence metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from database.db_manager import db
from modules.ner import ResumeNER, ENTITY_COLORS, ENTITY_TEXT_COLORS
from ui.styles import get_plotly_theme


def render_ner():
    """Renders the interactive Named Entity Recognition view."""
    theme_cfg = get_plotly_theme()
    st.markdown('<div class="brand-badge">CORE NLP FEATURE • TOKEN CLASSIFICATION</div>', unsafe_allow_html=True)
    st.title("Named Entity Recognition (NER)")
    st.caption("Identify and classify resume entities: PERSON, EMAIL, PHONE, ORGANIZATION, LOCATION, DEGREE, UNIVERSITY, SKILL, JOB_TITLE, and CERTIFICATION.")

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_label = st.selectbox("Select Resume for Entity Extraction:", list(resume_options.keys()))
    selected_id = resume_options[selected_label]

    resume = db.get_resume(selected_id)
    raw_text = resume["original_text"]

    # Load entities from DB or extract fresh
    entities = db.get_entities(selected_id)
    if not entities:
        ner = ResumeNER()
        entities = ner.extract_entities(raw_text)
        db.save_entities(selected_id, entities)

    # Visual Legend
    st.markdown("#### Entity Types & Color Legend")
    legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">'
    for label, bg in ENTITY_COLORS.items():
        fg = ENTITY_TEXT_COLORS.get(label, "#334155")
        legend_html += (
            f'<span style="background-color: {bg}; color: {fg}; padding: 4px 10px; '
            f'border-radius: 6px; font-size: 0.78rem; font-weight: 600; border: 1px solid rgba(0,0,0,0.06);">'
            f'{label}</span>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # Filter Controls
    all_categories = sorted(list(set(e["entity_label"] for e in entities)))
    selected_categories = st.multiselect(
        "Filter Entities by Category:",
        options=all_categories,
        default=all_categories
    )

    filtered_entities = [e for e in entities if e["entity_label"] in selected_categories]

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Entities Identified", len(entities))
    with c2:
        st.metric("Active Filtered Entities", len(filtered_entities))
    with c3:
        st.metric("Distinct Categories Present", len(all_categories))

    st.write("")

    col_view1, col_view2 = st.columns([1, 1])

    # Left: Color Highlighted Text
    with col_view1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Highlighted Document Text")
        highlighted_html = ResumeNER.render_highlighted_text(
            raw_text,
            [{"start": e["start_char"], "end": e["end_char"], "label": e["entity_label"]} for e in filtered_entities]
        )
        st.markdown(
            f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; '
            f'padding: 16px; height: 500px; overflow-y: auto; line-height: 1.7; font-size: 0.88rem;">'
            f'{highlighted_html}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Right: Extracted Entity Table & Distribution
    with col_view2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Extracted Entity Span Table")

        table_data = []
        for e in filtered_entities:
            table_data.append({
                "Entity Text": e["entity_text"],
                "Category": e["entity_label"],
                "Span": f"[{e['start_char']}:{e['end_char']}]",
                "Confidence": f"{round(e.get('confidence', 0.9) * 100)}%"
            })

        df_ents = pd.DataFrame(table_data)
        st.dataframe(df_ents, width="stretch", height=280, hide_index=True)

        st.subheader("Entity Frequency by Category")
        if not df_ents.empty:
            cat_counts = df_ents["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]

            fig_bar = px.bar(
                cat_counts, x="Category", y="Count",
                color="Category",
                color_discrete_sequence=theme_cfg["layout"]["colorway"]
            )
            fig_bar.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin={"t": 10, "b": 10, "l": 10, "r": 10},
                showlegend=False,
                font={"family": "Inter, sans-serif"},
                height=180
            )
            st.plotly_chart(fig_bar, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
