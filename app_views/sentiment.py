"""
SmartResume AI - Sentiment & Tone Analysis Page
Analyzes writing tone across professional summaries, project narratives,
and experience statements using NLTK VADER.
Includes prominent ethical disclaimers and tone distribution charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.db_manager import db
from modules.sentiment_analysis import ResumeSentimentAnalyzer
from modules.information_extraction import InformationExtractor
from ui.styles import get_plotly_theme


def render_sentiment():
    """Renders the descriptive sentiment and writing tone analysis page."""
    theme_cfg = get_plotly_theme()
    st.markdown('<div class="brand-badge">TONE & LINGUISTIC VALENCE</div>', unsafe_allow_html=True)
    st.title("Resume Sentiment & Tone Analysis")
    st.caption("Quantifies linguistic valence and communication tone in narrative resume sections using rule-based VADER sentiment.")

    # Ethical Disclaimer Banner
    st.warning(
        "⚠️ **Academic & Ethical Disclaimer:** Tone analysis measures descriptive vocabulary valence only. "
        "It must NOT be interpreted as a measure of candidate competence, intelligence, character, "
        "work ethic, or hiring suitability. Professional resumes naturally exhibit predominantly neutral/objective vocabulary."
    )

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_label = st.selectbox("Select Resume to Analyze:", list(resume_options.keys()))
    selected_id = resume_options[selected_label]

    resume = db.get_resume(selected_id)
    raw_text = resume["original_text"]

    # Extract sections and analyze
    extractor = InformationExtractor()
    sections = extractor.detect_sections(raw_text)
    sentiment_data = ResumeSentimentAnalyzer.analyze_resume_sections(sections)

    overall_compound = sentiment_data["overall_compound"]
    overall_tone = sentiment_data["overall_tone"]
    section_results = sentiment_data["section_results"]

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Compound Score", f"{overall_compound:+.4f}")
    with c2:
        st.metric("Overall Tone", overall_tone)
    with c3:
        st.metric("Analyzed Sections", len(section_results))
    with c4:
        st.metric("Polarity Index", "Enthusiastic" if overall_compound > 0.4 else ("Neutral" if overall_compound > -0.05 else "Problem-Focused"))

    st.write("")

    col_gauge, col_bar = st.columns([1, 1])

    # Gauge Chart for Compound Score
    with col_gauge:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Tone Valence Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_compound,
            title={'text': "Compound Polarity (-1.0 to +1.0)", 'font': {'size': 14, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [-1.0, 1.0], 'tickwidth': 1, 'tickcolor': "#64748B"},
                'bar': {'color': "#2563EB"},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [-1.0, -0.05], 'color': '#FEE2E2'},
                    {'range': [-0.05, 0.45], 'color': '#F1F5F9'},
                    {'range': [0.45, 1.0], 'color': '#DCFCE7'}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.75,
                    'value': overall_compound
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#FFFFFF",
            font={"family": "Inter, sans-serif"},
            height=260,
            margin={"t": 30, "b": 10, "l": 20, "r": 20}
        )
        st.plotly_chart(fig_gauge, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Breakdown by Section
    with col_bar:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Polarity Proportions by Section")

        if section_results:
            chart_rows = []
            for s in section_results:
                chart_rows.append({"Section": s["section_name"], "Type": "Positive", "Score": s["pos"]})
                chart_rows.append({"Section": s["section_name"], "Type": "Neutral", "Score": s["neu"]})
                chart_rows.append({"Section": s["section_name"], "Type": "Negative", "Score": s["neg"]})

            df_sec = pd.DataFrame(chart_rows)
            fig_sec = px.bar(
                df_sec, x="Section", y="Score", color="Type",
                barmode="stack",
                color_discrete_map={"Positive": "#10B981", "Neutral": "#94A3B8", "Negative": "#EF4444"}
            )
            fig_sec.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font={"family": "Inter, sans-serif"},
                height=260,
                margin={"t": 30, "b": 10, "l": 20, "r": 20}
            )
            st.plotly_chart(fig_sec, width="stretch")
        else:
            st.caption("No sections available for polarity breakdown.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Section Table
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("Section-by-Section Tone Analysis Breakdown")

    detail_table = []
    for s in section_results:
        detail_table.append({
            "Section Name": s["section_name"],
            "Tone Classification": s["tone_label"],
            "Compound": f"{s['compound']:+.4f}",
            "Positive": f"{round(s['pos'] * 100)}%",
            "Neutral": f"{round(s['neu'] * 100)}%",
            "Negative": f"{round(s['neg'] * 100)}%",
            "Word Count": s["word_count"],
            "Snippet": s["text_snippet"]
        })

    st.dataframe(pd.DataFrame(detail_table), width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
