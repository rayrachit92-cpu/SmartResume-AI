"""
SmartResume AI - Topic Modeling Page
Discovers career clusters and domain specializations across resumes using
Latent Dirichlet Allocation (LDA) with Scikit-learn.
Interactive topic slider, top keywords, and document-topic distributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from database.db_manager import db
from modules.topic_modeling import ResumeTopicModeler
from ui.styles import get_plotly_theme


def render_topics():
    """Renders the unsupervised LDA Topic Modeling page."""
    theme_cfg = get_plotly_theme()
    st.markdown('<div class="brand-badge">UNSUPERVISED LEARNING • LDA</div>', unsafe_allow_html=True)
    st.title("Career Domain Topic Modeling (LDA)")
    st.caption("Discovers latent technological themes, job family clusters, and domain distributions across the processed resume corpus.")

    resumes = db.get_all_resumes()
    n_docs = len(resumes)

    if n_docs < 2:
        st.info(f"Topic modeling requires at least 2 uploaded resumes to discover distributions across documents (Current: {n_docs}).")
        return

    # User parameter controls
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    col_param1, col_param2, col_param3 = st.columns([2, 2, 1])

    with col_param1:
        n_topics = st.slider("Number of Latent Topics (k):", min_value=2, max_value=min(8, max(2, n_docs)), value=min(4, max(2, n_docs)))
    with col_param2:
        n_words = st.slider("Top Keywords per Topic:", min_value=5, max_value=15, value=8)
    with col_param3:
        st.write("")
        st.write("")
        rerun_btn = st.button("🔄 Retrain LDA Model")

    st.markdown('</div>', unsafe_allow_html=True)

    # Execute LDA on corpus
    corpus_texts = [r["original_text"] for r in resumes]
    corpus_filenames = [f"RES-{r['id']:03d} • {r['filename']}" for r in resumes]

    modeler = ResumeTopicModeler(n_topics=n_topics, max_iter=30)
    lda_results = modeler.fit_transform(corpus_texts, n_words_per_topic=n_words)

    if lda_results["status"] != "success":
        st.warning(lda_results["message"])
        return

    topics = lda_results["topics"]
    doc_topics = lda_results["document_topics"]

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Resumes in Corpus", lda_results["total_documents"])
    with m2:
        st.metric("Topics Modeled", lda_results["n_topics"])
    with m3:
        st.metric("Model Perplexity", lda_results.get("perplexity", "N/A"))

    st.write("")

    # Topic Cards & Keywords
    st.subheader("Discovered Career Topics & Salient Keywords")
    cols = st.columns(len(topics))
    for i, t in enumerate(topics):
        with cols[i]:
            st.markdown(f"""
            <div class="saas-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; text-transform: uppercase;">
                    Topic {t['topic_id'] + 1}
                </div>
                <h4 style="margin: 4px 0 10px 0; color: #1E40AF; font-size: 1.05rem;">{t['label']}</h4>
                <p style="font-size: 0.82rem; color: #0F172A; margin: 0 0 8px 0;"><strong>Keywords:</strong></p>
                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                    {' '.join([f'<span style="background: #F1F5F9; color: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.74rem;">{kw}</span>' for kw in t['top_keywords'][:6]])}
                </div>
                <div style="margin-top: 14px; font-size: 0.78rem; color: #0D9488; font-weight: 600;">
                    👥 {t['assigned_document_count']} Resumes Assigned
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Charts Section
    col_pie, col_matrix = st.columns([1, 1])

    with col_pie:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Corpus Topic Distribution")

        pie_data = [{"Topic": f"Topic {t['topic_id'] + 1} ({t['label']})", "Resumes": t["assigned_document_count"]} for t in topics]
        fig_pie = px.pie(
            pd.DataFrame(pie_data), names="Topic", values="Resumes",
            hole=0.45,
            color_discrete_sequence=theme_cfg["layout"]["colorway"]
        )
        fig_pie.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin={"t": 10, "b": 10, "l": 10, "r": 10},
            font={"family": "Inter, sans-serif"}
        )
        st.plotly_chart(fig_pie, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_matrix:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.subheader("Document-Topic Probabilities")

        matrix_rows = []
        for i, dt in enumerate(doc_topics):
            row = {"Resume": corpus_filenames[i], "Dominant Topic": dt["dominant_topic_label"], "Confidence": f"{round(dt['confidence'] * 100)}%"}
            row.update(dt["distribution"])
            matrix_rows.append(row)

        df_matrix = pd.DataFrame(matrix_rows)
        st.dataframe(df_matrix[["Resume", "Dominant Topic", "Confidence"]], width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
