"""
SmartResume AI - Text Preprocessing & Vectorization Analyzer Page
Interactive before-and-after inspection of text cleaning, tokenization,
lemmatization, stopword removal, Bag of Words, and TF-IDF matrices.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from database.db_manager import db
from modules.preprocessing import TextPreprocessor
from modules.vectorization import ResumeVectorizer
from ui.styles import get_plotly_theme


def render_analyzer():
    """Renders the NLP preprocessing and vectorization analysis page."""
    theme_cfg = get_plotly_theme()
    st.markdown('<div class="brand-badge">NLP PIPELINE DEEP-DIVE</div>', unsafe_allow_html=True)
    st.title("Text Preprocessing & Vectorization")
    st.caption("Inspect tokenization, lemmatization, stopword filtering, vocabulary reduction, Bag-of-Words, and TF-IDF representations.")

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    # Select resume
    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_label = st.selectbox("Select Resume to Inspect:", list(resume_options.keys()))
    selected_id = resume_options[selected_label]
    resume = db.get_resume(selected_id)

    raw_text = resume["original_text"]

    # Preprocessing Pipeline Execution
    prep = TextPreprocessor.full_pipeline(raw_text)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw Tokens", f"{prep['raw_token_count']:,}")
    with m2:
        st.metric("Cleaned Lemmas", f"{prep['filtered_token_count']:,}")
    with m3:
        st.metric("Vocabulary Size", f"{prep['vocabulary_size']:,}")
    with m4:
        st.metric("Reduction Ratio", f"{prep['reduction_percentage']}%")

    st.write("")

    tabs = st.tabs([
        "🔄 Before & After Cleaning",
        "🔤 Token & Lemma Breakdown",
        "📊 Word Frequency",
        "🔢 Bag-of-Words (CountVectorizer)",
        "📈 TF-IDF Matrix"
    ])

    # Tab 1: Before and After
    with tabs[0]:
        st.subheader("Text Normalization Comparison")
        col_orig, col_clean = st.columns(2)
        with col_orig:
            st.markdown("**Original Extracted Document Text:**")
            st.text_area("Original", raw_text, height=350, disabled=True, label_visibility="collapsed")
        with col_clean:
            st.markdown("**Normalized & Cleaned Text (Whitespace & Unicode Standardized):**")
            st.text_area("Cleaned", prep["cleaned_text"], height=350, disabled=True, label_visibility="collapsed")

        with st.expander("Sentence Segmentation (NLTK sent_tokenize preview)"):
            for i, sent in enumerate(prep["sentences"]):
                st.markdown(f"**[{i+1}]** {sent}")

    # Tab 2: Token Breakdown
    with tabs[1]:
        st.subheader("Token-Level Lemmatization & Stopword Classification")
        st.caption("Inspecting sample tokens showing the raw surface form, lemmatized root form, and linguistic filters.")

        tokens_df = pd.DataFrame(prep["token_details"])
        st.dataframe(tokens_df, width="stretch", hide_index=True)

    # Tab 3: Word Frequency
    with tabs[2]:
        st.subheader("Top Lemmatized Terms by Frequency")
        top_words = prep["top_words"]
        if top_words:
            df_wf = pd.DataFrame(top_words, columns=["Term", "Count"])
            fig_wf = px.bar(
                df_wf, x="Count", y="Term",
                orientation='h',
                color="Count",
                color_continuous_scale="Blues"
            )
            fig_wf.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                yaxis={'categoryorder': 'total ascending'},
                font={"family": "Inter, sans-serif"},
                margin={"t": 20, "b": 20, "l": 20, "r": 20}
            )
            st.plotly_chart(fig_wf, width="stretch")
        else:
            st.caption("No frequent terms found.")

    # Tab 4: Bag of Words
    with tabs[3]:
        st.subheader("Bag of Words (CountVectorizer Representation)")
        st.caption("Sparse term frequency matrix computed using Scikit-Learn with stopword elimination.")

        corpus_texts = [r["original_text"] for r in resumes]
        bow_data = ResumeVectorizer.compute_bow(corpus_texts, max_features=30)

        st.markdown(f"**Vocabulary Size (Top Features):** `{bow_data['vocabulary_size']}` terms")
        st.dataframe(bow_data["dataframe"], width="stretch")

    # Tab 5: TF-IDF Matrix
    with tabs[4]:
        st.subheader("TF-IDF (Term Frequency - Inverse Document Frequency)")
        st.caption("Sublinear TF-IDF feature matrix highlighting distinguishing domain terms across resumes.")

        corpus_texts = [r["original_text"] for r in resumes]
        tfidf_data = ResumeVectorizer.compute_tfidf(corpus_texts, max_features=30)

        st.dataframe(tfidf_data["dataframe"], width="stretch")

        st.markdown("**Top Salient TF-IDF Keywords Across Corpus:**")
        if tfidf_data["top_terms"]:
            df_tf_top = pd.DataFrame(tfidf_data["top_terms"])
            st.dataframe(df_tf_top, width="stretch", hide_index=True)
