"""
SmartResume AI - Generative AI Resume Assistant Page
Evidence-grounded conversational Q&A, executive summaries, and resume critique
powered by Google Gemini API.
"""

import json

import streamlit as st
from database.db_manager import db
from modules.genai_assistant import GenAIAssistant


def render_ai_assistant():
    """Renders the Google Gemini-powered AI Resume Assistant view."""
    st.markdown('<div class="brand-badge">GENERATIVE AI • GOOGLE GEMINI</div>', unsafe_allow_html=True)
    st.title("AI Resume Assistant")
    st.caption("Ask questions, generate grounded executive summaries, and receive actionable resume optimization critiques.")

    assistant = GenAIAssistant()
    is_configured = assistant.is_configured()

    if not is_configured:
        st.warning(
            f"🔑 **Google Gemini API Key Required:** {assistant.configuration_message()} "
            "All other NLP modules (Extraction, NER, Linking, Sentiment, Topics) remain 100% operational offline without an API key."
        )
    elif st.button("Test Gemini connection", icon=":material/wifi:"):
        status = assistant.test_connection()
        if status["success"]:
            st.session_state["gemini_connection_verified"] = True
            st.success(f"Gemini connection verified with {status['model']}.")
        else:
            st.session_state["gemini_connection_verified"] = False
            st.error(status["error"])

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_label = st.selectbox("Select Target Resume for AI Analysis:", list(resume_options.keys()))
    selected_id = resume_options[selected_label]

    resume = db.get_resume(selected_id)
    raw_text = resume["original_text"]

    tabs = st.tabs([
        "💬 Interactive Resume Q&A",
        "📋 AI Executive Summary",
        "💡 Optimization & Critique Suggestions",
        "🧩 Structured resume insights",
    ])

    # Tab 1: Q&A
    with tabs[0]:
        st.subheader("Grounded Resume Question Answering")
        st.caption("All answers are strictly constrained to facts explicitly present in the candidate resume.")

        # Preset Question Chips
        st.markdown("**Quick Preset Prompts:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        preset_q = ""
        with col_q1:
            if st.button("🔍 What are the strongest technical skills?"):
                preset_q = "What are the candidate's strongest technical skills mentioned in the resume?"
        with col_q2:
            if st.button("💼 Summary of employment history?"):
                preset_q = "What companies has the candidate worked for, and what were their primary roles?"
        with col_q3:
            if st.button("🎓 What degrees & education are held?"):
                preset_q = "What is the candidate's educational background and degrees?"

        # Chat interface
        user_question = st.text_input(
            "Ask any question about this resume:",
            value=preset_q if preset_q else "",
            placeholder="e.g., Does the candidate have experience with Kubernetes or microservices?"
        )

        if st.button("Send Question", type="primary") and user_question:
            if not is_configured:
                st.error("Please configure GEMINI_API_KEY in Settings to use the generative assistant.")
            else:
                with st.spinner("SmartResume AI is consulting the resume..."):
                    res = assistant.answer_question(raw_text, user_question)
                    if res["success"]:
                        db.save_ai_interaction(selected_id, user_question, res["answer"])
                        st.markdown(f"""
                        <div class="saas-card" style="background: #FFFFFF; border-left: 4px solid #2563EB;">
                            <h4 style="margin: 0 0 8px 0; color: #1E40AF;">AI Grounded Response:</h4>
                            <div style="font-size: 0.92rem; line-height: 1.6; color: #0F172A;">
                                {res['answer']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(res["error"])

        # Session Conversation History
        st.write("")
        history = db.get_ai_interactions(selected_id)
        if history:
            with st.expander(f"📜 View Past Questions for this Resume ({len(history)})", expanded=False):
                for h in reversed(history):
                    st.markdown(f"**Q:** {h['question']}")
                    st.markdown(f"**A:** {h['answer']}")
                    st.caption(f"Timestamp: {h['timestamp'][:19].replace('T', ' ')}")
                    st.divider()

    # Tab 2: AI Summary
    with tabs[1]:
        st.subheader("Automated Executive Candidate Summary")
        st.caption("Generates a structured executive overview synthesized strictly from the resume content.")

        if st.button("⚡ Generate AI Summary", key="btn_gen_summary"):
            if not is_configured:
                st.error("Please configure GEMINI_API_KEY to generate AI summaries.")
            else:
                with st.spinner("Synthesizing executive summary with Gemini..."):
                    res = assistant.generate_summary(raw_text)
                    if res["success"]:
                        st.markdown(f"""
                        <div class="saas-card" style="line-height: 1.65; font-size: 0.92rem;">
                            {res['summary']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(res["error"])

    with tabs[3]:
        st.subheader("Structured resume insights")
        st.caption("JSON grounded only in the selected uploaded resume. Missing information is marked explicitly.")
        if st.button("Generate structured insights", key="btn_structured", type="primary"):
            if not is_configured:
                st.error(assistant.configuration_message())
            else:
                with st.spinner("Creating grounded structured insights..."):
                    result = assistant.generate_structured_insights(raw_text)
                if result["success"]:
                    st.json(result["data"], expanded=2)
                    db.save_ai_interaction(selected_id, "Generate structured resume insights", json.dumps(result["data"]))
                else:
                    st.error(result["error"])

    # Tab 3: Critique & Suggestions
    with tabs[2]:
        st.subheader("Resume Improvement Critique & ATS Guidance")
        st.caption("Analyzes clarity, metric quantification, missing standard sections, and ATS keyword relevance.")

        if st.button("🔍 Analyze for Improvements", key="btn_critique"):
            if not is_configured:
                st.error("Please configure GEMINI_API_KEY to analyze improvements.")
            else:
                with st.spinner("Reviewing resume structure and wording..."):
                    res = assistant.suggest_improvements(raw_text)
                    if res["success"]:
                        st.markdown(f"""
                        <div class="saas-card" style="line-height: 1.65; font-size: 0.92rem;">
                            {res['critique']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(res["error"])

