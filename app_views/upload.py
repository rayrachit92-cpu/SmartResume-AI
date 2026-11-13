"""
SmartResume AI - Resume Upload Module
Upload single or bulk PDF, DOCX, and TXT resumes.
Detects scanned documents, extracts text, allows manual correction,
and triggers the NLP processing pipeline.
"""

import streamlit as st
from pathlib import Path
from database.db_manager import db
from modules.text_extraction import DocumentExtractor
from modules.preprocessing import TextPreprocessor
from modules.ner import ResumeNER
from modules.information_extraction import InformationExtractor
from modules.entity_linking import EntityLinker
from modules.sentiment_analysis import ResumeSentimentAnalyzer


def render_upload():
    """Renders the resume upload and ingestion interface."""
    st.markdown('<div class="brand-badge">DOCUMENT INGESTION</div>', unsafe_allow_html=True)
    st.title("Upload & Ingest Resumes")
    st.caption("Upload candidate resumes in PDF, DOCX, or TXT format. Text is extracted, verified, and parsed into structured intelligence.")

    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Select one or multiple resumes to process",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, and TXT. Max file size: 15MB each."
        )

        if uploaded_files:
            st.subheader(f"Files Ready for Ingestion ({len(uploaded_files)})")

            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                file_size = uploaded_file.size
                file_bytes = uploaded_file.read()

                # Check if duplicate already exists
                existing = [r for r in db.get_all_resumes() if r["filename"] == filename]

                with st.expander(f"📄 {filename} ({(file_size / 1024):.1f} KB)", expanded=True):
                    if existing:
                        st.warning(f"⚠️ A document named **{filename}** has already been processed previously in the database.")

                    col_proc1, col_proc2 = st.columns([3, 1])

                    try:
                        extracted = DocumentExtractor.extract_from_bytes(file_bytes, filename)
                        raw_text = extracted["text"]

                        with col_proc1:
                            st.markdown(f"""
                            **Format:** `{extracted['file_type']}` | 
                            **Pages:** `{extracted['page_count']}` | 
                            **Words:** `{extracted['word_count']:,}` | 
                            **Scanned:** `{'Yes (OCR Applied)' if extracted['ocr_applied'] else ('Yes (Detected)' if extracted['is_scanned'] else 'No (Direct Text)')}`
                            """)

                        # Editable extracted text preview
                        edited_text = st.text_area(
                            "Extracted Document Text (Review or edit before final parsing):",
                            value=raw_text,
                            height=220,
                            key=f"edit_{filename}"
                        )

                        if st.button(f"Analyze resume: {filename}", key=f"btn_{filename}", type="primary"):
                            with st.spinner("Executing NLP pipeline (Preprocessing, NER, Information Extraction, Entity Linking, Sentiment)..."):
                                # 1. Preprocessing
                                prep = TextPreprocessor.full_pipeline(edited_text)

                                # 2. Database Insert
                                resume_id = db.insert_resume(
                                    filename=filename,
                                    file_type=extracted["file_type"],
                                    original_text=edited_text,
                                    file_size=file_size,
                                    is_scanned=extracted["is_scanned"],
                                    page_count=extracted["page_count"],
                                    cleaned_text=prep["cleaned_text"],
                                    tokens_count=prep["filtered_token_count"]
                                )

                                # 3. Structured Profile & NER
                                ner = ResumeNER()
                                extractor = InformationExtractor(ner_engine=ner)
                                profile = extractor.extract_full_profile(edited_text)
                                db.save_candidate_profile(resume_id, profile)

                                # 4. Save Entities
                                entities = profile.get("entities", [])
                                db.save_entities(resume_id, entities)

                                # 5. Entity Linking
                                linker = EntityLinker()
                                raw_skills = []
                                for cat, sk_list in profile.get("skills", {}).items():
                                    raw_skills.extend(sk_list)
                                for ent in entities:
                                    if ent["label"] == "SKILL":
                                        raw_skills.append(ent["text"])
                                linked = linker.link_terms(raw_skills, deduplicate=True)
                                db.save_entity_links(resume_id, linked)

                                # 6. Sentiment
                                sections = extractor.detect_sections(edited_text)
                                senti = ResumeSentimentAnalyzer.analyze_resume_sections(sections)
                                db.save_sentiment(resume_id, senti["section_results"])

                            st.success(f"✓ '{filename}' successfully parsed and saved into candidate records! (ID: RES-{resume_id:03d})")
                            st.session_state["active_resume_id"] = resume_id
                            st.rerun()

                    except Exception as e:
                        st.error(f"Failed to process '{filename}': {str(e)}")


