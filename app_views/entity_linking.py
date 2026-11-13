"""
SmartResume AI - Entity Linking & Normalization Page
Disambiguates extracted skill terms to canonical knowledge-base concepts:
Exact, Alias, Case-Insensitive, and Fuzzy Levenshtein matching.
Includes knowledge-base management and manual curation tools.
"""

import streamlit as st
import pandas as pd
import json
from database.db_manager import db
from modules.entity_linking import EntityLinker
from config.settings import SKILL_KB_PATH


def render_entity_linking():
    """Renders the interactive Entity Linking and Skill Canonicalization page."""
    st.markdown('<div class="brand-badge">CORE NLP FEATURE • KNOWLEDGE GRAPH LINKING</div>', unsafe_allow_html=True)
    st.title("Entity Linking & Normalization")
    st.caption("Grounds raw, heterogeneous skill mentions (e.g., 'py', 'JS', 'ML', 'k8s') to standardized canonical ontology concepts.")

    # Educational distinction box
    with st.expander("ℹ️ Academic Concept: NER vs. Entity Linking Explained", expanded=False):
        st.markdown("""
        - **Named Entity Recognition (NER)** is an information extraction task that detects **surface spans** in unstructured text and tags them with broad types (e.g., tagging `"Py"` or `"TensorFlow"` as `SKILL`). NER operates purely at the string token level.
        - **Entity Linking (EL)** is a knowledge grounding task that maps detected surface mentions to **canonical, unique entities** in a structured knowledge base (e.g., resolving both `"Py"` and `"python3"` to canonical entity `"Python"` [Wikidata Q28865]).
        - **Matching Priority:** (1) Exact Matching → (2) Case-Insensitive Matching → (3) Curated Alias Matching → (4) Fuzzy Levenshtein Distance.
        """)

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_label = st.selectbox("Select Resume to Disambiguate:", list(resume_options.keys()))
    selected_id = resume_options[selected_label]

    linker = EntityLinker()
    linked_records = db.get_entity_links(selected_id)

    if not linked_records:
        # Link terms on the fly
        prof = db.get_candidate_profile(selected_id)
        raw_terms = []
        if prof and isinstance(prof.get("skills"), dict):
            for cat, sk_list in prof["skills"].items():
                raw_terms.extend(sk_list)
        linked_records = linker.link_terms(raw_terms, deduplicate=True)
        db.save_entity_links(selected_id, linked_records)

    # Metrics
    total_linked = len(linked_records)
    exact_matches = sum(1 for l in linked_records if l["match_method"] in ["Exact", "Case-Insensitive"])
    alias_matches = sum(1 for l in linked_records if l["match_method"] == "Alias")
    fuzzy_matches = sum(1 for l in linked_records if l["match_method"] == "Fuzzy")
    unmatched = sum(1 for l in linked_records if l["match_method"] == "Unmatched")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Extracted Terms", total_linked)
    with c2:
        st.metric("Exact Matches", exact_matches)
    with c3:
        st.metric("Alias Matches", alias_matches)
    with c4:
        st.metric("Fuzzy Matches", fuzzy_matches)
    with c5:
        st.metric("Unmatched", unmatched)

    st.write("")

    tabs = st.tabs([
        "🔗 Canonical Mapping Table",
        "❓ Unmatched Terms",
        "📚 Knowledge Base Management",
        "🧪 Interactive Term Linker"
    ])

    # Tab 1: Mapping Table
    with tabs[0]:
        st.subheader("Extracted Term -> Standard Canonical Entity Mappings")

        table_rows = []
        for l in linked_records:
            method_badge = l["match_method"]
            table_rows.append({
                "Extracted Term": l["extracted_term"],
                "Standard Entity": l["canonical_name"],
                "Category": l["category"],
                "Match Method": method_badge,
                "Confidence / Similarity": f"{round(l.get('similarity_score', 1.0) * 100)}%",
                "Status": l.get("status", "Linked")
            })

        df_links = pd.DataFrame(table_rows)
        st.dataframe(df_links, width="stretch", hide_index=True)

    # Tab 2: Unmatched
    with tabs[1]:
        st.subheader("Unmatched & Low-Confidence Mentions")
        st.caption("Mentions where no confident canonical match was found (similarity < 82%). Review or add to knowledge base.")

        unmatched_rows = [l for l in linked_records if l["match_method"] == "Unmatched"]
        if unmatched_rows:
            st.dataframe(pd.DataFrame(unmatched_rows), width="stretch", hide_index=True)
        else:
            st.success("All extracted skill terms were successfully linked to canonical concepts!")

    # Tab 3: Knowledge Base Manager
    with tabs[2]:
        st.subheader("Skill Knowledge Base (Ontology)")
        st.caption(f"Currently managing `{len(linker.knowledge_base.get('skills', []))}` canonical skills.")

        with st.expander("➕ Add New Skill Concept to Knowledge Base"):
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                new_canon = st.text_input("Canonical Skill Name (e.g. 'GraphQL'):")
                new_cat = st.selectbox("Category:", [
                    "Programming Languages", "Web Frameworks & Libraries", "AI & Data Science",
                    "Databases & Storage", "Cloud & DevOps", "Developer Tools", "Big Data", "Methodologies & Soft Skills"
                ])
            with col_add2:
                new_aliases = st.text_input("Aliases (comma separated, e.g. 'gql, graph-ql'):")
                new_ext_id = st.text_input("External ID (e.g. Wikidata Q25110837):")

            if st.button("Add Skill to Knowledge Base"):
                if new_canon:
                    alias_list = [a.strip() for a in new_aliases.split(",") if a.strip()]
                    linker.add_custom_skill(new_canon, new_cat, alias_list, new_ext_id)
                    st.success(f"Added '{new_canon}' to Skill Knowledge Base!")
                    st.rerun()

        # Display KB records
        kb_skills = linker.knowledge_base.get("skills", [])
        kb_df = pd.DataFrame([{
            "ID": s["id"],
            "Canonical Name": s["canonical_name"],
            "Category": s["category"],
            "Aliases": ", ".join(s.get("aliases", [])),
            "Wikidata ID": s.get("external_id", "")
        } for s in kb_skills])
        st.dataframe(kb_df, width="stretch", height=350, hide_index=True)

    # Tab 4: Interactive Tester
    with tabs[3]:
        st.subheader("Test Disambiguation on Any Term")
        test_term = st.text_input("Enter any skill term or acronym (e.g. 'k8s', 'ML', 'reactjs', 'py'):", value="k8s")
        if test_term:
            res = linker.link_term(test_term)
            st.markdown(f"""
            <div class="saas-card">
                <h3>Result for '{test_term}'</h3>
                <p><strong>Standard Entity:</strong> <span style="color: #2563EB; font-weight: 600;">{res['canonical_name']}</span></p>
                <p><strong>Category:</strong> {res['category']}</p>
                <p><strong>Disambiguation Method:</strong> <code>{res['match_method']}</code></p>
                <p><strong>Similarity Score:</strong> {res['similarity_score'] * 100}%</p>
            </div>
            """, unsafe_allow_html=True)
