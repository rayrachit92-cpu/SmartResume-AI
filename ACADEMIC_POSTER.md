# 🎓 ACADEMIC PROJECT POSTER
## SmartResume AI: Intelligent Resume Information Extraction & AI-Powered Candidate Intelligence Platform

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          SMARTRESUME AI                                                │
│         Automated Resume Information Extraction, Grounded Entity Linking & Career Intelligence         │
│                        Academic NLP Mini-Project • Department of Computer Science                      │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. PROBLEM STATEMENT & MOTIVATION
Traditional recruitment systems struggle with unstructured documents across varying formats (PDF, DOCX). Standard ATS keyword search engines are brittle to surface-level synonyms (e.g. failing to match "py" to "Python"), lack structured section awareness, and frequently introduce opaque algorithmic biases. SmartResume AI addresses this challenge through an **interpretable, evidence-grounded NLP pipeline** combining local token classification, ontology entity linking, and conversational LLM assistance without qualification hallucination.

---

### 2. CORE OBJECTIVES
- **Document Ingestion:** Automated text extraction across single and bulk PDF/DOCX resumes with OCR fallback.
- **NLP Preprocessing & Vectorization:** Tokenization, lemmatization, stopword filtering, Bag-of-Words, and TF-IDF matrices.
- **Named Entity Recognition:** Hybrid transition-based token classification identifying 10+ resume categories.
- **Structured Profile Extraction:** Converting messy prose into normalized candidate profiles with completeness scoring.
- **Entity Linking:** Disambiguating messy skill mentions to canonical entities via a 4-stage matching hierarchy.
- **Sentiment & Tone Analysis:** Evaluating narrative communication tone with strict ethical hiring boundaries.
- **Topic Modeling (LDA):** Unsupervised discovery of career domain distributions and technology clusters.
- **AI Assistant:** Grounded resume Q&A and critique powered by Google Gemini.

---

### 3. SYSTEM ARCHITECTURE

```
┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│   DOCUMENT UPLOAD    │ ───► │     NLP PREPROCESSING   │ ───► │       HYBRID NER        │
│  PDF / DOCX / TXT    │      │  Tokenize, Lemma, BoW,  │      │  Transition Parser +    │
│  (PyMuPDF / OCR)     │      │  TF-IDF Feature Matrix  │      │  301 EntityRuler Rules  │
└──────────────────────┘      └─────────────────────────┘      └───────────┬─────────────┘
                                                                           │
┌──────────────────────┐      ┌─────────────────────────┐                  ▼
│  EXECUTIVE DASHBOARD │ ◄─── │   STRUCTURED EXTRACTION │ ◄─── ┌─────────────────────────┐
│  Linear SaaS Design  │      │  Relational Candidate   │      │     ENTITY LINKING      │
│  Live KPIs & Charts  │      │  Profile & Completeness │      │  Exact -> Alias ->      │
└──────────┬───────────┘      └─────────────────────────┘      │  Fuzzy -> Skill KB      │
           │                                                   └─────────────────────────┘
           ├────────────────────────────┬────────────────────────────┐
           ▼                            ▼                            ▼
┌──────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│  SENTIMENT ANALYSIS  │   │   TOPIC MODELING (LDA)  │   │   AI RESUME ASSISTANT   │
│  NLTK VADER Valence  │   │  Career Domain Clusters │   │  Grounded Google Gemini │
│  Ethical Safeguards  │   │  Perplexity & Weights   │   │  Q&A and Optimization   │
└──────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

---

### 4. DATASETS & MODEL TRAINING
- **Primary Corpus:** `yashpwr/resume-ner-training-data` (22,855 resumes with gold summaries, MIT License).
- **Independent Benchmark:** `dotin-inc/resume-dataset-NER-annotations` (360 annotated resumes, 18,224 entity labels, CC0-1.0).
- **Data Splitting:** 80% Train (288 resumes), 10% Validation (36 resumes), 10% Held-out Test (36 resumes). Zero data leakage.

---

### 5. EXPERIMENTAL RESULTS (HELD-OUT TEST SET)

| Metric | Measured Score | Notes |
| :--- | :---: | :--- |
| **Overall Micro F1-Score** | **29.38%** | Evaluated strictly on 1,102 unseen ground-truth spans |
| **PERSON Precision** | **75.00%** | Accurate name identification |
| **DATE Precision** | **66.67%** | Clean date range extraction |
| **LOCATION F1-Score** | **49.77%** | Geographic entity grounding |
| **UNIVERSITY F1-Score**| **48.28%** | Academic institution extraction |
| **Entity Linking Accuracy** | **98.5%** | Measured on curated alias and exact mappings |
| **Profile Completeness Avg**| **100.0%** | On standardized multi-section resumes |

---

### 6. KEY INNOVATIONS & SAAS FEATURES
1. **Light-Theme SaaS Interface:** Engineered with Linear, Notion, and Stripe design principles (clean white, Inter typography, soft borders).
2. **Transparent Knowledge Graph Grounding:** Every extracted skill is mapped to a canonical ontology with method transparency (Exact, Alias, or Fuzzy).
3. **Evidence-Grounded AI Assistant:** Strictly prohibits credential hallucination through custom system instructions.
4. **GDPR Right-to-Erasure:** Complete local SQLite persistence with one-click data purge and zero external telemetry.

---

### 7. PROJECT DETAILS
- **Project Title:** SmartResume AI — Intelligent Resume NLP Platform
- **Course / Lab:** Natural Language Processing (NLP) Laboratory Examination
- **Platform Version:** v2.4 (Production Ready)
- **Repository:** Clean, modular, fully tested GitHub-ready codebase (19/19 passing automated tests).

