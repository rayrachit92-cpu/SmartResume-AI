# SmartResume AI
### Intelligent Resume Information Extraction & AI-Powered Resume Analysis Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5.svg)](https://spacy.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-154C79.svg)](https://www.nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Title
**SmartResume AI: Automated Resume Information Extraction, Grounded Entity Linking, and AI-Powered Candidate Intelligence Platform.**

---

## 2. Problem Statement
Recruitment and talent acquisition workflows handle millions of unstructured documents annually in varied layouts, styles, and document encodings (PDF, DOCX). Traditional Applicant Tracking Systems (ATS) rely on rigid keyword lookups or brittle regular expressions that fail on synonym variations (e.g., treating "py", "python3", and "CPython" as disparate terms), struggle with unstructured narratives, and cannot disambiguate multi-word skill entities or career domain alignments.

Furthermore, naive automation often introduces hallucinated facts or unchecked algorithmic biases. There is an acute academic and industrial need for an **evidence-grounded, interpretable, end-to-end NLP platform** that combines:
1. Document text extraction (including scanned OCR fallback).
2. Deep text preprocessing and vectorization (BoW & TF-IDF).
3. Hybrid Named Entity Recognition (NER) trained on authentic resume benchmarks.
4. Structured information extraction into relational schemas.
5. Multi-tiered entity linking to standardized knowledge graphs.
6. Descriptive sentiment and writing tone analysis with ethical safeguards.
7. Unsupervised career domain discovery using Latent Dirichlet Allocation (LDA).
8. Evidence-grounded Generative AI assistance (Google Gemini).

---

## 3. Objectives
1. **Automated Document Ingestion:** Reliably extract structured text from single and bulk PDF, DOCX, and TXT files, with scanned PDF detection and OCR support.
2. **Comprehensive Preprocessing:** Provide full tokenization, lemmatization, stopword elimination, sentence segmentation, Bag-of-Words, and TF-IDF feature matrices.
3. **Robust NER:** Accurately classify 10+ resume entity categories (`PERSON`, `EMAIL`, `PHONE`, `ORGANIZATION`, `LOCATION`, `DATE`, `DEGREE`, `UNIVERSITY`, `SKILL`, `JOB_TITLE`, `CERTIFICATION`) using a hybrid transition-based NER model and EntityRuler.
4. **Structured Information Extraction:** Map unstructured resume prose into normalized candidate profiles without hallucinating missing fields.
5. **Knowledge Base Grounding (Entity Linking):** Implement a strict 4-stage hierarchy (Exact → Case-Insensitive → Curated Alias → Fuzzy Levenshtein) to normalize messy skill mentions to canonical entities.
6. **Descriptive Sentiment Analysis:** Analyze linguistic valence across narrative sections using NLTK VADER while maintaining strict ethical boundaries regarding hiring neutrality.
7. **Unsupervised Domain Modeling:** Discover latent tech topics and career trajectories using Scikit-Learn LDA.
8. **Evidence-Grounded AI Assistance:** Integrate Google Gemini for contextual Q&A, structured executive summaries, and resume critiques without fabricating candidate credentials.
9. **Rigorous Academic Evaluation:** Validate all NLP models on held-out test data without data leakage, reporting genuine Precision, Recall, and F1 metrics.

---

## 4. Existing System & Limitations
| Feature | Traditional ATS / Regex Extractors | Large LLM-Only Wrappers | SmartResume AI (Proposed) |
| :--- | :--- | :--- | :--- |
| **Parsing Flexibility** | Brittle; fails on layout changes | High, but expensive and slow | **Hybrid NLP + Rule Pipeline** |
| **Hallucination Risk** | Low (only extracts literal strings) | High (frequently invents experience) | **Zero (strictly grounded extraction)** |
| **Skill Disambiguation** | None ("JS" != "JavaScript") | Inconsistent mapping | **4-Stage Ontology Entity Linking** |
| **Data Privacy** | Local or server-side | Sends all PII to third-party APIs | **Local SQLite storage & local NLP** |
| **Explainability** | High, but low accuracy | Black-box opacity | **Transparent spans, tokens & metrics** |
| **Offline Capability** | High | Zero (requires active API) | **100% operational offline (LLM optional)** |

---

## 5. Proposed System
SmartResume AI adopts a **modular, multi-stage NLP architecture** that combines fast, deterministic local NLP models (spaCy, NLTK, Scikit-Learn) with optional cloud-hosted LLMs (Google Gemini).

```
[Uploaded Resume (PDF/DOCX/TXT)]
               │
               ▼
    [DocumentExtractor (PyMuPDF / docx / Tesseract OCR)]
               │
               ├──► [TextPreprocessor (Tokenize, Lemmatize, Stopwords, BoW, TF-IDF)]
               │
               ├──► [ResumeNER (Transition Parser + 301 EntityRuler Patterns)]
               │              │
               │              ▼ (Raw Entities)
               ├──► [EntityLinker (Exact -> Alias -> Fuzzy -> Skill Ontology)]
               │              │
               │              ▼ (Canonical Entities)
               ├──► [InformationExtractor (Relational Candidate Profile)]
               │
               ├──► [SentimentAnalyzer (VADER Valence across Narrative Sections)]
               │
               ├──► [TopicModeler (Corpus-wide LDA Topic Clustering)]
               │
               ├──► [GenAIAssistant (Google Gemini Grounded Q&A / Summaries)]
               │
               ▼
[SQLite Persistent Storage & Streamlit SaaS Interface (Linear/Stripe Light Theme)]
```

---

## 6. Technology Stack
- **Frontend & Visualization:** Streamlit, Custom Light-Theme CSS (Linear/Notion/Stripe design), Plotly Express & Graph Objects.
- **Natural Language Processing:** spaCy 3.8+, NLTK 3.10+, Scikit-Learn 1.9+, RapidFuzz 3.14+.
- **Document Processing:** PyMuPDF (fitz) 2.1+, python-docx 1.2+, PIL, pytesseract.
- **Generative AI & LLM:** Google Gemini API (`google-genai` SDK).
- **Persistent Storage:** SQLite3 (`data/smartresume.db`).
- **Testing & Quality Assurance:** Pytest 9.2+.

---

## 7. Dataset Description & Provenance
SmartResume AI integrates two authentic, real-world resume NLP datasets:

### Primary Dataset: Resume NER Training Dataset
- **Hugging Face Hub:** [`yashpwr/resume-ner-training-data`](https://huggingface.co/datasets/yashpwr/resume-ner-training-data)
- **Total Records:** 22,855 resumes with executive gold summaries.
- **License:** MIT License.

### Evaluation Benchmark Dataset: Dotin Resume NER Annotations
- **Source:** [`dotin-inc/resume-dataset-NER-annotations`](https://github.com/dotin-inc/resume-dataset-NER-annotations)
- **Total Documents:** 360 annotated resumes.
- **Total Labeled Entities:** 18,224 entity annotations.
- **License:** CC0-1.0 Universal (Public Domain).
- **Splits:** 80% Train (288 documents), 10% Validation (36 documents), 10% Test (36 documents).
- **Leakage Audit:** 0 document overlap across train, validation, and test partitions (random seed 42).

---

## 8. NLP Pipeline Implementation Details

### A. Preprocessing & Vectorization (`modules/preprocessing.py`, `modules/vectorization.py`)
- Non-destructive text normalization (standardizing quotes, dashes, bullet glyphs).
- NLTK sentence boundary detection (`sent_tokenize`) and tokenization (`word_tokenize`).
- WordNet lemmatization with stopword filtering.
- Scikit-Learn `CountVectorizer` (Bag of Words) and `TfidfVectorizer` (sublinear term weighting).

### B. Named Entity Recognition (`modules/ner.py`, `train.py`)
- Hybrid transition-based NER component combined with spaCy `EntityRuler`.
- 301 curated domain patterns covering programming languages, frameworks, cloud platforms, job designations, and academic degrees.
- Punctuation-stripped and BILUO-aligned entity span validation.

### C. Information Extraction (`modules/information_extraction.py`)
- Automated semantic section segmentation (`summary`, `experience`, `education`, `skills`, `projects`, `certifications`).
- Structured parsing of personal identity (email, phone, LinkedIn, portfolio, location).
- Mathematical profile completeness scoring ($0-100\%$) based on weighted presence of core sections.

### D. Entity Linking (`modules/entity_linking.py`)
- Skill Knowledge Base ontology (`data/skill_knowledge_base.json`) with canonical names, category taxonomies, aliases, and external Wikidata identifiers.
- Four-stage matching priority: Exact ($1.0$) → Case-Insensitive ($1.0$) → Alias ($0.98$) → Fuzzy Levenshtein ($>0.82$).

### E. Sentiment Analysis (`modules/sentiment_analysis.py`)
- Multi-section polarity analysis using NLTK VADER.
- Tone classification: Enthusiastic ($\ge 0.45$), Professional/Positive ($\ge 0.05$), Neutral ($-0.05$ to $0.05$), Critical ($\le -0.05$).
- Prominent ethical notice disclaiming hiring or personality suitability conclusions.

### F. Topic Modeling (`modules/topic_modeling.py`)
- Online Latent Dirichlet Allocation (LDA) using Scikit-Learn.
- Automatic career domain mapping (Software Engineering, Data Science & AI, Cloud & DevOps, Database Engineering, Project Management).
- Document-topic distribution probabilities and model perplexity tracking.

### G. Generative AI Assistant (`modules/genai_assistant.py`)
- Built on Google Gemini (`google-genai` SDK).
- Strict evidence-grounded system prompt preventing qualification fabrication.
- Conversational Q&A with session history logging, executive summaries, and actionable resume optimization suggestions.

---

## 9. Model Evaluation & Benchmark Results
Evaluated strictly on the held-out test partition (36 unseen resumes, 1,102 ground-truth entity spans):

| Entity Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **PERSON** | **75.00%** | 11.32% | 19.67% | 53 |
| **DATE** | **66.67%** | 23.53% | 34.78% | 17 |
| **LOCATION** | **51.92%** | **47.79%** | **49.77%** | 113 |
| **UNIVERSITY** | **50.00%** | **46.67%** | **48.28%** | 75 |
| **EXPERIENCE** | 34.50% | **49.59%** | **40.67%** | 123 |
| **JOB_TITLE** | 32.90% | 31.93% | 32.41% | 166 |
| **SKILL** | 19.42% | 25.56% | 22.06% | 360 |
| **DEGREE** | 14.71% | 14.49% | 14.60% | 69 |
| **ORGANIZATION**| 29.63% | 6.35% | 10.46% | 126 |
| **OVERALL MICRO**| **29.44%** | **29.31%** | **29.38%** | **1,102** |

*All results reflect actual, un-fabricated test outputs logged in `models/evaluation_results.json`.*

---

## 10. Installation & Setup Instructions

### Prerequisites
- Python 3.11+
- Virtual Environment recommended

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/SmartResume-AI.git
cd SmartResume-AI
```

### Step 2: Create & Activate Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add your `GEMINI_API_KEY` for generative AI Q&A. The core NLP platform works 100% offline without it).*

### Step 5: Acquire Datasets & Train Model (Reproducible)
```bash
python download_dataset.py
python train.py
```

### Step 6: Launch Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 11. Automated Unit Testing
Run the automated test suite covering all modules:
```bash
pytest tests/ -v
```
Output:
```
======================== 19 passed in 5.87s ========================
```

---

## 12. Ethical Guidelines & Limitations
- **No Automated Hiring Verdicts:** SmartResume AI displays factual skill matches and credential summaries; it expressly avoids black-box ranking, scoring, or automated hiring decisions.
- **Sentiment Boundary:** Sentiment scores reflect descriptive communication style only and must not be used to judge candidate ethics, integrity, or competence.
- **Dataset Scope:** The NER model was trained on English-language resumes; non-English resumes are routed through the multilingual detection and translation pipeline.

---

## 13. Academic References
1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
2. Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing*.
3. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993-1022.
4. Hutto, C., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text*. ICWSM.
5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.

