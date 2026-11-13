"""
SmartResume AI - Topic Modeling Module
Discovers latent career and technological themes across resumes using
Latent Dirichlet Allocation (LDA) with Scikit-learn.
Includes topic keyword extraction, document assignment, and corpus-level distributions.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Industry domain seed labels for semantic mapping
DOMAIN_KEYWORDS = {
    "Software & Web Engineering": ["software", "web", "application", "developer", "backend", "frontend", "javascript", "react", "api", "code"],
    "Data Science & AI / ML": ["data", "model", "machine", "learning", "python", "neural", "analytics", "deep", "nlp", "prediction"],
    "Cloud & DevOps Architecture": ["cloud", "aws", "azure", "ci", "cd", "docker", "kubernetes", "infrastructure", "devops", "pipeline"],
    "Database & Big Data Engineering": ["database", "sql", "spark", "kafka", "etl", "warehouse", "postgres", "queries", "hadoop", "migration"],
    "Enterprise & Project Management": ["management", "project", "business", "agile", "scrum", "lead", "team", "client", "strategy", "delivery"],
    "Cybersecurity & Infrastructure": ["security", "network", "firewall", "vulnerability", "risk", "audit", "compliance", "threat", "protocols"],
    "Mobile App Development": ["android", "ios", "swift", "kotlin", "flutter", "mobile", "react", "native", "apps", "store"]
}


class ResumeTopicModeler:
    """Discovers career clusters and technology domains using Latent Dirichlet Allocation."""

    def __init__(self, n_topics: int = 5, max_iter: int = 25, random_state: int = 42):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = []

    def fit_transform(self, documents: List[str], n_words_per_topic: int = 10) -> Dict[str, Any]:
        """
        Fits LDA topic model across a corpus of resumes.
        Handles small corpora gracefully with dynamic adjustment.
        """
        valid_docs = [d.strip() for d in documents if d and len(d.strip()) > 30]
        n_docs = len(valid_docs)

        if n_docs < 2:
            return {
                "status": "insufficient_data",
                "message": f"Topic modeling requires at least 2 distinct documents. Current count: {n_docs}.",
                "topics": [],
                "document_topics": []
            }

        # Dynamically bound number of topics by corpus size
        effective_n_topics = min(self.n_topics, max(2, n_docs // 2))

        # Vectorize using term counts with stopword removal
        self.vectorizer = CountVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            token_pattern=r'\b[A-Za-z][A-Za-z0-9+#]{1,25}\b'
        )

        dtm = self.vectorizer.fit_transform(valid_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()

        self.lda_model = LatentDirichletAllocation(
            n_components=effective_n_topics,
            max_iter=self.max_iter,
            learning_method='online',
            random_state=self.random_state,
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )

        doc_topic_matrix = self.lda_model.fit_transform(dtm)

        # Extract top keywords and identify domain label
        topics_info = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[:-n_words_per_topic - 1:-1]
            top_keywords = [self.feature_names[i] for i in top_word_indices]
            weights = [round(float(topic[i]), 2) for i in top_word_indices]

            # Label mapping based on overlap with domain keywords
            assigned_label = f"Topic {topic_idx + 1}"
            best_overlap = 0
            for domain, kws in DOMAIN_KEYWORDS.items():
                overlap = len(set(top_keywords).intersection(set(kws)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_label = domain

            topics_info.append({
                "topic_id": topic_idx,
                "label": assigned_label,
                "top_keywords": top_keywords,
                "weights": weights,
                "coherence_indicator": round(float(topic[top_word_indices[0]] / max(1.0, topic.sum())), 3)
            })

        # Document topic distributions
        doc_results = []
        for doc_idx, distribution in enumerate(doc_topic_matrix):
            dominant_id = int(np.argmax(distribution))
            dominant_prob = float(distribution[dominant_id])
            dominant_label = topics_info[dominant_id]["label"]

            dist_dict = {f"Topic {i+1} ({topics_info[i]['label']})": round(float(p), 4) for i, p in enumerate(distribution)}

            doc_results.append({
                "document_index": doc_idx,
                "dominant_topic_id": dominant_id,
                "dominant_topic_label": dominant_label,
                "confidence": round(dominant_prob, 3),
                "distribution": dist_dict
            })

        # Calculate resume count per topic
        counts = [0] * effective_n_topics
        for r in doc_results:
            counts[r["dominant_topic_id"]] += 1

        for i, t in enumerate(topics_info):
            t["assigned_document_count"] = counts[i]

        return {
            "status": "success",
            "n_topics": effective_n_topics,
            "total_documents": n_docs,
            "topics": topics_info,
            "document_topics": doc_results,
            "perplexity": round(float(self.lda_model.perplexity(dtm)), 2)
        }

