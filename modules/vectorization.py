"""
SmartResume AI - Vectorization Module
Provides Bag-of-Words (CountVectorizer) and TF-IDF representations,
feature matrix inspection, and vocabulary analysis.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeVectorizer:
    """Computes BoW and TF-IDF matrices for single or multiple resumes."""

    @staticmethod
    def compute_bow(texts: List[str], max_features: int = 50, ngram_range: Tuple[int, int] = (1, 2)) -> Dict[str, Any]:
        """Calculates Bag of Words (term counts)."""
        if not texts or all(not t.strip() for t in texts):
            return {"vocabulary": [], "matrix": [], "dataframe": pd.DataFrame()}

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )

        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame(matrix.toarray(), columns=feature_names)

        # Total counts across documents
        total_counts = df.sum(axis=0).sort_values(ascending=False).to_dict()

        return {
            "feature_names": list(feature_names),
            "matrix": matrix.toarray().tolist(),
            "total_counts": total_counts,
            "dataframe": df,
            "vocabulary_size": len(feature_names)
        }

    @staticmethod
    def compute_tfidf(texts: List[str], max_features: int = 50, ngram_range: Tuple[int, int] = (1, 2)) -> Dict[str, Any]:
        """Calculates TF-IDF (Term Frequency - Inverse Document Frequency)."""
        if not texts or all(not t.strip() for t in texts):
            return {"vocabulary": [], "matrix": [], "dataframe": pd.DataFrame(), "top_terms": []}

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame(matrix.toarray(), columns=feature_names)

        # Mean TF-IDF score per term across documents
        mean_scores = df.mean(axis=0).sort_values(ascending=False).to_dict()
        top_terms = [{"term": term, "tfidf_score": round(score, 4)} for term, score in mean_scores.items()][:20]

        return {
            "feature_names": list(feature_names),
            "matrix": matrix.toarray().tolist(),
            "mean_scores": mean_scores,
            "top_terms": top_terms,
            "dataframe": df,
            "vocabulary_size": len(feature_names)
        }

    @staticmethod
    def compute_similarity(texts: List[str]) -> pd.DataFrame:
        """Calculates pairwise cosine similarity matrix between documents."""
        if len(texts) < 2:
            return pd.DataFrame([[1.0]], columns=["Doc 1"], index=["Doc 1"])

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)

        labels = [f"Resume {i+1}" for i in range(len(texts))]
        return pd.DataFrame(sim_matrix, index=labels, columns=labels)

