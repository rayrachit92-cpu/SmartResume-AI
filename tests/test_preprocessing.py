"""
Unit tests for Text Preprocessing and Vectorization (BoW, TF-IDF).
"""

import pytest
from modules.preprocessing import TextPreprocessor
from modules.vectorization import ResumeVectorizer


def test_text_cleaning():
    raw = "Senior   Developer • Python / Django \n\n\n\r\nEmail: test@example.com"
    cleaned = TextPreprocessor.clean_text(raw)
    assert "•" in cleaned
    assert "\n\n\n" not in cleaned
    assert "  " not in cleaned


def test_tokenization_and_lemmatization():
    text = "The machine learning engineers are developing scalable neural models."
    tokens = TextPreprocessor.tokenize(text)
    assert "engineers" in tokens
    assert "developing" in tokens

    lemmas = TextPreprocessor.filter_and_lemmatize(tokens, remove_stop=True, remove_punct=True)
    lemma_words = [item["lemma"] for item in lemmas]
    assert "engineer" in lemma_words
    assert "model" in lemma_words
    assert "the" not in lemma_words  # stopword removed


def test_bow_vectorization():
    corpus = [
        "Python developer with Django and PostgreSQL experience",
        "Machine learning engineer with Python and PyTorch experience"
    ]
    bow = ResumeVectorizer.compute_bow(corpus, max_features=10)
    assert "feature_names" in bow
    assert len(bow["feature_names"]) > 0
    assert "matrix" in bow


def test_tfidf_vectorization():
    corpus = [
        "Kubernetes Docker cloud architect AWS",
        "React Node.js frontend web developer JavaScript"
    ]
    tfidf = ResumeVectorizer.compute_tfidf(corpus, max_features=10)
    assert "feature_names" in tfidf
    assert len(tfidf["top_terms"]) > 0
