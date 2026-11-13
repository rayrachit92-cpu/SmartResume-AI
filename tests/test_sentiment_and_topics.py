"""
Unit tests for Sentiment Analysis (VADER) and Topic Modeling (LDA).
"""

import pytest
from modules.sentiment_analysis import ResumeSentimentAnalyzer
from modules.topic_modeling import ResumeTopicModeler


def test_sentiment_analysis():
    positive_summary = "Proven track record of exceptional leadership, outstanding innovation, and delivering extraordinary value."
    res = ResumeSentimentAnalyzer.analyze_text(positive_summary, "Summary")
    assert res["compound"] > 0.05
    assert "Positive" in res["tone_label"]

    neutral_summary = "Managed daily inventory counts and scheduled weekly department shifts."
    res_neu = ResumeSentimentAnalyzer.analyze_text(neutral_summary, "Experience")
    assert -0.3 <= res_neu["compound"] <= 0.6


def test_topic_modeling():
    corpus = [
        "Python data science machine learning neural network PyTorch deep learning",
        "React frontend web development JavaScript TypeScript HTML CSS",
        "Kubernetes Docker AWS cloud infrastructure DevOps Terraform CI/CD",
        "SQL database PostgreSQL MySQL queries optimization relational database"
    ]
    modeler = ResumeTopicModeler(n_topics=2, max_iter=15)
    results = modeler.fit_transform(corpus)
    assert results["status"] == "success"
    assert len(results["topics"]) == 2
    assert len(results["document_topics"]) == 4

