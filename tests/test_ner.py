"""
Unit tests for ResumeNER (Entity recognition, contact regex, span validation).
"""

import pytest
from modules.ner import ResumeNER


@pytest.fixture(scope="module")
def ner_engine():
    return ResumeNER()


def test_contact_entities(ner_engine):
    text = "Candidate: Alex Morgan, Contact: alex.morgan@example.com, Phone: (415) 555-0192"
    entities = ner_engine.extract_entities(text)
    labels = {e["label"] for e in entities}
    assert "EMAIL" in labels
    assert "PHONE" in labels

    email_ent = next(e for e in entities if e["label"] == "EMAIL")
    assert email_ent["text"] == "alex.morgan@example.com"


def test_skill_entities(ner_engine):
    text = "Proficient in Python, React, PostgreSQL, Docker, and PyTorch."
    entities = ner_engine.extract_entities(text)
    skills = [e["text"] for e in entities if e["label"] == "SKILL"]
    assert "Python" in skills or "React" in skills or "PostgreSQL" in skills

