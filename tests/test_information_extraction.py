"""
Unit tests for InformationExtractor (Profile parsing, section detection, completeness).
"""

import pytest
from modules.information_extraction import InformationExtractor


@pytest.fixture(scope="module")
def extractor():
    return InformationExtractor()


def test_section_detection(extractor):
    resume_text = """
John Doe
New York, NY

PROFESSIONAL SUMMARY
Experienced backend engineer with 5 years building REST APIs.

EXPERIENCE
Senior Backend Developer | Acme Corp
Jan 2020 - Present
• Designed microservices with FastAPI and PostgreSQL.

EDUCATION
Bachelor of Science in Computer Science | Columbia University (2019)

SKILLS
Python, FastAPI, Docker, SQL
"""
    sections = extractor.detect_sections(resume_text)
    assert "summary" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "skills" in sections
    assert "Acme Corp" in sections["experience"]


def test_full_profile_extraction(extractor):
    resume_text = """
Jane Smith
Austin, TX | jane.smith@example.com | (512) 555-0123
LinkedIn: linkedin.com/in/janesmith

PROFESSIONAL SUMMARY
Frontend engineer specializing in React and Next.js.

EDUCATION
Master of Science in Software Engineering | University of Texas (2021)

SKILLS
React, JavaScript, TypeScript, Tailwind CSS

EXPERIENCE
Frontend Developer | TechFlow Inc
2021 - Present
• Implemented responsive design and reduced bundle size by 30%.
"""
    profile = extractor.extract_full_profile(resume_text)
    assert profile["candidate_name"] == "Jane Smith"
    assert profile["email"] == "jane.smith@example.com"
    assert profile["phone"] == "(512) 555-0123"
    assert profile["location"] == "Austin, TX"
    assert len(profile["education"]) > 0
    assert profile["completeness_score"] > 50.0

