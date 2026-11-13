"""
SmartResume AI - Sentiment & Tone Analysis Module
Analyzes the linguistic tone of professional descriptions (Summary, Experience, Projects)
using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).

IMPORTANT ACADEMIC & ETHICAL NOTE:
Sentiment scores measure descriptive vocabulary valence and communication style.
They MUST NOT be used or interpreted as a metric of candidate competence, personality,
trustworthiness, integrity, or hiring suitability.
"""

import os
from typing import Dict, List, Any
import nltk
from config.settings import DATA_DIR

local_nltk = DATA_DIR / "nltk_data"
if local_nltk.exists() and str(local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(local_nltk))

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER analyzer
try:
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon', download_dir=str(local_nltk), quiet=True)
    analyzer = SentimentIntensityAnalyzer()


class ResumeSentimentAnalyzer:
    """Analyzes tone valence of narrative resume sections."""

    @staticmethod
    def classify_tone(compound: float) -> str:
        """Classifies tone category based on standard VADER compound thresholds."""
        if compound >= 0.45:
            return "Highly Positive / Enthusiastic"
        elif compound >= 0.05:
            return "Positive / Professional"
        elif compound <= -0.05:
            return "Critical / Problem-Oriented"
        else:
            return "Neutral / Objective"

    @classmethod
    def analyze_text(cls, text: str, section_name: str = "General") -> Dict[str, Any]:
        """Calculates VADER positive, neutral, negative, and compound polarity scores."""
        clean_text = text.strip() if text else ""
        if not clean_text or len(clean_text) < 15:
            return {
                "section_name": section_name,
                "pos": 0.0,
                "neu": 1.0,
                "neg": 0.0,
                "compound": 0.0,
                "tone_label": "Neutral / Objective",
                "word_count": 0,
                "text_snippet": ""
            }

        scores = analyzer.polarity_scores(clean_text)
        tone = cls.classify_tone(scores["compound"])

        return {
            "section_name": section_name,
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
            "compound": round(scores["compound"], 4),
            "tone_label": tone,
            "word_count": len(clean_text.split()),
            "text_snippet": clean_text[:250] + ("..." if len(clean_text) > 250 else "")
        }

    @classmethod
    def analyze_resume_sections(cls, sections: Dict[str, str]) -> Dict[str, Any]:
        """
        Runs sentiment analysis across semantic narrative sections:
        Summary, Experience descriptions, and Projects.
        """
        targets = [
            ("Professional Summary", sections.get("summary", "")),
            ("Experience Narrative", sections.get("experience", "")),
            ("Projects Narrative", sections.get("projects", ""))
        ]

        section_results = []
        overall_compounds = []

        for name, content in targets:
            if content and len(content.strip()) > 20:
                res = cls.analyze_text(content, section_name=name)
                section_results.append(res)
                overall_compounds.append(res["compound"])

        # Fallback if no specific sections found
        if not section_results and sections.get("header"):
            fallback = cls.analyze_text(sections["header"][:600], section_name="Overview")
            section_results.append(fallback)
            overall_compounds.append(fallback["compound"])

        avg_compound = (sum(overall_compounds) / len(overall_compounds)) if overall_compounds else 0.0
        overall_tone = cls.classify_tone(avg_compound)

        return {
            "overall_compound": round(avg_compound, 4),
            "overall_tone": overall_tone,
            "section_results": section_results,
            "ethical_disclaimer": (
                "Ethical Notice: Tone analysis measures linguistic presentation and descriptive vocabulary only. "
                "It does not evaluate candidate competence, personality traits, ethics, or suitability for any role."
            )
        }
