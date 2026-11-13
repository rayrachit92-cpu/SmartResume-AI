"""
SmartResume AI - Multilingual NLP & Cloud Integration Module
Provides:
1. Automated language detection using langdetect.
2. Optional translation to English for international resumes.
3. Optional AWS Comprehend cloud entity detection connector.
"""

import os
import logging
from typing import Dict, List, Any, Optional

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 42
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

from config.settings import (
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
)
from modules.genai_assistant import GenAIAssistant

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "nl": "Dutch"
}


class MultilingualNLP:
    """Detects languages and routes non-English resumes through translation pipelines."""

    @staticmethod
    def detect_language(text: str) -> Dict[str, Any]:
        """Detects the primary language of the resume text."""
        clean = text.strip() if text else ""
        if not clean or len(clean) < 20 or not HAS_LANGDETECT:
            return {
                "lang_code": "en",
                "lang_name": "English",
                "confidence": 1.0,
                "is_english": True
            }

        try:
            detected = detect_langs(clean[:2000])
            primary = detected[0]
            code = primary.lang
            name = LANGUAGE_NAMES.get(code, code.upper())
            return {
                "lang_code": code,
                "lang_name": name,
                "confidence": round(float(primary.prob), 3),
                "is_english": (code == "en")
            }
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                "lang_code": "en",
                "lang_name": "English",
                "confidence": 1.0,
                "is_english": True
            }

    @staticmethod
    def translate_to_english(text: str, source_lang: str = "auto") -> Dict[str, Any]:
        """Translates non-English resume text to English using Gemini LLM if configured."""
        assistant = GenAIAssistant()
        if not assistant.is_configured():
            return {
                "success": False,
                "error": "Translation requires Gemini API key configured in .env or Settings.",
                "translated_text": text
            }

        prompt = f"""
Translate the following resume document accurately into professional English.
Preserve proper names, company names, job titles, and technical keywords.

SOURCE TEXT:
\"\"\"
{text}
\"\"\"
"""
        try:
            from google.genai import types
            config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=2500)
            res = assistant.client.models.generate_content(
                model=assistant.model_name,
                contents=prompt,
                config=config
            )
            return {
                "success": True,
                "translated_text": res.text,
                "source_lang": source_lang
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Translation failed: {e}",
                "translated_text": text
            }

    @staticmethod
    def check_aws_comprehend() -> Dict[str, Any]:
        """Verifies whether AWS Comprehend cloud service is configured."""
        is_configured = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and not AWS_ACCESS_KEY_ID.startswith("your_"))
        return {
            "is_configured": is_configured,
            "region": AWS_DEFAULT_REGION,
            "status_message": "AWS Comprehend Connected" if is_configured else "AWS Credentials Not Configured (Optional)"
        }

