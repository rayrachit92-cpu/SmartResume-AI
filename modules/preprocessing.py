"""
SmartResume AI - Text Preprocessing Pipeline
Provides non-destructive and normalized text processing, tokenization,
lemmatization, stopword removal, sentence segmentation, and frequency analysis.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Any, Tuple
import nltk
from config.settings import DATA_DIR

local_nltk = DATA_DIR / "nltk_data"
if local_nltk.exists() and str(local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(local_nltk))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK components
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

lemmatizer = WordNetLemmatizer()


class TextPreprocessor:
    """End-to-end NLP preprocessing pipeline while safely preserving original text."""

    @staticmethod
    def clean_text(text: str, remove_urls: bool = False, remove_emails: bool = False) -> str:
        """
        Cleans text by normalizing whitespace, standardizing unicode,
        and optionally scrubbing URLs/emails for pure linguistic processing.
        """
        if not text:
            return ""

        cleaned = text

        # Standardize quotes, dashes, and bullets
        cleaned = re.sub(r'[\u2018\u2019\u201A\u201B]', "'", cleaned)
        cleaned = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', cleaned)
        cleaned = re.sub(r'[\u2013\u2014]', '-', cleaned)
        cleaned = re.sub(r'[\u2022\u25E6\u2023\u25AA\u25AB\u2043]', ' • ', cleaned)

        if remove_urls:
            cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
        if remove_emails:
            cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', cleaned)

        # Replace excessive newlines and whitespace
        cleaned = re.sub(r'\r\n', '\n', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)

        return cleaned.strip()

    @staticmethod
    def segment_sentences(text: str) -> List[str]:
        """Segments text into sentences using NLTK."""
        if not text:
            return []
        try:
            return sent_tokenize(text)
        except Exception:
            return [s.strip() for s in re.split(r'[.!?]+(?:\s+|\n+)', text) if s.strip()]

    @staticmethod
    def tokenize(text: str, lowercase: bool = True) -> List[str]:
        """Tokenizes text into individual words/tokens."""
        if not text:
            return []
        raw_tokens = word_tokenize(text)
        if lowercase:
            return [t.lower() for t in raw_tokens]
        return raw_tokens

    @staticmethod
    def filter_and_lemmatize(tokens: List[str], remove_stop: bool = True,
                              remove_punct: bool = True) -> List[Dict[str, str]]:
        """
        Processes tokens with lemmatization, stopword removal, and punctuation filtering.
        Returns a rich list of dicts with token, lemma, and is_stop flags.
        """
        punct_set = set(string.punctuation).union({'•', '–', '—', '“', '”', '’', '‘'})
        processed = []

        for t in tokens:
            t_clean = t.strip()
            if not t_clean:
                continue

            is_p = t_clean in punct_set or all(c in string.punctuation for c in t_clean)
            is_s = t_clean.lower() in STOP_WORDS

            if remove_punct and is_p:
                continue
            if remove_stop and is_s:
                continue

            lemma = lemmatizer.lemmatize(t_clean.lower())
            processed.append({
                "token": t_clean,
                "lemma": lemma,
                "is_stop": is_s,
                "is_punct": is_p
            })

        return processed

    @staticmethod
    def get_word_frequencies(lemmas: List[str], top_n: int = 25) -> List[Tuple[str, int]]:
        """Returns the most frequent lemmas excluding short or single-char noise."""
        filtered = [l for l in lemmas if len(l) > 1 and l.isalpha()]
        counter = Counter(filtered)
        return counter.most_common(top_n)

    @classmethod
    def full_pipeline(cls, text: str) -> Dict[str, Any]:
        """
        Executes complete preprocessing pipeline and returns comprehensive structured metrics
        for interactive before-and-after UI comparison.
        """
        cleaned = cls.clean_text(text)
        sentences = cls.segment_sentences(cleaned)
        raw_tokens = cls.tokenize(cleaned, lowercase=False)
        lower_tokens = [t.lower() for t in raw_tokens]
        token_info = cls.filter_and_lemmatize(lower_tokens, remove_stop=True, remove_punct=True)
        lemmas = [item["lemma"] for item in token_info]
        top_words = cls.get_word_frequencies(lemmas, top_n=20)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "sentence_count": len(sentences),
            "sentences": sentences[:10],
            "raw_token_count": len(raw_tokens),
            "filtered_token_count": len(lemmas),
            "lemmas": lemmas,
            "token_details": token_info[:100],  # sample for UI preview
            "top_words": top_words,
            "vocabulary_size": len(set(lemmas)),
            "reduction_percentage": round((1 - (len(lemmas) / max(1, len(raw_tokens)))) * 100, 1)
        }
