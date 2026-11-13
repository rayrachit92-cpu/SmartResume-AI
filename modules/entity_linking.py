"""
SmartResume AI - Entity Linking & Normalization Engine
Maps raw extracted skill mentions and terms to standardized canonical entities
using a multi-stage disambiguation hierarchy:
1. Exact Match
2. Alias Match
3. Case-Insensitive Match
4. Fuzzy String Matching (Levenshtein / RapidFuzz)
5. Semantic Token Similarity
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from rapidfuzz import fuzz

from config.settings import SKILL_KB_PATH

logger = logging.getLogger(__name__)


class EntityLinker:
    """Links extracted skill terms to canonical knowledge base entities."""

    def __init__(self, kb_path: Path = SKILL_KB_PATH):
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()
        self._build_indexes()

    def _load_kb(self) -> Dict[str, Any]:
        """Loads canonical knowledge base from disk."""
        if self.kb_path.exists():
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load skill KB: {e}")
        return {"skills": []}

    def _build_indexes(self):
        """Builds in-memory lookup maps for O(1) matching."""
        self.exact_map = {}          # canonical name -> skill dict
        self.lower_map = {}          # lowercase canonical -> skill dict
        self.alias_map = {}          # lowercase alias -> skill dict
        self.all_canonical_names = []

        for skill in self.knowledge_base.get("skills", []):
            canon = skill["canonical_name"]
            self.exact_map[canon] = skill
            self.lower_map[canon.lower()] = skill
            self.all_canonical_names.append(canon)

            for alias in skill.get("aliases", []):
                self.alias_map[alias.lower()] = skill

    def link_term(self, term: str, fuzzy_threshold: float = 82.0) -> Dict[str, Any]:
        """
        Links a single candidate term using the strict priority hierarchy:
        1. Exact matching
        2. Alias matching
        3. Case-insensitive matching
        4. Fuzzy matching
        """
        raw_term = term.strip()
        if not raw_term:
            return self._build_unmatched_result(raw_term)

        # 1. Exact Match
        if raw_term in self.exact_map:
            ent = self.exact_map[raw_term]
            return {
                "extracted_term": raw_term,
                "canonical_name": ent["canonical_name"],
                "category": ent.get("category", "General"),
                "match_method": "Exact",
                "similarity_score": 1.0,
                "external_id": ent.get("external_id", ""),
                "status": "Linked"
            }

        # 2. Case-Insensitive Canonical Match
        lower_term = raw_term.lower()
        if lower_term in self.lower_map:
            ent = self.lower_map[lower_term]
            return {
                "extracted_term": raw_term,
                "canonical_name": ent["canonical_name"],
                "category": ent.get("category", "General"),
                "match_method": "Case-Insensitive",
                "similarity_score": 1.0,
                "external_id": ent.get("external_id", ""),
                "status": "Linked"
            }

        # 3. Alias Matching
        if lower_term in self.alias_map:
            ent = self.alias_map[lower_term]
            return {
                "extracted_term": raw_term,
                "canonical_name": ent["canonical_name"],
                "category": ent.get("category", "General"),
                "match_method": "Alias",
                "similarity_score": 0.98,
                "external_id": ent.get("external_id", ""),
                "status": "Linked"
            }

        # 4. Fuzzy Matching across canonical names and aliases
        best_match = None
        best_score = 0.0

        for canon_name, ent in self.exact_map.items():
            # Standard Levenshtein ratio
            score = fuzz.ratio(lower_term, canon_name.lower())

            if score > best_score:
                best_score = score
                best_match = ent

            # Also check aliases
            for alias in ent.get("aliases", []):
                alias_score = fuzz.ratio(lower_term, alias.lower())
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = ent

        if best_match and best_score >= fuzzy_threshold:
            return {
                "extracted_term": raw_term,
                "canonical_name": best_match["canonical_name"],
                "category": best_match.get("category", "General"),
                "match_method": "Fuzzy",
                "similarity_score": round(best_score / 100.0, 2),
                "external_id": best_match.get("external_id", ""),
                "status": "Linked"
            }

        return self._build_unmatched_result(raw_term)

    def _build_unmatched_result(self, raw_term: str) -> Dict[str, Any]:
        """Returns standard structure for terms with no confident knowledge-base link."""
        return {
            "extracted_term": raw_term,
            "canonical_name": raw_term,
            "category": "Uncategorized",
            "match_method": "Unmatched",
            "similarity_score": 0.0,
            "external_id": "",
            "status": "Unmatched"
        }

    def link_terms(self, terms: List[str], deduplicate: bool = True) -> List[Dict[str, Any]]:
        """Batch links a list of terms and optionally removes redundant duplicates."""
        results = []
        seen_canonical = set()

        for t in terms:
            linked = self.link_term(t)
            canon = linked["canonical_name"]

            if deduplicate:
                if canon.lower() in seen_canonical:
                    continue
                seen_canonical.add(canon.lower())

            results.append(linked)

        return results

    def add_custom_skill(self, canonical_name: str, category: str, aliases: List[str], external_id: str = ""):
        """Dynamically adds a new skill concept to the persistent knowledge base."""
        skill_id = f"SKILL-{len(self.knowledge_base.get('skills', [])) + 1:03d}"
        new_entry = {
            "id": skill_id,
            "canonical_name": canonical_name.strip(),
            "category": category.strip(),
            "aliases": [a.strip() for a in aliases if a.strip()],
            "related_concepts": [],
            "external_id": external_id.strip()
        }
        self.knowledge_base.setdefault("skills", []).append(new_entry)
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, indent=2)
        self._build_indexes()
