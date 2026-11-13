"""
Unit tests for EntityLinker (Exact, Alias, Case-insensitive, Fuzzy matching).
"""

import pytest
from modules.entity_linking import EntityLinker


@pytest.fixture(scope="module")
def linker():
    return EntityLinker()


def test_exact_match(linker):
    res = linker.link_term("Python")
    assert res["canonical_name"] == "Python"
    assert res["match_method"] == "Exact"
    assert res["similarity_score"] == 1.0


def test_alias_match(linker):
    res = linker.link_term("py")
    assert res["canonical_name"] == "Python"
    assert res["match_method"] == "Alias"

    res_k8s = linker.link_term("k8s")
    assert res_k8s["canonical_name"] == "Kubernetes"
    assert res_k8s["match_method"] == "Alias"


def test_case_insensitive_match(linker):
    res = linker.link_term("react")
    assert res["canonical_name"] == "React"
    assert res["match_method"] == "Case-Insensitive"


def test_fuzzy_match(linker):
    res = linker.link_term("PostgreSql")
    assert res["canonical_name"] == "PostgreSQL"
    assert res["match_method"] in ["Exact", "Case-Insensitive", "Fuzzy"]


def test_unmatched_term(linker):
    res = linker.link_term("CompletelyUnknownFictionalSkill123")
    assert res["match_method"] == "Unmatched"
    assert res["similarity_score"] == 0.0

