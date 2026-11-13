"""
SmartResume AI - Candidate Comparison Module
Performs multi-candidate side-by-side analysis:
- Skills Venn/Overlap & Unique Skill Breakdown
- Education & Degree Level Comparison
- Experience & Career Trajectory
- Topic Domain Alignment
- Objective, criteria-driven comparison without biased automated ranking
"""

from typing import Dict, List, Any, Set
import pandas as pd


class CandidateComparator:
    """Compares multiple candidate profiles across objective NLP metrics."""

    @staticmethod
    def compare_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a list of candidate dictionaries (from DB or extraction)
        and computes overlap, unique attributes, and alignment metrics.
        """
        if not candidates or len(candidates) < 2:
            return {
                "status": "insufficient_data",
                "message": "Please select at least 2 candidates to compare."
            }

        candidate_names = []
        all_skills_per_candidate: Dict[str, Set[str]] = {}
        all_skill_list_per_candidate: Dict[str, List[str]] = {}

        summary_rows = []

        for idx, c in enumerate(candidates):
            name = c.get("candidate_name") or f"Candidate {c.get('resume_id', idx + 1)}"
            candidate_names.append(name)

            # Flatten skills
            raw_skills = c.get("skills", {})
            flat_skills = set()
            if isinstance(raw_skills, dict):
                for cat, sk_list in raw_skills.items():
                    if isinstance(sk_list, list):
                        flat_skills.update(sk_list)
            elif isinstance(raw_skills, list):
                flat_skills.update(raw_skills)

            all_skills_per_candidate[name] = flat_skills
            all_skill_list_per_candidate[name] = sorted(list(flat_skills))

            # Degree summary
            edu = c.get("education", [])
            degree_str = ", ".join([e.get("degree", "") for e in edu if e.get("degree")]) or "Not specified"

            # Experience count
            exp = c.get("experience", [])

            summary_rows.append({
                "Candidate": name,
                "Email": c.get("email") or "N/A",
                "Location": c.get("location") or "N/A",
                "Education": degree_str,
                "Total Skills Extracted": len(flat_skills),
                "Experience Entries": len(exp),
                "Projects Count": len(c.get("projects", [])),
                "Certifications Count": len(c.get("certifications", [])),
                "Profile Completeness": f"{c.get('completeness_score', 0.0)}%"
            })

        # Calculate Common (Shared) Skills across ALL selected candidates
        common_skills = set.intersection(*all_skills_per_candidate.values()) if all_skills_per_candidate else set()

        # Calculate Unique Skills per candidate (skills possessed only by this candidate)
        unique_skills = {}
        for name, skills in all_skills_per_candidate.items():
            other_skills = set.union(*[s for n, s in all_skills_per_candidate.items() if n != name])
            unique_skills[name] = sorted(list(skills - other_skills))

        # Skill matrix dataframe for tabular visualization
        all_unique_skills_universe = sorted(list(set.union(*all_skills_per_candidate.values())))
        skill_matrix_data = []
        for sk in all_unique_skills_universe:
            row = {"Skill": sk}
            for name in candidate_names:
                row[name] = "✓" if sk in all_skills_per_candidate[name] else "—"
            skill_matrix_data.append(row)

        skill_matrix_df = pd.DataFrame(skill_matrix_data)
        summary_df = pd.DataFrame(summary_rows)

        return {
            "status": "success",
            "candidate_names": candidate_names,
            "summary_df": summary_df,
            "skill_matrix_df": skill_matrix_df,
            "common_skills": sorted(list(common_skills)),
            "unique_skills": unique_skills,
            "all_skills_per_candidate": all_skill_list_per_candidate,
            "total_skills_universe_count": len(all_unique_skills_universe),
            "disclaimer": "Transparency Note: This comparison presents objective extracted facts. It intentionally avoids black-box ranking or automated hiring verdicts."
        }

