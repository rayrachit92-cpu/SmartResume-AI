"""
SmartResume AI - Report Generation & Data Export Module
Exports comprehensive extracted candidate profiles and batch analytics in:
1. Structured JSON
2. Tabular CSV
3. Plain Text (Executive Brief)
4. Formatted Markdown / HTML reports
"""

import json
import io
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime


class ReportGenerator:
    """Generates structured exports and download payloads."""

    @staticmethod
    def generate_json_export(profile: Dict[str, Any], include_entities: bool = True) -> str:
        """Serializes candidate profile into clean, indented JSON."""
        data = dict(profile)
        if not include_entities and "entities" in data:
            del data["entities"]
        data["exported_at"] = datetime.now().isoformat()
        data["platform"] = "SmartResume AI"
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_csv_export(profiles: List[Dict[str, Any]]) -> str:
        """Flattens a list of candidate profiles into CSV format."""
        rows = []
        for p in profiles:
            # Flatten skills
            raw_skills = p.get("skills", {})
            flat_skills = []
            if isinstance(raw_skills, dict):
                for cat, sks in raw_skills.items():
                    if isinstance(sks, list):
                        flat_skills.extend(sks)
            elif isinstance(raw_skills, list):
                flat_skills = raw_skills

            edu_list = p.get("education", [])
            edu_str = " | ".join([f"{e.get('degree', '')} ({e.get('institution', '')})" for e in edu_list if e.get('degree')])

            rows.append({
                "Candidate Name": p.get("candidate_name", ""),
                "Email": p.get("email", ""),
                "Phone": p.get("phone", ""),
                "Location": p.get("location", ""),
                "LinkedIn": p.get("linkedin_url", ""),
                "Portfolio": p.get("portfolio_url", ""),
                "Completeness Score (%)": p.get("completeness_score", 0.0),
                "Education": edu_str,
                "Skills": ", ".join(sorted(list(set(flat_skills)))),
                "Total Skills Count": len(set(flat_skills)),
                "Experience Count": len(p.get("experience", [])),
                "Summary": p.get("summary", "")[:200]
            })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    @staticmethod
    def generate_txt_brief(profile: Dict[str, Any]) -> str:
        """Generates a formatted plain-text executive summary dossier."""
        name = profile.get("candidate_name") or "Candidate Profile"
        email = profile.get("email") or "Not provided"
        phone = profile.get("phone") or "Not provided"
        loc = profile.get("location") or "Not provided"
        completeness = profile.get("completeness_score", 0.0)

        lines = [
            "=" * 70,
            f"SMARTRESUME AI - CANDIDATE EXECUTIVE DOSSIER",
            "=" * 70,
            f"Candidate Name  : {name}",
            f"Email           : {email}",
            f"Phone           : {phone}",
            f"Location        : {loc}",
            f"Completeness    : {completeness}%",
            f"Export Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 70,
            "PROFESSIONAL SUMMARY:",
            profile.get("summary") or "No summary provided.",
            "-" * 70,
            "CATEGORIZED SKILLS:"
        ]

        skills = profile.get("skills", {})
        if isinstance(skills, dict):
            for cat, sk_list in skills.items():
                if sk_list:
                    lines.append(f"  • {cat}: {', '.join(sk_list)}")
        elif isinstance(skills, list):
            lines.append(f"  • Skills: {', '.join(skills)}")

        lines.extend([
            "-" * 70,
            "EDUCATION:"
        ])
        for edu in profile.get("education", []):
            deg = edu.get("degree", "Degree")
            inst = edu.get("institution", "")
            yr = edu.get("graduation_year", "")
            lines.append(f"  • {deg}{' - ' + inst if inst else ''}{' (' + yr + ')' if yr else ''}")

        lines.extend([
            "-" * 70,
            "EXPERIENCE:"
        ])
        for exp in profile.get("experience", []):
            jt = exp.get("job_title", "Position")
            co = exp.get("company", "")
            st = exp.get("start_date", "")
            en = exp.get("end_date", "")
            lines.append(f"  • {jt}{' at ' + co if co else ''} [{st} - {en}]")
            for resp in exp.get("responsibilities", []):
                lines.append(f"      - {resp}")

        lines.extend([
            "-" * 70,
            "PROJECTS:"
        ])
        for p in profile.get("projects", []):
            lines.append(f"  • {p.get('name', 'Project')}: {p.get('description', '')}")
            if p.get("technologies"):
                lines.append(f"    Tech: {', '.join(p['technologies'])}")

        lines.extend([
            "=" * 70,
            "Generated by SmartResume AI Platform",
            "=" * 70
        ])

        return "\n".join(lines)

