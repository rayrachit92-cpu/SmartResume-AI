"""
SmartResume AI - Information Extraction Module
Converts unstructured resume text into a structured, validated candidate profile:
- Personal Information (Name, Email, Phone, Location, LinkedIn, Portfolio)
- Education (Degree, Field, Institution, Year)
- Categorized Skills (Languages, Frameworks, Databases, Cloud, Tools, Soft Skills)
- Experience (Job Title, Company, Dates, Responsibilities)
- Projects (Name, Description, Technologies)
- Certifications (Name, Issuer, Date)
- Completeness Scoring & Structured JSON Export
"""

import re
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from modules.ner import ResumeNER
from config.settings import SKILL_KB_PATH


class InformationExtractor:
    """Extracts structured candidate profile from resume text using rules, NER, and section detection."""

    def __init__(self, ner_engine: Optional[ResumeNER] = None):
        self.ner = ner_engine or ResumeNER()
        self.skill_kb = self._load_skill_kb()

    def _load_skill_kb(self) -> Dict[str, Any]:
        """Loads canonical skills and aliases with categories."""
        if SKILL_KB_PATH.exists():
            try:
                with open(SKILL_KB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"skills": []}

    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Segments resume text into standard semantic sections:
        summary, experience, education, skills, projects, certifications.
        """
        section_headers = {
            "summary": [
                r'\b(?:professional\s+summary|executive\s+summary|career\s+objective|profile|about\s+me|summary)\b'
            ],
            "experience": [
                r'\b(?:work\s+experience|professional\s+experience|employment\s+history|experience|career\s+history)\b'
            ],
            "education": [
                r'\b(?:education|academic\s+background|educational\s+qualifications|academics|degrees)\b'
            ],
            "skills": [
                r'\b(?:technical\s+skills|skills\s+and\s+competencies|core\s+competencies|skills|technologies|expertise)\b'
            ],
            "projects": [
                r'\b(?:projects|key\s+projects|academic\s+projects|personal\s+projects)\b'
            ],
            "certifications": [
                r'\b(?:certifications|licenses|courses\s+and\s+certifications|credentials)\b'
            ]
        }

        # Build master regex of headers
        header_patterns = []
        for sec, patterns in section_headers.items():
            for p in patterns:
                header_patterns.append((sec, p))

        # Split text into lines
        lines = text.split("\n")
        current_section = "header"
        sections = {"header": [], "summary": [], "experience": [], "education": [], "skills": [], "projects": [], "certifications": []}

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if line looks like a section header (short, title-like, uppercase)
            is_header = False
            if len(line_str.split()) <= 4 and (line_str.isupper() or line_str.istitle() or line_str.endswith(":")):
                clean_line = line_str.rstrip(":").strip()
                for sec, pattern in header_patterns:
                    if re.search(pattern, clean_line, re.IGNORECASE):
                        current_section = sec
                        is_header = True
                        break

            if not is_header:
                if current_section in sections:
                    sections[current_section].append(line_str)
                else:
                    sections["header"].append(line_str)

        return {k: "\n".join(v) for k, v in sections.items()}

    def extract_personal_info(self, text: str, header_text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts contact and personal identity details."""
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', text)
        email = email_match.group() if email_match else ""

        # Phone
        phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        phone = phone_match.group() if phone_match else ""

        # LinkedIn
        linkedin_match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+', text, re.IGNORECASE)
        linkedin_url = linkedin_match.group() if linkedin_match else ""

        # Portfolio / GitHub
        portfolio_match = re.search(r'(?:https?://)?(?:www\.)?(?:github\.com/[A-Za-z0-9_-]+|[A-Za-z0-9_-]+\.github\.io|[A-Za-z0-9_-]+\.dev|[A-Za-z0-9_-]+\.me)', text, re.IGNORECASE)
        portfolio_url = portfolio_match.group() if portfolio_match else ""

        # Location
        location = ""
        # Match City, ST on a single line
        loc_match = re.search(r'\b([A-Z][a-zA-Z ]{2,30},\s*[A-Z]{2}(?:\s+\d{5})?)\b', text[:1000])
        if loc_match:
            location = loc_match.group(1).strip()
        else:
            for ent in entities:
                if ent["label"] == "LOCATION":
                    loc_c = ent["text"].strip(" ,.-|•\t\n")
                    if len(loc_c) > 2:
                        location = loc_c
                        break

        # Name: Look in header first, or first PERSON entity
        candidate_name = ""
        for ent in entities:
            if ent["label"] == "PERSON" and len(ent["text"].split()) in [2, 3, 4]:
                candidate_name = ent["text"]
                break

        if not candidate_name and header_text:
            first_lines = [l.strip() for l in header_text.split("\n") if l.strip()]
            for l in first_lines[:3]:
                # Candidate name is typically title-cased or uppercase, 2-4 words, no digits or symbols
                if re.match(r'^[A-Z][a-zA-Z\s\.\'-]{2,35}$', l) and len(l.split()) in [2, 3]:
                    candidate_name = l
                    break

        return {
            "candidate_name": candidate_name,
            "email": email,
            "phone": phone,
            "location": location,
            "linkedin_url": linkedin_url,
            "portfolio_url": portfolio_url
        }

    def extract_education(self, edu_text: str, full_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts structured degree, institution, field of study, and graduation year."""
        records = []
        source = edu_text if edu_text else full_text

        # Degree regex
        deg_regex = re.compile(
            r'\b(Bachelor(?:\'s)?(?:\s+of\s+[A-Za-z]+)?|B\.?S\.?|B\.?A\.?|B\.?Tech|B\.?E\.?|'
            r'Master(?:\'s)?(?:\s+of\s+[A-Za-z]+)?|M\.?S\.?|M\.?A\.?|M\.?Tech|M\.?E\.?|MBA|'
            r'Doctor(?:\s+of\s+Philosophy)?|Ph\.?D\.?|Associate(?:\'s)?(?:\s+of\s+[A-Za-z]+)?)\b',
            re.IGNORECASE
        )

        year_regex = re.compile(r'\b(19\d{2}|20\d{2})\b')

        lines = source.split("\n")
        current_entry = {}

        for line in lines:
            d_match = deg_regex.search(line)
            y_match = year_regex.search(line)

            # Look for institution keywords
            inst_match = re.search(r'\b([A-Z][A-Za-z\s&]+(?:University|College|Institute|Polytechnic|School))\b', line)

            if d_match or inst_match:
                if current_entry and (current_entry.get("degree") or current_entry.get("institution")):
                    records.append(current_entry)
                    current_entry = {}

                current_entry = {
                    "degree": d_match.group(1).strip() if d_match else "Degree",
                    "field_of_study": "",
                    "institution": inst_match.group(1).strip() if inst_match else "",
                    "graduation_year": y_match.group(1) if y_match else ""
                }

                # Attempt field detection
                field_match = re.search(r'in\s+([A-Za-z\s]+?)(?:,|\n|from|at|\.|\d|$)', line, re.IGNORECASE)
                if field_match:
                    current_entry["field_of_study"] = field_match.group(1).strip()

            elif current_entry and not current_entry.get("institution") and inst_match:
                current_entry["institution"] = inst_match.group(1).strip()
            elif current_entry and not current_entry.get("graduation_year") and y_match:
                current_entry["graduation_year"] = y_match.group(1)

        if current_entry and (current_entry.get("degree") or current_entry.get("institution")):
            records.append(current_entry)

        # Fallback to DEGREE entities if none found by line parser
        if not records:
            for ent in entities:
                if ent["label"] == "DEGREE":
                    records.append({
                        "degree": ent["text"],
                        "field_of_study": "",
                        "institution": "",
                        "graduation_year": ""
                    })

        return records[:4]

    def extract_categorized_skills(self, text: str) -> Dict[str, List[str]]:
        """Extracts and groups skills into categorized taxonomy."""
        categories = {
            "Programming Languages": set(),
            "Web Frameworks & Libraries": set(),
            "AI & Data Science": set(),
            "Databases & Storage": set(),
            "Cloud & DevOps": set(),
            "Developer Tools": set(),
            "Big Data": set(),
            "Methodologies & Soft Skills": set()
        }

        # Match skills against Knowledge Base
        for skill in self.skill_kb.get("skills", []):
            canon = skill["canonical_name"]
            cat = skill.get("category", "Developer Tools")
            if cat not in categories:
                categories[cat] = set()

            all_variants = [canon] + skill.get("aliases", [])
            for variant in all_variants:
                # Word boundary match
                pattern = r'(?<![A-Za-z0-9])' + re.escape(variant) + r'(?![A-Za-z0-9])'
                if re.search(pattern, text, re.IGNORECASE):
                    categories[cat].add(canon)
                    break

        return {k: sorted(list(v)) for k, v in categories.items() if v}

    def extract_experience(self, exp_text: str, full_text: str) -> List[Dict[str, Any]]:
        """Extracts work experience records."""
        source = exp_text if exp_text else full_text
        records = []

        date_range_regex = re.compile(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{2}/\d{4}|\d{4})\s*(?:-|to|–|—)\s*'
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{2}/\d{4}|\d{4}|Present|Current)',
            re.IGNORECASE
        )

        blocks = source.split("\n\n")
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue

            dates = date_range_regex.search(block)
            if dates:
                start_date = dates.group(1)
                end_date = dates.group(2)
                title = lines[0]
                company = lines[1] if len(lines) > 1 else ""

                # Filter title/company if they contain the dates
                title = date_range_regex.sub("", title).strip(" -|•,")
                company = date_range_regex.sub("", company).strip(" -|•,")

                bullets = [l.lstrip("•-* ").strip() for l in lines[2:] if len(l.strip()) > 10]

                records.append({
                    "job_title": title,
                    "company": company,
                    "start_date": start_date,
                    "end_date": end_date,
                    "responsibilities": bullets[:5]
                })

        return records[:6]

    def extract_projects(self, proj_text: str) -> List[Dict[str, Any]]:
        """Extracts projects, descriptions, and technology tags."""
        if not proj_text:
            return []

        projects = []
        blocks = proj_text.split("\n\n")

        for b in blocks:
            lines = [l.strip() for l in b.split("\n") if l.strip()]
            if not lines:
                continue

            p_name = lines[0].strip(" -|•*#")
            desc = " ".join(lines[1:]) if len(lines) > 1 else ""

            # Detect mentioned technologies
            techs = []
            for skill in self.skill_kb.get("skills", []):
                canon = skill["canonical_name"]
                if re.search(r'\b' + re.escape(canon) + r'\b', b, re.IGNORECASE):
                    techs.append(canon)

            projects.append({
                "name": p_name,
                "description": desc[:300],
                "technologies": list(set(techs))[:6]
            })

        return projects[:5]

    def extract_certifications(self, cert_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts certifications and credentials."""
        certs = []
        source_lines = cert_text.split("\n") if cert_text else []

        for line in source_lines:
            l = line.strip(" -•*")
            if len(l) > 5 and not l.isupper():
                certs.append({"name": l, "issuer": "", "date": ""})

        # Also add CERTIFICATION entities if found
        for ent in entities:
            if ent["label"] == "CERTIFICATION" and ent["text"] not in [c["name"] for c in certs]:
                certs.append({"name": ent["text"], "issuer": "", "date": ""})

        return certs[:6]

    def calculate_completeness(self, profile: Dict[str, Any]) -> float:
        """
        Calculates information completeness score (0-100%) based on populated core fields.
        """
        weights = {
            "candidate_name": 15,
            "email": 15,
            "phone": 10,
            "location": 5,
            "summary": 10,
            "education": 15,
            "skills": 15,
            "experience": 15
        }

        total = 0.0
        if profile.get("candidate_name"): total += weights["candidate_name"]
        if profile.get("email"): total += weights["email"]
        if profile.get("phone"): total += weights["phone"]
        if profile.get("location"): total += weights["location"]
        if profile.get("summary"): total += weights["summary"]
        if profile.get("education") and len(profile["education"]) > 0: total += weights["education"]
        if profile.get("skills") and sum(len(v) for v in profile["skills"].values()) > 0: total += weights["skills"]
        if profile.get("experience") and len(profile["experience"]) > 0: total += weights["experience"]

        return min(100.0, round(total, 1))

    def extract_full_profile(self, text: str) -> Dict[str, Any]:
        """Orchestrates end-to-end structured profile extraction."""
        sections = self.detect_sections(text)
        entities = self.ner.extract_entities(text)

        personal = self.extract_personal_info(text, sections.get("header", ""), entities)
        education = self.extract_education(sections.get("education", ""), text, entities)
        skills = self.extract_categorized_skills(text)
        experience = self.extract_experience(sections.get("experience", ""), text)
        projects = self.extract_projects(sections.get("projects", ""))
        certifications = self.extract_certifications(sections.get("certifications", ""), entities)

        profile = {
            **personal,
            "summary": sections.get("summary", "").strip()[:600],
            "education": education,
            "skills": skills,
            "experience": experience,
            "projects": projects,
            "certifications": certifications,
            "entities": entities
        }

        profile["completeness_score"] = self.calculate_completeness(profile)
        return profile
