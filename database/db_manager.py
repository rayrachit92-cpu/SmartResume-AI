"""
SmartResume AI - SQLite Database Manager
Provides persistent local storage for processed resumes, extracted profiles,
entities, linking results, sentiment analysis, and evaluation benchmarks.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from config.settings import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite persistent storage for SmartResume AI."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.init_db()

    def get_connection(self):
        """Returns a connection with dictionary cursor support."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Resumes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    is_scanned INTEGER DEFAULT 0,
                    page_count INTEGER DEFAULT 1,
                    original_text TEXT NOT NULL,
                    cleaned_text TEXT DEFAULT '',
                    tokens_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'processed'
                )
            """)

            # Extracted Profiles Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidate_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL UNIQUE,
                    candidate_name TEXT DEFAULT '',
                    email TEXT DEFAULT '',
                    phone TEXT DEFAULT '',
                    location TEXT DEFAULT '',
                    linkedin_url TEXT DEFAULT '',
                    portfolio_url TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    education_json TEXT DEFAULT '[]',
                    skills_json TEXT DEFAULT '[]',
                    experience_json TEXT DEFAULT '[]',
                    projects_json TEXT DEFAULT '[]',
                    certifications_json TEXT DEFAULT '[]',
                    completeness_score REAL DEFAULT 0.0,
                    FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE
                )
            """)

            # Extracted Entities Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    entity_text TEXT NOT NULL,
                    entity_label TEXT NOT NULL,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE
                )
            """)

            # Entity Linking Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    extracted_term TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    match_method TEXT NOT NULL,
                    similarity_score REAL DEFAULT 1.0,
                    FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE
                )
            """)

            # Sentiment Analysis Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    section_name TEXT NOT NULL,
                    pos REAL DEFAULT 0.0,
                    neu REAL DEFAULT 0.0,
                    neg REAL DEFAULT 0.0,
                    compound REAL DEFAULT 0.0,
                    tone_label TEXT DEFAULT 'Neutral',
                    FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE
                )
            """)

            # Topic Modeling Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resume_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL UNIQUE,
                    dominant_topic_id INTEGER DEFAULT 0,
                    dominant_topic_label TEXT DEFAULT '',
                    topic_distribution_json TEXT DEFAULT '{}',
                    FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE
                )
            """)

            # AI Q&A Session History
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_section TEXT DEFAULT '',
                    FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE SET NULL
                )
            """)

            conn.commit()

    def insert_resume(self, filename: str, file_type: str, original_text: str,
                      file_size: int = 0, is_scanned: bool = False,
                      page_count: int = 1, cleaned_text: str = "",
                      tokens_count: int = 0) -> int:
        """Inserts a new resume document and returns its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO resumes (
                    filename, file_type, upload_timestamp, file_size,
                    is_scanned, page_count, original_text, cleaned_text,
                    tokens_count, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename, file_type, now, file_size,
                1 if is_scanned else 0, page_count, original_text,
                cleaned_text, tokens_count, 'processed'
            ))
            conn.commit()
            return cursor.lastrowid

    def get_resume(self, resume_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single resume by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_resumes(self) -> List[Dict[str, Any]]:
        """Retrieves all stored resumes ordered by upload date descending."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM resumes ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def update_resume_text(self, resume_id: int, original_text: str, cleaned_text: str = "", tokens_count: int = 0):
        """Updates text of an existing resume (e.g. after manual editing)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE resumes
                SET original_text = ?, cleaned_text = ?, tokens_count = ?
                WHERE id = ?
            """, (original_text, cleaned_text, tokens_count, resume_id))
            conn.commit()

    def save_candidate_profile(self, resume_id: int, profile: Dict[str, Any]):
        """Inserts or updates candidate structured profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO candidate_profiles (
                    resume_id, candidate_name, email, phone, location,
                    linkedin_url, portfolio_url, summary, education_json,
                    skills_json, experience_json, projects_json,
                    certifications_json, completeness_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resume_id) DO UPDATE SET
                    candidate_name = excluded.candidate_name,
                    email = excluded.email,
                    phone = excluded.phone,
                    location = excluded.location,
                    linkedin_url = excluded.linkedin_url,
                    portfolio_url = excluded.portfolio_url,
                    summary = excluded.summary,
                    education_json = excluded.education_json,
                    skills_json = excluded.skills_json,
                    experience_json = excluded.experience_json,
                    projects_json = excluded.projects_json,
                    certifications_json = excluded.certifications_json,
                    completeness_score = excluded.completeness_score
            """, (
                resume_id,
                profile.get("candidate_name", ""),
                profile.get("email", ""),
                profile.get("phone", ""),
                profile.get("location", ""),
                profile.get("linkedin_url", ""),
                profile.get("portfolio_url", ""),
                profile.get("summary", ""),
                json.dumps(profile.get("education", [])),
                json.dumps(profile.get("skills", [])),
                json.dumps(profile.get("experience", [])),
                json.dumps(profile.get("projects", [])),
                json.dumps(profile.get("certifications", [])),
                profile.get("completeness_score", 0.0)
            ))
            conn.commit()

    def get_candidate_profile(self, resume_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves structured candidate profile by resume_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidate_profiles WHERE resume_id = ?", (resume_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["education"] = json.loads(data["education_json"] or "[]")
            data["skills"] = json.loads(data["skills_json"] or "[]")
            data["experience"] = json.loads(data["experience_json"] or "[]")
            data["projects"] = json.loads(data["projects_json"] or "[]")
            data["certifications"] = json.loads(data["certifications_json"] or "[]")
            return data

    def save_entities(self, resume_id: int, entities: List[Dict[str, Any]]):
        """Saves extracted NER entities for a resume."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM extracted_entities WHERE resume_id = ?", (resume_id,))
            for ent in entities:
                cursor.execute("""
                    INSERT INTO extracted_entities (
                        resume_id, entity_text, entity_label, start_char, end_char, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    resume_id,
                    ent.get("text", ""),
                    ent.get("label", ""),
                    ent.get("start", 0),
                    ent.get("end", 0),
                    ent.get("confidence", 1.0)
                ))
            conn.commit()

    def get_entities(self, resume_id: int) -> List[Dict[str, Any]]:
        """Fetches all entities for a given resume."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM extracted_entities WHERE resume_id = ?", (resume_id,))
            return [dict(row) for row in cursor.fetchall()]

    def save_entity_links(self, resume_id: int, links: List[Dict[str, Any]]):
        """Saves linked canonical skills/entities."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entity_links WHERE resume_id = ?", (resume_id,))
            for link in links:
                cursor.execute("""
                    INSERT INTO entity_links (
                        resume_id, extracted_term, canonical_name, category, match_method, similarity_score
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    resume_id,
                    link.get("extracted_term", ""),
                    link.get("canonical_name", ""),
                    link.get("category", ""),
                    link.get("match_method", "exact"),
                    link.get("similarity_score", 1.0)
                ))
            conn.commit()

    def get_entity_links(self, resume_id: int) -> List[Dict[str, Any]]:
        """Retrieves linked entities for a resume."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entity_links WHERE resume_id = ?", (resume_id,))
            return [dict(row) for row in cursor.fetchall()]

    def save_sentiment(self, resume_id: int, sentiments: List[Dict[str, Any]]):
        """Saves sentiment analysis scores."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sentiment_scores WHERE resume_id = ?", (resume_id,))
            for s in sentiments:
                cursor.execute("""
                    INSERT INTO sentiment_scores (
                        resume_id, section_name, pos, neu, neg, compound, tone_label
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    resume_id,
                    s.get("section_name", "General"),
                    s.get("pos", 0.0),
                    s.get("neu", 0.0),
                    s.get("neg", 0.0),
                    s.get("compound", 0.0),
                    s.get("tone_label", "Neutral")
                ))
            conn.commit()

    def get_sentiments(self, resume_id: int) -> List[Dict[str, Any]]:
        """Retrieves sentiment analysis records for a resume."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sentiment_scores WHERE resume_id = ?", (resume_id,))
            return [dict(row) for row in cursor.fetchall()]

    def save_topic(self, resume_id: int, dominant_id: int, dominant_label: str, distribution: Dict[str, float]):
        """Saves topic model assignment for a resume."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO resume_topics (resume_id, dominant_topic_id, dominant_topic_label, topic_distribution_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(resume_id) DO UPDATE SET
                    dominant_topic_id = excluded.dominant_topic_id,
                    dominant_topic_label = excluded.dominant_topic_label,
                    topic_distribution_json = excluded.topic_distribution_json
            """, (resume_id, dominant_id, dominant_label, json.dumps(distribution)))
            conn.commit()

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """Retrieves topic assignments for all resumes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM resume_topics")
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["distribution"] = json.loads(d["topic_distribution_json"] or "{}")
                results.append(d)
            return results

    def save_ai_interaction(self, resume_id: Optional[int], question: str, answer: str, source_section: str = ""):
        """Stores AI Q&A interaction in session log."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_interactions (resume_id, timestamp, question, answer, source_section)
                VALUES (?, ?, ?, ?, ?)
            """, (resume_id, datetime.now().isoformat(), question, answer, source_section))
            conn.commit()

    def get_ai_interactions(self, resume_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves AI Q&A history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if resume_id:
                cursor.execute("SELECT * FROM ai_interactions WHERE resume_id = ? ORDER BY id ASC", (resume_id,))
            else:
                cursor.execute("SELECT * FROM ai_interactions ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_kpi_metrics(self) -> Dict[str, Any]:
        """Calculates live KPI metrics from actual database contents."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM resumes")
            total_resumes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM candidate_profiles")
            total_candidates = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT canonical_name) FROM entity_links")
            unique_skills = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM extracted_entities")
            total_entities = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(completeness_score) FROM candidate_profiles")
            avg_completeness = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT COUNT(*) FROM resumes WHERE status = 'processed'")
            processed_count = cursor.fetchone()[0]
            success_rate = (processed_count / total_resumes * 100) if total_resumes > 0 else 0.0

            return {
                "total_resumes": total_resumes,
                "total_candidates": total_candidates,
                "unique_skills": unique_skills,
                "total_entities": total_entities,
                "avg_completeness": round(avg_completeness, 1),
                "success_rate": round(success_rate, 1)
            }

    def delete_resume(self, resume_id: int):
        """Completely deletes a resume and its associated extracted records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
            cursor.execute("DELETE FROM candidate_profiles WHERE resume_id = ?", (resume_id,))
            cursor.execute("DELETE FROM extracted_entities WHERE resume_id = ?", (resume_id,))
            cursor.execute("DELETE FROM entity_links WHERE resume_id = ?", (resume_id,))
            cursor.execute("DELETE FROM sentiment_scores WHERE resume_id = ?", (resume_id,))
            cursor.execute("DELETE FROM resume_topics WHERE resume_id = ?", (resume_id,))
            cursor.execute("DELETE FROM ai_interactions WHERE resume_id = ?", (resume_id,))
            conn.commit()

    def clear_all_data(self):
        """Clears all stored application data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM resumes")
            cursor.execute("DELETE FROM candidate_profiles")
            cursor.execute("DELETE FROM extracted_entities")
            cursor.execute("DELETE FROM entity_links")
            cursor.execute("DELETE FROM sentiment_scores")
            cursor.execute("DELETE FROM resume_topics")
            cursor.execute("DELETE FROM ai_interactions")
            conn.commit()


# Singleton instance
db = DatabaseManager()

