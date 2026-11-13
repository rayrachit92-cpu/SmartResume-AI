"""
SmartResume AI - Application Configuration & Settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
STORAGE_DIR = DATA_DIR / "storage"

# Create essential directories
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    STORAGE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "smartresume.db")))

# Skill Knowledge Base
SKILL_KB_PATH = DATA_DIR / "skill_knowledge_base.json"
DATASET_METADATA_PATH = DATA_DIR / "dataset_metadata.json"

# AI / LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")

# Cloud / AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# OCR Configuration
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() in ("true", "1", "yes")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")

# UI Theme Configuration
THEME = {
    "primary_bg": "#FFFFFF",
    "secondary_bg": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "primary_accent": "#2563EB",
    "secondary_accent": "#7C3AED",
    "teal_accent": "#0D9488",
    "text_dark": "#0F172A",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "border_subtle": "#F1F5F9",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}

