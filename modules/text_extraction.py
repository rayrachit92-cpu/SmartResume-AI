"""
SmartResume AI - Text Extraction Module
Extracts text from PDF, DOCX, and TXT files.
Includes scanned document detection and optional OCR using Tesseract.
"""

import io
import re
import os
import logging
from typing import Dict, Any, Tuple
from pathlib import Path

import pymupdf
import docx
from PIL import Image

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

from config.settings import ENABLE_OCR, TESSERACT_CMD

logger = logging.getLogger(__name__)

if HAS_PYTESSERACT and TESSERACT_CMD:
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except Exception as e:
        logger.warning(f"Could not configure Tesseract command: {e}")


class DocumentExtractor:
    """Extracts raw text and structural metadata from multiple resume formats."""

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Dispatches extraction based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return DocumentExtractor.extract_from_pdf_bytes(file_bytes)
        elif ext in [".docx", ".doc"]:
            return DocumentExtractor.extract_from_docx_bytes(file_bytes)
        elif ext in [".txt", ".text"]:
            return DocumentExtractor.extract_from_txt_bytes(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload PDF, DOCX, or TXT.")

    @staticmethod
    def extract_from_file_path(file_path: str) -> Dict[str, Any]:
        """Extracts text directly from local file path."""
        with open(file_path, "rb") as f:
            data = f.read()
        return DocumentExtractor.extract_from_bytes(data, file_path)

    @staticmethod
    def extract_from_pdf_bytes(file_bytes: bytes) -> Dict[str, Any]:
        """Extracts text from PDF with scanned page detection and OCR fallback."""
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        page_texts = []
        is_scanned = False
        ocr_applied = False

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            # If page text is very short (< 40 characters), page may be scanned or an image
            if len(text) < 40 and ENABLE_OCR and HAS_PYTESSERACT:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        is_scanned = True
                        ocr_applied = True
                except Exception as e:
                    logger.debug(f"OCR failed for page {page_num}: {e}")

            page_texts.append(text)

        full_text = "\n\n".join(filter(None, page_texts))

        # Check if entire document produced very little text
        if len(full_text.strip()) < 50:
            is_scanned = True

        return {
            "text": full_text,
            "page_count": total_pages,
            "is_scanned": is_scanned,
            "ocr_applied": ocr_applied,
            "file_type": "PDF",
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
        }

    @staticmethod
    def extract_from_docx_bytes(file_bytes: bytes) -> Dict[str, Any]:
        """Extracts text from DOCX paragraphs and tables."""
        doc_stream = io.BytesIO(file_bytes)
        doc = docx.Document(doc_stream)

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # Also extract text from any tables in the document
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_str = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_str:
                    table_texts.append(row_str)

        all_text = "\n".join(paragraphs)
        if table_texts:
            all_text += "\n\n" + "\n".join(table_texts)

        return {
            "text": all_text.strip(),
            "page_count": max(1, len(all_text) // 2500),
            "is_scanned": False,
            "ocr_applied": False,
            "file_type": "DOCX",
            "word_count": len(all_text.split()),
            "char_count": len(all_text),
        }

    @staticmethod
    def extract_from_txt_bytes(file_bytes: bytes) -> Dict[str, Any]:
        """Extracts text from plain text file with encoding detection."""
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(enc)
                return {
                    "text": text.strip(),
                    "page_count": 1,
                    "is_scanned": False,
                    "ocr_applied": False,
                    "file_type": "TXT",
                    "word_count": len(text.split()),
                    "char_count": len(text),
                }
            except UnicodeDecodeError:
                continue

        # Fallback with replacement
        text = file_bytes.decode("utf-8", errors="replace")
        return {
            "text": text.strip(),
            "page_count": 1,
            "is_scanned": False,
            "ocr_applied": False,
            "file_type": "TXT",
            "word_count": len(text.split()),
            "char_count": len(text),
        }
