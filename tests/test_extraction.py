"""
Unit tests for DocumentExtractor (PDF, DOCX, TXT, scanned handling, invalid inputs).
"""

import pytest
import io
import pymupdf
import docx
from modules.text_extraction import DocumentExtractor


def test_txt_extraction():
    sample = "Alex Morgan\nAI Engineer\nSan Francisco, CA\nEmail: alex@example.com"
    data = sample.encode("utf-8")
    result = DocumentExtractor.extract_from_bytes(data, "resume.txt")
    assert result["file_type"] == "TXT"
    assert "Alex Morgan" in result["text"]
    assert result["word_count"] > 0
    assert result["is_scanned"] is False


def test_pdf_extraction():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Sarah Jenkins\nSenior Full Stack Developer\nReact, Node.js, Python")
    pdf_bytes = doc.write()
    doc.close()

    result = DocumentExtractor.extract_from_bytes(pdf_bytes, "sarah.pdf")
    assert result["file_type"] == "PDF"
    assert "Sarah Jenkins" in result["text"]
    assert "React" in result["text"]
    assert result["page_count"] == 1


def test_docx_extraction():
    doc = docx.Document()
    doc.add_paragraph("Marcus Vance - Principal Cloud Architect")
    doc.add_paragraph("Expert in AWS, Kubernetes, and Terraform")

    stream = io.BytesIO()
    doc.save(stream)
    docx_bytes = stream.getvalue()

    result = DocumentExtractor.extract_from_bytes(docx_bytes, "marcus.docx")
    assert result["file_type"] == "DOCX"
    assert "Marcus Vance" in result["text"]
    assert "Terraform" in result["text"]


def test_invalid_format_rejection():
    with pytest.raises(ValueError) as excinfo:
        DocumentExtractor.extract_from_bytes(b"image data", "photo.png")
    assert "Unsupported file format" in str(excinfo.value)
