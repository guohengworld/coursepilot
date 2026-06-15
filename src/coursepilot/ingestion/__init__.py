"""Data ingestion pipeline: PDF/DOCX parsing, KP mapping, encoding."""

from coursepilot.ingestion.pdf_parser import parse_pdf
from coursepilot.ingestion.docx_parser import parse_docx
from coursepilot.ingestion.parser_utils import extract_knowledge_units

__all__ = [
    "parse_pdf",
    "parse_docx",
    "extract_knowledge_units",
]
