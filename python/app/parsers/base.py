"""Common document parser interface and parser dispatch."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class DocumentExtractionError(ValueError):
    """Raised when an uploaded document cannot yield usable text."""


@dataclass(slots=True)
class ExtractedDocument:
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ExtractedDocument: ...


def extract_document(path: str | Path) -> ExtractedDocument:
    """Extract a supported document while preserving useful structure."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        from app.parsers.pdf_parser import PdfParser

        return PdfParser().parse(source)
    if suffix == ".docx":
        from app.parsers.docx_parser import DocxParser

        return DocxParser().parse(source)
    if suffix in {".txt", ".md"}:
        try:
            text = source.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentExtractionError("Text files must use UTF-8 encoding.") from exc
        if not text.strip():
            raise DocumentExtractionError("The uploaded document is empty.")
        return ExtractedDocument(text=text.replace("\r\n", "\n").strip(), metadata={"format": suffix[1:]})
    raise DocumentExtractionError("Unsupported file type. Upload a PDF, DOCX, TXT, or MD file.")
