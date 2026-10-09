"""Extracts text from scanned PDFs using Tesseract OCR."""

import re
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path

from app.parsers.base import DocumentExtractionError, ExtractedDocument

# Tesseract often reads the "I" in Roman numerals as "|", "l" or "i" (ARTICLE II -> ARTICLE Il).
# Only the numeral right after "Article" is touched, so clause numbers elsewhere stay as read.
ARTICLE_NUMERAL_RE = re.compile(r"\b((?i:article)\s+)([IVXLCDM|l1i]*[|li][IVXLCDM|l1i]*)(?=[\s.,:;)-]|$)")


class OcrPdfParser:
    """Extracts text from scanned PDFs using Tesseract OCR."""

    def parse(self, path: Path) -> ExtractedDocument:
        try:
            info = pdfinfo_from_path(str(path))
            total_pages = info["Pages"]
        except Exception as exc:
            raise DocumentExtractionError(f"Unable to get PDF info: {exc}") from exc

        pages = []
        for page_num in range(1, total_pages + 1):
            pages.append(self.parse_page(path, page_num))

        full_text = "\n\n".join(p for p in pages if p)

        if not full_text.strip():
            raise DocumentExtractionError(
                "OCR could not extract any text from this PDF."
            )

        return ExtractedDocument(
            text=full_text,
            metadata={
                "format": "pdf",
                "page_count": total_pages,
                "ocr": True,
            },
        )

    def parse_page(self, path: Path, page_num: int) -> str:
        """Extract text from a single page using OCR."""
        try:
            images = convert_from_path(str(path), dpi=300, first_page=page_num, last_page=page_num)
        except Exception as exc:
            raise DocumentExtractionError(f"Unable to convert PDF page {page_num} to images: {exc}") from exc

        if not images:
            return ""

        try:
            text = pytesseract.image_to_string(images[0], lang="eng")
        except pytesseract.TesseractNotFoundError as exc:
            raise DocumentExtractionError(f"OCR failed: Tesseract not found: {exc}") from exc
        except Exception as exc:
            raise DocumentExtractionError(f"OCR failed on page {page_num}: {exc}") from exc

        return self._post_process(text)

    @staticmethod
    def _post_process(text: str) -> str:
        # Collapse repeated spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Rejoin hyphenated words and soft-wrapped lines, as PdfParser does for text pages
        text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
        text = re.sub(r"(?<![.:;!?\n])\n(?=[a-z(\[])", " ", text)
        text = ARTICLE_NUMERAL_RE.sub(
            lambda m: m.group(1) + re.sub(r"[|l1i]", "I", m.group(2)), text
        )
        return text.strip()
