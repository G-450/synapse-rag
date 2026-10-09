"""Extracts text from scanned PDFs using Tesseract OCR."""

import re
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path

from app.parsers.base import DocumentExtractionError, ExtractedDocument

# Tesseract often reads the "I" in Roman numerals as "|", "l" or "i" (ARTICLE II -> ARTICLE Il).
# Only the numeral right after "Article" is touched, so clause numbers elsewhere stay as read.
ARTICLE_NUMERAL_RE = re.compile(r"\b((?i:article)\s+)([IVXLCDM|l1i]*[|li][IVXLCDM|l1i]*)(?=[\s.,:;)-]|$)")
# Tesseract can also merge strokes and drop an "I" outright (ARTICLE III -> ARTICLE II). The first
# section number under a heading (Section 3.1) shows which article it really is.
ARTICLE_HEADING_RE = re.compile(r"^((?i:article)\s+)([IVXLCDM]+)(?=[\s.:-]|$)", re.M)
SECTION_NUMBER_RE = re.compile(r"^\s*(?:(?i:section)\s+)?(\d+)\.\d+", re.M)
ROMAN_VALUES = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


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
        text = OcrPdfParser._restore_dropped_article_strokes(text)
        return text.strip()

    @staticmethod
    def _restore_dropped_article_strokes(text: str) -> str:
        """Fix an article heading whose numeral lost an "I", using the section numbers under it.

        Only applied when the heading is the expected numeral with I's missing, so headings
        that are genuinely numbered differently from their sections are left as written.
        """
        headings = list(ARTICLE_HEADING_RE.finditer(text))
        parts, last = [], 0
        for i, heading in enumerate(headings):
            body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section = SECTION_NUMBER_RE.search(text, heading.end(), body_end)
            if not section or not 1 <= int(section.group(1)) < 4000:
                continue
            numeral = heading.group(2)
            expected = OcrPdfParser._to_roman(int(section.group(1)))
            if numeral != expected and OcrPdfParser._missing_only_i(numeral, expected):
                parts.append(text[last:heading.start(2)] + expected)
                last = heading.end(2)
        return "".join(parts) + text[last:]

    @staticmethod
    def _to_roman(number: int) -> str:
        numeral = ""
        for value, symbol in ROMAN_VALUES:
            count, number = divmod(number, value)
            numeral += symbol * count
        return numeral

    @staticmethod
    def _missing_only_i(read: str, expected: str) -> bool:
        """True when deleting one or more "I"s from expected gives read."""
        if len(read) >= len(expected):
            return False
        pos = 0
        for char in expected:
            if pos < len(read) and read[pos] == char:
                pos += 1
            elif char != "I":
                return False
        return pos == len(read)
