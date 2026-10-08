"""Selectable-text PDF extraction with conservative margin cleanup."""

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from app.parsers.base import DocumentExtractionError, ExtractedDocument
from app.parsers.ocr_parser import OcrPdfParser
from app.parsers.table_extractor import PdfTableExtractor
from pdf2image import convert_from_path
import pytesseract

class PdfParser:
    def parse(self, path: Path) -> ExtractedDocument:
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").replace("\r\n", "\n") for page in reader.pages]
        except Exception as exc:
            raise DocumentExtractionError(f"Unable to read PDF: {exc}") from exc

        if not any(page.strip() for page in pages):
            return OcrPdfParser().parse(path)

        repeated_margins = self._repeated_margin_lines(pages)
        
        final_pages = []
        for i, page in enumerate(pages):
            if page.strip():
                final_pages.append(self._clean_page(page, repeated_margins))
            else:
                # This page has no text - OCR just this page
                img = convert_from_path(str(path), dpi=300,
                                        first_page=i+1, last_page=i+1)[0]
                ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
                final_pages.append(ocr_text)

        tables_by_page = PdfTableExtractor().extract_tables(path)
        
        for i in range(len(final_pages)):
            if i in tables_by_page:
                tables_md = "\n\n".join(tables_by_page[i])
                final_pages[i] = f"{final_pages[i]}\n\n{tables_md}".strip()

        text = "\n\n".join(page for page in final_pages if page)
        return ExtractedDocument(text=text, metadata={"format": "pdf", "page_count": len(pages)})

    @staticmethod
    def _normalized_margin(line: str) -> str:
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip())).lower()

    def _repeated_margin_lines(self, pages: list[str]) -> set[str]:
        candidates: Counter[str] = Counter()
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            for line in lines[:2] + lines[-2:]:
                if len(line) <= 160:
                    candidates[self._normalized_margin(line)] += 1
        threshold = max(2, (len(pages) + 1) // 2)
        return {line for line, count in candidates.items() if count >= threshold}

    def _clean_page(self, page: str, repeated_margins: set[str]) -> str:
        lines = []
        for line in page.splitlines():
            stripped = re.sub(r"[ \t]+", " ", line).strip()
            if not stripped or re.fullmatch(r"(?:page\s+)?\d+(?:\s+of\s+\d+)?", stripped, re.I):
                continue
            if self._normalized_margin(stripped) in repeated_margins:
                continue
            lines.append(stripped)
        text = "\n".join(lines)
        text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
        text = re.sub(r"(?<![.:;!?])\n(?=[a-z(\[])", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
