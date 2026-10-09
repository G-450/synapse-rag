"""Selectable-text PDF extraction with conservative margin cleanup, OCR fallback, and structured table extraction."""

import re
from collections import Counter
from pathlib import Path

import pdfplumber

from app.parsers.base import DocumentExtractionError, ExtractedDocument
from app.parsers.ocr_parser import OcrPdfParser
from app.parsers.table_extractor import PdfTableExtractor


class PdfParser:
    def parse(self, path: Path) -> ExtractedDocument:
        tables_by_page = {}
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages_text = []
                for i, page in enumerate(pdf.pages):
                    tables = page.find_tables()
                    if tables:
                        bboxes = [t.bbox for t in tables]

                        def not_in_table(obj, bboxes=bboxes):
                            if obj.get("object_type") == "char":
                                x_mid = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
                                y_mid = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                                for bx0, btop, bx1, bbottom in bboxes:
                                    if bx0 <= x_mid <= bx1 and btop <= y_mid <= bbottom:
                                        return False
                            return True

                        p = page.filter(not_in_table)
                        
                        # Extract table Markdown directly without reopening the PDF
                        tables_data = page.extract_tables()
                        if tables_data:
                            tables_md = PdfTableExtractor().convert_tables(tables_data)
                            if tables_md:
                                tables_by_page[i] = tables_md
                    else:
                        p = page

                    text = p.extract_text() or ""
                    pages_text.append(text.replace("\r\n", "\n"))
        except Exception as exc:
            raise DocumentExtractionError(f"Unable to read PDF: {exc}") from exc

        if not any(page.strip() for page in pages_text):
            return OcrPdfParser().parse(path)

        repeated_margins = self._repeated_margin_lines(pages_text)

        final_pages = []
        has_ocr = False
        for i, page in enumerate(pages_text):
            if page.strip():
                final_pages.append(self._clean_page(page, repeated_margins))
            else:
                has_ocr = True
                ocr_text = OcrPdfParser().parse_page(path, i + 1)
                final_pages.append(ocr_text)

        for i in range(len(final_pages)):
            if i in tables_by_page:
                tables_md = "\n\n".join(tables_by_page[i])
                final_pages[i] = f"{final_pages[i]}\n\n{tables_md}".strip()

        text = "\n\n".join(page for page in final_pages if page)
        metadata = {"format": "pdf", "page_count": len(pages_text)}
        if has_ocr:
            metadata["ocr"] = True
        return ExtractedDocument(text=text, metadata=metadata)

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
