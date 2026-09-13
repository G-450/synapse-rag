"""Selectable-text PDF extraction with conservative margin cleanup."""

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from app.parsers.base import DocumentExtractionError, ExtractedDocument


class PdfParser:
    def parse(self, path: Path) -> ExtractedDocument:
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").replace("\r\n", "\n") for page in reader.pages]
        except Exception as exc:
            raise DocumentExtractionError(f"Unable to read PDF: {exc}") from exc

        if not any(page.strip() for page in pages):
            raise DocumentExtractionError(
                "No extractable text found in PDF. Scanned images are not yet supported."
            )

        repeated_margins = self._repeated_margin_lines(pages)
        cleaned = [self._clean_page(page, repeated_margins) for page in pages]
        text = "\n\n".join(page for page in cleaned if page)
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
