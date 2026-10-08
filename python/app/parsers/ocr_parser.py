from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from app.parsers.base import ExtractedDocument, DocumentExtractionError

class OcrPdfParser:
    """Extracts text from scanned PDFs using Tesseract OCR."""

    def parse(self, path: Path) -> ExtractedDocument:
        try:
            images = convert_from_path(str(path), dpi=300)
        except Exception as exc:
            raise DocumentExtractionError(
                f"Unable to convert PDF to images: {exc}"
            ) from exc

        pages = []
        for img in images:
            text = pytesseract.image_to_string(img, lang="eng")
            text = self._post_process(text)
            pages.append(text)

        full_text = "\n\n".join(p for p in pages if p)

        if not full_text.strip():
            raise DocumentExtractionError(
                "OCR could not extract any text from this PDF."
            )

        return ExtractedDocument(
            text=full_text,
            metadata={
                "format": "pdf",
                "page_count": len(images),
                "ocr": True,
            },
        )

    @staticmethod
    def _post_process(text: str) -> str:
        import re
        # Collapse repeated spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Fix common OCR substitutions: 1 -> l and 0 -> O when surrounded by letters or at word boundaries
        text = re.sub(r'\b1(?=[a-zA-Z])', 'l', text)
        text = re.sub(r'(?<=[a-zA-Z])1\b', 'l', text)
        text = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', text)
        text = re.sub(r'\b0(?=[a-zA-Z])', 'O', text)
        text = re.sub(r'(?<=[a-zA-Z])0\b', 'O', text)
        text = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'O', text)
        return text.strip()
