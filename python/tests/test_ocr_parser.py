import pytest
from unittest.mock import patch
from pathlib import Path
from app.parsers.ocr_parser import OcrPdfParser
from app.parsers.base import DocumentExtractionError

def test_ocr_parser_success():
    parser = OcrPdfParser()
    with patch("app.parsers.ocr_parser.pdfinfo_from_path") as mock_info, \
         patch("app.parsers.ocr_parser.convert_from_path") as mock_convert, \
         patch("app.parsers.ocr_parser.pytesseract.image_to_string") as mock_tesseract:

        mock_info.return_value = {"Pages": 2}
        mock_convert.return_value = ["img"]
        mock_tesseract.side_effect = ["Page 1 text", "Page 2 text"]

        result = parser.parse(Path("dummy.pdf"))

        assert result.text == "Page 1 text\n\nPage 2 text"
        assert result.metadata["ocr"] is True
        assert result.metadata["page_count"] == 2

def test_ocr_parser_no_text():
    parser = OcrPdfParser()
    with patch("app.parsers.ocr_parser.pdfinfo_from_path") as mock_info, \
         patch("app.parsers.ocr_parser.convert_from_path") as mock_convert, \
         patch("app.parsers.ocr_parser.pytesseract.image_to_string") as mock_tesseract:

        mock_info.return_value = {"Pages": 1}
        mock_convert.return_value = ["img1"]
        mock_tesseract.return_value = "   "

        with pytest.raises(DocumentExtractionError, match="OCR could not extract any text"):
            parser.parse(Path("dummy.pdf"))

def test_ocr_parser_conversion_error():
    parser = OcrPdfParser()
    with patch("app.parsers.ocr_parser.pdfinfo_from_path") as mock_info, \
         patch("app.parsers.ocr_parser.convert_from_path") as mock_convert:
        mock_info.return_value = {"Pages": 1}
        mock_convert.side_effect = Exception("Poppler missing")

        with pytest.raises(DocumentExtractionError, match="Unable to convert PDF page 1 to images"):
            parser.parse(Path("dummy.pdf"))
