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

def test_post_process_repairs_misread_article_numerals():
    text = (
        "ARTICLE |\nDEFINITIONS\n\nARTICLE Il\nCONFIDENTIALITY\n\nARTICLE ViiI\nWARRANTIES\n\n"
        "obligations under Article Il shall survive"
    )

    result = OcrPdfParser._post_process(text)

    assert result.startswith("ARTICLE I\nDEFINITIONS")
    assert "ARTICLE II\nCONFIDENTIALITY" in result
    assert "ARTICLE VIII\nWARRANTIES" in result
    assert "under Article II shall" in result

def test_post_process_leaves_other_numbers_alone():
    text = (
        "ARTICLE XL\nARTICLE 1\nARTICLE IV\nSection 1A applies on the 1st day per Exhibit B1\n"
        "Each article in this Agreement and the article lists"
    )

    assert OcrPdfParser._post_process(text) == text

def test_post_process_restores_article_stroke_from_section_numbers():
    text = (
        "ARTICLE Il\nCONFIDENTIALITY\nSection 2.1 Duty.\n\n"
        "ARTICLE Il\nEXCLUSIONS\nSection 3.1 Standard Exclusions.\n\n"
        "ARTICLE VII - REMEDIES\n8.1 Injunctive relief.\n\n"
        "obligations under Article Il shall survive"
    )

    result = OcrPdfParser._post_process(text)

    assert "ARTICLE II\nCONFIDENTIALITY" in result
    assert "ARTICLE III\nEXCLUSIONS" in result
    assert "ARTICLE VIII - REMEDIES" in result
    assert "under Article II shall" in result

def test_post_process_keeps_article_without_matching_evidence():
    text = (
        "ARTICLE II\nNo numbered sections here.\n\n"
        "ARTICLE VI\nSection 4.1 Numbered differently on purpose.\n\n"
        "ARTICLE IX\nSection 2.1 Restarted numbering."
    )

    assert OcrPdfParser._post_process(text) == text

def test_post_process_rejoins_wrapped_lines():
    text = "Section 2.1 The Receiving Party agrees to pro-\ntect the information of the\nDisclosing Party.\nSection 2.2 Next."

    assert OcrPdfParser._post_process(text) == (
        "Section 2.1 The Receiving Party agrees to protect the information of the\n"
        "Disclosing Party.\nSection 2.2 Next."
    )
