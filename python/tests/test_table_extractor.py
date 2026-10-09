from unittest.mock import patch
from app.parsers.table_extractor import PdfTableExtractor

def test_extract_tables_success():
    extractor = PdfTableExtractor()
    with patch("app.parsers.table_extractor.pdfplumber.open") as mock_open:
        mock_pdf = mock_open.return_value.__enter__.return_value
        mock_page = mock_pdf.pages[0]
        mock_page.extract_tables.return_value = [
            [["Header 1", "Header 2"], ["Val 1", "Val 2"]]
        ]
        mock_pdf.pages = [mock_page]

        result = extractor.extract_tables("dummy.pdf")

        assert 0 in result
        assert len(result[0]) == 1
        md_table = result[0][0]
        assert "| Header 1 | Header 2 |" in md_table
        assert "| Val 1 | Val 2 |" in md_table

def test_extract_tables_no_tables():
    extractor = PdfTableExtractor()
    with patch("app.parsers.table_extractor.pdfplumber.open") as mock_open:
        mock_pdf = mock_open.return_value.__enter__.return_value
        mock_page = mock_pdf.pages[0]
        mock_page.extract_tables.return_value = []
        mock_pdf.pages = [mock_page]

        result = extractor.extract_tables("dummy.pdf")

        assert result == {}

def test_table_to_markdown_empty():
    extractor = PdfTableExtractor()
    assert extractor._table_to_markdown([]) == ""
    assert extractor._table_to_markdown([[]]) == ""
