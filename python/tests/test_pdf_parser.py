from unittest.mock import patch, MagicMock
from pathlib import Path
from app.parsers.pdf_parser import PdfParser
from app.parsers.base import ExtractedDocument

def test_pdf_parser_delegates_to_ocr():
    parser = PdfParser()
    with patch("app.parsers.pdf_parser.pdfplumber.open") as mock_open, \
         patch("app.parsers.pdf_parser.OcrPdfParser") as mock_ocr_class:
         
        mock_pdf = mock_open.return_value.__enter__.return_value
        mock_page = MagicMock()
        mock_page.find_tables.return_value = []
        
        # We need to simulate the filter behavior since we changed it.
        # page.filter returns a new page or itself, let's just make it return itself
        mock_page.filter.return_value = mock_page
        mock_page.extract_text.return_value = "   "
        mock_pdf.pages = [mock_page]
        
        mock_ocr_parser = mock_ocr_class.return_value
        mock_ocr_parser.parse.return_value = ExtractedDocument(text="OCR TEXT", metadata={"ocr": True, "format": "pdf", "page_count": 1})
        
        result = parser.parse(Path("dummy.pdf"))
        
        assert result.text == "OCR TEXT"
        mock_ocr_parser.parse.assert_called_once()

def test_pdf_parser_mixed_pdf():
    parser = PdfParser()
    with patch("app.parsers.pdf_parser.pdfplumber.open") as mock_open, \
         patch("app.parsers.pdf_parser.PdfTableExtractor") as mock_table, \
         patch("app.parsers.pdf_parser.OcrPdfParser") as mock_ocr_class:
         
        mock_pdf = mock_open.return_value.__enter__.return_value
        
        page1 = MagicMock()
        page1.find_tables.return_value = []
        page1.filter.return_value = page1
        page1.extract_text.return_value = "Line 1\nLine 2\nPage 1 Text\nLine 4\nLine 5"
        
        page2 = MagicMock()
        page2.find_tables.return_value = []
        page2.filter.return_value = page2
        page2.extract_text.return_value = "   "
        
        mock_pdf.pages = [page1, page2]
        
        mock_ocr_inst = mock_ocr_class.return_value
        mock_ocr_inst.parse_page.return_value = "Page 2 OCR Text"
        
        result = parser.parse(Path("dummy.pdf"))
        
        assert "Page 1 Text" in result.text
        assert "Page 2 OCR Text" in result.text
        assert result.metadata["ocr"] is True
        assert result.metadata["page_count"] == 2

def test_pdf_parser_tables_merged():
    parser = PdfParser()
    with patch("app.parsers.pdf_parser.pdfplumber.open") as mock_open, \
         patch("app.parsers.pdf_parser.PdfTableExtractor") as mock_table:
         
        mock_pdf = mock_open.return_value.__enter__.return_value
        page1 = MagicMock()
        mock_table_obj = MagicMock()
        page1.find_tables.return_value = [mock_table_obj]
        page1.extract_tables.return_value = [[["Table", "Data"]]]
        page1.filter.return_value = page1
        page1.extract_text.return_value = "Line 1\nLine 2\nPage 1 Text Without Table\nLine 4\nLine 5"
        mock_pdf.pages = [page1]
        
        mock_table_inst = mock_table.return_value
        mock_table_inst.convert_tables.return_value = ["| Table | Data |"]
        
        result = parser.parse(Path("dummy.pdf"))
        
        assert "Page 1 Text Without Table" in result.text
        assert "| Table | Data |" in result.text
