import pdfplumber
from pathlib import Path

class PdfTableExtractor:
    """Extracts tables from PDFs as structured Markdown."""

    def extract_tables(self, path: Path) -> dict[int, list[str]]:
        """Returns {page_number: [table_as_markdown, ...]}"""
        result = {}
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue
                md_tables = []
                for table in tables:
                    md = self._table_to_markdown(table)
                    if md:
                        md_tables.append(md)
                if md_tables:
                    result[i] = md_tables
        return result

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        if not table or not table[0]:
            return ""
        # Clean cells
        rows = []
        for row in table:
            cells = [(c or "").strip().replace("\n", " ") for c in row]
            rows.append(cells)

        # Build markdown table
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
        body = "\n".join(
            "| " + " | ".join(r) + " |" for r in rows[1:]
        )
        return f"{header}\n{sep}\n{body}"
