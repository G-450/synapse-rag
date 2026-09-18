"""Rule-based legal hierarchy recognition for contracts."""

from dataclasses import dataclass
import re


ARTICLE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?ARTICLE\s+([0-9IVXLCDM]+)(?:\s*[-:.]\s*|\s+)?(.*)$", re.I)
SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:SECTION|Section)\s+([0-9]+(?:\.[0-9A-Za-z]+)*(?:\([a-z0-9ivx]+\))?)(?:\s*[-:.]\s*|\s+)?(.*)$"
)
NUMBERED_RE = re.compile(r"^\s*(?:#{1,6}\s*)?([0-9]+(?:\.[0-9]+)+(?:\([a-z0-9ivx]+\))?)\s+(.+)$", re.I)
SCHEDULE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(SCHEDULE|EXHIBIT|ANNEX|APPENDIX)\s+([A-Z0-9IVXLCDM-]+)(?:\s*[-:.]\s*|\s+)?(.*)$", re.I)
DEFINITION_RE = re.compile(r'^\s*["“]([^"”]{1,100})["”]\s+(?:means|shall mean|has the meaning)\b', re.I)
RECITAL_RE = re.compile(r"^\s*(?:WHEREAS\b|RECITALS?\b)", re.I)


@dataclass(slots=True)
class LegalSection:
    kind: str
    number: str
    title: str
    content: str
    article: str = ""
    header: str = ""
    start_char: int = 0
    end_char: int = 0

    @property
    def path(self) -> str:
        parts = [part for part in (self.article, self.header) if part]
        return " | ".join(parts)


class LegalHierarchyParser:
    """Split extracted text at semantic legal boundaries while retaining headings."""

    def parse(self, text: str) -> list[LegalSection]:
        lines = text.replace("\r\n", "\n").splitlines()
        sections: list[LegalSection] = []
        article = ""
        current_header = "Preamble"
        current_kind = "preamble"
        current_number = ""
        current_title = ""
        buffer: list[str] = []
        cursor = 0
        section_start = 0

        def flush(end: int) -> None:
            nonlocal buffer
            content = "\n".join(buffer).strip()
            article_lines = [line for line in buffer if line]
            if current_kind == "article" and len(article_lines) <= 2 and all(
                line == line.upper() for line in article_lines
            ):
                buffer = []
                return
            if content:
                sections.append(LegalSection(
                    kind=current_kind,
                    number=current_number,
                    title=current_title,
                    content=content,
                    article=article,
                    header=current_header,
                    start_char=section_start,
                    end_char=end,
                ))
            buffer = []

        for line in lines:
            raw = line.strip()
            line_start = cursor
            cursor += len(line) + 1
            if not raw:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue

            match = ARTICLE_RE.match(raw)
            if match:
                flush(line_start)
                article = f"Article {match.group(1).upper()}"
                current_kind, current_number = "article", match.group(1).upper()
                current_title = match.group(2).strip()
                current_header = " ".join(x for x in (article, current_title) if x)
                section_start, buffer = line_start, [raw]
                continue

            match = SECTION_RE.match(raw) or NUMBERED_RE.match(raw)
            if match:
                flush(line_start)
                rest = match.group(2).strip()
                title_match = re.match(r"^(.{1,120}?)\.\s+(?=[A-Z])", rest)
                current_kind, current_number = "section", match.group(1)
                current_title = title_match.group(1).strip() if title_match else rest
                current_header = f"Section {current_number}" + (f": {current_title}" if current_title else "")
                section_start, buffer = line_start, [raw]
                continue

            match = SCHEDULE_RE.match(raw)
            if match:
                flush(line_start)
                current_kind = match.group(1).lower()
                current_number, current_title = match.group(2), match.group(3).strip()
                current_header = f"{match.group(1).title()} {current_number}" + (f": {current_title}" if current_title else "")
                section_start, buffer = line_start, [raw]
                continue

            if RECITAL_RE.match(raw) and current_kind != "recital":
                flush(line_start)
                current_kind, current_number, current_title = "recital", "", "Recitals"
                current_header, section_start, buffer = "Recitals", line_start, [raw]
                continue

            definition = DEFINITION_RE.match(raw)
            if definition:
                flush(line_start)
                current_kind, current_number = "definition", ""
                current_title = definition.group(1)
                current_header = f'Definition: "{current_title}"'
                section_start, buffer = line_start, [raw]
                continue

            buffer.append(raw)

        flush(len(text))
        if not sections and text.strip():
            sections.append(LegalSection("document", "", "", text.strip(), header="Document", end_char=len(text)))
        return sections
