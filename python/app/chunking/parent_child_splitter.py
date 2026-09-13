"""Contextual parent-child splitting for structurally parsed contracts."""

from dataclasses import asdict, dataclass
import re
import uuid

from app.parsers.legal_hierarchy_parser import LegalHierarchyParser, LegalSection


TOKEN_RE = re.compile(r"\S+")


@dataclass(slots=True)
class ParentChunk:
    id: str
    document_id: str
    filename: str
    content: str
    parent_header: str
    parent_article: str
    section_number: str
    section_title: str
    section_kind: str
    start_char: int
    end_char: int


@dataclass(slots=True)
class ChildChunk:
    id: str
    document_id: str
    filename: str
    parent_id: str
    parent_header: str
    parent_article: str
    section_number: str
    section_title: str
    section_kind: str
    content: str
    child_content: str
    parent_content: str
    start_char: int
    end_char: int
    chunk_type: str = "child"

    def metadata(self, source_corpus: str = "user-upload") -> dict:
        data = asdict(self)
        data["chunk_id"] = data.pop("id")
        data["source_corpus"] = source_corpus
        return data


class ParentChildSplitter:
    def __init__(
        self,
        parent_max_tokens: int = 1200,
        child_tokens: int = 200,
        child_overlap: int = 40,
    ) -> None:
        if child_tokens <= 0 or not 0 <= child_overlap < child_tokens:
            raise ValueError("child_overlap must be non-negative and smaller than child_tokens")
        self.parent_max_tokens = parent_max_tokens
        self.child_tokens = child_tokens
        self.child_overlap = child_overlap
        self.hierarchy_parser = LegalHierarchyParser()

    def split(self, text: str, document_id: str, filename: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for section_index, section in enumerate(self.hierarchy_parser.parse(text)):
            for part_index, part in enumerate(self._split_large_parent(section)):
                parent_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"synapse:{document_id}:parent:{section_index}:{part_index}:{part.content}",
                ))
                parent = ParentChunk(
                    id=parent_id,
                    document_id=document_id,
                    filename=filename,
                    content=part.content,
                    parent_header=part.header,
                    parent_article=part.article,
                    section_number=part.number,
                    section_title=part.title,
                    section_kind=part.kind,
                    start_char=part.start_char,
                    end_char=part.end_char,
                )
                parents.append(parent)
                children.extend(self._children_for_parent(parent))
        return parents, children

    def _split_large_parent(self, section: LegalSection) -> list[LegalSection]:
        tokens = list(TOKEN_RE.finditer(section.content))
        if len(tokens) <= self.parent_max_tokens:
            return [section]

        parts: list[LegalSection] = []
        for index in range(0, len(tokens), self.parent_max_tokens):
            group = tokens[index:index + self.parent_max_tokens]
            start, end = group[0].start(), group[-1].end()
            suffix = f" (Part {len(parts) + 1})"
            parts.append(LegalSection(
                kind=section.kind,
                number=section.number,
                title=section.title,
                content=section.content[start:end],
                article=section.article,
                header=section.header + suffix,
                start_char=section.start_char + start,
                end_char=section.start_char + end,
            ))
        return parts

    def _children_for_parent(self, parent: ParentChunk) -> list[ChildChunk]:
        tokens = list(TOKEN_RE.finditer(parent.content))
        if not tokens:
            return []
        step = self.child_tokens - self.child_overlap
        breadcrumb_parts = [f"Document: {parent.filename}"]
        if parent.parent_article:
            breadcrumb_parts.append(parent.parent_article)
        if parent.section_kind == "section" and parent.section_number:
            label = f"Section {parent.section_number}"
            if parent.section_title:
                label += f": {parent.section_title}"
            breadcrumb_parts.append(label)
        elif parent.parent_header:
            breadcrumb_parts.append(parent.parent_header)
        breadcrumb = "[" + " | ".join(breadcrumb_parts) + "]"

        children: list[ChildChunk] = []
        for child_index, token_start in enumerate(range(0, len(tokens), step)):
            group = tokens[token_start:token_start + self.child_tokens]
            if not group:
                break
            start, end = group[0].start(), group[-1].end()
            child_text = parent.content[start:end]
            child_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"synapse:{parent.id}:child:{child_index}:{child_text}",
            ))
            children.append(ChildChunk(
                id=child_id,
                document_id=parent.document_id,
                filename=parent.filename,
                parent_id=parent.id,
                parent_header=parent.parent_header,
                parent_article=parent.parent_article,
                section_number=parent.section_number,
                section_title=parent.section_title,
                section_kind=parent.section_kind,
                content=f"{breadcrumb}\n{child_text}",
                child_content=child_text,
                parent_content=parent.content,
                start_char=parent.start_char + start,
                end_char=parent.start_char + end,
            ))
            if token_start + self.child_tokens >= len(tokens):
                break
        return children
