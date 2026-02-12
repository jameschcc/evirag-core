# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from .contracts import SourceLocator


class DocumentParseError(ValueError):
    """Raised when a supported document cannot be decoded or parsed."""


@dataclass(frozen=True)
class ParsedBlock:
    start: int
    end: int
    locator: SourceLocator


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    format: str
    blocks: list[ParsedBlock]


# 代码/配置类文件按纯文本解析，但使用代码切片器并标记 source_kind=code。
CODE_SUFFIXES = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".go", ".java",
    ".kt", ".kts", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs", ".rs",
    ".rb", ".php", ".phtml", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql",
    ".swift", ".m", ".mm", ".scala", ".lua", ".pl", ".pm", ".r", ".vue", ".svelte",
    ".dart", ".ex", ".exs", ".erl", ".hrl", ".clj", ".cljs", ".groovy", ".gradle",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".html",
    ".htm", ".css", ".scss", ".less", ".mod", ".sum",
}
CODE_BASENAMES = {
    "dockerfile", "makefile", "jenkinsfile", "vagrantfile", "gemfile", "rakefile",
    "procfile", ".gitignore", ".dockerignore", ".editorconfig",
}
_CODE_MIME_TYPES = {
    "application/javascript", "application/x-javascript", "application/typescript",
    "application/x-python", "application/x-sh", "application/sql", "application/xml",
    "application/x-httpd-php", "text/x-python", "text/x-java-source", "text/x-c",
    "text/javascript", "text/css", "text/html", "text/xml",
}


def detect_format(filename: str | None, content_type: str | None = None) -> str:
    suffix = PurePosixPath(filename or "").suffix.lower()
    by_suffix = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".txt": "text", ".md": "text", ".csv": "text"}
    if suffix in by_suffix:
        return by_suffix[suffix]
    if suffix in CODE_SUFFIXES:
        return "text"
    by_mime = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "text/plain": "text",
        "text/markdown": "text",
        "text/csv": "text",
    }
    return by_mime.get((content_type or "").split(";", 1)[0].lower(), "text")


def detect_source_kind(filename: str | None, content_type: str | None = None) -> str:
    """Classify an upload as ``code`` or ``document`` for chunking and retrieval.

    Extension wins because browsers often report code files as text/plain;
    content type is only a fallback for extension-less files.
    """
    suffix = PurePosixPath(filename or "").suffix.lower()
    basename = PurePosixPath(filename or "").name.lower()
    if basename in CODE_BASENAMES or suffix in CODE_SUFFIXES:
        return "code"
    if (content_type or "").split(";", 1)[0].strip().lower() in _CODE_MIME_TYPES:
        return "code"
    return "document"


def parse_document(filename: str | None, content: bytes, content_type: str | None = None) -> ParsedDocument:
    format_name = detect_format(filename, content_type)
    try:
        if format_name == "pdf":
            return _parse_pdf(content)
        if format_name == "docx":
            return _parse_docx(content)
        if format_name == "pptx":
            return _parse_pptx(content)
        return _parse_text(content)
    except DocumentParseError:
        raise
    except Exception as exc:  # parser libraries expose format-specific exception types
        raise DocumentParseError(f"{format_name}_parse_failed: {exc}") from exc


def _sanitize(text: str) -> str:
    """Remove NUL characters that PostgreSQL text/JSONB cannot store.

    PDF text extraction occasionally yields embedded ``\\x00`` bytes; a single
    one makes the whole chunk payload untranslatable and aborts ingestion.
    """
    return text.replace("\x00", "")


def _join_blocks(items: list[tuple[str, SourceLocator]], format_name: str) -> ParsedDocument:
    blocks: list[ParsedBlock] = []
    parts: list[str] = []
    cursor = 0
    for value, locator in items:
        text = _sanitize(value).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        blocks.append(ParsedBlock(start=start, end=cursor, locator=locator))
    return ParsedDocument(text="".join(parts), format=format_name, blocks=blocks)


def _parse_text(content: bytes) -> ParsedDocument:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("text_decode_failed: file must be UTF-8") from exc
    normalized = _sanitize(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    locator = SourceLocator(kind="text", line_start=1, line_end=normalized.count("\n") + 1)
    return ParsedDocument(text=normalized, format="text", blocks=[ParsedBlock(0, len(normalized), locator)] if normalized else [])


def _parse_pdf(content: bytes) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentParseError("pdf_parser_dependency_missing: install pypdf") from exc
    reader = PdfReader(BytesIO(content))
    items: list[tuple[str, SourceLocator]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        items.append((text, SourceLocator(kind="pdf", page=page_number)))
    return _join_blocks(items, "pdf")


def _parse_docx(content: bytes) -> ParsedDocument:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentParseError("docx_parser_dependency_missing: install python-docx") from exc
    document = Document(BytesIO(content))
    items: list[tuple[str, SourceLocator]] = []
    block_number = 0
    for paragraph in document.paragraphs:
        block_number += 1
        items.append((paragraph.text, SourceLocator(kind="docx", line_start=block_number, line_end=block_number)))
    for table_number, table in enumerate(document.tables, start=1):
        rows = ["\t".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        items.append(("\n".join(rows), SourceLocator(kind="docx", line_start=block_number + table_number, line_end=block_number + table_number)))
    return _join_blocks(items, "docx")


def _parse_pptx(content: bytes) -> ParsedDocument:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise DocumentParseError("pptx_parser_dependency_missing: install python-pptx") from exc
    presentation = Presentation(BytesIO(content))
    items: list[tuple[str, SourceLocator]] = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        texts = [shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
        items.append(("\n".join(texts), SourceLocator(kind="pptx", page=slide_number)))
    return _join_blocks(items, "pptx")
