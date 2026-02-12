# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

import re
from uuid import UUID, uuid4

from .contracts import Chunk, SourceLocator, SourceSpan
from .document_parser import ParsedBlock, ParsedDocument

_SENTENCE_ENDINGS = "。！？!?.；;\n…"
_CLOSERS = "\"'”’）)】》]›»"
_MIN_CUT_RATIO = 0.3


class SemanticChunker:
    """语义完整性优先的自适应切片：段落边界 > 句读/标点 > 空白 > 硬切。"""

    def __init__(self, target_chars: int = 900, overlap_chars: int = 120) -> None:
        if target_chars <= overlap_chars:
            raise ValueError("target_chars must be greater than overlap_chars")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def split(
        self,
        text: str,
        document_id: UUID,
        version_id: UUID,
        title: str,
        pipeline_revision: str,
        parsed_document: ParsedDocument | None = None,
    ) -> list[Chunk]:
        normalized = (parsed_document.text if parsed_document is not None else text).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        blocks = parsed_document.blocks if parsed_document is not None else None
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]

        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) > self.target_chars:
                units.extend(self._split_long_paragraph(paragraph))
            else:
                units.append(paragraph)

        chunks: list[Chunk] = []
        cursor = 0
        carry = ""
        for unit in units:
            candidate = f"{carry}\n{unit}" if carry else unit
            if len(candidate) <= self.target_chars:
                carry = candidate
                continue
            if carry:
                chunks.append(self._chunk(carry, normalized, cursor, document_id, version_id, title, pipeline_revision, blocks))
                overlap = self._overlap_tail(carry)
                cursor = max(0, cursor + len(carry) - len(overlap))
                carry = overlap
                if carry:
                    candidate = f"{carry}\n{unit}"
                    if len(candidate) <= self.target_chars:
                        carry = candidate
                        continue
            carry = unit
        if carry:
            chunks.append(self._chunk(carry, normalized, cursor, document_id, version_id, title, pipeline_revision, blocks))
        for index, chunk in enumerate(chunks):
            chunk.ordinal = index
        return chunks

    def _split_long_paragraph(self, text: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        n = len(text)
        while n - start > self.target_chars:
            cut = self._find_cut(text, start)
            piece = text[start:cut].strip()
            if piece:
                pieces.append(piece)
            start = cut
            while start < n and text[start].isspace():
                start += 1
        tail = text[start:].strip()
        if tail:
            if pieces and len(tail) < self.overlap_chars and len(pieces[-1]) + len(tail) + 1 <= self.target_chars:
                pieces[-1] = f"{pieces[-1]} {tail}"
            else:
                pieces.append(tail)
        return pieces if pieces else [text]

    def _find_cut(self, text: str, start: int) -> int:
        limit = start + self.target_chars
        min_cut = start + max(1, int(self.target_chars * _MIN_CUT_RATIO))
        i = limit - 1
        while i >= min_cut:
            if text[i] in _SENTENCE_ENDINGS:
                cut = i + 1
                while cut < limit and text[cut] in _CLOSERS:
                    cut += 1
                return cut
            i -= 1
        i = limit - 1
        while i >= min_cut:
            if text[i].isspace():
                return i
            i -= 1
        return limit

    def _overlap_tail(self, text: str) -> str:
        if self.overlap_chars <= 0 or len(text) <= self.overlap_chars:
            return ""
        window_start = len(text) - self.overlap_chars
        for i in range(window_start, len(text)):
            if text[i].isspace():
                window_start = i + 1
                break
        tail = text[window_start:].strip()
        return tail if tail else text[len(text) - self.overlap_chars :].strip()

    def _chunk(
        self,
        text: str,
        source: str,
        start_hint: int,
        document_id: UUID,
        version_id: UUID,
        title: str,
        pipeline_revision: str,
        blocks: list[ParsedBlock] | None,
    ) -> Chunk:
        start = source.find(text[: min(40, len(text))], max(0, start_hint))
        start = start if start >= 0 else start_hint
        end = min(len(source), start + len(text))
        source_spans = self._source_spans(text, source, start, end, blocks)
        return Chunk(
            document_id=document_id,
            version_id=version_id,
            title=title,
            text=text,
            source_spans=source_spans,
            pipeline_revision=pipeline_revision,
        )

    @staticmethod
    def _source_spans(text: str, source: str, start: int, end: int, blocks: list[ParsedBlock] | None) -> list[SourceSpan]:
        if blocks:
            spans = []
            for block in blocks:
                overlap_start = max(start, block.start)
                overlap_end = min(end, block.end)
                if overlap_start < overlap_end:
                    spans.append(
                        SourceSpan(
                            block_id=uuid4(),
                            text_start=overlap_start - start,
                            text_end=overlap_end - start,
                            locator=block.locator,
                        )
                    )
            if spans:
                return spans
        locator = SourceLocator(kind="text", line_start=source[:start].count("\n") + 1, line_end=source[:end].count("\n") + 1)
        return [SourceSpan(block_id=uuid4(), text_start=0, text_end=len(text), locator=locator)]


_CODE_DECLARATION = re.compile(
    r"^\s*(?:"
    r"(?:async\s+)?def\s+[A-Za-z_]\w*"
    r"|class\s+[A-Za-z_]\w*"
    r"|(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+[A-Za-z_]\w*"
    r"|(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+[A-Za-z_]\w*"
    r"|(?:export\s+)?(?:const|let|var)\s+[A-Za-z_$]\w*\s*="
    r"|(?:export\s+)?(?:declare\s+)?(?:interface|type|enum|namespace)\s+[A-Za-z_]\w*"
    r"|func\s+(?:\([^)]*\)\s*)?[A-Za-z_]\w*"
    r"|type\s+[A-Za-z_]\w*\s+(?:struct|interface)"
    r"|(?:pub\s+)?(?:async\s+)?fn\s+[A-Za-z_]\w*"
    r"|(?:pub\s+)?(?:struct|enum|impl|trait|mod|union)\s+[A-Za-z_]\w*"
    r"|(?:public|private|protected|internal|static|final|abstract|sealed|synchronized|override|virtual|async)\b[^\n]*\b(?:class|interface|enum|record|struct)\b"
    r"|(?:public|private|protected|internal|static|final|abstract|synchronized|override|virtual|async)\b[^\n]*[A-Za-z_]\w*\s*\("
    r"|(?:public|private|protected)\s+(?:static\s+)?function\s+[A-Za-z_]\w*"
    r"|function\s+[A-Za-z_]\w*"
    r"|CREATE\s+(?:TABLE|INDEX|VIEW|FUNCTION|PROCEDURE|TRIGGER)\b"
    r"|(?:SELECT|INSERT|UPDATE|DELETE)\s+"
    r")\s*",
    re.IGNORECASE,
)


class CodeChunker:
    """代码切片：优先在声明边界切开（函数/类/方法），保留原始行号。

    与 ``SemanticChunker`` 不同，代码不做滑动重叠（重复代码会污染引用），
    而是在声明行或达到 max_chars 时切分；过小的尾部片段并入前一段。
    """

    def __init__(self, target_chars: int = 1200, max_chars: int = 2400, min_chars: int = 300) -> None:
        if not (0 < min_chars <= target_chars <= max_chars):
            raise ValueError("require 0 < min_chars <= target_chars <= max_chars")
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split(
        self,
        text: str,
        document_id: UUID,
        version_id: UUID,
        title: str,
        pipeline_revision: str,
        parsed_document: ParsedDocument | None = None,
    ) -> list[Chunk]:
        normalized = (parsed_document.text if parsed_document is not None else text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not normalized.strip():
            return []
        lines = normalized.split("\n")
        units = self._units(lines)
        chunks: list[Chunk] = []
        for start, end in units:
            body = "\n".join(lines[start:end]).strip("\n")
            if not body.strip():
                continue
            chunks.append(Chunk(
                document_id=document_id,
                version_id=version_id,
                title=title,
                text=body,
                source_kind="code",
                source_spans=[SourceSpan(
                    block_id=uuid4(),
                    text_start=0,
                    text_end=len(body),
                    locator=SourceLocator(kind="text", line_start=start + 1, line_end=end),
                )],
                pipeline_revision=pipeline_revision,
            ))
        for index, chunk in enumerate(chunks):
            chunk.ordinal = index
        return chunks

    def _units(self, lines: list[str]) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        start = 0
        size = 0
        for index, line in enumerate(lines):
            size += len(line) + 1
            if index + 1 < len(lines) and _CODE_DECLARATION.match(line) and size >= self.min_chars:
                units.append((start, index))
                start = index
                size = 0
                continue
            if size >= self.max_chars:
                units.append((start, index + 1))
                start = index + 1
                size = 0
        if start < len(lines):
            units.append((start, len(lines)))
        return self._merge_small_tail(units, lines)

    def _merge_small_tail(self, units: list[tuple[int, int]], lines: list[str]) -> list[tuple[int, int]]:
        if len(units) < 2:
            return units
        start, end = units[-1]
        tail = "\n".join(lines[start:end])
        prev_start, _prev_end = units[-2]
        if len(tail.strip()) >= self.min_chars // 2:
            return units
        if len("\n".join(lines[prev_start:end])) > self.max_chars:
            return units
        return units[:-2] + [(prev_start, end)]
