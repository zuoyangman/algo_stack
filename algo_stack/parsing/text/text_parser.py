"""Text document parser — 文."""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser

from algo_stack.parsing._types import ContentBlock, Modality, ParsedDocument


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


class TextParser:
    """Parse plain-text documents (.txt, .md, .json, .csv, .html, …)."""

    modality = Modality.TEXT

    def __init__(self, *, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def parse_bytes(self, data: bytes, source: str) -> ParsedDocument:
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else "txt"
        text, meta = self._decode(data, ext)
        blocks = [
            ContentBlock(modality=Modality.TEXT, text=para, metadata={"index": i})
            for i, para in enumerate(self._split_paragraphs(text))
            if para.strip()
        ]
        return ParsedDocument(
            source=source,
            modality=Modality.TEXT,
            text=text,
            metadata={"format": ext, **meta},
            blocks=blocks,
        )

    def _decode(self, data: bytes, ext: str) -> tuple[str, dict]:
        raw = data.decode(self.encoding, errors="replace")
        meta: dict = {"char_count": len(raw), "encoding": self.encoding}

        if ext == "json":
            obj = json.loads(raw)
            meta["json_type"] = type(obj).__name__
            return json.dumps(obj, ensure_ascii=False, indent=2), meta

        if ext == "csv":
            reader = csv.reader(io.StringIO(raw))
            rows = list(reader)
            meta["row_count"] = len(rows)
            meta["col_count"] = len(rows[0]) if rows else 0
            lines = ["\t".join(row) for row in rows]
            return "\n".join(lines), meta

        if ext in ("html", "htm"):
            extractor = _HTMLTextExtractor()
            extractor.feed(raw)
            text = extractor.get_text()
            meta["html_stripped"] = True
            return text, meta

        if ext in ("md", "markdown"):
            meta["markdown"] = True
            return raw, meta

        return raw, meta

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        parts = re.split(r"\n\s*\n", text)
        return parts if parts else [text]
