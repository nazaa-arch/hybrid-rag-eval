"""Structure-aware recursive chunker supporting Markdown, Plain Text, and PDF documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybrid_rag.schemas import Chunk, Document
from hybrid_rag.text_utils import normalize_text, split_sentences

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class _SectionBlock:
    section: str
    text: str
    page: int | None
    char_start: int


class StructureAwareChunker:
    """Splits Markdown, text, and PDF files while preserving heading hierarchy and overlap."""

    def __init__(self, chunk_size: int = 680, chunk_overlap: int = 96) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        *,
        doc_id: str,
        title: str,
        content: str,
        source_type: str = "markdown",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Document, list[Chunk]]:
        """Chunk raw markdown or plain-text content into Document + Chunk list."""
        cleaned = normalize_text(content).strip()
        if not cleaned:
            raise ValueError("Document content cannot be empty.")
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        doc_meta = dict(metadata or {})

        if source_type == "markdown":
            blocks = self._extract_markdown_sections(cleaned, default_title=title)
        else:
            blocks = [_SectionBlock(section=title, text=cleaned, page=None, char_start=0)]

        chunks = self._build_chunks_from_blocks(
            doc_id=doc_id,
            title=title,
            blocks=blocks,
            base_metadata=doc_meta,
        )
        if not chunks:
            raise ValueError("Document produced no indexable text chunks.")

        document = Document(
            doc_id=doc_id,
            title=title,
            source_type="markdown" if source_type == "markdown" else "text",
            content_hash=content_hash,
            char_count=len(cleaned),
            chunk_count=len(chunks),
            metadata=doc_meta,
        )
        return document, chunks

    def chunk_pdf_bytes(
        self,
        *,
        doc_id: str,
        title: str,
        pdf_bytes: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Document, list[Chunk]]:
        """Extract text page-by-page from PDF bytes (PyMuPDF with pypdf fallback) and chunk."""
        pages = self._extract_pdf_pages(pdf_bytes)
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
        doc_meta = dict(metadata or {})
        doc_meta["page_count"] = len(pages)

        blocks: list[_SectionBlock] = []
        running_offset = 0
        for page_num, page_text in pages:
            cleaned_page = normalize_text(page_text).strip()
            if not cleaned_page:
                continue
            page_blocks = self._extract_markdown_sections(
                cleaned_page,
                default_title=f"Page {page_num}",
                page=page_num,
                base_offset=running_offset,
            )
            blocks.extend(page_blocks)
            running_offset += len(cleaned_page) + 2

        chunks = self._build_chunks_from_blocks(
            doc_id=doc_id,
            title=title,
            blocks=blocks,
            base_metadata=doc_meta,
        )
        if not chunks:
            raise ValueError("PDF contains no extractable text.")

        total_chars = sum(len(t) for _, t in pages)
        document = Document(
            doc_id=doc_id,
            title=title,
            source_type="pdf",
            content_hash=content_hash,
            char_count=total_chars,
            chunk_count=len(chunks),
            metadata=doc_meta,
        )
        return document, chunks

    def chunk_file(
        self,
        file_path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Document, list[Chunk]]:
        """Chunk a .md, .txt, or .pdf file from disk."""
        suffix = file_path.suffix.lower()
        resolved_doc_id = doc_id or self._slugify(file_path.stem)
        base_meta = {"filename": file_path.name, **(metadata or {})}

        if suffix == ".pdf":
            pdf_bytes = file_path.read_bytes()
            resolved_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()
            return self.chunk_pdf_bytes(
                doc_id=resolved_doc_id,
                title=resolved_title,
                pdf_bytes=pdf_bytes,
                metadata=base_meta,
            )

        raw_text = file_path.read_text(encoding="utf-8")
        source_type = "markdown" if suffix in {".md", ".markdown"} else "text"
        inferred_title = title or self._infer_title(raw_text, fallback=file_path.stem)
        return self.chunk_text(
            doc_id=resolved_doc_id,
            title=inferred_title,
            content=raw_text,
            source_type=source_type,
            metadata=base_meta,
        )

    @staticmethod
    def _extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
        """Extract (1-indexed page number, text) tuples using PyMuPDF (fitz) or pypdf fallback."""
        try:
            import fitz  # type: ignore[import-untyped]

            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                return [(idx + 1, page.get_text("text")) for idx, page in enumerate(doc)]
        except Exception:
            from io import BytesIO

            from pypdf import PdfReader

            try:
                reader = PdfReader(BytesIO(pdf_bytes))
                return [
                    (idx + 1, page.extract_text() or "") for idx, page in enumerate(reader.pages)
                ]
            except Exception as exc:
                raise ValueError("Invalid or corrupted PDF file.") from exc

    def _extract_markdown_sections(
        self,
        text: str,
        *,
        default_title: str,
        page: int | None = None,
        base_offset: int = 0,
    ) -> list[_SectionBlock]:
        """Parse markdown heading hierarchy into section blocks with breadcrumb paths."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [
                _SectionBlock(
                    section=default_title,
                    text=text.strip(),
                    page=page,
                    char_start=base_offset,
                )
            ]

        blocks: list[_SectionBlock] = []
        heading_stack: list[tuple[int, str]] = []

        # Leading text before the first heading
        first_start = matches[0].start()
        if first_start > 0:
            preamble = text[:first_start].strip()
            if preamble:
                blocks.append(
                    _SectionBlock(
                        section=default_title,
                        text=preamble,
                        page=page,
                        char_start=base_offset,
                    )
                )

        for idx, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))

            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            # Build concise breadcrumb (skip duplicating top H1 if deeper headings exist)
            if len(heading_stack) > 1 and heading_stack[0][0] == 1:
                breadcrumb = " > ".join(h[1] for h in heading_stack[1:])
            else:
                breadcrumb = " > ".join(h[1] for h in heading_stack)

            if body:
                blocks.append(
                    _SectionBlock(
                        section=breadcrumb or default_title,
                        text=body,
                        page=page,
                        char_start=base_offset + body_start,
                    )
                )

        return blocks or [
            _SectionBlock(
                section=default_title,
                text=text.strip(),
                page=page,
                char_start=base_offset,
            )
        ]

    def _build_chunks_from_blocks(
        self,
        *,
        doc_id: str,
        title: str,
        blocks: list[_SectionBlock],
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        for block in blocks:
            spans = self._recursive_split(block.text)
            local_cursor = 0
            for span_text in spans:
                span_clean = span_text.strip()
                if not span_clean:
                    continue
                found_at = block.text.find(span_clean[:40], max(0, local_cursor - self.chunk_overlap))
                rel_start = found_at if found_at >= 0 else local_cursor
                char_start = block.char_start + rel_start
                char_end = char_start + len(span_clean)
                local_cursor = rel_start + len(span_clean)

                chunk_id = f"c{chunk_index:03d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=title,
                        section=block.section,
                        page=block.page,
                        chunk_index=chunk_index,
                        char_start=char_start,
                        char_end=char_end,
                        text=span_clean,
                        metadata=dict(base_metadata),
                    )
                )
                chunk_index += 1

        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """Split section text along paragraphs -> sentences -> word boundaries with overlap."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        atomic_units: list[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                atomic_units.append(para)
            else:
                # Split oversized paragraph by newlines or sentences
                lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
                if len(lines) > 1 and all(len(ln) <= self.chunk_size for ln in lines):
                    atomic_units.extend(lines)
                else:
                    sentences = split_sentences(para) or [para]
                    for sent in sentences:
                        if len(sent) <= self.chunk_size:
                            atomic_units.append(sent)
                        else:
                            atomic_units.extend(self._hard_word_split(sent))

        # Pack atomic units into chunks respecting chunk_size and chunk_overlap
        chunks: list[str] = []
        current_units: list[str] = []
        current_len = 0

        for unit in atomic_units:
            sep_len = 2 if current_units else 0
            if current_units and current_len + sep_len + len(unit) > self.chunk_size:
                chunks.append("\n\n".join(current_units))
                # Carry trailing units / tail slice for overlap
                overlap_units: list[str] = []
                overlap_len = 0
                if self.chunk_overlap > 0:
                    for prev in reversed(current_units):
                        if overlap_len + len(prev) <= self.chunk_overlap:
                            overlap_units.insert(0, prev)
                            overlap_len += len(prev) + 2
                        elif not overlap_units:
                            tail = self._tail_on_word_boundary(prev, self.chunk_overlap)
                            if tail:
                                overlap_units.append(tail)
                                overlap_len = len(tail)
                            break
                        else:
                            break
                current_units = overlap_units
                current_len = sum(len(u) for u in current_units) + max(0, len(current_units) - 1) * 2

            current_units.append(unit)
            current_len += (2 if len(current_units) > 1 else 0) + len(unit)

        if current_units:
            final_chunk = "\n\n".join(current_units).strip()
            if final_chunk and (not chunks or final_chunk != chunks[-1]):
                chunks.append(final_chunk)

        return chunks

    def _hard_word_split(self, text: str) -> list[str]:
        """Word-boundary fallback split for unusually long unbroken text."""
        words = text.split()
        pieces: list[str] = []
        buf: list[str] = []
        length = 0
        for word in words:
            add = len(word) + (1 if buf else 0)
            if buf and length + add > self.chunk_size:
                pieces.append(" ".join(buf))
                buf = [word]
                length = len(word)
            else:
                buf.append(word)
                length += add
        if buf:
            pieces.append(" ".join(buf))
        return pieces

    @staticmethod
    def _tail_on_word_boundary(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        sliced = text[-max_chars:]
        first_space = sliced.find(" ")
        if 0 < first_space < len(sliced) - 10:
            return sliced[first_space + 1 :].strip()
        return sliced.strip()

    @staticmethod
    def _infer_title(content: str, fallback: str) -> str:
        match = _HEADING_RE.search(content)
        if match:
            return match.group(2).strip()
        return fallback.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
        return slug or "document"
