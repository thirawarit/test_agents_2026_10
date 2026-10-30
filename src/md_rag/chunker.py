"""Heading-aware markdown chunking.

A document becomes sections, one per heading. A section that fits stays whole
as a single chunk. A longer one is packed from sentences, with code fences and
tables kept intact even when they exceed the chunk size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from md_rag.loader import Document
from md_rag.textnorm import split_sentences, word_count, word_tokens

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_ATX_CLOSING = re.compile(r"[ \t]+#+$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT_RULE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_TABLE_RULE = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")


@dataclass(frozen=True)
class Section:
    section_id: str
    file: str
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_id: str
    file: str
    heading_path: tuple[str, ...]
    text: str

    @property
    def embed_text(self) -> str:
        """Chunk text prefixed with its heading path."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text


@dataclass(frozen=True)
class _Atom:
    text: str
    words: int
    splittable: bool  # True for prose sentences, False for code and tables


def _fence_closes(line: str, opener: str) -> bool:
    stripped = line.strip()
    return (
        bool(stripped)
        and set(stripped) == {opener[0]}
        and len(stripped) >= len(opener)
    )


def split_sections(document: Document) -> list[Section]:
    lines = document.text.split("\n")
    start = _front_matter_end(lines)
    stack: list[tuple[int, str]] = []
    buffer: list[str] = lines[:start]
    sections: list[Section] = []
    fence: str | None = None

    def flush() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            heading_path = tuple(title for _, title in stack)
            section_id = f"{document.path}#{len(sections)}"
            sections.append(Section(section_id, document.path, heading_path, text))

    def open_heading(level: int, title: str) -> None:
        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    for line in lines[start:]:
        if fence is not None:
            buffer.append(line)
            if _fence_closes(line, fence):
                fence = None
            continue

        opener = _FENCE.match(line)
        if opener:
            fence = opener.group(1)
            buffer.append(line)
            continue

        atx = _ATX_HEADING.match(line)
        if atx:
            title = _ATX_CLOSING.sub("", (atx.group(2) or "").strip()).strip()
            open_heading(len(atx.group(1)), title or "(untitled)")
            continue

        rule = _SETEXT_RULE.match(line)
        if rule and buffer and buffer[-1].strip() and not _is_table_line(buffer[-1]):
            title = buffer.pop().strip()
            open_heading(1 if rule.group(1)[0] == "=" else 2, title)
            continue

        buffer.append(line)

    flush()
    return sections


def _front_matter_end(lines: list[str]) -> int:
    """Return the line after a closing YAML marker, so it can't read as a heading."""
    if not lines or lines[0].strip() != "---":
        return 0
    for number, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return number + 1
    return 0


def _is_table_line(line: str) -> bool:
    return "|" in line


def _block_atoms(text: str) -> list[_Atom]:
    """Split section text into whole code fences, whole tables, and sentences."""
    atoms: list[_Atom] = []
    lines = text.split("\n")
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            joined = "\n".join(prose) + "\n"
            prose.clear()
            for sentence in split_sentences(joined):
                if sentence.strip():
                    atoms.append(_Atom(sentence, word_count(sentence), True))

    def add_whole(block: list[str]) -> None:
        flush_prose()
        joined = "\n".join(block) + "\n"
        atoms.append(_Atom(joined, word_count(joined), False))

    index = 0
    while index < len(lines):
        line = lines[index]
        opener = _FENCE.match(line)
        if opener:
            end = index + 1
            while end < len(lines) and not _fence_closes(lines[end], opener.group(1)):
                end += 1
            add_whole(lines[index : end + 1])
            index = end + 1
            continue
        if (
            _is_table_line(line)
            and index + 1 < len(lines)
            and _TABLE_RULE.match(lines[index + 1])
            and "-" in lines[index + 1]
        ):
            end = index + 2
            while end < len(lines) and _is_table_line(lines[end]) and lines[end].strip():
                end += 1
            add_whole(lines[index:end])
            index = end
            continue
        prose.append(line)
        index += 1

    flush_prose()
    return atoms


def _split_long_sentence(atom: _Atom, limit: int) -> list[_Atom]:
    """Cut a sentence longer than the chunk size at word boundaries.

    Unsegmented text, such as a long list on one line, can arrive as a single
    sentence. Offsets come from walking the original text so spacing survives.
    """
    pieces: list[_Atom] = []
    tokens = word_tokens(atom.text)
    position = 0
    cut_points = []
    for count, token in enumerate(tokens, start=1):
        position = atom.text.find(token, position) + len(token)
        if count % limit == 0:
            cut_points.append(position)
    start = 0
    for cut in [*cut_points, len(atom.text)]:
        piece = atom.text[start:cut]
        if piece.strip():
            pieces.append(_Atom(piece, word_count(piece), True))
        start = cut
    return pieces


def _pack(atoms: list[_Atom], chunk_words: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    current: list[_Atom] = []
    words = 0
    fresh = False  # True once `current` holds something not already emitted

    def emit() -> None:
        nonlocal fresh
        if fresh:
            chunks.append("".join(atom.text for atom in current).strip())
        fresh = False

    def carry_overlap() -> list[_Atom]:
        # Code and tables are never repeated: only trailing sentences carry over.
        kept: list[_Atom] = []
        total = 0
        for atom in reversed(current):
            if not atom.splittable or total + atom.words > overlap_words:
                break
            kept.insert(0, atom)
            total += atom.words
        return kept

    for atom in atoms:
        if words + atom.words > chunk_words and current:
            emit()
            current = carry_overlap()
            words = sum(item.words for item in current)
            if words + atom.words > chunk_words:
                current, words = [], 0
        current.append(atom)
        words += atom.words
        fresh = True
    emit()
    return chunks


def chunk_section(section: Section, chunk_words: int, overlap_words: int) -> list[Chunk]:
    if word_count(section.text) <= chunk_words:
        texts = [section.text]
    else:
        atoms: list[_Atom] = []
        for atom in _block_atoms(section.text):
            if atom.splittable and atom.words > chunk_words:
                atoms.extend(_split_long_sentence(atom, chunk_words))
            else:
                atoms.append(atom)
        texts = _pack(atoms, chunk_words, overlap_words)
    return [
        Chunk(f"{section.section_id}:{number}", section.section_id, section.file, section.heading_path, text)
        for number, text in enumerate(texts)
    ]


def chunk_document(
    document: Document, chunk_words: int, overlap_words: int
) -> tuple[list[Section], list[Chunk]]:
    sections = split_sections(document)
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunk_section(section, chunk_words, overlap_words))
    return sections, chunks
