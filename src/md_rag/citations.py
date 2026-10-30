"""Resolve the model's [n] markers into sources.

The source list comes from this code and never from the model, so a citation
can only ever name a passage that was really retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from md_rag.retrieve import Passage

_NUMBER_LIST = r"\d+(?:\s*[,;]\s*\d+)*"
_MARKER = re.compile(rf"(?P<space>[ \t]*)\[\s*(?P<numbers>{_NUMBER_LIST})\s*\]")
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class Source:
    number: int
    label: str


@dataclass(frozen=True)
class CitedAnswer:
    text: str
    sources: list[Source]
    removed_markers: list[int]

    @property
    def flagged(self) -> bool:
        """True when no claim in the answer can be traced to a document."""
        return not self.sources


def resolve_citations(answer: str, passages: list[Passage]) -> CitedAnswer:
    """Renumber valid markers by first use and delete unknown ones.

    Numbering follows the reader's order, and two passages with the same file
    and heading path become one source.
    """
    by_number = {passage.number: passage for passage in passages}
    numbers_by_label: dict[str, int] = {}
    removed: list[int] = []

    def replace(match: re.Match[str]) -> str:
        markers: list[str] = []
        for digits in _NUMBER.findall(match.group("numbers")):
            number = int(digits)
            passage = by_number.get(number)
            if passage is None:
                removed.append(number)
                continue
            new_number = numbers_by_label.setdefault(passage.label, len(numbers_by_label) + 1)
            marker = f"[{new_number}]"
            if marker not in markers:
                markers.append(marker)
        return match.group("space") + "".join(markers) if markers else ""

    text = _MARKER.sub(replace, answer).strip()
    sources = [Source(number, label) for label, number in numbers_by_label.items()]
    return CitedAnswer(text, sources, removed)


def format_sources(sources: list[Source]) -> str:
    return "\n".join(f"[{source.number}] {source.label}" for source in sources)
