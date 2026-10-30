"""Hybrid retrieval: dense plus BM25, fused, diversified, expanded to sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from md_rag.config import Config
from md_rag.index import Index, normalize_vectors
from md_rag.textnorm import bm25_tokens, estimate_tokens, normalize_text


@dataclass(frozen=True)
class Passage:
    number: int  # the [n] marker the model cites
    file: str
    heading_path: tuple[str, ...]
    text: str

    @property
    def label(self) -> str:
        return " > ".join((self.file, *self.heading_path))


def top_rows(scores: np.ndarray, count: int) -> list[int]:
    """Return rows with a positive score, best first."""
    order = np.argsort(-scores, kind="stable")[:count]
    return [int(row) for row in order if scores[row] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> dict[int, float]:
    """Merge ranked lists by rank, since cosine and BM25 scores aren't comparable."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank)
    return dict(sorted(fused.items(), key=lambda item: (-item[1], item[0])))


def maximal_marginal_relevance(
    fused: dict[int, float], vectors: np.ndarray, keep: int, relevance_weight: float
) -> list[int]:
    """Select varied chunks so a topic summary can draw on several files."""
    if not fused:
        return []
    rows = list(fused)
    best = max(fused.values())
    relevance = np.array([fused[row] / best for row in rows], dtype=np.float32)
    similarity = vectors[rows] @ vectors[rows].T

    selected: list[int] = []
    remaining = list(range(len(rows)))
    while remaining and len(selected) < keep:
        if selected:
            redundancy = similarity[np.ix_(remaining, selected)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining), dtype=np.float32)
        scores = relevance_weight * relevance[remaining] - (1 - relevance_weight) * redundancy
        choice = remaining[int(np.argmax(scores))]
        selected.append(choice)
        remaining.remove(choice)
    return [rows[position] for position in selected]


def expand_to_sections(index: Index, rows: list[int], token_limit: int) -> list[Passage]:
    """Replace chunks with their full sections, up to a per-section limit.

    An oversized section contributes only its selected chunks, in document
    order, until the limit is reached. The first always gets through, even
    when oversized, because a code fence or table is never cut.
    """
    selected_by_section: dict[str, list[int]] = {}
    for row in rows:
        selected_by_section.setdefault(index.chunks[row].section_id, []).append(row)

    passages: list[Passage] = []
    for number, (section_id, section_rows) in enumerate(selected_by_section.items(), start=1):
        section = index.sections[section_id]
        text = section.text
        if estimate_tokens(text) > token_limit:
            kept: list[str] = []
            used = 0
            for row in sorted(section_rows):
                chunk_text = index.chunks[row].text
                cost = estimate_tokens(chunk_text)
                if kept and used + cost > token_limit:
                    continue
                kept.append(chunk_text)
                used += cost
            text = "\n\n".join(kept)
        passages.append(Passage(number, section.file, section.heading_path, text))
    return passages


class Retriever:
    def __init__(self, config: Config, index: Index, embedder: Any) -> None:
        self.config = config
        self.index = index
        self.embedder = embedder

    def __call__(self, question: str) -> list[Passage]:
        config = self.config
        query = normalize_text(question)
        vector = normalize_vectors(np.asarray(self.embedder.embed_query(query), dtype=np.float32))
        dense = top_rows(self.index.vectors @ vector, config.dense_top_k)
        keyword = top_rows(self.index.bm25.scores(bm25_tokens(query)), config.bm25_top_k)
        fused = reciprocal_rank_fusion([dense, keyword], config.rrf_k)
        selected = maximal_marginal_relevance(
            fused, self.index.vectors, config.mmr_keep, config.mmr_lambda
        )
        return expand_to_sections(self.index, selected, config.section_token_limit)
