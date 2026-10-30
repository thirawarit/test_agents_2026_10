"""The on-disk index: vectors in NumPy, chunks and BM25 tokens in JSON.

A manifest names the data files for the current generation. Replacing the
manifest is the only step that publishes a new index, so a run that dies
midway leaves the previous index readable.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from md_rag.chunker import Chunk, Section, chunk_document
from md_rag.config import Config
from md_rag.loader import Document, load_documents
from md_rag.textnorm import bm25_tokens, document_identifiers

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
EMBED_BATCH_SIZE = 32


class IndexError_(Exception):
    """The index is missing, stale for this config, or can't be built."""


class BM25:
    """Okapi BM25 over pre-tokenized chunks, rebuilt from stored tokens at load."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.count = len(documents)
        self.lengths = np.array([len(tokens) for tokens in documents], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if self.count else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for row, tokens in enumerate(documents):
            for token, frequency in Counter(tokens).items():
                self.postings.setdefault(token, []).append((row, frequency))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.count, dtype=np.float32)
        for token in set(query_tokens):
            postings = self.postings.get(token)
            if not postings:
                continue
            matches = len(postings)
            idf = math.log(1 + (self.count - matches + 0.5) / (matches + 0.5))
            for row, frequency in postings:
                length_norm = 1 - self.b + self.b * self.lengths[row] / (self.average_length or 1.0)
                scores[row] += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
        return scores


@dataclass
class Index:
    sections: dict[str, Section]
    chunks: list[Chunk]
    tokens: list[list[str]]
    vectors: np.ndarray
    files: dict[str, str]
    bm25: BM25 = field(init=False)
    identifiers: Counter[str] = field(init=False)  # identifier -> chunks containing it

    def __post_init__(self) -> None:
        self.bm25 = BM25(self.tokens)
        self.identifiers = Counter()
        for chunk in self.chunks:
            self.identifiers.update(document_identifiers(chunk.embed_text))


@dataclass(frozen=True)
class IndexReport:
    added: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: int
    chunks: int
    embedded: int
    rebuilt_reason: str | None = None


def _fingerprint(config: Config) -> dict[str, Any]:
    """Settings that invalidate stored chunks or vectors when they change."""
    return {
        "schema": SCHEMA_VERSION,
        "embed_model": config.embed_model,
        "chunk_words": config.chunk_words,
        "overlap_words": config.overlap_words,
    }


def _read_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = index_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_generation(index_dir: Path, manifest: dict[str, Any]) -> Index:
    payload = json.loads((index_dir / manifest["chunks_file"]).read_text(encoding="utf-8"))
    vectors = np.load(index_dir / manifest["vectors_file"])
    sections = {
        item["section_id"]: Section(
            item["section_id"], item["file"], tuple(item["heading_path"]), item["text"]
        )
        for item in payload["sections"]
    }
    chunks = [
        Chunk(
            item["chunk_id"],
            item["section_id"],
            item["file"],
            tuple(item["heading_path"]),
            item["text"],
        )
        for item in payload["chunks"]
    ]
    tokens = [item["tokens"] for item in payload["chunks"]]
    if len(chunks) != vectors.shape[0]:
        raise IndexError_(
            f"Index at {index_dir} is inconsistent: {len(chunks)} chunks, "
            f"{vectors.shape[0]} vectors. Run: md-rag index --full"
        )
    return Index(sections, chunks, tokens, vectors, dict(manifest["files"]))


def load_index(config: Config) -> Index:
    manifest = _read_manifest(config.index_dir)
    if manifest is None:
        raise IndexError_(f"No index at {config.index_dir}. Run: md-rag index")
    if manifest.get("fingerprint") != _fingerprint(config):
        raise IndexError_(
            f"Index at {config.index_dir} was built with different settings. Run: md-rag index"
        )
    return _read_generation(config.index_dir, manifest)


def stale_files(config: Config, index: Index) -> list[str]:
    """Return documents that differ from what the index holds."""
    current = {
        document.path: document.sha256
        for document in load_documents(config.docs_dir, config.exclude)
    }
    paths = set(current) | set(index.files)
    return sorted(path for path in paths if current.get(path) != index.files.get(path))


def _embed(embedder: Any, texts: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        rows.extend(embedder.embed_documents(texts[start : start + EMBED_BATCH_SIZE]))
    vectors = np.asarray(rows, dtype=np.float32)
    return normalize_vectors(vectors)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise IndexError_("The embedding model returned an empty or invalid vector")
    return vectors / norms


def _write_generation(config: Config, index: Index) -> None:
    index_dir = config.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    previous = _read_manifest(index_dir)

    generation = uuid.uuid4().hex
    chunks_file = f"chunks-{generation}.json"
    vectors_file = f"vectors-{generation}.npy"
    payload = {
        "sections": [
            {
                "section_id": item.section_id,
                "file": item.file,
                "heading_path": list(item.heading_path),
                "text": item.text,
            }
            for item in index.sections.values()
        ],
        "chunks": [
            {
                "chunk_id": item.chunk_id,
                "section_id": item.section_id,
                "file": item.file,
                "heading_path": list(item.heading_path),
                "text": item.text,
                "tokens": tokens,
            }
            for item, tokens in zip(index.chunks, index.tokens)
        ],
    }
    (index_dir / chunks_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    np.save(index_dir / vectors_file, index.vectors)

    manifest = {
        "fingerprint": _fingerprint(config),
        "chunks_file": chunks_file,
        "vectors_file": vectors_file,
        "files": index.files,
    }
    temporary = index_dir / f"{MANIFEST_NAME}.{generation}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, index_dir / MANIFEST_NAME)

    if previous:
        for name in (previous["chunks_file"], previous["vectors_file"]):
            (index_dir / name).unlink(missing_ok=True)


def _prepare(documents: list[Document], config: Config) -> tuple[list[Section], list[Chunk]]:
    sections: list[Section] = []
    chunks: list[Chunk] = []
    for document in documents:
        new_sections, new_chunks = chunk_document(document, config.chunk_words, config.overlap_words)
        sections.extend(new_sections)
        chunks.extend(new_chunks)
    return sections, chunks


def update_index(config: Config, embedder: Any, full: bool = False) -> IndexReport:
    """Bring the index up to date, embedding only new and changed files."""
    if not config.docs_dir.is_dir():
        raise IndexError_(f"Documents directory not found: {config.docs_dir}")
    documents = load_documents(config.docs_dir, config.exclude)
    if not documents:
        # Refuse rather than publish an empty index over a good one, which is
        # what a mistyped or unmounted docs_dir would otherwise do.
        raise IndexError_(f"No markdown files found in {config.docs_dir}")

    manifest = _read_manifest(config.index_dir)
    previous: Index | None = None
    reason: str | None = None
    if full:
        reason = "requested with --full"
    elif manifest is not None:
        if manifest.get("fingerprint") == _fingerprint(config):
            previous = _read_generation(config.index_dir, manifest)
        else:
            reason = "model or chunking settings changed"

    old_files = previous.files if previous else {}
    current = {document.path: document.sha256 for document in documents}
    added = sorted(path for path in current if path not in old_files)
    changed = sorted(path for path in current if path in old_files and old_files[path] != current[path])
    removed = sorted(path for path in old_files if path not in current)
    unchanged = {path for path in current if old_files.get(path) == current[path]}

    if previous is not None and not added and not changed and not removed:
        return IndexReport([], [], [], len(unchanged), len(previous.chunks), 0)

    to_embed = [document for document in documents if document.path not in unchanged]
    new_sections, new_chunks = _prepare(to_embed, config)
    new_tokens = [bm25_tokens(item.embed_text) for item in new_chunks]
    new_vectors = _embed(embedder, [item.embed_text for item in new_chunks]) if new_chunks else None

    rows = (
        [row for row, item in enumerate(previous.chunks) if item.file in unchanged]
        if previous
        else []
    )
    sections = dict(previous.sections) if previous else {}
    sections = {key: item for key, item in sections.items() if item.file in unchanged}
    sections.update({item.section_id: item for item in new_sections})

    chunks = [previous.chunks[row] for row in rows] if previous else []
    tokens = [previous.tokens[row] for row in rows] if previous else []
    matrices = [previous.vectors[rows]] if previous and rows else []
    chunks += new_chunks
    tokens += new_tokens
    if new_vectors is not None:
        matrices.append(new_vectors)
    if not matrices:
        raise IndexError_(f"No indexable text found in {config.docs_dir}")

    _write_generation(
        config, Index(sections, chunks, tokens, np.concatenate(matrices), current)
    )
    return IndexReport(
        added, changed, removed, len(unchanged), len(chunks), len(new_chunks), reason
    )
