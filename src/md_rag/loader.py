"""Find and read markdown documents."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from md_rag.textnorm import normalize_text


@dataclass(frozen=True)
class Document:
    path: str  # POSIX path relative to the documents directory
    text: str
    sha256: str


def find_markdown(docs_dir: Path, exclude: tuple[str, ...]) -> list[Path]:
    excluded = {name.casefold() for name in exclude}
    root = docs_dir.resolve()
    found = []
    for path in docs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        if path.name.casefold() in excluded:
            continue
        # A symlink must not pull in a file from outside the documents directory.
        if not path.resolve().is_relative_to(root):
            continue
        found.append(path)
    return sorted(found)


def load_documents(docs_dir: Path, exclude: tuple[str, ...]) -> list[Document]:
    documents = []
    for path in find_markdown(docs_dir, exclude):
        raw = path.read_bytes()
        relative = path.relative_to(docs_dir).as_posix()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            print(f"warning: skipping {relative}: not valid UTF-8", file=sys.stderr)
            continue
        documents.append(Document(relative, normalize_text(text), hashlib.sha256(raw).hexdigest()))
    return documents
