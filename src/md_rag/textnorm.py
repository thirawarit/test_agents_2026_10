"""Normalization and tokenization for mixed Thai/English text.

The same functions run on documents and queries, so both sides of every
comparison see identical text and tokens.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache

from pythainlp.corpus import thai_stopwords, ttc
from pythainlp.tokenize import sent_tokenize, word_tokenize
from pythainlp.util import normalize as thai_normalize

_THAI_RANGE = "฀-๿"
_THAI_RUN = re.compile(f"[{_THAI_RANGE}]+")
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
# One Thai run, or one word of anything else (Latin, digits, other scripts).
_RUN = re.compile(f"(?P<thai>[{_THAI_RANGE}]+)|(?P<latin>[^\\W_{_THAI_RANGE}]+)")
_LATIN_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_THAI_REPEAT_MARKS = {"ๆ", "ฯ"}
# A Latin term with its joining characters, which BM25 tokens split apart.
_LITERAL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_./\-]*[A-Za-z0-9])?")
_LOWER_THEN_UPPER = re.compile(r"[a-z][A-Z]")
_LETTER_THEN_DIGIT = re.compile(r"^[A-Za-z].*[0-9]")
_FILE_SUFFIX = re.compile(r"^[A-Za-z0-9_\-]{2,}\.[a-z0-9]{1,5}$")

THAI_WORD_ENGINE = "newmm"
THAI_SENTENCE_ENGINE = "crfcut"


def normalize_text(text: str) -> str:
    """Apply NFC, unify line endings, and normalize Thai spelling.

    pythainlp's normalize collapses spaces and newlines, which would destroy
    markdown structure, so it only ever sees runs of Thai characters.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    return _THAI_RUN.sub(lambda match: thai_normalize(match.group()), text)


def split_runs(text: str) -> list[tuple[str, str]]:
    """Split text into ("thai", run) and ("latin", word) pairs."""
    return [(match.lastgroup or "latin", match.group()) for match in _RUN.finditer(text)]


def word_tokens(text: str) -> list[str]:
    """Return word units, sending each run to the tokenizer for its script."""
    tokens: list[str] = []
    for script, run in split_runs(text):
        if script == "thai":
            tokens.extend(
                token
                for token in word_tokenize(run, engine=THAI_WORD_ENGINE, keep_whitespace=False)
                if token not in _THAI_REPEAT_MARKS
            )
        else:
            tokens.append(run)
    return tokens


def bm25_tokens(text: str) -> list[str]:
    """Return lowercased word units for the keyword index."""
    return [token.lower() for token in word_tokens(text)]


def is_identifier(word: str) -> bool:
    """Return True for a Latin term shaped like a name, not an ordinary word.

    Rarity alone can't show this: in a small corpus "policy" and "office" appear
    in a single chunk, just as "MinIO" does. Shape is what tells them apart.
    """
    letters = [char for char in word if char.isalpha()]
    return bool(
        "_" in word
        or _LOWER_THEN_UPPER.search(word)
        or _LETTER_THEN_DIGIT.search(word)
        or _FILE_SUFFIX.match(word)
        or (len(letters) >= 2 and word.isupper())
    )


def literal_terms(text: str) -> list[str]:
    return _LITERAL.findall(text)


def identifier_key(word: str) -> str:
    """Return the lookup form of a document identifier.

    Plain all-caps words keep their case, so "SELECT" in a code block can't match
    the verb in "select a plan". Other identifiers match in any case, so "minio"
    finds "MinIO".
    """
    return word if word.isalpha() and word.isupper() else word.lower()


def document_identifiers(text: str) -> set[str]:
    return {identifier_key(word) for word in literal_terms(text) if is_identifier(word)}


def is_thai_word(token: str) -> bool:
    return bool(_THAI_RUN.search(token))


def is_thai_stopword(token: str) -> bool:
    return token in thai_stopwords()


@cache
def _thai_frequencies() -> dict[str, int]:
    return dict(ttc.word_freqs())


def thai_word_frequency(token: str) -> int:
    """Return a word's count in the Thai Textbook Corpus list bundled with PyThaiNLP."""
    return _thai_frequencies().get(token, 0)


def word_count(text: str) -> int:
    return len(word_tokens(text))


def estimate_tokens(text: str) -> int:
    """Estimate gemma3 prompt tokens, rounded up to plan context safely.

    Measured on Ollama: about 0.37 tokens per Thai character and 0.20 per
    other character. The rates below sit just above those. Digits and
    punctuation cost about a token each, so they are charged in full: citation
    lists like [1, 2, 3] are mostly digits, and at the plain rate they left
    notes undercounted by 1.8x.
    """
    thai = symbols = 0
    for char in text:
        if "฀" <= char <= "๿":
            thai += 1
        elif char.isascii() and not char.isalpha() and not char.isspace():
            symbols += 1
    return int(thai * 0.40 + symbols + (len(text) - thai - symbols) * 0.25) + 1


def split_sentences(text: str) -> list[str]:
    """Split text into sentences that concatenate back to the input.

    Lines containing Thai go through the Thai sentence model, because Thai
    marks sentences with spaces and an English word mid-sentence must not end
    it. Any Latin sentence boundaries inside the result are then split by rule.
    """
    sentences: list[str] = []
    for line in text.splitlines(keepends=True):
        # The Thai model drops trailing whitespace, so set it aside first.
        body = line.rstrip()
        ending = line[len(body) :]
        pieces = sent_tokenize(body, engine=THAI_SENTENCE_ENGINE) if _THAI_RUN.search(body) else [body]
        split_line = [part for piece in pieces for part in _split_latin(piece)]
        if split_line:
            split_line[-1] += ending
        elif ending:
            split_line = [ending]
        sentences.extend(split_line)
    return sentences


def _split_latin(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _LATIN_BOUNDARY.finditer(text):
        parts.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return parts


def question_language(text: str) -> str:
    """Return "th" for a Thai question, otherwise "other".

    English product names can outnumber the Thai letters in a Thai question,
    so Thai wins once it is at least half as frequent as other letters.
    """
    thai = sum(1 for char in text if "฀" <= char <= "๿" and char.isalpha())
    other = sum(1 for char in text if char.isalpha() and not "฀" <= char <= "๿")
    return "th" if thai and thai >= other * 0.5 else "other"
