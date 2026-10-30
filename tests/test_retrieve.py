import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from md_rag import cli
from md_rag.chunker import Chunk, Section
from md_rag.config import Config, load_config
from md_rag.index import BM25, Index, IndexError_, load_index, stale_files, update_index
from md_rag.retrieve import (
    Retriever,
    expand_to_sections,
    is_relevant,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
    top_rows,
)
from md_rag.textnorm import bm25_tokens

FIXTURES = Path(__file__).parent / "fixtures"
DIMENSIONS = 256


class HashEmbedder:
    """Deterministic bag-of-words vectors, so no Ollama is needed."""

    def __init__(self):
        self.embedded: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(DIMENSIONS)
        for token in bm25_tokens(text) or ["empty"]:
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % DIMENSIONS] += 1.0
        return vector.tolist()

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def make_config(tmp_path: Path, docs: Path, **overrides) -> Config:
    # Hash vectors have no meaningful cosine scale, so the cutoff is off here.
    # The tests under "Relevance cutoff" set it themselves.
    overrides.setdefault("min_relevance", 0.0)
    return Config(docs_dir=docs, index_dir=tmp_path / "index", **overrides)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    target = tmp_path / "docs"
    shutil.copytree(FIXTURES, target)
    return target


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def small_index(chunk_texts: dict[str, list[str]]) -> Index:
    sections, chunks, tokens = {}, [], []
    for number, (heading, texts) in enumerate(chunk_texts.items()):
        section_id = f"a.md#{number}"
        sections[section_id] = Section(section_id, "a.md", (heading,), "\n".join(texts))
        for position, text in enumerate(texts):
            chunks.append(Chunk(f"{section_id}:{position}", section_id, "a.md", (heading,), text))
            tokens.append(bm25_tokens(text))
    vectors = np.eye(len(chunks), dtype=np.float32)
    return Index(sections, chunks, tokens, vectors, {"a.md": "hash"})


def test_first_index_embeds_everything_with_heading_prefix(tmp_path, docs):
    embedder = HashEmbedder()
    report = update_index(make_config(tmp_path, docs), embedder)
    assert report.added == ["english.md", "mixed.md", "thai.md"]
    assert report.embedded == report.chunks == len(embedder.embedded)
    assert any(text.startswith("Platform Handbook > Backup Policy\n\n") for text in embedder.embedded)


def test_second_run_with_no_changes_embeds_nothing(tmp_path, docs):
    config = make_config(tmp_path, docs)
    update_index(config, HashEmbedder())
    embedder = HashEmbedder()
    report = update_index(config, embedder)
    assert embedder.embedded == []
    assert report.unchanged == 3


def test_only_the_changed_file_is_reembedded(tmp_path, docs):
    config = make_config(tmp_path, docs)
    update_index(config, HashEmbedder())
    write(docs / "english.md", "# Handbook\n\nBackups now run hourly.\n")
    embedder = HashEmbedder()
    report = update_index(config, embedder)
    assert report.changed == ["english.md"]
    assert embedder.embedded == ["Handbook\n\nBackups now run hourly."]
    index = load_index(config)
    assert index.vectors.shape[0] == len(index.chunks) == report.chunks
    assert {"thai.md", "mixed.md", "english.md"} == {chunk.file for chunk in index.chunks}
    assert not any("Singapore" in chunk.text for chunk in index.chunks)


def test_kept_vectors_stay_aligned_with_their_chunks(tmp_path, docs):
    config = make_config(tmp_path, docs)
    update_index(config, HashEmbedder())
    write(docs / "aaa.md", "# First\n\nA new file sorted before the others.\n")
    update_index(config, HashEmbedder())
    index = load_index(config)
    embedder = HashEmbedder()
    expected = np.asarray(embedder.embed_documents([c.embed_text for c in index.chunks]), dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(index.vectors, expected)


def test_removed_file_leaves_the_index(tmp_path, docs):
    config = make_config(tmp_path, docs)
    update_index(config, HashEmbedder())
    (docs / "mixed.md").unlink()
    report = update_index(config, HashEmbedder())
    assert report.removed == ["mixed.md"]
    index = load_index(config)
    assert "mixed.md" not in index.files
    assert not any(item.file == "mixed.md" for item in index.sections.values())
    assert not any(chunk.file == "mixed.md" for chunk in index.chunks)


def test_changed_chunk_settings_force_a_rebuild(tmp_path, docs):
    update_index(make_config(tmp_path, docs), HashEmbedder())
    report = update_index(make_config(tmp_path, docs, chunk_words=200), HashEmbedder())
    assert report.rebuilt_reason == "model or chunking settings changed"
    assert report.embedded == report.chunks


def test_index_built_with_another_model_is_refused_at_query_time(tmp_path, docs):
    update_index(make_config(tmp_path, docs), HashEmbedder())
    with pytest.raises(IndexError_, match="different settings"):
        load_index(make_config(tmp_path, docs, embed_model="nomic-embed-text"))


def test_empty_docs_dir_keeps_the_existing_index(tmp_path, docs):
    config = make_config(tmp_path, docs)
    first = update_index(config, HashEmbedder())
    for path in docs.iterdir():
        path.unlink()
    with pytest.raises(IndexError_, match="No markdown files"):
        update_index(config, HashEmbedder())
    assert len(load_index(config).chunks) == first.chunks


def test_failed_embedding_keeps_the_existing_index(tmp_path, docs):
    class Failing(HashEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("model crashed")

    config = make_config(tmp_path, docs)
    first = update_index(config, HashEmbedder())
    write(docs / "english.md", "# Changed\n\ntext\n")
    with pytest.raises(RuntimeError):
        update_index(config, Failing())
    assert len(load_index(config).chunks) == first.chunks
    assert stale_files(config, load_index(config)) == ["english.md"]


def test_old_generation_files_are_cleaned_up(tmp_path, docs):
    config = make_config(tmp_path, docs)
    update_index(config, HashEmbedder())
    write(docs / "english.md", "# Changed\n\ntext\n")
    update_index(config, HashEmbedder())
    names = sorted(path.name.split("-")[0] for path in config.index_dir.iterdir())
    assert names == ["chunks", "manifest.json", "vectors"]


def test_bm25_ranks_rare_term_first():
    bm25 = BM25([["a", "b"], ["a", "rare"], ["a", "c"]])
    assert top_rows(bm25.scores(["rare"]), 3) == [1]


def test_bm25_finds_thai_term_inside_unspaced_text():
    documents = [bm25_tokens("ระบบสำรองข้อมูลทำงานทุกคืน"), bm25_tokens("พนักงานลาพักร้อนได้ปีละสิบสองวัน")]
    assert top_rows(BM25(documents).scores(bm25_tokens("สำรองข้อมูล")), 2) == [0]


def test_top_rows_drops_zero_scores():
    assert top_rows(np.array([0.0, 0.5, 0.0]), 3) == [1]


def test_rrf_rewards_appearing_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4]], k=60)
    assert list(fused)[0] == 3
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)


def test_mmr_prefers_variety_over_a_near_duplicate():
    vectors = np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    fused = {0: 1.0, 1: 0.95, 2: 0.6}
    assert maximal_marginal_relevance(fused, vectors, 2, 0.7) == [0, 2]


def test_mmr_handles_empty_input():
    assert maximal_marginal_relevance({}, np.zeros((0, 2)), 5, 0.7) == []


def test_mmr_respects_keep():
    vectors = np.eye(5, dtype=np.float32)
    fused = {row: 1.0 for row in range(5)}
    assert len(maximal_marginal_relevance(fused, vectors, 3, 0.7)) == 3


def test_chunks_from_one_section_expand_to_one_whole_passage():
    index = small_index({"A": ["one", "two"], "B": ["three"]})
    passages = expand_to_sections(index, [1, 0, 2], token_limit=100)
    assert [item.text for item in passages] == ["one\ntwo", "three"]
    assert [item.number for item in passages] == [1, 2]
    assert passages[0].label == "a.md > A"


def test_oversized_section_keeps_selected_chunks_only():
    long_text = "word " * 200
    index = small_index({"A": [long_text, long_text, "selected"]})
    passages = expand_to_sections(index, [2], token_limit=100)
    assert passages[0].text == "selected"


def test_retrieval_finds_a_topic_in_all_three_fixture_files(tmp_path, docs):
    config = make_config(tmp_path, docs, mmr_keep=8)
    embedder = HashEmbedder()
    update_index(config, embedder)
    passages = Retriever(config, load_index(config), embedder)("backup สำรองข้อมูล").passages
    assert {item.file for item in passages} == {"english.md", "mixed.md", "thai.md"}


def test_query_is_normalized_before_keyword_search(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write(docs / "a.md", "# หัวข้อ\n\nแม่บ้านทำความสะอาดห้องประชุม\n")
    write(docs / "b.md", "# Other\n\nunrelated english text\n")
    config = make_config(tmp_path, docs)
    embedder = HashEmbedder()
    update_index(config, embedder)
    passages = Retriever(config, load_index(config), embedder)("เเม่บ้าน").passages
    assert passages[0].file == "a.md"


# Relevance cutoff


class FixedEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, text):
        return self.vector


def scored_index(chunk_cosines: dict[str, float], query_words: str = "") -> tuple[Index, list[float]]:
    """Build one chunk per heading, each with an exact cosine to the returned query.

    The query vector's last dimension belongs to no chunk, which is what lets
    its cosine with every chunk be set independently.
    """
    index = small_index({heading: [f"{heading} {query_words}"] for heading in chunk_cosines})
    size = len(chunk_cosines)
    cosines = np.array(list(chunk_cosines.values()), dtype=np.float64)
    index.vectors = np.hstack([np.eye(size), np.zeros((size, 1))]).astype(np.float32)
    query = [*cosines.tolist(), float(np.sqrt(1 - (cosines**2).sum()))]
    return index, query


def cutoff_config(tmp_path: Path, minimum: float = 0.5) -> Config:
    return Config(docs_dir=tmp_path, index_dir=tmp_path, min_relevance=minimum)


def test_is_relevant_includes_the_boundary():
    assert is_relevant(0.5, 0.5)
    assert not is_relevant(0.499, 0.5)


def test_question_below_the_cutoff_gets_no_passages(tmp_path):
    index, query = scored_index({"A": 0.49, "B": 0.2})
    result = Retriever(cutoff_config(tmp_path), index, FixedEmbedder(query))("anything")
    assert result.passages == []
    assert result.best_score == pytest.approx(0.49, abs=1e-5)


def test_question_above_the_cutoff_gets_passages(tmp_path):
    index, query = scored_index({"A": 0.51, "B": 0.2})
    result = Retriever(cutoff_config(tmp_path), index, FixedEmbedder(query))("anything")
    assert [item.heading_path for item in result.passages][0] == ("A",)


def test_cutoff_is_a_gate_and_keeps_other_candidates_once_passed(tmp_path):
    index, query = scored_index({"A": 0.51, "B": 0.2, "C": 0.1})
    result = Retriever(cutoff_config(tmp_path), index, FixedEmbedder(query))("anything")
    assert len(result.passages) == 3


def test_top_fused_rank_does_not_pass_the_cutoff(tmp_path):
    # This chunk is first in both rankings, so RRF would give it the best
    # possible score, yet the cosine says it isn't relevant.
    index, query = scored_index({"A": 0.3}, query_words="exact keyword match")
    assert list(reciprocal_rank_fusion([[0], [0]], 60)) == [0]
    result = Retriever(cutoff_config(tmp_path), index, FixedEmbedder(query))("exact keyword match")
    assert result.passages == []


def test_zero_turns_the_cutoff_off(tmp_path):
    index, query = scored_index({"A": 0.01})
    result = Retriever(cutoff_config(tmp_path, 0.0), index, FixedEmbedder(query))("anything")
    assert len(result.passages) == 1


def test_uncovered_question_through_the_cli_makes_no_model_call(tmp_path, docs, monkeypatch, capsys):
    class FailingChat:
        def invoke(self, messages):
            raise AssertionError("a refused question must not reach the model")

    settings = tmp_path / "config.toml"
    settings.write_text(
        f'[paths]\ndocs_dir = "{docs}"\nindex_dir = "{tmp_path / "index"}"\n'
        "[retrieval]\nmin_relevance = 0.99\n",
        encoding="utf-8",
    )
    update_index(load_config(settings), HashEmbedder())
    monkeypatch.setattr(cli, "ensure_local_model", lambda *args: {})
    monkeypatch.setattr(cli, "make_embedder", lambda *args: HashEmbedder())
    monkeypatch.setattr(cli, "make_chat", lambda *args: FailingChat())

    assert cli.main(["--config", str(settings), "ask", "cafeteria lunch menu"]) == cli.EXIT_NOT_COVERED
    captured = capsys.readouterr()
    assert captured.out == "The documents don't contain information on this topic.\n"
    assert "min_relevance is 0.99" in captured.err


# Rare-term pass


DOCUMENT_TEXT = [
    "Files are kept on MinIO for 14 days.",
    "SEV3 means a minor defect in the policy of the office.",
    "SELECT pg_size_pretty(pg_database_size('phoenix'));",
    "บริษัทมีศูนย์ข้อมูลที่เชียงใหม่ ทีมงานทำงานทุกไตรมาส",
]


def ask_below_cutoff(tmp_path, question, texts=DOCUMENT_TEXT, **overrides):
    """Run retrieval where every chunk scores 0.05, far below the cutoff."""
    index = small_index({f"Heading{number}": [text] for number, text in enumerate(texts)})
    size = len(texts)
    index.vectors = np.hstack([np.eye(size), np.zeros((size, 1))]).astype(np.float32)
    query = [0.05] * size + [float(np.sqrt(1 - size * 0.05**2))]
    config = Config(docs_dir=tmp_path, index_dir=tmp_path, **overrides)
    return Retriever(config, index, FixedEmbedder(query))(question)


@pytest.mark.parametrize(
    ("question", "term"),
    [
        ("MinIO", "MinIO"),
        ("minio", "minio"),
        ("SEV3", "SEV3"),
        ("pg_size_pretty", "pg_size_pretty"),
        ("What does pg_size_pretty do here?", "pg_size_pretty"),
        ("MinIO ใช้ทำอะไร", "MinIO"),
        ("SEV3 คืออะไร", "SEV3"),
    ],
)
def test_rare_identifier_passes_a_low_scoring_question(tmp_path, question, term):
    result = ask_below_cutoff(tmp_path, question)
    assert result.passages
    assert result.rare_terms == (term,)


@pytest.mark.parametrize(
    "question",
    [
        "What is the company policy on parental leave?",
        "What is the dress code at the office?",
        "How do I select a health plan?",
        "What is the size of the bonus?",
        "Is it pretty?",
    ],
)
def test_common_words_in_a_single_chunk_do_not_pass(tmp_path, question):
    assert ask_below_cutoff(tmp_path, question).passages == []


def test_rare_thai_term_passes(tmp_path):
    result = ask_below_cutoff(tmp_path, "เชียงใหม่")
    assert result.passages
    assert result.rare_terms == ("เชียงใหม่",)


def test_rare_thai_term_with_stopwords_passes(tmp_path):
    assert ask_below_cutoff(tmp_path, "ไตรมาสคืออะไร").rare_terms == ("ไตรมาส",)


@pytest.mark.parametrize("question", ["บริษัท", "ทำงาน", "ของ"])
def test_common_thai_words_in_a_single_chunk_do_not_pass(tmp_path, question):
    assert ask_below_cutoff(tmp_path, question).passages == []


@pytest.mark.xfail(reason="The word list comes from textbooks, so it rates ทีมงาน as rare", strict=True)
def test_workplace_thai_word_does_not_pass(tmp_path):
    assert ask_below_cutoff(tmp_path, "ทีมงานทำงาน").passages == []


def test_rare_thai_term_fails_when_another_content_word_is_missing(tmp_path):
    assert ask_below_cutoff(tmp_path, "เชียงใหม่มีร้านอาหารอะไรบ้าง").passages == []


def test_thai_frequency_limit_is_configurable(tmp_path):
    assert ask_below_cutoff(tmp_path, "บริษัท", rare_term_max_thai_freq=300).rare_terms == ("บริษัท",)
    assert ask_below_cutoff(tmp_path, "เชียงใหม่", rare_term_max_thai_freq=100).passages == []


def test_a_term_in_many_chunks_is_not_rare(tmp_path):
    texts = [f"MinIO note {number}" for number in range(3)] + [f"other {number}" for number in range(37)]
    assert ask_below_cutoff(tmp_path, "MinIO", texts).passages == []


def test_rarity_limit_scales_with_the_corpus_and_is_configurable(tmp_path):
    texts = [f"MinIO note {number}" for number in range(3)] + [f"other {number}" for number in range(37)]
    assert ask_below_cutoff(tmp_path, "MinIO", texts, rare_term_max_fraction=0.1).passages


def test_rare_term_absent_from_the_documents_does_not_pass(tmp_path):
    assert ask_below_cutoff(tmp_path, "SEV9").passages == []


def test_zero_turns_the_rare_term_rule_off(tmp_path):
    assert ask_below_cutoff(tmp_path, "MinIO", rare_term_max_fraction=0).passages == []


def test_question_passing_on_cosine_reports_no_rare_terms(tmp_path):
    index, query = scored_index({"A": 0.51}, query_words="MinIO")
    result = Retriever(cutoff_config(tmp_path), index, FixedEmbedder(query))("MinIO")
    assert result.passages
    assert result.rare_terms == ()


def test_heading_text_counts_as_document_text(tmp_path):
    index = small_index({"MinIO": ["storage notes"]})
    assert index.identifiers["minio"] == 1


def test_requirement_file_is_never_indexed(tmp_path, docs):
    write(docs / "REQ.local.md", "# Requirement\n\nsecret requirement text\n")
    config = make_config(tmp_path, docs)
    update_index(config, HashEmbedder())
    index = load_index(config)
    assert "REQ.local.md" not in index.files
    assert not any("secret requirement" in chunk.text for chunk in index.chunks)
