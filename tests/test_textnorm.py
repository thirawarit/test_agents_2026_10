import pytest

from md_rag.textnorm import (
    bm25_tokens,
    document_identifiers,
    estimate_tokens,
    is_identifier,
    is_thai_stopword,
    literal_terms,
    normalize_text,
    question_language,
    split_runs,
    split_sentences,
    thai_word_frequency,
    word_tokens,
)


def test_normalize_fixes_thai_spelling_variants():
    assert normalize_text("เเม่") == "แม่"
    assert normalize_text("นํา") == "นำ"


def test_normalize_applies_nfc_to_latin():
    assert normalize_text("Café") == "Café"


def test_normalize_keeps_markdown_structure():
    text = "# หัวข้อ\n\n```\n  code   here\n```\n\n| a  | b |\n"
    assert normalize_text(text) == text


def test_normalize_removes_zero_width_and_unifies_newlines():
    assert normalize_text("ก​ข\r\nค‌ง") == "กข\nคง"


def test_query_and_document_spelling_variants_match_after_normalizing():
    document = bm25_tokens(normalize_text("เเม่บ้านทำความสะอาด"))
    query = bm25_tokens(normalize_text("แม่บ้าน"))
    assert set(query) <= set(document)


def test_split_runs_separates_scripts():
    assert split_runs("ใช้ Redis v2 ทุกวัน") == [
        ("thai", "ใช้"),
        ("latin", "Redis"),
        ("latin", "v2"),
        ("thai", "ทุกวัน"),
    ]


def test_word_tokens_segments_thai_and_keeps_latin_whole():
    tokens = word_tokens("ระบบใช้ Redis เป็นแคช")
    assert "Redis" in tokens
    assert "ระบบ" in tokens
    assert "ใช้" in tokens


def test_word_tokens_drops_punctuation_and_repeat_marks():
    assert word_tokens("ดี ๆ, (จริง)!") == ["ดี", "จริง"]


def test_bm25_tokens_lowercase_latin_only_text():
    assert bm25_tokens("PostgreSQL 16") == ["postgresql", "16"]


def test_split_sentences_splits_thai_on_model_boundaries():
    text = "ระบบใช้ Redis เป็นแคช เพราะเร็วมาก ทีมงานจึงทดสอบทุกสัปดาห์ นอกจากนี้ยังมีการสำรองข้อมูลทุกคืน"
    sentences = split_sentences(text)
    assert len(sentences) >= 2
    assert "".join(sentences) == text


def test_split_sentences_splits_english_and_keeps_text():
    text = "First sentence here. Second one follows! Third is last."
    assert split_sentences(text) == ["First sentence here. ", "Second one follows! ", "Third is last."]


def test_split_sentences_handles_mixed_lines():
    text = "Phase 1 เริ่มในเดือนมีนาคม The rollback stays open. It closes after 72 hours.\n"
    assert "".join(split_sentences(text)) == text


def test_estimate_stays_above_measured_gemma3_counts():
    # Real prompt_eval_count values from Ollama for these exact strings.
    thai = "ระบบสำรองข้อมูลทำงานทุกคืนเวลาสองนาฬิกาและเก็บสำเนาไว้สามสิบวัน " * 40
    english = "The backup system runs every night at two and keeps copies for thirty days. " * 40
    assert estimate_tokens(thai) >= 940
    assert estimate_tokens(english) >= 620


def test_estimate_charges_for_citation_markers():
    plain = "note " * 100
    cited = "note [12] " * 100
    assert estimate_tokens(cited) - estimate_tokens(plain) >= 300


def test_estimate_charges_for_every_number_in_a_citation_list():
    short = "note [1]"
    long = "note [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
    assert estimate_tokens(long) - estimate_tokens(short) >= 25


@pytest.mark.parametrize(
    "word",
    ["MinIO", "SEV3", "pg_size_pretty", "deploy.sh", "PL/pgSQL", "PostgreSQL", "UTC", "v2", "node.js"],
)
def test_names_are_identifiers(word):
    assert is_identifier(word)


@pytest.mark.parametrize(
    "word",
    ["policy", "Office", "Singapore", "on-call", "Customer-facing", "16", "24h", "1st", "5G", "e.g", "A"],
)
def test_ordinary_words_and_numbers_are_not_identifiers(word):
    assert not is_identifier(word)


def test_literal_terms_keep_joined_terms_whole_and_drop_trailing_punctuation():
    assert literal_terms("Run deploy.sh, then pg_size_pretty.") == ["Run", "deploy.sh", "then", "pg_size_pretty"]


def test_literal_terms_find_latin_inside_thai():
    assert literal_terms("MinIO ใช้ทำอะไร") == ["MinIO"]


def test_identifier_lookup_keeps_case_only_for_plain_all_caps():
    assert document_identifiers("SELECT from MinIO and SEV3") == {"SELECT", "minio", "sev3"}


def test_thai_word_frequency_separates_common_words_from_rare_ones():
    assert thai_word_frequency("ทำ") > thai_word_frequency("ทำงาน") > thai_word_frequency("เชียงใหม่")
    assert thai_word_frequency("คำที่ไม่มีอยู่จริง") == 0


def test_thai_stopwords_cover_grammar_words():
    assert is_thai_stopword("ของ")
    assert not is_thai_stopword("เชียงใหม่")


def test_question_language():
    assert question_language("ระบบสำรองข้อมูลทำงานอย่างไร") == "th"
    assert question_language("How do backups work?") == "other"
    assert question_language("Docker Compose ใช้ทำอะไร") == "th"
    assert question_language("What does สวัสดี mean in this document?") == "other"
