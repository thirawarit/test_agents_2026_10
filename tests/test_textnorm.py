from md_rag.textnorm import (
    bm25_tokens,
    normalize_text,
    question_language,
    split_runs,
    split_sentences,
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


def test_question_language():
    assert question_language("ระบบสำรองข้อมูลทำงานอย่างไร") == "th"
    assert question_language("How do backups work?") == "other"
    assert question_language("Docker Compose ใช้ทำอะไร") == "th"
    assert question_language("What does สวัสดี mean in this document?") == "other"
