from pathlib import Path

from md_rag.chunker import Section, chunk_document, chunk_section, split_sections
from md_rag.loader import Document, load_documents
from md_rag.textnorm import word_count

FIXTURES = Path(__file__).parent / "fixtures"


def doc(text: str) -> Document:
    return Document("note.md", text, "hash")


def section(text: str) -> Section:
    return Section("note.md#0", "note.md", ("Title",), text)


def fixture(name: str) -> Document:
    return next(item for item in load_documents(FIXTURES, ()) if item.path == name)


def test_loader_excludes_requirement_file_and_non_markdown(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep\n\ntext", encoding="utf-8")
    (tmp_path / "REQ.local.md").write_text("# Requirement\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text("not markdown", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.md").write_text("deep", encoding="utf-8")
    paths = [item.path for item in load_documents(tmp_path, ("REQ.local.md",))]
    assert paths == ["keep.md", "sub/deep.md"]


def test_loader_ignores_symlink_outside_docs_dir(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("secret", encoding="utf-8")
    (docs / "link.md").symlink_to(outside)
    assert load_documents(docs, ()) == []


def test_loader_skips_invalid_utf8(tmp_path, capsys):
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\xff")
    assert load_documents(tmp_path, ()) == []
    assert "bad.md" in capsys.readouterr().err


def test_heading_path_tracks_nesting():
    text = "# A\n\none\n\n## B\n\ntwo\n\n### C\n\nthree\n\n## D\n\nfour\n"
    paths = [item.heading_path for item in split_sections(doc(text))]
    assert paths == [("A",), ("A", "B"), ("A", "B", "C"), ("A", "D")]


def test_text_before_first_heading_has_empty_path():
    sections = split_sections(doc("intro\n\n# A\n\nbody\n"))
    assert sections[0].heading_path == ()
    assert sections[0].text == "intro"


def test_heading_without_body_gets_no_section():
    sections = split_sections(doc("# A\n\n## B\n\nbody\n"))
    assert [item.heading_path for item in sections] == [("A", "B")]


def test_hash_inside_code_fence_is_not_a_heading():
    sections = split_sections(fixture("english.md"))
    deployment = next(item for item in sections if item.heading_path[-1] == "Deployment")
    assert "# Not a heading" in deployment.text
    assert not any("Not a heading" in title for item in sections for title in item.heading_path)


def test_setext_headings_are_recognized():
    sections = split_sections(doc("Top\n===\n\nbody\n\nSub\n---\n\nmore\n"))
    assert [item.heading_path for item in sections] == [("Top",), ("Top", "Sub")]


def test_table_and_front_matter_rules_are_not_setext_headings():
    sections = split_sections(fixture("mixed.md"))
    assert sections[0].heading_path == ()
    assert "title: Project Phoenix" in sections[0].text
    assert [item.heading_path[0] for item in sections[1:]] == ["โครงการ Phoenix"] * 4


def test_section_that_fits_stays_whole():
    text = "short section text"
    chunks = chunk_section(section(text), 350, 50)
    assert [item.text for item in chunks] == [text]


def test_embed_text_is_prefixed_with_heading_path():
    item = Section("a.md#0", "a.md", ("Guide", "Backups"), "body")
    assert chunk_section(item, 350, 50)[0].embed_text == "Guide > Backups\n\nbody"


def test_long_section_splits_near_target_with_overlap():
    text = " ".join(f"Sentence number {index} says something." for index in range(120))
    chunks = chunk_section(section(text), 60, 12)
    assert len(chunks) > 1
    assert all(word_count(item.text) <= 60 for item in chunks)
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.text.split(". ")[-1].rstrip(".") in later.text


def test_long_thai_section_splits_within_limit():
    text = " ".join(["ระบบสำรองข้อมูลทำงานทุกคืนและทีมงานตรวจสอบผลทุกเช้า"] * 80)
    chunks = chunk_section(section(text), 60, 10)
    assert len(chunks) > 1
    assert all(word_count(item.text) <= 60 for item in chunks)


def test_single_sentence_longer_than_limit_is_cut():
    text = " ".join(["word"] * 500)
    chunks = chunk_section(section(text), 100, 0)
    assert len(chunks) == 5
    assert all(word_count(item.text) <= 100 for item in chunks)


def test_code_fence_stays_whole_when_oversized():
    fence = "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(200)) + "\n```"
    text = f"Intro sentence.\n\n{fence}\n\nOutro sentence."
    chunks = chunk_section(section(text), 50, 10)
    assert sum(fence in item.text for item in chunks) == 1
    assert not any("value_1 = 1" in item.text and fence not in item.text for item in chunks)


def test_table_stays_whole_when_oversized():
    rows = "\n".join(f"| name{i} | value{i} |" for i in range(100))
    table = f"| Name | Value |\n| --- | --- |\n{rows}"
    text = f"Before the table.\n\n{table}\n\nAfter the table."
    chunks = chunk_section(section(text), 50, 10)
    assert sum(table in item.text for item in chunks) == 1


def test_all_fixture_text_reaches_a_chunk():
    for name in ("thai.md", "english.md", "mixed.md"):
        document = fixture(name)
        sections, chunks = chunk_document(document, 350, 50)
        assert sections and chunks
        assert {item.section_id for item in chunks} == {item.section_id for item in sections}


def test_thai_fixture_sections():
    paths = [item.heading_path[-1] for item in split_sections(fixture("thai.md"))]
    assert paths == [
        "คู่มือการดูแลระบบ",
        "การสำรองข้อมูล",
        "การกู้คืนข้อมูล",
        "การเฝ้าระวังระบบ",
        "การลาพักร้อน",
    ]
