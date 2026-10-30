import re
from pathlib import Path

from langchain_core.messages import AIMessage

from md_rag import cli
from md_rag.citations import Source, format_sources, resolve_citations
from md_rag.config import Config
from md_rag.retrieve import Passage, Retrieval
from md_rag.summarize import Summarizer


def passage(number: int, file: str = "a.md", *headings: str, text: str = "body") -> Passage:
    return Passage(number, file, headings or ("Heading",), text)


class ScriptedChat:
    """Answers by rule and records every prompt it receives."""

    def __init__(self, respond=None, metadata=None):
        self.respond = respond or (lambda messages: "fact [1]")
        self.metadata = metadata or {"done_reason": "stop"}
        self.prompts: list[list] = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self.respond(messages), response_metadata=self.metadata)


def echo_markers(messages) -> str:
    """Reply with every [n] the prompt contains, as a faithful model would."""
    return " ".join(f"note {marker}" for marker in sorted(set(re.findall(r"\[\d+\]", messages[1].content))))


def config(**overrides) -> Config:
    return Config(docs_dir=Path("."), index_dir=Path("."), **overrides)


def test_valid_markers_resolve_to_file_and_heading():
    passages = [passage(1, "guide.md", "Ops", "Backups")]
    cited = resolve_citations("Backups run nightly [1].", passages)
    assert cited.text == "Backups run nightly [1]."
    assert cited.sources == [Source(1, "guide.md > Ops > Backups")]
    assert not cited.flagged


def test_markers_are_renumbered_by_first_use():
    passages = [passage(1, "a.md"), passage(2, "b.md"), passage(3, "c.md")]
    cited = resolve_citations("First [3]. Then [1]. Again [3].", passages)
    assert cited.text == "First [1]. Then [2]. Again [1]."
    assert [item.label for item in cited.sources] == ["c.md > Heading", "a.md > Heading"]


def test_unknown_markers_are_removed():
    cited = resolve_citations("Real [1]. Invented [9].", [passage(1)])
    assert cited.text == "Real [1]. Invented."
    assert cited.removed_markers == [9]
    assert not cited.flagged


def test_only_unknown_markers_flags_the_answer():
    cited = resolve_citations("Invented claim [7].", [passage(1)])
    assert cited.flagged
    assert cited.sources == []
    assert cited.text == "Invented claim."


def test_answer_without_markers_is_flagged():
    assert resolve_citations("No citation at all.", [passage(1)]).flagged


def test_multiple_numbers_in_one_marker():
    passages = [passage(1, "a.md"), passage(2, "b.md")]
    assert resolve_citations("Both [1, 2].", passages).text == "Both [1][2]."
    assert resolve_citations("Both [1,2].", passages).text == "Both [1][2]."


def test_list_with_one_unknown_number_keeps_the_valid_one():
    cited = resolve_citations("Claim [1, 9].", [passage(1)])
    assert cited.text == "Claim [1]."
    assert cited.removed_markers == [9]


def test_same_file_and_heading_become_one_source():
    passages = [passage(1, "a.md", "H"), passage(2, "a.md", "H")]
    cited = resolve_citations("One [1]. Two [2].", passages)
    assert cited.text == "One [1]. Two [1]."
    assert len(cited.sources) == 1


def test_thai_digits_resolve():
    assert resolve_citations("ข้อมูล[๑]", [passage(1)]).text == "ข้อมูล[1]"


def test_sources_are_built_from_passages_not_model_text():
    cited = resolve_citations("Claim [1].\n\nSources:\n[1] invented.md > Fake", [passage(1, "real.md")])
    assert format_sources(cited.sources) == "[1] real.md > Heading"


def test_passage_without_headings_cites_the_file():
    item = Passage(1, "a.md", (), "body")
    assert resolve_citations("Claim [1].", [item]).sources == [Source(1, "a.md")]


def test_small_context_goes_in_one_call():
    chat = ScriptedChat()
    draft = Summarizer(chat, config())("topic", [passage(1), passage(2)])
    assert (draft.mode, draft.calls) == ("single", 1)
    assert "[1] a.md > Heading" in chat.prompts[0][1].content


def test_large_context_uses_map_reduce_and_keeps_citation_ids():
    passages = [passage(number, f"f{number}.md", text="word " * 400) for number in range(1, 21)]
    chat = ScriptedChat(echo_markers)
    settings = config(num_ctx=2048, num_predict=256)
    draft = Summarizer(chat, settings)("topic", passages)
    assert draft.mode == "map-reduce"
    assert draft.calls > 2
    assert set(re.findall(r"\[(\d+)\]", draft.text)) == {str(number) for number in range(1, 21)}
    assert not draft.warnings


def test_no_call_exceeds_the_budget_in_map_reduce():
    from md_rag.textnorm import estimate_tokens

    passages = [passage(number, text="word " * 400) for number in range(1, 21)]
    chat = ScriptedChat(echo_markers)
    settings = config(num_ctx=2048, num_predict=256)
    Summarizer(chat, settings)("topic", passages)
    for system, user in chat.prompts:
        assert estimate_tokens(system.content) + estimate_tokens(user.content) <= settings.context_budget


def test_map_results_of_none_are_dropped():
    def respond(messages):
        return "NONE" if "extract notes" in messages[0].content else "merged [1]"

    passages = [passage(number, text="word " * 400) for number in range(1, 21)]
    draft = Summarizer(ScriptedChat(respond), config(num_ctx=2048, num_predict=256))("topic", passages)
    assert draft.mode == "empty"
    assert "don't contain" in draft.text


def test_no_passages_makes_no_model_call():
    chat = ScriptedChat()
    draft = Summarizer(chat, config())("สำรองข้อมูล", [])
    assert chat.prompts == []
    assert draft.text == "ไม่พบข้อมูลเกี่ยวกับหัวข้อนี้ในเอกสาร"


def test_passage_larger_than_the_budget_is_trimmed_with_a_warning():
    chat = ScriptedChat(echo_markers)
    settings = config(num_ctx=2048, num_predict=256)
    draft = Summarizer(chat, settings)("topic", [passage(1, text="word " * 3000)])
    assert any("Trimmed" in warning for warning in draft.warnings)
    assert "[1]" in draft.text


def test_thai_question_asks_for_thai_output():
    chat = ScriptedChat()
    Summarizer(chat, config())("ระบบสำรองข้อมูลทำงานอย่างไร", [passage(1)])
    assert "Write the answer in Thai" in chat.prompts[0][0].content


def test_english_question_asks_for_the_question_language():
    chat = ScriptedChat()
    Summarizer(chat, config())("How do backups work?", [passage(1)])
    assert "same language as the topic" in chat.prompts[0][0].content


def test_cut_off_output_is_reported():
    chat = ScriptedChat(metadata={"done_reason": "length"})
    draft = Summarizer(chat, config())("topic", [passage(1)])
    assert any("num_predict" in warning for warning in draft.warnings)


def run_ask(
    monkeypatch, tmp_path, answer: str, retrieval=None, question="How do backups work?", extra=""
) -> int:
    settings = tmp_path / "config.toml"
    settings.write_text(f'[paths]\ndocs_dir = "docs"\n{extra}', encoding="utf-8")
    result = retrieval or Retrieval([passage(1, "guide.md", "Ops")], 0.8)
    monkeypatch.setattr(cli, "ensure_local_model", lambda *args: {})
    monkeypatch.setattr(cli, "load_index", lambda *args: None)
    monkeypatch.setattr(cli, "make_embedder", lambda *args: None)
    monkeypatch.setattr(cli, "make_chat", lambda *args: ScriptedChat(lambda messages: answer))
    monkeypatch.setattr("md_rag.retrieve.Retriever", lambda *args: (lambda question: result))
    return cli.main(["--config", str(settings), "ask", question])


def test_cited_answer_prints_code_built_sources(monkeypatch, tmp_path, capsys):
    assert run_ask(monkeypatch, tmp_path, "Backups run nightly [1].") == cli.EXIT_OK
    assert capsys.readouterr().out == "Backups run nightly [1].\n\nSources:\n[1] guide.md > Ops\n"


def test_answer_with_no_citation_exits_with_its_own_code(monkeypatch, tmp_path, capsys):
    assert run_ask(monkeypatch, tmp_path, "An unsupported claim [9].") == cli.EXIT_NO_CITATION
    captured = capsys.readouterr()
    assert "Sources:" not in captured.out
    assert "no valid citation" in captured.err


def test_refused_question_says_so_and_lists_no_sources(monkeypatch, tmp_path, capsys):
    refused = Retrieval([], 0.48)
    assert run_ask(monkeypatch, tmp_path, "unused", refused) == cli.EXIT_NOT_COVERED
    captured = capsys.readouterr()
    assert captured.out == "The documents don't contain information on this topic.\n"
    assert "best match scored 0.480" in captured.err
    assert "min_relevance is 0.50" in captured.err
    assert "no valid citation" not in captured.err


def test_refused_thai_question_gets_a_thai_reply(monkeypatch, tmp_path, capsys):
    code = run_ask(monkeypatch, tmp_path, "unused", Retrieval([], 0.48), "นโยบายการลาคลอดเป็นอย่างไร")
    assert code == cli.EXIT_NOT_COVERED
    assert capsys.readouterr().out == "ไม่พบข้อมูลเกี่ยวกับหัวข้อนี้ในเอกสาร\n"


def test_model_finding_nothing_in_map_reduce_also_exits_not_covered(monkeypatch, tmp_path, capsys):
    passages = [passage(number, text="word " * 400) for number in range(1, 21)]
    extra = "[ollama]\nnum_ctx = 2048\nnum_predict = 256\n"
    code = run_ask(monkeypatch, tmp_path, "NONE", Retrieval(passages, 0.7), extra=extra)
    assert code == cli.EXIT_NOT_COVERED
    assert "Sources:" not in capsys.readouterr().out


def test_exit_codes_are_documented():
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert f"| {cli.EXIT_NOT_COVERED} | Topic not covered" in readme
    assert f"| {cli.EXIT_NO_CITATION} | Answer has no valid citation" in readme
