"""End-to-end tests against the real local Ollama. Skipped when it isn't available.

Model output varies, so these check structure and facts that retrieval makes
inevitable. They don't judge the quality of the prose.
"""

import shutil
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from md_rag.config import Config
from md_rag.graph import build_graph
from md_rag.index import load_index, update_index
from md_rag.llm import OllamaError, ensure_local_model, make_chat, make_embedder
from md_rag.retrieve import Retriever
from md_rag.summarize import Draft, Summarizer
from md_rag.textnorm import question_language

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> Config:
    base = tmp_path_factory.mktemp("rag")
    docs = base / "docs"
    shutil.copytree(FIXTURES, docs)
    settings = Config(docs_dir=docs, index_dir=base / "index")
    try:
        ensure_local_model(settings, settings.embed_model)
        ensure_local_model(settings, settings.generation_model)
    except OllamaError as error:
        pytest.skip(str(error))
    return settings


@pytest.fixture(scope="module")
def graph(config):
    embedder = make_embedder(config)
    update_index(config, embedder)
    chat = make_chat(config)
    return build_graph(Retriever(config, load_index(config), embedder), Summarizer(chat, config), chat)


def test_generation_model_has_no_tool_capability_recorded(config):
    """Documents the fact the empty-registry design relies on for gemma3."""
    details = ensure_local_model(config, config.generation_model)
    assert "completion" in details["capabilities"]
    if config.generation_model == "gemma3:latest":
        assert "tools" not in details["capabilities"]


def test_bge_m3_retrieves_across_languages(config, graph):
    embedder = make_embedder(config)
    retriever = Retriever(config, load_index(config), embedder)
    english = retriever("backup retention policy").passages
    thai = retriever("นโยบายการสำรองข้อมูล").passages
    assert {"english.md", "thai.md", "mixed.md"} <= {item.file for item in english}
    assert {"english.md", "thai.md", "mixed.md"} <= {item.file for item in thai}
    assert any("สำรองข้อมูล" in title for title in thai[0].heading_path)


def test_english_question_gets_an_english_cited_answer(graph):
    state = graph.invoke({"question": "Summarize how backups are handled across the documents."})
    cited = state["cited"]
    assert not cited.flagged
    assert question_language(cited.text) == "other"
    files = {source.label.split(" > ")[0] for source in cited.sources}
    assert len(files) >= 2


def test_thai_question_gets_a_thai_cited_answer(graph):
    state = graph.invoke({"question": "สรุปแนวทางการสำรองข้อมูลจากเอกสารทั้งหมดให้หน่อย"})
    cited = state["cited"]
    assert not cited.flagged
    assert question_language(cited.text) == "th"
    files = {source.label.split(" > ")[0] for source in cited.sources}
    assert len(files) >= 2


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("How often are restore drills run?", id="en-to-en"),
        pytest.param("สรุปแนวทางการสำรองข้อมูลจากเอกสารทั้งหมดให้หน่อย", id="th-to-th"),
        pytest.param("ใครเป็นคนรดน้ำต้นไม้ในสำนักงาน", id="th-to-english-text-plants"),
        pytest.param("ระดับความรุนแรงของเหตุการณ์ SEV1 คืออะไร", id="th-to-english-text-severity"),
        pytest.param("ทีมซ้อมกู้คืนฐานข้อมูลปีละกี่ครั้ง", id="th-to-english-text-drills"),
        pytest.param("ปล่อยเวอร์ชันใหม่วันไหนบ้าง", id="th-to-english-text-releases"),
        pytest.param("How many vacation days do employees get?", id="en-to-thai-text"),
        pytest.param("Project Phoenix ใช้อะไรสำรองข้อมูล", id="mixed"),
    ],
)
def test_covered_question_passes_the_cutoff(config, graph, question):
    result = Retriever(config, load_index(config), make_embedder(config))(question)
    assert result.passages, f"refused at {result.best_score:.3f}"


@pytest.mark.xfail(reason="Measured wrong refusals: loose Thai wording against English text", strict=False)
@pytest.mark.parametrize(
    "question",
    ["สรุปนโยบายการปล่อยซอฟต์แวร์ขึ้นระบบจริง", "สรุปวิธีการปล่อยระบบแบบค่อยเป็นค่อยไป"],
)
def test_known_wrong_refusals(config, graph, question):
    assert Retriever(config, load_index(config), make_embedder(config))(question).passages


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("What is the company policy on parental leave?", id="parental-en"),
        pytest.param("นโยบายการลาคลอดของบริษัทเป็นอย่างไร", id="maternity-th"),
        pytest.param("What is the capital of Australia?", id="unrelated-en"),
        pytest.param("กติกาฟุตบอลมีอะไรบ้าง", id="unrelated-th"),
    ],
)
def test_uncovered_question_is_refused_without_citations(config, graph, question):
    state = graph.invoke({"question": question})
    assert state["passages"] == []
    assert state["draft"].calls == 0
    assert state["cited"].sources == []
    assert state["draft"].mode == "empty"


def test_map_reduce_works_on_the_real_model(config):
    # Only a window this small forces the fixtures through map-reduce. At 1024
    # one fixed prompt reports exactly the overflow count by chance.
    small = Config(
        docs_dir=config.docs_dir,
        index_dir=config.index_dir,
        num_ctx=1100,
        num_predict=128,
    )
    embedder = make_embedder(small)
    retriever = Retriever(small, load_index(small), embedder)
    chat = make_chat(small)
    state = build_graph(retriever, Summarizer(chat, small), chat).invoke(
        {"question": "Summarize how backups are handled."}
    )
    assert state["draft"].mode == "map-reduce"
    assert state["draft"].calls >= 3
    assert not any("truncated" in warning for warning in state.get("warnings", []))
    assert not state["cited"].flagged


def test_real_overflow_is_reported(config):
    # Calls past the planner, which would never send a prompt this large.
    small = Config(docs_dir=config.docs_dir, index_dir=config.index_dir, num_ctx=2048, num_predict=256)
    draft = Draft("", "single")
    system = "Reply with OK."
    user = "The backup system runs every night at two. " * 300
    Summarizer(make_chat(small), small)._call(system, user, draft)
    assert any("truncated" in warning for warning in draft.warnings)


def test_chat_returns_the_count_truncation_detection_needs(config):
    chat = make_chat(config)
    reply = chat.invoke([SystemMessage(content="Reply OK."), HumanMessage(content="Hello")])
    assert isinstance(reply.response_metadata.get("prompt_eval_count"), int)
