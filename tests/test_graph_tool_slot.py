"""The tools edge is unreachable with an empty registry, and works with a fake tool."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from md_rag.graph import MAX_TOOL_ROUNDS, build_graph
from md_rag.retrieve import Passage
from md_rag.summarize import Draft
from md_rag.tools import TOOLS

PASSAGES = [Passage(1, "a.md", ("Heading",), "body")]


def retriever(question):
    return PASSAGES


def summarizer(question, passages):
    return Draft("Drafted fact [1].", "single", calls=1)


class NoToolsChat:
    """Behaves like gemma3: any attempt to use tools is an error."""

    def bind_tools(self, tools):
        raise AssertionError("bind_tools must not be called with an empty registry")

    def invoke(self, messages):
        raise AssertionError("the agent node must not call the model with an empty registry")


class ToolChat:
    """Replays scripted replies from the bound model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name="lookup", args=None, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {"term": "x"}, "id": call_id}])


def visited(graph, question="topic"):
    nodes, final = [], {}
    for update in graph.stream({"question": question}, stream_mode="updates"):
        for node, values in update.items():
            nodes.append(node)
            final.update(values or {})
    return nodes, final


@tool
def lookup(term: str) -> str:
    """Look up a term."""
    return f"definition of {term}"


@tool
def broken(term: str) -> str:
    """Always fails."""
    raise ValueError("backend down")


def test_shipped_registry_is_empty():
    assert TOOLS == {}


def test_empty_registry_never_binds_tools_or_reaches_the_tools_node():
    nodes, final = visited(build_graph(retriever, summarizer, NoToolsChat()))
    assert nodes == ["retrieve", "summarize", "agent", "cite"]
    assert final["cited"].text == "Drafted fact [1]."
    assert final["cited"].sources[0].label == "a.md > Heading"


def test_empty_registry_passed_explicitly_behaves_the_same():
    nodes, _ = visited(build_graph(retriever, summarizer, NoToolsChat(), tools={}))
    assert "tools" not in nodes


def test_retrieval_always_runs_before_any_model_use():
    calls = []

    def tracking_retriever(question):
        calls.append("retrieve")
        return PASSAGES

    def tracking_summarizer(question, passages):
        calls.append("summarize")
        assert passages == PASSAGES
        return Draft("Fact [1].", "single")

    visited(build_graph(tracking_retriever, tracking_summarizer, NoToolsChat()))
    assert calls == ["retrieve", "summarize"]


def test_fake_tool_runs_and_feeds_its_result_back_to_the_agent():
    chat = ToolChat([call(), AIMessage(content="Final fact [1].")])
    nodes, final = visited(build_graph(retriever, summarizer, chat, tools={"lookup": lookup}))
    assert nodes == ["retrieve", "summarize", "agent", "tools", "agent", "cite"]
    assert chat.bound_tools == [lookup]
    assert final["cited"].text == "Final fact [1]."
    result = chat.seen[1][-1]
    assert isinstance(result, ToolMessage)
    assert result.content == "definition of x"
    assert result.tool_call_id == "call-1"


def test_tool_loop_stops_at_the_round_limit_and_keeps_the_draft():
    chat = ToolChat([call(call_id=f"call-{n}") for n in range(MAX_TOOL_ROUNDS + 2)])
    nodes, final = visited(build_graph(retriever, summarizer, chat, tools={"lookup": lookup}))
    assert nodes.count("agent") == MAX_TOOL_ROUNDS
    assert nodes.count("tools") == MAX_TOOL_ROUNDS - 1
    assert nodes[-1] == "cite"
    assert final["cited"].text == "Drafted fact [1]."


def test_failing_tool_is_reported_to_the_model():
    chat = ToolChat([call("broken"), AIMessage(content="Fact [1].")])
    visited(build_graph(retriever, summarizer, chat, tools={"broken": broken}))
    assert "backend down" in chat.seen[1][-1].content


def test_unknown_tool_is_reported_to_the_model():
    chat = ToolChat([call("invented"), AIMessage(content="Fact [1].")])
    visited(build_graph(retriever, summarizer, chat, tools={"lookup": lookup}))
    assert "Unknown tool" in chat.seen[1][-1].content


def test_answer_with_no_citation_is_flagged_with_a_warning():
    def uncited(question, passages):
        return Draft("Nothing cited here.", "single")

    _, final = visited(build_graph(retriever, uncited, NoToolsChat()))
    assert final["cited"].flagged
    assert any("no valid citation" in warning for warning in final["warnings"])


def test_answer_in_the_wrong_language_gets_a_warning():
    _, final = visited(build_graph(retriever, summarizer, NoToolsChat()), question="สรุปเรื่องการสำรองข้อมูล")
    assert any("language" in warning for warning in final["warnings"])


def test_summarizer_warnings_reach_the_final_state():
    def warned(question, passages):
        return Draft("Fact [1].", "single", warnings=["Prompt was probably truncated"])

    _, final = visited(build_graph(retriever, warned, NoToolsChat()))
    assert "Prompt was probably truncated" in final["warnings"]
