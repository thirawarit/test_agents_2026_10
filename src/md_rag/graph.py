"""The LangGraph flow: retrieve -> summarize -> agent <-> tools -> cite.

This is a fixed graph rather than a prebuilt ReAct agent, so retrieval always
runs first and the model can't skip it.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from md_rag.citations import CitedAnswer, resolve_citations
from md_rag.retrieve import Passage
from md_rag.summarize import Draft
from md_rag.textnorm import question_language
from md_rag.tools import TOOLS

MAX_TOOL_ROUNDS = 3

_AGENT_SYSTEM = """\
You finish an answer that is already drafted from documents. Call a tool only \
if the draft can't answer the question without it. Your final reply must be \
the complete answer. Keep every [n] citation from the draft exactly as written.
"""


class RagState(TypedDict, total=False):
    question: str
    passages: list[Passage]
    draft: Draft
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    tool_rounds: int
    cited: CitedAnswer
    warnings: list[str]


def _run_tool(tools: dict[str, BaseTool], call: dict[str, Any]) -> ToolMessage:
    tool = tools.get(call["name"])
    if tool is None:
        content = f"Unknown tool: {call['name']}"
    else:
        try:
            content = str(tool.invoke(call["args"]))
        except Exception as error:  # a failed tool is reported back to the model
            content = f"Tool failed: {error}"
    return ToolMessage(content=content, tool_call_id=call["id"])


def build_graph(
    retriever: Callable[[str], list[Passage]],
    summarizer: Callable[[str, list[Passage]], Draft],
    chat: Any,
    tools: dict[str, BaseTool] | None = None,
) -> Any:
    registry = TOOLS if tools is None else tools
    # bind_tools is only reached with real tools: gemma3 has no tool support,
    # and Ollama rejects any request that includes tool definitions for it.
    bound = chat.bind_tools(list(registry.values())) if registry else None

    def retrieve_node(state: RagState) -> dict[str, Any]:
        return {"passages": retriever(state["question"])}

    def summarize_node(state: RagState) -> dict[str, Any]:
        draft = summarizer(state["question"], state["passages"])
        return {"draft": draft, "warnings": list(draft.warnings)}

    def agent_node(state: RagState) -> dict[str, Any]:
        draft_text = state["draft"].text
        if bound is None:
            return {"answer": draft_text}
        history = state.get("messages", [])
        opening = [] if history else [
            SystemMessage(content=_AGENT_SYSTEM),
            HumanMessage(content=f"Question: {state['question']}\n\nDraft:\n{draft_text}"),
        ]
        reply = bound.invoke([*history, *opening])
        return {
            "messages": [*opening, reply],
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "answer": str(reply.content).strip() or draft_text,
        }

    def route_after_agent(state: RagState) -> str:
        if not registry:
            return "cite"
        last = state["messages"][-1]
        wants_tools = bool(getattr(last, "tool_calls", None))
        return "tools" if wants_tools and state["tool_rounds"] < MAX_TOOL_ROUNDS else "cite"

    def tools_node(state: RagState) -> dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        return {"messages": [_run_tool(registry, call) for call in calls]}

    def cite_node(state: RagState) -> dict[str, Any]:
        cited = resolve_citations(state["answer"], state["passages"])
        warnings = list(state.get("warnings", []))
        if cited.removed_markers:
            numbers = ", ".join(str(number) for number in cited.removed_markers)
            warnings.append(f"Removed citation markers with no matching passage: {numbers}")
        if cited.flagged:
            warnings.append("This answer has no valid citation, so it can't be traced to a document.")
        if question_language(state["question"]) != question_language(cited.text):
            warnings.append("The answer may not be in the language of the question.")
        return {"cited": cited, "warnings": warnings}

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("cite", cite_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "cite": "cite"})
    graph.add_edge("tools", "agent")
    graph.add_edge("cite", END)
    return graph.compile()
