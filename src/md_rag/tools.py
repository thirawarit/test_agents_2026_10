"""The tool registry, empty on purpose.

While it is empty the graph makes no tool-enabled model call, so normal runs
don't depend on the generation model supporting tools. gemma3 doesn't.

Before adding a tool, read the README's note on tools: a tool's arguments come
from the model, so a tool that reaches the network can send document text off
this machine, which the checks on the Ollama host can't prevent.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

TOOLS: dict[str, BaseTool] = {}
