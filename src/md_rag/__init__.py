"""Local-only RAG over mixed Thai/English markdown files."""

import os

# Must run before anything imports langchain or langsmith: tracing would send
# prompts, and therefore document text, to a hosted service.
for _name in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
):
    os.environ[_name] = "false"
del _name

__version__ = "0.1.0"
