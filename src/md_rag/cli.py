"""Command line interface: index, ask, doctor."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from md_rag.citations import format_sources
from md_rag.config import Config, ConfigError, load_config
from md_rag.index import IndexError_, load_index, stale_files, update_index
from md_rag.llm import (
    OllamaError,
    context_length,
    ensure_local_model,
    make_chat,
    make_embedder,
    server_version,
)
from md_rag.tools import TOOLS

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_CITATION = 3
EXIT_NOT_COVERED = 4


def cmd_index(config: Config, args: argparse.Namespace) -> int:
    ensure_local_model(config, config.embed_model)
    report = update_index(config, make_embedder(config), full=args.full)
    if report.rebuilt_reason:
        print(f"Full rebuild: {report.rebuilt_reason}")
    print(
        f"Added {len(report.added)}, changed {len(report.changed)}, removed {len(report.removed)}, "
        f"unchanged {report.unchanged} files."
    )
    print(f"Embedded {report.embedded} chunks. Index holds {report.chunks} chunks.")
    return EXIT_OK


def cmd_ask(config: Config, args: argparse.Namespace) -> int:
    from md_rag.graph import build_graph
    from md_rag.retrieve import Retriever
    from md_rag.summarize import Summarizer

    ensure_local_model(config, config.embed_model)
    ensure_local_model(config, config.generation_model)
    index = load_index(config)
    chat = make_chat(config)
    graph = build_graph(Retriever(config, index, make_embedder(config)), Summarizer(chat, config), chat)
    state = graph.invoke({"question": args.question})

    cited = state["cited"]
    print(cited.text)
    if cited.sources:
        print("\nSources:")
        print(format_sources(cited.sources))
    for warning in state.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    if state["draft"].mode == "empty":
        print(
            f"warning: topic not covered: best match scored {state['best_score']:.3f}, "
            f"retrieval.min_relevance is {config.min_relevance:.2f}",
            file=sys.stderr,
        )
        return EXIT_NOT_COVERED
    return EXIT_NO_CITATION if cited.flagged else EXIT_OK


def cmd_doctor(config: Config, args: argparse.Namespace) -> int:
    from langsmith.utils import tracing_is_enabled

    from md_rag.loader import find_markdown
    from md_rag.textnorm import bm25_tokens, split_sentences

    failures = 0

    def report(status: str, name: str, detail: str) -> None:
        nonlocal failures
        failures += status == "FAIL"
        print(f"{status:<5} {name}: {detail}")

    def check(name: str, probe: Callable[[], str]) -> None:
        try:
            report("OK", name, probe())
        except (OllamaError, ConfigError, IndexError_) as error:
            report("FAIL", name, str(error))

    report("OK", "config", str(config.source_path))
    if config.docs_dir.is_dir():
        count = len(find_markdown(config.docs_dir, config.exclude))
        report("OK" if count else "WARN", "documents", f"{count} markdown files in {config.docs_dir}")
    else:
        report("FAIL", "documents", f"directory not found: {config.docs_dir}")

    report("OK", "local-only host", config.ollama_host)
    report("FAIL" if tracing_is_enabled() else "OK", "tracing", "LangSmith tracing is off")

    tokens = bm25_tokens("ระบบสำรองข้อมูล backup")
    sentences = split_sentences("ระบบทำงานทุกคืน ทีมงานตรวจสอบทุกเช้า")
    report("OK" if len(tokens) > 2 and sentences else "FAIL", "thai tokenizer", " | ".join(tokens))

    check("ollama", lambda: f"version {server_version(config)}")
    check("embedding model", lambda: _describe_embedder(config))
    try:
        details = ensure_local_model(config, config.generation_model)
    except (OllamaError, ConfigError) as error:
        report("FAIL", "generation model", str(error))
    else:
        limit = context_length(details)
        report("OK", "generation model", f"{config.generation_model}, local")
        if limit is not None and config.num_ctx > limit:
            report("WARN", "context", f"num_ctx {config.num_ctx} exceeds the model limit {limit}")
        else:
            report("OK", "context", f"num_ctx {config.num_ctx}, model limit {limit}")
        supports_tools = "tools" in details.get("capabilities", [])
        detail = f"{config.generation_model} {'supports' if supports_tools else 'does not support'} tools"
        if TOOLS and not supports_tools:
            report("FAIL", "tool support", f"{detail}, and {len(TOOLS)} tools are registered")
        else:
            report("OK", "tool support", f"{detail}; {len(TOOLS)} tools registered")

    try:
        index = load_index(config)
        stale = stale_files(config, index) if config.docs_dir.is_dir() else []
    except IndexError_ as error:
        report("WARN", "index", str(error))
    else:
        detail = f"{len(index.chunks)} chunks from {len(index.files)} files"
        if stale:
            report("WARN", "index", f"{detail}; out of date for {len(stale)} files. Run: md-rag index")
        else:
            report("OK", "index", detail)

    return EXIT_ERROR if failures else EXIT_OK


def _describe_embedder(config: Config) -> str:
    ensure_local_model(config, config.embed_model)
    return f"{config.embed_model}, local"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="md-rag", description=__doc__)
    parser.add_argument("--config", help="path to the TOML config file")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="index new and changed markdown files")
    index.add_argument("--full", action="store_true", help="rebuild everything")
    index.set_defaults(handler=cmd_index)

    ask = commands.add_parser("ask", help="summarize a topic across the documents")
    ask.add_argument("question")
    ask.set_defaults(handler=cmd_ask)

    doctor = commands.add_parser("doctor", help="check configuration, Ollama and the index")
    doctor.set_defaults(handler=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(load_config(args.config), args)
    except (ConfigError, OllamaError, IndexError_) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
