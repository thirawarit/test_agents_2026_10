"""Configuration loading and the local-only checks that guard it."""

from __future__ import annotations

import ipaddress
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_CONFIG_NAME = "config.local.toml"
CONFIG_ENV_VAR = "MD_RAG_CONFIG"

_CLOUD_MODEL = re.compile(r"(^|[:\-])cloud$", re.IGNORECASE)
_CONTEXT_SAFETY = 0.8
_MIN_CONTEXT_BUDGET = 256


class ConfigError(Exception):
    """The configuration is missing, invalid, or not local-only."""


def assert_loopback_host(host: str) -> None:
    """Refuse any Ollama host that isn't this machine."""
    parts = urlsplit(host)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"Ollama host must be a full http(s) URL, got {host!r}")
    if parts.hostname == "localhost":
        return
    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise ConfigError(f"Refusing non-loopback Ollama host {host!r}: document text must stay local")


def assert_local_model_tag(model: str) -> None:
    """Refuse model tags that Ollama would run in its cloud."""
    if not model.strip():
        raise ConfigError("Model name is empty")
    if _CLOUD_MODEL.search(model.strip()):
        raise ConfigError(f"Refusing cloud model {model!r}: document text must stay local")


@dataclass(frozen=True)
class Config:
    docs_dir: Path
    index_dir: Path
    exclude: tuple[str, ...] = ("REQ.local.md",)

    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "bge-m3"
    generation_model: str = "gemma3:latest"
    temperature: float = 0.2
    num_ctx: int = 16384
    num_predict: int = 1536

    chunk_words: int = 350
    overlap_words: int = 50

    dense_top_k: int = 40
    bm25_top_k: int = 40
    rrf_k: int = 60
    mmr_keep: int = 24
    mmr_lambda: float = 0.7
    section_token_limit: int = 2000
    min_relevance: float = 0.5

    source_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        assert_loopback_host(self.ollama_host)
        assert_local_model_tag(self.embed_model)
        assert_local_model_tag(self.generation_model)
        if not 0 <= self.overlap_words < self.chunk_words:
            raise ConfigError("chunking: overlap_words must be at least 0 and below chunk_words")
        if not 0 <= self.mmr_lambda <= 1:
            raise ConfigError("retrieval: mmr_lambda must be between 0 and 1")
        if not 0 <= self.min_relevance <= 1:
            raise ConfigError("retrieval: min_relevance must be between 0 and 1")
        if self.context_budget < _MIN_CONTEXT_BUDGET:
            raise ConfigError(
                f"ollama: num_ctx {self.num_ctx} leaves only {self.context_budget} tokens for "
                f"prompts after num_predict {self.num_predict}; raise num_ctx or lower num_predict"
            )
        for name in ("dense_top_k", "bm25_top_k", "rrf_k", "mmr_keep", "section_token_limit"):
            if getattr(self, name) < 1:
                raise ConfigError(f"retrieval: {name} must be at least 1")

    @property
    def context_budget(self) -> int:
        """Estimated prompt tokens a single call may use.

        When a prompt overflows num_ctx, Ollama drops part of it without an
        error, and the system instructions can be lost with it. The margin
        absorbs error in the token estimate so that overflow stays rare.
        """
        return int(self.num_ctx * _CONTEXT_SAFETY) - self.num_predict


# TOML section -> keys it may contain, all named after Config fields.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "paths": ("docs_dir", "index_dir", "exclude"),
    "ollama": ("host", "embed_model", "generation_model", "temperature", "num_ctx", "num_predict"),
    "chunking": ("chunk_words", "overlap_words"),
    "retrieval": (
        "dense_top_k",
        "bm25_top_k",
        "rrf_k",
        "mmr_keep",
        "mmr_lambda",
        "section_token_limit",
        "min_relevance",
    ),
}


def resolve_config_path(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit)
    from_env = os.environ.get(CONFIG_ENV_VAR)
    return Path(from_env) if from_env else Path.cwd() / DEFAULT_CONFIG_NAME


def load_config(path: str | Path | None = None) -> Config:
    config_path = resolve_config_path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}. "
            f"Copy config.example.toml to {DEFAULT_CONFIG_NAME} and set paths.docs_dir."
        )
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML in {config_path}: {error}") from error

    values: dict[str, object] = {}
    for section, content in raw.items():
        allowed = _SECTIONS.get(section)
        if allowed is None or not isinstance(content, dict):
            raise ConfigError(f"Unknown config section [{section}] in {config_path}")
        for key, value in content.items():
            if key not in allowed:
                raise ConfigError(f"Unknown config key {section}.{key} in {config_path}")
            values["ollama_host" if (section, key) == ("ollama", "host") else key] = value

    if not values.get("docs_dir"):
        raise ConfigError(f"paths.docs_dir is required in {config_path}")

    base = config_path.parent
    values["docs_dir"] = _resolve(base, str(values["docs_dir"]))
    values["index_dir"] = _resolve(base, str(values.get("index_dir", ".rag_index")))
    if "exclude" in values:
        values["exclude"] = tuple(str(name) for name in values["exclude"])  # type: ignore[union-attr]
    try:
        return Config(source_path=config_path, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"Bad value in {config_path}: {error}") from error


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
