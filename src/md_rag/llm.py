"""Ollama access. Every model client in the project comes from this module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from md_rag.config import Config, ConfigError, assert_local_model_tag, assert_loopback_host


class OllamaError(Exception):
    """Ollama is unreachable, or a model is missing or not local."""


# trust_env=False stops httpx from sending loopback traffic through HTTP_PROXY.
# follow_redirects=False stops a redirect from carrying a prompt to another host.
def _client_kwargs() -> dict[str, Any]:
    return {"trust_env": False, "follow_redirects": False}


def make_embedder(config: Config) -> OllamaEmbeddings:
    assert_loopback_host(config.ollama_host)
    assert_local_model_tag(config.embed_model)
    return OllamaEmbeddings(
        model=config.embed_model,
        base_url=config.ollama_host,
        client_kwargs=_client_kwargs(),
    )


def make_chat(config: Config) -> ChatOllama:
    assert_loopback_host(config.ollama_host)
    assert_local_model_tag(config.generation_model)
    return ChatOllama(
        model=config.generation_model,
        base_url=config.ollama_host,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        client_kwargs=_client_kwargs(),
    )


def _request(config: Config, method: str, path: str, **kwargs: Any) -> httpx.Response:
    assert_loopback_host(config.ollama_host)
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=30) as client:
            return client.request(method, config.ollama_host.rstrip("/") + path, **kwargs)
    except httpx.HTTPError as error:
        raise OllamaError(f"Cannot reach Ollama at {config.ollama_host}: {error}") from error


def server_version(config: Config) -> str:
    response = _request(config, "GET", "/api/version")
    if response.status_code != 200:
        raise OllamaError(f"Ollama /api/version returned HTTP {response.status_code}")
    return str(response.json().get("version", "unknown"))


def show_model(config: Config, model: str) -> dict[str, Any]:
    response = _request(config, "POST", "/api/show", json={"model": model})
    if response.status_code == 404:
        raise OllamaError(f"Model {model!r} is not installed. Run: ollama pull {model}")
    if response.status_code != 200:
        raise OllamaError(f"Ollama /api/show returned HTTP {response.status_code} for {model!r}")
    return response.json()


def ensure_local_model(config: Config, model: str) -> dict[str, Any]:
    """Confirm a model runs here, including local aliases of cloud models.

    The tag check can't see an alias whose name hides a cloud model, so this
    asks Ollama where the model really runs before any text is sent to it.
    """
    assert_local_model_tag(model)
    details = show_model(config, model)
    if details.get("remote_host") or details.get("remote_model"):
        raise ConfigError(f"Refusing cloud model {model!r}: document text must stay local")
    return details


def context_length(details: dict[str, Any]) -> int | None:
    for key, value in details.get("model_info", {}).items():
        if key.endswith(".context_length"):
            return int(value)
    return None


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_eval_count: int | None
    done_reason: str | None


def complete(chat: Any, messages: list[BaseMessage]) -> Completion:
    reply = chat.invoke(messages)
    metadata = getattr(reply, "response_metadata", {}) or {}
    return Completion(
        text=str(reply.content).strip(),
        prompt_eval_count=metadata.get("prompt_eval_count"),
        done_reason=metadata.get("done_reason"),
    )
