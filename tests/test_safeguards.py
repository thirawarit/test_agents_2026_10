"""Local-only guarantees, tested against a real loopback HTTP server where possible."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from md_rag import cli
from md_rag.config import Config, ConfigError, assert_local_model_tag, assert_loopback_host, load_config
from md_rag.llm import OllamaError, context_length, ensure_local_model, make_chat, make_embedder, show_model
from md_rag.summarize import looks_truncated

ROOT = Path(__file__).parent.parent


class StubOllama(BaseHTTPRequestHandler):
    """Answers like Ollama, or redirects when the server asks it to."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.requests.append(self.path)
        if self.server.redirect_to:
            self.send_response(307)
            self.send_header("Location", self.server.redirect_to + self.path)
            self.end_headers()
            return
        self.send_json(self.server.responses.get(self.path, {"embeddings": [[1.0, 0.0]]}))

    def send_json(self, body):
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    servers = []

    def start(redirect_to=None, responses=None):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StubOllama)
        server.requests = []
        server.redirect_to = redirect_to
        server.responses = responses or {}
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def url_of(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def config_for(host: str, **overrides) -> Config:
    return Config(docs_dir=Path("."), index_dir=Path("."), ollama_host=host, **overrides)


@pytest.mark.parametrize(
    "host",
    [
        "http://127.0.0.1:11434",
        "http://127.8.8.8:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
        "https://127.0.0.1",
    ],
)
def test_loopback_hosts_are_accepted(host):
    assert_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "http://192.168.1.10:11434",
        "http://10.0.0.5:11434",
        "http://0.0.0.0:11434",
        "http://ollama.com",
        "https://api.example.com:11434",
        "http://127.0.0.1.evil.example",
        "http://localhost.evil.example",
        "http://127.0.0.1@evil.example:11434",
        "http://user:pw@evil.example",
        "http://[2001:db8::1]:11434",
        "127.0.0.1:11434",
        "ftp://127.0.0.1",
        "",
    ],
)
def test_other_hosts_are_refused(host):
    with pytest.raises(ConfigError):
        assert_loopback_host(host)


@pytest.mark.parametrize("model", ["kimi-k2:1t-cloud", "gpt-oss:120b-cloud", "gemma3:cloud", "x:CLOUD", "foo-cloud"])
def test_cloud_tags_are_refused(model):
    with pytest.raises(ConfigError):
        assert_local_model_tag(model)


@pytest.mark.parametrize("model", ["gemma3:latest", "bge-m3", "cloudy:7b", "mycloud"])
def test_local_tags_are_accepted(model):
    assert_local_model_tag(model)


def test_config_refuses_a_remote_host_at_load(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[paths]\ndocs_dir = "docs"\n[ollama]\nhost = "http://10.0.0.5:11434"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="non-loopback"):
        load_config(path)


def test_config_refuses_a_cloud_generation_model_at_load(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[paths]\ndocs_dir = "docs"\n[ollama]\ngeneration_model = "x:7b-cloud"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="cloud"):
        load_config(path)


def test_config_requires_docs_dir(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[ollama]\nnum_ctx = 16384\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="docs_dir"):
        load_config(path)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[paths]\ndocs_dir = "docs"\n[ollama]\nnum_cxt = 1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="num_cxt"):
        load_config(path)


def test_config_resolves_paths_from_its_own_directory(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[paths]\ndocs_dir = "docs"\n', encoding="utf-8")
    config = load_config(path)
    assert config.docs_dir == (tmp_path / "docs").resolve()
    assert config.index_dir == (tmp_path / ".rag_index").resolve()


def test_generation_model_is_configurable(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[paths]\ndocs_dir = "docs"\n[ollama]\ngeneration_model = "llama3.2:latest"\n', encoding="utf-8")
    assert load_config(path).generation_model == "llama3.2:latest"
    assert make_chat(load_config(path)).model == "llama3.2:latest"


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_relevance_outside_the_cosine_range_is_refused(value):
    with pytest.raises(ConfigError, match="min_relevance"):
        config_for("http://127.0.0.1:11434", min_relevance=value)


@pytest.mark.parametrize(
    ("name", "value"),
    [("rare_term_max_fraction", -0.1), ("rare_term_max_fraction", 1.5), ("rare_term_max_thai_freq", -1)],
)
def test_rare_term_settings_outside_their_range_are_refused(name, value):
    with pytest.raises(ConfigError, match=name):
        config_for("http://127.0.0.1:11434", **{name: value})


def test_example_config_states_the_default_cutoff(tmp_path):
    example = (ROOT / "config.example.toml").read_text(encoding="utf-8")
    copy = tmp_path / "config.toml"
    copy.write_text(example, encoding="utf-8")
    assert load_config(copy).min_relevance == config_for("http://127.0.0.1:11434").min_relevance == 0.5


def test_example_config_states_the_rare_term_defaults(tmp_path):
    copy = tmp_path / "config.toml"
    copy.write_text((ROOT / "config.example.toml").read_text(encoding="utf-8"), encoding="utf-8")
    loaded, defaults = load_config(copy), config_for("http://127.0.0.1:11434")
    assert loaded.rare_term_max_fraction == defaults.rare_term_max_fraction == 0.05
    assert loaded.rare_term_max_thai_freq == defaults.rare_term_max_thai_freq == 150


def test_context_budget_refuses_a_window_too_small_for_the_answer():
    with pytest.raises(ConfigError, match="num_ctx"):
        config_for("http://127.0.0.1:11434", num_ctx=1024, num_predict=1024)


def test_missing_config_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["--config", str(tmp_path / "absent.toml"), "doctor"]) == 1
    assert "Config file not found" in capsys.readouterr().err


def test_clients_ignore_proxy_settings(stub, monkeypatch):
    server = stub()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    embedder = make_embedder(config_for(url_of(server)))
    assert embedder.embed_query("text") == [1.0, 0.0]
    assert server.requests == ["/api/embed"]


def test_clients_do_not_follow_redirects(stub):
    target = stub()
    redirecting = stub(redirect_to=url_of(target))
    embedder = make_embedder(config_for(url_of(redirecting)))
    with pytest.raises(Exception):
        embedder.embed_query("text")
    assert target.requests == []


def test_clients_ignore_the_ollama_host_variable(stub, monkeypatch):
    server = stub()
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    assert make_embedder(config_for(url_of(server))).embed_query("text") == [1.0, 0.0]
    assert server.requests == ["/api/embed"]


def test_cloud_alias_is_refused_after_asking_ollama(stub):
    server = stub(responses={"/api/show": {"remote_host": "https://ollama.com:443", "remote_model": "x"}})
    with pytest.raises(ConfigError, match="cloud"):
        ensure_local_model(config_for(url_of(server)), "innocent-name:latest")


def test_local_model_passes_the_ollama_check(stub):
    details = {"capabilities": ["completion"], "model_info": {"gemma3.context_length": 131072}}
    server = stub(responses={"/api/show": details})
    assert context_length(ensure_local_model(config_for(url_of(server)), "gemma3:latest")) == 131072


def test_unreachable_ollama_is_a_clean_error():
    with pytest.raises(OllamaError, match="Cannot reach"):
        show_model(config_for("http://127.0.0.1:9"), "gemma3:latest")


def test_tracing_stays_off_even_when_the_environment_enables_it():
    code = (
        "import md_rag\n"
        "from langsmith.utils import tracing_is_enabled\n"
        "import langchain_core\n"
        "print(tracing_is_enabled())\n"
    )
    environment = {
        "LANGSMITH_TRACING": "true",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGSMITH_API_KEY": "fake",
        "PATH": "/usr/bin",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=True
    )
    assert result.stdout.strip() == "False"


def test_index_directory_is_ignored_by_git():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".rag_index/" in lines
    assert "external_file" in lines


@pytest.mark.parametrize(
    ("estimated", "reported"),
    [(5471, 2051), (5629, 2051), (5886, 2051), (6439, 2051), (7506, 2051), (9125, 2051), (10725, 2051), (16097, 2051)],
)
def test_overflow_of_any_size_is_detected(estimated, reported):
    # Measured on Ollama 0.40 at num_ctx 4096, from 2% to 200% over.
    assert looks_truncated(estimated, reported, 4096)


def test_overflow_at_a_different_window_is_detected():
    # Measured: 4520 real tokens at num_ctx 2048 reported 1027.
    assert looks_truncated(5700, 1027, 2048)


def test_planned_prompt_that_overflows_is_detected():
    # The largest prompt the planner sends at the defaults, reporting the fixed
    # overflow count. The ratio alone (0.71) would miss this.
    assert looks_truncated(11571, 16384 // 2 + 3, 16384)


@pytest.mark.parametrize(
    ("estimated", "reported"),
    [(2667, 2036), (3753, 2861), (4820, 3671), (5254, 4001)],
)
def test_prompts_up_to_the_window_are_not_reported(estimated, reported):
    # Measured on Ollama 0.40 at num_ctx 4096, from 50% to 98% full.
    assert not looks_truncated(estimated, reported, 4096)


def test_healthy_map_prompts_are_not_reported():
    # Measured map-reduce prompts. 8290 of 16384 is close to half the window.
    assert not looks_truncated(11141, 8290, 16384)
    assert not looks_truncated(8031, 6045, 16384)


def test_a_count_near_the_overflow_value_is_not_reported():
    overflow_count = 16384 // 2 + 3
    assert not looks_truncated(11141, overflow_count - 1, 16384)
    assert not looks_truncated(11141, overflow_count + 1, 16384)


def test_small_prompts_are_not_reported():
    assert not looks_truncated(100, 60, 16384)


def test_a_missing_count_is_not_reported_as_truncated():
    assert not looks_truncated(5700, None, 2048)
