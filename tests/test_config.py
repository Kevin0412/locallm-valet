"""Config loading / validation tests (new backend-agnostic schema)."""

import pytest

from locallm_valet.config import ConfigError, DEFAULT_COMMAND_TEMPLATE, load_config

SAMPLE = """
server: {host: 0.0.0.0, port: 8123}
backend:
  host: 127.0.0.1
  port: 30000
  startup_timeout_seconds: 90
  command_template: "{python} -m sglang.launch_server --model-path {model_path} --host {host} --port {port} {extra_args}"
  env: {SGLANG_USE_MODELSCOPE: "true"}
memory:
  device: 1
  safety_margin_gib: 3
  release_timeout_seconds: 90
idle: {timeout_seconds: 7200, check_interval_seconds: 10}
models:
  qwen:
    path: Qwen/Qwen3.6-35B-A3B-FP8
    required_vram_gib: 42
    required_ram_gib: 36
    backend:
      extra_args: [--kv-cache-dtype, fp8_e4m3]
      env: {LD_PRELOAD: /usr/lib/libstdc++.so.6}
"""


def write_sample(tmp_path, text=SAMPLE):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_load_config(tmp_path, monkeypatch):
    for var in ("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", "LOCALLM_VALET_PORT", "LOCALLM_VALET_HOST",
                "LLM_GATEWAY_IDLE_TIMEOUT_SECONDS", "LLM_GATEWAY_PORT", "LLM_GATEWAY_HOST",
                "SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", "SGLANG_MANAGER_PORT", "SGLANG_MANAGER_HOST"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(write_sample(tmp_path))
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 8123
    assert cfg.backend.base_url == "http://127.0.0.1:30000"
    assert "sglang.launch_server" in cfg.backend.command_template
    assert cfg.backend.env == {"SGLANG_USE_MODELSCOPE": "true"}
    assert cfg.backend.startup_timeout_seconds == 90
    assert cfg.memory.device == 1
    assert cfg.memory.safety_margin_gib == 3
    assert cfg.memory.release_timeout_seconds == 90
    assert cfg.idle.timeout_seconds == 7200
    q = cfg.models["qwen"]
    assert q.path == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert q.required_vram_gib == 42
    assert q.required_ram_gib == 36
    assert q.backend.extra_args == ["--kv-cache-dtype", "fp8_e4m3"]
    assert q.backend.env == {"LD_PRELOAD": "/usr/lib/libstdc++.so.6"}
    assert q.backend.max_concurrency is None  # unset = unknown, CLI decides


def test_max_concurrency_parsing(tmp_path, monkeypatch):
    """backend.max_concurrency is optional; validated when present."""
    monkeypatch.delenv("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", raising=False)
    # valid positive int
    cfg = load_config(write_sample(tmp_path, SAMPLE.replace(
        "      extra_args: [--kv-cache-dtype, fp8_e4m3]",
        "      extra_args: [--kv-cache-dtype, fp8_e4m3]\n      max_concurrency: 4",
    )))
    assert cfg.models["qwen"].backend.max_concurrency == 4
    # explicit null = unknown
    cfg2 = load_config(write_sample(tmp_path, SAMPLE.replace(
        "      extra_args: [--kv-cache-dtype, fp8_e4m3]",
        "      extra_args: [--kv-cache-dtype, fp8_e4m3]\n      max_concurrency: null",
    )))
    assert cfg2.models["qwen"].backend.max_concurrency is None
    # zero / negative / non-int are rejected
    for bad in ("0", "-1", "true", "'four'"):
        with pytest.raises(ConfigError, match="max_concurrency"):
            load_config(write_sample(tmp_path, SAMPLE.replace(
                "      extra_args: [--kv-cache-dtype, fp8_e4m3]",
                f"      extra_args: [--kv-cache-dtype, fp8_e4m3]\n      max_concurrency: {bad}",
            )))


def test_default_command_template_is_sglang():
    """The default template keeps SGLang working out of the box; every other
    backend overrides it with one string."""
    assert "{model_path}" in DEFAULT_COMMAND_TEMPLATE
    assert "{extra_args}" in DEFAULT_COMMAND_TEMPLATE


def test_legacy_sections_still_accepted(tmp_path, monkeypatch, caplog):
    """Old sglang:/gpu: section names must keep working (deprecated)."""
    for var in ("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", "LLM_GATEWAY_IDLE_TIMEOUT_SECONDS",
                "SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS",
                "LOCALLM_VALET_PORT", "LLM_GATEWAY_PORT", "SGLANG_MANAGER_PORT"):
        monkeypatch.delenv(var, raising=False)
    legacy = SAMPLE.replace("backend:", "sglang:").replace("memory:", "gpu:")
    cfg = load_config(write_sample(tmp_path, legacy))
    assert cfg.backend.port == 30000
    assert cfg.memory.device == 1
    assert cfg.models["qwen"].required_ram_gib == 36
    assert any("deprecated" in r.message for r in caplog.records)


def test_legacy_command_list_rejected(tmp_path):
    with pytest.raises(ConfigError, match="command_template"):
        load_config(write_sample(tmp_path, SAMPLE.replace(
            'command_template: "{python} -m sglang.launch_server --model-path {model_path} --host {host} --port {port} {extra_args}"',
            "command: [/opt/venv/bin/python, -m, sglang.launch_server]",
        )))


def test_idle_timeout_env_override(tmp_path, monkeypatch):
    """The idle timeout must be overridable, never hardcoded."""
    monkeypatch.setenv("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("LOCALLM_VALET_PORT", "9000")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.idle.timeout_seconds == 1800
    assert cfg.server.port == 9000


def test_legacy_env_names_fallback_chain(tmp_path, monkeypatch):
    """Both earlier project-name env vars still work as fallbacks:
    LOCALLM_VALET_* -> LLM_GATEWAY_* -> SGLANG_MANAGER_*."""
    monkeypatch.delenv("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_IDLE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", "2700")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.idle.timeout_seconds == 2700

    monkeypatch.delenv("SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_IDLE_TIMEOUT_SECONDS", "2400")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.idle.timeout_seconds == 2400

    monkeypatch.setenv("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", "1800")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.idle.timeout_seconds == 1800  # primary wins


def test_idle_timeout_env_bad_value(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", "abc")
    with pytest.raises(ConfigError):
        load_config(write_sample(tmp_path))


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/xyz.yaml")


def test_missing_path(tmp_path):
    with pytest.raises(ConfigError, match="path"):
        load_config(write_sample(tmp_path, SAMPLE.replace("path: Qwen/Qwen3.6-35B-A3B-FP8", "no_path: x")))


def test_negative_memory_requirement(tmp_path):
    with pytest.raises(ConfigError, match="required_ram_gib"):
        load_config(write_sample(tmp_path, SAMPLE.replace("required_ram_gib: 36", "required_ram_gib: -1")))


def test_no_models(tmp_path):
    with pytest.raises(ConfigError, match="at least one model"):
        load_config(write_sample(tmp_path, "server: {}\nmodels: {}\n"))


def test_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write_sample(tmp_path, "models: [unclosed\n"))


def test_api_key_string(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALLM_VALET_API_KEY", raising=False)
    monkeypatch.delenv("SGLANG_MANAGER_API_KEY", raising=False)
    cfg = load_config(write_sample(tmp_path, SAMPLE.replace(
        "server: {host: 0.0.0.0, port: 8123}",
        "server: {host: 0.0.0.0, port: 8123, api_key: sk-test-123}",
    )))
    assert cfg.server.api_keys == ["sk-test-123"]


def test_api_key_list(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALLM_VALET_API_KEY", raising=False)
    monkeypatch.delenv("SGLANG_MANAGER_API_KEY", raising=False)
    cfg = load_config(write_sample(tmp_path, SAMPLE.replace(
        "server: {host: 0.0.0.0, port: 8123}",
        "server: {host: 0.0.0.0, port: 8123, api_key: [sk-a, sk-b]}",
    )))
    assert cfg.server.api_keys == ["sk-a", "sk-b"]


def test_api_key_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALLM_VALET_API_KEY", "sk-env-1, sk-env-2")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.server.api_keys == ["sk-env-1", "sk-env-2"]


def test_api_key_invalid_type(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALLM_VALET_API_KEY", raising=False)
    monkeypatch.delenv("SGLANG_MANAGER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="api_key"):
        load_config(write_sample(tmp_path, SAMPLE.replace(
            "server: {host: 0.0.0.0, port: 8123}",
            "server: {host: 0.0.0.0, port: 8123, api_key: 42}",
        )))


def test_per_model_backend_overrides(tmp_path, monkeypatch):
    """Per-model command_template / health_path are parsed and fall back to
    the global backend.* settings when absent."""
    for var in ("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", "SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    lines = [
        "      command_template: llama-server -m {model_path} {extra_args}",
        "      health_path: /healthz",
        "      extra_args: [--ctx-size, 8192]",
    ]
    text = SAMPLE.replace(
        "      extra_args: [--kv-cache-dtype, fp8_e4m3]",
        "\n".join(lines),
    )
    cfg = load_config(write_sample(tmp_path, text))
    q = cfg.models["qwen"]
    assert q.backend.command_template == "llama-server -m {model_path} {extra_args}"
    assert q.backend.health_path == "/healthz"
    assert q.backend.extra_args == ["--ctx-size", "8192"]
    # another model without overrides falls back to the global template
    text2 = text.replace(
        "models:\n  qwen:",
        "models:\n  other:\n    path: /m/other\n    backend:\n      extra_args: []\n  qwen:",
    )
    cfg2 = load_config(write_sample(tmp_path, text2))
    assert cfg2.models["other"].backend.command_template is None
    assert cfg2.models["other"].backend.health_path is None


def test_switch_when_busy_option(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALLM_VALET_IDLE_TIMEOUT_SECONDS", raising=False)
    cfg = load_config(write_sample(tmp_path, SAMPLE.replace(
        "server: {host: 0.0.0.0, port: 8123}",
        "server: {host: 0.0.0.0, port: 8123, switch_when_busy: wait, switch_wait_timeout_seconds: 30}",
    )))
    assert cfg.server.switch_when_busy == "wait"
    assert cfg.server.switch_wait_timeout_seconds == 30
    with pytest.raises(ConfigError, match="switch_when_busy"):
        load_config(write_sample(tmp_path, SAMPLE.replace(
            "server: {host: 0.0.0.0, port: 8123}",
            "server: {host: 0.0.0.0, port: 8123, switch_when_busy: kill}",
        )))
