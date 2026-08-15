"""Config loading / validation tests."""

import pytest

from sglang_manager.config import ConfigError, load_config

SAMPLE = """
server: {host: 0.0.0.0, port: 8123}
sglang:
  host: 127.0.0.1
  port: 30000
  startup_timeout_seconds: 90
  command: [/opt/venv/bin/python, -m, sglang.launch_server]
  env: {SGLANG_USE_MODELSCOPE: "true"}
gpu: {device: 1, safety_margin_gib: 3}
idle: {timeout_seconds: 7200, check_interval_seconds: 10}
models:
  qwen:
    path: Qwen/Qwen3.6-35B-A3B-FP8
    required_vram_gib: 42
    sglang:
      mem_fraction_static: 0.87
      context_length: 262144
      extra_args: [--kv-cache-dtype, fp8_e4m3]
      env: {LD_PRELOAD: /usr/lib/libstdc++.so.6}
"""


def write_sample(tmp_path, text=SAMPLE):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_load_config(tmp_path, monkeypatch):
    for var in ("SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", "SGLANG_MANAGER_PORT", "SGLANG_MANAGER_HOST"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(write_sample(tmp_path))
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 8123
    assert cfg.sglang.base_url == "http://127.0.0.1:30000"
    assert cfg.sglang.command == ["/opt/venv/bin/python", "-m", "sglang.launch_server"]
    assert cfg.sglang.env == {"SGLANG_USE_MODELSCOPE": "true"}
    assert cfg.sglang.startup_timeout_seconds == 90
    assert cfg.gpu.device == 1
    assert cfg.gpu.safety_margin_gib == 3
    assert cfg.idle.timeout_seconds == 7200
    q = cfg.models["qwen"]
    assert q.path == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert q.required_vram_gib == 42
    assert q.sglang.mem_fraction_static == 0.87
    assert q.sglang.context_length == 262144
    assert q.sglang.extra_args == ["--kv-cache-dtype", "fp8_e4m3"]
    assert q.sglang.env == {"LD_PRELOAD": "/usr/lib/libstdc++.so.6"}


def test_idle_timeout_env_override(tmp_path, monkeypatch):
    """The idle timeout must be overridable, never hardcoded."""
    monkeypatch.setenv("SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("SGLANG_MANAGER_PORT", "9000")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.idle.timeout_seconds == 1800
    assert cfg.server.port == 9000


def test_idle_timeout_env_bad_value(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS", "abc")
    with pytest.raises(ConfigError):
        load_config(write_sample(tmp_path))


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/xyz.yaml")


def test_missing_required_vram(tmp_path):
    with pytest.raises(ConfigError, match="required_vram_gib"):
        load_config(write_sample(tmp_path, SAMPLE.replace("required_vram_gib: 42", "path_only: true")))


def test_no_models(tmp_path):
    with pytest.raises(ConfigError, match="at least one model"):
        load_config(write_sample(tmp_path, "server: {}\nmodels: {}\n"))


def test_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write_sample(tmp_path, "models: [unclosed\n"))


def test_api_key_string(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_MANAGER_API_KEY", raising=False)
    cfg = load_config(write_sample(tmp_path, SAMPLE.replace(
        "server: {host: 0.0.0.0, port: 8123}",
        "server: {host: 0.0.0.0, port: 8123, api_key: sk-test-123}",
    )))
    assert cfg.server.api_keys == ["sk-test-123"]


def test_api_key_list(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_MANAGER_API_KEY", raising=False)
    cfg = load_config(write_sample(tmp_path, SAMPLE.replace(
        "server: {host: 0.0.0.0, port: 8123}",
        "server: {host: 0.0.0.0, port: 8123, api_key: [sk-a, sk-b]}",
    )))
    assert cfg.server.api_keys == ["sk-a", "sk-b"]


def test_api_key_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_MANAGER_API_KEY", "sk-env-1, sk-env-2")
    cfg = load_config(write_sample(tmp_path))
    assert cfg.server.api_keys == ["sk-env-1", "sk-env-2"]


def test_api_key_invalid_type(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_MANAGER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="api_key"):
        load_config(write_sample(tmp_path, SAMPLE.replace(
            "server: {host: 0.0.0.0, port: 8123}",
            "server: {host: 0.0.0.0, port: 8123, api_key: 42}",
        )))
