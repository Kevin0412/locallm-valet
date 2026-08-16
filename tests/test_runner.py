"""Launch command template & process env construction tests."""

import sys

from llm_gateway.config import ModelBackendArgs, ModelSpec, BackendConfig
from llm_gateway.runner import build_backend_command, build_process_env


def make_spec(**kwargs) -> ModelSpec:
    kwargs.setdefault("path", "/models/qwen")
    kwargs.setdefault("name", "qwen")
    return ModelSpec(
        required_vram_gib=30,
        backend=ModelBackendArgs(
            extra_args=["--kv-cache-dtype", "fp8_e4m3"],
            env={"SGLANG_USE_MODELSCOPE": "true"},
        ),
        **kwargs,
    )


def test_default_sglang_template():
    cmd = build_backend_command(BackendConfig(), make_spec(), device=0)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    assert cmd[cmd.index("--model-path") + 1] == "/models/qwen"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "30000"
    assert cmd[cmd.index("--tp-size") + 1] == "1"
    assert cmd[-2:] == ["--kv-cache-dtype", "fp8_e4m3"]


def test_llama_cpp_template():
    """llama.cpp uses -m / --ctx-size; a custom template must win wholesale."""
    cfg = BackendConfig(command_template="llama-server -m {model_path} --host {host} --port {port} {extra_args}")
    cmd = build_backend_command(cfg, make_spec(), device=0)
    assert cmd[0] == "llama-server"
    assert cmd[cmd.index("-m") + 1] == "/models/qwen"
    assert "--tp-size" not in cmd  # no SGLang-specific flags leaked
    assert cmd[-2:] == ["--kv-cache-dtype", "fp8_e4m3"]


def test_vllm_template():
    cfg = BackendConfig(command_template="{python} -m vllm.entrypoints.openai.api_server --model {model_path} --port {port} {extra_args}")
    cmd = build_backend_command(cfg, make_spec(), device=0)
    assert cmd[0] == sys.executable
    assert cmd[cmd.index("--model") + 1] == "/models/qwen"


def test_template_without_extra_args_placeholder_appends():
    cfg = BackendConfig(command_template="my-server --model {model_path} --port {port}")
    cmd = build_backend_command(cfg, make_spec(), device=0)
    assert cmd[-2:] == ["--kv-cache-dtype", "fp8_e4m3"]  # appended at the end


def test_device_placeholder():
    cfg = BackendConfig(command_template="my-server --model {model_path} --device {device}")
    cmd = build_backend_command(cfg, make_spec(), device=3)
    assert cmd[cmd.index("--device") + 1] == "3"


def test_quoted_paths_with_spaces():
    spec = make_spec(path="/models/My Qwen Model")
    spec.backend.extra_args = []  # template has no {extra_args}; keep it clean
    cfg = BackendConfig(command_template="my-server -m {model_path}")
    cmd = build_backend_command(cfg, spec, device=0)
    assert cmd == ["my-server", "-m", "/models/My Qwen Model"]


# ------------------------------------------------------------ process env

def test_manager_pythonpath_does_not_leak():
    """The manager's PYTHONPATH must not reach the backend child — the child
    uses its own interpreter's site-packages."""
    base = {"PATH": "/usr/bin", "PYTHONPATH": "/home/test/llm-gateway/.deps:/home/test/llm-gateway"}
    env = build_process_env(BackendConfig(), make_spec(), device=3, base_env=base)
    assert "PYTHONPATH" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PATH"] == "/usr/bin"


def test_pythonpath_passthrough_when_configured():
    cfg = BackendConfig(env={"PYTHONPATH": "/opt/custom"})
    spec = make_spec()
    env = build_process_env(cfg, spec, device=0, base_env={"PYTHONPATH": "/leaked"})
    assert env["PYTHONPATH"] == "/opt/custom"


def test_model_env_overrides_global_env():
    cfg = BackendConfig(env={"SGLANG_USE_MODELSCOPE": "true", "A": "global"})
    spec = make_spec()
    spec.backend.env = {"A": "model", "LD_PRELOAD": "/lib/libstdc++.so.6"}
    env = build_process_env(cfg, spec, device=0, base_env={})
    assert env["A"] == "model"
    assert env["LD_PRELOAD"] == "/lib/libstdc++.so.6"
    assert env["SGLANG_USE_MODELSCOPE"] == "true"


# ------------------------------------------- per-model overrides (mixed backends)

def test_per_model_command_template_override():
    """One registry can mix backends: a per-model template wins over the
    global one (e.g. vLLM model + llama.cpp model side by side)."""
    global_cfg = BackendConfig()  # SGLang default
    vllm = make_spec(name="vllm-qwen", path="/models/Qwen2.5-7B")
    vllm.backend.command_template = (
        "{python} -m vllm.entrypoints.openai.api_server --model {model_path} "
        "--host {host} --port {port} {extra_args}"
    )
    llama = make_spec(name="llama3.2-3b", path="/models/llama-3.2-3b-q8.gguf")
    llama.backend.command_template = "llama-server -m {model_path} --host {host} --port {port} {extra_args}"

    cmd_vllm = build_backend_command(global_cfg, vllm, device=0)
    cmd_llama = build_backend_command(global_cfg, llama, device=0)

    assert cmd_vllm[0] == sys.executable
    assert "vllm.entrypoints.openai.api_server" in cmd_vllm
    assert cmd_vllm[cmd_vllm.index("--model") + 1] == "/models/Qwen2.5-7B"
    assert "--model-path" not in cmd_vllm  # no SGLang flags leaked
    assert cmd_llama[0] == "llama-server"
    assert cmd_llama[cmd_llama.index("-m") + 1] == "/models/llama-3.2-3b-q8.gguf"
    assert "--model-path" not in cmd_llama


class _FakeProc:
    returncode = None
    pid = 9999
    stdout = None
    stderr = None

    async def wait(self):
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class _FakeHttp:
    def __init__(self):
        self.urls = []

    async def get(self, url):
        self.urls.append(url)

        class _R:
            status_code = 200

        return _R()

    async def aclose(self):
        pass


async def test_per_model_health_path_override(monkeypatch):
    """Different backends may use different readiness paths; the per-model
    health_path must be used for polling."""
    import asyncio

    from llm_gateway.runner import BackendRunner

    async def fake_spawn(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    http = _FakeHttp()
    cfg = BackendConfig(host="127.0.0.1", port=30000, health_path="/health")
    spec = make_spec()
    spec.backend.health_path = "/healthz"  # e.g. a custom OpenVINO server
    runner = BackendRunner(cfg, http_client=http)  # type: ignore[arg-type]
    await runner.start(spec)
    await runner.wait_health(10)
    assert http.urls == ["http://127.0.0.1:30000/healthz"]
    assert runner.health_path == "/healthz"
    await runner.stop(1)
