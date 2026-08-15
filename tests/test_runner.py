"""Launch command construction tests."""

import sys

from sglang_manager.config import ModelSglangArgs, ModelSpec, SglangConfig
from sglang_manager.runner import build_launch_command


def make_spec(**kwargs) -> ModelSpec:
    return ModelSpec(
        name="qwen",
        path="/models/qwen",
        required_vram_gib=30,
        sglang=ModelSglangArgs(
            mem_fraction_static=0.87,
            context_length=262144,
            extra_args=["--kv-cache-dtype", "fp8_e4m3"],
            env={"SGLANG_USE_MODELSCOPE": "true"},
        ),
        **kwargs,
    )


def flag_value(cmd, flag):
    assert flag in cmd, f"{flag} missing in {cmd}"
    return cmd[cmd.index(flag) + 1]


def test_default_base_command():
    cmd = build_launch_command(SglangConfig(), make_spec(), gpu_device=0)
    assert cmd[:3] == [sys.executable, "-m", "sglang.launch_server"]
    assert flag_value(cmd, "--model-path") == "/models/qwen"
    assert flag_value(cmd, "--port") == "30000"
    assert flag_value(cmd, "--host") == "127.0.0.1"
    assert flag_value(cmd, "--tp-size") == "1"
    assert flag_value(cmd, "--mem-fraction-static") == "0.87"
    assert flag_value(cmd, "--context-length") == "262144"
    assert "--kv-cache-dtype" in cmd


def test_custom_base_command():
    cfg = SglangConfig(command=["/opt/conda/envs/qwen3.5/bin/python", "-m", "sglang.launch_server"])
    cmd = build_launch_command(cfg, make_spec(), gpu_device=0)
    assert cmd[:3] == ["/opt/conda/envs/qwen3.5/bin/python", "-m", "sglang.launch_server"]


def test_optional_args_omitted():
    spec = make_spec()
    spec.sglang = ModelSglangArgs()
    cmd = build_launch_command(SglangConfig(), spec, gpu_device=1)
    assert "--mem-fraction-static" not in cmd
    assert "--context-length" not in cmd


def test_model_extra_args_appended():
    cmd = build_launch_command(SglangConfig(), make_spec(), gpu_device=0)
    assert cmd[-2:] == ["--kv-cache-dtype", "fp8_e4m3"]
