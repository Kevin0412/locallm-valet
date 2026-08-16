# locallm-valet

[English](README.md) | **简体中文**

**本地 LLM 代客泊车**：空闲时把模型"泊"走（卸载释放显存），要用时"开"回来（按需启动）。
单设备、单活动模型，管理**你自己机器上的任意 OpenAI 兼容推理后端**——
SGLang / vLLM / llama.cpp / OpenVINO / ...

> 就像代客泊车：模型不用时被卸载，GPU/CPU/NPU 还给你；请求到来时模型被启动
> （仅当显存/内存充足）、健康检查后开始服务——一切由 OpenAI API 的 `model` 字段驱动。
> 模型繁忙时拒绝切换；连续空闲（默认 1 小时，**可配置**）后自动泊走。

客户端永远只访问一个固定地址：

```text
http://server:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://server:8000/v1", api_key="...")
client.chat.completions.create(model="qwen2.5-7b", messages=[...])  # 按需启动
client.chat.completions.create(model="llama3.1-8b", messages=[...])  # 自动切换后端/模型
```

## 特性

- **后端无关**：一行 `command_template` 适配任意 OpenAI 兼容服务（SGLang / vLLM /
  llama.cpp / OpenVINO ...）。无后端专属代码，无硬编码参数。
- **按需生命周期**：网关负责拉起、健康检查、停止、切换后端进程；空闲时后端绝非常驻。
- **资源感知**：双门控——GPU 显存（NVML，无 NVIDIA 驱动时自动跳过）+ 系统内存
  （psutil，跨平台）。绝不硬启动撞 OOM。
- **不抢占**：有请求在途时拒绝切换——流式连接永不被切断（除非 force-stop）。
- **正式状态机**（`STOPPED / STARTING / RUNNING / STOPPING / SWITCHING`）+ 全局
  生命周期锁：并发请求绝不可能拉起两个后端。
- **空闲看门狗**：空闲超时自动卸载（默认 1 小时，**可配置不硬编码**——YAML 或
  `LOCALLM_VALET_IDLE_TIMEOUT_SECONDS`）。
- **token 用量统计**：每次请求（普通 + 流式）真实 tokens 落 SQLite，内置用量看板。
- **API-key 认证**：可选 `Authorization: Bearer` 门禁，覆盖全部数据接口。
- **Windows 友好**：纯 asyncio、无 Unix 专属依赖；CPU/NPU 机器（如 Intel Core Ultra
  + OpenVINO）走 RAM 门控。

## 架构

```text
Client
  │
  │ OpenAI-compatible API (固定地址 :8000)
  ▼
locallm-valet :8000
  │
  ├── 请求路由（按 model 字段）
  ├── 状态机 STOPPED/STARTING/RUNNING/STOPPING/SWITCHING
  ├── 资源门控（VRAM via NVML 可选 + RAM via psutil）
  ├── 后端生命周期（启动/停止/切换，全局串行锁）
  ├── 健康检查轮询（绝不盲等 sleep）
  ├── active_requests / last_activity 记账
  └── idle watchdog（可配置超时）
          │
          ▼
     后端 127.0.0.1:30000（按需拉起，由网关托管）
     SGLang / vLLM / llama.cpp / OpenVINO ...（一行 command_template 适配）
          │
          ▼
     GPU（NVML）/ CPU / NPU（RAM）
```

## 快速开始

```bash
pip install -e ".[dev]"
# 或：pip install locallm-valet
cp config.example.yaml config.yaml      # 然后按需修改
python -m locallm_valet --config config.yaml
# 或：locallm-valet --config config.yaml
```

```bash
# 触发后端冷启动（首个请求要等模型加载；客户端 timeout 要 ≥ backend.startup_timeout_seconds）
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"你好"}]}'

curl http://127.0.0.1:8000/gateway/status   # 状态、模型、显存 + 内存
```

## 后端接入

启动命令 = **模板**（`backend.command_template`），占位符：
`{python} {model_path} {model_name} {host} {port} {device} {extra_args}`
（自动引号转义；模板没写 `{extra_args}` 时自动追加末尾）。

```yaml
# SGLang（默认模板，不配置即用）
command_template: "{python} -m sglang.launch_server --model-path {model_path} --host {host} --port {port} --tp-size 1 {extra_args}"

# vLLM
command_template: "{python} -m vllm.entrypoints.openai.api_server --model {model_path} --host {host} --port {port} {extra_args}"

# llama.cpp（Linux / macOS / Windows 通用，.exe 路径直接写）
command_template: "llama-server -m {model_path} --host {host} --port {port} {extra_args}"

# OpenVINO GenAI（自定义 OpenAI 兼容服务，CPU/NPU）
command_template: "{python} -m my_openvino_server --model {model_path} --port {port} {extra_args}"
```

后端只需满足：① 暴露 OpenAI 兼容 API；② `health_path` 就绪后返回 200。

> **vLLM 注意**：vLLM 会严格校验请求的 `model` 字段与 `--served-model-name`（默认是
> 模型路径名）是否一致。请在 `extra_args` 里加 `--served-model-name <registry 名称>`
> 与客户端发送的名称对齐。

### 一个 registry 混用多个后端

全局模板作用于所有**未定义自己的模板**的模型（`models.<name>.backend.command_template`
按模型覆盖；`backend.health_path` 同样支持按模型覆盖，适配就绪路径不同的后端）。
由于同一时间只跑一个模型，共用的内部端口（30000）不会冲突：

```yaml
models:
  vllm-qwen2.5-7b:
    path: /models/Qwen2.5-7B-Instruct
    required_vram_gib: 18
    backend:
      command_template: "{python} -m vllm.entrypoints.openai.api_server --model {model_path} --host {host} --port {port} {extra_args}"
      extra_args: ["--served-model-name", "vllm-qwen2.5-7b"]
  llama3.2-3b-gguf:
    path: C:\models\llama-3.2-3b-instruct-q8.gguf
    required_ram_gib: 8
    backend:
      command_template: "llama-server -m {model_path} --host {host} --port {port} {extra_args}"
      extra_args: ["--ctx-size", 8192]
```

`extra_args` 里的数字会自动转成字符串（不用加引号）。

## 资源门控（显存 + 内存）

每个模型配置 `required_vram_gib`（显存）与 `required_ram_gib`（系统内存），
门槛 = 需求 + `memory.safety_margin_gib`：

- **VRAM**（NVML）：**GPU 后端的首要门控**——显存不足直接拒绝。仅当有 NVIDIA
  驱动时检查；CPU/NPU 机器自动跳过。
- **RAM**（psutil `available`）：**可选**（`required_ram_gib > 0` 才检查）。主要用于
  CPU/NPU 后端（llama.cpp / OpenVINO）——那里 RAM 才是真正的算力资源。GPU 服务
  通常留 0：显存够就能起（内存紧只是加载慢/换页，不是不能跑）。
- `0`（默认）= 该资源不检查。
- 切换时以**停止旧模型后重新读取的实际资源**为准（等内存释放完再判断）。

```text
模型需要 VRAM 40G + margin 4G = 44G；当前 free 44.1G ✓ 启动
模型需要 RAM  32G + margin 4G = 36G；当前 available 34G ✗ 503 insufficient_memory
```

## 端口与路由

| 端口 | 绑定 | 用途 | 配置 |
|---|---|---|---|
| **8000** | `0.0.0.0`（默认） | 客户端唯一入口（OpenAI API + 管理面） | `server.port` / `LOCALLM_VALET_PORT` |
| **30000** | `127.0.0.1`（默认） | 后端内部端口，仅网关可达 | `backend.port` |

### OpenAI 兼容面（客户端用，API-key 保护）

| 方法 | 路径 | 行为 |
|---|---|---|
| POST | `/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`、任意 `/v1/*` | 按 `model` 门控（启动/切换/直通）后原样转发，含 SSE |
| GET | `/v1/models` | registry 列表（后端停止时也可列） |
| GET | 其他 `/v1/*` | 仅当有模型已加载时转发 |

流式请求自动注入 `stream_options.include_usage=true` 以捕获真实 token 数，对客户端透明。

### 管理面（`/gateway/*`，API-key 保护）

| 方法 | 路径 | 行为 |
|---|---|---|
| GET | `/gateway/status` | 状态机 + 活跃请求 + idle 秒数 + 内存（vram/ram）+ `max_context_tokens`（已加载模型的真实 KV 容量，加载时实测——配置的 `--context-length` 只是声明，真实上限因机器而异，例如 48G 卡上 27B 稠密 FP8 约 15.8 万 tokens） |
| GET | `/gateway/models` | registry + 每个模型的加载状态 + `max_context_tokens`（仅已加载模型有值；加载过才知道） |
| POST | `/gateway/stop` | 正常关闭——空闲时任意状态接受（含取消进行中的启动/切换）；忙时 503 |
| POST | `/gateway/force-stop` | 强制关闭——无条件清理（切断活跃流式连接），用于抢回资源 |
| POST | `/gateway/preload/{model}` | 提前暖模型，等 ready |
| GET | `/gateway/usage` | 用量聚合：`model` / `since` / `until` / `group_by=hour\|day\|none` / `limit` |
| GET | `/gateway/dashboard` | 内置用量看板（自包含 HTML/JS，无 CDN） |

`/docs`、`/redoc`、`/openapi.json` 为 FastAPI 自带（仅 schema，免认证）。

### API-key 认证

配置 `server.api_key`（字符串或列表）或 `LOCALLM_VALET_API_KEY`（逗号分隔多 key）。
之后所有 `/v1/*` 与 `/gateway/*` 数据接口要求 `Authorization: Bearer <key>`
（否则 401 `authentication_error`）。免认证例外：`/docs`、`/redoc`、`/openapi.json`、
`/gateway/dashboard`（静态壳，页面 401 时自动弹窗要 key）。**未配置 key = 全开**。

## 状态机与并发

```text
STOPPED ──start(model)──▶ STARTING ──health ok──▶ RUNNING(model)
   ▲                        │                       │
   │◀──failed/timeout───────┘                       │ stop/空闲
   │                                                ▼
   └─────────────────────────────────────────── STOPPING ──▶ STOPPED
RUNNING(a) ──switch(b)──▶ SWITCHING(a→b) ──▶ RUNNING(b)  （失败回 STOPPED）
```

| 当前状态 | 请求 model=当前启动中的模型 | 请求 model=其他模型 |
|---|---|---|
| STOPPED | 触发启动 | 触发启动（串行） |
| STARTING(m) | 等待启动完成，再转发 | 503 `model_switch_busy` |
| RUNNING(m) | **直接转发**（不抢锁） | 忙 → busy；空闲 → 切换 |
| SWITCHING(a→b) | 等待切换完成 | 503 `model_switch_busy` |
| STOPPING | 503 `backend_unavailable` | 同左 |

客户端中途断开不会卡死状态机（取消清理、状态回 STOPPED）。

## 冷启动与超时

未命中已加载模型时请求会挂起等待：后端启动 → health ready → 转发。三个时间：

| 时间 | 默认 | 说明 |
|---|---|---|
| 模型加载耗时 | 实测 SGLang + 35B FP8 86~188s | 真正的等待时长 |
| `backend.startup_timeout_seconds` | 180（示例配置 600） | 服务端兜底 → 503 `backend_startup_timeout`，不无限挂 |
| 客户端 timeout | openai-python 600s；**httpx/requests 5s** | 真正风险点：设 `timeout=600` 或 `None` |

启动期间同模型并发请求共享同一次启动；异模型请求立即 503 busy。

## 错误类型

| HTTP | error.type | 含义 |
|---|---|---|
| 503 | `insufficient_memory` | 显存和/或内存不足（消息指明哪一项） |
| 503 | `model_switch_busy` | 模型忙/切换中，拒绝切换 |
| 404 | `model_not_found` | registry 里没有该模型 |
| 503 | `backend_startup_failed` / `backend_startup_timeout` | 后端启动失败/超时 |
| 503 | `backend_unavailable` | 后端不可达/正在停止 |
| 400 | `invalid_request` | body 非法 / 缺 `model` 字段 |
| 401 | `authentication_error` | API key 缺失/错误 |

## 配置

```yaml
server:   # manager 监听地址 + 可选 api_key
backend:  # 内部端口、health path、command_template、env
memory:   # device（仅 CUDA）、safety_margin_gib、释放等待超时
idle:     # timeout_seconds（可配置，不硬编码）、检查间隔
usage:    # SQLite token 统计 + 看板（enabled / db_path）
models:   # 每模型：path、required_vram_gib、required_ram_gib、backend.extra_args/env
```

环境变量覆盖：`LOCALLM_VALET_CONFIG`、`LOCALLM_VALET_HOST`、`LOCALLM_VALET_PORT`、
`LOCALLM_VALET_API_KEY`、`LOCALLM_VALET_IDLE_TIMEOUT_SECONDS`。
更早项目名下的环境变量与配置段仍作为弃用别名兼容，便于平滑迁移（见 `locallm_valet/config.py`）。

## Windows 支持

- 纯 Python 3.10+ / asyncio；无 Unix 专属依赖（无 signal/fcntl/resource）；psutil 全平台读内存。
- Windows 上 `terminate()` 即硬终止，SIGTERM→SIGKILL 升级流程同样有效。
- 模板可直接写 `.exe` 路径（含空格用引号包住）。
- 无 NVIDIA 驱动（Intel CPU/NPU）→ 显存检查自动跳过，RAM 检查照常。
- 控制台 / NSSM / `deploy/locallm-valet.bat` 运行（Linux 用 `deploy/locallm-valet.service`）。

## 开发

```bash
pip install -e ".[dev]"
# 或：pip install locallm-valet
pytest -v     # 97 个测试，fake memory/runner——无需 GPU
```

目录结构：

```text
locallm_valet/
├── __main__.py     CLI 入口
├── api.py          FastAPI：/v1 代理 + /gateway 管理 + usage/看板 + API-key 认证
├── config.py       YAML 配置（backend 模板 / 双内存门控 / 兼容别名）
├── dashboard.py    用量看板页面（内联 HTML/CSS/JS）
├── errors.py       typed 错误体系
├── memory.py       VRAM（NVML 可选）+ RAM（psutil）监控
├── manager.py      状态机 + 生命周期锁 + watchdog + 记账
├── proxy.py        OpenAI 兼容反向代理（SSE + usage 捕获）
├── runner.py       后端子进程：模板命令 / health / 停止
├── state.py        状态枚举
└── usage.py        SQLite token 用量记录与聚合
```

## License

MIT
