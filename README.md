# llm-gateway

后端无关的 **LLM 生命周期网关**（原 sglang-manager，已泛化改名）：单设备、单活动模型，
管理**任意 OpenAI 兼容推理后端**——SGLang / vLLM / llama.cpp / OpenVINO / ...

> 根据 OpenAI API 的 `model` 字段按需启动和切换模型；后端已运行且模型匹配时直接路由，
> 未运行时仅在**显存/内存充足**时启动，模型繁忙时拒绝切换，资源不足直接报错，
> 并在连续空闲（默认 1 小时，**可配置**）无活跃请求后自动卸载释放资源。

客户端永远只访问一个固定地址：

```text
http://server:8000/v1
```

```python
client = OpenAI(base_url="http://server:8000/v1", api_key="...")
client.chat.completions.create(model="qwen3.6-35b", messages=[...])   # 按需启动
client.chat.completions.create(model="llama3.2-3b", messages=[...])   # 自动切换后端/模型
```

## 架构

```text
Client
  │
  │ OpenAI-compatible API (固定地址 :8000)
  ▼
llm-gateway :8000
  │
  ├── 请求路由（按 model 字段）
  ├── 状态机 STOPPED/STARTING/RUNNING/STOPPING/SWITCHING
  ├── 资源检查（VRAM via NVML 可选 + RAM via psutil 跨平台）
  ├── 后端生命周期（启动/停止/切换，全局串行锁）
  ├── 健康检查轮询（不 sleep 硬等）
  ├── active_requests / last_activity 记账
  └── idle watchdog（可配置超时，默认 1 小时）
          │
          ▼
     推理后端 127.0.0.1:30000（按需拉起，由网关托管）
     SGLang / vLLM / llama.cpp / OpenVINO ...（command_template 一行适配）
          │
          ▼
     GPU（NVML）/ CPU / NPU（RAM）
```

## 后端无关性（高泛用性）

启动命令 = **模板字符串**（`backend.command_template`），占位符：

| 占位符 | 含义 |
|---|---|
| `{python}` | 当前解释器（自动引号转义） |
| `{model_path}` | 模型路径 / hub id（自动引号转义） |
| `{model_name}` | registry 名称 |
| `{host}` / `{port}` | 后端监听地址 |
| `{device}` | `memory.device` |
| `{extra_args}` | 模型级 `backend.extra_args`（自动引号转义；模板没写时自动追加末尾） |

各后端一行示例（详见 `config.example.yaml`）：

```yaml
# SGLang（默认模板，不配置即用）
command_template: "{python} -m sglang.launch_server --model-path {model_path} --host {host} --port {port} --tp-size 1 {extra_args}"
# vLLM
command_template: "{python} -m vllm.entrypoints.openai.api_server --model {model_path} --host {host} --port {port} {extra_args}"
# llama.cpp（Windows 同样适用，.exe 路径直接写）
command_template: "llama-server -m {model_path} --host {host} --port {port} {extra_args}"
# OpenVINO GenAI（自定义 OpenAI 兼容服务，CPU/NPU）
command_template: "{python} -m my_openvino_server --model {model_path} --port {port} {extra_args}"
```

后端只需满足：① 暴露 OpenAI 兼容 API；② `health_path` 就绪后返回 200。
网关不做任何后端专属假设——没有硬编码的 SGLang 参数。

## 资源检查（显存 + 内存）

每个模型配置 `required_vram_gib`（显存）与 `required_ram_gib`（系统内存），
门槛 = 需求 + `memory.safety_margin_gib`：

- **VRAM**：NVML 读取；**仅当 NVML 可用时检查**（有 NVIDIA 驱动的机器）。
  CPU/NPU 机器（Windows + OpenVINO / llama.cpp CPU）没有 NVIDIA 驱动时
  **自动跳过显存检查**，只查 RAM——这就是为你的 Intel CPU/NPU 机器准备的行为。
- **RAM**：psutil 读取 `available`（Linux / Windows / macOS 全平台），
  覆盖 llama.cpp 权重加载、OpenVINO 模型 + 运行时开销、NPU 共享内存等。
- `0`（默认）= 该资源不检查。
- 切换时以**停止旧模型后重新读取的实际资源**为准（等显存释放完再判断）。

```text
模型需要 VRAM 40G + margin 4G = 44G；当前 free 44.1G ✓ 启动
模型需要 RAM  32G + margin 4G = 36G；当前 available 34G ✗ 503 insufficient_memory
```

## 核心规则（V1 边界）

1. **单设备单活动模型**：同一时间只加载一个模型。
2. **请求按 `model` 决定行为**：`RUNNING(qwen)` + `model=qwen` → 直接转发，
   不重新检查资源；`STOPPED` + `model=qwen` → 检查资源 → 启动 → health → 转发，
   资源不足直接 503（绝不硬启动然后撞 OOM）。
3. **切换**：`RUNNING(a)` + `model=b` 且 `active_requests == 0` → 停 a → 等进程退出 →
   等资源释放 → 重新读取 → 判断 b → 启动 → health → 转发。
4. **忙时不抢占**：`active_requests > 0` 时收到异模型请求 → 503 `model_switch_busy`。
5. **记账**：请求进入 `active_requests += 1`；流式请求等 SSE 完整关闭后才 `-= 1`。
6. **空闲自动卸载**：`RUNNING` 且无请求且超时（可配置，不硬编码）→ 停止后端释放资源。
7. **资源分 managed / external**：只读取，只管理自己拉起的进程，不杀其他 workload。
8. **生命周期全局串行**：启动/停止/切换互斥；`RUNNING` 同模型请求不经过该锁。
9. **健康检查**：轮询 `health_path`（默认 `/health`），`startup_timeout_seconds` 内
   不 ready → 503 `backend_startup_timeout` / `backend_startup_failed`。
10. **进程管理**：网关常驻（systemd / Windows 服务），后端按需拉起，后端本身不自启。

## 状态机与并发

```text
STOPPED ──start(model)──▶ STARTING ──health ok──▶ RUNNING(model)
   ▲                        │                       │
   │◀──failed/timeout───────┘                       │ stop/无请求
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

客户端中途断开不会卡死状态机（启动/切换被安全取消，进程清理、状态回 STOPPED）。

## 冷启动等待与超时

未命中已加载模型时请求会挂起等待：后端启动 → health ready → 转发。
三个时间：模型加载耗时（实测 SGLang+35B FP8 86~188s）、服务端兜底
`backend.startup_timeout_seconds`（超时 503，不无限挂）、**客户端 timeout**
（openai-python 默认 600s；httpx/requests 默认 5s 必须调大，如 `timeout=600` 或 `None`）。
启动期间同模型并发请求共享同一次启动；异模型请求立即 503 busy。

## 错误类型

| HTTP | error.type | 含义 |
|---|---|---|
| 503 | `insufficient_memory` | VRAM 和/或 RAM 不足（消息里说明哪一项） |
| 503 | `model_switch_busy` | 模型忙/切换中，拒绝切换 |
| 404 | `model_not_found` | registry 里没有该模型 |
| 503 | `backend_startup_failed` / `backend_startup_timeout` | 后端启动失败/超时 |
| 503 | `backend_unavailable` | 后端不可达/正在停止 |
| 400 | `invalid_request` | body 非法 / 缺 `model` 字段 |
| 401 | `authentication_error` | API key 缺失/错误 |

## API

### API-key 认证

配置 `server.api_key`（字符串或列表）或环境变量 `LLM_GATEWAY_API_KEY`
（逗号分隔多 key）后，所有 `/v1/*` 与 `/gateway/*` 数据接口要求
`Authorization: Bearer <key>`（缺失/错误 → 401 `authentication_error`）。
免认证例外：`/docs`、`/redoc`、`/openapi.json`、`/gateway/dashboard`（静态壳；
页面的数据请求 401 时自动弹窗要 key）。**未配置 key = 完全开放**。

### OpenAI 兼容（客户端入口）

- 任意 `/v1/*` POST（chat/completions、completions、responses、embeddings...）：
  按 body 的 `model` 字段门控后原样转发（含 `stream=true` SSE；自动注入
  `stream_options.include_usage` 以便真实 token 记账）。
- `GET /v1/models`：返回 registry 全部模型（后端停止时也可列出）。

### 管理接口

- `GET /gateway/status`：状态机 + 活跃请求 + `memory`（nvml_available /
  vram_* / ram_*，无 NVML 时 vram 为 null）。
- `GET /gateway/models`：registry + 加载状态。
- `POST /gateway/stop`：**正常关闭**——空闲时（任意状态）接受并清理，
  包括取消进行中的启动/切换；正在服务时 503。
- `POST /gateway/force-stop`：**强制关闭**——无条件清理（切断活跃流式连接）。
- `POST /gateway/preload/{model}`：提前暖模型。
- `GET /gateway/usage` + `GET /gateway/dashboard`：token 用量统计与看板
  （SQLite 落盘；普通响应解析 body usage，流式解析 SSE 尾帧 usage；
  记录失败绝不影响请求）。

## 配置

复制 `config.example.yaml` 为 `config.yaml`。环境变量（旧 `SGLANG_MANAGER_*`
名称仍兼容回退）：`LLM_GATEWAY_CONFIG`、`LLM_GATEWAY_IDLE_TIMEOUT_SECONDS`、
`LLM_GATEWAY_HOST`、`LLM_GATEWAY_PORT`、`LLM_GATEWAY_API_KEY`。
旧配置段名 `sglang:` / `gpu:` 仍作为 `backend:` / `memory:` 的别名接受（告警）。

## Windows 兼容性

- 纯 Python 3.10+ / asyncio（Proactor 事件循环原生支持子进程）；无 Unix 专属依赖
  （无 signal/fcntl/resource）；`psutil` 跨平台读内存。
- 进程停止：Windows 上 `terminate()` 即硬终止，`stop_timeout` → `kill()` 流程同样有效。
- `command_template` 可直接写 `.exe`（如 `C:\llama.cpp\llama-server.exe`）；
  含空格路径用引号包住（模板自动 shell-quote 占位符）。
- NVML 缺失（Intel CPU/NPU 机器）→ 显存检查自动跳过，RAM 检查照常。
- 服务化：Linux 用 `deploy/llm-gateway.service`；Windows 用 NSSM 或
  `deploy/llm-gateway.bat` 示例。

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m llm_gateway --config config.yaml
# 或 .venv/bin/llm-gateway --config config.yaml
```

## 测试

无 GPU 环境用 fake memory/runner 验证状态机全链路：`.venv/bin/pytest -v`

## 目录

```text
llm_gateway/
├── __main__.py     CLI 入口
├── api.py          FastAPI：/v1 代理 + /gateway 管理 + usage/看板 + API-key 认证
├── config.py       YAML 配置（backend 模板 / memory 双检查 / 兼容别名）
├── dashboard.py    用量看板页面（内联 HTML/CSS/JS）
├── errors.py       typed 错误体系
├── memory.py       VRAM(NVML 可选) + RAM(psutil) 监控
├── manager.py      状态机 + 生命周期锁 + watchdog + 记账
├── proxy.py        OpenAI 兼容反向代理（含 SSE + usage 捕获）
├── runner.py       后端子进程：模板命令/health/停止
├── state.py        状态枚举
└── usage.py        SQLite token 用量记录与聚合
```
