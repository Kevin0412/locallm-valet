# sglang-manager

面向**单 GPU、单活动模型**的 GPU-aware SGLang 生命周期代理。

> 根据 OpenAI API 的 `model` 字段按需启动和切换模型；SGLang 已运行且模型匹配时直接路由，未运行时仅在显存充足时启动，模型繁忙时拒绝切换，显存不足直接报错，并在连续空闲（默认 1 小时，**可配置**）无活跃请求后自动卸载模型释放 GPU。

客户端永远只访问一个固定地址：

```text
http://server:8000/v1
```

不直接访问 SGLang，现有 OpenAI 客户端零改动：

```python
client = OpenAI(base_url="http://server:8000/v1", api_key="...")
client.chat.completions.create(model="qwen3.6-35b", messages=[...])  # 自动按需启动
client.chat.completions.create(model="qwen3-coder-30b", messages=[...])  # 自动切换
```

## 架构

```text
Client
  │
  │ OpenAI-compatible API (固定地址 :8000)
  ▼
sglang-manager :8000
  │
  ├── 请求路由（按 model 字段）
  ├── 模型状态机 STOPPED/STARTING/RUNNING/STOPPING/SWITCHING
  ├── GPU 显存检查（NVML：required_vram + safety_margin）
  ├── SGLang 生命周期管理（启动/停止/切换，全局串行锁）
  ├── 健康检查轮询（不 sleep 硬等）
  ├── active_requests / last_activity 记账
  └── idle watchdog（可配置超时，默认 1 小时）
          │
          ▼
     SGLang 127.0.0.1:30000（按需拉起，由 manager 托管）
          │
          ▼
       GPU 48G（只管理自己的 SGLang，不碰其他 CUDA 进程）
```

## 核心规则（V1 边界）

1. **单卡单活动模型**：同一时间只加载一个模型，不做多模型显存拼装。
2. **请求按 `model` 决定行为**：
   - `RUNNING(qwen)` + `model=qwen` → 直接转发，**不重新检查显存**（已加载成功即为事实）。
   - `STOPPED` + `model=qwen` → 先查 NVML free VRAM，`free >= required_vram_gib + safety_margin_gib` 才启动，否则 `503 insufficient_gpu_memory`（绝不硬启动然后撞 CUDA OOM）。
3. **切换**：`RUNNING(qwen)` + `model=gemma` 且 `active_requests == 0` → 停 Qwen → 等进程退出 → **等显存真正释放（free 趋于稳定）** → 重新读 NVML free VRAM → 判断 Gemma → 启动 → 等 health → 转发。判断永远以「停止旧模型后重新读取的实际 free VRAM」为准。
4. **忙时不抢占**：`active_requests > 0` 时收到异模型请求 → `503 model_switch_busy`，绝不 kill 正在生成（含 `stream=true` 长连接）的请求。
5. **记账**：请求进入 `active_requests += 1`；**流式请求等 SSE 流完整关闭后才 `-= 1`** 并刷新 `last_activity`。
6. **空闲自动卸载**：`RUNNING` 且 `active_requests == 0` 且 `now - last_activity >= idle.timeout_seconds` → 停止 SGLang 释放显存。超时可配置（YAML `idle.timeout_seconds` 或环境变量 `SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS`），**不是硬编码**。期间收到同模型请求直接路由并重置计时。
7. **显存分 managed / external**：只读 NVML，只管理自己拉起的 SGLang，不杀其他 CUDA 程序。
8. **生命周期全局串行**：启动/停止/切换互斥（`_transition_lock`），并发请求不可能同时拉起多个 SGLang；但 `RUNNING` 同模型请求不经过该锁，直接转发。
9. **健康检查**：启动后轮询 `sglang.health_path`（默认 `/health`，可换更严格的 `/health_generate`），`startup_timeout_seconds`（默认 180s）内不 ready → `503 sglang_startup_timeout` / `sglang_startup_failed`，状态回 `STOPPED`。
10. **进程管理**：manager 常驻（systemd），SGLang 由 manager 按需 `start/stop/restart`，SGLang 本身不开机自启。

## 状态机

```text
STOPPED ──start(model)──▶ STARTING ──health ok──▶ RUNNING(model)
   ▲                        │                       │
   │◀──failed/timeout───────┘                       │ stop/无请求
   │                                                ▼
   └─────────────────────────────────────────── STOPPING ──▶ STOPPED
RUNNING(a) ──switch(b)──▶ SWITCHING(a→b) ──▶ RUNNING(b)  （失败回 STOPPED）
```

并发请求的行为：

| 当前状态 | 请求 model=当前启动中的模型 | 请求 model=其他模型 |
|---|---|---|
| STOPPED | 触发启动 | 触发启动（串行） |
| STARTING(m) | 等待启动完成，再转发 | `503 model_switch_busy` |
| RUNNING(m) | **直接转发**（不抢锁） | `active_requests>0` → busy；`==0` → 切换 |
| SWITCHING(a→b) | 等待切换完成 | `503 model_switch_busy` |
| STOPPING | `503 sglang_unavailable`（稍后重试） | 同左 |

## 错误类型

| HTTP | error.type | 含义 |
|---|---|---|
| 503 | `insufficient_gpu_memory` | free VRAM < required + margin |
| 503 | `model_switch_busy` | 模型忙/切换中，拒绝切换 |
| 404 | `model_not_found` | registry 里没有该模型 |
| 503 | `sglang_startup_failed` | 启动过程中进程退出 |
| 503 | `sglang_startup_timeout` | 超时未 ready |
| 503 | `sglang_unavailable` | SGLang 不可达/正在停止 |
| 503 | `gpu_unavailable` | NVML 不可用 |
| 400 | `invalid_request` | body 非法 / 缺 `model` 字段 |

```json
{
  "error": {
    "type": "insufficient_gpu_memory",
    "message": "cannot start model 'qwen3.6-35b': needs 46.0 GiB (42.0 required + 4.0 safety margin), only 10.2 GiB free on device 0",
    "code": "insufficient_gpu_memory"
  }
}
```

## API

### OpenAI 兼容（客户端入口）

- `POST /v1/chat/completions`、`POST /v1/completions`、`POST /v1/responses`、`POST /v1/embeddings`……任意 `/v1/*` POST：按 body 里 `model` 字段门控后**原样转发**（含 `stream=true` SSE）。
- `GET /v1/models`：返回 registry 全部模型（SGLang 停止时也可列出）。
- 其他 `/v1/*` GET：仅在已加载模型时转发。

### 管理接口

- `GET /gateway/status`：

```json
{
  "state": "running",
  "model": "qwen3.6-35b",
  "starting_model": null,
  "switch_from": null,
  "switch_to": null,
  "active_requests": 0,
  "idle_seconds": 952.0,
  "idle_timeout_seconds": 3600.0,
  "uptime_seconds": 1234.5,
  "gpu": { "available": true, "device": 0, "total_gib": 48.0, "free_gib": 14.2, "used_gib": 33.8 }
}
```

- `GET /gateway/models`：registry + 每个模型的加载状态。
- `POST /gateway/stop`：手动卸载（忙时 `503 model_switch_busy`）。
- `POST /gateway/preload/{model}`：提前暖模型，等 ready 后返回（用于预热场景）。

## 配置

复制 `config.example.yaml` 为 `config.yaml`：

```bash
cp config.example.yaml config.yaml
```

关键项：

| 配置 | 说明 |
|---|---|
| `sglang.command` | SGLang 启动命令基底，指向装有 SGLang 的 python（如 `/home/test/anaconda3/envs/qwen3.5/bin/python -m sglang.launch_server`） |
| `sglang.env` | 全局 env（如 `SGLANG_USE_MODELSCOPE=true`），自动附加 `CUDA_VISIBLE_DEVICES=<gpu.device>` |
| `models.<name>.path` | ModelScope ID 或本地目录，透传给 `--model-path` |
| `models.<name>.required_vram_gib` | 启动门槛的基准值（`+ gpu.safety_margin_gib`） |
| `models.<name>.sglang.*` | `mem_fraction_static` / `context_length` / `extra_args` / `env`（对应原启动脚本参数，如 `--reasoning-parser`、`--kv-cache-dtype fp8_e4m3`、`LD_PRELOAD`） |
| `idle.timeout_seconds` | 空闲自动卸载超时，**可配置**，默认 3600 |

环境变量覆盖：`SGLANG_MANAGER_CONFIG`（配置文件路径）、`SGLANG_MANAGER_IDLE_TIMEOUT_SECONDS`、`SGLANG_MANAGER_HOST`、`SGLANG_MANAGER_PORT`。

模型显存门槛示例（48G GPU）：

```text
模型预计需要   30G
安全余量        4G
────────────────
启动门槛       34G   →  free VRAM < 34G 直接 503，避免卡着显存启动
```

注意：`required_vram_gib` 要和 `mem_fraction_static × 总显存` 匹配，比如 `0.87 × 48 ≈ 42G`，则填 42，且 `42 + 4 = 46 < 48` 才可能启动（还要留出外部进程占用）。

## 运行

```bash
# 安装依赖（建议独立 venv；qwen3.5 环境留给 SGLang）
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 启动
.venv/bin/python -m sglang_manager --config config.yaml
# 或
.venv/bin/sglang-manager --config config.yaml
```

> 环境没有 venv/不可写时（如只读 home），可用 `pip install --target .deps ...` 把依赖装进项目目录，
> 再用 `PYTHONPATH=.deps:. python3 -m sglang_manager --config config.yaml` 启动。

systemd（`deploy/sglang-manager.service`）：

```bash
sudo cp deploy/sglang-manager.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now sglang-manager
```

SGLang 不需要 systemd 单元——manager 按需拉起/停止它。

## 测试

无 GPU 环境用 fake GPU/runner 验证状态机全链路：

```bash
.venv/bin/pytest -v
```

覆盖：按需启动、同模型直连不重启、显存不足 503、空闲切换、忙时拒绝切换、启动超时、并发同模型请求排队、进程意外退出、idle watchdog 卸载、SSE 流式代理计数等。

## 目录

```text
sglang_manager/
├── __main__.py     CLI 入口
├── api.py          FastAPI：/v1 代理 + /gateway 管理
├── config.py       YAML 配置加载/校验（idle 可配置）
├── errors.py       typed 错误体系
├── gpu.py          NVML 显存读取 + 释放等待
├── manager.py      状态机 + 生命周期锁 + watchdog + 记账
├── proxy.py        OpenAI 兼容反向代理（含 SSE）
├── runner.py       SGLang 子进程：启动/health/停止
└── state.py        状态枚举
```
