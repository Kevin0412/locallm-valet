# AGENTS.md — locallm-valet 开发与运维约定

本文件是给 AI 代理（以及人类维护者）的操作守则。**凡与本文件冲突的操作，以本文件为准。**
这些规则是从真实事故中提炼的——违反它们曾导致数据丢失、看板指标消失、GPU 占用失控。

---

## 0. 最高原则：用项目已有的功能，不自己造轮子

本项目是**自托管生产服务**，绝大多数操作已有内置入口。**不要**为以下场景写临时脚本
（shell/Python 一次性文件）绕过项目：历史教训是 17 个临时 bench 脚本 + 12 个 vLLM
测试脚本绕过内置入口，直接导致 benchmark 数据被覆盖清空。

| 想做的事 | ✅ 用这个 | ❌ 不要这样 |
|---|---|---|
| 跑 benchmark | `POST /gateway/benchmark/run`（前端"开始运行"入口，`job.py`）| 自己写 shell 循环调 CLI `benchmark run` |
| 加载/切换模型 | `POST /gateway/preload/{model}`、`POST /gateway/stop`、`/force-stop` | 直接 `nohup` 起 SGLang/vLLM 进程 |
| 查模型/槽位状态 | `GET /gateway/status`、`GET /gateway/models` | 手敲 `nvidia-smi` + `ps` 推断 |
| 测速/探测 | CLI `benchmark probe`（走网关 base_url）| 写独立 vLLM 脚本起第二个服务 |
| 查看结果 | 看板 `/gateway/benchmark`、`/gateway/dashboard` | 直接 grep JSONL |

**为什么**：`job.py`（前端入口）每个模型写**独立 JSONL 文件**
`{model}_{dataset}_results.jsonl`，天生不会互相覆盖；CLI `benchmark run` 的
`render_report` 曾用 `to_jsonl` **覆盖写单文件**，逐模型跑会把之前模型的记录清空
（已修复为合并写，但入口仍以 job.py 为准）。

---

## 1. Benchmark（评测）

### 正确入口
```bash
# 通过网关启动（前端页面同款）
curl -X POST http://127.0.0.1:8000/gateway/benchmark/run \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dataset":"mmlu","models":["qwen3.6-35b","qwen3.8-27b-fp8","gemma-4-31b-w4a16"],
       "concurrency":4,"max_tokens":64000,"enable_thinking":true}'
```
支持：多模型一次跑完、`sample`、`concurrency`、`enable_thinking`、暂停/恢复/停止
（`/gateway/benchmark/pause|resume|stop`）、每模型独立文件、进度查询。

### 统一标准
- **thinking 默认开**（`enable_thinking=true`）——所有模型在同一标准下对比
- **不做任何思考限制**（不截断 max_tokens、不加 reasoning budget）——要真实数据，
  模型想多久想多久；超长思考是模型特性，不是 bug
- `max_tokens` 全模型一致（64000）；`--timeout` 只是请求超时保护，非思考限制

### 数据集
- mmlu / mmlu_pro / bfcl / humaneval / mbpp 已有缓存（`dataset_cache/`），无需重新下载
- BFCL 是函数调用基准（agent 能力）；humaneval/mbpp 是代码基准（跑完自动评分）
- 同一批题目用 `sample_500_indices.json`（seed 42 分层采样），保证跨模型可比

### 结果文件
- 每模型独立文件：`benchmark_results/{model}_{dataset}_results.jsonl`（job.py 格式）
- 合并写逻辑：其他模型记录保留；同 (model, item_id) 用最新；HTTP 错误/空响应
  失败记录**绝不覆盖**已有有效答案（回归测试 `test_report_to_jsonl_merges_other_models`）

---

## 2. 模型 / 后端

### 注册表（config.yaml，**gitignored**——含 API key，绝不提交）
- `qwen3.6-35b`：SGLang，FP8 KV，`--enable-cache-report`
- `qwen3.8-27b-fp8`：SGLang，FP8 KV，`--enable-cache-report`
- `gemma-4-31b-w4a16`：**vLLM**（compressed-tensors W4A16），FP8 KV，
  `--tool-call-parser gemma4 --reasoning-parser gemma4 --enable-prompt-tokens-details`
  ——agent 调用部署，BFCL 99.2%

### 生命周期一律走网关
```bash
curl -X POST http://127.0.0.1:8000/gateway/preload/gemma-4-31b-w4a16 -H "Authorization: Bearer $API_KEY"
curl -X POST http://127.0.0.1:8000/gateway/stop        -H "Authorization: Bearer $API_KEY"
curl -X POST http://127.0.0.1:8000/gateway/force-stop  -H "Authorization: Bearer $API_KEY"
```
- `stop`：空闲时优雅卸载（busy 时 503）；`force-stop`：无条件释放（切断流式）
- **绝不在网关外直接起后端进程**——那会让看板显示 stopped 但 GPU 被占、无法管理

### GPU 占用纪律
- **峰时（约 06:00–22:00）不跑长任务**，GPU 空闲给用户
- 长 benchmark 只在**谷电（22:00–06:00）**跑
- 任务结束必须确认 GPU 释放（`nvidia-smi` 只剩桌面进程）

---

## 3. 数据与看板

### 看板
- `/gateway/benchmark`：评测结果表（准确率/分项/思考 tokens/缓存命中/prefill/decode）
- `/gateway/dashboard`：用量趋势（token/请求，UTC bucket 对齐，本地时间显示）

### 数据安全
- `benchmark_results/*.jsonl` 是**唯一结果来源**，不 commit（benchmark_results gitignored）
- 修改任何写 JSONL 的代码必须保证：合并而非覆盖、失败不覆盖有效、其他模型不丢
- 结果变更后用 `pytest tests/ -q` 验证（115 个测试，含数据持久化回归）

---

## 4. 环境事实（写死，别猜）

| 项 | 值 |
|---|---|
| 网关 | `http://127.0.0.1:8000`（systemd `locallm-valet`）|
| 后端端口 | SGLang 30000 / vLLM 按槽位 |
| API key | `config.yaml` 的 `server.api_key`（gitignored）|
| GPU | RTX 4090 48G（49140 MiB），驱动 595.84 |
| Python | 网关用系统 python3.12 + `.deps`；SGLang 在 `qwen3.5` conda env；vLLM 在 `vllm` conda env |
| 跑测试 | `PYTHONPATH=.deps:. /usr/bin/python3.12 -m pytest tests/ -q` |
| 测试 | 115 个（API/manager/usage/benchmark audit/runner）|
| transformers | vLLM env 的 gemma4 配置打过补丁（`global_head_dim` 兼容），脚本在 `scripts/patch_vllm_gemma4_transformers.py` |
| 网络 | GitHub 直连被墙，走 Clash 代理 `127.0.0.1:7897`；ModelScope 快 |

### 模型路径
- qwen3.6-35b：`/home/test/models/modelscope/models/Qwen/Qwen3___6-35B-A3B-FP8`
- qwen3.8-27b-fp8：`/home/test/models/modelscope/models/Qwen/Qwen3___8-27B-FP8`
- gemma w4a16：`/home/test/models/gemma-4-31B-it-qat-w4a16-ct/.../snapshots/master`
- GGUF 已退役删除（gemma 统一 vLLM）

---

## 5. 工作流纪律

1. **先查项目有没有现成入口**（grep `def ` / `@app.` / README），有就用
2. **不写一次性脚本**——如果必须（如探针），跑完即删，绝不留在 /tmp 复用
3. **改动后跑全量测试** `pytest tests/ -q`
4. **config.yaml 含密钥**：改动 gitignore 文件时说明"本地生效不提交"
5. **谷电纪律**：22:00 后启动长任务，05:45 前收尾（看板 6 点前更新）
6. **不凌晨让用户做决策**——能自主判断的（引擎选择、数据取舍）自己做，除非有不可逆破坏

---

## 6. 已知模型特性（实测）

| 模型 | 强项 | 注意 |
|---|---|---|
| qwen3.6-35b | 世界知识（mmlu 87.6%）| think 模式全开 |
| qwen3.8-27b-fp8 | **agent/工具调用**（bfcl 98.8%）| mmlu 仅 29.4%（定位不同）；思考上瘾（单题可达 2 万 tokens/45 分钟）|
| gemma-4-31b-w4a16 | 工具调用 bfcl 99.2%、humaneval 98.2% | vLLM 跑；think 数据在 reasoning+content 双字段 |

**27b 的思考上瘾**：少数难题会输出 1-2 万 reasoning tokens，单题拖 45 分钟——
这是模型特性，**不截断**（要真实数据）；安排任务时把 27b 排后面/单独留足时间。
