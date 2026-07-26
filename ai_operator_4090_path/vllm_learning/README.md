# vLLM Learning Lab：RTX 4090 单卡学习工程

这是一套独立于仓库其他项目的 vLLM 学习工程，目标环境是：

- NVIDIA RTX 4090 24GB
- Ubuntu 24.04
- Python 3.12
- 本机 CUDA Toolkit 12.6

默认模型是 `Qwen/Qwen2.5-1.5B-Instruct`。它的 BF16/FP16 权重约占 3GB，给 vLLM 执行开销和 KV cache 留有充足空间。可通过 `VLLM_MODEL` 换成其他 Hugging Face 模型 ID 或本地模型目录。

第一次接触 vLLM，建议先阅读
[《vLLM 入门学习指导：从离线推理到 OpenAI 服务》](docs/vllm_beginner_guide.md)，
再按本文的命令实际运行。

> vLLM/PyTorch 的预编译 wheel 会携带它自己的 CUDA 运行时；本工程为了严格匹配
> CUDA 12.6，默认固定到官方仍提供 cu126 wheel 的
> `vLLM 0.10.0+cu126`，并显式使用 PyTorch cu126 后端。这样既不依赖机器上
> `nvcc` 的补丁版本，也不会误装当前默认的 cu129/cu130 wheel。

## 目录

```text
vllm_learning/
├── data/prompts.jsonl              # 离线批量输入样例
├── docs/vllm_beginner_guide.md     # 配合代码阅读的入门指导
├── examples/
│   ├── 01_basic_inference.py       # 基础推理
│   ├── 02_offline_batch.py         # JSONL 离线批量推理
│   ├── 03_sampling_params.py       # greedy / balanced / creative
│   ├── 04_kv_cache_observe.py      # 模型加载及长短请求显存快照
│   └── 05_openai_client.py         # OpenAI Python 客户端请求
├── requirements/                   # GPU、客户端、开发依赖分组
├── scripts/                        # 环境检查、安装、运行、服务、验证
├── src/vllm_lab/                   # 无 GPU 也可导入和测试的公共代码
└── tests/                          # 最小单元测试
```

## 1. Ubuntu 服务器初始化

进入本目录后执行：

```bash
bash scripts/check_env.sh
bash scripts/setup_cuda.sh
source .venv/bin/activate
```

`setup_cuda.sh` 会创建全新的 Python 3.12 环境，避免现有 PyTorch/CUDA 包与 vLLM wheel 二进制不兼容。需要事先安装 `uv`；安装方式见 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

安装脚本使用 vLLM 官方 wheel 索引
`https://wheels.vllm.ai/0.10.0/cu126/` 和 PyTorch cu126 后端。vLLM 的旧版
[GPU 安装说明](https://docs.vllm.ai/en/v0.10.0/getting_started/installation/gpu.html)
记录了 cu126 构建方式；当前版
[GPU 安装说明](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
已以更新的 CUDA 运行时为默认值。

快速确认：

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash scripts/verify.sh
```

首次运行会从 Hugging Face 下载模型。若使用受限模型，先设置 `HF_TOKEN`；若模型已下载，可将 `VLLM_MODEL` 指向本地目录。

## 2. 配置

复制模板并按需修改：

```bash
cp .env.example .env
set -a
source .env
set +a
```

常用变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | 模型 ID 或本地路径 |
| `VLLM_DTYPE` | `auto` | 权重/计算数据类型 |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | 张量并行 GPU 数 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | 当前 vLLM 实例可使用的 GPU 显存比例 |
| `VLLM_MAX_MODEL_LEN` | `4096` | 最大上下文长度 |
| `VLLM_TRUST_REMOTE_CODE` | `false` | 是否允许模型仓库自定义代码 |
| `VLLM_SEED` | `42` | 随机种子 |

4090 单卡应保持 `VLLM_TENSOR_PARALLEL_SIZE=1`。`tensor_parallel_size=1` 就是非分布式兼容路径；张量并行的意义是将模型切到多张 GPU 上，不能让单张 4090 模拟 `2` 路并行。只有服务器确实有 N 张可见 GPU，且模型层/注意力头支持 N 路切分时，才设置为 N。

## 3. 学习顺序

### 3.1 基础推理

```bash
bash scripts/run_example.sh basic
```

学习 `LLM`、`SamplingParams`、prompt 到输出的最短路径。所有离线示例都使用相同的环境变量配置。

### 3.2 离线批量推理

```bash
bash scripts/run_example.sh batch
```

默认读取 `data/prompts.jsonl`，批量送入一次 `llm.generate()`，结果写入 `reports/offline_results.jsonl`。也可直接指定文件：

```bash
python examples/02_offline_batch.py \
  --input data/prompts.jsonl \
  --output reports/my_results.jsonl
```

每行输入格式：

```json
{"id": "question-1", "prompt": "用一句话解释 continuous batching。"}
```

### 3.3 采样参数

```bash
bash scripts/run_example.sh sampling
```

同一提示词依次比较：

- `temperature=0`：greedy，适合确定性输出。
- `temperature=0.7, top_p=0.9`：质量与多样性较均衡。
- `temperature=1.0, top_p=0.95, top_k=50`：更发散。

### 3.4 KV cache 与显存

```bash
watch -n 0.5 nvidia-smi
bash scripts/run_example.sh kv-cache
```

示例会记录四个时点的 `nvidia-smi` 快照：加载前、模型加载后、短请求后、长批量请求后。要注意：

1. vLLM 在初始化时会按 `gpu_memory_utilization` 预留执行器和 KV cache 空间，所以显存通常在“模型加载后”就明显上升。
2. 更长上下文和更高并发消耗更多 KV block，但预留式分配意味着 `nvidia-smi` 数字不一定随每次请求线性增长。
3. OOM 时先减小 `VLLM_MAX_MODEL_LEN` 或 `VLLM_GPU_MEMORY_UTILIZATION`；运行期频繁 preemption 时，可降低并发/批量 token 数，或在确有多卡时增加张量并行。

服务模式可观察 vLLM 自身指标：

```bash
curl -s http://127.0.0.1:8000/metrics \
  | grep -E 'vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_preemptions|num_requests_running)'
```

不同引擎版本可能使用 `kv_cache_usage_perc` 或旧名称
`gpu_cache_usage_perc`；值 `1` 表示 cache block 使用率 100%。指标定义见
[vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)。

## 4. OpenAI 兼容服务

终端 A：

```bash
source .venv/bin/activate
bash scripts/serve_openai.sh
```

启动脚本等价于以下核心命令，并额外读取本工程环境变量：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name vllm-lab \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --generation-config vllm
```

`--generation-config vllm` 用于避免模型仓库的 `generation_config.json` 静默覆盖本次请求的采样默认值。

终端 B 可选择 curl：

```bash
bash scripts/request_curl.sh
```

或 OpenAI Python 客户端：

```bash
python examples/05_openai_client.py
```

服务会提供 `/v1/models`、`/v1/completions`、`/v1/chat/completions` 等兼容接口；详情见 [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)。

## 5. 依赖说明

- `requirements/vllm.txt`：GPU 服务端核心依赖。固定为目标环境基线
  `vLLM 0.10.0+cu126`；这是官方索引中有明确 cu126 wheel 的版本。
- `requirements/client.txt`：OpenAI 兼容客户端；不需要 GPU。
- `requirements/dev.txt`：测试和静态检查；不需要 GPU。
- `pyproject.toml`：本地 `vllm_lab` 公共包和工具配置。

如果已升级 NVIDIA 驱动并希望试验新版本，可显式覆盖版本和 PyTorch 后端：

```bash
VLLM_SPEC='vllm==目标版本' \
VLLM_TORCH_BACKEND=auto \
bash scripts/setup_cuda.sh
```

这会重用 `.venv` 并替换 vLLM 版本。新版本是否仍提供 CUDA 12.6 wheel
需要逐版确认；不要只根据本机安装了 Toolkit 12.6 就假定 cu129/cu130 wheel
能够运行。生产环境建议在验证后把具体版本和 wheel 变体固定下来。

## 6. 验证清单

任何机器都能运行：

```bash
bash scripts/verify.sh
```

脚本执行 Python 语法编译、包导入、单元测试、shell 语法检查；安装了 Ruff 时还会执行静态检查。

RTX 4090 服务器完整验收：

```bash
source .venv/bin/activate
bash scripts/check_env.sh
bash scripts/verify.sh
bash scripts/run_example.sh basic
bash scripts/run_example.sh batch
bash scripts/run_example.sh sampling
bash scripts/run_example.sh kv-cache
```

然后启动服务并从另一个终端验证：

```bash
bash scripts/serve_openai.sh
# 另一个终端
curl -fsS http://127.0.0.1:8000/health
bash scripts/request_curl.sh
python examples/05_openai_client.py
curl -s http://127.0.0.1:8000/metrics \
  | grep -E 'vllm:(kv_cache_usage_perc|gpu_cache_usage_perc)'
```

若显存中已有其他进程，先用 `nvidia-smi` 确认剩余空间；不要直接把 `VLLM_GPU_MEMORY_UTILIZATION` 调到 `1.0`。
