# vLLM 掌握路径：RTX 4090 调试实验手册

本文把学习目标转换为可重复实验。每个实验只改变明确的自变量，保留 JSON 原始证据，
再生成独立的[实验结果记录](vllm_experiment_results.md)。不要把本机语法检查当成 GPU
验收，也不要把尚未运行的预期现象写成实测结论。

默认基线是 RTX 4090 24GB、Python 3.12、PyTorch/vLLM cu126、
`Qwen/Qwen2.5-1.5B-Instruct`、`tensor_parallel_size=1`、
`max_model_len=4096` 和 `gpu_memory_utilization=0.85`。单卡实验必须保持 TP=1；
TP>1 需要真实多卡，不能在一张 4090 上模拟。

## 运行约定

在本目录完成安装并激活虚拟环境：

```bash
bash scripts/setup_cuda.sh
source .venv/bin/activate
bash scripts/verify.sh
```

离线实验 00–05 各自启动一个 Python 进程；在线实验 06–07 需要先在终端 A 执行
`bash scripts/serve_openai.sh`，再在终端 B 运行实验。每次改变模型或引擎参数后，应重启
服务或实验进程，避免把旧引擎状态混入新对照。

## 实验 00：环境与版本分层检查

| 项目         | 内容                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| 要回答的问题 | 当前解释器、vLLM wheel、PyTorch CUDA runtime、可见 GPU 和 TP 配置是否构成同一套可运行环境？             |
| 对应概念     | Python 环境、CUDA Toolkit 与 wheel runtime、Compute Capability、GPU 可见性、张量并行                    |
| 代码         | `experiments/00_preflight.py`                                                                           |
| 前置条件     | 已激活项目 `.venv`；4090 对当前进程可见                                                                 |
| 自变量       | 正常不改变；只有在非 4090 兼容机验证时才使用 `--allow-non-4090`                                         |
| 观察量       | Python/vLLM/PyTorch 版本、`torch.version.cuda`、GPU 名称、显存、Compute Capability、可见卡数与 TP       |
| 默认命令     | `bash scripts/run_experiment.sh preflight`                                                              |
| 调试顺序     | 先确认 `which python`；再确认 `import torch` 和 `import vllm`；再看 CUDA 可用性；最后检查 GPU 型号和 TP |
| 通过线       | Python 3.12、vLLM 0.10.0、PyTorch CUDA 12.6、CUDA 可用、至少一张 4090、可见卡数不小于 TP                |
| 停止条件     | 任一基础检查失败时不进入模型加载实验，先修复解释器或 wheel/runtime 组合                                 |
| 原始结果     | `reports/experiments/00_preflight.json`                                                                 |

## 实验 01：引擎初始化、预热与稳态延迟

| 项目         | 内容                                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| 要回答的问题 | 为什么第一次创建 `LLM`、第一次 `generate` 和后续相同请求的耗时不能混为一个指标？                                |
| 对应概念     | 模型权重加载、执行器初始化、KV cache 规划、CUDA kernel 预热、稳态推理                                           |
| 代码         | `experiments/01_engine_lifecycle.py`                                                                            |
| 控制变量     | 同一模型、prompt、采样参数、引擎参数和进程；显式关闭 APC；稳态阶段重复相同请求                                  |
| 自变量       | 生命周期阶段：初始化、首次生成、后续生成                                                                        |
| 观察量       | 初始化秒数、预热秒数、稳态延迟 min/mean/P50/P95/max、加载前后进程显存快照                                       |
| 默认命令     | `bash scripts/run_experiment.sh engine-lifecycle`                                                               |
| 预期但待验证 | 初始化通常最慢；首次生成可能含额外一次性开销；稳态重复值更适合做对照基线                                        |
| 调试顺序     | 初始化失败先看模型下载和显存；首轮异常看 kernel/graph 日志；稳态抖动再排查其他 GPU 进程和温度频率               |
| 通过线       | 生成成功，至少 5 个稳态样本写入报告，且初始化、预热、稳态采用不同字段记录                                       |
| 停止条件     | 初始化 OOM 时先确认空闲显存，再降低 `VLLM_GPU_MEMORY_UTILIZATION` 或 `VLLM_MAX_MODEL_LEN`，不要直接进入压力实验 |
| 原始结果     | `reports/experiments/01_engine_lifecycle.json`                                                                  |

## 实验 02：Prefill 与 Decode 负载二维扫描

| 项目         | 内容                                                                                          |
| ------------ | --------------------------------------------------------------------------------------------- |
| 要回答的问题 | 固定输出时增加 prompt，和固定 prompt 时增加输出 token，分别怎样改变离线总延迟？               |
| 对应概念     | Prefill、Decode、输入 token、输出 token、KV cache、同步离线总延迟                             |
| 代码         | `experiments/02_prefill_decode_sweep.py`                                                      |
| 控制变量     | 同一引擎、同一重复文本、greedy、`ignore_eos=True`、显式关闭 APC、每格相同重复次数             |
| 自变量       | prompt 目标长度 `128,512,2048` 与输出目标长度 `16,64,256` 的笛卡尔积                          |
| 观察量       | 实际输入 token、实际输出 token、每格总延迟及 P50/P95                                          |
| 默认命令     | `bash scripts/run_experiment.sh prefill-decode`                                               |
| 预期但待验证 | 输入增长主要增加 Prefill 工作；输出增长会增加串行 Decode 步数；两者都增加 KV 占用             |
| 口径边界     | 本实验测同步 `generate` 总延迟，不直接测 TTFT/TPOT；不能用总延迟除 token 后声称得到严格 TPOT  |
| 调试顺序     | 先核对实际 token 数；再确认输入加输出不超过 `max_model_len`；最后比较同一行或同一列的受控样本 |
| 通过线       | 九个长度组合均完成，每个组合保存重复样本，比较时只改变一个维度                                |
| 停止条件     | 某格超上下文上限时先调整扫描范围；不得默默截断后继续比较                                      |
| 原始结果     | `reports/experiments/02_prefill_decode_sweep.json`                                            |

## 实验 03：离线列表批量与同步逐条调用

| 项目         | 内容                                                                                  |
| ------------ | ------------------------------------------------------------------------------------- |
| 要回答的问题 | 相同请求集合一次传给 `llm.generate(prompts, ...)`，相较逐条同步调用能否提高整体吞吐？ |
| 对应概念     | Offline batching、调度机会、批处理吞吐、请求延迟、固定工作量比较                      |
| 代码         | `experiments/03_offline_batching.py`                                                  |
| 控制变量     | 同一已加载引擎、请求数、prompt 长度、输出上限、greedy、强制输出长度、显式关闭 APC     |
| 自变量       | 调用方式：列表批量与同步逐条；每次重复交替执行顺序以减弱顺序偏差                      |
| 观察量       | 完成固定请求集合的总秒数、实际输出 token、输出 tokens/s、重复样本的 P50               |
| 默认命令     | `bash scripts/run_experiment.sh offline-batching`                                     |
| 预期但待验证 | 列表批量给调度器更多同时可见请求，通常有更高吞吐，但绝对收益依赖形状和显存余量        |
| 口径边界     | 这里只比较离线调用方式，不等价于在线 continuous batching，也不代表单请求延迟一定更低  |
| 调试顺序     | 先核对两种模式完成相同请求数和输出 token；再看执行顺序；最后解释吞吐差异              |
| 通过线       | 两种模式各至少两个样本，固定工作量一致，报告能重算 tokens/s                           |
| 停止条件     | 若实际输出 token 不同，先修复 EOS/长度控制，不能直接比较吞吐                          |
| 原始结果     | `reports/experiments/03_offline_batching.json`                                        |

## 实验 04：采样、复现性与停止原因

| 项目         | 内容                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------ |
| 要回答的问题 | temperature、top_p、top_k、seed、EOS 与 max_tokens 如何共同决定候选集合、随机性和停止行为？      |
| 对应概念     | logits、temperature、概率重归一化、top-p、top-k、随机种子、finish reason                         |
| 代码         | `experiments/04_sampling_diagnostics.py`                                                         |
| 控制变量     | 同一模型、prompt、引擎 seed、最大输出长度；每组运行三次                                          |
| 自变量       | greedy、低温、balanced、top-k 限制、`ignore_eos=True` 强制短长度                                 |
| 观察量       | 每组唯一 token 序列数、token IDs、文本、输出 token 数、finish/stop reason                        |
| 默认命令     | `bash scripts/run_experiment.sh sampling`                                                        |
| 预期但待验证 | greedy 应确定；同 seed 的随机采样应可复现；改变过滤集合可改变输出；忽略 EOS 后通常以 length 停止 |
| 调试顺序     | 先比较 token IDs 而非只看文本；再看 seed 和采样参数；最后核对 finish reason 与 stop reason       |
| 通过线       | 五组设置均保存三次原始 token 序列，能够解释确定性、多样性与停止原因                              |
| 停止条件     | 若模型或版本无法接受某参数，记录兼容错误并先确认 vLLM 版本，不随意删掉该对照组                   |
| 原始结果     | `reports/experiments/04_sampling_diagnostics.json`                                               |

## 实验 05：KV 容量与负载压力递增

| 项目         | 内容                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| 要回答的问题 | batch 和上下文长度增加时，活跃 token、执行时间、进程显存与失败边界怎样变化？                                  |
| 对应概念     | KV block、活跃序列、容量预算、预留式显存、preemption、OOM、进程显存快照                                       |
| 代码         | `experiments/05_kv_pressure.py`                                                                               |
| 控制变量     | 同一引擎、相同 prompt 构造、固定输出 64 token、greedy、显式关闭 APC、逐级递增而非直接冲击上限                 |
| 自变量       | batch `1,8,16` 与 prompt `512,2048`；需要探索边界时再显式覆盖                                                 |
| 观察量       | 潜在活跃 token、实际输出 token、总耗时、加载前后显存、运行中 `nvidia-smi` 采样、异常类型与信息                |
| 默认命令     | `bash scripts/run_experiment.sh kv-pressure`                                                                  |
| 安全升级     | 先跑默认值；确认余量后逐步增加 batch；使用 `--continue-on-error` 时仍要保留首个失败样本                       |
| 关键边界     | `nvidia-smi` 是进程级显存，不是 vLLM 内部 KV block 使用率；内部压力应结合在线 `/metrics` 判断                 |
| 调试顺序     | 初始化 OOM 与运行期压力分开；先核对其他进程和配置，再看活跃 token；服务模式再查 waiting、KV usage、preemption |
| 通过线       | 默认六个场景有完整状态；成功和失败均留下结构化证据；结论不把进程显存等同于 KV 使用量                          |
| 停止条件     | 出现 CUDA OOM 或连续失败时停止增压并重启进程；不要在 CUDA 错误后的污染进程中继续取数                          |
| 原始结果     | `reports/experiments/05_kv_pressure.json`                                                                     |

## 实验 06：OpenAI 兼容服务分层验收

| 项目         | 内容                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------ |
| 要回答的问题 | 服务进程存活，是否就等于认证、模型路由、Completion、Chat 和 metrics 都正确？                     |
| 对应概念     | HTTP health、Bearer auth、served model name、Completion API、Chat template、Prometheus metrics   |
| 代码         | `experiments/06_service_smoke.py`                                                                |
| 前置条件     | 终端 A 执行 `bash scripts/serve_openai.sh` 并等待服务 ready                                      |
| 控制变量     | `VLLM_HOST`、`VLLM_PORT`、`VLLM_API_KEY`、`VLLM_SERVED_MODEL_NAME` 与服务端一致                  |
| 自变量       | 服务层级：health、models、completions、chat completions、metrics                                 |
| 观察量       | 每层通过状态、模型 ID、返回文本或异常类型、metrics 行数                                          |
| 默认命令     | `bash scripts/run_experiment.sh service-smoke`                                                   |
| 故障注入     | 可临时用错误 API key 验证 401；用错误 served model 验证路由失败；恢复正确变量后必须回归通过      |
| 调试顺序     | health → 认证 → `/v1/models` → Completion → Chat/template → metrics，按层定位而非笼统归因于 vLLM |
| 通过线       | 五层检查全部通过，且客户端使用的模型名来自 served model name 而非假定 Hugging Face ID            |
| 停止条件     | health 失败时先查看服务日志；不要继续把下游连接错误解释为采样或模型问题                          |
| 原始结果     | `reports/experiments/06_service_smoke.json`                                                      |

## 实验 07：错峰请求与 Continuous Batching 时间线

| 项目         | 内容                                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| 要回答的问题 | 长请求已经 Decode 时，稍后到达的短请求能否加入调度，并在长请求全部结束前收到首 token？                     |
| 对应概念     | Continuous batching、调度迭代、流式响应、TTFT、E2E、running/waiting、KV usage、preemption                  |
| 代码         | `experiments/07_continuous_batching.py`                                                                    |
| 前置条件     | 在线服务已通过实验 06；`/metrics` 可访问；客户端环境变量与服务一致                                         |
| 控制变量     | 同一服务进程、模型、prompt 类型、采样设置；长请求先到，短请求延迟 0.2 秒到达                               |
| 自变量       | 请求类型与输出长度：2 个长请求各 256 token，4 个短请求各 32 token                                          |
| 观察量       | arrival、first-token、finish 时间，TTFT、E2E、finish reason，定时采样的 running/waiting/KV/preemption 指标 |
| 默认命令     | `bash scripts/run_experiment.sh continuous-batching`                                                       |
| 动态加入证据 | 至少一个短请求的 first-token 时间早于最后一个长请求的 finish 时间；这是时间线证据，不单凭吞吐猜测          |
| 口径边界     | 客户端 TTFT 含网络与序列化；本地回环可减小但不能消除这部分；指标名会随 vLLM 版本演化                       |
| 调试顺序     | 先保证所有流完成；再看时间线；再关联 running/waiting；最后用 KV usage 与 preemption 解释压力               |
| 通过线       | 6 个请求均完成，有首 token 时间和 metrics 样本，并能明确判断是否存在动态加入证据                           |
| 停止条件     | 服务出现 OOM、持续超时或 preemption 激增时停止加压，保存服务日志和 JSON 后降低并发                         |
| 原始结果     | `reports/experiments/07_continuous_batching.json`                                                          |

## 实验 08：Chunked Prefill 的 TTFT/ITL 取舍

| 项目         | 内容                                                                                                                                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 要回答的问题 | 已有请求持续 Decode 时注入长 Prefill，Chunked Prefill 和每轮 token 预算怎样改变新请求 TTFT 与旧请求 ITL？                                                                                                  |
| 对应概念     | Chunked Prefill、`max_num_batched_tokens`、调度预算、TTFT、ITL、吞吐与延迟取舍                                                                                                                             |
| 代码         | `experiments/08_chunked_prefill.py`                                                                                                                                                                        |
| 控制变量     | 同一模型、GPU、长 Decode 请求、注入 prompt、输出长度和注入时刻；每个 profile 使用全新服务进程                                                                                                              |
| 自变量       | Profile A 显式关闭 Chunked Prefill；Profile B 开启并设置 `max_num_batched_tokens=512`                                                                                                                      |
| 观察量       | 服务端返回的实际 prompt token、两个请求 TTFT/E2E、注入前/中/后客户端流事件间隔 P50                                                                                                                         |
| Profile A    | 服务端：`VLLM_ENABLE_CHUNKED_PREFILL=false bash scripts/serve_openai.sh`；客户端：`VLLM_ENABLE_CHUNKED_PREFILL=false bash scripts/run_experiment.sh chunked-prefill --profile disabled`                    |
| Profile B    | 服务端：`VLLM_ENABLE_CHUNKED_PREFILL=true VLLM_MAX_NUM_BATCHED_TOKENS=512 bash scripts/serve_openai.sh`；客户端使用相同两个变量执行 `bash scripts/run_experiment.sh chunked-prefill --profile chunked-512` |
| 预期但待验证 | 切分长 Prefill 可能减小它对既有 Decode ITL 的尖峰，但新长 prompt 的 TTFT、整体吞吐会随预算变化，不存在统一“越小越好”                                                                                       |
| 口径边界     | 流式 chunk 事件间隔是客户端观察到的 ITL 近似值，包含回环网络和序列化；真正 kernel 时间需另用 profiler                                                                                                      |
| 调试顺序     | 先确认两份报告实际 prompt token 接近；再比较注入中 ITL；再比较长 prompt TTFT；最后讨论吞吐和尾延迟                                                                                                         |
| 通过线       | 两种服务配置各产生一份报告，配置字段不是 `not-recorded`，请求均完成且注入中至少有一个 ITL 样本                                                                                                             |
| 停止条件     | 长 prompt 超 `max_model_len`、请求超时或服务 OOM 时停止，先减小 `--prefill-repetitions`，不得用截断样本比较                                                                                                |
| 原始结果     | `reports/experiments/08_chunked_prefill_<profile>.json`                                                                                                                                                    |

## 实验 09：Automatic Prefix Caching 共享前缀对照

| 项目         | 内容                                                                                                  |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| 要回答的问题 | 启用 APC 后，相同长前缀加不同问题，是否比同长度但未见过的前缀减少重复 Prefill 工作？                  |
| 对应概念     | Prefix block hash、完整 block 命中、共享 KV、Prefill 复用、Decode 不可复用边界                        |
| 代码         | `experiments/09_prefix_caching.py`                                                                    |
| 控制变量     | 同一启用 APC 的引擎、相同目标前缀长度、问题长度、输出长度和 greedy；交替执行两种 case                 |
| 自变量       | `shared_cached` 使用已预热的共享长前缀；`unseen_control` 每轮使用同长度但首块不同的新前缀             |
| 观察量       | tokenizer 实际前缀 token、每轮离线总延迟、输出 token 和文本、两类样本 P50/P95                         |
| 默认命令     | `bash scripts/run_experiment.sh prefix-caching`                                                       |
| 预期但待验证 | 共享前缀的完整 KV blocks 可命中并减少重复 Prefill；末尾未满 block、不同前缀和新 Decode token 仍需计算 |
| 口径边界     | 离线总延迟只是 APC 证据之一，不是在线 TTFT；APC 不会直接让长答案的每个 Decode 步变快                  |
| 调试顺序     | 先核对两类实际 token 接近；再确认共享前缀已预热；再看命中/日志和延迟；最后检查输出正确性              |
| 通过线       | 两类各至少三个样本，控制前缀未被复用，共享组延迟差异能与 block 命中机制一致解释                       |
| 停止条件     | 如果共享前缀短于有效 block 或两组 token 差距过大，先修正负载，不能据此判断 APC 无效                   |
| 原始结果     | `reports/experiments/09_prefix_caching.json`                                                          |

## 结果固化与推荐顺序

建议按 `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09` 执行。
实验 08 必须重启两种服务配置；每完成一个实验，就重新生成结果文档：

```bash
bash scripts/run_experiment.sh render-results
```

提交或分享结果时同时保留服务启动命令、环境变量、服务日志和 JSON。不同模型、引擎版本、
`max_model_len` 或 `gpu_memory_utilization` 的结果应视为不同实验批次，不能覆盖后直接横向比较。
