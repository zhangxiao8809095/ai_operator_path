# vLLM 调试实验结果记录

> 本文档只记录实验事实，不预填或猜测 RTX 4090 结果。
> 运行实验后执行
> `bash scripts/run_experiment.sh render-results`，表格将由
> `reports/experiments/*.json` 重新生成。该目录默认不提交 Git。

结果解释必须同时保留模型、vLLM、PyTorch/CUDA、GPU、
引擎参数和负载参数；
缺少这些上下文时，不应跨机器比较绝对延迟或吞吐。

## 00_preflight：环境与版本分层检查

| 状态   | 结果文件                              | 下一步                  |
| ------ | ------------------------------------- | ----------------------- |
| 未运行 | reports/experiments/00_preflight.json | 先执行实验 00_preflight |

## 01_engine_lifecycle：引擎初始化、预热与稳态延迟

| 状态   | 结果文件                                     | 下一步                         |
| ------ | -------------------------------------------- | ------------------------------ |
| 未运行 | reports/experiments/01_engine_lifecycle.json | 先执行实验 01_engine_lifecycle |

## 02_prefill_decode_sweep：Prompt/Output 长度二维扫描

| 状态   | 结果文件                                         | 下一步                             |
| ------ | ------------------------------------------------ | ---------------------------------- |
| 未运行 | reports/experiments/02_prefill_decode_sweep.json | 先执行实验 02_prefill_decode_sweep |

## 03_offline_batching：离线列表批量与同步逐条调用对照

| 状态   | 结果文件                                     | 下一步                         |
| ------ | -------------------------------------------- | ------------------------------ |
| 未运行 | reports/experiments/03_offline_batching.json | 先执行实验 03_offline_batching |

## 04_sampling_diagnostics：采样参数、复现性与停止原因

| 状态   | 结果文件                                         | 下一步                             |
| ------ | ------------------------------------------------ | ---------------------------------- |
| 未运行 | reports/experiments/04_sampling_diagnostics.json | 先执行实验 04_sampling_diagnostics |

## 05_kv_pressure：KV 容量与负载压力递增

| 状态   | 结果文件                                | 下一步                    |
| ------ | --------------------------------------- | ------------------------- |
| 未运行 | reports/experiments/05_kv_pressure.json | 先执行实验 05_kv_pressure |

## 06_service_smoke：OpenAI 兼容服务分层验收

| 状态   | 结果文件                                  | 下一步                      |
| ------ | ----------------------------------------- | --------------------------- |
| 未运行 | reports/experiments/06_service_smoke.json | 先执行实验 06_service_smoke |

## 07_continuous_batching：错峰到达 Continuous Batching 与指标时间线

| 状态   | 结果文件                                        | 下一步                            |
| ------ | ----------------------------------------------- | --------------------------------- |
| 未运行 | reports/experiments/07_continuous_batching.json | 先执行实验 07_continuous_batching |

## 08_chunked_prefill：长 Prefill 注入与 Decode ITL 干扰

| 状态   | 结果文件                                      | 下一步                        |
| ------ | --------------------------------------------- | ----------------------------- |
| 未运行 | reports/experiments/08_chunked_prefill_*.json | 先执行实验 08_chunked_prefill |

## 09_prefix_caching：Automatic Prefix Caching 共享前缀对照

| 状态   | 结果文件                                   | 下一步                       |
| ------ | ------------------------------------------ | ---------------------------- |
| 未运行 | reports/experiments/09_prefix_caching.json | 先执行实验 09_prefix_caching |
