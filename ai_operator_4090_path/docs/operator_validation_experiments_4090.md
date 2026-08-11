# RTX 4090 全算子验证与调试实验手册

本文用于在单张 RTX 4090 上验收当前工程中的全部前向算子。目标不是要求自定义实现超过 PyTorch 或 cuBLAS，而是建立完整证据链：

```text
输入契约 -> 正确性 -> 数值与边界 -> Sanitizer -> 可信计时
         -> NSYS 归层 -> NCU 定位 -> 因果结论 -> 回归测试
```

只有上一层通过后才进入下一层。正确性失败时禁止继续采集性能结论；Sanitizer 失败时禁止把对应 kernel 标记为验收通过。

## 1. 范围与实现清单

当前验收范围是单卡、前向推理、CUDA contiguous Tensor。除 WMMA 路径外，当前算子输入以 FP32 为主。

| 算子族    | 导出接口                                          | 当前独立路径                         |
| --------- | ------------------------------------------------- | ------------------------------------ |
| GEMM      | `gemm_naive`、`gemm_tiled`、`gemm_tiled_padding`  | 标量、shared tile、padding tile      |
| GEMM      | `gemm_regtile2x2`、`gemm_regtile4x4`              | 2x2、4x4 register tile               |
| GEMM      | `gemm_vectorized_float4`、`gemm_wmma_fp16`        | float4 与 WMMA/fallback              |
| Softmax   | `softmax_row`                                     | 单线程负责一行的朴素基线             |
| Softmax   | `softmax_block_reduce`                            | shared-memory 树形归约               |
| Softmax   | `softmax_warp_reduce`                             | warp shuffle 与 warp 间归约          |
| Softmax   | `softmax_online`                                  | Online Softmax 状态合并              |
| LayerNorm | `layernorm_row`、`layernorm_block_reduce`         | 单线程行基线、shared block reduction |
| LayerNorm | `layernorm_warp_reduce`、`layernorm_vectorized`   | warp reduction、float4/fallback      |
| RMSNorm   | `rmsnorm_row`、`rmsnorm_block_reduce`             | 单线程行基线、shared block reduction |
| RMSNorm   | `rmsnorm_warp_reduce`                             | warp reduction                       |
| RMSNorm   | `rmsnorm_vectorized`、`rmsnorm_vectorized_float4` | float2/fallback、float4/fallback     |
| Attention | `attention_naive`                                 | causal/non-causal 编译期特化         |
| Attention | `attention_causal_naive`                          | 固定 causal 特化                     |
| Attention | `attention_kv_cache_decode`                       | 单 token Query + KV cache            |
| Attention | `attention_tiled_online_softmax`                  | tiled key 遍历 + Online Softmax      |

本轮不验收 Autograd、多 GPU、NCCL、分布式并行和生产发布。

## 2. 每项实验的统一登记卡

每做一项实验，先复制并填写下面的登记卡。预测可以错误，但不能在看完结果后补写预测。为避免十列宽表难以阅读，后续每个算子族都拆成两张竖列对齐的表：第一张登记目标、对照、输入和指标，第二张用同一实验编号登记命令、预测、通过标准和产物；两张表合起来就是一张完整实验卡。

| 字段       | 填写内容                                  |
| ---------- | ----------------------------------------- |
| 实验编号   | 例如 `SM-P02`                             |
| 实验目标   | 本实验要回答的一个问题                    |
| 对照对象   | PyTorch reference 或两个 kernel 版本      |
| 输入矩阵   | shape、dtype、分布、alignment、seed       |
| 执行命令   | pytest、Benchmark、Sanitizer、NSYS 或 NCU |
| 正确性指标 | max abs/rel error、有限性、性质检查       |
| 性能指标   | 固定八指标；再按实验问题补充专项指标      |
| 运行前预测 | 预计变化、原因、可能代价                  |
| 通过标准   | 正确性和证据要求                          |
| 产物       | 日志、CSV、NSYS/NCU 报告、结论            |

统一错误统计至少记录：

```text
max_abs_error
max_rel_error
P99_abs_error（低精度或长归约时）
首个最大误差位置
NaN / +Inf / -Inf 数量
重复运行间最大差值
```

## 3. 分级闸门与服务器命令

### Gate 0：环境、代码与构建

从服务器上的工程目录开始，不要根据本机路径猜测服务器路径：

```bash
pwd
git rev-parse HEAD
git status --short
python scripts/verify_workspace.py
bash scripts/00_check_env.sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
bash scripts/00_check_env.sh --strict
bash scripts/10_build.sh
```

`--system-site-packages`用于复用租赁镜像中已经安装的CUDA版PyTorch。若镜像没有CUDA版PyTorch，必须先安装与服务器驱动匹配的官方CUDA wheel；主算子工程与`vllm_learning/`使用两个独立虚拟环境。只上传本子目录、没有Git元数据时，`git rev-parse`和`git status`记录为N/A，不应阻断构建。

首次上传完成后可用统一入口重新执行静态检查、环境检查、干净构建、24接口/sm_89检查和GPU测试：

```bash
bash scripts/15_verify_4090.sh smoke
bash scripts/15_verify_4090.sh full
```

`smoke`用于先确认24个接口能够在非默认stream执行；`full`再运行全部pytest。两者都不自动运行Sanitizer、NSYS或NCU，重型实验仍按后续Gate逐项执行。

导出接口检查：

```bash
python - <<'PY'
import aiop4090

names = [name for name in dir(aiop4090) if name.startswith((
    "gemm_", "softmax_", "layernorm_", "rmsnorm_", "attention_"
))]
print("\n".join(sorted(names)))
print("count:", len(names))
PY
```

期望：GPU 为 RTX 4090、Compute Capability 为 8.9、CUDA/PyTorch 可用、扩展构建成功、导出 24 个算子接口。

### Gate 1：基础正确性

```bash
python -m pytest -q tests
python -m pytest -q tests/test_gemm.py tests/test_operator_validation.py -k gemm
python -m pytest -q tests/test_softmax_norm.py tests/test_operator_validation.py -k softmax
python -m pytest -q tests/test_softmax_norm.py tests/test_operator_validation.py -k "layernorm or rmsnorm"
python -m pytest -q tests/test_attention.py tests/test_operator_validation.py -k attention
```

停止条件：任何失败、CUDA context 错误、偶发不一致或未解释的 NaN/Inf。

### Gate 2：Compute Sanitizer

先跑单个算子族，Sanitizer 很慢时不要直接使用 `all`：

```bash
bash scripts/40_sanitize.sh memcheck gemm
bash scripts/40_sanitize.sh memcheck softmax
bash scripts/40_sanitize.sh memcheck norm
bash scripts/40_sanitize.sh memcheck attention

bash scripts/40_sanitize.sh racecheck softmax
bash scripts/40_sanitize.sh racecheck norm
bash scripts/40_sanitize.sh racecheck attention

bash scripts/40_sanitize.sh initcheck gemm
bash scripts/40_sanitize.sh synccheck softmax
bash scripts/40_sanitize.sh synccheck norm
bash scripts/40_sanitize.sh synccheck attention
```

`40_sanitize.sh`使用`tests/test_sanitizer_smoke.py`中的专用小shape：覆盖24个正式接口以及Norm向量路径和tail fallback，但不重复大型性能shape与数百次稳定性循环。完整shape、数值和性质验证属于Gate 1；Sanitizer在这里专门回答越界、竞态、未初始化读取和同步问题。

验收要求：没有未解释的 invalid access、race hazard、uninitialized read 或 synchronization error。每个 warning 都要记录并判断，不能只看进程退出码。

### Gate 3：Benchmark

```bash
mkdir -p reports/benchmark
python benchmark/bench_ops.py --op gemm --csv reports/benchmark/gemm.csv
python benchmark/bench_ops.py --op softmax_norm --csv reports/benchmark/softmax_norm.csv
python benchmark/bench_ops.py --op attention --csv reports/benchmark/attention.csv
```

Benchmark 输出每个用例的 median、P90、min、max 和 spread，并把原始分轮汇总写入 CSV。若 P90 与 median 差距明显，先检查频率、温度、后台任务、同步和输入分配，再讨论 kernel 优化。

### Gate 4：NSYS 归层

全链路只在需要总览时运行；日常优先选择单个算子：

```bash
bash scripts/profile_nsys.sh gemm_tiled
bash scripts/profile_nsys.sh softmax_online
bash scripts/profile_nsys.sh layernorm_vectorized
bash scripts/profile_nsys.sh rmsnorm_vectorized_float4
bash scripts/profile_nsys.sh attention_tiled_online_softmax
bash scripts/profile_nsys.sh attention_kv_cache_decode
```

NSYS 先回答时间属于哪一层：Python/C++ 准备、dtype 转换、临时分配、拷贝、同步、launch gap 或目标 kernel。

### Gate 5：NCU

Speed-of-Light 先覆盖目标版本：

```bash
bash scripts/profile_ncu.sh gemm_naive
bash scripts/profile_ncu.sh gemm_tiled
bash scripts/profile_ncu.sh gemm_tiled_padding
bash scripts/profile_ncu.sh gemm_regtile2x2
bash scripts/profile_ncu.sh gemm_regtile4x4
bash scripts/profile_ncu.sh gemm_vectorized_float4
bash scripts/profile_ncu.sh gemm_wmma_fp16
bash scripts/profile_ncu.sh softmax_row
bash scripts/profile_ncu.sh softmax_block_reduce
bash scripts/profile_ncu.sh softmax_warp_reduce
bash scripts/profile_ncu.sh softmax_online
bash scripts/profile_ncu.sh layernorm_row
bash scripts/profile_ncu.sh layernorm_block_reduce
bash scripts/profile_ncu.sh layernorm_warp_reduce
bash scripts/profile_ncu.sh layernorm_vectorized
bash scripts/profile_ncu.sh rmsnorm_row
bash scripts/profile_ncu.sh rmsnorm_block_reduce
bash scripts/profile_ncu.sh rmsnorm_warp_reduce
bash scripts/profile_ncu.sh rmsnorm_vectorized
bash scripts/profile_ncu.sh rmsnorm_vectorized_float4
bash scripts/profile_ncu.sh attention_naive
bash scripts/profile_ncu.sh attention_causal_naive
bash scripts/profile_ncu.sh attention_kv_cache_decode
bash scripts/profile_ncu.sh attention_tiled_online_softmax
```

为填写所有版本的八指标总表，每个版本至少采集一次 `Full` 报告。必须先通过正确性和 Sanitizer；按算子族分批执行，前一族失败时停止后续重型采集：

```bash
OPS=(
  gemm_naive
  gemm_tiled
  gemm_tiled_padding
  gemm_regtile2x2
  gemm_regtile4x4
  gemm_vectorized_float4
  gemm_wmma_fp16
  softmax_row
  softmax_block_reduce
  softmax_warp_reduce
  softmax_online
  layernorm_row
  layernorm_block_reduce
  layernorm_warp_reduce
  layernorm_vectorized
  rmsnorm_row
  rmsnorm_block_reduce
  rmsnorm_warp_reduce
  rmsnorm_vectorized
  rmsnorm_vectorized_float4
  attention_naive
  attention_causal_naive
  attention_kv_cache_decode
  attention_tiled_online_softmax
)

for op in "${OPS[@]}"; do
  ITERS=1 bash scripts/profile_ncu_full.sh "$op"
done
```

固定八指标诊断顺序：

```text
Duration
-> Compute (SM) Throughput vs Memory Throughput
-> DRAM Throughput
-> L2 Cache Throughput
-> Achieved Occupancy
-> Registers / Thread
-> Top Stall Reason
-> 回到 Duration 和正确性
```

`Launch Stats` 和 `Roofline` 不属于固定八指标。只有小shape、短kernel或疑似launch-bound时补查 `Launch Stats`；只有需要判断算术强度与理论上限时补查 `Roofline`。

### Gate 5.1：统一八指标定义与提取

八指标是所有版本的第一轮性能体检，不代替正确性，也不直接等于根因。每个版本都必须先填完下面八项，再进入本算子章节的专项指标。`Memory`、`DRAM` 和 `L2 Cache` 除吞吐百分比外，还要结合绝对 bytes、requests 或 sectors 判断；`Top Stall Reason` 不能只写名称。

| 固定八指标              | 每个版本统一填写格式                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------- |
| Duration                | Benchmark median/P90/min/max；NCU kernel Duration；单位统一为 us 或 ms                 |
| Compute (SM) Throughput | 吞吐百分比、主要计算pipeline、是否接近计算侧上限                                       |
| Memory Throughput       | 吞吐百分比；L1/L2/shared的关键bytes、requests或sectors                                 |
| DRAM Throughput         | 吞吐百分比、DRAM read/write bytes；低DRAM时不得直接写“非访存瓶颈”                      |
| L2 Cache Throughput     | 吞吐百分比、L2 read/write bytes或sectors；区分缓存流量和显存流量                       |
| Achieved Occupancy      | Achieved/Theoretical Occupancy、Blocks/SM及限制资源                                    |
| Registers / Thread      | 每线程寄存器数、是否限制驻留block/warp、是否出现local load/store或spill                |
| Top Stall Reason        | Stall名称、cycles、占比及Estimated Speedup；结合Eligible/Issued Warps和Source/SASS定位 |

完成固定八指标后，可按具体问题补查 `Launch Stats`、`Roofline`、绝对流量、指令统计、shared bank conflict或源码/SASS证据。

各算子章节中的横向对比表是唯一结果登记位置：每个正式导出版本占一行，八个指标分别填写“原始数值 + 一句首轮判断”。报告路径统一为 `reports/ncu/<op>_full.ncu-rep`。

每份NCU Full报告可用现有脚本提取这八项；`kernel_name` 使用报告中显示的实际kernel名称，默认读取第6次调用（前5次为warmup）：

```bash
bash scripts/extract_ncu_metrics.sh \
  reports/ncu/<op>_full.ncu-rep \
  <kernel_name> \
  6
```

为了避免逐份打开NCU UI或手工填写`kernel_name`，优先使用批量Python工具。它会按报告文件名自动匹配实际kernel，提取固定八指标和各算子专项指标，并输出可直接筛选或回填的横向CSV与Markdown：

```bash
# 先做不依赖GPU的脚本自检，并检查报告到kernel的映射
python scripts/extract_ncu_results.py --self-test
python scripts/extract_ncu_results.py --list-mappings

# 先预览本次会处理哪些报告，不调用ncu
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family all \
  --dry-run

# 一次提取目录内所有已知Full报告
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family all \
  --output-dir reports/ncu_summary \
  --strict
```

NCU入口默认执行5次warmup和1次正式调用，因此提取器默认读取第6次kernel调用。显式设置`ITERS=<N>`只增加正式调用次数，第一个正式调用仍是第6次；只有修改`benchmark/profile_entry.py`中的warmup次数时才需要同步修改`--invocation <N>`。NCU不在默认路径时使用`--ncu-bin /path/to/ncu`或设置`NCU_BIN`。生成结果中的`N/A`表示报告未包含该metric，或该证据只能由NSYS、源码或SASS确认，不能按测量值`0`处理。旧的两个Shell脚本仍保留，用于排查某一份报告的kernel匹配或原始指标。

## 4. GEMM 实验与指标

GEMM 在本轮作为普通算子验收，不要求重做此前的专项学习报告。

### GEMM版本演进表

下表描述代码结构的演进关系，不代表性能排名。GEMM在`tiled`之后分成padding、register tiling、float4和WMMA等不同优化方向，因此“直接对照”比文件中的排列顺序更重要。

| 顺序/分支 | 当前版本                 | 直接对照          | 相较前驱的核心优化                                                                | 预期收益与新增代价                                                              |
| --------- | ------------------------ | ----------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1 基线    | `gemm_naive`             | —                 | 每个线程计算一个C元素，直接循环读取global A/B。                                   | 结构最直观；同一A/B元素被不同线程重复读取，global流量大。                       |
| 2 主线    | `gemm_tiled`             | `gemm_naive`      | Block协作把16×16的A/B tile搬入shared memory，再在K维复用。                        | 减少global重复读取；增加shared容量、两次同步和尾部补零。                        |
| 3A 支线   | `gemm_tiled_padding`     | `gemm_tiled`      | shared数组第二维由16改为17，改变相邻行的bank映射。                                | 用于验证潜在bank conflict；当前访问模式未必受益，必须以冲突指标和Duration确认。 |
| 3B 主线   | `gemm_regtile2x2`        | `gemm_tiled`      | 每线程从1个输出扩展为相邻2×2输出，用4个累加器复用加载到寄存器的A/B值。            | 提高每次shared读取对应的FMA数量；shared占用和寄存器压力上升。                   |
| 4 主线    | `gemm_regtile4x4`        | `gemm_regtile2x2` | 每线程进一步计算4×4输出，用16个累加器增加数据复用和指令级并行。                   | 理论复用更高；更容易受Registers/Thread、Occupancy或spill限制。                  |
| 独立支线  | `gemm_vectorized_float4` | `gemm_naive`      | 每线程计算连续4列，并在满足条件时用float4读取B；不继承register-tile的shared路径。 | 减少load指令并利用连续访问；要求16字节对齐且N为4倍数，否则走标量尾部路径。      |
| 硬件支线  | `gemm_wmma_fp16`         | `gemm_tiled`      | 用16×16×16 WMMA fragment和Tensor Core执行FP16乘法、FP32累加及输出。               | 获得Tensor pipeline吞吐；引入FP32→FP16转换、低精度误差和非16倍数fallback。      |

### GEMM：7个版本的八指标横向对比表

下面每个正式导出版本占一行，八个固定指标按统一顺序横向排列。执行后将“待在4090填写”替换为“原始数值 + 一句首轮判断”；专项指标只在完成这8项之后补充。

| 算子版本                 | Duration     | Compute (SM) Throughput | Memory Throughput | DRAM Throughput | L2 Cache Throughput | Achieved Occupancy | Registers / Thread | Top Stall Reason |
| ------------------------ | ------------ | ----------------------- | ----------------- | --------------- | ------------------- | ------------------ | ------------------ | ---------------- |
| `gemm_naive`             | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `gemm_tiled`             | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `gemm_tiled_padding`     | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `gemm_regtile2x2`        | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `gemm_regtile4x4`        | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `gemm_vectorized_float4` | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `gemm_wmma_fp16`         | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |

### GEMM补充指标说明

固定八指标完成后，再用下面指标解释GEMM的数据复用、寄存器分块和Tensor Core路径。百分比指标必须和绝对工作量一起判断。

| GEMM补充指标          | 含义与统一记录方法                                                            |
| --------------------- | ----------------------------------------------------------------------------- |
| Achieved TFLOP/s      | 按实际执行shape和Duration计算；同时注明FP32、FP16或Tensor Core路径            |
| L2 Absolute Traffic   | 记录L2 read/write bytes或sectors，用于判断数据复用是否真的减少了缓存流量      |
| Shared Bank Conflicts | 记录冲突次数或相关比率，并对照padding前后的shared访问方式                     |
| Register Spill        | 记录local load/store或spill bytes，判断寄存器分块是否把数据溢出到local memory |
| FP32/Tensor Pipeline  | 记录主要pipeline利用率，区分普通FMA路径和Tensor Core路径                      |
| MMA Instructions      | 记录MMA指令数量；非WMMA版本填写`0`或`N/A + 原因`                              |

### GEMM补充指标结果表

每个版本占一行。填写“原始数值 + 一句版本差异判断”；确实不适用时填写`N/A + 原因`。

| 算子版本                 | Achieved TFLOP/s | L2 Absolute Traffic | Shared Bank Conflicts | Register Spill | FP32/Tensor Pipeline | MMA Instructions |
| ------------------------ | ---------------- | ------------------- | --------------------- | -------------- | -------------------- | ---------------- |
| `gemm_naive`             | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |
| `gemm_tiled`             | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |
| `gemm_tiled_padding`     | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |
| `gemm_regtile2x2`        | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |
| `gemm_regtile4x4`        | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |
| `gemm_vectorized_float4` | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |
| `gemm_wmma_fp16`         | 待在4090填写     | 待在4090填写        | 待在4090填写          | 待在4090填写   | 待在4090填写         | 待在4090填写     |

### GEMM NCU结果自动提取

完成本章产生的NCU Full报告后执行：

```bash
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family gemm \
  --output-dir reports/ncu_summary
```

| 自动产物                                    | 用途                                                                                         |
| ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `reports/ncu_summary/gemm_fixed8.csv`       | 每个正式版本及aligned、misaligned、tail、fallback场景的固定八指标横向表                      |
| `reports/ncu_summary/gemm_supplemental.csv` | L2/DRAM绝对流量、bank conflict、spill、pipeline、MMA指令，以及由shape和Duration计算的TFLOP/s |
| `reports/ncu_summary/gemm_summary.md`       | 固定八指标和GEMM补充指标的可读横向对照，适合直接整理结论                                     |

TFLOP/s按每个profile场景的实际`M/N/K`计算；WMMA fallback会记录实际的`tiled` kernel，不能误填为Tensor Core路径。

下面8项实验均可从项目根目录用统一入口执行：`bash scripts/run_gemm_experiment.sh <实验编号>`。脚本会创建所需的`reports/gemm`、`reports/benchmark`、`reports/ncu`和`reports/nsys`目录；正确性闸门失败时会立即停止，不继续采集性能报告。NCU默认不使用`sudo`，服务器限制性能计数器时显式设置`NCU_USE_SUDO=1`。

### GEMM-C01：验证全部普通FP32版本

| 字段           | 内容                                                                |
| -------------- | ------------------------------------------------------------------- |
| 实验编号       | `GEMM-C01`                                                          |
| 实验目标       | 验证6个普通FP32 GEMM在规则、非整除、极小和长K输入下都满足数学正确性 |
| 对照对象       | 6个FP32 kernel与`torch.matmul`                                      |
| 输入矩阵       | `(1,1,1)`、规则shape、非整除shape、长K；FP32随机值和固定seed        |
| 正确性指标     | 输出shape/dtype、max abs/rel error、首个最大误差位置、NaN/Inf数量   |
| 性能与因果指标 | N/A；正确性失败时禁止进入性能实验                                   |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-C01`                      |
| 运行前预测     | 分别预测各版本最可能出错的尾部shape，运行后保留预测与反例           |
| 通过标准       | 6个FP32版本在记录容差内；有限输入全部产生有限输出                   |
| 产物           | `reports/gemm/GEMM-C01.log`                                         |

### GEMM-C02：验证空维度契约

| 字段           | 内容                                                                  |
| -------------- | --------------------------------------------------------------------- |
| 实验编号       | `GEMM-C02`                                                            |
| 实验目标       | 验证`M/N/K=0`时的返回shape、零结果和launch前处理                      |
| 对照对象       | 7个GEMM版本与数学上的空矩阵或零矩阵结果                               |
| 输入矩阵       | `M=0`、`N=0`、`K=0`分别构造；WMMA使用FP16，其余版本使用FP32           |
| 正确性指标     | 输出shape、numel、`K=0`非空输出是否全零、异常类型和数量               |
| 性能与因果指标 | N/A；空输入不用于性能排名                                             |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-C02`                        |
| 运行前预测     | `M/N=0`直接返回空输出；`K=0`在接口层返回全零，不产生非法零grid launch |
| 通过标准       | 无invalid configuration；返回shape正确；`K=0`的非空结果全零           |
| 产物           | `reports/gemm/GEMM-C02.log`                                           |

### GEMM-C03：验证WMMA数值与fallback路径

| 字段           | 内容                                                                           |
| -------------- | ------------------------------------------------------------------------------ |
| 实验编号       | `GEMM-C03`                                                                     |
| 实验目标       | 验证WMMA低精度误差随K变化的趋势，并确认非16倍数shape走正确fallback             |
| 对照对象       | `gemm_wmma_fp16`与FP16输入的PyTorch矩阵乘结果                                  |
| 输入矩阵       | K=`16/64/256/1024/4096`；M/N/K为16倍数的WMMA shape；三组非16倍数fallback shape |
| 正确性指标     | max abs/rel error、P99 abs error、NaN/Inf数量、输出是否为FP32                  |
| 性能与因果指标 | NSYS中的类型转换、临时分配、fallback kernel名称；本实验不做版本性能排名        |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-C03`                                 |
| 运行前预测     | K增长可能放大累计误差；非16倍数shape不出现MMA路径                              |
| 通过标准       | 误差趋势可解释且在记录容差内；无意外NaN/Inf；fallback结果正确                  |
| 产物           | `reports/gemm/GEMM-C03.csv`                                                    |

### GEMM-D01：排除内存、初始化与同步错误

| 字段           | 内容                                                                                  |
| -------------- | ------------------------------------------------------------------------------------- |
| 实验编号       | `GEMM-D01`                                                                            |
| 实验目标       | 排除普通对拍可能无法暴露的越界、未初始化读取和同步问题                                |
| 对照对象       | tiled、regtile、float4和WMMA代表实现                                                  |
| 输入矩阵       | 从C01/C02中选择最小非整除、尾部、空维度和错位输入                                     |
| 正确性指标     | Sanitizer错误数量、首个错误位置、重复运行结果是否稳定                                 |
| 性能与因果指标 | memcheck/initcheck/synccheck对应的thread、访问地址、访问类型和同步位置；Duration为N/A |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-D01`                                        |
| 运行前预测     | 尾部判断、float4错位和shared尾部初始化最容易暴露地址或初始化问题                      |
| 通过标准       | 正式GEMM用例没有未解释的invalid access或uninitialized read                            |
| 产物           | `reports/gemm/GEMM-D01.log`                                                           |

### GEMM-P01：解释shared-memory演进

| 字段           | 内容                                                                                  |
| -------------- | ------------------------------------------------------------------------------------- |
| 实验编号       | `GEMM-P01`                                                                            |
| 实验目标       | 判断tiled是否减少global/L2重复流量，以及padding是否改善shared bank conflict           |
| 对照对象       | `gemm_naive`、`gemm_tiled`、`gemm_tiled_padding`                                      |
| 输入矩阵       | `512/1024/2048`方阵；相同dtype、seed、warmup和采样轮数                                |
| 正确性指标     | Profile前重新与`torch.matmul`对拍                                                     |
| 性能与因果指标 | 固定八指标，加DRAM/L2绝对bytes、requests/sectors、shared bank conflict、barrier stall |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-P01`                                        |
| 运行前预测     | tiled减少重复DRAM读取；padding只有在bank conflict确实下降时才可能带来收益             |
| 通过标准       | Duration变化能够由绝对流量和shared证据解释，不能只比较吞吐百分比                      |
| 产物           | `reports/benchmark/gemm.csv`及三个`reports/ncu/*_full.ncu-rep`                        |

### GEMM-P02：解释register tiling的资源交换

| 字段           | 内容                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------- |
| 实验编号       | `GEMM-P02`                                                                                     |
| 实验目标       | 判断2x2和4x4分块的复用收益是否超过寄存器占用与Occupancy损失                                    |
| 对照对象       | `gemm_tiled`、`gemm_regtile2x2`、`gemm_regtile4x4`                                             |
| 输入矩阵       | 与P01相同方阵，并补一组非整除shape                                                             |
| 正确性指标     | 三个版本在相同shape下均与reference一致                                                         |
| 性能与因果指标 | Registers / Thread、Achieved Occupancy、Blocks/SM、Eligible Warps、spill、shared指令、Duration |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-P02`                                                 |
| 运行前预测     | 更大register tile可能减少shared指令，但也可能增加寄存器并降低驻留warp                          |
| 通过标准       | 明确记录复用收益与资源代价，不能只写“4x4更快”                                                  |
| 产物           | `reports/gemm/GEMM-P02.md`及三个NCU报告                                                        |

### GEMM-P03：验证float4收益边界

| 字段           | 内容                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| 实验编号       | `GEMM-P03`                                                                 |
| 实验目标       | 验证16-byte对齐时的向量访问收益，以及错位和尾部shape的安全fallback         |
| 对照对象       | float4对齐快路径、首地址错位路径、`N%4!=0`尾部路径和`torch.matmul`         |
| 输入矩阵       | 对齐输入、首地址偏移4B的连续Tensor、`N%4!=0`                               |
| 正确性指标     | fallback与reference一致、首错位置、是否出现misaligned access               |
| 性能与因果指标 | vector load指令、requests/sectors、LG Throttle、L2/DRAM绝对bytes、Duration |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-P03`                             |
| 运行前预测     | 对齐快路径load指令更少；fallback必须安全，但不要求比标量路径更快           |
| 通过标准       | 错位输入不崩溃；结果正确；对齐路径出现向量指令证据                         |
| 产物           | 测试日志和`reports/ncu/gemm_vectorized_float4_full.ncu-rep`                |

### GEMM-P04：证明Tensor Core执行路径

| 字段           | 内容                                                                             |
| -------------- | -------------------------------------------------------------------------------- |
| 实验编号       | `GEMM-P04`                                                                       |
| 实验目标       | 证明规则shape实际执行MMA，并判断接口转换和临时分配是否抵消kernel收益             |
| 对照对象       | `gemm_wmma_fp16`、FP32 tiled和PyTorch                                            |
| 输入矩阵       | 16倍数FP16方阵；另采FP32输入和非16倍数fallback                                   |
| 正确性指标     | max abs/rel error、P99 abs error、输出dtype、NaN/Inf数量                         |
| 性能与因果指标 | Tensor pipeline、MMA指令、Achieved TFLOP/s、Duration；NSYS转换、分配和端到端时间 |
| 执行命令       | `bash scripts/run_gemm_experiment.sh GEMM-P04`                                   |
| 运行前预测     | 规则shape出现MMA；FP32输入的内部转换会降低端到端收益                             |
| 通过标准       | NCU出现MMA/Tensor证据；NSYS区分转换、分配和WMMA kernel；数值误差在容差内         |
| 产物           | WMMA NCU报告；原生FP16、FP32转FP16、非16倍数fallback三份NSYS报告                 |

GEMM 额外注意：吞吐率上升不代表总工作量减少；同时记录绝对 bytes、指令和 Duration。

## 5. Softmax 实验与指标

### Softmax版本演进表

Softmax形成较清晰的线性主线：先把“一线程一行”的串行工作改成Block协作，再减少归约同步，最后改变数值归约算法。表中的“输入遍历次数”按当前源码统计。

| 顺序   | 当前版本               | 直接对照               | 相较前驱的核心优化                                                    | 预期收益与新增代价                                                              |
| ------ | ---------------------- | ---------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1 基线 | `softmax_row`          | —                      | 一个线程串行处理一整行，依次完成max、sum和归一化写回。                | 实现简单且稳定；列数增大时单线程串行，输入共遍历3次。                           |
| 2      | `softmax_block_reduce` | `softmax_row`          | 一行交给一个Block，线程跨列并行；max和sum使用shared-memory树形归约。  | 长行并行度提高；每级归约都有shared访问和`__syncthreads()`。                     |
| 3      | `softmax_warp_reduce`  | `softmax_block_reduce` | 先在warp内用shuffle归约，只把各warp结果写入shared，再由首个warp合并。 | 减少shared指令和Block级同步；收益取决于列数、warp利用率和MIO/scoreboard stall。 |
| 4      | `softmax_online`       | `softmax_warp_reduce`  | 把max与指数和合并为在线状态`(m,l)`，归约时按新max重标定。             | 输入遍历由3次降为2次；增加exp、状态合并和寄存器工作量，不保证一定更快。         |

### Softmax：4个版本的八指标横向对比表

下面每个正式导出版本占一行，八个固定指标按统一顺序横向排列。执行后将“待在4090填写”替换为“原始数值 + 一句首轮判断”；专项指标只在完成这8项之后补充。

| 算子版本               | Duration     | Compute (SM) Throughput | Memory Throughput | DRAM Throughput | L2 Cache Throughput | Achieved Occupancy | Registers / Thread | Top Stall Reason |
| ---------------------- | ------------ | ----------------------- | ----------------- | --------------- | ------------------- | ------------------ | ------------------ | ---------------- |
| `softmax_row`          | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `softmax_block_reduce` | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `softmax_warp_reduce`  | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `softmax_online`       | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |

### Softmax补充指标说明

固定八指标完成后，再用下面指标解释不同Reduction和Online Softmax实现的实际工作量与同步代价。

| Softmax补充指标             | 含义与统一记录方法                                                       |
| --------------------------- | ------------------------------------------------------------------------ |
| Input Passes                | 依据代码和load证据记录输入被完整遍历的次数，区分理论遍历与编译后实际访问 |
| Load Bytes                  | 记录输入相关绝对load bytes，判断Online版本是否真的减少访存               |
| Exp/Arithmetic Instructions | 记录exp及主要算术指令量，观察Online状态合并增加的计算工作                |
| Shared Access               | 记录shared load/store指令或bytes，比较block与warp归约                    |
| Barrier Wait                | 记录barrier数量及等待相关stall，判断同步是否限制执行                     |
| MIO Throttle                | 记录对应stall占比或cycles，判断MIO指令队列压力                           |
| Short Scoreboard            | 记录对应stall占比或cycles，结合shared依赖分析短延迟等待                  |

### Softmax补充指标结果表

每个版本占一行。填写“原始数值 + 一句版本差异判断”；确实不适用时填写`N/A + 原因`。

| 算子版本               | Input Passes | Load Bytes   | Exp/Arithmetic Instructions | Shared Access | Barrier Wait | MIO Throttle | Short Scoreboard |
| ---------------------- | ------------ | ------------ | --------------------------- | ------------- | ------------ | ------------ | ---------------- |
| `softmax_row`          | 待在4090填写 | 待在4090填写 | 待在4090填写                | 待在4090填写  | 待在4090填写 | 待在4090填写 | 待在4090填写     |
| `softmax_block_reduce` | 待在4090填写 | 待在4090填写 | 待在4090填写                | 待在4090填写  | 待在4090填写 | 待在4090填写 | 待在4090填写     |
| `softmax_warp_reduce`  | 待在4090填写 | 待在4090填写 | 待在4090填写                | 待在4090填写  | 待在4090填写 | 待在4090填写 | 待在4090填写     |
| `softmax_online`       | 待在4090填写 | 待在4090填写 | 待在4090填写                | 待在4090填写  | 待在4090填写 | 待在4090填写 | 待在4090填写     |

### Softmax NCU结果自动提取

```bash
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family softmax \
  --output-dir reports/ncu_summary
```

| 自动产物                                       | 用途                                                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `reports/ncu_summary/softmax_fixed8.csv`       | row、block、warp、online及small场景的固定八指标横向表                                         |
| `reports/ncu_summary/softmax_supplemental.csv` | 输入遍历次数、绝对流量、MUFU/算术指令、shared、barrier、MIO和Short Scoreboard原始值           |
| `reports/ncu_summary/softmax_summary.md`       | 将固定八指标与Softmax补充指标按版本横向放在一起，便于判断Online是否用额外计算换取更少输入遍历 |

`Input Passes`来自当前kernel源码：row/block/warp为3次，online为2次；它是源码证据，不是NCU硬件计数器。小shape的端到端launch时间仍须查看NSYS，脚本会明确标为`N/A (requires NSYS)`。

下面6项实验均可从项目根目录执行`bash scripts/run_operator_experiment.sh <实验编号>`。正确性或Sanitizer闸门失败时停止，不继续对应的性能采集。

### SM-C01：验证列边界

| 字段           | 内容                                                     |
| -------------- | -------------------------------------------------------- |
| 实验编号       | `SM-C01`                                                 |
| 实验目标       | 验证列边界                                               |
| 对照对象       | 4个kernel与`torch.softmax`                               |
| 输入矩阵       | cols=`1/31/32/33/255/256/257/1024/1026/4096`，FP32随机值 |
| 正确性指标     | max abs/rel、行和误差、非负性、有限性                    |
| 性能与因果指标 | N/A                                                      |
| 执行命令       | `bash scripts/run_operator_experiment.sh SM-C01`         |
| 运行前预测     | 预测32、256附近最容易暴露分支和尾部问题                  |
| 通过标准       | 所有列数在容差内，行和接近1                              |
| 产物           | `reports/softmax/SM-C01.log`                             |

### SM-C02：验证数值稳定语义

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `SM-C02`                                         |
| 实验目标       | 验证数值稳定语义                                 |
| 对照对象       | 4个kernel与PyTorch                               |
| 输入矩阵       | 平移、大正负、混合极值、NaN、Inf                 |
| 正确性指标     | 平移不变性、有限输出比例、首个NaN、特殊值语义    |
| 性能与因果指标 | N/A                                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh SM-C02` |
| 运行前预测     | 减max后有限极值稳定；NaN/Inf跟随reference语义    |
| 通过标准       | 有限输入不意外NaN/Inf；特殊值逐项记录            |
| 产物           | `reports/softmax/SM-C02.log`                     |

### SM-D01：验证重复和同步

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `SM-D01`                                         |
| 实验目标       | 验证重复和同步                                   |
| 对照对象       | 相同输入的100次结果                              |
| 输入矩阵       | `(5,257)`及归约尾部shape                         |
| 正确性指标     | 运行间最大差值、失败次数                         |
| 性能与因果指标 | racecheck、synccheck                             |
| 执行命令       | `bash scripts/run_operator_experiment.sh SM-D01` |
| 运行前预测     | 同步缺陷会表现为偶发误差或工具告警               |
| 通过标准       | 100次bitwise稳定且无未解释工具告警               |
| 产物           | `reports/softmax/SM-D01.log`                     |

### SM-P01：比较Block与Warp

| 字段           | 内容                                                                |
| -------------- | ------------------------------------------------------------------- |
| 实验编号       | `SM-P01`                                                            |
| 实验目标       | 比较Block与Warp                                                     |
| 对照对象       | block_reduce、warp_reduce                                           |
| 输入矩阵       | rows=`4096/8192`，cols=`1024/4096`                                  |
| 正确性指标     | profile前重跑reference                                              |
| 性能与因果指标 | shared指令、barrier、Eligible Warps、MIO/Short Scoreboard、Duration |
| 执行命令       | `bash scripts/run_operator_experiment.sh SM-P01`                    |
| 运行前预测     | warp路径减少shared/barrier，但未必降低Duration                      |
| 通过标准       | 指标变化和源码差异一致                                              |
| 产物           | `reports/benchmark/softmax.csv`、两份NCU Full报告及指标摘录         |

### SM-P02：判断Online的取舍

| 字段           | 内容                                                                                 |
| -------------- | ------------------------------------------------------------------------------------ |
| 实验编号       | `SM-P02`                                                                             |
| 实验目标       | 判断Online的取舍                                                                     |
| 对照对象       | block、warp、online                                                                  |
| 输入矩阵       | 与SM-P01相同                                                                         |
| 正确性指标     | max abs/rel                                                                          |
| 性能与因果指标 | 输入遍历次数、load bytes、exp/算术、Registers / Thread、Achieved Occupancy、Duration |
| 执行命令       | `bash scripts/run_operator_experiment.sh SM-P02`                                     |
| 运行前预测     | 不预设online更快，先比较总load与exp工作量                                            |
| 通过标准       | 用工作量、资源和Duration共同下结论                                                   |
| 产物           | `reports/softmax/SM-P02-{block,warp,online}.md`及三份NCU报告                         |

### SM-P03：判断是否launch-bound

| 字段           | 内容                                                        |
| -------------- | ----------------------------------------------------------- |
| 实验编号       | `SM-P03`                                                    |
| 实验目标       | 判断是否launch-bound                                        |
| 对照对象       | row/block/warp/online                                       |
| 输入矩阵       | 小rows，cols=`1/31/32/33/257`                               |
| 正确性指标     | max abs/rel、行和                                           |
| 性能与因果指标 | Launch Stats、kernel latency、CPU launch gap、端到端latency |
| 执行命令       | `bash scripts/run_operator_experiment.sh SM-P03`            |
| 运行前预测     | 小shape的端到端时间可能由launch主导                         |
| 通过标准       | NSYS能区分CPU gap和kernel时间                               |
| 产物           | 四份NSYS报告                                                |

运行前预测示例：Warp版可能减少 shared 与 barrier，但如果工作量很小，Duration可能主要由launch决定；Online版减少统计遍历，但状态合并和exp工作增加，不能仅凭名称判断更快。

## 6. LayerNorm 实验与指标

### LayerNorm版本演进表

LayerNorm沿“串行行基线 → Block归约 → warp归约 → float4访存”的顺序演进。向量化版本只改变访存宽度，mean和variance仍然是两轮独立归约。

| 顺序   | 当前版本                 | 直接对照                 | 相较前驱的核心优化                                                         | 预期收益与新增代价                                                                |
| ------ | ------------------------ | ------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1 基线 | `layernorm_row`          | —                        | 一个线程串行计算一行的mean、variance和仿射输出。                           | 逻辑清楚；两次统计遍历和一次输出遍历都由单线程完成。                              |
| 2      | `layernorm_block_reduce` | `layernorm_row`          | 一行交给一个Block，线程并行累加sum/variance，并用shared树形归约。          | 长行并行度提高；两轮归约带来较多shared访问和barrier。                             |
| 3      | `layernorm_warp_reduce`  | `layernorm_block_reduce` | 用warp shuffle完成warp内求和，仅通过shared合并warp摘要。                   | 减少shared与同步开销；仍需分别完成mean和variance两轮归约。                        |
| 4      | `layernorm_vectorized`   | `layernorm_warp_reduce`  | 统计和仿射阶段改用float4读取/写回X、gamma、beta和Y，归约继续复用warp方案。 | 减少向量路径的访存指令；要求cols为4倍数且所有指针16字节对齐，否则回退到warp版本。 |

### LayerNorm：4个版本的八指标横向对比表

下面每个正式导出版本占一行，八个固定指标按统一顺序横向排列。执行后将“待在4090填写”替换为“原始数值 + 一句首轮判断”；专项指标只在完成这8项之后补充。

| 算子版本                 | Duration     | Compute (SM) Throughput | Memory Throughput | DRAM Throughput | L2 Cache Throughput | Achieved Occupancy | Registers / Thread | Top Stall Reason |
| ------------------------ | ------------ | ----------------------- | ----------------- | --------------- | ------------------- | ------------------ | ------------------ | ---------------- |
| `layernorm_row`          | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `layernorm_block_reduce` | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `layernorm_warp_reduce`  | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `layernorm_vectorized`   | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |

### LayerNorm补充指标说明

固定八指标完成后，再用下面指标解释均值/方差归约、同步和向量化访问的代价。

| LayerNorm补充指标         | 含义与统一记录方法                                              |
| ------------------------- | --------------------------------------------------------------- |
| Mean/Variance Reductions  | 记录均值和方差的Reduction次数、遍历次数及使用的归约层次         |
| Shared Access             | 记录shared load/store指令或bytes，比较row、block和warp实现      |
| Barrier Wait              | 记录barrier数量及等待相关stall，判断block归约的同步代价         |
| Absolute Read/Write Bytes | 记录输入、输出及中间结果的绝对读写bytes，避免只比较吞吐百分比   |
| Float4 Load/Store         | 记录向量load/store指令数量，并注明使用快路径还是fallback        |
| Requests/Sectors          | 记录关键内存层级的requests和sectors，判断向量化是否改善访问合并 |

### LayerNorm补充指标结果表

每个版本占一行。填写“原始数值 + 一句版本差异判断”；确实不适用时填写`N/A + 原因`。

| 算子版本                 | Mean/Variance Reductions | Shared Access | Barrier Wait | Absolute Read/Write Bytes | Float4 Load/Store | Requests/Sectors |
| ------------------------ | ------------------------ | ------------- | ------------ | ------------------------- | ----------------- | ---------------- |
| `layernorm_row`          | 待在4090填写             | 待在4090填写  | 待在4090填写 | 待在4090填写              | 待在4090填写      | 待在4090填写     |
| `layernorm_block_reduce` | 待在4090填写             | 待在4090填写  | 待在4090填写 | 待在4090填写              | 待在4090填写      | 待在4090填写     |
| `layernorm_warp_reduce`  | 待在4090填写             | 待在4090填写  | 待在4090填写 | 待在4090填写              | 待在4090填写      | 待在4090填写     |
| `layernorm_vectorized`   | 待在4090填写             | 待在4090填写  | 待在4090填写 | 待在4090填写              | 待在4090填写      | 待在4090填写     |

### LayerNorm NCU结果自动提取

```bash
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family layernorm \
  --output-dir reports/ncu_summary
```

| 自动产物                                         | 用途                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `reports/ncu_summary/layernorm_fixed8.csv`       | 4个正式版本以及vectorized aligned、misaligned和tail fallback场景的固定八指标          |
| `reports/ncu_summary/layernorm_supplemental.csv` | 两次归约的源码事实、shared/barrier、绝对流量、requests/sectors和实际向量/fallback路径 |
| `reports/ncu_summary/layernorm_summary.md`       | 将八指标与LayerNorm专项指标按场景横向对照，直接检查向量化收益是否来自访问合并         |

脚本根据报告名记录vectorized实际落到`float4 kernel`还是`warp fallback`；向量指令数量仍需结合NCU Source/SASS页确认，不能只凭profile场景名称推断硬件执行。

下面6项实验使用统一入口`bash scripts/run_operator_experiment.sh <实验编号>`，测试日志、NCU指标摘录和NSYS报告分别写入`reports/layernorm`、`reports/ncu`和`reports/nsys`。

### LN-C01：验证4个实现

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `LN-C01`                                         |
| 实验目标       | 验证4个实现                                      |
| 对照对象       | row、block、warp、vectorized与`torch.layer_norm` |
| 输入矩阵       | rows=`1/3/64`，规则和非整除cols，FP32随机值      |
| 正确性指标     | max abs/rel、shape、dtype、有限性                |
| 性能与因果指标 | N/A                                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh LN-C01` |
| 运行前预测     | 预测row最慢但可作为独立朴素基线                  |
| 通过标准       | 4个实现均在记录容差内                            |
| 产物           | `reports/layernorm/LN-C01.log`                   |

### LN-C02：验证方差和eps稳定性

| 字段           | 内容                                                                   |
| -------------- | ---------------------------------------------------------------------- |
| 实验编号       | `LN-C02`                                                               |
| 实验目标       | 验证方差和eps稳定性                                                    |
| 对照对象       | 4个实现与PyTorch                                                       |
| 输入矩阵       | 常量、极小方差、大偏置小波动使用eps=`1e-3/1e-5`；非恒定输入补测eps=`0` |
| 正确性指标     | 均值相关误差、max abs/rel、NaN/Inf、eps敏感性                          |
| 性能与因果指标 | N/A                                                                    |
| 执行命令       | `bash scripts/run_operator_experiment.sh LN-C02`                       |
| 运行前预测     | 大偏置小波动最容易放大两遍方差公式误差                                 |
| 通过标准       | 输出有限，误差随eps变化可解释                                          |
| 产物           | `reports/layernorm/LN-C02.csv`                                         |

### LN-C03：验证边界和fallback

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `LN-C03`                                         |
| 实验目标       | 验证边界和fallback                               |
| 对照对象       | vectorized与warp fallback                        |
| 输入矩阵       | cols=`1/31/32/33/255/256/257/1026`，首地址偏移4B |
| 正确性指标     | 尾部、首错位置、fallback一致性                   |
| 性能与因果指标 | 实际kernel路径                                   |
| 执行命令       | `bash scripts/run_operator_experiment.sh LN-C03` |
| 运行前预测     | 非4倍数和4B错位应走warp fallback                 |
| 通过标准       | 无错位访问，fallback与reference一致              |
| 产物           | `reports/layernorm/LN-C03.log`                   |

### LN-P01：比较Row与Block

| 字段           | 内容                                              |
| -------------- | ------------------------------------------------- |
| 实验编号       | `LN-P01`                                          |
| 实验目标       | 比较Row与Block                                    |
| 对照对象       | 单线程row、shared block                           |
| 输入矩阵       | `(4096,1024)`、`(8192,4096)`                      |
| 正确性指标     | 同shape结果一致                                   |
| 性能与因果指标 | Duration、shared用量、barrier stall、Memory bytes |
| 执行命令       | `bash scripts/run_operator_experiment.sh LN-P01`  |
| 运行前预测     | block并行归约更快，但增加shared和barrier          |
| 通过标准       | 用Duration和同步/流量共同解释                     |
| 产物           | `reports/benchmark/layernorm.csv`及两份NCU报告    |

### LN-P02：评价Warp归约

| 字段           | 内容                                                    |
| -------------- | ------------------------------------------------------- |
| 实验编号       | `LN-P02`                                                |
| 实验目标       | 评价Warp归约                                            |
| 对照对象       | block_reduce、warp_reduce                               |
| 输入矩阵       | 与LN-P01相同                                            |
| 正确性指标     | 同shape结果一致                                         |
| 性能与因果指标 | shared指令、Eligible Warps、Top Stall Reason、Duration  |
| 执行命令       | `bash scripts/run_operator_experiment.sh LN-P02`        |
| 运行前预测     | warp shuffle预计减少shared指令和barrier                 |
| 通过标准       | 结论同时引用Eligible Warps与Stall                       |
| 产物           | `reports/layernorm/LN-P02-{block,warp}.md`及两份NCU报告 |

### LN-P03：评价float4

| 字段           | 内容                                                              |
| -------------- | ----------------------------------------------------------------- |
| 实验编号       | `LN-P03`                                                          |
| 实验目标       | 评价float4                                                        |
| 对照对象       | vectorized对齐路径与warp fallback                                 |
| 输入矩阵       | 对齐cols=4096、错位地址、cols=4098                                |
| 正确性指标     | fallback正确                                                      |
| 性能与因果指标 | vector load/store、requests/sectors、Registers / Thread、Duration |
| 执行命令       | `bash scripts/run_operator_experiment.sh LN-P03`                  |
| 运行前预测     | 对齐时向量指令减少事务；错位只要求安全fallback                    |
| 通过标准       | 快路径有指令证据，fallback不牺牲正确性                            |
| 产物           | NCU、NSYS和测试日志                                               |

常量或近常量输入主要用于数值验收，不用于性能排名。

## 7. RMSNorm 实验与指标

### RMSNorm版本演进表

前三个版本是归约方式的线性演进；float2和float4是建立在warp归约上的两条向量宽度分支。`rmsnorm_vectorized_float4`不是在同一个kernel里由float2逐步加宽，而是独立kernel和独立fallback判断。

| 顺序/分支   | 当前版本                    | 直接对照               | 相较前驱的核心优化                                       | 预期收益与新增代价                                                                 |
| ----------- | --------------------------- | ---------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 1 基线      | `rmsnorm_row`               | —                      | 一个线程串行完成平方和归约及缩放写回，不计算mean和beta。 | 作为RMSNorm基线；长行仍受单线程串行限制。                                          |
| 2           | `rmsnorm_block_reduce`      | `rmsnorm_row`          | 一行交给一个Block，用shared树形归约并行计算平方和。      | 提高长行并行度；引入shared访问和逐级barrier。                                      |
| 3           | `rmsnorm_warp_reduce`       | `rmsnorm_block_reduce` | warp内使用shuffle，只在warp间通过shared合并平方和。      | 减少shared指令和同步；仍是标量load/store。                                         |
| 4A 向量支线 | `rmsnorm_vectorized`        | `rmsnorm_warp_reduce`  | 平方和及写回改用float2，归约结构保持warp方案。           | 减少访存指令；要求8字节对齐且cols为2倍数，否则回退到warp版本。                     |
| 4B 向量支线 | `rmsnorm_vectorized_float4` | `rmsnorm_vectorized`   | 把独立向量路径从float2加宽到float4，一次处理4个元素。    | 可能进一步减少指令；对16字节对齐和4倍数shape要求更严格，失败时同样回退到warp版本。 |

### RMSNorm：5个版本的八指标横向对比表

下面每个正式导出版本占一行，八个固定指标按统一顺序横向排列。执行后将“待在4090填写”替换为“原始数值 + 一句首轮判断”；专项指标只在完成这8项之后补充。

| 算子版本                    | Duration     | Compute (SM) Throughput | Memory Throughput | DRAM Throughput | L2 Cache Throughput | Achieved Occupancy | Registers / Thread | Top Stall Reason |
| --------------------------- | ------------ | ----------------------- | ----------------- | --------------- | ------------------- | ------------------ | ------------------ | ---------------- |
| `rmsnorm_row`               | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `rmsnorm_block_reduce`      | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `rmsnorm_warp_reduce`       | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `rmsnorm_vectorized`        | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `rmsnorm_vectorized_float4` | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |

### RMSNorm补充指标说明

固定八指标完成后，再用下面指标解释RMS归约层次和float2/float4向量化路径。

| RMSNorm补充指标           | 含义与统一记录方法                                                        |
| ------------------------- | ------------------------------------------------------------------------- |
| RMS Reduction Work        | 记录平方和Reduction次数、遍历次数及归约层次                               |
| Shared/Barrier            | 记录shared访问、barrier数量及等待相关stall，比较block与warp路径           |
| Float2/Float4 Width       | 记录实际向量指令宽度，并注明float2、float4快路径或fallback                |
| Requests/Sectors          | 记录关键内存层级的requests和sectors，判断不同向量宽度的访问合并效果       |
| Absolute Read/Write Bytes | 记录输入、权重和输出的绝对读写bytes，用于比较工作量而不是只比较吞吐百分比 |

### RMSNorm补充指标结果表

每个版本占一行。填写“原始数值 + 一句版本差异判断”；确实不适用时填写`N/A + 原因`。

| 算子版本                    | RMS Reduction Work | Shared/Barrier | Float2/Float4 Width | Requests/Sectors | Absolute Read/Write Bytes |
| --------------------------- | ------------------ | -------------- | ------------------- | ---------------- | ------------------------- |
| `rmsnorm_row`               | 待在4090填写       | 待在4090填写   | 待在4090填写        | 待在4090填写     | 待在4090填写              |
| `rmsnorm_block_reduce`      | 待在4090填写       | 待在4090填写   | 待在4090填写        | 待在4090填写     | 待在4090填写              |
| `rmsnorm_warp_reduce`       | 待在4090填写       | 待在4090填写   | 待在4090填写        | 待在4090填写     | 待在4090填写              |
| `rmsnorm_vectorized`        | 待在4090填写       | 待在4090填写   | 待在4090填写        | 待在4090填写     | 待在4090填写              |
| `rmsnorm_vectorized_float4` | 待在4090填写       | 待在4090填写   | 待在4090填写        | 待在4090填写     | 待在4090填写              |

### RMSNorm NCU结果自动提取

```bash
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family rmsnorm \
  --output-dir reports/ncu_summary
```

| 自动产物                                       | 用途                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `reports/ncu_summary/rmsnorm_fixed8.csv`       | 5个正式版本及float2-only、float4 fallback、misaligned场景的固定八指标                      |
| `reports/ncu_summary/rmsnorm_supplemental.csv` | 平方和归约源码事实、shared/barrier、float2/float4/fallback路径、requests/sectors和绝对流量 |
| `reports/ncu_summary/rmsnorm_summary.md`       | 将八指标与RMSNorm专项指标按场景横向比较，便于分离向量宽度收益与寄存器/Occupancy代价        |

`Float2/Float4 Width`记录的是代码实际选择的kernel路径；要证明编译后的向量指令宽度，仍需以Source/SASS证据为准。

下面6项实验使用统一入口`bash scripts/run_operator_experiment.sh <实验编号>`。Float2、Float4、float2-only和float4 fallback会生成可区分的Profiler报告。

### RMS-C01：验证5个实现

| 字段           | 内容                                              |
| -------------- | ------------------------------------------------- |
| 实验编号       | `RMS-C01`                                         |
| 实验目标       | 验证5个实现                                       |
| 对照对象       | row、block、warp、float2、float4与PyTorch公式     |
| 输入矩阵       | rows=`1/3/64`，边界cols，FP32随机值               |
| 正确性指标     | max abs/rel、shape、dtype、有限性                 |
| 性能与因果指标 | N/A                                               |
| 执行命令       | `bash scripts/run_operator_experiment.sh RMS-C01` |
| 运行前预测     | 预测结果一致，细小差异来自归约顺序                |
| 通过标准       | 5个实现均在记录容差内                             |
| 产物           | `reports/rmsnorm/RMS-C01.log`                     |

### RMS-C02：验证数值和eps

| 字段           | 内容                                              |
| -------------- | ------------------------------------------------- |
| 实验编号       | `RMS-C02`                                         |
| 实验目标       | 验证数值和eps                                     |
| 对照对象       | 5个实现与reference                                |
| 输入矩阵       | 零、常量、`1e10/1e-10`、极小RMS；多个eps          |
| 正确性指标     | 误差分布、NaN/Inf、eps敏感性                      |
| 性能与因果指标 | N/A                                               |
| 执行命令       | `bash scripts/run_operator_experiment.sh RMS-C02` |
| 运行前预测     | 零和极小输入受eps控制，`1e10`仍应有限             |
| 通过标准       | 有限输入保持有限，误差分布可解释                  |
| 产物           | `reports/rmsnorm/RMS-C02.csv`                     |

### RMS-C03：验证向量路径边界

| 字段           | 内容                                              |
| -------------- | ------------------------------------------------- |
| 实验编号       | `RMS-C03`                                         |
| 实验目标       | 验证向量路径边界                                  |
| 对照对象       | float2、float4及warp fallback                     |
| 输入矩阵       | 奇数、2倍数非4倍数、4倍数、首地址偏移4B           |
| 正确性指标     | fallback、尾部、首错位置                          |
| 性能与因果指标 | 实际选择的kernel路径                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh RMS-C03` |
| 运行前预测     | 258走float2；float4版本fallback；奇数都fallback   |
| 通过标准       | 所有路径正确且Sanitizer干净                       |
| 产物           | `reports/rmsnorm/RMS-C03.log`                     |

### RMS-P01：比较归约层次

| 字段           | 内容                                                           |
| -------------- | -------------------------------------------------------------- |
| 实验编号       | `RMS-P01`                                                      |
| 实验目标       | 比较归约层次                                                   |
| 对照对象       | row、block、warp                                               |
| 输入矩阵       | `(4096,1024)`、`(8192,4096)`                                   |
| 正确性指标     | 结果一致                                                       |
| 性能与因果指标 | shared、barrier、Top Stall Reason、Memory Throughput、Duration |
| 执行命令       | `bash scripts/run_operator_experiment.sh RMS-P01`              |
| 运行前预测     | 并行归约降低Duration，warp路径减少shared同步                   |
| 通过标准       | 用Duration、shared和Stall形成证据链                            |
| 产物           | `reports/benchmark/rmsnorm.csv`及三份NCU报告                   |

### RMS-P02：区分Float2和Float4

| 字段           | 内容                                                                  |
| -------------- | --------------------------------------------------------------------- |
| 实验编号       | `RMS-P02`                                                             |
| 实验目标       | 区分Float2和Float4                                                    |
| 对照对象       | 两个独立向量kernel                                                    |
| 输入矩阵       | 对齐cols=`1024/4096`；另用258确认float2-only                          |
| 正确性指标     | 结果一致                                                              |
| 性能与因果指标 | 向量指令宽度、requests/sectors、Registers / Thread、Duration          |
| 执行命令       | `bash scripts/run_operator_experiment.sh RMS-P02`                     |
| 运行前预测     | float4指令更宽但可能增加寄存器压力                                    |
| 通过标准       | NCU能区分两个kernel和向量宽度                                         |
| 产物           | Float2、Float4、float2-only和float4 fallback四份NCU报告及错位NSYS报告 |

### RMS-P03：理解与LayerNorm工作量

| 字段           | 内容                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| 实验编号       | `RMS-P03`                                                                                             |
| 实验目标       | 理解与LayerNorm工作量                                                                                 |
| 对照对象       | 同层次的RMSNorm和LayerNorm                                                                            |
| 输入矩阵       | 完全相同rows/cols和计时条件                                                                           |
| 正确性指标     | 数学输出不互比                                                                                        |
| 性能与因果指标 | Reduction次数、读写bytes、算术指令、Duration                                                          |
| 执行命令       | `bash scripts/run_operator_experiment.sh RMS-P03`                                                     |
| 运行前预测     | RMSNorm少均值相关工作，流量可能相近                                                                   |
| 通过标准       | 只报告工作量差异，不宣称数学等价                                                                      |
| 产物           | `reports/benchmark/norm.csv`、两份NCU报告及`reports/rmsnorm/RMS-P03-{layernorm-warp,rmsnorm-warp}.md` |

## 8. Attention 与 KV-cache 实验和指标

### Attention与KV-cache版本演进表

Attention不是四个版本首尾相接的单线优化：`attention_causal_naive`是接口特化，KV-cache是Decode工作负载分支，tiled online-softmax才是当前Prefill计算主线。比较性能时必须先确认两行是否解决相同shape和相同问题。

| 顺序/分支      | 当前版本                         | 直接对照                       | 相较前驱的核心优化                                                                                | 预期收益与新增代价                                                                               |
| -------------- | -------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1 通用基线     | `attention_naive`                | —                              | Host按causal参数分派到编译期`<true>/<false>`kernel；每个输出元素先求max，再重算QK以求sum和加权V。 | 不物化完整score矩阵且便于对拍；QK点积被计算两遍，计算量高。                                      |
| 2A 特化接口    | `attention_causal_naive`         | `attention_naive(causal=True)` | 直接调用固定`<true>`kernel，去掉Host侧causal分支。                                                | 验证causal专用接口；底层kernel相同，不能预期设备端Duration产生本质变化。                         |
| 2B Decode支线  | `attention_kv_cache_decode`      | `attention_naive`的prefill形状 | 把Query限制为长度1，只遍历`kv_len`范围内的K/V cache，Grid不再包含Query序列维。                    | 匹配逐token解码并避免计算无关cache位置；通常转为小kernel、launch和cache读取问题。                |
| 2C Prefill主线 | `attention_tiled_online_softmax` | `attention_naive`              | 按128个key分段遍历，用在线`(m,l,acc)`状态在一次QK遍历中同时合并max、sum和加权V。                  | 消除naive的第二次QK点积；增加在线重标定exp、三组shared状态和寄存器压力，causal判断位于kernel内。 |

### Attention与KV-cache：4个版本的八指标横向对比表

下面每个正式导出版本占一行，八个固定指标按统一顺序横向排列。执行后将“待在4090填写”替换为“原始数值 + 一句首轮判断”；专项指标只在完成这8项之后补充。

| 算子版本                         | Duration     | Compute (SM) Throughput | Memory Throughput | DRAM Throughput | L2 Cache Throughput | Achieved Occupancy | Registers / Thread | Top Stall Reason |
| -------------------------------- | ------------ | ----------------------- | ----------------- | --------------- | ------------------- | ------------------ | ------------------ | ---------------- |
| `attention_naive`                | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `attention_causal_naive`         | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `attention_kv_cache_decode`      | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |
| `attention_tiled_online_softmax` | 待在4090填写 | 待在4090填写            | 待在4090填写      | 待在4090填写    | 待在4090填写        | 待在4090填写       | 待在4090填写       | 待在4090填写     |

### Attention补充指标说明

固定八指标完成后，再用下面指标解释Prefill、Decode、Online Softmax和causal特化的差异。

| Attention补充指标            | 含义与统一记录方法                                                  |
| ---------------------------- | ------------------------------------------------------------------- |
| Q/K/V and Cache Bytes        | 记录Q/K/V或KV cache的绝对读取bytes，并注明Prefill或Decode形状       |
| L2 Hit Rate/Absolute Traffic | 同时记录L2命中率和绝对流量，避免把高命中率直接等同于低访存成本      |
| Score Intermediate           | 记录是否物化score中间量及其bytes；需要代码和Profiler证据共同确认    |
| Register Spill               | 记录local load/store或spill bytes，判断Online状态是否造成寄存器压力 |
| Long Scoreboard              | 记录对应stall的cycles或占比，判断长延迟memory dependency            |
| Causal Branch Efficiency     | 记录分支指令、有效线程比例或分支效率，解释causal特化收益            |
| Duration vs S/kv_len         | 固定其他维度，记录Duration随序列长度或`kv_len`变化的曲线和拐点      |

### Attention补充指标结果表

每个版本占一行。填写“原始数值 + 一句版本差异判断”；确实不适用时填写`N/A + 原因`。

| 算子版本                         | Q/K/V and Cache Bytes | L2 Hit Rate/Absolute Traffic | Score Intermediate | Register Spill | Long Scoreboard | Causal Branch Efficiency | Duration vs S/kv_len |
| -------------------------------- | --------------------- | ---------------------------- | ------------------ | -------------- | --------------- | ------------------------ | -------------------- |
| `attention_naive`                | 待在4090填写          | 待在4090填写                 | 待在4090填写       | 待在4090填写   | 待在4090填写    | 待在4090填写             | 待在4090填写         |
| `attention_causal_naive`         | 待在4090填写          | 待在4090填写                 | 待在4090填写       | 待在4090填写   | 待在4090填写    | 待在4090填写             | 待在4090填写         |
| `attention_kv_cache_decode`      | 待在4090填写          | 待在4090填写                 | 待在4090填写       | 待在4090填写   | 待在4090填写    | 待在4090填写             | 待在4090填写         |
| `attention_tiled_online_softmax` | 待在4090填写          | 待在4090填写                 | 待在4090填写       | 待在4090填写   | 待在4090填写    | 待在4090填写             | 待在4090填写         |

### Attention与KV-cache NCU结果自动提取

```bash
python scripts/extract_ncu_results.py \
  --report-dir reports/ncu \
  --family attention \
  --output-dir reports/ncu_summary
```

| 自动产物                                         | 用途                                                                                    |
| ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `reports/ncu_summary/attention_fixed8.csv`       | 4个正式版本及causal/non-causal、S=64/128、kv_len场景的固定八指标                        |
| `reports/ncu_summary/attention_supplemental.csv` | L2/DRAM流量、L2 hit rate、spill、Long Scoreboard、分支指标、score物化源码证据和Duration |
| `reports/ncu_summary/attention_summary.md`       | 将不同S和`kv_len`保留为独立结果行，可直接整理Prefill/Decode的Duration变化               |

NCU只能给出kernel总流量，不能自动把bytes精确拆成Q/K/V/cache各张量；该列必须同时写shape并结合源码。`Score Intermediate`由源码确认是否物化，`Duration vs S/kv_len`则需要至少两份不同shape报告后再比较。

下面8项实验使用统一入口`bash scripts/run_operator_experiment.sh <实验编号>`。Prefill、Decode、causal/non-causal均使用独立的profile场景和报告名称。

### AT-C01：验证Naive非因果

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `AT-C01`                                         |
| 实验目标       | 验证Naive非因果                                  |
| 对照对象       | `attention_naive(false)`与显式reference          |
| 输入矩阵       | 多组`B/H/S/D`，含D=5等非整除值                   |
| 正确性指标     | max abs/rel、shape、NaN/Inf                      |
| 性能与因果指标 | N/A                                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-C01` |
| 运行前预测     | 非整除D不应影响正确性，只改变循环尾部            |
| 通过标准       | 所有shape在容差内                                |
| 产物           | `reports/attention/AT-C01.log`                   |

### AT-C02：验证Causal mask

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `AT-C02`                                         |
| 实验目标       | 验证Causal mask                                  |
| 对照对象       | naive causal、固定causal接口、PyTorch reference  |
| 输入矩阵       | 短序列、首尾query、D非整除                       |
| 正确性指标     | 首错query、max abs/rel、mask边界                 |
| 性能与因果指标 | 编译期causal路径                                 |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-C02` |
| 运行前预测     | 编译期causal消除路径选择，但mask工作仍存在       |
| 通过标准       | query 0只能看key 0；末query可见全部历史          |
| 产物           | `reports/attention/AT-C02.log`                   |

### AT-C03：验证KV Decode边界

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `AT-C03`                                         |
| 实验目标       | 验证KV Decode边界                                |
| 对照对象       | decode kernel与截断后的K/V reference             |
| 输入矩阵       | `kv_len=1/中间/最大/0/越界`                      |
| 正确性指标     | 输出误差、错误消息、不同长度稳定性               |
| 性能与因果指标 | N/A                                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-C03` |
| 运行前预测     | Duration随kv_len近似增长；非法值在launch前失败   |
| 通过标准       | 合法值正确，0和越界明确失败                      |
| 产物           | `reports/attention/AT-C03.csv`                   |

### AT-C04：验证空和错误shape

| 字段           | 内容                                             |
| -------------- | ------------------------------------------------ |
| 实验编号       | `AT-C04`                                         |
| 实验目标       | 验证空和错误shape                                |
| 对照对象       | 4个接口的契约                                    |
| 输入矩阵       | `B/H=0`、`S/D=0`、维度/shape/dtype/stride错误    |
| 正确性指标     | 返回shape或异常类型和消息                        |
| 性能与因果指标 | N/A                                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-C04` |
| 运行前预测     | B/H为空直接返回；S/D为空因Reduction无定义而拒绝  |
| 通过标准       | 无非法grid，异常稳定且可读                       |
| 产物           | `reports/attention/AT-C04.log`                   |

### AT-P01：比较Naive与Online

| 字段           | 内容                                                                         |
| -------------- | ---------------------------------------------------------------------------- |
| 实验编号       | `AT-P01`                                                                     |
| 实验目标       | 比较Naive与Online                                                            |
| 对照对象       | naive、tiled online                                                          |
| 输入矩阵       | causal/non-causal相同shape                                                   |
| 正确性指标     | max abs/rel                                                                  |
| 性能与因果指标 | DRAM/L2 bytes、score中间量、Registers / Thread、Achieved Occupancy、Duration |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-P01`                             |
| 运行前预测     | online不物化score，但当前实现仍可能算术受限                                  |
| 通过标准       | 正确性保持；流量和Duration结论有NCU证据                                      |
| 产物           | `reports/benchmark/attention.csv`及causal/non-causal四份NCU报告              |

### AT-P02：分析Prefill

| 字段           | 内容                                                                              |
| -------------- | --------------------------------------------------------------------------------- |
| 实验编号       | `AT-P02`                                                                          |
| 实验目标       | 分析Prefill                                                                       |
| 对照对象       | naive、online、PyTorch SDPA                                                       |
| 输入矩阵       | 较长S=`64/128`，D=`64`                                                            |
| 正确性指标     | profile前正确性抽查                                                               |
| 性能与因果指标 | 固定八指标；工作量稳定时补查Roofline                                              |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-P02`                                  |
| 运行前预测     | 长S提高算术工作量，先判断SM还是Memory方向                                         |
| 通过标准       | 按固定NCU顺序得出瓶颈                                                             |
| 产物           | `reports/attention/AT-P02-{naive,online}-s{64,128}.md`、四份NCU报告及一份NSYS报告 |

### AT-P03：分析Decode

| 字段           | 内容                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| 实验编号       | `AT-P03`                                                                 |
| 实验目标       | 分析Decode                                                               |
| 对照对象       | KV decode在不同kv_len                                                    |
| 输入矩阵       | Q长度1，KV=`1/32/128/256`                                                |
| 正确性指标     | 输出误差                                                                 |
| 性能与因果指标 | 单kernel latency、Launch Stats、cache bytes、Long Scoreboard、端到端时间 |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-P03`                         |
| 运行前预测     | 小KV可能launch-bound，长KV读取和scoreboard增大                           |
| 通过标准       | 报告kernel与端到端两种时间及随KV变化                                     |
| 产物           | CSV、NSYS、NCU报告                                                       |

### AT-P04：分析Causal特化

| 字段           | 内容                                                       |
| -------------- | ---------------------------------------------------------- |
| 实验编号       | `AT-P04`                                                   |
| 实验目标       | 分析Causal特化                                             |
| 对照对象       | `attention_naive(true/false)`与固定causal                  |
| 输入矩阵       | 相同B/H/S/D和输入                                          |
| 正确性指标     | 各自与reference一致                                        |
| 性能与因果指标 | 分支指令、有效线程比例、Duration、kernel名称               |
| 执行命令       | `bash scripts/run_operator_experiment.sh AT-P04`           |
| 运行前预测     | 特化消除bool分支，但causal有效工作量也更少                 |
| 通过标准       | profiler中kernel可区分，且都通过各自reference              |
| 产物           | naive causal、naive non-causal、固定causal三组NSYS/NCU报告 |

Attention 的 `S×S` 中间矩阵是否真正物化，应以代码和实际内存流量共同判断，不能只根据算法名称推断。

## 9. 公共工程实验

本节每项实验独立执行和记录。正确性或接口契约实验不强行填写固定八指标；只有`ENG-P01`确认时间主要位于目标kernel后，才进入对应算子章节的NCU分析。

在4090服务器先运行`bash scripts/run_debug_experiment.sh preflight`。以下5项实验统一使用`bash scripts/run_debug_experiment.sh <实验编号>`执行，日志写入`reports/debug_labs/<实验编号>.log`。

预检必须确认CUDA可用、当前GPU为Compute Capability 8.9、正式24接口可导入，并能找到`nvcc`、`nsys`、`compute-sanitizer`、`cuobjdump`、`ldd`和C++编译器。缺少任何必需项都停止本节；只有跨GPU实验可以因单卡标记N/A。

### ENG-C01：验证非默认stream语义

| 字段           | 内容                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| 实验编号       | `ENG-C01`                                                                                               |
| 实验目标       | 验证全部24个导出接口都在PyTorch当前stream上分配和launch，不依赖default stream或全局同步                 |
| 对照对象       | 在`torch.cuda.Stream()`中调用的正式算子与同输入reference                                                |
| 输入/故障条件  | 在非默认stream内创建输入、调用算子并记录Event；覆盖GEMM、Softmax、LayerNorm、RMSNorm和Attention         |
| 正确性指标     | 输出误差、输出device、Event完成顺序、24个接口通过数量                                                   |
| 性能与因果指标 | NSYS中的stream ID、kernel所属stream和非必要全局同步；固定八指标为N/A                                    |
| 执行命令       | `bash scripts/run_debug_experiment.sh ENG-C01`                                                          |
| 运行前预测     | 所有kernel应出现在调用时的current stream；测试中的最终同步只用于读取结果，不应掩盖producer/consumer依赖 |
| 通过标准       | 24个接口结果正确且Event顺序成立；实现中使用当前stream，不依赖隐式default-stream排序                     |
| 产物           | `reports/debug_labs/ENG-C01.log`；需要归层时补充`reports/nsys/<representative-op>.nsys-rep`             |

### ENG-C02：验证输入契约在launch前失败

| 字段           | 内容                                                                                     |
| -------------- | ---------------------------------------------------------------------------------------- |
| 实验编号       | `ENG-C02`                                                                                |
| 实验目标       | 验证CPU tensor、错误dtype、非连续布局和错误shape在CUDA launch前产生可定位的接口错误      |
| 对照对象       | 各算子正式支持契约与对应的非法输入                                                       |
| 输入/故障条件  | CPU输入、FP16传给仅支持FP32的接口、transpose非连续输入、维度或shape不匹配                |
| 正确性指标     | 异常类型、错误消息、失败层、是否产生CUDA kernel launch                                   |
| 性能与因果指标 | N/A；本实验关注API契约，不进行性能排名                                                   |
| 执行命令       | `bash scripts/run_debug_experiment.sh ENG-C02`                                           |
| 运行前预测     | 错误应由`TORCH_CHECK`或输入检查直接报告，不能延迟成非法访问或静默fallback                |
| 通过标准       | 所有非法输入在launch前明确失败；消息能够指出device、dtype、contiguous或shape中的实际问题 |
| 产物           | `reports/debug_labs/ENG-C02.log`和“输入 → 预期错误 → 实际错误层”记录表                   |

### ENG-C03：验证same-device与Device Guard

| 字段           | 内容                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------- |
| 实验编号       | `ENG-C03`                                                                                         |
| 实验目标       | 验证多输入必须位于同一CUDA device，并确认当前device不同于输入device时仍由Device Guard选择正确设备 |
| 对照对象       | `device-guard`正确路径与`wrong-device`错误路径                                                    |
| 输入/故障条件  | 两张GPU时分别构造“current device与输入device不同”和“两个输入分属不同device”；单卡4090仅做静态检查 |
| 正确性指标     | 输出device、输出误差、同设备检查错误类型和消息                                                    |
| 性能与因果指标 | N/A；不比较跨device性能                                                                           |
| 执行命令       | `bash scripts/run_debug_experiment.sh ENG-C03`                                                    |
| 运行前预测     | Device Guard路径应在输入device上正确执行；混合device应在launch前失败                              |
| 通过标准       | 多GPU时两条路径均符合预期；单卡环境记录`N/A：仅有一张GPU`，不得填写PASS                           |
| 产物           | `reports/debug_labs/ENG-C03.log`、`stream_device-guard.json`和`stream_wrong-device.json`          |

### ENG-D01：区分launch错误与异步执行错误

| 字段           | 内容                                                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验编号       | `ENG-D01`                                                                                                                                      |
| 实验目标       | 观察立即launch检查与kernel异步执行错误的报错位置差异，并理解`CUDA_LAUNCH_BLOCKING=1`只用于临时定位                                             |
| 对照对象       | 独立故障扩展中的invalid launch、illegal-address执行错误和单元素OOB                                                                             |
| 输入/故障条件  | invalid launch单独执行；illegal-address分别在`CUDA_LAUNCH_BLOCKING=0/1`的新进程执行；单元素OOB只由memcheck取证                                 |
| 正确性指标     | 异常是否出现、首次报错API/源码位置、blocking前后堆栈变化、memcheck定位的thread/address                                                         |
| 性能与因果指标 | N/A；`CUDA_LAUNCH_BLOCKING=1`会改变执行时序，不能用于性能测试                                                                                  |
| 执行命令       | `bash scripts/run_debug_experiment.sh ENG-D01`                                                                                                 |
| 运行前预测     | invalid configuration在launcher处立即失败；illegal-address在异步模式接近synchronize暴露，blocking模式把错误推近故障调用；memcheck定位单元素OOB |
| 通过标准       | 两个blocking进程均非零退出且含CUDA illegal-address证据；能区分launch错误、执行错误、暴露位置和memcheck地址证据                                 |
| 产物           | `invalid_launch.log`、`async_error_blocking_{0,1}.log`、`async_error_summary.log`、`memcheck_oob.log`及`ENG-D01.log`                           |

### ENG-P01：完成Python到kernel的端到端归层

| 字段           | 内容                                                                                                                  |
| -------------- | --------------------------------------------------------------------------------------------------------------------- |
| 实验编号       | `ENG-P01`                                                                                                             |
| 实验目标       | 用NSYS区分输入准备、分配、转换、拷贝、同步、launch gap和kernel时间，决定问题是否应该进入NCU                           |
| 对照对象       | baseline、hidden-copy、hidden-sync、WMMA FP16输入和WMMA FP32输入五个pipeline场景                                      |
| 输入/故障条件  | 相同GEMM目标与size/iters；每次只改变拷贝、同步或dtype转换中的一个因素                                                 |
| 正确性指标     | 每个场景profile前结果与reference一致，记录max abs error                                                               |
| 性能与因果指标 | NVTX区间、CUDA API时间、Memcpy、同步、CPU launch gap、kernel Duration及端到端时间；只有kernel占主导后才分析固定八指标 |
| 执行命令       | `bash scripts/run_debug_experiment.sh ENG-P01`                                                                        |
| 运行前预测     | hidden-copy增加拷贝和分配；hidden-sync增加CPU阻塞；FP32输入增加转换；目标GEMM kernel本身可以保持不变                  |
| 通过标准       | 能指出每个端到端增量属于哪一层；不得把拷贝、同步或转换时间归因给kernel八指标                                          |
| 产物           | `reports/debug_labs/ENG-P01.log`、五份`pipeline_*.nsys-rep`及对应JSON                                                 |

单卡服务器无法验证跨GPU输入时，在结果表中写 `N/A：仅有一张GPU`，不能写“通过”。

## 10. 故障注入练习

故障实验必须在独立临时分支或工程副本中完成。一次只制造一个故障；修复后重新干净构建，并把最小失败输入变成回归测试。不要把错误kernel提交到正式版本。

| 算子族    | 故障                   | 预期工具             | 必须解释的理论          |
| --------- | ---------------------- | -------------------- | ----------------------- |
| GEMM      | 删除尾部判断           | memcheck             | thread到地址、异步错误  |
| GEMM      | 删除一次barrier        | racecheck/synccheck  | shared可见性、CTA同步   |
| GEMM      | 不写shared尾部identity | initcheck/正确性测试 | tile尾部和零填充        |
| Softmax   | 去掉减最大值           | 数值对拍             | exp溢出与稳定Softmax    |
| Softmax   | 删除归约barrier        | racecheck            | Reduction依赖           |
| Norm      | 强制错位输入走float4   | memcheck             | alignment与安全fallback |
| Norm      | 改坏方差/RMS公式       | 数值矩阵             | 累加、eps和消减误差     |
| Attention | 改坏causal条件         | reference对拍        | mask可见范围            |
| Attention | 改坏Online状态缩放     | 极值/长序列对拍      | Online Softmax合并公式  |
| Attention | 放宽`kv_len`边界       | memcheck             | KV cache有效范围        |

统一复盘：

```text
现象：
最小复现（shape / dtype / seed / 命令）：
错误层（构建 / API / launch / kernel / async / numerical / performance）：
候选假设1：
候选假设2：
排除证据：
确认根因：
修复：
新增回归测试：
预防措施：
```

## 11. 故障驱动调试能力专项

本节保留前面的正确性、Benchmark和八指标路径，但验收对象不同：前面的实验回答“正式算子是否正确、性能如何”，本节回答“出现未知故障时，能否定位错误层、选择工具、找到根因并完成回归”。只运行正常代码并得到全绿，不能替代本节。

所有训练代码位于 `debug_labs/`，不加入24个正式导出接口。故意越界、竞态和未初始化读取位于独立的 `debug_labs/fault_extension/`；每次只在一个新Python子进程中运行。正式算子Sanitizer报告出现错误仍然是失败，只有本节明确标记的故障kernel允许出现预期告警。

统一入口：

```bash
source .venv/bin/activate
mkdir -p reports/debug_labs
bash scripts/run_debug_experiment.sh preflight
bash scripts/run_debug_experiment.sh help
```

本节19项实验统一使用`bash scripts/run_debug_experiment.sh <实验编号>`。底层case仍可通过`bash scripts/50_debug_labs.sh help`查看；统一入口负责正确性闸门、独立故障进程、日志路径和Profiler报告命名。

### 11.1 Extension构建、加载与集成故障

这组实验训练从“import失败或代码修改未生效”反查实际加载二进制、绑定导出、构建source列表、动态库和GPU架构，而不是反复修改kernel公式。

#### DBG-E01：确认实际加载的Extension

| 字段           | 内容                                                                               |
| -------------- | ---------------------------------------------------------------------------------- |
| 实验编号       | `DBG-E01`                                                                          |
| 实验目标       | 确认Python实际加载的是当前工程、当前源码构建出的唯一Extension                      |
| 对照对象       | package路径、`_C.so`路径、最新源码时间、二进制时间、Python/C++导出列表             |
| 输入/故障条件  | 干净激活当前虚拟环境，在RTX 4090上加载正式扩展并启用loader检查                     |
| 正确性指标     | 24个Python接口与24个binding符号是否齐全、CUDA是否可用、Compute Capability是否为8.9 |
| 性能与因果指标 | C++ ABI、`ldd`依赖、`cuobjdump`中的sm_89、源码是否新于`.so`；固定八指标为N/A       |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-E01`                                     |
| 运行前预测     | 当前工程路径唯一；源码不新于二进制；动态库依赖完整；二进制包含sm_89                |
| 通过标准       | 严格诊断返回0；24个符号齐全；无`not found`；4090和sm_89证据成立                    |
| 产物           | `reports/debug_labs/DBG-E01.log`和`extension_diagnostic.json`                      |

#### DBG-E02：识别常见构建集成故障

| 字段           | 内容                                                                                               |
| -------------- | -------------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-E02`                                                                                          |
| 实验目标       | 训练把旧二进制、缺失导出、漏编译source、undefined symbol和错误架构配置定位到正确工程层             |
| 对照对象       | `stale-binary`、`missing-export`、`source-omission`、`undefined-symbol`和`arch-config`五个安全案例 |
| 输入/故障条件  | 案例在临时目录或只读诊断路径中制造，不修改正式CUDA源码和24个导出接口                               |
| 正确性指标     | 五个案例的实际现象、异常类型和分类层                                                               |
| 性能与因果指标 | build/import/binding/loader/arch证据；性能指标为N/A                                                |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-E02`                                                     |
| 运行前预测     | 五种故障分别落在构建、导出、source列表、动态加载和架构层，不应首先修改kernel公式                   |
| 通过标准       | 能依据证据正确分类全部五种现象，并给出每种故障的下一条验证命令                                     |
| 产物           | `reports/debug_labs/DBG-E02.log`和`build_integration.log`                                          |

#### DBG-E03：验证隔离故障Extension可构建

| 字段           | 内容                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-E03`                                                                                     |
| 实验目标       | 验证故障训练kernel位于独立模块，可在4090上构建和导入，同时不污染正式接口                      |
| 对照对象       | `debug_labs/fault_extension`的identity基线与正式`aiop4090`扩展                                |
| 输入/故障条件  | `TORCH_CUDA_ARCH_LIST=8.9`；独立执行`setup.py build_ext --inplace`；随后运行安全identity case |
| 正确性指标     | identity输出与输入完全一致、故障模块路径独立、正式接口数量仍为24                              |
| 性能与因果指标 | 编译命令、目标架构和模块加载路径；固定八指标为N/A                                             |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-E03`                                                |
| 运行前预测     | 隔离模块能生成sm_89二进制并通过identity，不改变正式Extension                                  |
| 通过标准       | 构建和导入成功；identity正确；正式源码和导出接口未被修改                                      |
| 产物           | `reports/debug_labs/DBG-E03.log`、故障扩展模块路径和identity输出                              |

#### DBG-E04：区分binding缺失与动态链接失败

| 字段           | 内容                                                                              |
| -------------- | --------------------------------------------------------------------------------- |
| 实验编号       | `DBG-E04`                                                                         |
| 实验目标       | 根据错误形态区分pybind未导出、动态依赖解析失败和CUDA kernel执行错误               |
| 对照对象       | E02的`missing-export`与`undefined-symbol`案例，以及E01的binding和`ldd`结果        |
| 输入/故障条件  | 分别触发不存在的Python绑定符号和带未解析依赖的临时动态库                          |
| 正确性指标     | `AttributeError`、`undefined symbol`或`not found`的实际消息及出现阶段             |
| 性能与因果指标 | Python属性查找、动态loader和kernel launch三层边界；固定八指标为N/A                |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-E04`                                    |
| 运行前预测     | 缺少pybind导出表现为属性错误；链接失败发生在模块加载期；两者都不会进入CUDA kernel |
| 通过标准       | 能写出“现象 → 错误层 → 证据 → 排除kernel层”的完整判断                             |
| 产物           | `reports/debug_labs/DBG-E04.log`和人工填写的`DBG-E04.md`                          |

E01失败时停止后续GPU实验，依次执行`bash scripts/clean_build.sh`和`bash scripts/10_build.sh`，再确认Python实际加载的是当前工程生成的`.so`。不要在未确认加载路径时根据测试现象修改CUDA源码。

### 11.2 Stream、Event与Device真实故障

`current-stream`验证正式算子的正确实现；`missing-event`故意让producer和consumer位于两个非默认stream且不建立依赖；`fixed-event`只增加Event，其他条件不变。

#### DBG-S01：验证算子使用PyTorch当前stream

| 字段           | 内容                                                                                   |
| -------------- | -------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-S01`                                                                              |
| 实验目标       | 验证正式GEMM在consumer current stream执行，并正确消费producer Event建立的数据依赖      |
| 对照对象       | producer stream填充输入、consumer stream等待Event后调用`gemm_naive`与数学期望          |
| 输入/故障条件  | 两个非默认stream、显式Event、延迟producer、`64×64`矩阵                                 |
| 正确性指标     | max abs error、exact equality、producer/consumer stream ID和Event依赖状态              |
| 性能与因果指标 | current stream归属；Duration和固定八指标为N/A                                          |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-S01`                                         |
| 运行前预测     | consumer等待Event后读取到完整输入，GEMM结果等于期望值                                  |
| 通过标准       | JSON状态为PASS且误差为0；能说明正确性来自current-stream与Event依赖，不是default stream |
| 产物           | `reports/debug_labs/stream_current-stream.json`                                        |

#### DBG-S02：复现缺少Event的数据竞争

| 字段           | 内容                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-S02`                                                                                     |
| 实验目标       | 复现producer和consumer跨stream但没有Event依赖时的陈旧数据读取                                 |
| 对照对象       | 与S01相同输入和两个stream，只删除consumer的`wait_event`                                       |
| 输入/故障条件  | producer中使用`torch.cuda._sleep`延迟写入；consumer立即调用GEMM；必要时增加sleep cycles       |
| 正确性指标     | max abs error、重复次数、`EXPECTED_RACE_REPRODUCED`或`INCONCLUSIVE`状态                       |
| 性能与因果指标 | 调度顺序与数据依赖；固定八指标为N/A                                                           |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-S02`                                                |
| 运行前预测     | consumer可能在producer写完前读取旧值；偶尔正确只代表调度暂时掩盖race                          |
| 通过标准       | 至少一次得到`EXPECTED_RACE_REPRODUCED`；未复现时增加sleep并重跑，不能把`INCONCLUSIVE`写成PASS |
| 产物           | `reports/debug_labs/stream_missing-event.json`和重复运行记录                                  |

#### DBG-S03：用Event修复stream依赖

| 字段           | 内容                                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| 实验编号       | `DBG-S03`                                                                           |
| 实验目标       | 在S02基础上只增加Event依赖，验证单变量修复能够消除陈旧读取                          |
| 对照对象       | S02 missing-event与S03 fixed-event                                                  |
| 输入/故障条件  | shape、输入、producer/consumer和sleep cycles保持一致；只增加record/wait Event       |
| 正确性指标     | max abs error、exact equality、修复前后状态变化                                     |
| 性能与因果指标 | Event依赖范围；确认没有使用device-wide synchronize作为修复；固定八指标为N/A         |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-S03`                                      |
| 运行前预测     | 只增加Event后max error回到0，producer与consumer仍可异步执行其他无关工作             |
| 通过标准       | 状态为PASS且误差为0；能够解释Event建立的是两条stream间的数据依赖，不是让GPU整体同步 |
| 产物           | `reports/debug_labs/stream_fixed-event.json`和S02/S03对照结论                       |

#### DBG-S04：验证Device Guard和同设备契约

| 字段           | 内容                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-S04`                                                                                    |
| 实验目标       | 验证输入device决定分配和launch位置，并验证混合device多输入在API层被拒绝                      |
| 对照对象       | `device-guard`正确路径与`wrong-device`错误路径                                               |
| 输入/故障条件  | 至少两张可见GPU；一组输入全部在非current device，另一组输入分属`cuda:0`和`cuda:1`            |
| 正确性指标     | 输出device、输出误差、错误消息和失败层                                                       |
| 性能与因果指标 | Device Guard与same-device契约；性能指标为N/A                                                 |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-S04`                                               |
| 运行前预测     | 同device输入在其device上正确执行；混合device在launch前报告`same CUDA device`                 |
| 通过标准       | 多GPU时两项通过；单张4090明确填写`N/A：仅有一张GPU`                                          |
| 产物           | `reports/debug_labs/stream_device-guard.json`和`reports/debug_labs/stream_wrong-device.json` |

S02是预期失败实验，S03才是修复验证。不能把“没有复现race”记为通过，因为调度顺序可能暂时掩盖缺失依赖。

`DBG-S02`在状态为`INCONCLUSIVE`时会非零退出。增加延迟后重跑，例如：`STREAM_SLEEP_CYCLES=400000000 bash scripts/run_debug_experiment.sh DBG-S02`。

### 11.3 Python到kernel的跨层性能定位

这组实验先用NSYS判断时间在哪一层，再决定是否进入NCU。五个scenario使用相同GEMM目标，但分别保留或注入隐藏`.contiguous()`、逐次同步和WMMA内部类型转换。

#### DBG-L01：区分baseline与hidden copy

| 字段           | 内容                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------ |
| 实验编号       | `DBG-L01`                                                                                  |
| 实验目标       | 判断端到端时间增加是否来自接口层`.contiguous()`拷贝，而不是GEMM kernel退化                 |
| 对照对象       | `baseline`与`hidden-copy`两个NVTX pipeline                                                 |
| 输入/故障条件  | 相同size和iters；hidden-copy只把转置view转换为contiguous后再调用同一`gemm_tiled`           |
| 正确性指标     | 两个场景均与各自reference一致并记录max abs error                                           |
| 性能与因果指标 | NVTX输入准备、Memcpy/拷贝kernel、分配、operator区间、目标kernel Duration和端到端时间       |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-L01`                                             |
| 运行前预测     | hidden-copy增加准备和拷贝时间；目标GEMM kernel的工作量和Duration不应承担这部分增量         |
| 通过标准       | 能用NSYS时间线指出增量所在区间，并明确“不进入NCU分析拷贝开销”                              |
| 产物           | `reports/debug_labs/pipeline_baseline.nsys-rep`、`pipeline_hidden-copy.nsys-rep`及对应JSON |

#### DBG-L02：区分baseline与hidden sync

| 字段           | 内容                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------ |
| 实验编号       | `DBG-L02`                                                                                  |
| 实验目标       | 定位每次调用后的`torch.cuda.synchronize()`如何把异步提交变成CPU阻塞                        |
| 对照对象       | `baseline`与`hidden-sync`两个NVTX pipeline                                                 |
| 输入/故障条件  | 相同输入、size、iters和`gemm_tiled`；只增加逐次device synchronize                          |
| 正确性指标     | 两个场景结果一致、max abs error在容差内                                                    |
| 性能与因果指标 | CUDA同步API时间、CPU阻塞、launch gap、并发损失、kernel Duration和端到端时间                |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-L02`                                             |
| 运行前预测     | kernel实现和单次Duration近似不变，但CPU调用时间与端到端延迟明显增加                        |
| 通过标准       | 能把延迟归到同步API并说明为何kernel八指标无法解释接口层阻塞                                |
| 产物           | `reports/debug_labs/pipeline_baseline.nsys-rep`、`pipeline_hidden-sync.nsys-rep`及对应JSON |

#### DBG-L03：区分WMMA kernel与内部dtype转换

| 字段           | 内容                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| 实验编号       | `DBG-L03`                                                                  |
| 实验目标       | 比较FP16输入和FP32输入调用WMMA接口时，内部转换与临时分配对端到端时间的影响 |
| 对照对象       | `wmma-fp16-input`与`wmma-fp32-input`                                       |
| 输入/故障条件  | 相同M/N/K和iters；一组预先准备FP16输入，另一组由接口内部执行FP32到FP16转换 |
| 正确性指标     | 两个场景分别与对应低精度reference一致，记录max abs error                   |
| 性能与因果指标 | FP32→FP16转换kernel、临时分配、额外流量、WMMA kernel Duration和端到端时间  |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-L03`                             |
| 运行前预测     | WMMA kernel可以同样快，但FP32输入路径因转换和临时tensor导致接口时间更长    |
| 通过标准       | 同时报告kernel Duration和端到端时间，能够解释“kernel快但接口不一定快”      |
| 产物           | 两份`reports/debug_labs/pipeline_wmma-*.nsys-rep`及对应JSON                |

#### DBG-L04：区分launch检查与执行期OOB

| 字段           | 内容                                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| 实验编号       | `DBG-L04`                                                                           |
| 实验目标       | 对比invalid configuration和kernel内越界写，建立launch/API与异步kernel错误的分层判断 |
| 对照对象       | 故障扩展中的`invalid_launch`与`out_of_bounds`                                       |
| 输入/故障条件  | 每个故障使用独立Python进程；OOB只在Compute Sanitizer下作为正式证据                  |
| 正确性指标     | launcher异常位置、memcheck的kernel/thread/address/access type/源码行                |
| 性能与因果指标 | 错误暴露时机和CUDA context隔离；性能指标为N/A                                       |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-L04`                                      |
| 运行前预测     | invalid configuration由launch check立即报告；OOB需要执行和同步或memcheck才能定位    |
| 通过标准       | 能解释两类错误为何出现在不同层，并说明OOB后为何不能复用同一CUDA context             |
| 产物           | `reports/debug_labs/invalid_launch.log`和`reports/debug_labs/memcheck_oob.log`      |

只有NSYS确认主要时间位于目标kernel后，才使用固定八指标和专项NCU指标。隐藏拷贝、同步或内部转换占主导时，直接分析kernel八指标属于错误归层。

### 11.4 独立Sanitizer故障与上下文隔离

#### DBG-T01：用memcheck定位越界写

| 字段           | 内容                                                                  |
| -------------- | --------------------------------------------------------------------- |
| 实验编号       | `DBG-T01`                                                             |
| 实验目标       | 从Compute Sanitizer报告反推出thread索引如何写到输出末尾之后           |
| 对照对象       | 安全identity kernel与故意OOB kernel                                   |
| 输入/故障条件  | 故障kernel令一个thread写入`index == count`；使用独立Python进程        |
| 正确性指标     | 工具错误数量、kernel名、thread/block、地址、写访问宽度和源码行        |
| 性能与因果指标 | 地址计算与边界判断；Duration和固定八指标为N/A                         |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-T01`                        |
| 运行前预测     | memcheck报告invalid global write，地址位于合法分配末尾之后            |
| 通过标准       | 只出现预期OOB；能由thread索引推导出`index == count`并提出正确尾部判断 |
| 产物           | `reports/debug_labs/memcheck_oob.log`                                 |

#### DBG-T02：用racecheck定位shared-memory竞态

| 字段           | 内容                                                                                   |
| -------------- | -------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-T02`                                                                              |
| 实验目标       | 识别多个thread对同一shared位置产生的WAW/RAW hazard，并理解结果偶尔正确也不代表同步合法 |
| 对照对象       | 正确同步的shared访问与故意缺少依赖的race kernel                                        |
| 输入/故障条件  | 多thread读写同一shared地址；独立进程运行racecheck                                      |
| 正确性指标     | hazard类型、冲突thread、shared地址、访问顺序和源码行                                   |
| 性能与因果指标 | shared可见性、CTA同步与barrier缺失；Duration和固定八指标为N/A                          |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-T02`                                         |
| 运行前预测     | racecheck报告WAW或RAW hazard，即使普通运行输出有时看起来正确                           |
| 通过标准       | 只出现预期shared竞态；能够说明所需同步点以及为什么不能依赖warp/调度偶然顺序            |
| 产物           | `reports/debug_labs/racecheck_race.log`                                                |

#### DBG-T03：用initcheck定位未初始化读取

| 字段           | 内容                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 实验编号       | `DBG-T03`                                                                     |
| 实验目标       | 区分未初始化device memory读取、普通数值误差和越界访问                         |
| 对照对象       | 已初始化输入路径与从原始未初始化CUDA分配读取的故障kernel                      |
| 输入/故障条件  | 不写入目标device分配便直接读取并传播到输出；独立进程运行initcheck             |
| 正确性指标     | uninitialized read、访问位置、受影响thread以及传播到输出的路径                |
| 性能与因果指标 | 数据初始化契约；Duration和固定八指标为N/A                                     |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-T03`                                |
| 运行前预测     | initcheck报告未初始化读取；普通误差容差或memcheck不能替代这项证据             |
| 通过标准       | 只出现预期未初始化读取；能够给出初始化位置或identity填充值修复，并区分它与OOB |
| 产物           | `reports/debug_labs/initcheck_init.log`                                       |

脚本使用Compute Sanitizer的非零`error-exitcode`确认工具确实发现了故障，再把预期发现转换为实验PASS。OOB进程结束后必须启动新Python进程；不得继续复用可能已经损坏的CUDA context。

### 11.5 未知故障与根因复盘验收

运行前不要查看`unknown_fault_lab.py`的实现或答案。可先执行`bash scripts/50_debug_labs.sh unknown`随机抽题；正式验收时固定case并至少重复3次。每题必须先按第10节模板完成复盘，最后才允许执行`python debug_labs/unknown_fault_lab.py --case <U01-U04> --reveal`。

#### DBG-U01：未知故障案例U01

| 字段           | 内容                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-U01`                                                                                               |
| 实验目标       | 从U01给出的现象开始，在不知道答案的情况下完成最小复现、错误分层、假设证伪、根因定位和回归               |
| 对照对象       | U01实际输出、所选reference或修复前后单变量对照                                                          |
| 输入/故障条件  | 固定case U01、shape、dtype、seed和命令；相同条件重复不少于3次                                           |
| 正确性指标     | 现象稳定性、首个异常位置、修复前后差异和正式pytest回归结果                                              |
| 性能与因果指标 | 根据现象选择pytest、Sanitizer、NSYS或NCU；记录选择理由和未选择其他工具的理由                            |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-U01`                                                          |
| 运行前预测     | 看答案前写错误层和至少两个可证伪假设；保留错误预测和排除证据                                            |
| 通过标准       | 20至30分钟内完成分层和定位；至少一项工具证据加一项代码/契约证据；单变量修复后最小复现和全量pytest均通过 |
| 产物           | `reports/debug_labs/unknown_U01.json`和`reports/debug_labs/DBG-U01.md`                                  |

#### DBG-U02：未知故障案例U02

| 字段           | 内容                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-U02`                                                                                               |
| 实验目标       | 从U02给出的现象开始，在不知道答案的情况下完成最小复现、错误分层、假设证伪、根因定位和回归               |
| 对照对象       | U02实际输出、所选reference或修复前后单变量对照                                                          |
| 输入/故障条件  | 固定case U02、shape、dtype、seed和命令；相同条件重复不少于3次                                           |
| 正确性指标     | 现象稳定性、首个异常位置、修复前后差异和正式pytest回归结果                                              |
| 性能与因果指标 | 根据现象选择pytest、Sanitizer、NSYS或NCU；记录选择理由和未选择其他工具的理由                            |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-U02`                                                          |
| 运行前预测     | 看答案前写错误层和至少两个可证伪假设；保留错误预测和排除证据                                            |
| 通过标准       | 20至30分钟内完成分层和定位；至少一项工具证据加一项代码/契约证据；单变量修复后最小复现和全量pytest均通过 |
| 产物           | `reports/debug_labs/unknown_U02.json`和`reports/debug_labs/DBG-U02.md`                                  |

#### DBG-U03：未知故障案例U03

| 字段           | 内容                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-U03`                                                                                               |
| 实验目标       | 从U03给出的现象开始，在不知道答案的情况下完成最小复现、错误分层、假设证伪、根因定位和回归               |
| 对照对象       | U03实际输出、所选reference或修复前后单变量对照                                                          |
| 输入/故障条件  | 固定case U03、shape、dtype、seed和命令；相同条件重复不少于3次                                           |
| 正确性指标     | 现象稳定性、首个异常位置、修复前后差异和正式pytest回归结果                                              |
| 性能与因果指标 | 根据现象选择pytest、Sanitizer、NSYS或NCU；记录选择理由和未选择其他工具的理由                            |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-U03`                                                          |
| 运行前预测     | 看答案前写错误层和至少两个可证伪假设；保留错误预测和排除证据                                            |
| 通过标准       | 20至30分钟内完成分层和定位；至少一项工具证据加一项代码/契约证据；单变量修复后最小复现和全量pytest均通过 |
| 产物           | `reports/debug_labs/unknown_U03.json`和`reports/debug_labs/DBG-U03.md`                                  |

#### DBG-U04：未知故障案例U04

| 字段           | 内容                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| 实验编号       | `DBG-U04`                                                                                               |
| 实验目标       | 从U04给出的现象开始，在不知道答案的情况下完成最小复现、错误分层、假设证伪、根因定位和回归               |
| 对照对象       | U04实际输出、所选reference或修复前后单变量对照                                                          |
| 输入/故障条件  | 固定case U04、shape、dtype、seed和命令；相同条件重复不少于3次                                           |
| 正确性指标     | 现象稳定性、首个异常位置、修复前后差异和正式pytest回归结果                                              |
| 性能与因果指标 | 根据现象选择pytest、Sanitizer、NSYS或NCU；记录选择理由和未选择其他工具的理由                            |
| 执行命令       | `bash scripts/run_debug_experiment.sh DBG-U04`                                                          |
| 运行前预测     | 看答案前写错误层和至少两个可证伪假设；保留错误预测和排除证据                                            |
| 通过标准       | 20至30分钟内完成分层和定位；至少一项工具证据加一项代码/契约证据；单变量修复后最小复现和全量pytest均通过 |
| 产物           | `reports/debug_labs/unknown_U04.json`和`reports/debug_labs/DBG-U04.md`                                  |

### 11.6 调试能力专项验收表

| 能力                   | 必做实验                 | 状态 | 关键证据路径 | 根因复盘编号 |
| ---------------------- | ------------------------ | ---- | ------------ | ------------ |
| Extension构建与集成    | DBG-E01至DBG-E04         |      |              |              |
| Stream/Event/Device    | DBG-S01至DBG-S04         |      |              |              |
| Python到kernel跨层定位 | DBG-L01至DBG-L04         |      |              |              |
| Sanitizer报告解读      | DBG-T01至DBG-T03         |      |              |              |
| 未知故障闭环           | DBG-U01至DBG-U04至少两题 |      |              |              |

## 12. 最终验收表

状态只允许填写：`PASS`、`FAIL`、`BLOCKED`、`N/A`。

| 算子族    | 版本数 | 基础正确性 | 深度边界/数值 | Stream/契约 | Sanitizer | Benchmark | NSYS | 八指标完成数 | 补充指标完成数 | 专项NCU | 总状态 |
| --------- | ------ | ---------- | ------------- | ----------- | --------- | --------- | ---- | ------------ | -------------- | ------- | ------ |
| GEMM      | 7      |            |               |             |           |           |      | /7           | /7             |         |        |
| Softmax   | 4      |            |               |             |           |           |      | /4           | /4             |         |        |
| LayerNorm | 4      |            |               |             |           |           |      | /4           | /4             |         |        |
| RMSNorm   | 5      |            |               |             |           |           |      | /5           | /5             |         |        |
| Attention | 4      |            |               |             |           |           |      | /4           | /4             |         |        |

每个算子族的最终结论必须包含：

```text
已验证环境：
已验证shape/dtype：
未验证范围：
最优版本及适用shape：
主要瓶颈：
支持结论的正确性证据：
支持结论的NSYS/NCU证据：
已知代价和fallback：
下一步：
```

## 13. 通过标准

- `bash scripts/run_debug_experiment.sh preflight`返回PASS，确认RTX 4090、sm_89、24个正式接口和调试工具链齐全。
- 全量和分组 pytest 通过。
- 支持契约内的有限输入不产生意外 NaN/Inf。
- 非默认stream下结果正确。
- Compute Sanitizer没有未解释错误。
- Benchmark至少记录median、P90、min和max。
- 24个正式导出版本全部填写Duration、Compute (SM) Throughput、Memory Throughput、DRAM Throughput、L2 Cache Throughput、Achieved Occupancy、Registers / Thread和Top Stall Reason。
- 24个正式导出版本的补充指标结果表全部填写；不适用项必须记录`N/A + 原因`，不得留空。
- Launch Stats与Roofline只作为按需补充指标；使用时记录触发原因和结论，不计入八指标完成数。
- 性能结论同时包含Duration、工作量/流量和资源或Top Stall Reason证据。
- 不要求超过PyTorch/cuBLAS，但不得用“某个百分比更高”代替因果解释。
- 每个失败都保留最小复现和回归测试。
- Extension严格诊断通过，能区分旧`.so`、缺失binding、动态链接和架构配置问题。
- invalid launch、异步illegal-address和memcheck OOB三类错误均被独立进程捕获；`CUDA_LAUNCH_BLOCKING=0/1`对照日志能够解释错误暴露位置变化。
- 缺少Event的stream race至少复现一次，增加Event后在相同输入和stream条件下修复。
- NSYS能区分hidden copy、hidden sync、dtype转换和目标kernel；只有确认kernel占主导后才进入NCU。
- 隔离故障扩展的memcheck、racecheck和initcheck均找到预期源码证据，且正式扩展Sanitizer保持干净。
- 至少两道未知故障在查看答案前完成完整复盘，其中至少一道在30分钟内定位根因。

## 14. 官方工具参考

- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/pdf/UserGuide.pdf)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

当前本地环境只进行静态检查。最终结论必须在 RTX 4090 服务器实际构建和执行后填写。
