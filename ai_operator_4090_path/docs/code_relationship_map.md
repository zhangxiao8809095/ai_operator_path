# aiop4090 算子代码关系图

这份文档按“开发调试时该看哪里、改哪里”的角度整理当前仓库。整个工程的主线是：

CUDA 编程模型基础动画：

[打开 CUDA 动画：线程、Warp 与内存模型](./cuda_model_animation/index.html)

NCU 性能指标动画：

[打开 NCU 动画：五个指标与算子瓶颈](./ncu_metrics_animation/index.html)

GEMM 三个版本的交互动画：

[打开 GEMM 动画：Naive → Tiled → Register Tiling](./gemm_animation/index.html)

```text
CUDA/C++ kernel
  -> PyTorch extension binding: aiop4090._C
  -> Python wrapper: aiop4090
  -> correctness tests
  -> benchmark / Nsight profiling
```

## 1. 顶层目录职责

| 路径 | 作用 | 开发时通常改什么 |
|---|---|---|
| `src/aiop4090/csrc/*.cu` | 算子的 CUDA kernel 和 C++ host launcher | 算法实现、block/grid 配置、shape/dtype 检查、输出分配 |
| `src/aiop4090/csrc/bindings.cpp` | pybind11 导出 C++ 函数到 `aiop4090._C` | 新增/改名算子时增加声明和 `m.def` |
| `src/aiop4090/__init__.py` | Python 侧薄包装，用户和测试直接 import 这里 | 暴露 Python API、设置默认参数、做轻量参数转换 |
| `tests/*.py` | correctness 对拍 PyTorch baseline | 增加 shape、dtype、边界情况、容差 |
| `benchmark/*.py` | benchmark 和 profiling 的 Python 入口 | 增加性能 shape、计时逻辑、profile 单算子入口 |
| `scripts/*.sh` | 构建、测试、benchmark、Nsight 包装脚本 | 改 Python 环境、CUDA/Nsight 路径、输出目录 |
| `setup.py` | 编译 PyTorch CUDA extension | 新增 `.cu/.cpp` 文件、改编译参数、改扩展名 |
| `docs/*.md/.py` | 路线图、报告模板、算法 reference | 写报告、保存算法推导和实验结论 |

## 2. 构建和导出链路

### 2.1 编译入口

`setup.py` 通过 `torch.utils.cpp_extension.CUDAExtension` 编译扩展模块：

```text
extension name: aiop4090._C
sources:
  bindings.cpp
  gemm.cu
  softmax.cu
  norm.cu
  attention.cu
```

`scripts/10_build.sh` 执行：

```bash
python -m pip install -e . --no-build-isolation
```

所以修改 `.cu/.cpp` 后，通常执行：

```bash
bash scripts/clean_build.sh
bash scripts/10_build.sh
bash scripts/20_test.sh
```

### 2.2 Python 调用到 CUDA kernel 的固定路径

以 `gemm_tiled` 为例：

```text
Python test/benchmark
  calls ops.gemm_tiled(a, b)
    -> src/aiop4090/__init__.py: gemm_tiled(a, b)
      -> aiop4090._C.gemm_tiled(a, b)
        -> src/aiop4090/csrc/bindings.cpp: m.def("gemm_tiled", &gemm_tiled, ...)
          -> src/aiop4090/csrc/gemm.cu: torch::Tensor gemm_tiled(...)
            -> check_gemm_inputs(...)
            -> allocate C
            -> launch gemm_tiled_kernel<<<grid, block>>>(...)
```

所有算子都遵循这个模式：

```text
Python wrapper name
  == binding export name
  == C++ host function name
  -> launches one CUDA kernel in the same .cu file
```

## 3. 通用输入约束

`src/aiop4090/csrc/common.h` 定义了通用检查：

```text
CHECK_CUDA(x)       要求 CUDA tensor
CHECK_CONTIGUOUS(x) 要求 contiguous
CHECK_FLOAT32(x)    要求 torch.float32
CHECK_INPUT(x)      同时检查以上三项
ceil_div_int(a,b)   grid 维度上取整辅助函数
```

当前所有 kernel 都只支持 CUDA contiguous float32。开发时如果遇到这些需求：

| 需求 | 主要修改点 |
|---|---|
| 支持 fp16/bf16 | `common.h` 的 dtype 检查、`.cu` kernel 的 pointer 类型/计算类型、测试容差、benchmark dtype |
| 支持非 contiguous 输入 | Python wrapper 里主动 `.contiguous()`，或 C++ 支持 stride；当前更推荐调用侧显式 contiguous |
| 支持 CPU fallback | `__init__.py` 或新增 Python reference；当前工程目标是 CUDA kernel，不含 CPU 实现 |
| 更好的错误信息 | 各 `.cu` host function 的 `TORCH_CHECK` |

## 4. GEMM 关系图

### 4.1 函数链

| Python API | Binding | C++ host launcher | CUDA kernel | 所在文件 |
|---|---|---|---|---|
| `ops.gemm_naive(a, b)` | `_C.gemm_naive` | `torch::Tensor gemm_naive(A, B)` | `gemm_naive_kernel` | `src/aiop4090/csrc/gemm.cu` |
| `ops.gemm_tiled(a, b)` | `_C.gemm_tiled` | `torch::Tensor gemm_tiled(A, B)` | `gemm_tiled_kernel` | `src/aiop4090/csrc/gemm.cu` |
| `ops.gemm_regtile2x2(a, b)` | `_C.gemm_regtile2x2` | `torch::Tensor gemm_regtile2x2(A, B)` | `gemm_regtile2x2_kernel` | `src/aiop4090/csrc/gemm.cu` |

### 4.2 当前实现分工

`check_gemm_inputs(A, B)`：

- 检查 `A/B` 是 CUDA contiguous float32。
- 检查 `A` 和 `B` 是 2D。
- 检查 `A.size(1) == B.size(0)`。

`gemm_naive_kernel`：

- 一个 thread 计算一个 `C[row, col]`。
- 每个输出元素串行遍历 `K`。
- 用于 correctness 和最基础 profiling。

`gemm_tiled_kernel`：

- `TILE = 16`。
- 每个 block 计算一个 `16x16` 输出 tile。
- 使用 shared memory 缓存 `A/B` tile。
- 支持非 16 倍数 shape，通过越界处填 0。

`gemm_regtile2x2_kernel`：

- `RT_TILE = 16, RM = 2, RN = 2`。
- 每个 thread 计算相邻 `2x2` 输出。
- block 覆盖 `32x32` 输出区域。
- 是 register tiling 的演示版本，后续可扩到 `4x4/8x4`。

### 4.3 测试和 benchmark

| 文件 | 调用 | 用途 |
|---|---|---|
| `tests/test_gemm.py` | `ops.gemm_*` vs `a @ b` | correctness，对 square 和非整倍数 shape 做对拍 |
| `benchmark/bench_ops.py::bench_gemm` | `torch_mm`、`gemm_naive`、`gemm_tiled`、`gemm_regtile2x2` | 通用 benchmark，输出 ms 和 TFLOP/s |
| `benchmark/bench_gemm_shapes.py` | 只跑 GEMM 多个大 shape | GEMM 专项 benchmark |
| `benchmark/profile_entry.py` | `--op gemm_naive/gemm_tiled/gemm_regtile2x2` | Nsight 单算子重复执行入口 |

### 4.4 GEMM 开发时改哪里

| 需求 | 主要文件 | 同步修改 |
|---|---|---|
| 优化已有 `gemm_tiled` kernel | `src/aiop4090/csrc/gemm.cu` | 跑 `tests/test_gemm.py` 和 GEMM benchmark |
| 新增 `gemm_padding` 或 `gemm_wmma` | `gemm.cu`、`bindings.cpp`、`__init__.py` | `tests/test_gemm.py`、`bench_ops.py`、`profile_entry.py`、README/docs |
| 改 tile/register 参数 | `gemm.cu` 里的 `TILE/RT_TILE/RM/RN` 和对应 grid/block | benchmark shape 和 Nsight 指标 |
| 增加非整倍数 shape 测试 | `tests/test_gemm.py` | 无需改 kernel，除非测试失败 |

## 5. Softmax / Norm 关系图

### 5.1 函数链

| Python API | Binding | C++ host launcher | CUDA kernel | 所在文件 |
|---|---|---|---|---|
| `ops.softmax_row(x)` | `_C.softmax_row` | `torch::Tensor softmax_row(X)` | `softmax_row_kernel` | `src/aiop4090/csrc/softmax.cu` |
| `ops.layernorm_row(x, gamma, beta, eps)` | `_C.layernorm_row` | `torch::Tensor layernorm_row(X, gamma, beta, eps)` | `layernorm_row_kernel` | `src/aiop4090/csrc/norm.cu` |
| `ops.rmsnorm_row(x, gamma, eps)` | `_C.rmsnorm_row` | `torch::Tensor rmsnorm_row(X, gamma, eps)` | `rmsnorm_row_kernel` | `src/aiop4090/csrc/norm.cu` |

### 5.2 当前实现分工

`softmax_row_kernel`：

- 输入 `X` shape 是 `[rows, cols]`。
- 一个 block 处理一行。
- `block = 256`。
- 第一遍规约 row max。
- 第二遍规约 exp sum。
- 第三遍写出 `exp(x - max) / sum`。
- shared memory 用同一段 `smem` 复用做 max/sum。

`layernorm_row_kernel`：

- 输入 `X` shape 是 `[rows, cols]`。
- `gamma/beta` shape 是 `[cols]`。
- 一个 block 处理一行。
- 第一遍规约 mean。
- 第二遍规约 variance。
- 第三遍写出 `(x - mean) * inv_std * gamma + beta`。

`rmsnorm_row_kernel`：

- 输入 `X` shape 是 `[rows, cols]`。
- `gamma` shape 是 `[cols]`。
- 一个 block 处理一行。
- 规约 `mean(x^2)`，写出 `x * rsqrt(mean(x^2)+eps) * gamma`。

### 5.3 测试和 benchmark

| 文件 | 调用 | reference |
|---|---|---|
| `tests/test_softmax_norm.py::test_softmax_row` | `ops.softmax_row` | `torch.softmax(x, dim=-1)` |
| `tests/test_softmax_norm.py::test_layernorm_row` | `ops.layernorm_row` | `torch.nn.functional.layer_norm` |
| `tests/test_softmax_norm.py::test_rmsnorm_row` | `ops.rmsnorm_row` | Python 表达式 `x * rsqrt(mean(x^2)+eps) * gamma` |
| `benchmark/bench_ops.py::bench_softmax_norm` | `ops.softmax_row/layernorm_row/rmsnorm_row` | 和 PyTorch softmax/layer_norm 同场计时 |
| `benchmark/profile_entry.py` | `--op softmax/layernorm/rmsnorm` | Nsight 单算子入口 |

### 5.4 Softmax / Norm 开发时改哪里

| 需求 | 主要文件 | 同步修改 |
|---|---|---|
| 把 block reduction 改成 warp shuffle | `softmax.cu` 或 `norm.cu` kernel | correctness + Nsight 指标 |
| 加 online softmax 版本 | `softmax.cu` | `bindings.cpp`、`__init__.py`、tests、benchmark、profile_entry |
| 加 fp16/bf16 norm | `common.h`、`norm.cu` | 测试 dtype、容差、benchmark dtype |
| 加 fused softmax/dropout 或 fused norm | 新 `.cu` 或现有 `.cu` | `setup.py` sources、bindings、tests、benchmark |
| 调整列数/行数测试覆盖 | `tests/test_softmax_norm.py` | 可能同步 benchmark shape |

## 6. Attention 关系图

### 6.1 函数链

| Python API | Binding | C++ host launcher | CUDA kernel | 所在文件 |
|---|---|---|---|---|
| `ops.attention_naive(q, k, v, causal=True)` | `_C.attention_naive` | `torch::Tensor attention_naive(Q, K, V, causal)` | `attention_naive_kernel` | `src/aiop4090/csrc/attention.cu` |

### 6.2 当前实现分工

`attention_naive(Q, K, V, causal)` host launcher：

- 检查 `Q/K/V` 是 CUDA contiguous float32。
- 检查 shape 都是 `[B, H, S, D]`。
- 检查三者 shape 完全一致。
- 输出 `O = torch::empty_like(Q)`。
- `block = 128`。
- `grid = dim3(D, S, B * H)`。

`attention_naive_kernel`：

- 每个 block 计算一个输出元素 `O[b, h, i, d_out]`。
- `blockIdx.x = d_out`，`blockIdx.y = query position i`，`blockIdx.z = b*h`。
- 对所有 key position `j` 计算 `Q[i] dot K[j] / sqrt(D)`。
- causal 模式下跳过 `j > i`。
- 先规约 max，再规约 softmax denominator，再规约 `sum(p_j * V[j, d_out])`。
- 为了直观，dot product 会重复计算，是故意低效的 naive baseline。

### 6.3 测试和 benchmark

| 文件 | 调用 | reference |
|---|---|---|
| `tests/test_attention.py::_ref_attention` | PyTorch matmul + mask + softmax + matmul | attention correctness baseline |
| `tests/test_attention.py::test_attention_naive_causal` | `ops.attention_naive(..., True)` | causal attention |
| `tests/test_attention.py::test_attention_naive_non_causal` | `ops.attention_naive(..., False)` | non-causal attention |
| `benchmark/bench_ops.py::bench_attention` | `torch.nn.functional.scaled_dot_product_attention` 和 `ops.attention_naive` | attention benchmark |
| `benchmark/profile_entry.py` | `--op attention_naive` | Nsight 单算子入口 |

### 6.4 Attention 开发时改哪里

| 需求 | 主要文件 | 同步修改 |
|---|---|---|
| 优化 naive attention 内部实现 | `src/aiop4090/csrc/attention.cu` | attention 测试 + profile_entry |
| 新增 tiled attention | `attention.cu` | `bindings.cpp`、`__init__.py`、tests、bench_ops、profile_entry |
| 新增 online softmax attention | `attention.cu`，参考 `docs/online_softmax_reference.py` | tests 覆盖更大 `S/D`，benchmark shape |
| 支持 KV cache/decode | `attention.cu`，参考 `docs/kv_cache_reference.py` | 新增 test/reference 和 profiling 入口 |
| 支持不同 Q/K/V shape | `attention.cu` host shape 检查和 kernel indexing | tests 中增加 cross-attention 或 decode shape |

## 7. 测试脚本调用关系

测试统一从 `aiop4090` 包进入，不直接调用 `_C` 或 kernel：

```text
tests/test_gemm.py
  -> import aiop4090 as ops
  -> ops.gemm_naive / ops.gemm_tiled / ops.gemm_regtile2x2
  -> PyTorch a @ b reference

tests/test_softmax_norm.py
  -> import aiop4090 as ops
  -> ops.softmax_row / layernorm_row / rmsnorm_row
  -> PyTorch reference

tests/test_attention.py
  -> import aiop4090 as ops
  -> ops.attention_naive
  -> _ref_attention: matmul + causal mask + softmax + matmul
```

测试的职责是确认“数值正确”。如果你只改 kernel 内部优化，测试 API 通常不用变；如果你新增 Python API、shape 支持、dtype 支持，就要同步加测试。

## 8. Benchmark 和 Profiling 调用关系

### 8.1 日常 benchmark

`scripts/30_bench.sh`：

```text
python benchmark/bench_ops.py --op all
```

`benchmark/bench_ops.py`：

```text
main()
  -> bench_gemm()
       -> cuda_bench(lambda: ops.gemm_*(...))
  -> bench_softmax_norm()
       -> cuda_bench(lambda: ops.softmax_row/layernorm_row/rmsnorm_row(...))
  -> bench_attention()
       -> cuda_bench(lambda: ops.attention_naive(...))
```

`cuda_bench(fn)`：

- warmup 多次执行 `fn()`。
- `torch.cuda.synchronize()`。
- 用 CUDA event 记录 repeat 次调用总耗时。
- 返回平均 `ms/op`。

### 8.2 GEMM 专项 benchmark

`benchmark/bench_gemm_shapes.py`：

- 从 `bench_ops` 复用 `cuda_bench`。
- 跑更多 GEMM 大 shape。
- 输出 `ms` 和 `TFLOP/s`。

### 8.3 Nsight Compute

`scripts/profile_ncu.sh OP`：

```text
ncu --set speed-of-light
  -> python benchmark/profile_entry.py --op OP --iters ITERS
```

`scripts/profile_ncu_full.sh OP`：

```text
ncu --set full
  -> python benchmark/profile_entry.py --op OP --iters ITERS
```

`benchmark/profile_entry.py`：

```text
main()
  -> parse --op
  -> create fixed input shape for that op
  -> fn = lambda: ops.<selected_op>(...)
  -> repeat(fn, iters)
```

这里的重点不是输出耗时，而是让 Nsight 抓到足够多次 kernel launch。

### 8.4 Nsight Systems

`scripts/profile_nsys.sh`：

```text
nsys profile
  -> python benchmark/bench_ops.py --op all
```

适合看整体时间线、launch overhead、Python/CUDA 调用节奏。

## 9. 新增一个算子的标准步骤

假设要新增 `my_op`：

1. 在合适的 `.cu` 文件里实现：

```text
namespace {
__global__ void my_op_kernel(...) { ... }
}

torch::Tensor my_op(torch::Tensor X, ...) {
    CHECK_INPUT(X);
    TORCH_CHECK(...);
    auto Y = torch::empty_like(X);
    my_op_kernel<<<grid, block, shared_mem>>>(...);
    return Y;
}
```

2. 在 `src/aiop4090/csrc/bindings.cpp` 增加声明和导出：

```text
torch::Tensor my_op(torch::Tensor X, ...);
m.def("my_op", &my_op, "...");
```

3. 在 `src/aiop4090/__init__.py` 增加 Python 包装：

```python
def my_op(x, ...):
    return _C.my_op(x, ...)
```

4. 如果新增了新的 `.cu` 文件，在 `setup.py` 的 `sources` 里加入路径。

5. 在 `tests/` 新增或扩展 correctness test，与 PyTorch/Python reference 对拍。

6. 在 `benchmark/bench_ops.py` 或专项 benchmark 中加入性能 case。

7. 在 `benchmark/profile_entry.py` 的 `choices` 和分支里加入 `--op my_op`。

8. 如需 Nsight，复用：

```bash
bash scripts/profile_ncu.sh my_op
bash scripts/profile_ncu_full.sh my_op
```

9. 更新 README 或 `docs/phase_map.md`，记录当前版本和下一步优化点。

## 10. 常见需求到文件的速查表

| 具体需求 | 优先看/改 |
|---|---|
| kernel 计算结果错 | 对应 `tests/test_*.py` 看 reference，再看对应 `.cu` kernel indexing/reduction |
| shape 不支持或报错 | 对应 `.cu` host launcher 的 `TORCH_CHECK` 和 grid/block 计算 |
| dtype 不支持 | `common.h`、对应 `.cu` pointer/accumulator、tests 容差 |
| Python 找不到函数 | `__init__.py` 是否包装，`bindings.cpp` 是否 `m.def`，是否重新 build |
| build 后仍像旧代码 | `scripts/clean_build.sh` 后重新 `scripts/10_build.sh` |
| 新 `.cu` 没被编译 | `setup.py` 的 `CUDAExtension.sources` |
| pytest 失败但 benchmark 能跑 | benchmark 没做 correctness，对照 `tests` 查数值 |
| benchmark 想加新 shape | `benchmark/bench_ops.py` 或 `benchmark/bench_gemm_shapes.py` |
| Nsight 想 profile 新 op | `benchmark/profile_entry.py` choices/分支，加上固定输入 |
| ncu 找不到 Python/CUDA | `scripts/profile_ncu.sh` / `profile_ncu_full.sh` 的 `PYTHON_BIN`、CUDA path、`LD_LIBRARY_PATH` |
| 想看整体 launch 时间线 | `scripts/profile_nsys.sh` |
| 想写性能报告 | `docs/report_template.md` |

## 11. 推荐调试顺序

开发一个 kernel 优化版本时，建议按这个顺序走：

```text
1. 改 .cu kernel / host launcher
2. clean build + rebuild
3. 跑对应 pytest
4. 跑小 benchmark 看是否明显退化
5. 用 profile_entry + ncu 看单 kernel 指标
6. 把结果写到 report_template
7. 再决定下一轮优化方向
```

对应命令：

```bash
bash scripts/clean_build.sh
bash scripts/10_build.sh
pytest -q tests/test_gemm.py
python benchmark/bench_gemm_shapes.py
bash scripts/profile_ncu.sh gemm_tiled
```

把 `test_gemm.py` 和 `gemm_tiled` 换成当前正在开发的算子即可。

## 12. 逐文件函数作用速查

本节按文件列出当前仓库里的自定义函数、CUDA kernel、C++ host launcher 和测试/benchmark 入口。没有自定义函数、但有顶层执行逻辑的脚本也单独说明，方便开发时判断入口在哪里。

### 12.1 `setup.py`

| 函数/调用 | 作用 | 开发关注点 |
|---|---|---|
| `setup(...)` | setuptools 打包入口；配置包名、源码目录、扩展模块和构建命令 | 新增 `.cu/.cpp` 文件时，改 `CUDAExtension.sources` |
| `CUDAExtension(name="aiop4090._C", sources=[...])` | 定义 PyTorch CUDA extension，最终生成 Python 可 import 的 `aiop4090._C` | 改扩展名、编译文件列表、nvcc/cxx 编译参数 |
| `BuildExtension` | PyTorch 提供的 C++/CUDA extension 构建后端 | 一般不改，构建异常时再查 |

### 12.2 `src/aiop4090/__init__.py`

| 函数 | 作用 | 调用下游 |
|---|---|---|
| `gemm_naive(a, b)` | Python API；调用 naive FP32 GEMM | `_C.gemm_naive(a, b)` |
| `gemm_tiled(a, b)` | Python API；调用 shared-memory tiled GEMM | `_C.gemm_tiled(a, b)` |
| `gemm_regtile2x2(a, b)` | Python API；调用每线程 2x2 register tiling GEMM | `_C.gemm_regtile2x2(a, b)` |
| `softmax_row(x)` | Python API；调用按行 softmax | `_C.softmax_row(x)` |
| `layernorm_row(x, gamma, beta, eps=1e-5)` | Python API；调用按行 LayerNorm，并把 `eps` 转成 Python `float` | `_C.layernorm_row(x, gamma, beta, float(eps))` |
| `rmsnorm_row(x, gamma, eps=1e-6)` | Python API；调用按行 RMSNorm，并把 `eps` 转成 Python `float` | `_C.rmsnorm_row(x, gamma, float(eps))` |
| `attention_naive(q, k, v, causal=True)` | Python API；调用 naive attention，并把 `causal` 转成 `bool` | `_C.attention_naive(q, k, v, bool(causal))` |

这个文件只做薄包装。新增算子时，如果希望测试和 benchmark 能 `import aiop4090 as ops` 后直接调用，就要在这里加同名 Python 函数。

### 12.3 `src/aiop4090/csrc/common.h`

| 函数/宏 | 作用 | 开发关注点 |
|---|---|---|
| `CHECK_CUDA(x)` | 检查输入 tensor 在 CUDA 上 | 支持 CPU fallback 时要绕开或改逻辑 |
| `CHECK_CONTIGUOUS(x)` | 检查输入 tensor 是 contiguous | 支持 stride tensor 时要改 indexing 和检查 |
| `CHECK_FLOAT32(x)` | 检查 dtype 是 `torch.float32` | 支持 fp16/bf16 时需要扩展这里和 kernel |
| `CHECK_INPUT(x)` | 顺序执行 CUDA、contiguous、float32 三项检查 | 当前所有 host launcher 都依赖这个宏 |
| `ceil_div_int(int a, int b)` | 整数上取整除法，用于计算 grid 维度 | 改 block/tile 大小时常用 |

### 12.4 `src/aiop4090/csrc/bindings.cpp`

| 函数/声明 | 作用 | 开发关注点 |
|---|---|---|
| `torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B);` | 声明 GEMM naive host launcher | 声明必须和 `.cu` 里的定义签名一致 |
| `torch::Tensor gemm_tiled(torch::Tensor A, torch::Tensor B);` | 声明 tiled GEMM host launcher | 同上 |
| `torch::Tensor gemm_regtile2x2(torch::Tensor A, torch::Tensor B);` | 声明 2x2 register tiling GEMM host launcher | 同上 |
| `torch::Tensor softmax_row(torch::Tensor X);` | 声明 row-wise softmax host launcher | 同上 |
| `torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps);` | 声明 row-wise LayerNorm host launcher | 注意 `eps` 是 `double`，kernel 内转 `float` |
| `torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps);` | 声明 row-wise RMSNorm host launcher | 同上 |
| `torch::Tensor attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);` | 声明 naive attention host launcher | 注意 `causal` 是 C++ `bool` |
| `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` | PyTorch extension 的 Python 模块初始化入口；用 `m.def` 暴露函数 | 新增算子时必须在这里 `m.def("python_name", &cpp_func, "...")` |

`bindings.cpp` 是 Python `_C.xxx` 能否找到函数的关键。新增算子但 Python 报 `AttributeError` 时，优先查这里和是否重新 build。

### 12.5 `src/aiop4090/csrc/gemm.cu`

| 函数/kernel | 类型 | 作用 | 开发关注点 |
|---|---|---|---|
| `gemm_naive_kernel(...)` | CUDA kernel | 每个 thread 计算一个 `C[row, col]`，串行遍历 `K` 完成点积 | 最容易读懂，用于 baseline；性能很低 |
| `gemm_tiled_kernel(...)` | CUDA kernel | 每个 block 计算 `16x16` 输出 tile，把 A/B tile 载入 shared memory 后累加 | 优化 shared memory、padding、vectorized load 时改这里 |
| `gemm_regtile2x2_kernel(...)` | CUDA kernel | 每个 thread 计算相邻 `2x2` 输出，演示 register tiling | 扩展 `4x4/8x4` register tiling 时参考这里 |
| `check_gemm_inputs(const torch::Tensor& A, const torch::Tensor& B)` | C++ helper | 检查 A/B 是 CUDA contiguous float32、2D，且 `A.K == B.K` 对齐 | GEMM shape/dtype 报错时先看这里 |
| `gemm_naive(torch::Tensor A, torch::Tensor B)` | C++ host launcher | 解析 `M/N/K`、分配 `C`、配置 `16x16` block、launch `gemm_naive_kernel` | 改输出 shape、grid/block、stream/error check 时改这里 |
| `gemm_tiled(torch::Tensor A, torch::Tensor B)` | C++ host launcher | 配置 `TILE x TILE` block 和按 tile 上取整的 grid，launch `gemm_tiled_kernel` | 改 tile 大小后要同步 grid/block |
| `gemm_regtile2x2(torch::Tensor A, torch::Tensor B)` | C++ host launcher | 配置 `RT_TILE x RT_TILE` block，grid 覆盖 `RT_TILE*RN` by `RT_TILE*RM` 输出 | 改 `RM/RN/RT_TILE` 后必须同步 grid 计算 |

### 12.6 `src/aiop4090/csrc/softmax.cu`

| 函数/kernel | 类型 | 作用 | 开发关注点 |
|---|---|---|---|
| `softmax_row_kernel(...)` | CUDA kernel | 一个 block 处理一行，先规约 max，再规约 exp sum，最后写出 normalized probability | warp shuffle、online softmax、vectorized load 都主要改这里 |
| `softmax_row(torch::Tensor X)` | C++ host launcher | 检查输入 `[rows, cols]`，分配 `Y`，用 `rows` 个 block、每 block 256 threads launch kernel | 改 block size、shared memory 大小、shape 支持时改这里 |

### 12.7 `src/aiop4090/csrc/norm.cu`

| 函数/kernel | 类型 | 作用 | 开发关注点 |
|---|---|---|---|
| `layernorm_row_kernel(...)` | CUDA kernel | 一个 block 处理一行；规约 mean 和 variance，再写出 `norm * gamma + beta` | 优化规约、融合、fp16/bf16 支持时改这里 |
| `rmsnorm_row_kernel(...)` | CUDA kernel | 一个 block 处理一行；规约 `mean(x^2)`，再写出 `x * inv_rms * gamma` | RMSNorm 向量化加载和 dtype 支持主要改这里 |
| `layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps)` | C++ host launcher | 检查 X 是 2D、gamma/beta 是 `[cols]`，分配输出并 launch `layernorm_row_kernel` | shape/dtype/eps 行为和 block size 在这里控制 |
| `rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps)` | C++ host launcher | 检查 X 是 2D、gamma 是 `[cols]`，分配输出并 launch `rmsnorm_row_kernel` | 同上 |

### 12.8 `src/aiop4090/csrc/attention.cu`

| 函数/kernel | 类型 | 作用 | 开发关注点 |
|---|---|---|---|
| `attention_naive_kernel(...)` | CUDA kernel | 每个 block 计算一个 `O[b,h,i,d_out]`；对 key position 做 score、softmax、weighted sum | tiled attention、online softmax、FlashAttention 化都主要改这里或新增 kernel |
| `attention_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal)` | C++ host launcher | 检查 Q/K/V 是 `[B,H,S,D]` 且 shape 一致，分配 O，配置 `grid(D, S, B*H)` launch kernel | 支持 cross-attention、KV cache、不同 shape 时先改这里 |

### 12.9 `tests/test_gemm.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `_check_gemm(fn, m, n, k, tol=1e-3)` | GEMM correctness helper；构造随机 A/B，调用待测函数，与 `a @ b` 比较最大误差 | 新增 GEMM 版本时复用这个 helper |
| `test_gemm_naive_square()` | 测试 `ops.gemm_naive` 的 128 square shape | naive 基础正确性 |
| `test_gemm_tiled_square()` | 测试 `ops.gemm_tiled` 的 256 square shape | tiled 基础正确性 |
| `test_gemm_regtile2x2_square()` | 测试 `ops.gemm_regtile2x2` 的 256 square shape | register tiling 基础正确性 |
| `test_gemm_non_multiple_shape()` | 测试 tiled/regtile 在非 tile 整倍数 shape 下是否正确 | 修改边界处理时必须跑 |

### 12.10 `tests/test_softmax_norm.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `test_softmax_row()` | 构造 `[64,1024]` 输入，比较 `ops.softmax_row` 和 `torch.softmax` | softmax 数值稳定性和容差 |
| `test_layernorm_row()` | 构造 X/gamma/beta，比较 `ops.layernorm_row` 和 PyTorch `layer_norm` | mean/variance 规约和 affine 是否正确 |
| `test_rmsnorm_row()` | 构造 X/gamma，比较 `ops.rmsnorm_row` 和 Python reference | RMSNorm 公式、eps 和 broadcast 是否正确 |

### 12.11 `tests/test_attention.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `_ref_attention(q, k, v, causal: bool)` | PyTorch reference：`QK^T / sqrt(D)`、可选 causal mask、softmax、乘 V | 新 attention 版本都应先和它对拍 |
| `test_attention_naive_causal()` | 测试 causal `ops.attention_naive` | causal mask 和 softmax 范围 |
| `test_attention_naive_non_causal()` | 测试 non-causal `ops.attention_naive` | 非 causal 全量注意力 |

### 12.12 `benchmark/bench_ops.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `cuda_bench(fn, warmup=20, repeat=100)` | 通用 CUDA 计时 helper；warmup 后用 CUDA event 测平均 ms/op | benchmark 新算子时复用 |
| `bench_gemm()` | 跑 GEMM PyTorch baseline 和三种自定义 GEMM，输出 ms 和 TFLOP/s | 加 GEMM 新版本或新 shape 时改这里 |
| `bench_softmax_norm()` | 跑 softmax、LayerNorm、RMSNorm 的 PyTorch baseline 和自定义实现 | 加 norm/softmax 新版本时改这里 |
| `bench_attention()` | 跑 PyTorch scaled dot product attention 和 naive attention | 加 tiled/online attention benchmark 时改这里 |
| `main()` | 解析 `--op`，按 `all/gemm/softmax_norm/attention` 调用对应 benchmark | 新增 benchmark 分组时改 choices 和分支 |

### 12.13 `benchmark/bench_gemm_shapes.py`

| 函数/逻辑 | 作用 | 开发关注点 |
|---|---|---|
| 顶层 `shapes = [...]` | 定义 GEMM 专项 benchmark 的 shape 列表 | 加实际业务 shape 或压力测试 shape 时改这里 |
| 顶层双层 `for` 循环 | 为每个 shape 构造 A/B，并依次计时 `torch`、`naive`、`tiled`、`regtile2x2` | 加 GEMM 新版本时在 case 列表里增加一项 |
| `cuda_bench` 导入 | 复用 `bench_ops.py` 的 CUDA event 计时逻辑 | 计时逻辑统一改 `bench_ops.py` 即可 |

这个文件没有自定义 `def`，它作为脚本运行时直接执行顶层 benchmark 逻辑。

### 12.14 `benchmark/profile_entry.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `repeat(fn, iters=30)` | 先 warmup 5 次，再重复执行目标函数 `iters` 次并同步；用于给 Nsight 足够的 kernel launch | 不输出耗时，只服务 profiling |
| `main()` | 解析 `--op/--iters`，为指定 op 构造固定输入和 lambda，然后交给 `repeat` | 新增可 profile 算子时改 `choices` 和分支 |

### 12.15 `docs/online_softmax_reference.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `online_softmax_vector(x: torch.Tensor, block_size: int = 16)` | 单个 1D vector 的 online softmax reference；维护 running max `m` 和 running exp-sum `l` | 实现 online softmax attention 或 softmax kernel 时用来对照公式 |
| `if __name__ == "__main__": ...` | 构造随机向量，对比 online softmax 和 `torch.softmax` 的最大误差 | 算法推导自检入口，不参与 package 构建 |

### 12.16 `docs/kv_cache_reference.py`

| 函数 | 作用 | 开发关注点 |
|---|---|---|
| `decode_without_cache(xs, w_k, w_v)` | 每个 decode step 都重新计算历史 token 的 K/V，用作无 cache reference | 说明 KV cache 节省了哪些重复计算 |
| `decode_with_cache(xs, w_k, w_v)` | 每个 decode step 只计算当前 token 的 K/V 并 append 到 cache | 后续写 KV cache attention 时参考数据流 |
| `if __name__ == "__main__": ...` | 构造小例子，对比 cache 和 no-cache 每一步的 K/V 是否一致 | 算法概念自检入口 |

### 12.17 Shell 脚本入口

这些文件没有定义 shell 函数，但每个文件本身都是一个可执行流程入口：

| 文件 | 顶层逻辑 | 开发关注点 |
|---|---|---|
| `scripts/00_check_env.sh` | 依次打印 GPU、nvcc、ncu/nsys、Python/PyTorch CUDA 信息 | 环境排查第一步 |
| `scripts/10_build.sh` | 安装构建依赖并执行 editable install | 构建 extension |
| `scripts/20_test.sh` | 执行 `pytest -q tests` | correctness 一键测试 |
| `scripts/30_bench.sh` | 执行 `python benchmark/bench_ops.py --op all` | benchmark 一键入口 |
| `scripts/clean_build.sh` | 删除 build、dist、egg-info、`.so`、`__pycache__` | 修改 `.cu/.cpp` 后干净重编 |
| `scripts/profile_ncu.sh` | 用 Nsight Compute `--set speed-of-light` profile `benchmark/profile_entry.py --op OP` | 快速看核心性能指标 |
| `scripts/profile_ncu_full.sh` | 用 Nsight Compute `--set full` profile 单个 op | 需要完整指标时用 |
| `scripts/profile_nsys.sh` | 用 Nsight Systems profile 全量 `bench_ops.py` | 看整体时间线和 launch overhead |

### 12.18 无自定义函数的文档文件

这些文件不包含可调用函数，主要是说明、路线图或模板：

| 文件 | 作用 |
|---|---|
| `README.md` | 仓库总说明、环境、构建、测试、benchmark、已包含算子 |
| `docs/phase_map.md` | 分阶段开发路线和每阶段涉及文件 |
| `docs/report_template.md` | 性能 profiling 报告模板 |
| `docs/interview_resume.md` | 项目简历和面试表达材料 |
| `docs/code_relationship_map.md` | 当前这份代码关系和函数作用速查文档 |
