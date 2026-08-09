# 阶段 1 自学教材：从零交付最小 CUDA 算子

> 版本：2.0，自包含教材版。  
> 适用对象：有芯片软件开发经验，但第一次承担 AI 芯片 Kernel/算子任务的人。  
> 唯一目标：只依靠本教材的正文、代码、实验和答案，完成阶段 1 的两个独立交付。  
> 学习原则：先理解问题，再运行例子，随后关闭答案独立重写；验收通过就停止扩展。

## 教材使用说明

这不是资料索引，也不是“还需要自己找视频补完”的提纲。正文负责提供阶段 1 所需的：

- 概念解释和一行记忆法。
- 能直接运行的完整工程代码。
- 从输入 shape 到线程、地址、同步、精度和性能的推导。
- 正确性测试、benchmark、故障制造与定位方法。
- 练习题、预期现象和参考答案。
- 原阶段 1 表格的逐项验收映射。

正常情况下，不需要再找一本 CUDA 教材配合阅读。只有下面三类信息可能需要查看目标环境的官方说明：

1. 本机驱动、CUDA Toolkit 与 PyTorch 的具体版本组合。
2. 目标 GPU 的精确 SM、寄存器、shared memory 等硬件上限。
3. profiler 在不同版本中的指标名称变化。

这些是环境参数，不是缺失课程。正文会告诉你需要查什么、为什么查以及拿到结果后如何使用。

### 推荐学习方法

每一章严格执行四步：

```text
先读“它解决什么问题”
    ↓
自己完成章内手算
    ↓
运行教材工程并修改一个变量
    ↓
关闭教材，完成闭卷自查
```

第一次允许照着参考工程敲代码，目的是建立完整心智模型。阶段结束前必须另建空目录独立重写一次；独立重写才是最终成绩。

你已经学习过的内容不必从头重读。先直接做该章“闭卷自查”和实验：两者都能独立通过，就只记录证据并跳到下一章；任何一项只能讲概念却做不出来，再回读对应正文。这样教材既能给新人从零使用，也能供你按现状查漏补缺。

---

## 0. 学完后，你可以开始做什么

你将能够独立完成两个新人任务。

### 工单 A：最小 PyTorch CUDA Extension 算子

实现一个 `scaled_add(x, y, alpha)`：

```text
out[i] = x[i] + alpha * y[i]
```

它必须具备：

- Python API、C++ binding、CUDA launcher、CUDA kernel 和构建配置。
- FP32、FP16、BF16 输入；低精度输入使用合理的累加方式。
- device、dtype、shape、stride/contiguous、空输入和错误输入检查。
- 当前 PyTorch CUDA stream 语义，不能偷偷切到另一个 stream。
- PyTorch 参考实现、正确性测试、错误测试和可重复 benchmark。
- 一次构建/链接错误复盘和一次异步 CUDA 错误复盘。

### 工单 B：Warp/Block Reduction 实验

实现二维 Tensor 最后一维求和 `row_sum(x)`：

```text
输入：[rows, cols]
输出：[rows]
```

至少包含三条路径：

1. naive：作为最容易验证的正确性基线。
2. warp shuffle：一个 warp 内完成归约。
3. block reduction：多个 warp 合作完成一行归约。

它必须覆盖：

- `cols = 1、31、32、33`。
- 非 2 次幂、尾部元素、多个 block、大输入和空输入。
- FP32、FP16、BF16 的累加精度和容差。
- 不同 block size 的性能对比。
- 对同步次数、访存、occupancy 和寄存器压力的解释。

### 真正的完成标准

“看懂代码”不算完成。下面四件事必须同时成立：

1. **能解释**：不用术语堆砌，能从线程推到地址、同步和结果。
2. **能实现**：从空目录开始，不复制教程核心代码。
3. **能验证**：用测试和 benchmark 证明正确、稳定、可复现。
4. **能定位**：故意制造错误后，能判断错误位于哪一层。

一周后，在不看教程的情况下重新构建、测试并解释两个工单，仍然通过，阶段 1 才算结束。

---

## 1. 新人先记住这一张工作地图

一个 CUDA 算子不是只有 `.cu` 文件，而是一条完整调用链：

```text
Python 调用
  └─ 输入 Tensor 和参数
      ↓
C++ binding
  └─ 暴露 Python 接口、进入 C++ 世界
      ↓
C++/CUDA launcher
  ├─ 检查 device、dtype、shape、stride、空输入
  ├─ 选择模板类型和 launch 配置
  └─ 取得当前 device 与当前 CUDA stream
      ↓
CUDA kernel
  ├─ thread/block 找到自己负责的数据
  ├─ 从全局内存读数据
  ├─ 在 register/shared memory 中计算或归约
  └─ 写回全局内存
      ↓
异步返回 Python
  └─ 错误可能到后面的同步点才暴露
```

构建时还有另一条链：

```text
Python/C++ 头文件
      ↓ 编译
C++ object + CUDA object
      ↓ 链接
共享库 .so
      ↓ 动态加载
Python import
```

以后遇到问题，先判断它属于哪一层：

- import 失败：环境、动态库、ABI 或符号问题。
- 调用时报参数错误：API 契约或 binding/launcher 问题。
- 结果错误：索引、边界、同步、数据竞争或精度问题。
- 后续代码才报 CUDA 错误：异步执行问题。
- 结果正确但慢：访存、并行度、资源或测量方法问题。

**一句话记忆：先定位层，再定位行。**

---

## 2. 最短学习顺序

每个模块都必须按照“知道最少事实 → 马上动手 → 闭卷验收”的顺序完成。

| 顺序  | 最少知识模块             | 立即用于哪个工单 | 通过证据                      |
| :---: | :----------------------- | :--------------- | :---------------------------- |
| 1     | 开发环境、构建链和调用链 | 工单 A           | 从空目录成功 import 并运行    |
| 2     | CUDA 执行模型和索引      | 工单 A           | 手算线程、warp、尾块和地址    |
| 3     | 访存、片上存储和资源     | 工单 A/B         | 先预测两种访问模式，再实验    |
| 4     | Reduction、同步和边界    | 工单 B           | 1/31/32/33及大输入全部正确    |
| 5     | 数值、精度和算术强度     | 工单 A/B         | 定义累加类型、容差和误差报告  |
| 6     | Stream、异步执行和计时   | 工单 A/B         | 演示错误计时与正确计时的差异  |
| 7     | 输入契约、测试和调试     | 两个工单         | 六类测试和两次故障复盘        |
| 8     | Transformer 最小上下文   | 后续岗位衔接     | 能手算主要 shape、FLOPs/bytes |

建议工作量为 32～48 小时。每周能投入 8 小时时，可以把四周计划自然拉长，不要为了赶日历跳过验收。

---

## 3. 模块 1：环境、C++ 和完整构建链

### 3.1 这个模块解决什么工作问题

你接到一个算子任务后，要能从空目录创建工程；import 失败时，能区分 Python、编译、链接、动态库和 ABI 问题。

### 3.2 必须知道的最少事实

#### Linux/Python 环境

- Python 虚拟环境决定使用哪套 Python 包，但系统动态链接器还会决定加载哪一个 `.so`。
- `import` 大致经历“找到 Python 模块 → 加载共享库 → 解析依赖库 → 解析符号”。
- 环境变量会从父进程传给子进程；“终端能运行，IDE 不能运行”常常是两边环境不同。
- 一个进程可以有多个 CPU 线程；CUDA kernel 在设备上异步执行，不等同于一个新的 Linux 进程。
- 虚拟内存地址不是实际物理地址；CPU 内存、锁页内存和 GPU device memory 也不是一回事。
- 文件描述符是进程访问文件、管道和 socket 的编号。日志不输出、管道阻塞或端口占用时要想到它。
- signal 是操作系统通知进程的机制。普通终止、强制终止和崩溃不是同一种情况。

达到阶段 1 所需的 Linux 深度，不是背操作系统教材，而是能处理：进程异常、CPU/GPU 卡住、内存持续增长、僵死进程、日志/端口占用和环境不一致。

最小诊断命令表：

| 问题                   | 先运行什么                                        | 重点看什么                            |
| :--------------------- | :------------------------------------------------ | :------------------------------------ |
| 当前到底用了哪个Python | `which python`、`python -m pip -V`                | 两者是否指向同一虚拟环境              |
| PyTorch从哪里导入      | `python -c "import torch; print(torch.__file__)"` | 是否误用了另一套包                    |
| 进程是否存在/僵死      | `ps -ef`、`ps -o pid,ppid,stat,rss,cmd -p PID`    | `STAT`、父进程、RSS和命令             |
| CPU线程是否忙等        | `top -H -p PID`                                   | 哪个线程持续占用CPU                   |
| GPU进程和显存          | `nvidia-smi`                                      | PID、显存、利用率是否符合预期         |
| 动态库依赖             | `ldd extension.so`                                | 是否出现 `not found` 或加载了错误版本 |
| 端口被谁占用           | `lsof -i :PORT`                                   | 占用进程PID                           |
| 文件/管道未释放        | `lsof -p PID`                                     | fd数量是否持续增长                    |
| 温和终止进程           | `kill -TERM PID`                                  | 让进程有机会清理资源                  |

只有进程无响应且无法正常退出时才考虑更强制的终止方式。诊断时先记录PID和现象，不要用模糊匹配批量结束无关进程。

#### C++ 资源生命周期

- RAII：资源的生命周期绑定到对象生命周期；构造成功后由析构负责释放。
- 所有权必须明确：谁创建、谁持有、谁释放；异常路径也必须安全。
- 不在本阶段手写复杂智能指针框架，但必须看懂 `unique_ptr`、`shared_ptr` 和引用的基本所有权区别。
- PyTorch Tensor 自己管理底层存储。kernel 中使用的裸指针只是临时视图，不能超过 Tensor 生命周期。

最小RAII示意：

```cpp
{
    c10::cuda::CUDAGuard guard(x.device());
    // 这个作用域内当前device切到x所在device。
    // 无论正常return还是抛异常，离开作用域时都会恢复原device。
}
```

它回答了“异常发生时谁负责恢复device”。阶段1遇到文件、事件或临时资源，也使用同样思路：把释放动作绑定到对象析构，而不是要求每条return路径手工清理。

#### 模板和类型分派

- 模板代码只有在某个具体类型被使用时才会生成对应实例。
- “代码写在这里”不代表 `.so` 中一定存在对应符号；没有实例化或链接遗漏都可能导致 undefined symbol。
- Python dtype 最终必须映射到 C++/CUDA 的具体类型，再 launch 对应模板 kernel。
- 阶段 1 只要求能读懂并修改常见的 dtype dispatch，不要求学习模板元编程。

#### 编译、链接、ABI 和符号

- 编译：每个源文件被编译为 object，主要检查语法、类型和可见声明。
- 链接：把多个 object 和依赖库组合成 `.so`，解决“这个函数实现在哪里”。
- 动态加载：Python import 时，操作系统加载 `.so` 及其依赖并解析符号。
- ABI 是二进制层面的约定，包括函数名修饰、参数传递、对象布局和 C++ 标准库兼容性。
- 编译器、PyTorch、CUDA 或 C++ ABI 不匹配时，源码可能编译成功，但 import 时出现 undefined symbol。

### 3.3 工单 A 的最小目录

```text
stage1_scaled_add/
  ├─ setup.py 或 pyproject.toml
  ├─ scaled_add.cpp       # binding、输入检查、声明
  ├─ scaled_add_cuda.cu   # launcher、kernel
  ├─ test_scaled_add.py
  ├─ benchmark.py
  └─ README.md
```

你必须能指着每个文件回答：它在哪个阶段被使用、最终进入 `.so` 的是什么、Python 为什么能调用到 kernel。

### 3.4 马上动手

1. 从空目录建立工单 A，先只支持 FP32 contiguous Tensor。
2. 打印或记录 Python、PyTorch、CUDA、编译器和 GPU 环境。
3. 成功构建、import，并跑通一个 10 元素输入。
4. 删除或改名一个 launcher 实现，主动制造一次 undefined symbol。
5. 根据报错判断：声明是否存在、实现是否编译、object 是否链接、`.so` 是否导出符号、依赖库是否能找到。
6. 修复后写一页故障复盘：现象、错误层、证据、根因、修复和预防。

### 3.5 闭卷自查

- [ ] 能从 Python 调用一路画到 kernel。
- [ ] 能解释“编译成功但 import 失败”为什么可能发生。
- [ ] 能说明 RAII 如何保证异常路径也释放资源。
- [ ] 能解释模板为什么可能没有生成所需符号。
- [ ] 能定位 Python 包、动态库和 import 路径冲突。
- [ ] 新环境能根据 README 一键构建并运行测试。

### 3.6 停止线

暂不学习复杂 CMake、模板元编程、ELF 全部格式、CUDA Driver API 和自定义内存分配器。能独立构建并诊断一次符号/ABI问题就进入下一模块。

---

## 4. 模块 2：CUDA 执行模型、索引和资源限制

### 4.1 这个模块解决什么工作问题

给定输入 shape 和 block 配置，你能算清启动多少线程、每个线程访问什么地址、最后一个 block 浪费多少线程，以及 kernel 能否在硬件上启动。

### 4.2 必须知道的最少事实

#### 层次关系

```text
一次 kernel launch
  └─ grid：所有 block
      └─ block：一组可使用 shared memory 和 block 同步的线程
          └─ warp：硬件以通常 32 个线程为一组执行
              └─ thread：通过索引找到自己的数据
```

- block 被调度到 SM 上执行；一个 block 不会拆到多个 SM。
- 同一 SM 可以同时驻留多个 block，具体数量受线程、warp、register、shared memory 和硬件 block 上限共同限制。
- 普通 kernel 内没有跨 block 的全局 barrier，不能假定“block 0 先于 block 1”。

#### 一维索引

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
num_blocks = ceil(N / blockDim.x)
```

因为最后一个 block 往往不满，所以最基本的保护是：

```text
if idx < N:
    访问第 idx 个元素
```

二维/三维 block 的总线程数是各维度乘积。计算 warp 数时用总线程数：

```text
threads_per_block = blockDim.x * blockDim.y * blockDim.z
warps_per_block = ceil(threads_per_block / 32)
```

#### wave 和尾部效应

若每个 SM 同时能驻留 `resident_blocks_per_sm` 个 block：

```text
一次可并行驻留的 block 数
  = SM 数量 * resident_blocks_per_sm

waves
  = ceil(总 block 数 / 一次可并行驻留的 block 数)
```

最后一个 wave 只有少数 block 时，部分 SM 空闲，这就是 wave tail。它与最后一个 block 中线程越界造成的 thread tail 是两件事。

#### SIMT 与分支发散

- 同一个 warp 的线程执行同一条指令，但每个 lane 可以有自己的数据。
- 如果 warp 内部分线程走 `if`，另一部分走 `else`，两条路径通常需要分别执行，未参与的 lane 被屏蔽。
- 不同 warp 走不同分支不是 warp 内发散。
- 短小分支可能被编译成 predication，是否变慢必须用受控实验确认。

#### 资源上限

驻留 block 数近似由以下最小值决定：

```text
硬件 blocks/SM 上限
threads/SM 上限 ÷ threads/block
registers/SM ÷ (registers/thread × threads/block)
shared memory/SM ÷ shared memory/block
```

任何单项超出 kernel launch 上限都可能导致启动失败；没有超出也不代表配置性能最好。

Occupancy 是“当前活跃 warp 数 / 硬件允许的最大活跃 warp 数”。它反映隐藏延迟的能力，不是性能分数。Occupancy 高仍可能因为访存不合并、指令依赖或工作量过大而很慢。

### 4.3 马上动手

对工单 A 选择 `block_size = 256`，分别使用：

```text
N = 1、31、32、33、255、256、257、1000003
```

每个 N 在运行前手算：

- block 数。
- warp 数。
- 最后一个 block 的有效线程数。
- 最后一个 warp 的有效 lane 数。
- `idx` 和输入/输出地址关系。

再用 profiler 或 kernel 中的受控调试结果核对。随后比较 block size 64、128、256、512，记录资源和时间，但不要仅凭一次时间选“最佳值”。

### 4.4 闭卷自查

- [ ] 给定任意 N 和 block size，能手算 block、warp 和尾部。
- [ ] 能区分 thread tail 与 wave tail。
- [ ] 能解释 block 为什么不能跨 SM，以及为什么不能在 kernel 内直接跨 block 同步。
- [ ] 能根据分支条件判断是否发生 warp 内发散。
- [ ] 能解释 register/shared memory 为什么会限制驻留 block 数。
- [ ] 能解释 occupancy 高为什么仍可能慢。

### 4.5 停止线

暂不学习 cooperative groups、动态并行、MIG、线程块集群和架构级调度细节。能从 launch 配置推导执行形态和主要资源限制即可。

---

## 5. 模块 3：访存、片上存储、bank conflict 和 spill

### 5.1 这个模块解决什么工作问题

看到一条线程地址公式时，你能够预判它会产生多少无效流量；看到 shared memory 索引时，能够预判 bank conflict；看到寄存器增加时，知道为什么 occupancy 或性能可能下降。

### 5.2 必须知道的最少事实

| 存储层             | 谁能直接使用    | 新人必须记住的作用                             |
| :----------------- | :-------------- | :--------------------------------------------- |
| Register           | 单个 thread     | 最快，保存局部变量；过多会限制驻留或发生 spill |
| Shared memory      | 同一个 block    | 软件管理的片上复用区；需要正确同步             |
| L1/L2 cache        | 硬件管理        | 缓存全局/局部内存访问；不能把命中当成程序契约  |
| DRAM/global memory | grid 中所有线程 | 容量大、延迟高；访问模式决定有效带宽           |

#### 合并访问、transaction 和 sector

- warp 发出的线程访问会被硬件合并为一个或多个内存请求。
- 最有利的基本模式是：相邻 lane 访问相邻、正确对齐的数据。
- stride、错位或随机地址会让同样的有效数据触发更多 sector/transaction，产生多余流量。
- “代码只有一次 load”不代表硬件只搬了一份最少数据。

工单 A 中若 `lane k` 访问 `x[base + k]`，通常有利于合并；若访问 `x[base + k * stride]`，stride 增大通常会触碰更多内存区域。

#### Shared memory bank

常见教学模型是 32 个 bank、4 字节粒度。对 4 字节数据，可以先用：

```text
bank = (byte_address / 4) % 32
```

- 一个 warp 中多个线程访问不同地址但落到同一 bank，可能发生 bank conflict，访问被拆分。
- 多个线程读取完全相同地址通常可以广播，不能机械地判定为 conflict。
- 具体 bank 宽度和规则与架构、数据类型有关；工作中必须结合目标硬件文档和 profiler 验证。

#### Register pressure、local memory 和 spill

- 编译器会为线程局部变量分配寄存器。
- 活跃变量过多可能增加 registers/thread，从而减少 blocks/SM 或 occupancy。
- 寄存器不足时，部分变量可能 spill 到 local memory。
- local memory 是“线程私有的地址空间”，不代表物理上位于芯片本地；它通常需要经过 cache，最终可能访问设备内存。
- 强行降低寄存器数量可能减少 occupancy 限制，也可能增加 spill，必须测量总效果。

### 5.3 两个最小对照实验

#### 实验 1：合并访问

只改变地址模式，其他条件保持一致：

```text
模式 A：lane k 访问 base + k
模式 B：lane k 访问 base + k * 32
```

运行前写下预测：哪一种会触碰更多 sector、有效带宽为什么不同。运行后同时记录时间和请求/sector方向，不允许只看 Duration。

#### 实验 2：bank conflict

用一个 warp 访问二维 shared memory，比较：

```text
shared[32][32]
shared[32][33]
```

先根据索引公式手算 bank，再观察 shared conflict 指标。Padding 是否有效必须由访问方向决定，不能背诵“加 1 一定更快”。

### 5.4 Alignment 的最少知识

- `float4`、`half2` 等向量化访问通常要求地址和元素数量满足对齐条件。
- 快路径必须显式检查 alignment 和长度；不满足时走标量 fallback，并处理尾部。
- 工单 A 的基础版本可以全部走标量路径，因此不依赖额外 alignment；但必须增加一个很小的 FP32 `float4` 练习，验证“对齐快路径 + 非对齐/尾部安全回退”。

### 5.5 闭卷自查

- [ ] 能说清 register/shared/L1/L2/DRAM 的作用域和复用方式。
- [ ] 给定 warp 地址序列，能预测合并访问和多余流量方向。
- [ ] 给定 shared 索引，能按常见模型计算 bank。
- [ ] 能解释 register pressure 如何影响驻留 block。
- [ ] 能解释 local memory 为什么不一定快，以及如何从编译/profiler 信息发现 spill。
- [ ] 能为向量化路径设计 alignment 检查、尾部处理和标量 fallback。

### 5.6 停止线

暂不学习复杂 cache replacement、PTX cache hint、TMA、异步拷贝和手写汇编。能从地址和资源数据解释基本性能方向即可。

---

## 6. 模块 4：Reduction、同步、竞态和边界

### 6.1 这个模块解决什么工作问题

你能实现一个不会在 31、33 或大输入上悄悄算错的归约，并能说明每一次同步到底保护了什么。

### 6.2 必须知道的最少事实

#### Identity

Identity 是“没有有效输入时不改变结果”的初始值：

- sum：0。
- product：1。
- max：负无穷或该 dtype 的合理最小值。

尾部无效 lane 必须使用正确 identity，不能随便填 0；对 max 来说，0 会让全负数输入得到错误结果。

#### Warp shuffle

- shuffle 让一个 lane 读取同一 warp 中另一个 lane 的寄存器值。
- 它减少了 shared memory 读写和 block barrier，但只能在参与的 warp/lane 范围内成立。
- 非满 warp 必须使用正确 active mask，并确保读取的源 lane 有定义。
- 不要假定所有 32 个 lane 都包含有效输入。

#### Block reduction

最小可靠结构：

```text
每个 thread 累加自己负责的若干元素
    ↓
每个 warp 内 shuffle reduction
    ↓
各 warp leader 把部分和写入 shared memory
    ↓
整个 block 同步
    ↓
第一个 warp 归约所有 warp 的部分和
    ↓
写出本行结果
```

#### 同步范围

- warp 同步：只约束指定 warp 中参与的 lane。
- `__syncthreads()`：同一 block 所有参与线程到达 barrier，并提供相应 shared/global memory 可见性保证。
- host/device synchronize：等待设备工作完成，范围远大于 block barrier。
- 普通 kernel 内没有全 grid 同步。

绝不能让部分线程在 `__syncthreads()` 前提前 return，而其他线程继续到达 barrier；这可能导致死锁或未定义行为。边界线程应以 identity 参与，而不是提前离开。

#### 竞态

如果多个线程在没有正确同步或原子操作的情况下读写同一位置，结果取决于执行时序，这就是数据竞争。一次运行正确不能证明没有竞态。

### 6.3 工单 B 的实现顺序

1. 写 PyTorch reference：`x.sum(dim=-1)`。
2. 写 naive 版本，先让所有 shape 正确。
3. 写单 warp 版本，覆盖 1、31、32；再处理 33，观察为什么一个 warp 不够。
4. 写 block 版本，让多个 warp 先各自产生部分和，再用 shared memory 合并。
5. 每个 thread 用循环处理多个列，支持任意 cols 和大输入。
6. grid 中每个 block 处理一行；使用多行输入验证多个 block 独立执行。
7. 对 64、128、256、512 threads/block 做受控对照。

### 6.4 必测 shape

```text
rows：0、1、2、17、1024
cols：0、1、2、31、32、33、63、64、65、127、255、256、257、1000、4097
```

空维度的输出语义必须事先定义并与 PyTorch reference 对齐。

### 6.5 必做故障练习

主动制造一次越界访问：去掉尾部判断，让某个非整除 shape 触发 illegal memory access。然后：

1. 记录最早出现异常的调用位置。
2. 在调试边界增加同步或错误检查，让异步错误在更接近根因的位置暴露。
3. 使用最小失败 shape 缩小范围。
4. 检查索引、分配大小、launch 配置和 stream。
5. 修复后增加回归测试。
6. 移除 benchmark 热路径中不必要的同步。

### 6.6 闭卷自查

- [ ] 能从空文件写出 warp 和 block reduction 核心结构。
- [ ] 能解释每个 barrier 保护的写入与读取。
- [ ] 能处理非 2 次幂、非满 warp、尾部和大输入。
- [ ] 能为 sum 和 max 选择正确 identity。
- [ ] 能解释为什么部分线程在 barrier 前 return 是错误的。
- [ ] 能定位一次竞态、越界或异步错误，并留下回归测试。

### 6.7 停止线

阶段 1 不要求跨 block 单次 kernel 全局归约、cooperative launch、原子性能优化和 online softmax。先把一个 warp、多个 warp、一个 block 的正确性做扎实。

---

## 7. 模块 5：数值精度、容差和算术强度

### 7.1 这个模块解决什么工作问题

你能回答“结果差多少算错误”“低精度为什么溢出或失真”“这个算子理论上更可能受算力还是带宽限制”。

### 7.2 四种格式的最少知识

| 格式 | 动态范围      | 有效精度  | 阶段 1 必须记住的用途                                         |
| :--- | :------------ | :-------- | :------------------------------------------------------------ |
| FP32 | 大            | 较高      | reference、输出或累加的常用基准                               |
| TF32 | 接近 FP32     | 低于 FP32 | NVIDIA Tensor Core 的一种计算路径，不是普通 Tensor 存储 dtype |
| FP16 | 明显小于 FP32 | 高于 BF16 | 容易溢出；常用 FP32 累加                                      |
| BF16 | 接近 FP32     | 低于 FP16 | 不易因范围溢出，但舍入误差更大；常用 FP32 累加                |

阶段 1 的默认策略：FP16/BF16 输入在 reduction 中先转换为 FP32 累加，再根据 API 契约决定输出 dtype。必须写在代码和 README 中，不能让调用者猜。

### 7.3 容差和误差报告

常用逐元素判断：

```text
abs(out - ref) <= atol + rtol * abs(ref)
```

- `atol` 处理参考值接近 0 时的绝对误差。
- `rtol` 处理参考值较大时的相对误差。
- 容差必须根据 dtype、累加方式和输入规模设定，不允许为了让测试通过无限放宽。

每个算子至少报告：

- max absolute error。
- max relative error，并对 reference 接近 0 的位置单独处理。
- mean absolute error 或一个分位数误差。
- 是否出现 NaN/Inf。
- 最坏误差对应的 shape、dtype 和输入范围。

### 7.4 极值、NaN 和 Inf

必须覆盖：

- 全 0、全 1、正负混合。
- 很小值、很大值和大量累加。
- FP16 接近范围边界的输入。
- 明确包含 NaN/Inf 的输入，并定义是传播、报错还是与 PyTorch 对齐。
- reduction 中顺序不同导致的浮点加法误差。

浮点加法不满足严格结合律，因此并行 reduction 与串行 reference 最末位不同并不自动等于错误；但误差必须被量化和解释。

### 7.5 Softmax 数值稳定性的最低要求

即使阶段 1 不实现 Softmax，也必须会解释 max trick：

```text
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

减去最大值不改变数学结果，却能避免 `exp(大正数)` 溢出。你需要知道 Softmax 同时包含 max reduction 和 sum reduction；online softmax 的状态合并推导留到阶段 2。

### 7.6 FLOPs、bytes 和 Arithmetic Intensity

```text
Arithmetic Intensity = FLOPs / 最低数据搬运 bytes
```

判断方法：

- 强度低：每搬很多数据只做很少计算，通常更可能 memory-bound。
- 强度高：数据复用多、每个 byte 做很多计算，更可能 compute-bound。
- 这只是理论方向；实际还可能受 launch、并行度、依赖、资源和实现质量限制。

对工单 A 的每个元素，先估算读取 `x、y`、写 `out` 的最低 bytes，再估算乘加 FLOPs。对工单 B，估算读取所有元素和写每行结果的最低 bytes。然后与 profiler 的实际流量方向比较。

### 7.7 闭卷自查

- [ ] 能解释 FP32、TF32、FP16、BF16 的范围、精度和常见硬件路径差异。
- [ ] 能为低精度 reduction 明确 accumulation dtype。
- [ ] 能根据 dtype 和输入规模设置并解释 atol/rtol。
- [ ] 能报告误差分布，而不是只写 `allclose=True`。
- [ ] 能解释 max trick，以及 NaN/Inf 和极值测试的意义。
- [ ] 能估算一个小算子的 FLOPs、最低 bytes 和算术强度。

### 7.8 停止线

暂不学习完整数值分析、FP8/INT8量化算法、随机舍入和所有 Tensor Core 指令。阶段 1 只要求能为当前算子选择累加类型、容差并判断理论瓶颈方向。

---

## 8. 模块 6：Stream、异步执行和可信计时

### 8.1 这个模块解决什么工作问题

你不会因为 CPU 只测到 launch 时间而声称 kernel 很快，也不会因为使用了错误 stream 导致结果时好时坏。

### 8.2 必须知道的最少事实

- CUDA kernel launch 通常对 CPU 异步：CPU 提交工作后可以立即继续。
- 同一 stream 内的工作按序执行；不同 stream 之间除非建立依赖，否则不能假定顺序。
- PyTorch 算子必须在调用线程当前选择的 CUDA stream 上执行，不应硬编码旧式默认 stream。
- 当前 device 和 Tensor device 必须一致；多卡环境尤其要显式处理 device guard。
- 隐式同步会破坏并发，也会污染 benchmark。
- 异步错误可能在后面的同步、拷贝甚至另一个 API 调用处才暴露。

### 8.3 错误计时与正确计时

#### 错误示范

```text
CPU start
launch kernel
CPU end
```

这常常只测到 CPU launch 开销，没有等待 GPU 完成。

#### CUDA Event 计时

```text
warmup 多次
record start event on current stream
重复 launch
record end event on the same stream
等待 end event 完成
读取 elapsed time
```

CUDA Event 适合测同一设备 stream 上的 GPU elapsed time。CPU wall time适合测完整端到端调用，包括 Python、调度、同步和其他开销。两者回答的问题不同。

### 8.4 最小 benchmark 协议

每次性能报告固定并记录：

- GPU、驱动、CUDA、PyTorch 和编译配置。
- shape、dtype、stride、输入范围。
- block size 和实现版本。
- warmup 次数、正式重复次数和同步位置。
- median、P90、P95、最小/最大或波动范围。
- PyTorch baseline，并确保采用相同输入和相同统计方法。
- 每次实验只改变一个主要变量，运行前先写预测。

### 8.5 必做实验

对同一 kernel 做三次计时：

1. CPU wall time，不同步。
2. CPU wall time，计时区间末尾同步。
3. 当前 stream 上的 CUDA Event。

解释三者为什么不同；增加后台工作或多个连续 launch，观察差异如何变化。最终 benchmark 同时保留 GPU kernel 时间和端到端时间，但不能混为同一个指标。

### 8.6 闭卷自查

- [ ] 能解释 launch 异步和错误延迟暴露。
- [ ] 能区分默认 stream、当前 stream 和不同 stream 的顺序关系。
- [ ] 算子能在 PyTorch 当前 stream 和正确 device 上执行。
- [ ] 能选择 CUDA Event 或 CPU wall time回答对应问题。
- [ ] benchmark 有 warmup、同步、重复统计和固定环境。
- [ ] 能解释 median/P90/P95 和异常值，而不是只报一次最好成绩。

### 8.7 停止线

阶段 1 不要求实现传输计算重叠、多 stream pipeline、CUDA Graph 和多 GPU 通信。先保证 stream 语义正确、错误可定位、计时可信。

---

## 9. 模块 7：API 契约、测试分层、调试和可复现

### 9.1 这个模块解决什么工作问题

其他人能够安全调用你的算子；错误输入会明确失败；边界输入不会悄悄产生错误结果；换一台环境仍能构建和验证。

### 9.2 输入契约必须明确

对两个工单逐项写清：

- **device**：只允许 CUDA Tensor，还是提供 CPU fallback；多个输入必须在同一 device。
- **dtype**：支持哪些 dtype；混合 dtype 是拒绝还是转换。
- **shape**：维度数、各维关系和允许的空维度。
- **stride/contiguous**：支持任意 stride，还是明确要求 contiguous；不能默默按 contiguous 地址解释非连续 Tensor。
- **alignment**：标量路径无额外要求；向量快路径不满足对齐时必须安全 fallback。
- **empty input**：返回什么 shape 和 dtype，是否 launch kernel。
- **错误语义**：错误类型和信息应指出哪个条件不满足。
- **device/stream**：输出在哪个 device，工作提交到哪个 stream。

### 9.3 六类最低测试

| 测试类型 | 必须回答的问题                         | 最小样例                           |
| :------- | :------------------------------------- | :--------------------------------- |
| 正常测试 | 常见输入是否与 reference 一致          | 常用 shape、三种 dtype             |
| 边界测试 | 线程、warp、block 和空输入边界是否正确 | 0/1/31/32/33/255/256/257           |
| 随机测试 | 是否只在人工样例上正确                 | 随机 shape、值和多次种子           |
| 极值测试 | 溢出、下溢、NaN/Inf和累加误差如何      | 大小值、正负混合、NaN/Inf          |
| 错误测试 | 非法输入是否明确失败                   | CPU、错 dtype、错 shape、跨 device |
| 回归测试 | 曾经出现的故障是否永久被拦截           | undefined symbol、尾部越界对应样例 |

除此之外建立 shape matrix：

```text
业务正常 shape
边界 shape
奇数/非整除 shape
很小 shape
大输入/压力 shape
非连续或错误 stride
对齐和非对齐输入
```

### 9.4 测试层次

- unit：小函数、shape推导、参数检查。
- correctness：算子与 PyTorch reference 对比。
- property：例如输入交换、线性关系或 shape不变量等性质。
- regression：固定住曾失败的输入和错误。
- performance：只在正确性通过后运行，检测明显退化。

性能变快不能抵消正确性失败；正确性通过也不能证明没有不稳定竞态，所以随机和重复执行都需要。

### 9.5 一次完整调试流程

```text
固定失败输入和随机种子
    ↓
判断错误层：构建 / API / launch / kernel / async / numerical
    ↓
缩小到最小失败 shape
    ↓
在边界增加检查或同步，让错误靠近根因暴露
    ↓
一次只验证一个假设
    ↓
修复后增加回归测试
    ↓
恢复正式异步语义并重跑全套测试
```

### 9.6 README 和复现要求

README 最少包含：

1. 支持的环境、GPU 和 dtype。
2. 一条构建命令。
3. 一条运行正确性测试的命令。
4. 一条运行 benchmark 的命令。
5. 输入契约和暂不支持项。
6. 已验证环境和没有验证的边界。
7. 两次故障复盘的链接或摘要。

### 9.7 闭卷自查

- [ ] 能在写 kernel 前先定义输入契约。
- [ ] 能覆盖 dtype、shape、stride、device、alignment、空输入和错误语义。
- [ ] 有正常、边界、随机、极值、错误、回归六类测试。
- [ ] 能把一次故障缩小为最小失败 shape 并增加回归测试。
- [ ] 一键构建、测试和 benchmark 能在干净环境复现。
- [ ] 文档明确写出已验证与未验证内容，不夸大结果。

### 9.8 停止线

暂不建设大型 CI 集群、复杂发布系统和完整跨平台矩阵。阶段 1 只要求一个干净环境能够一键构建测试，并诚实记录 GPU 验收边界。

---

## 10. 模块 8：Transformer 和推理所需的最小数学上下文

这一模块不是为了让你转做算法，而是让你知道 Kernel 在真实模型中的 shape、工作量和瓶颈。

### 10.1 统一符号

```text
B：batch size
S：sequence length
H：hidden size
heads：attention head 数
D：每个 head 的维度，D = H / heads
I：FFN intermediate size
X：[B, S, H]
M：把 batch 和 sequence 展平后，M = B * S
```

### 10.2 主要 GEMM shape 和 FLOPs

矩阵乘 `A[M,K] × W[K,N] → C[M,N]` 的常用 FLOPs 估算：

```text
FLOPs ≈ 2 * M * N * K
```

Transformer block 中至少会算：

| 操作              | 矩阵 shape           | 近似 FLOPs                |
| :---------------- | :------------------- | :------------------------ |
| QKV projection    | `[BS,H] × [H,3H]`    | `2 × BS × H × 3H`         |
| Output projection | `[BS,H] × [H,H]`     | `2 × BS × H²`             |
| QKᵀ               | 每头 `[S,D] × [D,S]` | `2 × B × heads × S² × D`  |
| Attention × V     | 每头 `[S,S] × [S,D]` | `2 × B × heads × S² × D`  |
| FFN up/gate       | `[BS,H] × [H,I]`     | 每个投影 `2 × BS × H × I` |
| FFN down          | `[BS,I] × [I,H]`     | `2 × BS × I × H`          |

最低 bytes 至少包含必须读取的输入和权重、必须写出的输出；如果中间结果不能复用、发生重复读取或额外 layout 转换，实际 bytes 会更高。

### 10.3 必做推导

任选一组真实参数，例如：

```text
B = 1
S = 2048
H = 4096
heads = 32
I = 11008
```

闭卷写出：

1. D。
2. Q、K、V 和 attention score 的 shape。
3. QKV、QKᵀ、Attention×V、FFN 的 FLOPs。
4. 每个操作至少需要读写哪些 Tensor。
5. 粗略算术强度和预期 compute-bound/memory-bound 方向。
6. FP16/BF16 输入为什么通常使用 FP32 累加，以及 Tensor Core 可能出现在哪类矩阵运算中。

### 10.4 停止线

阶段 1 不学习完整 Transformer 训练、反向传播、FlashAttention、KV Cache、量化和 vLLM 源码。能从 B/S/H/heads 推到 shape、FLOPs、bytes、精度和瓶颈方向即可。

---

## 11. 教材配套工程：完整可运行参考实现

前十章解释了“为什么”。这一章给出“怎样做”的完整参考工程。第一次学习时逐行敲入并运行；完成教材后关闭本章，从空目录独立重写。

> 验证边界：本文档结构、表格和Python代码块已经做静态检查；当前文档编写环境没有安装PyTorch/CUDA，C++/CUDA扩展尚未在本机实际编译。第一次在目标GPU服务器构建时，应把编译结果、GPU型号和软件版本记录到学习日志；若发生版本接口差异，按模块1的构建链方法定位。

### 11.1 开始前只检查五件事

在 Linux GPU 环境中运行：

```bash
nvidia-smi
nvcc --version
python3 --version
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

你需要得到：

- `nvidia-smi` 能看到 GPU，说明驱动基本可用。
- `nvcc` 存在，说明能够编译 `.cu` 文件。
- Python 能 import 带 CUDA 的 PyTorch。
- `torch.cuda.is_available()` 为 True。

如果第四或第五条失败，先不要改 kernel。此时问题还在环境层：记录驱动版本、`nvcc`版本、PyTorch版本和`torch.version.cuda`，确认当前 shell 与安装 PyTorch 的虚拟环境一致。不要为了试错随意更换系统驱动。

### 11.2 创建目录

```bash
mkdir -p stage1_cuda_ops/src stage1_cuda_ops/tests
cd stage1_cuda_ops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel pytest
```

如果当前虚拟环境还没有 CUDA 版 PyTorch，应安装与服务器约定一致的版本。PyTorch、CUDA wheel 和驱动组合会随环境变化，本教材不把某一个版本硬编码成永久答案。

最终目录：

```text
stage1_cuda_ops/
  ├─ setup.py
  ├─ src/
  │   ├─ bindings.cpp
  │   └─ kernels.cu
  ├─ tests/
  │   └─ test_ops.py
  └─ benchmark.py
```

### 11.3 构建文件 `setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="stage1_cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="stage1_cuda_ops",
            sources=["src/bindings.cpp", "src/kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

逐行解释：

- `CUDAExtension` 告诉 PyTorch：同时使用 C++ 编译器和 `nvcc` 构建扩展。
- 两个 source 分别生成 object，最后链接为同一个 Python 可加载共享库。
- `-O3` 开启优化；`-lineinfo` 保留 profiler 映射所需行信息。
- 阶段 1 不启用 `--use_fast_math`，避免一开始把性能变化和数值近似混在一起。
- `BuildExtension` 把 setuptools 的构建命令连接到 PyTorch 扩展构建流程。

### 11.4 Python/C++ 绑定 `src/bindings.cpp`

```cpp
#include <torch/extension.h>

torch::Tensor scaled_add_cuda(
    torch::Tensor x,
    torch::Tensor y,
    double alpha);

torch::Tensor row_sum_naive_cuda(torch::Tensor x);
torch::Tensor row_sum_warp_cuda(torch::Tensor x);
torch::Tensor row_sum_block_cuda(torch::Tensor x, int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_add", &scaled_add_cuda,
          "scaled_add: out = x + alpha * y (CUDA)");
    m.def("row_sum_naive", &row_sum_naive_cuda,
          "row-wise sum, serial reference kernel (CUDA)");
    m.def("row_sum_warp", &row_sum_warp_cuda,
          "row-wise sum, one warp per row (CUDA)");
    m.def("row_sum_block", &row_sum_block_cuda,
          "row-wise sum, one block per row (CUDA)");
}
```

这里没有 kernel 实现，只完成两件事：

1. 声明 C++ 世界中存在这些函数。
2. 把函数名注册为 Python 模块方法。

如果声明与 `.cu` 中的定义在函数名、参数或命名空间上不一致，编译可能通过，但链接或 import 时会找不到对应符号。

### 11.5 CUDA 实现 `src/kernels.cu`

下面代码故意保持“新人可解释”，没有追求极致性能。

```cpp
#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


#define CHECK_CUDA(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")


static bool is_supported_dtype(const torch::Tensor& x) {
    return x.scalar_type() == at::kFloat ||
           x.scalar_type() == at::kHalf ||
           x.scalar_type() == at::kBFloat16;
}


static void check_scaled_add_inputs(
    const torch::Tensor& x,
    const torch::Tensor& y) {
    CHECK_CUDA(x);
    CHECK_CUDA(y);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(y);
    TORCH_CHECK(x.device() == y.device(),
                "x and y must be on the same device");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "x and y must have the same dtype");
    TORCH_CHECK(x.sizes() == y.sizes(),
                "x and y must have the same shape");
    TORCH_CHECK(is_supported_dtype(x),
                "supported dtypes: float32, float16, bfloat16");
}


static void check_row_sum_input(const torch::Tensor& x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
    TORCH_CHECK(is_supported_dtype(x),
                "supported dtypes: float32, float16, bfloat16");
}


template <typename scalar_t>
__global__ void scaled_add_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    int64_t n,
    double alpha) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        using acc_t = at::acc_type<scalar_t, true>;
        acc_t xv = static_cast<acc_t>(x[idx]);
        acc_t yv = static_cast<acc_t>(y[idx]);
        acc_t av = static_cast<acc_t>(alpha);
        out[idx] = static_cast<scalar_t>(xv + av * yv);
    }
}


template <typename scalar_t>
__global__ void row_sum_naive_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols) {
    int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            sum += static_cast<float>(x[row * cols + col]);
        }
        out[row] = static_cast<scalar_t>(sum);
    }
}


__device__ __forceinline__ float warp_reduce_sum(float value) {
    constexpr unsigned int full_mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(full_mask, value, offset);
    }
    return value;
}


template <typename scalar_t>
__global__ void row_sum_warp_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols) {
    int64_t row = blockIdx.x;
    int lane = threadIdx.x;

    float local = 0.0f;
    for (int64_t col = lane; col < cols; col += 32) {
        local += static_cast<float>(x[row * cols + col]);
    }

    float sum = warp_reduce_sum(local);
    if (lane == 0) {
        out[row] = static_cast<scalar_t>(sum);
    }
}


template <typename scalar_t>
__global__ void row_sum_block_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols) {
    __shared__ float warp_sums[32];

    int64_t row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (blockDim.x + 31) / 32;

    float local = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        local += static_cast<float>(x[row * cols + col]);
    }

    float warp_sum = warp_reduce_sum(local);
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        float block_value = lane < num_warps ? warp_sums[lane] : 0.0f;
        float block_sum = warp_reduce_sum(block_value);
        if (lane == 0) {
            out[row] = static_cast<scalar_t>(block_sum);
        }
    }
}


torch::Tensor scaled_add_cuda(
    torch::Tensor x,
    torch::Tensor y,
    double alpha) {
    check_scaled_add_inputs(x, y);

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) {
        return out;
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "scaled_add_cuda",
        [&] {
            scaled_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                n,
                alpha);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor row_sum_naive_cuda(torch::Tensor x) {
    check_row_sum_input(x);

    int64_t rows = x.size(0);
    int64_t cols = x.size(1);
    if (rows == 0) {
        return torch::empty({0}, x.options());
    }
    if (cols == 0) {
        return torch::zeros({rows}, x.options());
    }

    auto out = torch::empty({rows}, x.options());
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;
    int blocks = static_cast<int>((rows + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "row_sum_naive_cuda",
        [&] {
            row_sum_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows,
                cols);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor row_sum_warp_cuda(torch::Tensor x) {
    check_row_sum_input(x);

    int64_t rows = x.size(0);
    int64_t cols = x.size(1);
    if (rows == 0) {
        return torch::empty({0}, x.options());
    }
    if (cols == 0) {
        return torch::zeros({rows}, x.options());
    }

    auto out = torch::empty({rows}, x.options());
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "row_sum_warp_cuda",
        [&] {
            row_sum_warp_kernel<scalar_t><<<rows, 32, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows,
                cols);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor row_sum_block_cuda(torch::Tensor x, int64_t threads) {
    check_row_sum_input(x);
    TORCH_CHECK(threads >= 32 && threads <= 1024,
                "threads must be in [32, 1024]");
    TORCH_CHECK(threads % 32 == 0,
                "threads must be a multiple of 32");

    int64_t rows = x.size(0);
    int64_t cols = x.size(1);
    if (rows == 0) {
        return torch::empty({0}, x.options());
    }
    if (cols == 0) {
        return torch::zeros({rows}, x.options());
    }

    auto out = torch::empty({rows}, x.options());
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "row_sum_block_cuda",
        [&] {
            row_sum_block_kernel<scalar_t><<<
                rows, static_cast<int>(threads), 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    rows,
                    cols);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
```

### 11.6 逐段读懂，而不是整段背诵

#### 输入检查为什么放在 launcher

Python 只负责发起调用，真正的内存地址解释发生在 C++/CUDA。若把非连续 Tensor 当作连续数组访问，代码可能不越界，却会按错误地址读取。因此，当前教材选择“明确拒绝非连续输入”，而不是偷偷调用 `.contiguous()`：调用者能看到真实的数据复制成本。

#### 为什么使用 device guard 和当前 stream

`CUDAGuard` 临时把当前 CUDA device 切到输入所在设备，退出函数时自动恢复，这就是 RAII。`getCurrentCUDAStream()`取得 PyTorch 当前上下文选择的 stream，保证前序操作与本算子保持调用者期望的顺序。

#### 为什么 launch check 后不立即 synchronize

launch check 能捕获非法配置等启动错误，但不等待 kernel 完成。正式算子保持异步，才能与 PyTorch 其余工作正确流水。调试非法访存时，可以在临时调试边界同步；找到问题后应移除热路径中的额外同步。

#### 为什么 Reduction 统一用 FP32 局部累加

FP16/BF16 每次加法都舍入会快速积累误差。转换成 FP32 后在寄存器中累加，再转换回输出 dtype，是阶段 1 可以解释且常见的策略。输出仍是低精度，因此最终还有一次舍入。

#### 为什么 block 版本只需要一个 barrier

每个 warp 内先用 shuffle 完成寄存器归约；只有 warp leader 写 shared memory。第一个 warp 读取这些部分和之前，必须保证所有 warp 已经完成写入，所以这里需要一次 block barrier。之后只有第一个 warp 工作，不再需要整个 block 同步。

### 11.7 构建与第一次运行

```bash
python -m pip install -v -e .
python -c "import stage1_cuda_ops as ops; print(dir(ops))"
```

第一次运行：

```bash
python - <<'PY'
import torch
import stage1_cuda_ops as ops

x = torch.arange(10, device="cuda", dtype=torch.float32)
y = torch.ones_like(x)
out = ops.scaled_add(x, y, 2.0)
print(out)
print(torch.equal(out, x + 2.0 * y))

m = torch.arange(35, device="cuda", dtype=torch.float32).reshape(5, 7)
print(ops.row_sum_block(m, 128))
print(m.sum(dim=-1))
PY
```

预期：两个结果分别与 PyTorch reference 一致。第一次成功后先不要优化，立即进入测试章，把“看起来正确”升级为“有证据正确”。

### 11.8 访存与bank实验 `memory_lab.cu`

模块3要求“先预测，再用最小实验验证”。下面是完整独立实验，不依赖PyTorch。保存为工程根目录的`memory_lab.cu`：

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>


#define CUDA_CHECK(call) do {                                      \
    cudaError_t error = (call);                                    \
    if (error != cudaSuccess) {                                    \
        std::fprintf(stderr, "%s:%d CUDA error: %s\n",             \
                     __FILE__, __LINE__, cudaGetErrorString(error));\
        std::exit(1);                                               \
    }                                                              \
} while (0)


__global__ void copy_contiguous(
    const float* input,
    float* output,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}


__global__ void copy_strided(
    const float* input,
    float* output,
    int n,
    int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * stride];
    }
}


__global__ void shared_bank_pattern(
    float* output,
    int stride,
    int iterations) {
    volatile __shared__ float shared[32 * 33];
    int lane = threadIdx.x;
    int position = lane * stride;
    float value = static_cast<float>(lane);

    for (int i = 0; i < iterations; ++i) {
        shared[position] = value;
        __syncwarp();
        value = shared[position] + 1.0f;
        __syncwarp();
    }
    output[lane] = value;
}


template <typename Function>
float benchmark_ms(Function launch, int warmup = 20, int repeats = 100) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaGetLastError());
    return total_ms / repeats;
}


int main() {
    constexpr int n = 1 << 20;
    constexpr int stride = 32;
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float* input = nullptr;
    float* output = nullptr;
    float* bank_output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, sizeof(float) * n * stride));
    CUDA_CHECK(cudaMalloc(&output, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&bank_output, sizeof(float) * 32));
    CUDA_CHECK(cudaMemset(input, 0, sizeof(float) * n * stride));

    float contiguous_ms = benchmark_ms([&] {
        copy_contiguous<<<blocks, threads>>>(input, output, n);
    });
    float strided_ms = benchmark_ms([&] {
        copy_strided<<<blocks, threads>>>(input, output, n, stride);
    });

    float conflict_ms = benchmark_ms([&] {
        shared_bank_pattern<<<1, 32>>>(bank_output, 32, 4096);
    });
    float padded_ms = benchmark_ms([&] {
        shared_bank_pattern<<<1, 32>>>(bank_output, 33, 4096);
    });

    std::printf("contiguous copy: %.6f ms\n", contiguous_ms);
    std::printf("stride-32 copy:  %.6f ms\n", strided_ms);
    std::printf("bank stride 32:  %.6f ms\n", conflict_ms);
    std::printf("bank stride 33:  %.6f ms\n", padded_ms);

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(bank_output));
    return 0;
}
```

编译运行：

```bash
nvcc -O3 -lineinfo memory_lab.cu -o memory_lab
./memory_lab
```

预期方向：

- contiguous访问通常明显优于stride-32，因为warp触碰的内存区域更集中。
- shared stride 32让不同lane的不同地址落到同一个bank；stride 33把lane分散到不同bank，通常更快。
- 时间差的具体倍数不做标准答案，因为cache、时钟和GPU架构会改变绝对值。

若环境安装了Nsight Compute，可运行：

```bash
ncu --set full ./memory_lab
```

在对应kernel中比较实际global memory sectors/requests和shared bank conflict方向。不同NCU版本的指标名可能变化，所以先按kernel名称定位，再搜索带`sector`、`request`、`bank conflict`含义的指标；不要把一个版本的完整指标名背成知识点。

### 11.9 Alignment快路径与安全fallback

向量化不是阶段1主项目，但必须理解“条件不满足时不能硬跑”。下面是FP32的最小模式：

```cpp
__global__ void scaled_add_float4_kernel(
    const float4* x,
    const float4* y,
    float4* out,
    int64_t n4,
    float alpha) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 xv = x[idx];
        float4 yv = y[idx];
        out[idx] = make_float4(
            xv.x + alpha * yv.x,
            xv.y + alpha * yv.y,
            xv.z + alpha * yv.z,
            xv.w + alpha * yv.w);
    }
}
```

launcher只有在以下条件全部满足时才能选它：

```cpp
bool aligned16 =
    reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0;

bool can_use_float4 =
    x.scalar_type() == at::kFloat &&
    x.numel() % 4 == 0 &&
    aligned16;

if (can_use_float4) {
    // 把float*解释为float4*，处理n/4个向量。
} else {
    // 走本教材已经实现的标量kernel，安全处理非对齐和尾部。
}
```

测试fallback时不要只改变元素数量，还要制造有storage offset的连续视图：

```python
base_x = torch.randn(1025, device="cuda")
base_y = torch.randn(1025, device="cuda")
x = base_x[1:1025]
y = base_y[1:1025]
assert x.is_contiguous()
# x的首地址相对base_x偏移4 bytes，通常不满足16-byte alignment。
# 算子必须走标量fallback并保持结果正确。
```

完成标准不是“float4一定更快”，而是能证明：条件满足时进入向量路径；非对齐或尾部时安全回退；两条路径使用同一正确性和benchmark协议。

---

## 12. 教材配套测试：把输入契约变成证据

将以下内容保存为 `tests/test_ops.py`。

```python
import pytest
import torch

import stage1_cuda_ops as ops


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
COLS = [0, 1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1000, 4097]


def tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 2e-2, 2e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 255, 256, 257, 1000003])
def test_scaled_add(dtype, n):
    torch.manual_seed(7)
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    actual = ops.scaled_add(x, y, 0.25)
    expected = x + 0.25 * y
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_add_property_alpha_zero(dtype):
    x = torch.randn(257, device="cuda", dtype=dtype)
    y = torch.randn_like(x)
    actual = ops.scaled_add(x, y, 0.0)
    torch.testing.assert_close(actual, x, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [0, 1, 2, 17, 1024])
@pytest.mark.parametrize("cols", COLS)
@pytest.mark.parametrize("name,threads", [
    ("row_sum_naive", None),
    ("row_sum_warp", None),
    ("row_sum_block", 64),
    ("row_sum_block", 128),
    ("row_sum_block", 256),
    ("row_sum_block", 512),
])
def test_row_sum(dtype, rows, cols, name, threads):
    torch.manual_seed(rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)

    fn = getattr(ops, name)
    actual = fn(x) if threads is None else fn(x, threads)
    expected = x.sum(dim=-1)

    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_noncontiguous_is_rejected():
    x = torch.randn(4, 7, device="cuda").t()
    y = torch.randn_like(x)
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.scaled_add(x, y, 1.0)


def test_wrong_device_is_rejected():
    x = torch.randn(8)
    y = torch.randn(8)
    with pytest.raises(RuntimeError, match="CUDA"):
        ops.scaled_add(x, y, 1.0)


def test_wrong_dtype_is_rejected():
    x = torch.ones(8, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="dtypes"):
        ops.row_sum_block(x, 128)


def test_wrong_shape_is_rejected():
    x = torch.randn(2, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match="2D"):
        ops.row_sum_block(x, 128)


def test_current_stream():
    stream = torch.cuda.Stream()
    x = torch.randn(1024, device="cuda")
    y = torch.randn_like(x)

    with torch.cuda.stream(stream):
        actual = ops.scaled_add(x, y, 0.5)
        done = torch.cuda.Event()
        done.record(stream)

    done.synchronize()
    torch.testing.assert_close(actual, x + 0.5 * y)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extreme_values(dtype):
    values = torch.tensor(
        [0.0, 1.0, -1.0, 1e-4, -1e-4, 100.0, -100.0],
        device="cuda",
        dtype=dtype,
    )
    x = values.repeat(17, 1)
    actual = ops.row_sum_block(x, 128)
    expected = x.sum(dim=-1)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
```

运行：

```bash
pytest -q
```

### 12.1 这套测试覆盖了什么

- 正常：随机 FP32/FP16/BF16。
- 边界：0/1/31/32/33以及block整除边界。
- 随机：固定 seed 后可复现。
- 极值：正负、大值、小值。
- 错误：CPU、错误dtype、错误维度、非连续输入。
- 回归：后续每修一个故障，就把最小失败输入加入这里。
- stream：在非默认的当前 stream 中调用。

### 12.2 还需要亲自补上的测试

参考代码不能替代你的判断。请独立补充：

1. `scaled_add` 的 x/y shape 不同。
2. x/y dtype 不同。
3. 如果有两块 GPU，x/y 位于不同 device。
4. NaN/Inf 的传播行为，并与 PyTorch reference 对齐。
5. 一个曾经失败的最小 shape 回归测试。
6. `threads=31、33、2048` 的错误语义测试。

### 12.3 如何读失败结果

```text
import 失败
  → 构建、链接、动态库、ABI

pytest.raises 没触发
  → 输入契约检查缺失或检查顺序错误

只有 31/33/257 失败
  → 尾部、非满warp、越界或identity

只有 FP16/BF16 大输入失败
  → 累加精度、溢出或容差

重复执行偶尔失败
  → 竞态、错误stream或未初始化数据
```

---

## 13. 教材配套 Benchmark：正确测量而不是报一个数字

将以下内容保存为 `benchmark.py`。

```python
import statistics
import time

import torch

import stage1_cuda_ops as ops


def percentile(values, p):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def bench_cuda_event(fn, warmup=20, repeats=50, inner=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / inner)

    return {
        "median_ms": statistics.median(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def wrong_cpu_timer(fn, repeats=100):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeats


def synced_cpu_timer(fn, repeats=100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def main():
    torch.manual_seed(7)
    device = torch.cuda.current_device()
    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    print("CUDA in PyTorch:", torch.version.cuda)

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    implementations = {
        "torch.sum": lambda: x.sum(dim=-1),
        "naive": lambda: ops.row_sum_naive(x),
        "warp": lambda: ops.row_sum_warp(x),
        "block64": lambda: ops.row_sum_block(x, 64),
        "block128": lambda: ops.row_sum_block(x, 128),
        "block256": lambda: ops.row_sum_block(x, 256),
        "block512": lambda: ops.row_sum_block(x, 512),
    }

    print("\n计时方法差异：")
    fn = implementations["block256"]
    print("错误CPU计时 ms:", wrong_cpu_timer(fn))
    print("同步CPU计时 ms:", synced_cpu_timer(fn))
    print("CUDA Event:", bench_cuda_event(fn))

    print("\n受控实现对比：")
    for name, current_fn in implementations.items():
        print(name, bench_cuda_event(current_fn))


if __name__ == "__main__":
    main()
```

运行：

```bash
python benchmark.py
```

### 13.1 运行前必须先写预测

- naive 每个输出只有一个线程串行处理整行，并行度和单线程依赖链较差。
- warp 每行只使用32个线程；短行可能合适，长行每个lane循环次数多。
- block 使用更多warp分担长行，但会增加一个block barrier和shared部分和。
- block size不是越大越快。线程增多可能减少每线程循环，却增加资源和调度开销。

### 13.2 结果报告模板

```text
问题：比较 row_sum 的 block size
固定项：GPU、软件版本、shape、dtype、输入、warmup、repeats
变量：threads/block = 64/128/256/512
预测：长行增加线程可减少每线程循环，过大后收益下降
结果：median/P90/P95及波动
解释：结合循环次数、同步、occupancy、寄存器和实际流量
限制：只对当前shape/dtype/GPU成立
下一步：换短行和不规则shape验证边界
```

### 13.3 算术强度手算示范

对 FP32 `scaled_add` 的一个元素：

```text
读 x：4 bytes
读 y：4 bytes
写 out：4 bytes
最低流量：12 bytes
计算：1次乘法 + 1次加法 ≈ 2 FLOPs
Arithmetic Intensity ≈ 2 / 12 = 0.167 FLOPs/byte
```

它的理论强度很低，因此通常先怀疑访存和launch，而不是算力不足。

对 `[rows, cols]` FP32 row sum，每行最低读取 `4 × cols` bytes、写4 bytes，大约执行 `cols-1` 次加法。cols很大时强度仍约为 `1/4 FLOPs/byte`，同样偏向memory-bound。

### 13.4 阶段1够用的profiler读法

创建`profile_once.py`：

```python
import torch
import stage1_cuda_ops as ops

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
for _ in range(5):
    ops.row_sum_block(x, 256)
torch.cuda.synchronize()
```

运行：

```bash
ncu --set full --target-processes all python profile_once.py
```

阶段1不要在数百个指标里漫游，只按下面顺序回答问题：

1. **Duration**：这个kernel到底用了多久，是否稳定。
2. **Launch Stats**：grid、block和threads是否与手算一致。
3. **Memory Workload**：实际读取/写入流量、requests/sectors是否符合地址模式预测。
4. **Occupancy**：理论和实际active warps受到什么资源限制。
5. **Registers/Shared**：registers/thread和shared/block是否压低驻留block。
6. **Scheduler/Stall**：warp主要在等待数据、依赖、同步还是缺少可调度工作。
7. **Source对应**：热点或stall能否回到具体循环/访存位置。
8. **回到Duration**：指标变化最终有没有转化为真实时间改善。

每次只比较两个受控版本，例如block128与block256。结论写成：

```text
代码变化
  → 预期改变的线程/访问/资源
  → profiler中支持或反驳预测的指标
  → Duration是否真的改善
  → 对正确性、适用shape和资源的代价
```

查看编译器报告时，可临时在`setup.py`的nvcc参数中加入：

```python
"nvcc": ["-O3", "-lineinfo", "--ptxas-options=-v"]
```

重新构建后关注每个kernel的register数量、shared memory、stack frame和spill stores/loads。出现spill时先判断根因，不要立刻强制限制寄存器；减少register可能提高驻留，也可能制造更多local memory访问。

---

## 14. 四个自包含故障实验

故障实验不是破坏正式代码。每次先建立临时分支或复制目录，完成后恢复正确版本，并把最小失败输入加入回归测试。

### 14.1 undefined symbol

操作：让 `bindings.cpp` 保留 `row_sum_block_cuda` 声明，但临时改掉 `.cu` 中定义的函数名，然后重新构建/import。

你需要建立的推理链：

```text
声明存在
  → C++调用处可以编译
实现符号名称不匹配
  → 链接或动态加载时无法解析
Python import失败
  → 根因不在kernel索引
```

Linux上可使用：

```bash
find . -name '*.so'
ldd path/to/extension.so
nm -D -C path/to/extension.so | grep row_sum
```

- `ldd` 看依赖库是否找到。
- `nm -D -C` 看动态符号及反修饰后的C++函数名。
- 如果 `.so` 中根本没有目标定义，回到source列表、模板实例化和链接步骤。
- 如果符号看似相同但仍失败，比较参数签名、命名空间和ABI。

### 14.2 illegal memory access 与异步报错

操作：在临时版本中把 `scaled_add_kernel` 的 `idx < n` 改成 `idx <= n`，使用 `n=257`。

可能现象：launch处不立即失败，后面的同步、打印Tensor或其他CUDA API才报告错误。这是因为CPU只提交了异步工作。

调试时运行：

```bash
CUDA_LAUNCH_BLOCKING=1 pytest -q tests/test_ops.py -k scaled_add
```

它把异步执行临时变得更接近同步，便于让报错靠近根因，但会改变性能和时序，不能用于正式benchmark。illegal access后CUDA上下文可能已处于错误状态，修复代码后应重启Python进程再验证。

### 14.3 race condition

操作：临时删除 block reduction 中的 `__syncthreads()`，对大输入重复运行数百次。

解释：第一个warp可能在其他warp写完 `warp_sums` 之前读取，结果依赖调度时序。它可能每次都错、偶尔错，甚至在某一块GPU上看似一直正确；这些都不能建立正确性。

### 14.4 错误计时

操作：运行 `benchmark.py` 中三种计时。错误CPU计时通常更接近launch开销；同步CPU计时包含端到端提交和等待；CUDA Event主要测同一stream上的GPU elapsed time。

故障复盘统一使用：

```text
现象：用户能看到什么
最小复现：最小shape、dtype、命令
错误层：环境/构建/API/launch/kernel/异步/数值
假设：至少两个可能原因
证据：如何排除和确认
根因：哪条契约或哪行逻辑错误
修复：改了什么
回归：新增什么测试
预防：以后在哪个检查点提前发现
```

---

## 15. 四轮执行安排

“一轮”不强制等于一周。时间不足时延长日历，不能删掉验收。

### 第一轮：把完整调用链跑通

- 学模块 1。
- 创建工单 A，只支持 FP32 contiguous。
- 从空目录构建、import、运行、测试。
- 制造并修复一次 undefined symbol/ABI/依赖问题。

**本轮出口：** 能脱离教程重新画出调用链，并从空目录跑通最小算子。

### 第二轮：能从线程推到地址和硬件请求

- 学模块 2、3。
- 为工单 A 加 FP16/BF16、边界、当前 stream 和标量路径。
- 完成合并访问、bank conflict、alignment fallback 三个小实验。
- 手算 block、warp、wave、尾块和资源限制，再与工具结果对照。

**本轮出口：** 面对一个地址公式，能先预测性能方向再做实验。

### 第三轮：独立完成 Reduction

- 学模块 4、5。
- 完成工单 B 的 naive、warp、block 三条路径。
- 覆盖非 2 次幂、尾部、大输入和三种 dtype。
- 建立误差报告，估算 FLOPs/bytes 和算术强度。
- 制造并修复一次 illegal memory access/异步错误。

**本轮出口：** 工单 B 在规定 shape matrix 上正确，并能解释同步、精度和资源取舍。

### 第四轮：让结果可信、可交接

- 学模块 6、7、8。
- 完成错误计时/正确计时实验。
- 建立六类测试、README 和一键命令。
- 完成 Transformer shape、FLOPs、bytes 推导。
- 清理代码后冻结阶段 1 版本。

**本轮出口：** 别人可以在干净环境复现，你能用证据解释正确性和性能。

### 延迟复测

完成一周后，不看教程重新完成：

- 从空目录构建工单 A 的核心链路。
- 从空文件写出 reduction 核心结构。
- 手算一个新 shape 的线程、warp、尾部和算术强度。
- 解释两次故障的定位过程。

任一项需要照抄才能完成，就回到对应模块，不重复已经通过的模块。

---

## 16. 阶段 1 原表格逐项覆盖检查

这一节是阶段 1 的唯一总验收表。每一行都必须能指向代码、测试、推导或故障记录，不能只填写“已学习”。

| 原技术模块         | 原表要求                                                                          | 本教材位置 | 必须留下的证据                             |
| :----------------- | :-------------------------------------------------------------------------------- | :--------- | :----------------------------------------- |
| C++/CUDA 构建链    | RAII、模板实例化、`.so`符号、编译器/CUDA/PyTorch ABI、完整调用链                  | 模块 1     | 工单 A、环境快照、undefined symbol/ABI复盘 |
| CUDA 执行模型      | grid/block/thread/warp到SM、wave、尾部、SIMT、同步范围、资源限制                  | 模块 2、4  | 手算记录、block size对照、profiler核对     |
| 访存与片上存储     | register/shared/L1/L2/DRAM、transaction/sector、bank、spill/local                 | 模块 3     | 合并访问实验、bank实验、寄存器/资源记录    |
| Reduction 与正确性 | shuffle、block reduction、identity、非2次幂、尾部、竞态、异步错误                 | 模块 4     | 工单 B、shape matrix、illegal access复盘   |
| 数值与精度         | FP32/FP16/BF16存储与累加、容差、极值、NaN/Inf、FLOPs/bytes、算术强度              | 模块 5     | dtype协议、误差报告、极值测试、手算记录    |
| Stream 与计时      | 当前/default stream、异步launch、CUDA Event、CPU wall、warmup、同步、重复统计     | 模块 6     | 三种计时对照、median/P90/P95报告           |
| 测试分层           | dtype、shape、stride、device、alignment、空输入、边界、错误语义、真实shape matrix | 模块 7     | 六类测试、输入契约、README、一键命令       |
| Transformer 推导   | B/S/H/heads、主要GEMM/Attention shape和FLOPs                                      | 模块 10    | 一份闭卷shape/FLOPs/bytes推导              |
| 精度与硬件路径     | FP32/TF32/FP16/BF16动态范围、Tensor Core路径、累加精度                            | 模块 5、10 | 对比说明和工单中的accumulation实现         |
| 两次故障复盘       | ABI/编译错误、illegal memory access/异步错误                                      | 模块 1、4  | 两份“现象—证据—根因—修复—回归”记录         |

### 工程检查项出口

- [ ] ENG-01：能处理进程、线程、虚拟内存、fd、signal相关的常见开发异常。
- [ ] ENG-02：能解释并处理 Python环境、模块加载和依赖冲突。
- [ ] ENG-04：能用 RAII 和明确所有权避免资源泄漏、悬空引用和异常路径清理失败。
- [ ] ENG-05：能读懂并修改常见模板、dtype dispatch 和编译期实例化代码。
- [ ] ENG-07：能定位编译、链接、ABI、符号和共享库问题。
- [ ] ENG-08：能独立完成 Python → binding → launcher → kernel 全链路。
- [ ] ENG-09：能定义 shape、dtype、device、stride、alignment、空输入和错误语义。
- [ ] ENG-10：能建立 unit、correctness、property、regression 和 performance 测试。
- [ ] ENG-12：干净环境能一键构建测试，文档明确已验证和未验证边界。

### CUDA 检查项出口

- [ ] CUDA-01：能从 launch 配置推算 block、warp、wave 和尾部。
- [ ] CUDA-02：能根据分支条件预测发散并设计对照实验。
- [ ] CUDA-03：能解释 register、shared、L1/L2、DRAM 的作用域和复用。
- [ ] CUDA-04：能从线程地址判断合并访问和多余流量。
- [ ] CUDA-05：能计算常见 bank 映射、判断 conflict 并实验验证。
- [ ] CUDA-06：能区分理论/实际 occupancy，并解释高 occupancy 仍可能慢。
- [ ] CUDA-07：能从编译/profiler信息判断 register pressure 和 spill。
- [ ] CUDA-08：能独立实现 warp/block reduction 并处理非2次幂。
- [ ] CUDA-09：能区分 warp、block、host/device同步，避免竞态和错误可见性假设。
- [ ] CUDA-10：能正确使用当前 stream、event和异步语义。
- [ ] CUDA-15：能定位 illegal memory access、异步错误、竞态和数值错误。
- [ ] CUDA-17：能结合 register、shared和block limit选择合法配置。

### Kernel、数学与性能检查项出口

- [ ] KER-01～06：输入契约、尾块、精度、stream、alignment fallback和shape matrix全部有测试。
- [ ] MATH-02：能解释 Softmax max trick和归约中的主要误差来源。
- [ ] MATH-09：能解释 FP32/TF32/FP16/BF16的范围、精度、累加和硬件路径。
- [ ] MATH-11：能估算 FLOPs/bytes和算术强度，判断理论瓶颈方向。
- [ ] OPS-02：能说明一行由一个warp、多个warp或一个block处理的选择边界。
- [ ] PERF-01～04：固定测量协议、统计噪声、受控变量，并正确选择Event或wall time。

---

## 17. 新人正式接活前的最终口试

不看文档，用自己的话回答：

1. Python 调用为什么最终能够运行一个 GPU kernel？
2. 编译成功后为什么还可能出现 undefined symbol？
3. 给定 `N=1000、block=256`，有多少 block、warp和无效线程？
4. 相邻 lane 访问相邻地址为什么通常更好？sector代表什么方向的证据？
5. `shared[row][col]` 如何映射到 bank？为什么 padding 不一定有效？
6. occupancy 高为什么仍可能慢？register增加会产生什么连锁影响？
7. 33个数怎样用多个 warp 做正确 reduction？
8. 为什么 barrier 前不能让部分线程提前 return？
9. FP16 和 BF16 的主要区别是什么？为什么 reduction常用FP32累加？
10. `atol + rtol × abs(ref)` 分别解决什么问题？
11. 为什么 CPU 直接包住 kernel launch 的计时通常是错的？
12. PyTorch 当前 stream 与默认 stream 有什么区别？
13. 非连续 Tensor、空输入、非对齐地址分别如何处理？
14. 一次 illegal memory access为什么可能在后面的API才报错？
15. 给定 B/S/H/heads，如何写出 QKᵀ 的 shape和FLOPs？
16. 如何用 FLOPs/bytes判断一个算子的理论瓶颈方向？
17. 一个性能优化结果至少要报告哪些统计量和环境信息？
18. 你的两个工单各有哪些已知限制，为什么这些限制在阶段1可以接受？

如果任何回答只能背结论、不能联系到自己的代码和实验，这一项仍未掌握。

---

## 18. 课后题与参考答案

先在纸上完成题目，再看答案。能看懂答案不等于会做；至少隔一天换一组数字重算。

### 18.1 线程、warp 和尾部

**题目：** `N=1000`，`blockDim.x=256`。计算 block数、总线程数、总warp数、无效线程数和最后一个warp的有效lane数。

**答案：**

```text
blocks = ceil(1000 / 256) = 4
总线程 = 4 × 256 = 1024
总warp = 4 × (256 / 32) = 32
无效线程 = 1024 - 1000 = 24
最后一个block有232个有效线程
232 = 7 × 32 + 8
所以最后一个warp有8个有效lane
```

必须保留 `idx < N`。block数向上取整负责“覆盖全部数据”，边界判断负责“不访问分配范围外的数据”。

### 18.2 资源限制与 Occupancy

**题目：** 假设某GPU每个SM最多2048 threads、65536 registers、100 KiB shared memory、32 blocks。某kernel使用256 threads/block、64 registers/thread、16 KiB shared/block。估算资源分别允许多少blocks/SM，哪个先成为限制。

**答案：**

```text
线程限制：floor(2048 / 256) = 8 blocks
寄存器限制：floor(65536 / (64 × 256)) = 4 blocks
shared限制：floor(100 / 16) = 6 blocks
block硬上限：32 blocks

最终：min(8, 4, 6, 32) = 4 blocks/SM
```

每个block有8个warp，4个block就是32个active warp。若硬件最大64个active warp，理论occupancy约50%。这不说明性能一定只有峰值的一半；它只说明可用于隐藏延迟的活跃warp容量。

### 18.3 合并访问

**题目：** 一个warp中，lane `k` 分别访问 `x[base+k]` 和 `x[base+32*k]`。哪一个通常产生更少sector？

**答案：** 第一种。相邻lane访问相邻元素，地址聚集在少量连续内存区域。第二种相邻lane相隔128 bytes（FP32），warp会触碰许多分散区域。精确请求数量与对齐、cache line/sector规则和GPU架构有关，所以结论需要profiler验证，但方向可以先预测。

### 18.4 Shared memory bank

**题目：** 按32 banks、4-byte word模型，一个warp的lane `k` 访问地址 `4 × k` 和 `4 × 32 × k`，bank分别是什么？

**答案：**

```text
地址 4 × k：bank = k % 32，每个lane落到不同bank
地址 4 × 32 × k：bank = (32 × k) % 32 = 0，全部落到bank 0
```

第二种通常形成严重conflict。若所有lane读取完全相同地址，硬件可能广播；“同bank”与“同地址广播”不能混为一谈。

### 18.5 Reduction 与非2次幂

**题目：** 一行有33个元素，一个warp能不能处理？

**答案：** 能。lane 0可以在循环中处理col 0和32，其余lane处理各自一个元素，然后32个lane归约局部和。一个warp并不是只能处理32个元素，只是并行lane数为32。长行时每lane循环次数增加，是否改用多个warp是性能选择，不是正确性硬边界。

**题目：** 为什么block reduction中无数据线程使用0，而不是提前return？

**答案：** 0是sum的identity，不改变结果；让所有线程继续执行可以保证它们都到达block barrier。部分线程提前return、其他线程执行`__syncthreads()`会破坏barrier的参与约定。

### 18.6 数值精度

**题目：** FP16和BF16哪个范围大，哪个精度通常更高？

**答案：** BF16使用与FP32相近的指数宽度，因此动态范围大；FP16尾数位更多，因此在可表示范围内通常精度更高，但最大有限值较小、更容易溢出。两者做长reduction时通常使用FP32累加。

**题目：** `ref=0` 时为什么不能只看相对误差？

**答案：** 相对误差需要除以或相对于`abs(ref)`衡量，参考值为0或很小时会失去意义。因此判定式同时使用绝对容差`atol`和相对容差`rtol`。

### 18.7 Stable Softmax

**题目：** 对 `[1000, 1001]` 直接计算指数有什么风险？减去max后是什么？

**答案：** 直接计算`exp(1000)`和`exp(1001)`会溢出。减去max=1001后得到`[-1, 0]`，只需计算`exp(-1)`和`exp(0)`；数学softmax不变，数值范围稳定。

### 18.8 Stream 与计时

**题目：** 为什么下面代码通常低估kernel时间？

```python
t0 = time.perf_counter()
ops.row_sum_block(x, 256)
t1 = time.perf_counter()
```

**答案：** CPU在launch后通常立刻继续，只测到Python/C++调用和提交开销，没有等待GPU完成。若要测GPU elapsed time，使用同一当前stream上的CUDA Event；若要测端到端wall time，在计时边界正确同步。

### 18.9 Transformer shape 推导

使用：

```text
B=1, S=2048, H=4096, heads=32, I=11008
```

**答案：**

```text
D = H / heads = 128
Q/K/V shape = [1, 32, 2048, 128]
每头QKᵀ = [2048,128] × [128,2048] → [2048,2048]
全部attention score = [1,32,2048,2048]

QKV projection FLOPs
= 2 × (B×S) × H × (3H)
= 2 × 2048 × 4096 × 12288

QKᵀ FLOPs
= 2 × B × heads × S² × D

Attention×V FLOPs
= 2 × B × heads × S² × D

一个FFN H→I投影 FLOPs
= 2 × (B×S) × H × I

FFN I→H投影 FLOPs
= 2 × (B×S) × I × H
```

推导时不要只写数字，还要写每个维度代表什么。这样shape变化时才能重算，而不是背某个模型的结果。

### 18.10 故障归层

**题目：** 为以下现象选择最先检查的层。

1. 编译成功，Python import报告undefined symbol。
2. cols=256正确，cols=257错误。
3. 第一次运行正确，重复数百次偶尔错误。
4. CPU计时比CUDA Event小一个数量级。
5. 只有非连续输入错误。

**答案：**

1. 链接、动态加载、ABI和符号。
2. 尾部判断、索引或非2次幂归约。
3. 竞态、stream依赖或未初始化数据。
4. 异步launch导致CPU只测到提交开销。
5. stride契约；当前教材明确拒绝non-contiguous。

---

## 19. 最小术语表

| 术语                 | 在本阶段的准确含义                                   |
| :------------------- | :--------------------------------------------------- |
| Kernel               | 在GPU设备上并行执行的函数                            |
| Launcher             | 检查输入、选择配置并启动kernel的主机端函数           |
| Binding              | 把C++函数暴露给Python的接口层                        |
| Grid                 | 一次launch中的全部blocks                             |
| Block/CTA            | 一组能共享shared memory并进行block同步的threads      |
| Thread               | 执行kernel代码并拥有私有寄存器状态的逻辑线程         |
| Warp                 | GPU以SIMT方式调度的一组线程，通常32个lane            |
| Lane                 | warp中的一个线程位置                                 |
| SM                   | 执行warp/block的GPU流式多处理器                      |
| SIMT                 | 多个线程执行同一指令流、处理不同数据的执行方式       |
| Divergence           | 同一warp的lane走不同控制路径，造成路径分段执行       |
| Register             | 线程私有的高速片上存储资源                           |
| Shared memory        | block内threads共享、由程序管理的片上存储             |
| Global memory        | GPU设备大容量内存的CUDA地址空间                      |
| Coalescing           | 把warp中线程的相邻访问合并为较少内存请求             |
| Transaction/sector   | 硬件实际搬运内存区域的请求/粒度概念                  |
| Bank conflict        | warp内多个不同地址请求落到同一shared bank而被拆分    |
| Spill                | 寄存器放不下时，局部值被放到local memory             |
| Local memory         | 线程私有地址空间，物理访问可能落到cache/设备内存     |
| Occupancy            | 活跃warps相对于硬件最大warps的比例                   |
| Latency hiding       | 一个warp等待时调度其他ready warp执行                 |
| Reduction            | 把多个输入按sum/max等结合为更少输出                  |
| Identity             | 不改变归约结果的初始值，例如sum的0、max的负无穷      |
| Race condition       | 多线程无正确同步地访问同一数据，结果依赖时序         |
| Stream               | CUDA工作按序排队执行的队列抽象                       |
| CUDA Event           | 记录stream进度并测量GPU elapsed time的对象           |
| ABI                  | 不同二进制模块之间的调用、符号和对象布局约定         |
| `.so`                | Linux共享库，PyTorch扩展最终可被Python动态加载的产物 |
| Baseline             | 使用相同输入和协议进行比较的可信参考实现             |
| Arithmetic Intensity | FLOPs除以最低或实际搬运bytes                         |
| Compute-bound        | 性能理论上主要受计算吞吐限制                         |
| Memory-bound         | 性能理论上主要受数据搬运限制                         |
| Warmup               | 正式计时前运行若干次，排除初始化和冷启动影响         |
| P90/P95              | 90%/95%样本不超过的耗时，用于观察尾部和波动          |
| Shape matrix         | 系统覆盖正常、边界、不规则和压力shape的测试集合      |
| Fallback             | 快路径条件不满足时使用的安全通用实现                 |

---

## 20. 阶段 1 明确不学什么

为了让新人尽快开始干活，以下内容全部后置：

- 不追求 GEMM 超过 cuBLAS。
- 不学习完整 CUTLASS、Triton、FlashAttention 或 vLLM 源码。
- 不学习多 GPU、通信库、分布式推理和服务调度。
- 不学习复杂编译器、MLIR和自动算子生成。
- 不因为 occupancy 或带宽百分比高就直接下性能结论。
- 不为了展示技巧而写内联 PTX 或架构专属汇编。
- 不提前优化没有正确性测试和可信 benchmark 的代码。

**阶段 1 的工作边界只有一句话：能够独立交付一个最小 CUDA 算子和一个可靠 Reduction，并能证明它们正确、可复现、可解释。**
