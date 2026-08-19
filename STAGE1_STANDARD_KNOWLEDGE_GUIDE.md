# 阶段 1 正常掌握版教材：从不会到独立交付 CUDA 算子

> 版本：3.3，正常掌握、自包含教材、概念定义与关系双维、章内自查答案版。
> 适用对象：有芯片软件开发经验，但第一次承担 AI 芯片 Kernel/算子任务的人。  
> 目标：只依靠本教材的正文、代码、实验和答案，从零形成工程、C++、CUDA、数值、调试和性能基础，并完成四类独立交付。  
> 学习原则：先理解问题，再跟随例题，随后运行实验，最后关闭答案独立重写。

## 教材使用说明

这不是资料索引，也不是只告诉你“应该会什么”的验收提纲。正文负责提供阶段 1 正常掌握所需的：

- 概念解释和一行记忆法。
- 能直接运行的完整工程代码。
- 从输入 shape 到线程、地址、同步、精度和性能的推导。
- 正确性测试、benchmark、故障制造与定位方法。
- 练习题、预期现象和参考答案。
- 原阶段 1 表格的逐项验收映射。
- Python/C++并发、Host↔Device内存、Transpose、CUDA Graph等正常工作知识。
- 每一道最终口试题在正文中的教学依据与参考答案。

正常情况下，不需要再找一本 CUDA 教材配合阅读。主线先用小算子和Reduction建立心智模型，后续扩展章节再补齐正常岗位知识。只有下面三类信息需要查看目标环境的官方说明：

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

## 双维学习总入口：先掌握概念，再打通关系

你学习阶段 1 时，不需要把正文当成一串互不相关的知识点。每一项知识都从两个维度掌握：

1. **基础概念维度**：它是什么，解决什么问题，输入和输出是什么，代码中在哪里出现。
2. **概念关系维度**：它依赖什么，会影响什么，变化后沿哪条因果链影响正确性或性能，怎样用实验验证。

只会解释单个术语，说明“点”已经记住；能把多个概念连成因果链并拿出证据，才说明已经具备接活能力。

### 维度一：阶段 1 基础概念地图

下表不是新的背诵清单，而是后续章节的导航。先确认“这个概念回答什么问题”，再进入对应正文学习细节。

| 概念组       | 基础概念                                      | 它首先回答的问题                               | 正文出口        |
| :----------- | :-------------------------------------------- | :--------------------------------------------- | :-------------- |
| 工程调用链   | Python API、Binding、Launcher、Kernel         | 一次 Python 调用怎样最终在 GPU 上执行？        | 第 1、3、11 章  |
| 构建加载链   | 编译、链接、共享库、符号、ABI、动态加载       | 为什么编译成功仍可能 import 失败？             | 第 3、14 章     |
| 执行层级     | Grid、Block、Thread、Warp、Lane、SM           | 一个元素由谁计算，线程最终在哪个硬件上执行？   | 第 4 章         |
| SIMT 控制流  | Warp 调度、分支发散、活跃线程、尾块           | 同一 Warp 的线程为什么可能不能同时做有效工作？ | 第 4、6 章      |
| 地址与边界   | 线性索引、多维索引、stride、越界保护          | 线程编号怎样变成合法的元素地址？               | 第 4、6、9 章   |
| 存储层级     | Register、Shared、L1/L2、Global/DRAM          | 数据放在哪里，访问代价和共享范围有何不同？     | 第 5 章         |
| Global 访存  | 合并访问、request、sector、对齐、向量化       | 一组 Lane 的地址怎样变成硬件访存请求？         | 第 5、11、13 章 |
| Shared 访存  | Bank、Bank conflict、Padding                  | Shared 地址为什么可能被串行处理？              | 第 5、11 章     |
| 资源与并行度 | Registers/thread、Shared/block、Blocks/SM     | 一个 SM 能同时驻留多少工作？                   | 第 4、5 章      |
| 协作与正确性 | Barrier、Warp shuffle、Atomic、Race condition | 多线程怎样安全交换数据并完成 Reduction？       | 第 6、14 章     |
| 数值语义     | FP32、FP16、BF16、累加类型、NaN/Inf、容差     | 数学相同的程序为什么会出现数值差异？           | 第 7 章         |
| 异步语义     | Current stream、Default stream、Event、同步点 | Kernel 何时真正完成，错误和时间何时可见？      | 第 8、14 章     |
| 接口与证据   | Device、dtype、shape、layout、测试、benchmark | 怎样证明算子对、可用且结果可复现？             | 第 9、12、13 章 |
| 模型上下文   | GEMM、Reduction、Softmax、Attention shape     | 基础 Kernel 怎样进入 Transformer 算子？        | 第 7、10 章     |

下面逐组定义总表中的基础概念。定义回答“它是什么”；这些概念怎样互相影响，由后面的关系地图回答。

#### 概念组 1：工程调用链

| 基础概念   | 定义                                                                          |
| :--------- | :---------------------------------------------------------------------------- |
| Python API | Python 使用者可以直接调用的函数、类及参数约定，是算子对上层暴露的入口。       |
| Binding    | 把 Python 对象和调用转换为 C++ 函数及原生数据结构的语言桥接层。               |
| Launcher   | 在 CPU 侧检查输入契约、选择实现和类型、计算启动配置并发射 Kernel 的宿主函数。 |
| Kernel     | 在 GPU 上由大量线程并行执行的设备函数；CUDA C++ 中通常用 `__global__` 声明。  |

#### 概念组 2：构建加载链

| 基础概念 | 定义                                                                                   |
| :------- | :------------------------------------------------------------------------------------- |
| 编译     | 把一个源文件及其可见头文件转换为目标文件或中间代码，并检查该翻译单元内的语法和类型。   |
| 链接     | 合并目标文件和依赖库、解析跨文件符号引用，并生成可执行文件或共享库。                   |
| 共享库   | 可在运行时被一个或多个进程加载的二进制库；Linux 上常见扩展名为 `.so`。                 |
| 符号     | 二进制中用于标识函数、全局变量等实体的名称及相关信息，是链接和动态加载解析引用的依据。 |
| ABI      | 不同二进制模块之间关于名称修饰、参数传递、对象布局和调用约定的兼容规则。               |
| 动态加载 | 程序运行时把共享库映射进进程地址空间、加载其依赖并解析所需符号的过程。                 |

#### 概念组 3：执行层级

| 基础概念 | 定义                                                                                                   |
| :------- | :----------------------------------------------------------------------------------------------------- |
| Grid     | 一次 Kernel Launch 创建的全部 Block 的集合。                                                           |
| Block    | 可共享 Shared Memory、可执行 Block 级同步，并在整个生命周期内驻留于同一个 SM 的线程组。                |
| Thread   | CUDA 编程模型中的单个执行实例，拥有自己的索引、寄存器状态和局部控制流。                                |
| Warp     | GPU 调度和执行线程指令的基本线程组；当前 NVIDIA CUDA 架构通常每个 Warp 有 32 个 Lane。                 |
| Lane     | Warp 内某个线程的位置编号，通常为 `0～31`，用于描述同一 Warp 各线程的地址和活跃掩码。                  |
| SM       | Streaming Multiprocessor，负责驻留和调度 Block/Warp、执行指令并提供寄存器与 Shared Memory 的硬件单元。 |

#### 概念组 4：SIMT 控制流

| 基础概念  | 定义                                                                                       |
| :-------- | :----------------------------------------------------------------------------------------- |
| Warp 调度 | 调度器从当前具备发射条件的 Warp 中选择一个或多个，使其下一条指令进入执行管线的过程。       |
| 分支发散  | 同一 Warp 的 Lane 对一个分支条件得到不同结果，导致不同路径在不同活跃掩码下分批执行的现象。 |
| 活跃线程  | 在当前指令的活跃掩码中实际参与执行的线程；未活跃 Lane 不产生该指令的有效结果。             |
| 尾块      | 因任务元素数不能整除 Block 覆盖范围而只有部分线程处理有效数据的最后一个或若干 Block。      |

#### 概念组 5：地址与边界

| 基础概念 | 定义                                                                          |
| :------- | :---------------------------------------------------------------------------- |
| 线性索引 | 把线程编号或多维坐标换算成一维元素序号，便于访问线性存储的数据。              |
| 多维索引 | 用行、列、批次等多个坐标描述逻辑元素位置，并按 Layout/Stride 转换为存储偏移。 |
| Stride   | 某一维坐标增加 1 时，底层存储地址需要跨过的元素数或字节数。                   |
| 越界保护 | 在读写前判断逻辑坐标或线性索引是否合法，使无效线程不访问分配范围之外的内存。  |

#### 概念组 6：存储层级

| 基础概念      | 定义                                                                              |
| :------------ | :-------------------------------------------------------------------------------- |
| Register      | 通常由单线程私有使用的最快片上存储，由编译器分配，容量不足时可能发生 Spill。      |
| Shared Memory | 同一 Block 内线程显式读写和共享的片上存储，生命周期通常与 Block 相同。            |
| L1 Cache      | 靠近 SM 的硬件管理缓存，用于减少部分局部、全局或纹理访问的更远层级请求。          |
| L2 Cache      | GPU 上跨 SM 共享的硬件管理缓存，是访问 DRAM 前的最后一级通用缓存。                |
| Global Memory | CUDA 编程模型中可被 Grid 内线程访问的设备全局地址空间，并不等同于某一块物理芯片。 |
| DRAM          | GPU 板上承载大部分 Global Memory 数据的高容量片外显存，延迟高于片上存储。         |

#### 概念组 7：Global 访存

| 基础概念 | 定义                                                                                      |
| :------- | :---------------------------------------------------------------------------------------- |
| 合并访问 | 一个 Warp 的同一条 Global Memory 指令所访问的地址可由较少内存事务覆盖的性质。             |
| Request  | 一个内存指令经过地址合并后向 Cache/内存层级提出的访问请求；具体统计口径随架构和指标而异。 |
| Sector   | Cache Line 中可独立传输或统计的较小固定粒度区域；在常见 NVIDIA 指标中通常为 32 Byte。     |
| 对齐     | 地址是某个访问粒度的整数倍，从而满足标量、向量或硬件事务的地址约束。                      |
| 向量化   | 用一条宽指令为单个线程加载、存储或计算多个相邻元素的实现方式。                            |

#### 概念组 8：Shared 访存

| 基础概念      | 定义                                                                                 |
| :------------ | :----------------------------------------------------------------------------------- |
| Bank          | Shared Memory 被划分出的可并行服务访问的存储分区，地址按硬件规则映射到不同 Bank。    |
| Bank Conflict | 同一 Warp 的同一条 Shared 指令访问同一 Bank 的不同地址，导致请求需要分批服务的现象。 |
| Padding       | 在逻辑数据之间增加不参与计算的存储元素，用于改变地址对齐、行跨度或 Bank 映射。       |

#### 概念组 9：资源与并行度

| 基础概念         | 定义                                                                        |
| :--------------- | :-------------------------------------------------------------------------- |
| Registers/thread | 编译后平均分配给每个线程的寄存器数量，是限制线程和 Block 驻留的重要资源量。 |
| Shared/block     | 每个 Block 静态与动态申请的 Shared Memory 总量。                            |
| Blocks/SM        | 一个 SM 在资源和架构上限约束下可以同时驻留的 Block 数量。                   |

#### 概念组 10：协作与正确性

| 基础概念       | 定义                                                                                           |
| :------------- | :--------------------------------------------------------------------------------------------- |
| Barrier        | 要求指定作用域内的参与线程都到达后才能继续，并提供相应内存可见性保证的同步点。                 |
| Warp Shuffle   | 让同一 Warp 的 Lane 直接交换寄存器值的指令机制，无需先把数据写入 Shared Memory。               |
| Atomic         | 对同一内存位置执行不可分割的读—改—写操作，使并发更新不会彼此覆盖。                             |
| Race Condition | 多个执行者在缺少所需同步或原子保证时并发访问同一数据，且至少一个执行者写入，导致结果依赖时序。 |

#### 概念组 11：数值语义

| 基础概念 | 定义                                                                                        |
| :------- | :------------------------------------------------------------------------------------------ |
| FP32     | IEEE 754 Binary32 浮点格式，通常由 1 位符号、8 位指数和 23 位显式尾数组成。                 |
| FP16     | IEEE 754 Binary16 浮点格式，通常由 1 位符号、5 位指数和 10 位显式尾数组成。                 |
| BF16     | BFloat16 浮点格式，通常由 1 位符号、8 位指数和 7 位显式尾数组成，范围接近 FP32 而精度更低。 |
| 累加类型 | Reduction、点积等连续累加过程中用于保存中间和的 dtype，可与输入和输出 dtype 不同。          |
| NaN      | Not a Number，表示未定义或不可表示的浮点结果，并会按浮点规则传播。                          |
| Inf      | 正无穷或负无穷的浮点特殊值，通常由溢出、除零或输入本身产生。                                |
| 容差     | 判断近似数值结果是否可接受的误差界，通常由绝对容差 `atol` 和相对容差 `rtol` 共同定义。      |

#### 概念组 12：异步语义

| 基础概念       | 定义                                                                                      |
| :------------- | :---------------------------------------------------------------------------------------- |
| Current Stream | 框架或线程当前上下文为某个设备选中的 CUDA Stream，扩展算子通常必须沿用它。                |
| Default Stream | CUDA 为设备提供的默认 Stream；它是一个具体执行队列，并不等同于任意时刻的 Current Stream。 |
| Event          | 插入 CUDA Stream 时间线的设备侧标记，可用于建立依赖、查询完成状态或测量设备执行区间。     |
| 同步点         | CPU、Stream 或设备必须等待指定异步工作完成后才能继续的程序位置。                          |

#### 概念组 13：接口与证据

| 基础概念  | 定义                                                                       |
| :-------- | :------------------------------------------------------------------------- |
| Device    | Tensor 所在和算子应执行的计算设备，例如某一块具体 GPU。                    |
| Dtype     | Tensor 每个元素的存储与解释类型，例如 FP32、FP16、BF16 或整数类型。        |
| Shape     | Tensor 各逻辑维度的长度组合，决定元素数量和算子输出维度。                  |
| Layout    | 逻辑坐标到物理存储地址的排列规则，由维度顺序、Stride、打包和对齐共同体现。 |
| 测试      | 用预期结果或预期错误自动检查实现行为是否符合契约的可重复验证。             |
| Benchmark | 在固定任务、环境和统计协议下测量时间或吞吐，用于公平比较实现性能。         |

#### 概念组 14：模型上下文

| 基础概念        | 定义                                                                                         |
| :-------------- | :------------------------------------------------------------------------------------------- |
| GEMM            | General Matrix Multiplication，计算矩阵乘积及可选累加，是 Transformer 线性层的主要计算形式。 |
| Reduction       | 用满足要求的合并操作把多个输入压缩成更少输出的运算，例如 Sum、Max 或 Norm 统计。             |
| Softmax         | 把一组分数变换为和为 1 的非负权重：`softmax(x_i)=exp(x_i)/Σexp(x_j)`。                       |
| Attention Shape | Q、K、V、Score 和输出 Tensor 在 Batch、Head、Sequence、Head Dimension 等维度上的形状关系。   |

学习单个概念时，统一写一张“概念卡”：

```text
概念名称：
它解决的问题：
一句话定义：
输入、输出或作用对象：
上游依赖：
下游影响：
代码中的位置：
能证明它的测试或指标：
最容易混淆的概念：
```

如果只能填写“一句话定义”，说明只是科普级理解；九项都能填写，才是岗位级理解。

### 维度二：阶段 1 核心关系地图

#### 关系链 A：从源码到一次 Kernel 执行

```text
源码与头文件
  → 编译得到目标文件
  → 链接解析跨文件符号并生成共享库
  → Python import 触发动态加载和 ABI/依赖检查
  → Binding 把 Python 对象转换为 C++/Tensor 参数
  → Launcher 检查契约、选择类型与 Launch 配置
  → 在 Current Device、Current Stream 上发射 Kernel
  → Kernel 异步执行
  → Event、显式同步或后续依赖让结果与错误变得可见
```

这条链用于定位工程问题。必须能分别解释“编译错误、链接错误、import 错误、参数错误、Launch 错误、异步执行错误”发生在哪一段，而不是把所有问题统称为 CUDA 报错。

#### 关系链 B：从 Shape 到线程，再到地址

```text
输入 shape 与算子公式
  → 确定输出元素和并行任务
  → Grid/Block 划分任务
  → blockIdx/threadIdx 得到逻辑索引
  → 索引与 stride 得到元素偏移
  → 元素偏移 × sizeof(dtype) 得到字节地址
  → 边界判断决定该线程是否有效
```

索引正确只证明“访问的是目标元素”，不自动证明访问高效。下一条关系链才负责解释硬件访存。

#### 关系链 C：从 Lane 地址到访存效率

```text
同一 Warp 中各 Lane 的字节地址
  → 地址是否连续、对齐、跨越多少 segment
  → 形成若干 memory request/sector
  → 经过 L1/L2 命中或访问 DRAM
  → 实际传输字节与等待时间
  → 影响 eligible warp、stall 和最终 Duration
```

若数据先进入 Shared memory，还要追加一条分支：

```text
Shared 字节地址
  → 映射到 Bank
  → 同一 Warp 是否访问同一 Bank 的不同地址
  → 是否产生 Bank conflict 和串行化
```

因此，“相邻线程访问相邻元素”“使用 `float4`”“给 Shared 加 Padding”都只是实现手段，是否有效必须回到实际地址、request/sector 或 Bank 指标验证。

#### 关系链 D：从算法映射到资源与性能

```text
每个线程/Block 承担的工作
  → Registers/thread、Shared memory/block、Threads/block
  → 每个 SM 可驻留的 Block 与 Warp 数
  → Occupancy 和可供调度的 Warp
  → 指令依赖、内存等待时是否仍有 Eligible Warp
  → Issue 是否充足
  → 固定工作量的 Duration
```

这里存在取舍关系：增加寄存器或 Shared 可能提高复用、减少 Global 访问，也可能降低驻留并行度。因此 `Occupancy↑ → 一定更快` 和 `寄存器越少越好` 都是错误结论。阶段 1 要会把资源、复用和 Duration 放在同一条链上解释。

#### 关系链 E：从数学定义到可信正确性

```text
数学公式
  → API 契约：device/dtype/shape/layout/空输入
  → 映射与边界：每个输出由谁写
  → 协作规则：Barrier/Shuffle/Atomic/可见性
  → 数值路径：输入精度、累加精度、运算顺序
  → Reference 与 atol/rtol
  → 正常、边界、错误、随机、极值和压力测试
```

任何一层缺失都不能只用“一组随机输入通过”来替代。Race 可能偶发，低精度误差会随规模累积，错误输入还必须验证拒绝行为。

#### 关系链 F：从异步执行到正确计时

```text
CPU 发射 Kernel
  → Launch 很快返回
  → GPU 在 Stream 中按依赖执行
  → CUDA Event 记录设备时间线位置
  → 同步后才能读取完成时间或异步错误
  → Warmup + 多次重复 + 稳健统计
  → 与同一工作量、同一精度的 Baseline 比较
```

CPU 包围 Launch 得到的通常主要是发射时间。Event 解决设备执行区间，Warmup 和统计解决首次构建、频率波动与离群值；三者回答的问题不同，不能互相替代。

#### 关系链 G：从基础 Kernel 到 Transformer

```text
[B,S,H] 激活与权重
  → GEMM 产生 Q/K/V 或 FFN 中间结果
  → reshape/split heads 得到 [B,heads,S,D]
  → QKᵀ 产生 [B,heads,Sq,Sk]
  → Scale/Mask/Softmax 做逐元素、归约和归一化
  → 概率矩阵 × V 再次进入矩阵乘
```

这条链的目的不是在阶段 1 实现完整推理系统，而是能把“索引、访存、Reduction、数值稳定性、GEMM shape”放回真实模型中理解。

### 四种关系必须都能说清楚

| 关系类型 | 要回答的问题                       | 合格示例                                                         |
| :------- | :--------------------------------- | :--------------------------------------------------------------- |
| 依赖关系 | 没有 A，B 为什么无法成立？         | 没有统一 API 契约，就无法公平比较 Reference 与 Kernel            |
| 映射关系 | 软件概念怎样落到线程、地址或硬件？ | `threadIdx.x` → 元素索引 → 字节地址 → Warp 的 request/sector     |
| 因果关系 | A 变化后，通过什么中间量影响结果？ | Shared 增大 → Blocks/SM 可能下降 → Eligible Warp 变化 → Duration |
| 取舍关系 | 改善一个因素时可能牺牲什么？       | FP32 累加改善误差，但可能增加转换、寄存器或计算成本              |

### 双维闭卷自查：用关系判断是否真正掌握

对任意一个概念，关闭教材后完成下面五步：

1. 用一句话定义它，并说明它解决的具体问题。
2. 画出至少一条“上游 → 它 → 下游”的三节点关系。
3. 修改一个输入、shape 或实现参数，预测这条关系中哪些量会变化。
4. 指出能验证预测的代码位置、测试现象或 profiler 指标。
5. 说明一个反例或适用边界，避免把局部规律背成绝对结论。

例如，学习 Occupancy 不能止于定义；还要能写出：

```text
Registers/thread 增加
  → Blocks/SM 可能减少
  → Occupancy 可能下降
  → 但如果寄存器换来了更高数据复用或 ILP
  → Duration 仍可能下降
```

五步中有任一步无法完成，就回到关系链指出的前置章节查漏补缺。后文原有闭卷口试仍是阶段出口；本节是帮助你组织和定位知识的学习入口，不额外扩大考试范围。

#### 双维自查详细答案：以“合并访问”为完整示范

这组自查没有唯一固定概念，因此答案应当是一套可重复使用的作答方法。下面用“合并访问”示范五步的合格深度；换成 Occupancy、Barrier、Stream 等概念时，也要保持相同结构。

1. **定义和问题。** 合并访问描述的是：一个 Warp 执行同一条 Global Load/Store 指令时，32 个 Lane 给出的地址能否由较少的内存事务覆盖。它解决的不是“某个线程有没有读到正确元素”，而是“一组线程为了这些有效数据，硬件实际发出了多少请求、搬运了多少额外字节”。
2. **上游与下游。** `线程到元素的映射 → 每个 Lane 的字节地址 → request/sector 数量 → 实际内存流量和等待 → Duration`。Shape、stride、dtype、向量宽度和首地址对齐都是上游条件；内存请求、缓存行为、eligible warp 和时间是下游结果。
3. **改变变量并预测。** 把连续访问 `x[warp_base + lane]` 改成跨步访问 `x[(warp_base + lane) * 32]`。有效读取元素数不变，但地址分散到更多内存区域，预计 request/sector 和实际传输字节增加，Kernel Duration 在内存受限且规模足够大时上升。
4. **验证证据。** 代码中核对最终字节地址，而不只看数组下标；正确性测试保证两版完成相同任务；CUDA Event 或 NCU Duration 判断结果；NCU 的 Global Load/Store、Sector、DRAM/L2 流量用于验证原因。Cache 命中可能让实际 DRAM bytes 小于按代码 Load 次数估算的值。
5. **边界和反例。** 连续地址通常有利，但不保证整体一定更快：小输入可能被 Launch 固定开销主导；计算受限 Kernel 可能看不到明显收益；对齐、缓存命中、向量指令和额外索引计算也会改变结果。因此最后结论必须限定 Shape、dtype、实现和 GPU，并回到相同工作量的 Duration。

达到这个深度的标志是：即使 profiler 结果与预测相反，也能沿关系链检查“地址是否真的改变、Cache 是否掩盖 DRAM、工作量是否一致、瓶颈是否在内存”，而不是继续背原结论。

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

#### 3.5.1 详细答案

1. **从 Python 到 Kernel 的完整链路。** Python 调用扩展函数后，Binding 接收并转换 Python/Tensor 参数；Launcher 检查 device、dtype、shape、stride 和空输入，设置正确 device，取得 PyTorch Current Stream，计算 Grid/Block 并发射 Kernel；Kernel 用线程索引读写 Device Memory，然后异步返回。Binding 是语言边界，Launcher 是契约和调度边界，Kernel 才是设备执行函数，三者不能混为一层。
2. **编译成功但 import 仍可能失败。** 编译只证明单个源文件能生成目标文件。之后还可能在链接时缺少符号，在动态加载时找不到依赖 `.so`，因 C++ ABI、CUDA/PyTorch 版本不兼容而解析失败，或 Python 实际导入了另一个同名旧模块。定位顺序应是：确认导入文件实际路径，查看缺失库和符号，再核对构建时与运行时环境。
3. **RAII 如何保护异常路径。** RAII 把资源生命周期绑定到栈对象：构造时取得资源，析构时释放资源。正常 return、异常抛出或中间检查失败时，已经构造的局部对象都会析构，所以 `std::vector`、智能指针、文件对象和锁守卫不会依赖程序员记住每条手工释放路径。边界是：裸指针、跨语言所有权或 CUDA 异步使用中的对象生命周期仍要显式设计。
4. **模板为什么可能没有生成所需符号。** 模板只有在某个具体类型组合被实例化后才生成机器代码。如果模板定义只放在 `.cpp/.cu` 中，而另一个翻译单元只看到了声明，链接器可能找不到所需实例；类型签名、命名空间或编译宏不一致也会造成符号不匹配。常见处理是把定义放到可见头文件，或在实现文件中对所需类型显式实例化。
5. **Python 包、动态库和 import 路径冲突。** 先打印模块的 `__file__`，确认加载的是当前工程产物；再核对 `sys.path`、虚拟环境和扩展后缀；若共享库已找到但依赖缺失，检查动态依赖和运行时库搜索路径。不要看到 `ImportError` 就重新编译，必须先区分“模块没找到”“共享库没加载”“符号没解析”三类问题。
6. **新环境的一键复现。** README 至少写明驱动/Toolkit/PyTorch/Python 版本、环境创建、构建、测试和 benchmark 命令，以及预期结果。验收方式是在干净环境按文档执行，不依赖当前终端遗留变量或未声明文件；成功 import、全部测试通过并能运行最小 benchmark，才算可复现。

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

#### 4.4.1 详细答案

1. **手算 Block、Warp 和尾部。** 一维任务常用 `grid = ceil(N / block_size)`。例如 `N=1000, block=256`，需要 4 个 Block；每个满 Block 有 8 个 Warp，最后一个 Block 只有 232 个有效线程，但硬件仍按 8 个 Warp 调度，其中最后一个 Warp 只有 8 个有效 Lane。必须同时写出总线程数 `1024`、无效线程数 `24` 和边界判断 `idx < N`。
2. **Thread tail 与 Wave tail。** Thread tail 是最后一个 Warp/Block 中部分线程因边界而无效；Wave tail 是 Grid 的 Block 数不能均匀填满所有 SM，最后一轮只有少数 SM 在工作。前者由元素数量与 Block 大小决定，后者由 Block 总数、SM 数和 Blocks/SM 决定，优化方法也不同。
3. **Block 与 SM、跨 Block 同步。** 一个 Block 在其整个生命周期内驻留在一个 SM 上，才能共享 Shared Memory 并执行 `__syncthreads()`。不同 Block 可能以任意顺序在不同 SM 上运行，普通 Kernel 内没有可靠的全 Grid Barrier；如果等待尚未被调度的 Block，还可能死锁。跨 Block 阶段通常拆成多个 Kernel，利用 Kernel 边界或 Cooperative Groups 的受限机制同步。
4. **判断 Warp 分支发散。** 看同一 Warp 在同一分支指令上的 Lane 条件是否不同。若 `if (lane < 16)`，一个 Warp 内两条路径都要分批执行，发生发散；若 `if (blockIdx.x < 4)`，同一 Block 内所有 Warp 的条件通常一致，不构成 Warp 内发散。边界判断会让最后一个 Warp 发散，但若只占极少比例，未必是主要瓶颈。
5. **资源怎样限制驻留 Block。** Blocks/SM 同时受最大线程数、最大 Block 数、寄存器总量和 Shared Memory 总量约束。例如每 Block 使用的 Shared 增大，`floor(shared_per_SM / shared_per_block)` 会下降；每线程寄存器增加，也会通过每 Block 总寄存器数限制驻留数。最终取所有限制中的最小值，还要考虑硬件分配粒度。
6. **Occupancy 高仍可能慢。** Occupancy 只表示 Active Warp 相对硬件上限的比例，不表示这些 Warp 当前 Eligible，也不表示访存合并、指令数量或算法工作量更优。高 Occupancy Kernel 仍可能因长依赖链、Bank Conflict、低缓存命中或大量无效工作而慢；较低 Occupancy Kernel 也可能因更高复用和 ILP 更快。结论要用 Eligible Warp、Stall、访存/资源指标解释，并以固定工作量 Duration 判定。

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

#### 5.5.1 详细答案

1. **存储层级与复用范围。** Register 通常是单线程私有、延迟最低但容量有限；Shared Memory 是同 Block 线程显式共享的片上存储；L1/L2 是硬件管理的 Cache，L1 更靠近 SM，L2 跨 SM 共享；Global Memory 是编程模型中的 Device Memory 地址空间，Cache 未命中后通常访问 DRAM。速度不是唯一差异，还要看容量、作用域、显式/隐式管理和生命周期。
2. **从 Warp 地址预测合并访问。** 先把元素下标乘 `sizeof(dtype)` 得到 32 个 Lane 的字节地址，再观察它们覆盖多少连续对齐区域。FP32 的 Lane `0..31` 读取连续 128B 通常比每 Lane 跨 128B 读取更少 Sector；实际事务还受架构、Cache、对齐和活跃掩码影响。预测后用 Request/Sector、L1/L2/DRAM bytes 和 Duration 验证。
3. **Shared Bank 计算。** 常见模型下，FP32 元素 `shared[row][col]` 的 Bank 可写为 `(row × LD + col) % 32`。必须针对“同一 Warp 的同一条指令”列出各 Lane 的地址；多个 Lane 访问同一 Bank 的不同地址会产生冲突，而广播同一地址可能被硬件特殊处理。Padding 只有改变了这组实际地址的 Bank 映射才有效。
4. **Register pressure 到驻留 Block。** `registers/thread × threads/block` 给出每 Block 的近似寄存器需求，经过硬件分配粒度取整后，与 SM 寄存器总量共同限制 Blocks/SM。寄存器减少可能提高驻留并行度，但若造成 Spill 或破坏复用，反而会增加内存访问，所以不能只追求更低寄存器数。
5. **Local Memory 与 Spill。** CUDA 的 Local Memory 是“线程私有地址空间”，不代表物理上位于片上；大数组、动态索引或寄存器不足时，局部变量可能落到由 L1/L2 缓存的 Device Memory。先看编译资源报告中的 spill load/store，再用 NCU 的 Local Load/Store、内存流量和 Duration 验证，不能仅凭源码出现局部变量就断言发生 Spill。
6. **安全向量化路径。** 以 `float4` 为例，至少检查所有相关指针 16B 对齐、连续布局/正确 stride、可处理长度和行首对齐；主体按 4 元素处理，尾部用标量路径，条件不满足时整体走标量 Fallback。`contiguous` 不保证任意偏移后的地址仍然 16B 对齐。测试应覆盖对齐、非对齐切片、长度余数 1/2/3 和空输入，并用 SASS/指令指标确认宽指令真的生成。

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

#### 6.6.1 详细答案

1. **Warp 与 Block Reduction 核心。** 每个线程先加载合法输入或 identity 到寄存器；Warp 内用 `__shfl_down_sync(mask, value, offset)` 按 16、8、4、2、1 合并；每个 Warp 的 Lane 0 把部分和写入 Shared；全 Block 同步；第一个 Warp 再读取各 Warp 的部分和并做第二次 Warp Reduction；最终由一个线程写输出。空文件重写时必须先确定参与掩码、Warp 数和输出所有者。
2. **Barrier 保护什么。** 在 Shared Reduction 中，第一个 Barrier 保证所有线程完成当前阶段写入后，任何线程才读取这些值；若循环复用同一 Shared 区域，下一轮写入前还可能需要防止上一轮读取未完成。解释 Barrier 时必须明确“哪些线程写哪个地址、哪些线程随后读哪个地址”，不能只说“这里为了同步”。
3. **非整齐输入。** 对 `idx >= N` 的线程装入 sum identity `0` 或 max identity `-∞`，而不是直接在所有 Barrier 前 return；非满 Warp 要使用正确 active mask，第二阶段只让合法 Warp 部分结果参与。超大输入可让每线程 Grid-stride 累加或多 Block 产生部分结果，再由第二 Kernel/原子方式合并。
4. **Identity 的选择。** Identity 满足 `x ⊕ identity = x`。Sum 用 `0`，Product 用 `1`，Max 用该 dtype 可表示的 `-∞` 或最低值，Min 用 `+∞` 或最高值；逻辑 All/Any 分别用 `true/false`。错误 identity 会让补齐线程改变结果，尤其在非 2 次幂输入中暴露。
5. **Barrier 前部分线程不能提前 return。** `__syncthreads()` 要求 Block 中所有应参与的线程按一致控制流到达。如果越界线程提前退出而其他线程等待 Barrier，行为未定义，可能死锁或得到错误结果。正确做法是让无效线程装入 identity 后继续参加同步，只限制最终读写。
6. **故障定位与回归。** 先把失败缩小到最小 Shape 并同步错误暴露点；越界用 Compute Sanitizer 和边界手算，Race 用 Racecheck/重复运行和 Barrier 检查，异步错误用 Launch 后检查加同步定位。修复后把最小失败 Shape、触发条件和期望行为写成自动回归测试，避免只保存一段日志。

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

#### 7.7.1 详细答案

1. **四种浮点格式。** FP32 是 1/8/23 位，范围和精度都较高；TF32 通常保持接近 FP32 的指数范围但降低有效尾数，用于支持的矩阵乘路径，存储仍常是 FP32；FP16 是 1/5/10 位，精度较 BF16 高但范围较窄；BF16 是 1/8/7 位，范围接近 FP32但精度较低。实际速度还取决于 GPU 是否支持对应 Tensor Core/指令路径，不能从 dtype 名字直接推出性能。
2. **低精度 Reduction 的累加类型。** 接口要分别声明输入/存储 dtype、运算转换 dtype、accumulator dtype 和输出 dtype。常见做法是 FP16/BF16 输入转 FP32 累加，减少逐步舍入和溢出风险，再按契约转换输出；这不能恢复输入量化前已经丢失的信息，也不保证不同归约顺序 bitwise 一致。
3. **设置 atol/rtol。** 判定通常是 `abs(out-ref) <= atol + rtol*abs(ref)`：`atol`保护接近 0 的结果，`rtol`允许误差随参考值尺度增长。容差应结合 dtype、运算次数、数据分布和参考精度，通过误差实验确定；不能把容差放宽到足以掩盖索引错误。
4. **误差报告。** 除通过率外，至少报告最大绝对误差、最大/分位相对误差、均值或 P95、最坏元素位置及输入尺度，并单独统计 NaN/Inf。对不同 K 或归约长度画误差随规模变化，才能判断是正常舍入累积还是实现错误。
5. **Max trick 与极值。** Softmax 用 `exp(x_i-m) / Σexp(x_j-m)`，其中 `m=max(x)`；平移不改变数学比值，却把最大指数变成 1，避免大正数指数溢出。极大正负值、全相等、NaN、`+Inf/-Inf` 用来验证稳定性和接口语义；Max trick 主要解决上溢，并不能自动定义所有 Inf/NaN 组合的结果。
6. **FLOPs、最低 bytes 与 AI。** 先按数学任务计有效 FLOPs，再只计算理论上不可避免的 Global 读写字节，`AI = FLOPs / bytes`。例如 FP32 `scaled_add` 每元素约 2 FLOPs，最低读 `x,y` 8B、写 `out` 4B，AI 约 `2/12=1/6 FLOP/B`，通常偏内存受限。最低 bytes 不等于实际 DRAM bytes，Cache、重复读取和事务浪费要再测。

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

#### 8.6.1 详细答案

1. **异步 Launch 与延迟报错。** CPU 发射 Kernel 后通常立即继续，GPU 在 Stream 中稍后执行；因此 CPU 计时可能只覆盖 Launch，非法访存也可能到后续同步 API 才报出。调试时在可疑 Launch 后检查错误并临时同步，可把报错归到正确位置；正式性能测试不要把多余同步混入被测区间。
2. **Default、Current 与不同 Stream。** Default Stream 是一个具体 Stream；Current Stream 是框架当前上下文选中的 Stream，可能不是 Default。相同 Stream 内操作按顺序执行，不同 Stream 只有显式 Event/依赖或特定默认流语义才有顺序保证。扩展应使用 PyTorch Current Stream，否则在自定义 Stream 中可能读取尚未就绪的数据或提前暴露输出。
3. **正确 Device 和 Current Stream。** Launcher 用输入 Tensor 的 device 建立 Device Guard，再获取该设备的 Current CUDA Stream并用于 Launch。验证时在非默认 Stream 上异步生成输入、调用扩展、继续消费输出，不做全局同步；再覆盖多 GPU 条件，确认没有发射到错误 Device。
4. **Event 与 CPU Wall Time。** CUDA Event 记录同一设备时间线上的位置，适合测 Kernel 或一段 GPU 工作；CPU Wall Time 适合包含 Python、Launch、同步、数据准备的端到端时间，但结束前必须同步需要完成的 GPU 工作。先写清要回答的是“Kernel 执行多久”还是“调用端到端多久”，再选工具。
5. **最小 Benchmark 协议。** 固定 Shape/dtype/device/实现和输入生成方式；先 Warmup 排除 JIT、缓存和频率爬升；用 Event 记录多次执行，结束后同步；重复多轮并报告分布；保存 GPU、驱动、CUDA、PyTorch、编译参数与时钟/负载环境。所有版本必须完成相同工作和精度契约。
6. **Median、P90/P95 与异常值。** Median 代表典型水平，对少量尖峰不敏感；P90/P95 表示 90%/95% 样本不超过的尾部延迟，能暴露抖动。异常值可能来自首次运行、系统抢占、温度/频率或隐式同步，应该调查并说明，不能只挑最小值，也不能在没有规则时随意删除。

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

#### 9.7.1 详细答案

1. **先定义输入契约。** 在写 Kernel 前写清数学公式、输入输出 Shape、支持 dtype/device/layout、空维度和错误行为、累加精度、Stream 语义及别名/原地修改规则。契约决定 Launcher 检查、Reference、测试矩阵和 Fallback；若实现过程中偷偷改变契约，性能比较就失去意义。
2. **七类接口条件。** Dtype 决定模板和数值路径；Shape 决定输出与边界；Stride/layout 决定地址公式；Device 决定执行位置；Alignment 决定宽访问是否合法；空输入决定是否零 Launch 及 Reduction identity；错误语义决定拒绝还是 Fallback。每项都应有明确代码分支和至少一个测试。
3. **六类测试。** 正常测试验证典型输入；边界测试覆盖 0/1、Warp/Block边界和非整齐 Shape；随机测试扩大组合；极值测试覆盖数值范围、NaN/Inf；错误测试验证非法输入被正确拒绝；回归测试固定保存曾经失败的最小案例。六类回答不同风险，不能用大量随机测试替代其余类别。
4. **最小失败 Shape。** 固定随机种子并逐步缩小维度、dtype、布局和执行路径，直到保留能稳定复现问题的最小输入；记录期望、实际、错误位置和环境。修复后先让该测试失败、再通过，并加入自动测试套件，证明测试确实捕获了原错误。
5. **干净环境复现。** 提供一条构建入口、一条测试入口和一条 benchmark 入口；锁定或记录依赖版本；脚本不得依赖个人绝对路径、未声明环境变量或旧构建产物。清理构建缓存后按 README 从头执行，结果仍一致才算可以交接。
6. **诚实的验证边界。** 文档区分静态检查、CPU 测试、成功编译、真实 GPU 正确性、Sanitizer、性能与 profiler 证据。没有目标 GPU 运行就明确写“未在目标 GPU 验证”，不能把语法检查描述为 CUDA 编译成功，也不能从一次 Shape 的结果推广到所有输入和架构。

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
9. FP16、BF16同为16-bit，它们的指数位、尾数位、动态范围和有效精度有何区别？在低精度reduction中，输入/存储dtype、运算dtype、累加dtype和输出dtype分别是什么？为什么常用FP32累加，它能减少哪些错误、又不能消除哪些误差，应怎样实验验证？
10. `atol + rtol × abs(ref)` 分别解决什么问题？
11. 为什么 CPU 直接包住 kernel launch 的计时通常是错的？
12. PyTorch 当前 stream 与默认 stream 有什么区别？
13. 非连续 Tensor、空输入、非对齐地址分别如何处理？
14. 一次 illegal memory access为什么可能在后面的API才报错？
15. 给定`B、H、query长度Sq、KV长度Skv、query heads数Nh`，怎样先求head维度`D`，再从Q/K分头后的shape推导`Kᵀ`和`QKᵀ`的shape？一个score元素包含多少乘法与加法，为什么常用FLOPs公式是`2×B×Nh×Sq×Skv×D`？该公式包含和不包含哪些Attention步骤，在self-attention、decode、GQA/MQA和causal实现中又怎样变化？
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

**阶段 1 的工作边界：能够独立交付普通 CUDA 小算子、可靠 Reduction 和基础数据变换，并能处理正常工程中的构建、并发、内存、Stream、调试与测量问题。**

---

## 21. 扩展教材一：Python/C++并发——从概念到可回答

### 21.1 先回答什么问题

当GPU时间只有0.1 ms，接口却耗时5 ms时，慢点可能在Python调度、数据准备、进程通信或同步。并发知识不是为了写服务器，而是为了不把Host问题误诊为Kernel问题。

### 21.2 线程、进程、协程的直观模型

把程序想成一家厨房：

- 线程：同一厨房里的多个厨师，共享冰箱和工具；沟通快，但会争用共享物品。
- 进程：不同厨房，各有自己的冰箱；隔离好，但传菜和复制原料成本高。
- 协程：一个厨师在等待烤箱时切菜；适合大量等待，不会让CPU计算自动并行。

| 机制 | 地址空间 | 适合问题                 | 主要风险                  |
| :--- | :------- | :----------------------- | :------------------------ |
| 线程 | 共享     | C++并行、I/O、任务提交   | 数据竞争、死锁、线程安全  |
| 进程 | 隔离     | Python CPU并行、故障隔离 | 启动、序列化、GPU context |
| 协程 | 共享     | 网络/磁盘等待            | 阻塞调用卡住事件循环      |

### 21.3 C++竞态例题

```cpp
int counter = 0;

void work() {
    for (int i = 0; i < 100000; ++i) {
        counter += 1;
    }
}
```

两个线程同时执行时，`counter += 1`包含“读→加→写”，不是不可分割操作，结果可能小于200000。

三种工具解决不同问题：

- `mutex`：保护一段包含多个操作的临界区。
- `atomic`：保证受支持的单个读改写操作原子化。
- `condition_variable`：让线程等待某个状态变化，避免忙等。

`atomic`不能自动保护由多个变量组成的不变量；`mutex`也不能修复错误锁顺序造成的死锁。

### 21.4 Host多线程与CUDA

每个Host线程都要明确：

- 当前CUDA device是什么。
- 当前框架stream是什么。
- 输入Tensor在哪个device。
- 哪个stream生产数据、哪个stream消费数据。
- Event或框架依赖是否建立。

“Python函数返回”不代表GPU任务完成；异步Kernel仍可能在stream中执行。

### 21.5 必做实验与答案

实验：分别用单线程、两个线程、两个进程执行“CPU准备10 ms + GPU Kernel 1 ms”。记录CPU wall time、Kernel event time和时间线。

预期：

- CUDA Event只覆盖GPU工作，不包含Python/进程准备。
- 多线程可能重叠等待，但共享状态需保护。
- 多进程有启动/通信成本，且每个进程需要正确初始化GPU上下文。
- 协程只有在等待可让出控制权时有效。

---

## 22. 扩展教材二：Pinned Memory、异步拷贝与CUDA Graph

### 22.1 Pageable与Pinned

普通Host内存可能被操作系统换页。DMA传输需要稳定物理页，因此异步H2D/D2H通常要求pinned host memory。

直观过程：

```text
Pageable Host → 临时Pinned缓冲 → DMA → Device
Pinned Host --------------------→ DMA → Device
```

Pinned内存不是越多越好：它占用不可换出的物理内存，分配成本也较高，应复用而不是频繁申请释放。

### 22.2 生命周期

异步拷贝提交后，源/目标缓冲不能立即释放或改写。安全点来自：

- 同一stream后续顺序。
- Event完成。
- 明确stream/device同步。

错误做法是提交`cudaMemcpyAsync`后马上复用Host缓冲，而没有等待依赖完成。

### 22.3 拷贝与计算重叠

重叠至少要求：

1. 使用支持异步传输的内存和API。
2. 拷贝与计算位于可并行的stream/依赖关系。
3. 硬件有相应copy engine能力。
4. 工作量足够大，固定开销不会掩盖收益。

### 22.4 CUDA Graph

普通重复执行：

```text
CPU launch A → CPU launch B → CPU launch C → 重复
```

Graph：

```text
Capture A/B/C及依赖 → Instantiate → 多次Replay
```

Graph主要减少重复Host launch和调度开销，不会缩短A/B/C各自的Kernel内部时间。Capture和instantiate是一次性或低频成本，必须与稳态replay分开测。

### 22.5 最小实验

用三个10～20微秒的小Kernel组成固定序列，分别测：

- 普通逐次launch端到端时间。
- Graph首次capture+instantiate。
- Graph重复replay的median/P95。

若单个Kernel本身很长，Graph相对收益通常变小；若shape和控制流频繁变化，Graph管理成本可能不值得。

---

## 23. 扩展教材三：Transpose、Scan与Histogram

### 23.1 Tiled Transpose完整推导

朴素转置：

```cpp
out[x * height + y] = in[y * width + x];
```

同一warp读取`in`通常连续，但写`out`跨越`height`，写访问可能不合并。Tiled版本先连续读取到shared，再交换索引连续写出：

```cpp
__global__ void transpose(float* out, const float* in, int h, int w) {
    __shared__ float tile[32][33];  // +1改变列访问的bank映射
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < w && y < h) tile[threadIdx.y][threadIdx.x] = in[y*w+x];
    __syncthreads();
    int ox = blockIdx.y * 32 + threadIdx.x;
    int oy = blockIdx.x * 32 + threadIdx.y;
    if (ox < h && oy < w) out[oy*h+ox] = tile[threadIdx.x][threadIdx.y];
}
```

为什么是`[32][33]`：按列读取`tile[lane][固定列]`时，若leading dimension为32，相邻lane可能落到同一bank；加1后bank随lane轮转。必须用目标GPU的bank规则和Profiler验证。

### 23.2 Scan

Reduction把多个值变成一个值；scan保留每个前缀结果：

```text
输入：1 2 3 4
inclusive scan：1 3 6 10
exclusive scan：0 1 3 6
```

大输入通常分三层：block内scan、扫描各block总和、把block前缀加回。它教会你跨block算法不能只靠`__syncthreads()`。

### 23.3 Histogram

多个线程可能更新同一bin，直接普通写会竞态。常见方法：

1. Global atomic：简单，冲突高时慢。
2. Block/warp私有直方图：先片上聚合，再合并到global。
3. 分桶/排序：减少冲突，但增加预处理。

性能必须使用真实数据分布；均匀输入和热点bin会产生完全不同的atomic竞争。

---

## 24. 34道阶段出口题参考答案

下面答案用于完成练习后核对。先遮住答案独立作答；需要项目证据的题必须同时给出自己的代码或实验。

1. **调用链：** Python API→动态扩展加载→C++ binding→launcher检查/配置→当前device/stream上的Kernel launch→异步执行→同步点读取结果或暴露错误。
2. **错误分层：** 编译错误发生在源文件生成目标文件前；链接错误是符号无法组合；动态加载错误是`.so`及依赖加载失败；运行错误发生在已加载代码执行时，CUDA错误可能异步延迟出现。
3. **RAII：** 把资源释放绑定对象析构，正常返回和异常展开都会执行析构，减少遗漏清理分支。
4. **线程计算：** `grid=ceil(1000/256)=4`，总线程1024，有效1000，无效24。
5. **层次：** Thread组成warp，warp是SIMT调度组；一个block含多个warp并在一个SM上执行；一个SM可驻留多个block，grid由所有block组成。
6. **Wave tail：** 所有SM并行处理一批可驻留block形成wave；最后一批不足以占满SM就是尾波。小grid直接没有足够block填满设备。
7. **发散：** 主体warp中lane长期走不同重路径时严重；只有最后一个边界warp少量lane退出时通常只是尾部成本。
8. **Occupancy：** 它只描述活跃warp比例。更多warp若都在等待同一瓶颈不会变快；较低occupancy若换来更高复用或指令级并行反而可能快。
9. **连续FP32地址：** lane `l`访问`base + 4×l`，32个lane覆盖连续128 bytes；对齐位置决定需要的sector/transaction。
10. **访存层次：** 一条warp访存形成request，请求被拆成覆盖地址的sector/transaction；cache命中会让L1/L2流量与最终DRAM bytes不同，多余sector造成浪费。
11. **float4：** 16-byte宽访问要求基址/行跨度对齐且尾部不越界；不满足时用标量或masked fallback。
12. **Bank：** 常用概念式为`bank=(byte_address/bank_width)%bank_count`；同一warp不同地址落同bank会拆分，相同地址可广播。具体bank宽度/数量查目标GPU。
13. **Padding：** 只有原访问确有leading-dimension冲突时才可能有用；它还增加shared占用和地址跨度，可能降低驻留或无收益。
14. **Register：** 增加可提高线程私有复用/ILP；代价是降低blocks/SM，过多还可能spill。
15. **Local与spill：** local是线程私有地址空间但物理通常在显存/cache；寄存器不足时编译器把值spill为local load/store。
16. **同步范围：** `__syncthreads()`只同步同一block；不同block可能在不同SM、不同时间执行，需要分Kernel、atomic或支持的全局协作机制。
17. **Reduction映射：** warp reduction用shuffle在寄存器间交换；block reduction先各warp归约，再用shared和barrier合并warp结果。
18. **非2次幂：** 越界lane装载identity，循环只合并有效数据；sum identity为0，max为负无穷，并对尾部显式guard。
19. **Atomic：** 保证单个受支持读改写不可分割；不保证多个变量组成的算法、不保证顺序，也可能高冲突串行化。
20. **FP16/BF16：** FP16尾数较多但指数范围小；BF16指数接近FP32但尾数少。两者通常用FP32累加。
21. **三种dtype：** 输入决定存储/读取，计算与累加决定中间误差/溢出，输出决定最终量化；“支持FP16”无法说明完整语义。
22. **allclose不足：** 它只给通过/失败，还需max/mean/P99误差、NaN/Inf数量、输入范围和reference路径。
23. **Softmax max trick：** 减去最大值不改变归一化比例，并让指数自变量不大于0，避免大正数exp溢出。
24. **Attention shape：** 常见Q为`[B,Nh,Sq,D]`，K/V为`[B,Nkv,Skv,D]`；展开共享后QK为`[B,Nh,Sq,Skv]`，PV输出`[B,Nh,Sq,D]`。
25. **Prefill/Decode：** Prefill一次处理多个query token，矩阵大、并行高；Decode常为`Sq=1`，读取增长的KV，容易带宽/并行度受限。
26. **Current stream：** 框架可能在非默认stream生产输入；错误stream会破坏依赖或产生隐藏同步。用非默认stream生产→调用→event验证。
27. **计时：** CPU wall time测Host端到端，需正确同步；CUDA Event测同stream上的GPU elapsed time。
28. **Warmup：** 首次运行含模块加载、JIT、缓存、分配和时钟爬升，不代表稳态。
29. **偶发错误：** 固定seed并初始化输出；sanitizer查未初始化/越界；重复和racecheck查竞态；用非默认stream+event验证依赖。一次只改变一个因素。
30. **支持边界：** 必须来自自己的API契约，例如CUDA、contiguous、指定dtype/shape/alignment；无法用教材替代，需要在项目README和错误测试中回答。
31. **Python并发：** 线程共享内存、适合I/O/扩展调用；进程隔离、适合CPU并行但有通信/context成本；协程适合可让出控制权的等待。
32. **C++并发：** 数据竞争是无同步共享访问；mutex保护临界区；atomic保证单个操作；条件变量等待状态变化并与mutex配合。
33. **Pinned：** DMA需要稳定物理页；异步拷贝提交后缓冲必须存活且不能提前复用，直到stream/event确认完成。
34. **CUDA Graph：** 主要减少重复Host launch/调度开销；它重放相同Kernel图，不改变单Kernel内部指令和数据路径。

---

## 25. 怎样判断自己是真的“从不会到会”

依次完成三次：

1. **跟写：** 对照正文完成代码，逐行解释索引和契约。
2. **变式：** 修改dtype、shape、block或布局，预测并验证结果。
3. **闭卷：** 一周后从空目录重写，并完成34题和故障实验。

只读完正文属于“认识”；能跟写属于“复现”；能完成变式和故障定位属于“独立应用”；闭卷重写并可交接才是阶段1完成。

---

## 26. 阶段1正式口试18题详细答案

先回答第17章原题，再查看本章。每题达到岗位可用程度，必须同时包含原理、推导或代码路径、自己的实验结果以及适用边界。

### 26.1 Python调用为什么能运行GPU Kernel

**标准回答：** Python先导入编译生成的扩展共享库；pybind或PyTorch Dispatcher把Python调用转换为C++函数调用；C++ binding接收Tensor，launcher完成device、dtype、shape、stride和空输入检查，取得当前device/stream并计算grid/block；随后用CUDA launch语法把设备指针和参数提交到stream；GPU异步调度并执行`__global__` Kernel；后续依赖或同步点读取结果并暴露错误。

**为什么分层：** Python适合用户接口，C++负责框架对象和错误契约，launcher负责实现选择与launch配置，Kernel只负责并行数据计算。分层使输入错误能在Host端尽早报告，Kernel也可独立分析。

**项目证据：** 指向`setup.py → bindings.cpp → launcher → kernels.cu`，用一个元素说明Python索引怎样落到`blockIdx/threadIdx`和地址。

**常见误区：** 认为Python直接执行CUDA源码；认为Python函数返回代表GPU已经完成；只讲“调用C++”而说不出binding、launcher和current stream。

### 26.2 编译成功后为什么仍可能undefined symbol

**标准回答：** 编译只保证单个翻译单元能生成目标文件；链接和动态加载还需要为被引用符号找到兼容定义。声明存在但定义未加入链接、函数签名不一致、模板未实例化、C++ ABI或PyTorch/CUDA动态库版本不一致，都可能在链接或`import .so`时产生undefined symbol。

**分层判断：** 源码语法/类型问题属于编译；构建扩展时找不到符号属于链接；`.so`生成后Python import才失败属于动态加载或运行时依赖解析。

**项目证据：** 保留一次故意删除实现或改变函数签名的复现，记录第一条错误、`nm/ldd`或构建日志和修复方式。

**常见误区：** 看到`.so`存在就认为二进制一定完整；反复重装Python包而不检查符号和ABI。

### 26.3 `N=1000、block=256`怎样计算

```text
blocks = ceil(1000/256) = (1000+255)//256 = 4
threads/block = 256
warps/block = 256/32 = 8
总warps = 4×8 = 32
总threads = 4×256 = 1024
无效threads = 1024-1000 = 24
最后一个warp覆盖索引992～1023，其中992～999有效，所以8个有效lane
```

Kernel必须使用`if (idx < N)`保护尾部。无效lane仍属于已launch线程，但不能访问越界地址。

**项目证据：** 把`N`换成1、31、32、33、255、256、257重复计算，并与Kernel输出/Profiler launch配置核对。

### 26.4 相邻lane访问相邻地址为什么通常更好

同一warp执行一条load/store时，硬件收集32个lane地址。若FP32地址为`base+4×lane`且对齐，所需数据集中在少量连续sector/transaction中；若地址跨很大stride，同样有效字节可能触发更多sector，产生多余传输。

`request`可理解为warp发出的访存需求，`sector/transaction`表示为覆盖这些地址实际访问的内存片段。Sector数明显高于有效字节需要的最低数量，支持访问不合并或对齐浪费的假设；但cache会让L1/L2/DRAM层级流量不同。

**项目证据：** 连续访问与大stride访问保持元素数相同，对比Duration、requests、sectors和DRAM bytes。

**常见误区：** 只因数组在内存中连续就说访问合并；关键是同一warp同一条指令的地址集合。

### 26.5 Shared地址怎样映射到Bank，Padding为何不一定有效

先把二维下标展开成线性地址。对按行存储的`T shared[ROWS][LD]`：

```text
element_index = row × LD + col
byte_address  = shared_base + element_index × sizeof(T)
bank_id       = floor(byte_address / bank_width) % bank_count
```

在常用的教学模型“32个bank、bank宽4字节、`T=float`、基地址按4字节对齐”下，它可简化为：

```text
bank(row, col) = (row × LD + col) % 32
```

但bank conflict不是数组自身的属性，而是**同一warp的同一条shared-memory指令所产生的地址集合**的属性。要把每个lane的`row(lane)`和`col(lane)`代入公式，不能只看声明是`[32][32]`还是`[32][33]`。

#### 例1：同一行连续访问，padding没有东西可优化

若lane `k`访问`shared[r][k]`：

```text
bank(k) = (r × LD + k) % 32
```

`k=0..31`仍会覆盖32个不同bank。`LD=32`时如此，`LD=33`时也只是整体旋转bank编号，不会把“无冲突”变得更好。

#### 例2：同一列跨行访问，`+1`才可能有效

若lane `k`访问`shared[k][c]`：

```text
LD = 32: bank(k) = (k × 32 + c) % 32 = c
         32个lane访问不同地址，却全落到bank c，是经典的32-way冲突

LD = 33: bank(k) = (k × 33 + c) % 32 = (k + c) % 32
         32个lane分布到32个bank，这个访问模式中padding有效
```

#### 例3：项目的`16×16` block中，`+1`甚至可能引入冲突

CUDA中`threadIdx.x`先变化。对`blockDim=(16,16)`，一个warp的32个lane覆盖两行：

```text
row(k) = k / 16
col(k) = k % 16

LD = 16: element_index = (k/16) × 16 + k%16 = k
         bank为0..31，没有冲突

LD = 17: 第0行落到bank 0..15
         第1行落到bank 17..31, 0
         lane 0和lane 31的地址不同却都候选映射到bank 0
```

该项目的NCU对照也显示：`[16][16]`版本的shared store bank conflict为0，`[16][17]`版本反而出现2-way shared-store conflict且Duration增加。这正是“padding是否有效取决于warp的真实访问方向”的项目证据。

还有三个边界需要同时判断：

- 多个lane读取**同一个地址**通常走广播/多播，不能按“同bank”机械地判成冲突。
- 上面的简化公式对FP32最直观；8字节或2字节访问还要考虑一次访问覆盖的bank、指令宽度和硬件拆分的transaction，最终以目标架构和Profiler为准。
- Padding会增加shared占用，可能降低blocks/SM；也可能改变对齐或编译后指令数。即使conflict下降，也必须再看Duration才能宣布优化有效。

**闭卷回答顺序：** 先写`row×LD+col`，再代入每个lane的`row/col`，然后区分“同地址广播”与“不同地址同bank”，最后用NCU的Bank Conflicts/Wavefronts和Duration验证。

### 26.6 Occupancy高为什么仍可能慢

这里的“慢”不是指Occupancy低、Stall百分比高或SM Throughput低，而是指**完成同一份有效工作需要更长时间**。对单个kernel，首先看同一shape、dtype和输出正确性下的CUDA Event/NCU `Duration`；对固定工作量，它等价于有效吞吐下降：

```text
GEMM有效吞吐 = 2 × M × N × K / Duration       （TFLOP/s）
访存类有效带宽 = 完成任务所需的有效bytes / Duration（GB/s）
```

若工作目标是整个API或服务，则“慢”可改用端到端latency或tokens/s表示，但必须写清计时边界。不能把一个kernel的Occupancy与整个服务的latency直接对比。

Occupancy表示active warps相对硬件上限的比例，它只说明SM上有多少候选warp可用于隐藏延迟，但不说明这些warp是否eligible、每条指令是否高效，也不说明为完成任务执行了多少额外工作。例如，所有active warps都在等待内存依赖，或bank conflict/spill使一次逻辑操作被拆成更多实际工作时，Occupancy可以很高，`Duration`仍然可以更长。

增加register可能让每线程保留更多数据、提高复用和ILP；同时可能降低blocks/SM和occupancy。继续增加还可能spill到local memory，产生额外load/store。相反，强行限制register虽提高occupancy，却可能增加spill或重复计算。

**指标分工：** `Duration`、TFLOP/s、GB/s或端到端latency是最终结果指标；Occupancy、eligible warps、issue rate、stall reasons、bank conflicts和spill是解释结果的诊断证据。SM Throughput高只表示SM很忙，也可能是在忙着执行额外指令或等待受限的路径，不能单独当成“更快”。

**正确因果链：** registers/thread → blocks/SM与active warps → eligible warps/issue → spill和数据复用 → Duration/有效吞吐。最终结论不能停在Occupancy或Stall百分比。

### 26.7 33个数怎样用多个Warp归约

“归约”是一类操作：用合并函数`⊕`把多个输入压缩成一个结果，不是特指求和。例如：

| 目标    | 合并操作`⊕` | identity（单位元） |
| :------ | :---------- | :----------------- |
| sum     | `a + b`     | `0`                |
| product | `a × b`     | `1`                |
| max     | `max(a,b)`  | `-∞`               |
| min     | `min(a,b)`  | `+∞`               |
| all     | `a && b`    | `true`             |
| any     | `a \|\| b`  | `false`            |

下面的33个数示例默认是**sum reduction**。例如block有64线程：

```text
thread 0..32 加载 x[0..32]
thread 33..63 加载 sum identity 0
        ↓
每个warp内用 shuffle 和 `+` 做树形合并
        ↓
两个warp的 lane 0 把 warp partial sum 写入 shared
        ↓
__syncthreads()
        ↓
第一个warp的 lane 0..1 读两个 partial，其余 lane 读 0
        ↓
再做一次 warp sum reduction，lane 0 写最终和
```

对应的最小Kernel如下。启动约定是`reduce_sum_33_kernel<<<1, 64>>>(input, output)`，`input`至少有33个`float`，`output`至少有1个`float`：

```cpp
// __global__表示该函数由Host发起、在GPU上执行。
// input指向33个待求和的float，output[0]用于保存最终结果。
__global__ void reduce_sum_33_kernel(const float* input, float* output) {
    // 一个block只有2个warp，因此只需保存两个warp的partial sum。
    // __shared__使同一block的64个线程都能读写这个数组。
    __shared__ float warp_sums[2];

    // 取当前线程在block内的线性编号，启动约定下范围是0..63。
    const int tid = threadIdx.x;

    // warp有32个lane；tid对32取余得到当前线程的lane编号0..31。
    const int lane = tid & 31;

    // tid整除32得到warp编号：tid 0..31属warp 0，tid 32..63属warp 1。
    const int warp_id = tid >> 5;

    // 32个bit全为1，表示当前warp的32个lane都参与shuffle。
    // 这里可以使用FULL_MASK，因为两次shuffle都由完整warp执行且无提前return。
    constexpr unsigned FULL_MASK = 0xffffffffu;

    // tid 0..32各读取一个真实输入：tids 33..63不能访问input，所以装载0。
    // 0是sum的identity：它参与加法不会改变结果，又能让所有线程继续参与同步。
    float value = (tid < 33) ? input[tid] : 0.0f;

    // 第一级归约：两个warp各自在寄存器中计算partial sum。
    // offset按16、8、4、2、1递减，每轮把已覆盖的数量翻倍。
    for (int offset = 16; offset > 0; offset >>= 1) {
        // lane k读取同一warp内lane k+offset的value，再加到自己的value。
        // 循环结束时，每个warp的lane 0持有该warp全32个lane的和。
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }

    // 只让每个warp的lane 0写入partial sum，避免多个lane重复写同一位置。
    if (lane == 0) {
        // warp 0写warp_sums[0]，warp 1写warp_sums[1]。
        // 第二个值实际上就是input[32]，因为warp 1其他31个lane都装载了0。
        warp_sums[warp_id] = value;
    }

    // block级barrier：等待64个线程都到达这里，并保证两次shared写入对后续读取可见。
    // 若没有这一行，warp 0可能在warp 1写完warp_sums[1]之前就开始读取。
    __syncthreads();

    // 第二级只需warp 0执行；warp 1在barrier之后已经完成任务。
    if (warp_id == 0) {
        // lane 0读warp 0的partial，lane 1读warp 1的partial，其他lane装载sum identity 0。
        value = (lane < 2) ? warp_sums[lane] : 0.0f;

        // 再用同样的shuffle树形求和，把两个partial sum合并起来。
        for (int offset = 16; offset > 0; offset >>= 1) {
            // 循环结束后，warp 0的lane 0持有全33个输入的最终和。
            value += __shfl_down_sync(FULL_MASK, value, offset);
        }

        // 只有唯一的最终结果拥有者lane 0向global memory写一次，避免写冲突。
        if (lane == 0) {
            // 将input[0]+...+input[32]写到Host约定的输出位置。
            output[0] = value;
        }
    }
}
```

若改成max reduction，整体的“warp内归约 → 写warp partial → block同步 → 归约partials”骨架不变，但必须同时替换：

```text
合并操作：a + b  → max(a, b)
无效lane填充：0 → -∞
```

不能只把`+`改成`max`却仍用0填尾部，否则全负数输入会错误地得到0。

#### 哪些操作只需“换合并函数”，哪些需要改数据结构

- sum/max/min/product等有合适identity、且可以树形合并的操作：并行归约骨架基本相同，更换`⊕`和identity即可。
- argmax：每个lane保存`(value,index)`，合并时比较value并保留对应index，还要明确相等值时选较小还是较大index。
- mean：不能对不同长度分块的平均值再直接取平均；应归约`(sum,count)`，最后再计算`sum/count`。
- variance：通常归约`(count,mean,M2)`等复合状态，并使用Welford merge；不是单纯把`+`换成另一个符号。
- softmax：可归约局部状态`(m,l)`，合并时需按新的最大值对指数和重缩放。
- subtraction/division不满足常见树形归约所需的结合律，改变合并顺序会改变语义，不能直接套用这个并行骨架。

严格来说，浮点加法也不满足数学上的完全结合律：`(a+b)+c`可能不等于`a+(b+c)`。因此并行sum与CPU串行sum可能有小的舍入差异，需用合理容差验证；若接口要求bitwise deterministic，还要固定归约顺序或使用专门算法。

**不变的并行结构：** 越界线程装载当前操作的identity、warp partial写shared、barrier保证可见性、最后只有一个线程写输出。**会变的是：** 每个lane保存的状态、合并函数、identity、数值精度要求和最终后处理。

**项目证据：** 对sum测`cols=1/31/32/33/63/64/65`，与FP32 reference比较并运行racecheck。若改成max，还必须加入“全负数”和重复最大值测试，用来验证identity与并列tie-breaking。

### 26.8 为什么Barrier前不能让部分线程提前Return

`__syncthreads()`要求同一block中应参与该阶段的线程以一致控制流到达。部分线程提前return而其他线程等待barrier，可能导致未定义行为或死锁；即使某次运行没有挂起，也不能认为正确。

尾部线程应继续参与控制流：将无效输入设置为identity，参加同步和归约，只在内存访问处使用边界guard。只有当整个block在所有barrier之前做出一致决定时，block级提前退出才安全。

**常见误区：** 把`if(idx>=N) return`机械放在任何Kernel开头，而不检查后面是否有block barrier。

### 26.9 FP16/BF16的范围与精度怎样不同，为什么Reduction常用FP32累加

**明确题目：** FP16、BF16同为16-bit，它们的指数位、尾数位、动态范围和有效精度有何区别？在低精度sum reduction中，请分别说明输入/存储dtype、运算dtype、accumulator dtype和输出dtype。为什么常用FP32累加？它能减少哪些溢出或舍入问题，又不能恢复哪些误差？请给出一组能隔离输入量化、累加误差和输出舍入的对照实验。

#### 这道题具体考察什么

合格回答必须覆盖五个点：

1. 能从指数位和尾数位解释“动态范围”与“表示精度”，而不只背“BF16范围大”。
2. 能分清输入/存储dtype、运算dtype、accumulator dtype和输出dtype，不把“FP16输入”误说成“全程FP16”。
3. 能说明长reduction中的溢出、小数被大数吞掉和逐步舍入误差，以及FP32累加为什么能改善它们。
4. 知道FP32累加不能恢复输入量化时已丢失的信息，也不能保证结果绝对精确或bitwise一致。
5. 能设计受控对照，用误差分布、NaN/Inf和最坏输入证明选择，不只说“FP32更准”。

#### 1. 位宽怎样决定范围和精度

| 格式 | 符号位 | 指数位 | 尾数存储位 | 1附近相邻数间隔   | 最小正规格正数 | 最大有限值  | 核心特点                                 |
| :--- | -----: | -----: | ---------: | ----------------: | -------------: | ----------: | :--------------------------------------- |
| FP16 | 1      | 5      | 10         | `2^-10 ≈ 9.77e-4` | 约`6.10e-5`    | `65504`     | 在可表示范围内比BF16精细，但容易范围溢出 |
| BF16 | 1      | 8      | 7          | `2^-7 ≈ 7.81e-3`  | 约`1.18e-38`   | 约`3.39e38` | 指数范围接近FP32，但相邻可表示数间隔更大 |
| FP32 | 1      | 8      | 23         | `2^-23 ≈ 1.19e-7` | 约`1.18e-38`   | 约`3.40e38` | 范围大且精度高，常作累加与reference基准  |

指数位主要决定可表示数的尺度范围；尾数位主要决定同一尺度附近能分得多细。因此：

- FP16比BF16多3个尾数位，在1附近的间隔约小8倍，所以通常表示精度更高。
- BF16和FP32都有8个指数位，所以它们的数量级范围接近；这不表示BF16和FP32一样精确。
- “范围大”回答的是能否表示极大/极小数，“精度高”回答的是附近的两个数能否被区分，两者不是同一件事。

表中列的是最小正规格正数。Subnormal还能表示更小的数，但有效尾数位会逐渐减少；实际运算是否保留subnormal还要核对目标硬件与编译模式。

#### 2. 一条低精度Reduction究竟有哪些dtype

阶段1默认路径是：

```text
FP16/BF16 input存储
    → 每个线程读取后转成FP32
    → 线程局部求和、warp partial和block partial都用FP32
    → 按API契约输出FP32，或最后一次转回FP16/BF16
```

所以必须分别回答：

- **input/storage dtype：** 决定读入前数据已经以多少位存储，也决定了输入量化误差。
- **arithmetic dtype：** 决定当前算术指令按什么精度执行。在本项目的纯sum路径中，输入先转成FP32，加法也以FP32执行。
- **accumulator dtype：** 决定每次部分和的范围与舍入精度；本项目中是FP32。
- **output dtype：** 由API契约决定。若输出低精度，最合理的FP32部分和在写出时仍会再舍入一次，甚至可能溢出。

对纯sum reduction，没有乘法dtype；对dot/GEMM，还必须另外说明乘法输入精度与accumulator精度，不能用“FP32累加”推断乘法也是完整FP32精度。

#### 3. 为什么常用FP32累加

归约会进行多次加法，中间partial sum通常比单个输入大，并且每次加法都可能舍入。FP32累加主要改善：

- **范围：** 例如两个FP16输入`40000 + 40000`。用FP16保存partial sum会超过`65504`；转成FP32后可以在accumulator中表示`80000`。
- **精度：** accumulator越大，低精度中相邻数的绝对间隔越大，后续加入的小数可能不再改变partial sum。FP32的尾数更长，能显著减少这类“小数被大数吞掉”和长链舍入误差。

这是精度与范围的折中，不是“使用了FP32就数学精确”。

#### 4. FP32累加不能解决什么

- 输入从FP32转成FP16/BF16时已经丢失的位无法恢复；转成FP32只是精确保存“量化后的低精度值”。
- FP32加法仍然会舍入，而且不满足严格结合律；不同的并行归约树可能产生小的末位差异。
- 正负大数相消造成的cancellation仍可能放大相对误差。
- 若最后输出转回FP16/BF16，会再舍入；上述`80000`转回FP16时仍会变成Inf。
- FP32累加只规定了accumulator，不自动证明Tensor Core/指令路径、输入转换时机或乘法精度；这些需要结合源码、编译结果和Profiler确认。

#### 5. 怎样做能说明因果的实验

先把同一份FP32输入转成FP16或BF16，然后使用**量化后的同一份输入**做对照：

```text
路径A：低精度input → 低精度accumulator → output
路径B：低精度input → FP32 accumulator → FP32 output
reference：量化后input → FP64或可接受的高精度求和
```

先固定input，才能把A/B的差异归因于accumulator，而不是输入量化不同。至少测：

- `cols=1/31/32/33/255/256/257/4097`，观察误差如何随归约长度增长。
- 全相等值，包括FP16长求和的范围溢出。
- 大数与小数混合，观察小数是否被partial sum吞掉。
- 正负混合和大数相消，观察归约顺序敏感性。
- 分别使用FP32输出和低精度输出，隔离“累加误差”与“最终输出舍入”。

**项目证据：** 报告max/mean/P99 absolute error、合理处理近零reference后的relative error、NaN/Inf数量、最坏shape和输入范围。源码中的FP32局部变量证明设计意图；若要声称具体硬件路径，再用Profiler/SASS证实。

### 26.10 `atol + rtol×abs(ref)`分别解决什么

常见判断为：

```text
abs(actual-ref) <= atol + rtol × abs(ref)
```

#### 先说每个符号是什么

- `actual`：Kernel实际计算的结果。
- `ref`：更可信的参考结果，例如PyTorch/FP32/FP64 reference。
- `abs(actual-ref)`：本次实际产生的绝对误差。
- `atol`：absolute tolerance，**绝对容差**。它是一个与ref大小无关的固定误差底线，单位与结果相同。
- `rtol`：relative tolerance，**相对容差**。它是无单位的比例；`rtol × abs(ref)`把这个比例换算成当前ref尺度下允许的绝对误差。

右边整体`atol + rtol×abs(ref)`是该元素的**最大允许绝对误差**。两项是相加，不是“满足atol或满足rtol任意一个”。

#### `atol`解决接近0的值

若`ref=0`，则相对项`rtol×abs(ref)=0`。此时只能由`atol`给出一个可接受的绝对误差范围：

```text
ref = 0
atol = 1e-5
rtol = 1e-3

允许误差 = 1e-5 + 1e-3 × 0 = 1e-5
actual = 8e-6  → 通过
actual = 2e-5  → 不通过
```

没有`atol`时，近零reference即使只有很小的浮点误差，也可能被不合理地判错。

#### `rtol`解决数值尺度变大

同样的绝对误差，相对于`0.001`可能很大，相对于`1000`却可能很小。`rtol`让允许误差随`abs(ref)`按比例增长：

```text
ref = 100
atol = 0.01
rtol = 0.001                 // 允许0.1%的相对误差

允许误差 = 0.01 + 0.001 × 100 = 0.11
actual = 100.08  → abs(actual-ref)=0.08，通过
actual = 100.20  → abs(actual-ref)=0.20，不通过
```

**一句话记忆：** `atol`是近零时的固定余量，`rtol`是随参考值变大的比例余量。

#### 使用边界

容差必须结合dtype、归约长度和reference路径制定。过严会拒绝合理浮点差异，过松会掩盖索引错误。还应单独检查NaN/Inf，因为普通比较可能产生误导。

- `atol`和`rtol`是测试事先设定的标准，不是从当前失败结果反向放宽到刚好通过。
- 该公式以`ref`为尺度，因此不是对称距离；交换`actual`和`ref`可能改变阈值。
- 只看`allclose=True`不足以证明数值质量，还应报告max/mean/P99 absolute error、relative error和NaN/Inf数量。

### 26.11 为什么CPU直接包Kernel Launch计时错误

CUDA launch通常异步：CPU提交Kernel后立即继续，`t1-t0`主要测Host launch开销，而非GPU完成时间。

单Kernel稳态时间应用同一stream上的CUDA Event：warmup后记录start event、重复launch、记录end event并同步end，再计算elapsed。端到端API时间可用CPU wall time，但结束前要在正确边界同步，并写清是否包含分配、拷贝和转换。

**项目证据：** 对同一Kernel展示“无同步CPU计时”“同步CPU计时”“CUDA Event”三组结果并解释差异。

### 26.12 Current Stream与Default Stream

#### 1. Stream先是什么

CUDA stream可以理解为**一条按顺序排列GPU工作的命令队列**。CPU会把Kernel launch、异步内存拷贝和event等工作放入某条stream，然后通常不等GPU完成就继续执行Host代码。

可以把GPU想成一个厨房，把stream想成一张按顺序写好的出菜单：

```text
stream S:
    1. 把输入拷到GPU
    2. 运行kernel A产生x
    3. 运行kernel B读取x并产生y
```

同一条stream中，GPU必须保证前面的工作先于后面的工作完成其可见效果。因此kernel B可以安全读取kernel A写出x，不需要在两者之间调用全局同步。

Stream不是CPU thread，也不是一个Kernel；它是GPU工作的排序与依赖语义。

#### 2. 多条Stream意味着什么

不同stream中的工作在硬件资源和依赖允许时**可能**并发或重叠，但“放到不同stream”不保证一定同时执行。更重要的是，不能默认一条stream会等待另一条stream的普通工作：

```text
stream A: kernel A 写 x
stream B: kernel B 读 x
```

若B依赖A，必须建立明确顺序，例如让B等待A上记录的CUDA Event，或使用PyTorch的`wait_stream`。否则B可能在A写完x之前就读取，造成竞态。

#### 3. Default Stream是什么

Default Stream是CUDA为每个设备提供的特殊stream。当代码没有显式选择其他stream时，GPU工作通常会提交到它。

“default”只表示缺省选择，不表示它比其他stream更快或更正确。CUDA还有legacy default stream与per-thread default stream等模式，它们与其他stream的隐式同步规则不完全相同。阶段1不需要背完这些规则，但不能依赖模糊的“default stream应该会自动等”假设。

#### 4. PyTorch Current Stream是什么

PyTorch Current Stream是**PyTorch当前上下文为某个CUDA device选中的stream**。它初始时通常是default stream，但用户进入`torch.cuda.stream(s)`上下文后，current stream就会暂时变成`s`：

```python
s = torch.cuda.Stream()

with torch.cuda.stream(s):
    x = make_input_on_cuda()  # 提交到s
    y = my_cuda_extension(x)  # 扩展也应提交到s
```

在这个例子中，`x`的生产和扩展对`x`的读取放在同一条stream `s`上，因此顺序自然正确。

如果扩展忽略current stream，却私自launch到default stream，就变成：

```text
stream s:       生产x
default stream: 读取x
```

这两件工作不再天然位于同一条有序队列中，可能在`x`尚未生产完成时读取，或因为隐式同步而丢失并发性能。所以PyTorch CUDA扩展应在正硤的device上取得并使用current stream，例如：

```cpp
at::cuda::CUDAGuard device_guard(input.device());
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
my_kernel<<<grid, block, 0, stream>>>(/* arguments */);
```

`CUDAGuard`负责选对输入所在的CUDA device，`getCurrentCUDAStream()`负责选对该device当前的工作队列。

#### 5. Stream和同步的关系

- 同一stream内：已有提交顺序，一般不需要在每个Kernel后调用全局同步。
- 不同stream之间有依赖：用Event、`wait_stream`等显式建立顺序。
- `stream.synchronize()`：Host等待该stream之前提交的工作完成。
- `cudaDeviceSynchronize()`：Host等待该device上的工作，范围更大，调试时有用，但不应随意放进性能路径。

**一句话记忆：** Stream是GPU工作的有序队列；Default Stream是CUDA的缺省队列，Current Stream是PyTorch此刻要你跟随的队列。

**验证：** 在自建stream中先异步生成输入，再调用扩展并记录event；不做全局同步，确认输出正确和依赖成立。这个实验证明的是扩展遵守current-stream语义，不只是“在default stream上碰巧算对”。

### 26.13 非连续、空输入和非对齐怎样处理

这三个词描述的不是同一件事：

| 问题   | 它在问什么                               | 典型风险                                                        |
| :----- | :--------------------------------------- | :-------------------------------------------------------------- |
| 非连续 | 逻辑上相邻的元素，在storage中是否也相邻  | Kernel按错误地址读数，结果错却可能不越界                        |
| 空输入 | 是否根本没有元素，或某个归约维长度为0    | 零block launch、越界读取或输出shape/语义错误                    |
| 非对齐 | 某个地址是否为向量指令所需字节数的整数倍 | 强行向量load/store可能非法、被拆成更多transaction或读到尾部之外 |

它们要分别判断。一个Tensor可以“连续但非16-byte对齐”，也可以“首地址16-byte对齐但逻辑布局不连续”。

#### 1. 什么是非连续Tensor

Tensor不只有shape，还有stride。Stride表示“某个维度的下标增加1时，需要在storage中跨过多少个元素”。对二维Tensor：

```text
真实element offset = row × stride[0] + col × stride[1]
```

例如：

```python
a = torch.arange(6).reshape(2, 3)
# a = [[0, 1, 2],
#      [3, 4, 5]]
# shape=(2, 3), stride=(3, 1)，按行连续

x = a.t()
# x = [[0, 3],
#      [1, 4],
#      [2, 5]]
# shape=(3, 2), stride=(1, 3)，逻辑上转置了，但storage没有重排
```

若Kernel无视stride，对`x[row][col]`仍使用连续假设：

```text
错误offset = row × cols + col
```

它会按`[[0,1],[2,3],[4,5]]`的顺序读取，而不是用户看到的`[[0,3],[1,4],[2,5]]`。这种错误往往仍在已分配storage范围内，所以可能不报illegal memory access，只是悄悄算错。

##### 非连续输入有三种合法契约

**方案A：明确拒绝。** Kernel只实现连续索引，launcher在取`data_ptr`之前检查：

```cpp
TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
```

这是当前阶段1项目的选择。好处是Kernel简单，调用者也能清楚看到接口限制。

**方案B：显式转成连续副本。**

```cpp
auto x_contiguous = x.contiguous();
```

这样可以复用连续Kernel，但非连续输入会多一次内存分配和数据拷贝。API若选择该契约，benchmark必须明确copy是否包含在计时中，不能偷偷copy后只报Kernel时间。

**方案C：Kernel真正支持stride。** Host把stride传给Kernel，Kernel按逻辑坐标计算真实offset：

```cpp
int64_t offset = row * stride0 + col * stride1;
float value = input[offset];
```

这不需要copy，但索引更复杂，且某些stride会让warp访问变得不合并。“支持正确”不等于“与contiguous一样快”。

**项目结论：** 本项目选方案A：拒绝non-contiguous，并用错误测试证明契约；不能把非连续Tensor的`data_ptr`当成紧密一维数组使用。

#### 2. 什么是空输入，为什么要在Launch前处理

只要任意一个维度为0，Tensor的`numel()`就是0，例如`shape=(0, 7)`和`shape=(3, 0)`。它们都没有输入元素，但对行求和来说，输出语义不同：

```text
x.shape = [0, 7]
没有任何行
输出shape = [0]，返回空Tensor

x.shape = [3, 0]
有3行，但每行都没有数
sum的identity是0
输出shape = [3]，值为[0, 0, 0]
```

所以不能只写“`numel()==0`就随便返回空Tensor”，必须先按算子的数学定义推导输出shape和值。Sum空集合的identity是0，product是1。浮点max/min可分别把`-Inf/+Inf`当作identity，但整数dtype、argmax的index语义或框架API未定义初值时仍可能要求报错；必须与目标API契约对齐。

还要在launch前提前返回，原因是：

- Kernel不应解引用没有元素的`data_ptr`。
- 常见grid公式`blocks=(n+threads-1)/threads`在`n=0`时得到0；启动零个block不是一个有意义的正常Kernel launch。
- 即使可以安全绕过访存，也没必要为“什么都不做”支付launch开销。

当前项目的Host端处理是：

```cpp
// 元素级scaled_add：输入是什么shape，输出就是什么shape。
auto out = torch::empty_like(x);
if (x.numel() == 0) {
    return out;                         // 不launch Kernel
}

// 二维行sum：分开处理“没有行”与“行是空集合”。
if (rows == 0) {
    return torch::empty({0}, x.options());
}
if (cols == 0) {
    return torch::zeros({rows}, x.options());
}
```

**项目结论：** 空输入不是统一地“拒绝”或“返回空”；先定义算子语义，再在Host launcher中构造正确输出并避免Kernel launch。

#### 3. 什么是地址对齐

对齐的意思是“地址是某个字节边界的整数倍”。例如`float4`一次处理4个FP32，共16字节，因此向量路径通常要求首地址满足：

```text
address % 16 == 0
```

例如地址`0x1000`可被16整除，是16-byte aligned；`0x1004`只相差4字节，对FP32标量读取没问题，但不满足`float4`的16-byte对齐要求。

为什么“contiguous”不能代替“aligned”？因为切片可以保持紧密布局，却改变首地址：

```python
base = torch.randn(1025, device="cuda", dtype=torch.float32)
x = base[1:1025]

assert x.is_contiguous()       # 元素仍然一个接一个
assert x.data_ptr() % 16 != 0  # 首地址相对base偏移了4字节
```

向量化路径不只要检查首地址，还要检查访问范围：

- `x/y/out`每个参与向量load/store的指针都要满足16-byte对齐。
- 平铺`float4`路径需要`numel % 4 == 0`，否则最后不足4个元素的是尾部。
- 行级向量化还要保证每行起点对齐。对contiguous FP32矩阵，`cols % 4 == 0`才能使下一行仍从16-byte边界开始。
- 对齐只说明地址满足指令要求，不能代替边界检查；对齐的指针也可能在最后一次`float4`读取时越过分配范围。

强行把不满足条件的`float*`解释成`float4*`不是“可能只慢一点”的正确fallback。它可能导致非法/非预期访问，也可能让一次访问跨越更多memory transaction。

##### 正确的快路径/fallback选择

```cpp
bool aligned16 =
    reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0;

bool can_use_float4 =
    x.scalar_type() == at::kFloat &&  // float4此处表示4个FP32
    x.numel() % 4 == 0 &&             // 没有1..3个元素的尾部
    aligned16;                        // 所有向量指针都16-byte对齐

if (can_use_float4) {
    // 快路径：每个线程使用float4处理4个元素。
} else {
    // 安全fallback：使用float标量load/store和普通idx<n边界判断。
}
```

也可以设计“对齐的主体走向量路径，最后1～3个元素走标量tail”，但必须使用正确的元素索引和边界，不能让两条路径重复或漏算。

**项目结论：** 标量基础Kernel不需要额外的16-byte对齐契约；`float4`练习只在基址、长度和行跨度都满足条件时走快路径，否则自动走标量fallback，不应报错或强行向量访问。

#### 4. Launcher中的完整判断顺序

```text
1. 检查device、dtype、shape及多输入关系
2. 检查layout契约：支持stride，还是拒绝non-contiguous
3. 根据算子语义处理空shape，需要时直接返回，不launch
4. 创建输出，选择正确device和current stream
5. 若有向量快路径，检查dtype、所有基址、行跨度和tail
6. 条件全部满足才launch快路径，否则launch安全fallback
7. 做launch error check，保持正常异步语义
```

#### 5. 怎样用测试证明契约

- **non-contiguous：** 使用`x = torch.randn(4,7,device="cuda").t()`，确认本项目抛出包含`contiguous`的明确错误。
- **空元素算子：** 对scaled-add测shape `[0]`、`[2,0,3]`，确认输出shape/dtype/device与契约一致。
- **空归约：** 对row-sum分别测`[0,7] → [0]`和`[3,0] → [0,0,0]`。
- **非对齐但连续：** 用`base[1:]`制造`data_ptr()%16!=0`，确认进入标量fallback且结果正确。
- **tail：** 使用元素数不能被4整除的shape，确认最后1～3个元素没有越界、丢失或重复计算。
- **路径证据：** 测试正确性只能证明“算对”；还应通过调试标记、独立Kernel名称或Profiler确认aligned样例走快路径，misaligned/tail样例走fallback。

**一句话记忆：** 非连续看stride，空输入看数学语义，非对齐看地址和访问宽度；三者都必须在launch前有明确契约。

### 26.14 Illegal Memory Access为何延迟报错

Kernel launch异步，Host提交时只完成参数入队。GPU稍后执行到越界访问，错误状态可能在下一个同步API、内存拷贝、event同步或其他CUDA调用才返回，因此报错栈不一定是真正错误行。

定位方法：缩小shape、固定seed；在调试版本中在可疑launch后同步；检查launch错误；用compute-sanitizer定位地址；从第一个错误输出反推线程与索引。调试同步不能保留在性能路径。

### 26.15 Attention中QKᵀ的Shape和FLOPs

**明确题目：** 给定batch size `B`、hidden size `H`、query长度`Sq`、KV长度`Skv`和query head数`Nh`，先计算每个head的维度`D`，再从Q/K分头后的shape推导`Kᵀ`及attention scores的shape。随后从“输出元素个数 × 每个点积的运算量”推导`QKᵀ` FLOPs，并说明该公式的计数范围。最后解释self-attention、decode、GQA/MQA和causal mask分别改变哪个维度或计数。

#### 这道题具体考察什么

合格回答需要覆盖以下六点：

1. 知道`H`、head数`Nh`和head dimension `D`的关系，并检查`H % Nh == 0`。
2. 能从`[B,S,H]`推到`[B,Nh,S,D]`，知道“分head”不是丢掉或新增元素。
3. 知道`Kᵀ`只交换K最后两个矩阵维度`[Skv,D] → [D,Skv]`，batch和head维不动。
4. 能解释scores的每个元素代表“一个query位置与一个key位置在某个head中的D维点积”。
5. 能从点积推导FLOPs，而不是死背`2BNS²D`。
6. 能明确公式只计算`QKᵀ`，并区分逻辑有效FLOPs和Kernel实际执行FLOPs。

这题不要求在阶段1推导完整FlashAttention或反向传播；重点是shape、点积和计数边界。

#### 1. 先定义每个符号

```text
B：batch size
H：模型hidden size
Nh：query/attention head数量
D：每个head的维度；普通MHA中 D = H / Nh
Sq：query token数量
Skv：key/value token数量
```

必须先检查：

```text
H % Nh == 0
```

否则无法把`H`均匀拆成`Nh`个相同大小的head。普通self-attention中`Sq=Skv=S`；cross-attention、带KV cache的decode或其他非对称场景中，`Sq`和`Skv`可以不同。

#### 2. Q和K是怎样变成多头Shape的

先看普通MHA。Q投影后的逻辑shape通常是：

```text
Q projection output: [B, Sq, H]
H = Nh × D
reshape:             [B, Sq, Nh, D]
transpose/reorder:   [B, Nh, Sq, D]
```

K同理：

```text
K projection output: [B, Skv, H]
reshape:             [B, Skv, Nh, D]
transpose/reorder:   [B, Nh, Skv, D]
```

`reshape`只是把最后一个`H`拆成`Nh×D`；随后交换head维与sequence维，使每个batch、每个head都能看到一张独立矩阵。

因此分头后的Q和K是：

```text
Q: [B, Nh, Sq,  D]
K: [B, Nh, Skv, D]
```

#### 3. Kᵀ到底转置什么，QKᵀ为什么得到这个Shape

这里的转置不是把整个四维Tensor倒过来，而是只交换K最后两个矩阵维度：

```text
K:  [B, Nh, Skv, D]
Kᵀ: [B, Nh, D, Skv]
```

`B`和`Nh`是batched matrix multiplication的批次维。对固定的batch `b`和head `h`，真正相乘的是：

```text
Q[b,h]:  [Sq, D]
Kᵀ[b,h]: [D, Skv]
          ─────────
Scores:   [Sq, Skv]
```

把所有batch和head放回来：

```text
Scores = Q @ K.transpose(-2, -1)
Scores shape = [B, Nh, Sq, Skv]
```

每个元素的定义是：

```text
Scores[b,h,i,j]
    = Σ(d=0..D-1) Q[b,h,i,d] × K[b,h,j,d]
```

所以`Scores[b,h,i,j]`表示：第`b`个样本、第`h`个head中，第`i`个query token和第`j`个key token的未缩放相似度。

一个常见错误是把输出写成`[B,Nh,Sq,D]`。这是后面的`softmax(Scores) @ V`输出shape；`QKᵀ`本身的最后一维是key位置`Skv`，不是head维度`D`。

#### 4. FLOPs怎样一步步推导

先数输出元素：

```text
Scores元素数 = B × Nh × Sq × Skv
```

每个score元素是长度`D`的点积：

```text
D次乘法
D-1次加法
精确标量计数 = 2D - 1 FLOPs
```

性能分析通常把一次multiply-add/FMA按2 FLOPs计算，并忽略最后少一次加法，于是每个点积近似为`2D` FLOPs：

```text
FLOPs_QK
≈ 输出元素数 × 每个元素的点积FLOPs
= (B × Nh × Sq × Skv) × (2D)
= 2 × B × Nh × Sq × Skv × D
```

若题目要求严格的朴素标量运算次数，可写：

```text
Exact scalar FLOPs = B × Nh × Sq × Skv × (2D - 1)
```

但GPU性能报告通常使用约定公式`2×B×Nh×Sq×Skv×D`，因为硬件FMA按乘法与加法各1 FLOP计算。

#### 5. 用一组数字完整计算

给定：

```text
B = 2
Sq = Skv = S = 128
H = 768
Nh = 12
```

先算：

```text
D = H / Nh = 768 / 12 = 64

Q  = [2, 12, 128, 64]
K  = [2, 12, 128, 64]
Kᵀ = [2, 12, 64, 128]
Scores = [2, 12, 128, 128]
```

再算FLOPs：

```text
FLOPs_QK
= 2 × 2 × 12 × 128 × 128 × 64
= 50,331,648 FLOPs
≈ 50.3 MFLOPs
```

可用最小PyTorch shape实验核对：

```python
import torch

B, Nh, Sq, Skv, D = 2, 12, 128, 128, 64
q = torch.randn(B, Nh, Sq, D, device="cuda")
k = torch.randn(B, Nh, Skv, D, device="cuda")
scores = q @ k.transpose(-2, -1)
assert scores.shape == (B, Nh, Sq, Skv)
```

该实验只验证shape和运算定义，不等于证明某个自定义Kernel的实际指令路径或性能。

#### 6. 这个FLOPs公式包含什么、不包含什么

`2×B×Nh×Sq×Skv×D`只表示`QKᵀ`这一次batched matrix multiplication的常用算法FLOPs。它不包含：

- 从输入`X`生成Q/K/V的线性projection。
- Q/K的reshape、transpose或实际layout copy。
- 对scores乘`1/sqrt(D)`的缩放。
- 添加causal/padding mask。
- softmax、dropout。
- `softmax(Scores) @ V`。
- output projection。

其中scale大约还会对每个score做一次乘法；mask和softmax有自己的操作量。做性能报告时必须明确报告的是“仅QK Kernel”还是“完整Attention端到端”，不能把两者的时间与同一个FLOPs分子混用。

#### 7. Self-Attention、Decode、GQA/MQA和Causal怎样变化

**普通self-attention：**

```text
Sq = Skv = S
Scores shape = [B, Nh, S, S]
FLOPs ≈ 2 × B × Nh × S² × D
```

这里的`S²`来自每个query位置都与每个key位置配对。

**带KV cache的单token decode：**

```text
Sq = 1
Skv = 当前KV cache长度
Scores shape = [B, Nh, 1, Skv]
FLOPs ≈ 2 × B × Nh × Skv × D
```

因此decode的QK计算量对当前cache长度近似线性增长，而不是对`Skv`平方增长。Prefill通常有多个query token，self-attention时才表现为`S²`。

**GQA/MQA：**

设query heads为`Nq`，KV heads为`Nkv`，通常`Nq % Nkv == 0`。多个query head共享一个K/V head：

```text
Q: [B, Nq,  Sq,  D]
K: [B, Nkv, Skv, D]
Scores after head mapping: [B, Nq, Sq, Skv]
FLOPs ≈ 2 × B × Nq × Sq × Skv × D
```

最终scores按query heads `Nq`计数，因为每个query head仍要产生自己的score矩阵。`Nkv`减少主要降低K/V projection、KV cache容量和读取量；它不会直接把score输出的head维从`Nq`改成`Nkv`。MQA是`Nkv=1`的特殊情况。

只给`H`和`Nh`不足以完整描述GQA，还必须知道`Nq`、`Nkv`及head映射规则。

**Causal self-attention：**

逻辑上，第`i`个query只能看不晚于自己的key。若Kernel真正跳过上三角，允许的query-key对数是：

```text
S × (S + 1) / 2
```

相应的有用FLOPs近似为：

```text
2 × B × Nh × D × S(S+1)/2
```

但许多朴素实现仍计算完整`S×S`分数后再加mask，这时Kernel实际执行的QK FLOPs仍接近完整公式。不能仅因语义上是causal就直接把Profiler中的实际FLOPs除以2；必须看实现是否真的跳过被mask区域。

#### 8. 闭卷回答顺序

```text
先写 D = H / Nh
→ 写 Q/K 分头后的shape
→ K只转置最后两维
→ 写 Scores = [B,Nh,Sq,Skv]
→ 数输出元素 B×Nh×Sq×Skv
→ 每个元素是长度D的点积，约2D FLOPs
→ 得到 2×B×Nh×Sq×Skv×D
→ 最后声明公式边界及self/decode/GQA/causal变化
```

**项目证据：** 至少保留一组手算shape和FLOPs，用PyTorch matmul核对输出shape；若报告自定义Kernel的Achieved TFLOP/s，则用同一shape的约定FLOPs除以CUDA Event/NCU `Duration`，并明确是否实际计算完整causal矩阵。

### 26.16 怎样用FLOPs/Bytes判断理论瓶颈

先计算有效FLOPs和所选层级最低必要bytes：

```text
AI = FLOPs / bytes
ridge_point = peak_compute / peak_bandwidth
```

若AI远低于ridge point，Roofline上限倾向带宽侧；若远高于，倾向计算侧。但这是宏观上限，不是自动根因：小grid、launch、同步、指令依赖、cache和无效工作都可能让Kernel达不到roof。

**项目证据：** 为scaled_add/reduction写FLOPs和最低bytes，比较有效GB/s、实际DRAM bytes和Duration，再说明判断边界。

### 26.17 性能结果至少报告什么

环境：GPU型号、驱动、CUDA、PyTorch/扩展版本、编译参数、代码commit。契约：shape、dtype、layout、累加、输入分布、是否含转换/分配。协议：warmup、重复次数、同步、计时器。结果：median、P90/P95、波动、吞吐或GB/s、正确性误差。基线：PyTorch/reference和前一版本。结论：为什么变化、Profiler证据、代价、适用shape和失败结果。

只报“快了30%”无法复现，也无法判断是否少算、精度不同或噪声造成。

### 26.18 两个工单的限制怎样回答

这题没有统一项目数字，必须依据自己的README。合格答案示例：scaled_add只支持CUDA contiguous FP32/FP16/BF16、相同shape，不支持广播/stride；低精度在FP32中计算；向量路径未对齐时fallback。Reduction只处理二维最后一维sum，使用固定block策略，超长行不是最优，不支持任意axis或autograd。

阶段1允许这些限制，因为目标是证明完整调用链、正确性、stream、边界、Reduction和测量方法，而不是构建通用算子库。前提是限制显式、错误安全、测试覆盖，并说明进入生产前需要补什么。

### 26.19 复测标准

每题制作卡片：正面写原题，背面只保留“三句核心答案、代码位置、证据文件”。一周后随机抽12题，至少10题能在不看文档时回答；其中第3～14题不能有结构性错误。只能复述本章、不能指向自己的工程证据时，最高记为“能解释”，不能记为“独立交付”。
