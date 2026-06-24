const matrices = {
  A: [
    [1, 2, 1, 0],
    [0, 1, 3, 1],
    [2, 0, 1, 2],
    [1, 1, 0, 2],
  ],
  B: [
    [1, 0, 2, 1],
    [2, 1, 0, 1],
    [1, 3, 1, 0],
    [0, 1, 2, 2],
  ],
};

matrices.C = matrices.A.map((row, i) =>
  matrices.B[0].map((_, j) =>
    row.reduce((sum, value, k) => sum + value * matrices.B[k][j], 0)
  )
);

const modeConfig = {
  naive: {
    title: "一个线程计算一个输出",
    summary: "线程直接从 Global Memory 读取 A 的一行和 B 的一列，完成一个点积。",
    stats: ["1 个 C 元素", "每次乘加都读取", "几乎没有", "16 × 16 输出"],
    concept: 0,
  },
  tiled: {
    title: "一个 Block 协作加载并复用 Tile",
    summary: "线程先把 A/B 小块搬进 Shared Memory，再共同使用这份数据完成多个输出。",
    stats: ["1 个 C 元素", "每个 Tile 协作读取一次", "Block 内复用", "16 × 16 输出"],
    concept: 1,
  },
  regtile: {
    title: "一个线程计算相邻 2×2 输出",
    summary: "线程从 Shared Memory 读取 2 个 A 和 2 个 B，在寄存器中组合成 4 次乘加。",
    stats: ["4 个 C 元素", "按 Tile 协作读取", "Block 内 + 线程内", "32 × 32 输出"],
    concept: 2,
  },
};

function rangeCells(matrix, coordinates, className) {
  return coordinates.map(([row, col]) => ({ matrix, row, col, className }));
}

function sourceLine(number, text, active = false) {
  return { number, text, active };
}

const sourceByMode = {
  naive: [
    {
      location: "gemm.cu:11-15",
      explanation: "动画中选中的 C[0,0] 对应 row=0、col=0；acc 是当前线程私有的寄存器累加器。",
      lines: [
        sourceLine(11, "int row = blockIdx.y * blockDim.y + threadIdx.y;", true),
        sourceLine(12, "int col = blockIdx.x * blockDim.x + threadIdx.x;", true),
        sourceLine(13, "if (row >= M || col >= N) return;"),
        sourceLine(15, "float acc = 0.0f;", true),
      ],
    },
    ...[0, 1, 2, 3].map((k) => ({
      location: "gemm.cu:16-17",
      explanation: `动画把循环展开到 k=${k}。代码每轮直接从 Global Memory 读取 A[row,k] 和 B[k,col]，再累加到 acc。`,
      lines: [
        sourceLine(16, "for (int k = 0; k < K; ++k) {", true),
        sourceLine(17, "    acc += A[row * K + k] * B[k * N + col];", true),
        sourceLine(18, "}"),
      ],
    })),
    {
      location: "gemm.cu:19",
      explanation: "动画中 C[0,0] 从圆点变成最终数值，对应线程把寄存器 acc 写回 Global Memory 中的 C。",
      lines: [
        sourceLine(19, "C[row * N + col] = acc;", true),
      ],
    },
    {
      location: "gemm.cu:11-19, 122-124",
      explanation: "Host 端用 16×16 线程组成 Block；kernel 中每个线程定位并计算一个输出，所以一个 Block 覆盖 16×16 个 C 元素。",
      lines: [
        sourceLine(11, "int row = blockIdx.y * blockDim.y + threadIdx.y;"),
        sourceLine(12, "int col = blockIdx.x * blockDim.x + threadIdx.x;"),
        sourceLine(16, "for (int k = 0; k < K; ++k) {"),
        sourceLine(17, "    acc += A[row * K + k] * B[k * N + col];", true),
        sourceLine(19, "C[row * N + col] = acc;"),
        sourceLine(122, "dim3 block(16, 16);", true),
        sourceLine(123, "dim3 grid(ceil_div_int(N, block.x), ceil_div_int(M, block.y));"),
        sourceLine(124, "gemm_naive_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);", true),
      ],
    },
  ],
  tiled: [
    {
      location: "gemm.cu:22, 28-35, 134-136",
      explanation: "TILE=16 决定 Shared Memory 小块、线程 Block 和输出 Tile 的尺寸；动画为便于观察缩小成 2×2。",
      lines: [
        sourceLine(22, "constexpr int TILE = 16;", true),
        sourceLine(28, "__shared__ float As[TILE][TILE];", true),
        sourceLine(29, "__shared__ float Bs[TILE][TILE];", true),
        sourceLine(31, "int row = blockIdx.y * TILE + threadIdx.y;"),
        sourceLine(32, "int col = blockIdx.x * TILE + threadIdx.x;"),
        sourceLine(35, "for (int t = 0; t < K; t += TILE) {"),
        sourceLine(134, "dim3 block(TILE, TILE);", true),
        sourceLine(136, "gemm_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);"),
      ],
    },
    {
      location: "gemm.cu:35-39",
      explanation: "动画中的 Global→Shared 搬运对应每个线程各加载一个 A 元素和一个 B 元素，越界位置填 0。",
      lines: [
        sourceLine(35, "for (int t = 0; t < K; t += TILE) {"),
        sourceLine(36, "    int a_col = t + threadIdx.x;"),
        sourceLine(37, "    int b_row = t + threadIdx.y;"),
        sourceLine(38, "    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;", true),
        sourceLine(39, "    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;", true),
      ],
    },
    {
      location: "gemm.cu:40",
      explanation: "动画中的等待屏障就是这次 __syncthreads()：确保 As/Bs 已被整个 Block 完整写入。",
      lines: [
        sourceLine(38, "As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;"),
        sourceLine(39, "Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;"),
        sourceLine(40, "__syncthreads();", true),
      ],
    },
    {
      location: "gemm.cu:42-45",
      explanation: "动画中多个输出线程复用同一 Tile，对应每个线程从 Shared Memory 读取自己需要的一行 As 和一列 Bs。",
      lines: [
        sourceLine(42, "#pragma unroll"),
        sourceLine(43, "for (int kk = 0; kk < TILE; ++kk) {", true),
        sourceLine(44, "    acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];", true),
        sourceLine(45, "}"),
      ],
    },
    {
      location: "gemm.cu:35-46",
      explanation: "动画进入下一个 K Tile，对应 t 增加 TILE；Shared Memory 被下一块数据覆盖，acc 保留并继续累加。",
      lines: [
        sourceLine(35, "for (int t = 0; t < K; t += TILE) {", true),
        sourceLine(38, "    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;"),
        sourceLine(39, "    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;"),
        sourceLine(44, "    acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];", true),
        sourceLine(46, "    __syncthreads();"),
        sourceLine(47, "}"),
      ],
    },
    {
      location: "gemm.cu:49",
      explanation: "动画中的四个输出由四个线程分别写回；每个线程仍只持有一个 acc，并在边界检查后写入 C。",
      lines: [
        sourceLine(49, "if (row < M && col < N) C[row * N + col] = acc;", true),
      ],
    },
    {
      location: "gemm.cu:28-49",
      explanation: "完整 Tiled 数据流：Global Memory 协作加载到 As/Bs，Block 同步，Shared Memory 中复用，最后写回 C。",
      lines: [
        sourceLine(28, "__shared__ float As[TILE][TILE];", true),
        sourceLine(29, "__shared__ float Bs[TILE][TILE];", true),
        sourceLine(38, "As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;"),
        sourceLine(39, "Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;"),
        sourceLine(40, "__syncthreads();"),
        sourceLine(44, "acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];", true),
        sourceLine(49, "C[row * N + col] = acc;"),
      ],
    },
  ],
  regtile: [
    {
      location: "gemm.cu:54-70",
      explanation: "RM=RN=2 让一个线程负责 2×2 输出。base_row/base_col 定位该线程负责区域，四个 acc 保存四个结果。",
      lines: [
        sourceLine(54, "constexpr int RT_TILE = 16;"),
        sourceLine(55, "constexpr int RM = 2;", true),
        sourceLine(56, "constexpr int RN = 2;", true),
        sourceLine(67, "int base_row = blockIdx.y * (RT_TILE * RM) + local_row * RM;", true),
        sourceLine(68, "int base_col = blockIdx.x * (RT_TILE * RN) + local_col * RN;", true),
        sourceLine(70, "float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;", true),
      ],
    },
    {
      location: "gemm.cu:87-92",
      explanation: "动画中的 a0/a1/b0/b1 正是代码从 Shared Memory 读取的两个 A 值和两个 B 值。",
      lines: [
        sourceLine(87, "#pragma unroll"),
        sourceLine(88, "for (int kk = 0; kk < RT_TILE; ++kk) {"),
        sourceLine(89, "    float a0 = As[local_row * RM + 0][kk];", true),
        sourceLine(90, "    float a1 = As[local_row * RM + 1][kk];", true),
        sourceLine(91, "    float b0 = Bs[kk][local_col * RN + 0];", true),
        sourceLine(92, "    float b1 = Bs[kk][local_col * RN + 1];", true),
      ],
    },
    {
      location: "gemm.cu:93-96",
      explanation: "动画中的四种组合逐行对应四个 FMA 累加，每个 a 值和 b 值在当前线程内被复用两次。",
      lines: [
        sourceLine(93, "acc00 += a0 * b0;", true),
        sourceLine(94, "acc01 += a0 * b1;", true),
        sourceLine(95, "acc10 += a1 * b0;", true),
        sourceLine(96, "acc11 += a1 * b1;", true),
      ],
    },
    {
      location: "gemm.cu:72, 87-98",
      explanation: "外层 t 遍历 K Tile，内层 kk 遍历 Tile 内部；四个寄存器累加器贯穿整个 K 维度。",
      lines: [
        sourceLine(72, "for (int t = 0; t < K; t += RT_TILE) {", true),
        sourceLine(88, "    for (int kk = 0; kk < RT_TILE; ++kk) {", true),
        sourceLine(89, "        float a0 = As[local_row * RM + 0][kk];"),
        sourceLine(91, "        float b0 = Bs[kk][local_col * RN + 0];"),
        sourceLine(93, "        acc00 += a0 * b0;", true),
        sourceLine(97, "    }"),
        sourceLine(98, "    __syncthreads();"),
      ],
    },
    {
      location: "gemm.cu:101-104",
      explanation: "动画中一个线程写回四个相邻输出，正好对应四条带边界检查的 C 写入语句。",
      lines: [
        sourceLine(101, "if (base_row + 0 < M && base_col + 0 < N) C[(base_row + 0) * N + base_col + 0] = acc00;", true),
        sourceLine(102, "if (base_row + 0 < M && base_col + 1 < N) C[(base_row + 0) * N + base_col + 1] = acc01;", true),
        sourceLine(103, "if (base_row + 1 < M && base_col + 0 < N) C[(base_row + 1) * N + base_col + 0] = acc10;", true),
        sourceLine(104, "if (base_row + 1 < M && base_col + 1 < N) C[(base_row + 1) * N + base_col + 1] = acc11;", true),
      ],
    },
    {
      location: "gemm.cu:54-68, 146-148",
      explanation: "Host 仍启动 16×16 个线程，但 grid 按 RT_TILE×RN 和 RT_TILE×RM 计算，因此一个 Block 覆盖 32×32 输出。",
      lines: [
        sourceLine(54, "constexpr int RT_TILE = 16;"),
        sourceLine(55, "constexpr int RM = 2;"),
        sourceLine(56, "constexpr int RN = 2;"),
        sourceLine(67, "int base_row = blockIdx.y * (RT_TILE * RM) + local_row * RM;"),
        sourceLine(68, "int base_col = blockIdx.x * (RT_TILE * RN) + local_col * RN;"),
        sourceLine(146, "dim3 block(RT_TILE, RT_TILE);", true),
        sourceLine(147, "dim3 grid(ceil_div_int(N, RT_TILE * RN), ceil_div_int(M, RT_TILE * RM));", true),
        sourceLine(148, "gemm_regtile2x2_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);"),
      ],
    },
    {
      location: "gemm.cu:62-104",
      explanation: "完整 Register Tiling 数据流：Block 共享 As/Bs，每线程读取 2×2 操作数组合，四个寄存器累加并写回四个输出。",
      lines: [
        sourceLine(62, "__shared__ float As[RT_TILE * RM][RT_TILE];", true),
        sourceLine(63, "__shared__ float Bs[RT_TILE][RT_TILE * RN];", true),
        sourceLine(70, "float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;", true),
        sourceLine(89, "float a0 = As[local_row * RM + 0][kk];"),
        sourceLine(91, "float b0 = Bs[kk][local_col * RN + 0];"),
        sourceLine(93, "acc00 += a0 * b0;", true),
        sourceLine(101, "if (base_row + 0 < M && base_col + 0 < N) C[(base_row + 0) * N + base_col + 0] = acc00;"),
      ],
    },
  ],
};

const naiveSteps = [
  {
    label: "定位输出",
    title: "线程 T(0,0) 负责 C[0,0]",
    calculation: "C[0,0] = Σ A[0,k] × B[k,0]",
    detail: "blockIdx 和 threadIdx 共同确定 row、col。一个线程只拥有一个输出位置。",
    cells: [{ matrix: "C", row: 0, col: 0, className: "is-c-active" }],
    registers: ["acc = 0"],
  },
  ...[0, 1, 2, 3].map((k) => {
    const terms = Array.from({ length: k + 1 }, (_, index) =>
      `${matrices.A[0][index]}×${matrices.B[index][0]}`
    ).join(" + ");
    const acc = Array.from({ length: k + 1 }, (_, index) =>
      matrices.A[0][index] * matrices.B[index][0]
    ).reduce((sum, value) => sum + value, 0);
    return {
      label: `点积 k = ${k}`,
      title: `直接读取 A[0,${k}] 与 B[${k},0]`,
      calculation: `acc = ${terms} = ${acc}`,
      detail: "数据直接来自 Global Memory。其他输出线程可能再次读取同一个 A 或 B 元素。",
      cells: [
        { matrix: "A", row: 0, col: k, className: "is-a-active" },
        { matrix: "B", row: k, col: 0, className: "is-b-active" },
        { matrix: "C", row: 0, col: 0, className: "is-c-active" },
      ],
      global: [`A[0,${k}]`, `B[${k},0]`],
      registers: [`acc = ${acc}`],
      globalArrow: false,
      sharedArrow: true,
    };
  }),
  {
    label: "写回结果",
    title: "线程把累加结果写入 C[0,0]",
    calculation: `C[0,0] = ${matrices.C[0][0]}`,
    detail: "每个线程独立重复这一过程，最终填满整个 C 矩阵。",
    cells: [{ matrix: "C", row: 0, col: 0, className: "is-c-active is-written" }],
    registers: [`acc = ${matrices.C[0][0]}`],
    writeCells: [[0, 0]],
  },
  {
    label: "Naive 总结",
    title: "并行完成了，但数据没有被主动复用",
    calculation: "一个线程 → 一个输出；每次乘加 → Global Memory",
    detail: "下一步要解决的问题：同一个 A/B 元素被相邻线程重复读取。",
    cells: [
      ...rangeCells("A", [[0, 0], [0, 1], [0, 2], [0, 3]], "is-a-active"),
      ...rangeCells("B", [[0, 0], [1, 0], [2, 0], [3, 0]], "is-b-active"),
      { matrix: "C", row: 0, col: 0, className: "is-c-active" },
    ],
    registers: ["1 thread", "1 acc"],
  },
];

const tiledSteps = [
  {
    label: "矩阵分块",
    title: "先选择 C 的一个 2×2 教学 Tile",
    calculation: "Ctile = Atile₀ × Btile₀ + Atile₁ × Btile₁",
    detail: "真实 kernel 使用 16×16 Tile；动画缩小为 2×2，便于观察。",
    cells: rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    shared: ["等待加载"],
    registers: ["acc = 0"],
  },
  {
    label: "协作加载",
    title: "Block 中的线程把第一个 A/B Tile 搬入 Shared Memory",
    calculation: "Global A/B → Shared As/Bs",
    detail: "每个线程只搬一小部分。协作完成后，整个 Block 都能访问 As 和 Bs。",
    cells: [
      ...rangeCells("A", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-a-active is-tile"),
      ...rangeCells("B", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-b-active is-tile"),
    ],
    global: ["A tile₀", "B tile₀"],
    shared: ["As tile₀", "Bs tile₀"],
    registers: ["acc = 0"],
    globalArrow: true,
  },
  {
    label: "同步",
    title: "__syncthreads() 等待 Tile 完整",
    calculation: "所有线程加载完成 → 所有线程开始计算",
    detail: "同步避免某个线程读取到其他线程尚未写入的 Shared Memory 数据。",
    cells: [
      ...rangeCells("A", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-a-active is-tile"),
      ...rangeCells("B", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-b-active is-tile"),
    ],
    shared: ["As ✓", "Bs ✓"],
    registers: ["等待计算"],
  },
  {
    label: "复用计算",
    title: "同一份 Shared Tile 服务 4 个输出线程",
    calculation: "C[0:2,0:2] += As[:,0:2] × Bs[0:2,:]",
    detail: "A 的一个值可服务多个输出列，B 的一个值可服务多个输出行。",
    cells: [
      ...rangeCells("A", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-a-active is-tile"),
      ...rangeCells("B", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-b-active is-tile"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    shared: ["As reused", "Bs reused"],
    registers: ["acc00", "acc01", "acc10", "acc11"],
    sharedArrow: true,
  },
  {
    label: "下一个 K Tile",
    title: "覆盖 Shared Memory，继续累加剩余 K 范围",
    calculation: "t = 2：加载 k=2..3，再累加到原有 acc",
    detail: "分块只改变计算顺序，不改变完整点积的数学结果。",
    cells: [
      ...rangeCells("A", [[0, 2], [0, 3], [1, 2], [1, 3]], "is-a-active is-tile"),
      ...rangeCells("B", [[2, 0], [2, 1], [3, 0], [3, 1]], "is-b-active is-tile"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    global: ["A tile₁", "B tile₁"],
    shared: ["As tile₁", "Bs tile₁"],
    registers: ["继续累加"],
    globalArrow: true,
  },
  {
    label: "写回结果",
    title: "每个线程写回自己负责的一个输出",
    calculation: "4 threads → C[0,0], C[0,1], C[1,0], C[1,1]",
    detail: "与 Naive 相比，线程分工没变，变化在于数据先进入 Shared Memory。",
    cells: rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-written"),
    writeCells: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared: ["复用完成"],
    registers: ["4 threads", "4 acc"],
  },
  {
    label: "Tiled 总结",
    title: "减少 Global Memory 重复读取",
    calculation: "Global 读取一次 → Shared Memory 复用多次",
    detail: "下一步要解决的问题：每个线程仍然只计算一个输出，线程内复用还不充分。",
    cells: [
      ...rangeCells("A", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-a-active is-tile"),
      ...rangeCells("B", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-b-active is-tile"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    shared: ["Block 共享"],
    registers: ["1 output / thread"],
  },
];

const regtileSteps = [
  {
    label: "扩大线程职责",
    title: "一个线程负责 C 中相邻的 2×2 区域",
    calculation: "1 thread → acc00, acc01, acc10, acc11",
    detail: "动画中的紫色 2×2 区域由一个线程完成，而不是四个线程。",
    cells: rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    shared: ["As", "Bs"],
    registers: ["acc00=0", "acc01=0", "acc10=0", "acc11=0"],
  },
  {
    label: "读取操作数",
    title: "从 Shared Memory 读取 2 个 A 和 2 个 B",
    calculation: "a0, a1 ← As；b0, b1 ← Bs",
    detail: "这些值进入当前线程，准备在寄存器中组合复用。",
    cells: [
      ...rangeCells("A", [[0, 0], [1, 0]], "is-a-active"),
      ...rangeCells("B", [[0, 0], [0, 1]], "is-b-active"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    shared: ["a0", "a1", "b0", "b1"],
    registers: ["4 accumulators"],
    sharedArrow: true,
  },
  {
    label: "四种组合",
    title: "2 个 A × 2 个 B 形成 4 次乘加",
    calculation: "a0b0 → acc00；a0b1 → acc01；a1b0 → acc10；a1b1 → acc11",
    detail: "一次 Shared Memory 读取被当前线程用于四个输出，形成线程内数据复用。",
    cells: [
      ...rangeCells("A", [[0, 0], [1, 0]], "is-a-active"),
      ...rangeCells("B", [[0, 0], [0, 1]], "is-b-active"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    shared: ["a0", "a1", "b0", "b1"],
    registers: ["acc00 += a0b0", "acc01 += a0b1", "acc10 += a1b0", "acc11 += a1b1"],
    sharedArrow: true,
  },
  {
    label: "沿 K 累加",
    title: "对 Tile 中每个 kk 重复四路乘加",
    calculation: "for kk in 0..15：更新 4 个 acc",
    detail: "真实 kernel 的 RT_TILE=16；四个累加器始终保留在同一线程中。",
    cells: [
      ...rangeCells("A", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-a-active is-tile"),
      ...rangeCells("B", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-b-active is-tile"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    shared: ["As tile", "Bs tile"],
    registers: ["acc00", "acc01", "acc10", "acc11"],
    sharedArrow: true,
  },
  {
    label: "写回四个结果",
    title: "同一个线程连续写回相邻 2×2 输出",
    calculation: `C tile = [[${matrices.C[0][0]}, ${matrices.C[0][1]}], [${matrices.C[1][0]}, ${matrices.C[1][1]}]]`,
    detail: "边界判断确保 M/N 不是 Tile 整倍数时不会越界写入。",
    cells: rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-written"),
    writeCells: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared: ["完成"],
    registers: [
      `acc00=${matrices.C[0][0]}`,
      `acc01=${matrices.C[0][1]}`,
      `acc10=${matrices.C[1][0]}`,
      `acc11=${matrices.C[1][1]}`,
    ],
  },
  {
    label: "Block 覆盖扩大",
    title: "相同 16×16 线程 Block 计算 32×32 输出",
    calculation: "256 threads × 4 outputs = 1024 outputs",
    detail: "更多计算分摊了循环、同步和地址计算开销，但也会消耗更多寄存器。",
    cells: rangeCells("C", [
      [0, 0], [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 1], [1, 2], [1, 3],
      [2, 0], [2, 1], [2, 2], [2, 3],
      [3, 0], [3, 1], [3, 2], [3, 3],
    ], "is-c-active is-tile"),
    shared: ["Block 共享"],
    registers: ["4 outputs / thread"],
  },
  {
    label: "Regtile 总结",
    title: "让一次读取完成更多计算",
    calculation: "Block 内复用 + 线程内复用",
    detail: "优化层次完整串联：并行分工 → Shared Memory → Register Tiling。",
    cells: [
      ...rangeCells("A", [[0, 0], [1, 0]], "is-a-active"),
      ...rangeCells("B", [[0, 0], [0, 1]], "is-b-active"),
      ...rangeCells("C", [[0, 0], [0, 1], [1, 0], [1, 1]], "is-c-active is-tile"),
    ],
    shared: ["As", "Bs"],
    registers: ["acc00", "acc01", "acc10", "acc11"],
  },
];

const stepsByMode = {
  naive: naiveSteps,
  tiled: tiledSteps,
  regtile: regtileSteps,
};

const state = {
  mode: "naive",
  step: 0,
  playing: false,
  intervalId: null,
  intervalMs: 1050,
};

const elements = {
  modeTitle: document.querySelector("#mode-title"),
  modeSummary: document.querySelector("#mode-summary"),
  phaseLabel: document.querySelector("#phase-label"),
  phaseTitle: document.querySelector("#phase-title"),
  phaseDetail: document.querySelector("#phase-detail"),
  calculation: document.querySelector("#calculation"),
  stepCount: document.querySelector("#step-count"),
  progressBar: document.querySelector("#progress-bar"),
  sharedContent: document.querySelector("#shared-content"),
  globalContent: document.querySelector("#global-content"),
  registerContent: document.querySelector("#register-content"),
  globalArrow: document.querySelector("#global-arrow"),
  sharedArrow: document.querySelector("#shared-arrow"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  speedInput: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
  statOutput: document.querySelector("#stat-output"),
  statLoads: document.querySelector("#stat-loads"),
  statReuse: document.querySelector("#stat-reuse"),
  statBlock: document.querySelector("#stat-block"),
  sourceLocation: document.querySelector("#source-location"),
  sourceCode: document.querySelector("#source-code"),
  sourceExplanation: document.querySelector("#source-explanation"),
};

function createMatrix(containerId, matrixName) {
  const container = document.querySelector(containerId);
  matrices[matrixName].forEach((rowValues, row) => {
    rowValues.forEach((value, col) => {
      const cell = document.createElement("span");
      cell.className = "matrix-cell";
      cell.dataset.matrix = matrixName;
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.dataset.value = String(value);
      cell.textContent = matrixName === "C" ? "·" : String(value);
      container.appendChild(cell);
    });
  });
}

function renderMemory(container, items, type) {
  container.replaceChildren();
  const resolvedItems = items && items.length ? items : [type === "shared" ? "未使用" : "—"];
  resolvedItems.forEach((item) => {
    const chip = document.createElement("span");
    if (item === "未使用" || item === "—") {
      chip.className = "empty-state";
    } else if (type === "register") {
      chip.className = "register-chip";
    } else if (String(item).startsWith("A") || String(item).startsWith("a")) {
      chip.className = "memory-a";
    } else if (String(item).startsWith("B") || String(item).startsWith("b")) {
      chip.className = "memory-b";
    }
    chip.textContent = item;
    container.appendChild(chip);
  });
}

function resetCells() {
  document.querySelectorAll(".matrix-cell").forEach((cell) => {
    cell.className = "matrix-cell";
    if (cell.dataset.matrix === "C") {
      cell.textContent = "·";
    }
  });
}

function revealWrittenCells(coordinates = []) {
  coordinates.forEach(([row, col]) => {
    const cell = document.querySelector(
      `.matrix-cell[data-matrix="C"][data-row="${row}"][data-col="${col}"]`
    );
    if (cell) {
      cell.textContent = String(matrices.C[row][col]);
      cell.classList.add("is-written");
    }
  });
}

function applyCellHighlights(cells = []) {
  cells.forEach(({ matrix, row, col, className }) => {
    const cell = document.querySelector(
      `.matrix-cell[data-matrix="${matrix}"][data-row="${row}"][data-col="${col}"]`
    );
    if (cell) {
      className.split(" ").forEach((name) => cell.classList.add(name));
    }
  });
}

function updateConcepts(activeIndex) {
  document.querySelectorAll(".concept-panel li").forEach((item, index) => {
    item.classList.toggle("is-current", index === activeIndex);
    item.classList.toggle("is-complete", index < activeIndex);
  });
}

function renderSource(source) {
  elements.sourceLocation.textContent = source.location;
  elements.sourceExplanation.textContent = source.explanation;
  elements.sourceCode.replaceChildren();
  source.lines.forEach((line) => {
    const row = document.createElement("span");
    row.className = "source-line";
    if (line.active) row.classList.add("is-source-active");

    const number = document.createElement("i");
    number.textContent = String(line.number);
    const code = document.createElement("code");
    code.textContent = line.text;

    row.append(number, code);
    elements.sourceCode.appendChild(row);
  });
}

function render() {
  const config = modeConfig[state.mode];
  const steps = stepsByMode[state.mode];
  const step = steps[state.step];
  const source = sourceByMode[state.mode][state.step];

  elements.modeTitle.textContent = config.title;
  elements.modeSummary.textContent = config.summary;
  elements.phaseLabel.textContent = step.label;
  elements.phaseTitle.textContent = step.title;
  elements.phaseDetail.textContent = step.detail;
  elements.calculation.textContent = step.calculation;
  elements.stepCount.textContent = `${String(state.step + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  elements.progressBar.style.width = `${((state.step + 1) / steps.length) * 100}%`;

  [elements.statOutput, elements.statLoads, elements.statReuse, elements.statBlock]
    .forEach((element, index) => {
      element.textContent = config.stats[index];
    });

  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.mode === state.mode);
  });

  resetCells();
  applyCellHighlights(step.cells);
  revealWrittenCells(step.writeCells);

  renderMemory(elements.globalContent, step.global || ["A", "B"], "global");
  renderMemory(elements.sharedContent, step.shared, "shared");
  renderMemory(elements.registerContent, step.registers || ["acc = 0"], "register");

  elements.globalArrow.classList.toggle("is-active", Boolean(step.globalArrow));
  elements.sharedArrow.classList.toggle("is-active", Boolean(step.sharedArrow));
  updateConcepts(config.concept);
  renderSource(source);
}

function stopPlayback() {
  state.playing = false;
  if (state.intervalId) {
    window.clearInterval(state.intervalId);
    state.intervalId = null;
  }
  elements.playIcon.textContent = "▶";
  elements.playButton.title = "播放";
  elements.playButton.setAttribute("aria-label", "播放");
}

function advanceStep(direction = 1) {
  const steps = stepsByMode[state.mode];
  const next = state.step + direction;

  if (next >= steps.length) {
    if (state.playing) {
      state.step = 0;
    } else {
      state.step = steps.length - 1;
    }
  } else if (next < 0) {
    state.step = 0;
  } else {
    state.step = next;
  }
  render();
}

function startPlayback() {
  stopPlayback();
  state.playing = true;
  elements.playIcon.textContent = "Ⅱ";
  elements.playButton.title = "暂停";
  elements.playButton.setAttribute("aria-label", "暂停");
  state.intervalId = window.setInterval(() => advanceStep(1), state.intervalMs);
}

function setMode(mode) {
  if (!stepsByMode[mode]) return;
  stopPlayback();
  state.mode = mode;
  state.step = 0;
  render();
}

document.querySelectorAll(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

document.querySelectorAll("[data-jump-mode]").forEach((card) => {
  card.addEventListener("click", () => {
    setMode(card.dataset.jumpMode);
    document.querySelector(".mode-tabs").scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

elements.playButton.addEventListener("click", () => {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

document.querySelector("#next-button").addEventListener("click", () => {
  stopPlayback();
  advanceStep(1);
});

document.querySelector("#prev-button").addEventListener("click", () => {
  stopPlayback();
  advanceStep(-1);
});

document.querySelector("#reset-button").addEventListener("click", () => {
  stopPlayback();
  state.step = 0;
  render();
});

elements.speedInput.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  state.intervalMs = 2150 - value;
  const speed = (1050 / state.intervalMs).toFixed(1);
  elements.speedOutput.textContent = `${speed}×`;
  if (state.playing) {
    startPlayback();
  }
});

createMatrix("#matrix-a", "A");
createMatrix("#matrix-b", "B");
createMatrix("#matrix-c", "C");
render();
