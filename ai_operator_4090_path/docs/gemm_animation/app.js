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

matrices.C = matrices.A.map((row) =>
  matrices.B[0].map((_, j) =>
    row.reduce((sum, value, k) => sum + value * matrices.B[k][j], 0)
  )
);

function rangeCells(matrix, coordinates, className) {
  return coordinates.map(([row, col]) => ({ matrix, row, col, className }));
}

function blockCells(rowStart, rowEnd, colStart, colEnd) {
  const cells = [];
  for (let row = rowStart; row <= rowEnd; row += 1) {
    for (let col = colStart; col <= colEnd; col += 1) {
      cells.push([row, col]);
    }
  }
  return cells;
}

function sourceLine(number, text, active = false) {
  return { number, text, active };
}

function src(location, explanation, lines) {
  return { location, explanation, lines };
}

const commonCells = {
  c00: [{ matrix: "C", row: 0, col: 0, className: "is-c-active" }],
  c2x2: rangeCells("C", blockCells(0, 1, 0, 1), "is-c-active is-tile"),
  c4x4: rangeCells("C", blockCells(0, 3, 0, 3), "is-c-active is-tile"),
  aTile0: rangeCells("A", blockCells(0, 1, 0, 1), "is-a-active is-tile"),
  bTile0: rangeCells("B", blockCells(0, 1, 0, 1), "is-b-active is-tile"),
  aTile1: rangeCells("A", blockCells(0, 1, 2, 3), "is-a-active is-tile"),
  bTile1: rangeCells("B", blockCells(2, 3, 0, 1), "is-b-active is-tile"),
};

const sourceCatalog = {
  naiveMap: src("gemm.cu:16-24", "每个线程用 blockIdx/threadIdx 定位一个 C 元素，然后用一个 acc 做完整 K 维点积。", [
    sourceLine(16, "int row = blockIdx.y * blockDim.y + threadIdx.y;", true),
    sourceLine(17, "int col = blockIdx.x * blockDim.x + threadIdx.x;", true),
    sourceLine(18, "if (row >= M || col >= N) return;"),
    sourceLine(20, "float acc = 0.0f;", true),
    sourceLine(21, "for (int k = 0; k < K; ++k) {"),
    sourceLine(24, "C[row * N + col] = acc;"),
  ]),
  naiveLoop: src("gemm.cu:21-23", "动画把 for k 循环展开。代码每轮直接从 Global Memory 读取 A[row,k] 和 B[k,col]。", [
    sourceLine(21, "for (int k = 0; k < K; ++k) {", true),
    sourceLine(22, "    acc += A[row * K + k] * B[k * N + col];", true),
    sourceLine(23, "}"),
  ]),
  naiveWrite: src("gemm.cu:24, 301-310", "kernel 写回一个 C 元素；host launcher 用 grid 覆盖整个 M×N 输出矩阵。", [
    sourceLine(24, "C[row * N + col] = acc;", true),
    sourceLine(306, "auto C = torch::empty({M, N}, A.options());"),
    sourceLine(307, "dim3 block(16, 16);", true),
    sourceLine(308, "dim3 grid(ceil_div_int(N, block.x), ceil_div_int(M, block.y));", true),
    sourceLine(309, "gemm_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(...);"),
  ]),
  tiledShared: src("gemm.cu:27-54, 314-323", "TILE=16 决定一个 block 覆盖 16×16 个 C 元素；A/B 子块先进入 shared memory。", [
    sourceLine(27, "constexpr int TILE = 16;", true),
    sourceLine(33, "__shared__ float As[TILE][TILE];", true),
    sourceLine(34, "__shared__ float Bs[TILE][TILE];", true),
    sourceLine(36, "int row = blockIdx.y * TILE + threadIdx.y;"),
    sourceLine(37, "int col = blockIdx.x * TILE + threadIdx.x;"),
    sourceLine(320, "dim3 block(TILE, TILE);", true),
    sourceLine(321, "dim3 grid(ceil_div_int(N, TILE), ceil_div_int(M, TILE));"),
  ]),
  tiledLoad: src("gemm.cu:40-45", "每轮 K tile 中，每个线程加载一个 A 元素和一个 B 元素；越界位置补 0。", [
    sourceLine(40, "for (int t = 0; t < K; t += TILE) {", true),
    sourceLine(41, "    int a_col = t + threadIdx.x;"),
    sourceLine(42, "    int b_row = t + threadIdx.y;"),
    sourceLine(43, "    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;", true),
    sourceLine(44, "    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;", true),
    sourceLine(45, "    __syncthreads();", true),
  ]),
  tiledCompute: src("gemm.cu:47-54", "所有线程同步后，从 As/Bs 读取数据完成当前 tile 的累加，最后写回自己的一个 C。", [
    sourceLine(47, "#pragma unroll"),
    sourceLine(48, "for (int kk = 0; kk < TILE; ++kk) {", true),
    sourceLine(49, "    acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];", true),
    sourceLine(51, "__syncthreads();"),
    sourceLine(54, "if (row < M && col < N) C[row * N + col] = acc;", true),
  ]),
  paddedDeclare: src("gemm.cu:57-82", "padding 版本与 tiled 数学一致，只把 shared memory 第二维从 TILE 改为 TILE+1。", [
    sourceLine(57, "__global__ void gemm_tiled_padding_kernel(...)", true),
    sourceLine(61, "__shared__ float As[TILE][TILE + 1];", true),
    sourceLine(62, "__shared__ float Bs[TILE][TILE + 1];", true),
    sourceLine(64, "int row = blockIdx.y * TILE + threadIdx.y;"),
    sourceLine(65, "int col = blockIdx.x * TILE + threadIdx.x;"),
  ]),
  paddedFlow: src("gemm.cu:68-82, 327-336", "搬运、同步、计算、写回都沿用 tiled 思路；+1 padding 主要用于改变 shared memory 行跨度。", [
    sourceLine(68, "for (int t = 0; t < K; t += TILE) {", true),
    sourceLine(71, "As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;", true),
    sourceLine(72, "Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;", true),
    sourceLine(73, "__syncthreads();"),
    sourceLine(77, "acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];", true),
    sourceLine(82, "if (row < M && col < N) C[row * N + col] = acc;"),
    sourceLine(335, "gemm_tiled_padding_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(...);"),
  ]),
  reg2Map: src("gemm.cu:87-138, 340-349", "RM=RN=2 让一个线程负责 2×2 输出，四个 acc 是线程私有寄存器累加器。", [
    sourceLine(87, "constexpr int RT_TILE = 16;"),
    sourceLine(88, "constexpr int RM = 2;", true),
    sourceLine(89, "constexpr int RN = 2;", true),
    sourceLine(100, "int base_row = blockIdx.y * (RT_TILE * RM) + local_row * RM;", true),
    sourceLine(101, "int base_col = blockIdx.x * (RT_TILE * RN) + local_col * RN;", true),
    sourceLine(103, "float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;", true),
  ]),
  reg2Load: src("gemm.cu:105-118", "每个线程加载 2 行 A 和 2 列 B 到 shared memory；整个 block 合起来覆盖更大的 A/B tile。", [
    sourceLine(105, "for (int t = 0; t < K; t += RT_TILE) {", true),
    sourceLine(107, "for (int r = 0; r < RM; ++r) {", true),
    sourceLine(110, "As[local_row * RM + r][local_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;"),
    sourceLine(113, "for (int c = 0; c < RN; ++c) {", true),
    sourceLine(116, "Bs[local_row][local_col * RN + c] = (row < K && col < N) ? B[row * N + col] : 0.0f;"),
    sourceLine(118, "__syncthreads();", true),
  ]),
  reg2Compute: src("gemm.cu:120-138", "从 shared memory 取 a0/a1/b0/b1，一次组合更新 4 个输出，最后分别写回。", [
    sourceLine(121, "for (int kk = 0; kk < RT_TILE; ++kk) {", true),
    sourceLine(122, "    float a0 = As[local_row * RM + 0][kk];", true),
    sourceLine(123, "    float a1 = As[local_row * RM + 1][kk];", true),
    sourceLine(124, "    float b0 = Bs[kk][local_col * RN + 0];", true),
    sourceLine(125, "    float b1 = Bs[kk][local_col * RN + 1];", true),
    sourceLine(126, "    acc00 += a0 * b0;", true),
    sourceLine(127, "    acc01 += a0 * b1;", true),
    sourceLine(128, "    acc10 += a1 * b0;", true),
    sourceLine(129, "    acc11 += a1 * b1;", true),
    sourceLine(134, "if (base_row + 0 < M && base_col + 0 < N) C[...] = acc00;"),
  ]),
  reg4Map: src("gemm.cu:140-156, 353-362", "4×4 register tiling 把每线程输出从 4 个扩展到 16 个，block 覆盖 64×64 输出。", [
    sourceLine(140, "constexpr int RT4_TILE = 16;"),
    sourceLine(141, "constexpr int RT4_RM = 4;", true),
    sourceLine(142, "constexpr int RT4_RN = 4;", true),
    sourceLine(153, "int base_row = blockIdx.y * (RT4_TILE * RT4_RM) + local_row * RT4_RM;", true),
    sourceLine(154, "int base_col = blockIdx.x * (RT4_TILE * RT4_RN) + local_col * RT4_RN;", true),
    sourceLine(156, "float acc[RT4_RM][RT4_RN] = {};", true),
    sourceLine(360, "dim3 grid(ceil_div_int(N, RT4_TILE * RT4_RN), ceil_div_int(M, RT4_TILE * RT4_RM));"),
  ]),
  reg4LoadCompute: src("gemm.cu:158-197", "每个线程加载 4 个 A 和 4 个 B，读入 a_vals/b_vals 后用双重循环更新 16 个 acc。", [
    sourceLine(158, "for (int t = 0; t < K; t += RT4_TILE) {", true),
    sourceLine(159, "for (int r = 0; r < RT4_RM; ++r) {", true),
    sourceLine(163, "As[local_row * RT4_RM + r][local_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;"),
    sourceLine(167, "for (int c = 0; c < RT4_RN; ++c) {", true),
    sourceLine(170, "Bs[local_row][local_col * RT4_RN + c] = (row < K && col < N) ? B[row * N + col] : 0.0f;"),
    sourceLine(181, "a_vals[r] = As[local_row * RT4_RM + r][kk];", true),
    sourceLine(186, "b_vals[c] = Bs[kk][local_col * RT4_RN + c];", true),
    sourceLine(193, "acc[r][c] += a_vals[r] * b_vals[c];", true),
  ]),
  reg4Write: src("gemm.cu:200-208", "写回阶段遍历 4×4 acc，每个元素都带边界判断，支持非整倍数 M/N。", [
    sourceLine(200, "#pragma unroll"),
    sourceLine(201, "for (int r = 0; r < RT4_RM; ++r) {", true),
    sourceLine(203, "    for (int c = 0; c < RT4_RN; ++c) {", true),
    sourceLine(204, "        int row = base_row + r;"),
    sourceLine(205, "        int col = base_col + c;"),
    sourceLine(206, "        if (row < M && col < N) C[row * N + col] = acc[r][c];", true),
  ]),
  float4Map: src("gemm.cu:211-244, 366-375", "float4 版本让一个线程负责同一行相邻 4 列，用一次向量读取拿到 B 的 4 个值。", [
    sourceLine(215, "int row = blockIdx.y * blockDim.y + threadIdx.y;", true),
    sourceLine(216, "int base_col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;", true),
    sourceLine(219, "float acc0 = 0.0f;"),
    sourceLine(220, "float acc1 = 0.0f;"),
    sourceLine(221, "float acc2 = 0.0f;"),
    sourceLine(222, "float acc3 = 0.0f;"),
    sourceLine(373, "dim3 grid(ceil_div_int(N, block.x * 4), ceil_div_int(M, block.y));"),
  ]),
  float4Compute: src("gemm.cu:223-244", "当 N 是 4 的倍数且列未越界时，B[k,base_col:base_col+4] 走 float4 向量加载。", [
    sourceLine(223, "bool can_vectorize_b = (N % 4 == 0) && (base_col + 3 < N);", true),
    sourceLine(225, "for (int k = 0; k < K; ++k) {", true),
    sourceLine(226, "    float a = A[row * K + k];", true),
    sourceLine(228, "    float4 b = reinterpret_cast<const float4*>(B + k * N + base_col)[0];", true),
    sourceLine(229, "    acc0 += a * b.x;"),
    sourceLine(230, "    acc1 += a * b.y;"),
    sourceLine(231, "    acc2 += a * b.z;"),
    sourceLine(232, "    acc3 += a * b.w;"),
    sourceLine(241, "if (base_col + 0 < N) C[row * N + base_col + 0] = acc0;"),
  ]),
  float4Fallback: src("gemm.cu:233-244", "非 4 对齐或尾部列不足 4 个时，退回逐元素读取和逐元素写回，保证正确性。", [
    sourceLine(233, "} else {", true),
    sourceLine(234, "    if (base_col + 0 < N) acc0 += a * B[k * N + base_col + 0];", true),
    sourceLine(235, "    if (base_col + 1 < N) acc1 += a * B[k * N + base_col + 1];"),
    sourceLine(236, "    if (base_col + 2 < N) acc2 += a * B[k * N + base_col + 2];"),
    sourceLine(237, "    if (base_col + 3 < N) acc3 += a * B[k * N + base_col + 3];"),
    sourceLine(241, "if (base_col + 0 < N) C[row * N + base_col + 0] = acc0;", true),
    sourceLine(244, "if (base_col + 3 < N) C[row * N + base_col + 3] = acc3;", true),
  ]),
  wmmaMap: src("gemm.cu:247-274, 379-405", "WMMA 版本把计算粒度提升到 warp：一个 warp 计算 16×16 C tile，使用 Tensor Core 指令。", [
    sourceLine(247, "constexpr int WMMA_M = 16;", true),
    sourceLine(248, "constexpr int WMMA_N = 16;", true),
    sourceLine(249, "constexpr int WMMA_K = 16;", true),
    sourceLine(256, "int warp_id = threadIdx.y;", true),
    sourceLine(257, "int tile_m = blockIdx.y * WMMA_WARPS_PER_BLOCK + warp_id;", true),
    sourceLine(258, "int tile_n = blockIdx.x;"),
    sourceLine(398, "dim3 block(32, WMMA_WARPS_PER_BLOCK);"),
  ]),
  wmmaFragments: src("gemm.cu:263-274", "fragment 是 WMMA API 的寄存器级 tile 容器；mma_sync 在 Tensor Core 上累加。", [
    sourceLine(263, "wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;", true),
    sourceLine(264, "wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;", true),
    sourceLine(265, "wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;", true),
    sourceLine(266, "wmma::fill_fragment(acc_frag, 0.0f);"),
    sourceLine(269, "wmma::load_matrix_sync(a_frag, A + row * K + k0, K);", true),
    sourceLine(270, "wmma::load_matrix_sync(b_frag, B + k0 * N + col, N);", true),
    sourceLine(271, "wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);", true),
    sourceLine(274, "wmma::store_matrix_sync(C + row * N + col, acc_frag, N, wmma::mem_row_major);"),
  ]),
  wmmaHost: src("gemm.cu:379-405", "host 侧允许 float32/float16 输入；非 16 整倍数 shape 会退回 FP32 tiled 路径保证正确性。", [
    sourceLine(379, "torch::Tensor gemm_wmma_fp16(torch::Tensor A, torch::Tensor B) {", true),
    sourceLine(386, "if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {", true),
    sourceLine(387, "    torch::Tensor A_float = A.scalar_type() == torch::kFloat32 ? A : A.to(torch::kFloat32);"),
    sourceLine(391, "    gemm_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(...);"),
    sourceLine(396, "torch::Tensor A_half = A.scalar_type() == torch::kFloat16 ? A : A.to(torch::kFloat16);", true),
    sourceLine(400, "gemm_wmma_fp16_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(...);", true),
  ]),
};

const modeConfig = {
  naive: {
    title: "一个线程计算一个输出",
    summary: "最直接的 FP32 GEMM：每个线程从 Global Memory 直接读取一行 A 和一列 B。",
    stats: ["1 个 C 元素", "每次乘加都读 Global", "几乎没有", "16 × 16 输出"],
    concept: 0,
  },
  tiled: {
    title: "Shared Memory Tiling",
    summary: "线程协作把 A/B tile 搬进 shared memory，再在 block 内复用。",
    stats: ["1 个 C 元素", "每个 tile 协作读取", "Block 内复用", "16 × 16 输出"],
    concept: 1,
  },
  padded: {
    title: "Tiled + Shared Padding",
    summary: "保持 tiled 数学不变，把 shared memory 行跨度改成 TILE+1，给 bank conflict 优化留入口。",
    stats: ["1 个 C 元素", "同 Tiled", "Block 内复用", "16 × 16 输出"],
    concept: 2,
  },
  regtile2: {
    title: "Register Tiling 2×2",
    summary: "一个线程计算 2×2 共 4 个输出，把 a0/a1/b0/b1 在线程内组合复用。",
    stats: ["4 个 C 元素", "按 tile 协作读取", "Block 内 + 线程内", "32 × 32 输出"],
    concept: 3,
  },
  regtile4: {
    title: "Register Tiling 4×4",
    summary: "一个线程计算 4×4 共 16 个输出，提升计算密度，同时显著增加寄存器压力。",
    stats: ["16 个 C 元素", "每线程搬 4A+4B", "更强线程内复用", "64 × 64 输出"],
    concept: 4,
  },
  float4: {
    title: "float4 向量化读取",
    summary: "一个线程负责同一行相邻 4 列，用 float4 一次读取 B 的 4 个连续值。",
    stats: ["4 个 C 元素", "B 侧 float4 读取", "向量化连续列", "16 × 64 输出"],
    concept: 5,
  },
  wmma: {
    title: "WMMA FP16 / Tensor Core",
    summary: "一个 warp 计算 16×16 tile，使用 WMMA fragment 和 Tensor Core 完成 FP16×FP16→FP32 累加。",
    stats: ["1 warp → 16×16", "FP16 tile load", "Tensor Core", "4 warps / block"],
    concept: 6,
  },
};

const naiveSteps = [
  {
    label: "定位输出",
    title: "线程 T(0,0) 负责 C[0,0]",
    calculation: "C[0,0] = Σ A[0,k] × B[k,0]",
    detail: "blockIdx 和 threadIdx 共同确定 row、col。一个线程只拥有一个输出位置。",
    cells: commonCells.c00,
    registers: ["acc = 0"],
    source: sourceCatalog.naiveMap,
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
      sharedArrow: true,
      source: sourceCatalog.naiveLoop,
    };
  }),
  {
    label: "写回结果",
    title: "线程把累加结果写入 C[0,0]",
    calculation: `C[0,0] = ${matrices.C[0][0]}`,
    detail: "每个线程独立重复这一过程，grid 中所有线程合起来填满整个 C 矩阵。",
    cells: [{ matrix: "C", row: 0, col: 0, className: "is-c-active is-written" }],
    registers: [`acc = ${matrices.C[0][0]}`],
    writeCells: [[0, 0]],
    source: sourceCatalog.naiveWrite,
  },
  {
    label: "Naive 总结",
    title: "并行完成了，但数据没有被主动复用",
    calculation: "一个线程 → 一个输出；每次乘加 → Global Memory",
    detail: "下一步要解决的问题：同一个 A/B 元素被相邻线程重复读取。",
    cells: [
      ...rangeCells("A", blockCells(0, 0, 0, 3), "is-a-active"),
      ...rangeCells("B", blockCells(0, 3, 0, 0), "is-b-active"),
      { matrix: "C", row: 0, col: 0, className: "is-c-active" },
    ],
    registers: ["1 thread", "1 acc"],
    source: sourceCatalog.naiveWrite,
  },
];

const tiledSteps = [
  {
    label: "矩阵分块",
    title: "一个 block 负责一个 16×16 C tile",
    calculation: "Ctile += As × Bs",
    detail: "动画缩小成 2×2 观察；真实代码中 TILE=16，一个 block 有 16×16 个线程。",
    cells: commonCells.c2x2,
    shared: ["等待加载"],
    registers: ["acc = 0"],
    source: sourceCatalog.tiledShared,
  },
  {
    label: "协作加载",
    title: "线程把 A/B 子块搬入 Shared Memory",
    calculation: "Global A/B → Shared As/Bs",
    detail: "每个线程加载一小块。之后同一份 As/Bs 会被 block 内多个线程读取。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0],
    global: ["A tile₀", "B tile₀"],
    shared: ["As tile₀", "Bs tile₀"],
    registers: ["acc = 0"],
    globalArrow: true,
    source: sourceCatalog.tiledLoad,
  },
  {
    label: "同步",
    title: "__syncthreads() 等待 Tile 完整",
    calculation: "所有线程加载完成 → 所有线程开始计算",
    detail: "同步避免某个线程读取到其他线程尚未写入的 shared memory 数据。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0],
    shared: ["As ✓", "Bs ✓"],
    registers: ["等待计算"],
    source: sourceCatalog.tiledLoad,
  },
  {
    label: "复用计算",
    title: "同一份 Shared Tile 服务多个输出线程",
    calculation: "acc += As[threadIdx.y][kk] × Bs[kk][threadIdx.x]",
    detail: "A 的一个值可服务多个输出列，B 的一个值可服务多个输出行。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0, ...commonCells.c2x2],
    shared: ["As reused", "Bs reused"],
    registers: ["acc00", "acc01", "acc10", "acc11"],
    sharedArrow: true,
    source: sourceCatalog.tiledCompute,
  },
  {
    label: "下一个 K Tile",
    title: "覆盖 Shared Memory，继续累加剩余 K 范围",
    calculation: "t += TILE：加载下一块 K，再累加到原有 acc",
    detail: "分块只改变数据流动和复用方式，不改变完整点积的数学结果。",
    cells: [...commonCells.aTile1, ...commonCells.bTile1, ...commonCells.c2x2],
    global: ["A tile₁", "B tile₁"],
    shared: ["As tile₁", "Bs tile₁"],
    registers: ["继续累加"],
    globalArrow: true,
    source: sourceCatalog.tiledCompute,
  },
  {
    label: "写回结果",
    title: "每个线程写回自己负责的一个输出",
    calculation: "4 threads → C[0,0], C[0,1], C[1,0], C[1,1]",
    detail: "与 Naive 相比，线程分工没变；变化在于 Global → Shared → Register 的读取路径。",
    cells: rangeCells("C", blockCells(0, 1, 0, 1), "is-c-active is-written"),
    writeCells: blockCells(0, 1, 0, 1),
    shared: ["复用完成"],
    registers: ["4 threads", "4 acc"],
    source: sourceCatalog.tiledCompute,
  },
  {
    label: "Tiled 总结",
    title: "减少 Global Memory 重复读取",
    calculation: "Global 读取一次 → Shared Memory 复用多次",
    detail: "下一步可以优化 shared memory 访问模式，或让一个线程计算更多输出。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0, ...commonCells.c2x2],
    shared: ["Block 共享"],
    registers: ["1 output / thread"],
    source: sourceCatalog.tiledShared,
  },
];

const paddedSteps = [
  {
    label: "Shared Padding",
    title: "As/Bs 的第二维增加 1 个 padding",
    calculation: "As[TILE][TILE + 1]，Bs[TILE][TILE + 1]",
    detail: "这个版本数学和 tiled 完全一样，重点是改变 shared memory 中每一行的跨度。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0],
    shared: ["As 16×17", "Bs 16×17"],
    registers: ["acc = 0"],
    source: sourceCatalog.paddedDeclare,
  },
  {
    label: "协作加载",
    title: "加载逻辑保持不变",
    calculation: "Global A/B → padded As/Bs",
    detail: "线程仍然写入 As[ty][tx] 和 Bs[ty][tx]；多出来的一列不保存数学数据，只改变行跨度。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0],
    global: ["A tile", "B tile"],
    shared: ["As + pad", "Bs + pad"],
    registers: ["acc = 0"],
    globalArrow: true,
    source: sourceCatalog.paddedFlow,
  },
  {
    label: "复用计算",
    title: "计算公式与 Tiled 相同",
    calculation: "acc += As[ty][kk] × Bs[kk][tx]",
    detail: "padding 不改变读写的数学坐标，只改变 shared memory 物理布局，可用于观察 bank conflict 变化。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0, ...commonCells.c2x2],
    shared: ["As padded", "Bs padded"],
    registers: ["acc"],
    sharedArrow: true,
    source: sourceCatalog.paddedFlow,
  },
  {
    label: "写回",
    title: "输出仍是一线程一个 C 元素",
    calculation: "if (row < M && col < N) C[row*N+col] = acc",
    detail: "所以这个版本适合和 gemm_tiled 对比，看 padding 对 shared memory 访问效率的影响。",
    cells: rangeCells("C", blockCells(0, 1, 0, 1), "is-c-active is-written"),
    writeCells: blockCells(0, 1, 0, 1),
    shared: ["复用完成"],
    registers: ["acc"],
    source: sourceCatalog.paddedFlow,
  },
  {
    label: "Padding 总结",
    title: "这是 shared memory 物理布局优化",
    calculation: "数学不变；内存行跨度改变",
    detail: "当某些访问模式触发 shared memory bank conflict 时，padding 常用于打散冲突。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0, ...commonCells.c2x2],
    shared: ["TILE+1 stride"],
    registers: ["1 output / thread"],
    source: sourceCatalog.paddedDeclare,
  },
];

const regtile2Steps = [
  {
    label: "扩大线程职责",
    title: "一个线程负责相邻 2×2 输出",
    calculation: "1 thread → acc00, acc01, acc10, acc11",
    detail: "动画中的紫色 2×2 区域由一个线程完成，而不是四个线程。",
    cells: commonCells.c2x2,
    shared: ["As", "Bs"],
    registers: ["acc00=0", "acc01=0", "acc10=0", "acc11=0"],
    source: sourceCatalog.reg2Map,
  },
  {
    label: "协作加载",
    title: "每线程加载 2 行 A 和 2 列 B",
    calculation: "As[local_row*2+r][local_col]，Bs[local_row][local_col*2+c]",
    detail: "整个 block 合作填满更大的 As/Bs tile，因此一个 block 覆盖 32×32 输出。",
    cells: [...rangeCells("A", [[0, 0], [1, 0]], "is-a-active"), ...rangeCells("B", [[0, 0], [0, 1]], "is-b-active")],
    global: ["2×A", "2×B"],
    shared: ["As 32×16", "Bs 16×32"],
    registers: ["4 acc"],
    globalArrow: true,
    source: sourceCatalog.reg2Load,
  },
  {
    label: "读取操作数",
    title: "a0/a1/b0/b1 进入当前线程寄存器",
    calculation: "a0,a1 ← As；b0,b1 ← Bs",
    detail: "这些值会在线程内部被组合复用。",
    cells: [...rangeCells("A", [[0, 0], [1, 0]], "is-a-active"), ...rangeCells("B", [[0, 0], [0, 1]], "is-b-active"), ...commonCells.c2x2],
    shared: ["a0", "a1", "b0", "b1"],
    registers: ["acc00", "acc01", "acc10", "acc11"],
    sharedArrow: true,
    source: sourceCatalog.reg2Compute,
  },
  {
    label: "四路乘加",
    title: "2 个 A × 2 个 B 形成 4 次 FMA",
    calculation: "a0b0→acc00；a0b1→acc01；a1b0→acc10；a1b1→acc11",
    detail: "一次 shared memory 读取被当前线程用于四个输出，形成线程内复用。",
    cells: [...rangeCells("A", [[0, 0], [1, 0]], "is-a-active"), ...rangeCells("B", [[0, 0], [0, 1]], "is-b-active"), ...commonCells.c2x2],
    shared: ["a0", "a1", "b0", "b1"],
    registers: ["4 FMA"],
    sharedArrow: true,
    source: sourceCatalog.reg2Compute,
  },
  {
    label: "写回 2×2",
    title: "同一个线程连续写回四个相邻输出",
    calculation: `[[${matrices.C[0][0]}, ${matrices.C[0][1]}], [${matrices.C[1][0]}, ${matrices.C[1][1]}]]`,
    detail: "每个写回位置都有边界判断，支持 M/N 不是 tile 整倍数。",
    cells: rangeCells("C", blockCells(0, 1, 0, 1), "is-c-active is-written"),
    writeCells: blockCells(0, 1, 0, 1),
    shared: ["完成"],
    registers: ["acc00", "acc01", "acc10", "acc11"],
    source: sourceCatalog.reg2Compute,
  },
  {
    label: "Regtile2 总结",
    title: "Block 内复用 + 线程内复用",
    calculation: "256 threads × 4 outputs = 1024 outputs",
    detail: "比 tiled 多用了寄存器，但每个线程完成更多乘加，减少地址计算和调度开销。",
    cells: commonCells.c4x4,
    shared: ["As/Bs"],
    registers: ["4 outputs / thread"],
    source: sourceCatalog.reg2Map,
  },
];

const regtile4Steps = [
  {
    label: "4×4 线程职责",
    title: "一个线程负责 4×4 共 16 个输出",
    calculation: "float acc[4][4] = {}",
    detail: "这是 regtile2x2 的自然扩展：更多线程内复用，也带来更高寄存器压力。",
    cells: commonCells.c4x4,
    shared: ["As 64×16", "Bs 16×64"],
    registers: ["acc[4][4]"],
    source: sourceCatalog.reg4Map,
  },
  {
    label: "加载 4A + 4B",
    title: "每个线程装入 4 行 A 和 4 列 B",
    calculation: "for r in 0..3 load A；for c in 0..3 load B",
    detail: "整个 block 的 shared tile 扩大，一个 block 覆盖 64×64 输出。",
    cells: [
      ...rangeCells("A", [[0, 0], [1, 0], [2, 0], [3, 0]], "is-a-active"),
      ...rangeCells("B", [[0, 0], [0, 1], [0, 2], [0, 3]], "is-b-active"),
    ],
    global: ["4×A", "4×B"],
    shared: ["As 64×16", "Bs 16×64"],
    registers: ["acc[4][4]"],
    globalArrow: true,
    source: sourceCatalog.reg4LoadCompute,
  },
  {
    label: "读取 a_vals/b_vals",
    title: "4 个 A 和 4 个 B 进入线程寄存器",
    calculation: "a_vals[4] × b_vals[4]",
    detail: "当前 kk 下，4 个 A 与 4 个 B 会组合成 16 次乘加。",
    cells: [
      ...rangeCells("A", [[0, 0], [1, 0], [2, 0], [3, 0]], "is-a-active"),
      ...rangeCells("B", [[0, 0], [0, 1], [0, 2], [0, 3]], "is-b-active"),
      ...commonCells.c4x4,
    ],
    shared: ["a_vals[4]", "b_vals[4]"],
    registers: ["acc[4][4]"],
    sharedArrow: true,
    source: sourceCatalog.reg4LoadCompute,
  },
  {
    label: "16 路乘加",
    title: "双重循环更新 16 个 acc",
    calculation: "for r,c：acc[r][c] += a_vals[r] × b_vals[c]",
    detail: "这是更强的线程内复用：同一个 a_vals[r] 会被 4 个输出列复用。",
    cells: commonCells.c4x4,
    shared: ["a_vals", "b_vals"],
    registers: ["16 acc"],
    sharedArrow: true,
    source: sourceCatalog.reg4LoadCompute,
  },
  {
    label: "写回 4×4",
    title: "边界检查后写回 16 个输出",
    calculation: "if (row < M && col < N) C[row*N+col] = acc[r][c]",
    detail: "4×4 register tile 对非整倍数矩阵也能正确处理尾部输出。",
    cells: rangeCells("C", blockCells(0, 3, 0, 3), "is-c-active is-written"),
    writeCells: blockCells(0, 3, 0, 3),
    shared: ["完成"],
    registers: ["acc[4][4]"],
    source: sourceCatalog.reg4Write,
  },
  {
    label: "Regtile4 总结",
    title: "计算密度更高，但寄存器压力更大",
    calculation: "256 threads × 16 outputs = 4096 outputs",
    detail: "这个版本适合观察寄存器使用量、occupancy 和算术强度之间的权衡。",
    cells: commonCells.c4x4,
    shared: ["Block 共享"],
    registers: ["16 outputs / thread"],
    source: sourceCatalog.reg4Map,
  },
];

const float4Steps = [
  {
    label: "相邻 4 列",
    title: "一个线程负责同一行的 4 个 C 输出",
    calculation: "base_col = thread_x × 4",
    detail: "线程职责沿列方向扩展，方便一次读取 B 的 4 个连续 float。",
    cells: rangeCells("C", blockCells(0, 0, 0, 3), "is-c-active is-tile"),
    global: ["A row", "B vec4"],
    shared: ["未使用"],
    registers: ["acc0", "acc1", "acc2", "acc3"],
    source: sourceCatalog.float4Map,
  },
  {
    label: "float4 读取",
    title: "B 的连续 4 列被一次向量加载",
    calculation: "float4 b = B[k, base_col:base_col+4]",
    detail: "A 仍是标量读取；B 的四个连续列进入 b.x/b.y/b.z/b.w。",
    cells: [
      { matrix: "A", row: 0, col: 0, className: "is-a-active" },
      ...rangeCells("B", blockCells(0, 0, 0, 3), "is-b-active"),
      ...rangeCells("C", blockCells(0, 0, 0, 3), "is-c-active is-tile"),
    ],
    global: ["A[0,k]", "float4 B"],
    shared: ["未使用"],
    registers: ["b.x", "b.y", "b.z", "b.w"],
    sharedArrow: true,
    source: sourceCatalog.float4Compute,
  },
  {
    label: "四路累加",
    title: "同一个 A 标量更新 4 个 acc",
    calculation: "acc0..acc3 += a × b.x..b.w",
    detail: "这个版本强调连续访存和向量化读取，不使用 shared memory tiling。",
    cells: [
      { matrix: "A", row: 0, col: 0, className: "is-a-active" },
      ...rangeCells("B", blockCells(0, 0, 0, 3), "is-b-active"),
      ...rangeCells("C", blockCells(0, 0, 0, 3), "is-c-active is-tile"),
    ],
    global: ["A scalar", "B float4"],
    shared: ["未使用"],
    registers: ["acc0", "acc1", "acc2", "acc3"],
    sharedArrow: true,
    source: sourceCatalog.float4Compute,
  },
  {
    label: "尾部保护",
    title: "N 不是 4 的倍数时退回标量路径",
    calculation: "if (base_col+i < N) acci += ...",
    detail: "float4 读取只在安全对齐范围内使用；尾部列逐元素处理保证正确性。",
    cells: [
      ...rangeCells("B", [[0, 1], [0, 2], [0, 3]], "is-b-active"),
      ...rangeCells("C", [[0, 1], [0, 2], [0, 3]], "is-c-active is-tile"),
    ],
    global: ["scalar fallback"],
    shared: ["未使用"],
    registers: ["acc0..3"],
    source: sourceCatalog.float4Fallback,
  },
  {
    label: "写回 4 列",
    title: "一个线程写回 C[row, base_col:base_col+4]",
    calculation: `C[0,0:4] = [${matrices.C[0].join(", ")}]`,
    detail: "每个输出列都有边界判断；一个 block 在列方向覆盖 16×4=64 个输出。",
    cells: rangeCells("C", blockCells(0, 0, 0, 3), "is-c-active is-written"),
    writeCells: blockCells(0, 0, 0, 3),
    global: ["C vec4"],
    shared: ["未使用"],
    registers: ["acc0", "acc1", "acc2", "acc3"],
    source: sourceCatalog.float4Fallback,
  },
];

const wmmaSteps = [
  {
    label: "Warp Tile",
    title: "一个 warp 计算一个 16×16 C tile",
    calculation: "32 lanes → WMMA 16×16×16",
    detail: "这里的计算粒度不再是单个线程，而是一个 warp 协同执行 Tensor Core 指令。",
    cells: commonCells.c4x4,
    global: ["A half tile", "B half tile"],
    shared: ["fragment"],
    registers: ["acc_frag"],
    source: sourceCatalog.wmmaMap,
  },
  {
    label: "Fragment",
    title: "A/B/Accumulator fragment 进入寄存器级 tile 容器",
    calculation: "a_frag, b_frag, acc_frag",
    detail: "fragment 是 WMMA API 的抽象。每个 lane 持有 fragment 的一部分数据。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0, ...commonCells.c4x4],
    global: ["A half", "B half"],
    shared: ["WMMA fragment"],
    registers: ["acc_frag = 0"],
    sharedArrow: true,
    source: sourceCatalog.wmmaFragments,
  },
  {
    label: "加载矩阵片段",
    title: "load_matrix_sync 读取 A/B 的 16×16 half tile",
    calculation: "load_matrix_sync(a_frag), load_matrix_sync(b_frag)",
    detail: "当前代码使用 row_major 布局；shape 必须是 16 的倍数，否则 host 侧退回 tiled。",
    cells: [...commonCells.aTile0, ...commonCells.bTile0],
    global: ["A half tile", "B half tile"],
    shared: ["fragment load"],
    registers: ["a_frag", "b_frag"],
    sharedArrow: true,
    source: sourceCatalog.wmmaFragments,
  },
  {
    label: "Tensor Core",
    title: "mma_sync 完成 FP16×FP16→FP32 累加",
    calculation: "acc_frag = a_frag × b_frag + acc_frag",
    detail: "这一步由 Tensor Core 执行，是 WMMA 版本区别于 FP32 CUDA core kernel 的核心。",
    cells: commonCells.c4x4,
    global: ["Tensor Core"],
    shared: ["mma_sync"],
    registers: ["FP32 acc_frag"],
    sharedArrow: true,
    source: sourceCatalog.wmmaFragments,
  },
  {
    label: "写回 Fragment",
    title: "store_matrix_sync 写回 16×16 输出 tile",
    calculation: "store_matrix_sync(C + row*N + col, acc_frag, N)",
    detail: "输出 C 是 float32；输入可以是 float16，也可以由 host 侧从 float32 临时转 half。",
    cells: rangeCells("C", blockCells(0, 3, 0, 3), "is-c-active is-written"),
    writeCells: blockCells(0, 3, 0, 3),
    global: ["C FP32"],
    shared: ["fragment done"],
    registers: ["acc_frag"],
    source: sourceCatalog.wmmaFragments,
  },
  {
    label: "Host 选择",
    title: "非 16 整倍数 shape 自动退回 FP32 tiled",
    calculation: "if M/N/K not multiple of 16 → gemm_tiled_kernel",
    detail: "这样测试中的非整倍数 shape 也能通过；真正 WMMA 路径用于 16 对齐的矩阵。",
    cells: commonCells.c4x4,
    global: ["half path", "fallback path"],
    shared: ["WMMA / Tiled"],
    registers: ["acc_frag / acc"],
    source: sourceCatalog.wmmaHost,
  },
];

const stepsByMode = {
  naive: naiveSteps,
  tiled: tiledSteps,
  padded: paddedSteps,
  regtile2: regtile2Steps,
  regtile4: regtile4Steps,
  float4: float4Steps,
  wmma: wmmaSteps,
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
  renderSource(step.source);
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
  const activeTab = document.querySelector(`.mode-tab[data-mode="${mode}"]`);
  if (activeTab) {
    const tabs = activeTab.closest(".mode-tabs");
    if (tabs) {
      const left = activeTab.offsetLeft - (tabs.clientWidth - activeTab.offsetWidth) / 2;
      tabs.scrollTo({ left: Math.max(0, left), behavior: "smooth" });
    }
  }
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
