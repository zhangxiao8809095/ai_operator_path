const X = [
  [1.2, -0.4, 0.8, 2.0, -1.1, 0.5, 1.6, -0.8],
  [0.3, 1.1, -1.4, 0.7, 2.2, -0.5, 0.9, -1.0],
  [-0.7, 0.2, 1.4, -1.8, 0.6, 1.0, -0.2, 2.4],
];

const gamma = [0.9, 1.1, 0.8, 1.2, 1.0, 0.7, 1.3, 0.95];
const beta = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.05, -0.15];
const activeRow = 0;
const cols = X[0].length;
const layerEps = 1e-5;
const rmsEps = 1e-6;

function fmt(value) {
  if (typeof value === "string") return value;
  if (!Number.isFinite(value)) return "";
  const rounded = Math.abs(value) < 0.005 ? 0 : value;
  return rounded.toFixed(2).replace(/\.?0+$/, "");
}

function sum(values) {
  return values.reduce((acc, value) => acc + value, 0);
}

function mean(values) {
  return sum(values) / values.length;
}

function layerStats(row) {
  const rowMean = mean(row);
  const rowVar = mean(row.map((value) => (value - rowMean) ** 2));
  const invStd = 1 / Math.sqrt(rowVar + layerEps);
  return { rowMean, rowVar, invStd };
}

function rmsStats(row) {
  const rowMeanSq = mean(row.map((value) => value * value));
  const invRms = 1 / Math.sqrt(rowMeanSq + rmsEps);
  return { rowMeanSq, invRms };
}

function layerOut(row) {
  const stats = layerStats(row);
  return row.map((value, col) => (value - stats.rowMean) * stats.invStd * gamma[col] + beta[col]);
}

function rmsOut(row) {
  const stats = rmsStats(row);
  return row.map((value, col) => value * stats.invRms * gamma[col]);
}

function grouped(values, groupSize) {
  const groups = [];
  for (let index = 0; index < values.length; index += groupSize) {
    groups.push(values.slice(index, index + groupSize));
  }
  return groups;
}

const demoRow = X[activeRow];
const demoLayer = layerStats(demoRow);
const demoRms = rmsStats(demoRow);
const layerY = X.map(layerOut);
const rmsY = X.map(rmsOut);
const demoSquares = demoRow.map((value) => value * value);
const demoCenteredSquares = demoRow.map((value) => (value - demoLayer.rowMean) ** 2);
const vecGroups = grouped(demoRow, 4);
const vecSquareSums = vecGroups.map((group) => sum(group.map((value) => value * value)));
const vecSums = vecGroups.map(sum);
const vecVarSums = grouped(demoCenteredSquares, 4).map(sum);

function sourceLine(number, text, active) {
  return { number, text, active: Boolean(active) };
}

function src(location, explanation, lines) {
  return { location, explanation, lines };
}

const sourceCatalog = {
  rowLaunchers: src("norm.cu:229-258", "基础入口还是最直接：检查输入，拿 rows/cols，分配 Y，然后启动 row kernel。", [
    sourceLine(229, "torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {", true),
    sourceLine(230, "    CHECK_INPUT(X);"),
    sourceLine(236, "    int rows = static_cast<int>(X.size(0));", true),
    sourceLine(237, "    int cols = static_cast<int>(X.size(1));", true),
    sourceLine(238, "    auto Y = torch::empty_like(X);"),
    sourceLine(239, "    int block = 256;", true),
    sourceLine(240, "    layernorm_row_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(246, "torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps) {", true),
    sourceLine(255, "    rmsnorm_row_kernel<<<rows, block, block * sizeof(float)>>>("),
  ]),
  warpLaunchers: src("norm.cu:261-318", "warp-reduce 入口多了空矩阵快速返回；规约逻辑进入 block_reduce_sum，kernel 本身更短。", [
    sourceLine(261, "torch::Tensor layernorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {", true),
    sourceLine(268, "    int rows = static_cast<int>(X.size(0));"),
    sourceLine(269, "    int cols = static_cast<int>(X.size(1));"),
    sourceLine(270, "    auto Y = torch::empty_like(X);"),
    sourceLine(271, "    if (rows == 0 || cols == 0) return Y;", true),
    sourceLine(273, "    layernorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(305, "torch::Tensor rmsnorm_warp_reduce(torch::Tensor X, torch::Tensor gamma, double eps) {", true),
    sourceLine(315, "    rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>("),
  ]),
  vectorLaunchers: src("norm.cu:279-343", "vectorized 入口先判断 cols 和指针是否适合 float4；条件不满足就退回 warp_reduce kernel，保证正确性。", [
    sourceLine(279, "torch::Tensor layernorm_vectorized(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {", true),
    sourceLine(289, "    if (rows == 0 || cols == 0) return Y;"),
    sourceLine(291, "    bool can_vectorize = (cols % 4 == 0) && is_aligned_16(X) && is_aligned_16(gamma) &&", true),
    sourceLine(292, "                          is_aligned_16(beta) && is_aligned_16(Y);", true),
    sourceLine(293, "    if (can_vectorize) {", true),
    sourceLine(294, "        layernorm_vectorized_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(297, "    } else {", true),
    sourceLine(298, "        layernorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(321, "torch::Tensor rmsnorm_vectorized(torch::Tensor X, torch::Tensor gamma, double eps) {"),
    sourceLine(333, "    if (can_vectorize) {"),
    sourceLine(338, "        rmsnorm_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>("),
  ]),
  align: src("norm.cu:31-32, 291-292", "float4 一次读写 16 字节，所以 X、gamma、beta、Y 的地址都要 16 字节对齐。", [
    sourceLine(31, "bool is_aligned_16(const torch::Tensor& tensor) {", true),
    sourceLine(32, "    return (reinterpret_cast<std::uintptr_t>(tensor.data_ptr<float>()) % 16) == 0;", true),
    sourceLine(291, "    bool can_vectorize = (cols % 4 == 0) && is_aligned_16(X) && is_aligned_16(gamma) &&", true),
    sourceLine(292, "                          is_aligned_16(beta) && is_aligned_16(Y);", true),
  ]),
  rowRmsMap: src("norm.cu:75-87", "基础 RMSNorm kernel 里，一个 block 处理一行，线程用 tid 跨步扫描 cols。", [
    sourceLine(75, "__global__ void rmsnorm_row_kernel(const float* __restrict__ X,", true),
    sourceLine(79, "    extern __shared__ float smem[];"),
    sourceLine(80, "    int row = blockIdx.x;", true),
    sourceLine(81, "    int tid = threadIdx.x;", true),
    sourceLine(83, "    float local_sum_sq = 0.0f;"),
    sourceLine(84, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(85, "        float v = X[row * cols + col];"),
    sourceLine(86, "        local_sum_sq += v * v;", true),
  ]),
  rowRmsReduce: src("norm.cu:88-98", "基础版把每个线程的局部平方和写进 smem，再用 stride 逐半缩小做树形规约。", [
    sourceLine(88, "    smem[tid] = local_sum_sq;", true),
    sourceLine(89, "    __syncthreads();", true),
    sourceLine(90, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(91, "        if (tid < stride) smem[tid] += smem[tid + stride];", true),
    sourceLine(92, "        __syncthreads();"),
    sourceLine(94, "    float inv_rms = rsqrtf(smem[0] / cols + eps);", true),
    sourceLine(96, "    for (int col = tid; col < cols; col += blockDim.x) {"),
    sourceLine(97, "        Y[row * cols + col] = X[row * cols + col] * inv_rms * gamma[col];", true),
  ]),
  rowLayerMean: src("norm.cu:35-54", "基础 LayerNorm 先求整行 mean；这一段和 RMSNorm 的树形规约形式一样。", [
    sourceLine(35, "__global__ void layernorm_row_kernel(const float* __restrict__ X,", true),
    sourceLine(40, "    extern __shared__ float smem[];"),
    sourceLine(41, "    int row = blockIdx.x;"),
    sourceLine(42, "    int tid = threadIdx.x;"),
    sourceLine(44, "    float local_sum = 0.0f;", true),
    sourceLine(45, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(48, "    smem[tid] = local_sum;", true),
    sourceLine(50, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {"),
    sourceLine(54, "    float mean = smem[0] / cols;", true),
  ]),
  rowLayerVar: src("norm.cu:56-72", "LayerNorm 知道 mean 后，还要第二遍扫描计算 variance，最后套 gamma/beta。", [
    sourceLine(56, "    float local_var = 0.0f;", true),
    sourceLine(57, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(58, "        float v = X[row * cols + col] - mean;", true),
    sourceLine(59, "        local_var += v * v;", true),
    sourceLine(61, "    smem[tid] = local_var;"),
    sourceLine(67, "    float inv_std = rsqrtf(smem[0] / cols + eps);", true),
    sourceLine(70, "        float norm = (X[row * cols + col] - mean) * inv_std;", true),
    sourceLine(71, "        Y[row * cols + col] = norm * gamma[col] + beta[col];", true),
  ]),
  warpReduceSum: src("norm.cu:8-12", "warp_reduce_sum 用 shuffle 在一个 warp 内交换寄存器值，不需要每一轮都写 shared memory。", [
    sourceLine(8, "__forceinline__ __device__ float warp_reduce_sum(float val) {", true),
    sourceLine(9, "    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {", true),
    sourceLine(10, "        val += __shfl_down_sync(0xffffffff, val, offset);", true),
    sourceLine(11, "    }"),
    sourceLine(12, "    return val;", true),
  ]),
  blockReduceSum: src("norm.cu:15-29", "block_reduce_sum 先让每个 warp 内部规约，再把每个 warp 的结果放进 smem，最后由 warp 0 合并。", [
    sourceLine(15, "__forceinline__ __device__ float block_reduce_sum(float val, float* smem) {", true),
    sourceLine(16, "    int lane = threadIdx.x & (warpSize - 1);", true),
    sourceLine(17, "    int warp_id = threadIdx.x / warpSize;", true),
    sourceLine(18, "    int warp_count = (blockDim.x + warpSize - 1) / warpSize;"),
    sourceLine(20, "    val = warp_reduce_sum(val);", true),
    sourceLine(21, "    if (lane == 0) smem[warp_id] = val;", true),
    sourceLine(24, "    val = (threadIdx.x < warp_count) ? smem[lane] : 0.0f;"),
    sourceLine(25, "    if (warp_id == 0) val = warp_reduce_sum(val);", true),
    sourceLine(26, "    if (threadIdx.x == 0) smem[0] = val;"),
    sourceLine(28, "    return smem[0];", true),
  ]),
  warpRmsKernel: src("norm.cu:130-148", "RMSNorm 的 warp-reduce 版保留数学流程，只把 smem 树形规约替换成 block_reduce_sum。", [
    sourceLine(130, "__global__ void rmsnorm_warp_reduce_kernel(const float* __restrict__ X,", true),
    sourceLine(135, "    int row = blockIdx.x;"),
    sourceLine(137, "    if (row >= rows) return;", true),
    sourceLine(139, "    float local_sum_sq = 0.0f;"),
    sourceLine(140, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(142, "        local_sum_sq += v * v;", true),
    sourceLine(144, "    float inv_rms = rsqrtf(block_reduce_sum(local_sum_sq, smem) / cols + eps);", true),
    sourceLine(147, "        Y[row * cols + col] = X[row * cols + col] * inv_rms * gamma[col];"),
  ]),
  warpLayerKernel: src("norm.cu:101-127", "LayerNorm 的 warp-reduce 版调用两次 block_reduce_sum：一次求 mean，一次求 variance。", [
    sourceLine(101, "__global__ void layernorm_warp_reduce_kernel(const float* __restrict__ X,", true),
    sourceLine(107, "    int row = blockIdx.x;"),
    sourceLine(109, "    if (row >= rows) return;", true),
    sourceLine(111, "    float local_sum = 0.0f;"),
    sourceLine(115, "    float mean = block_reduce_sum(local_sum, smem) / cols;", true),
    sourceLine(117, "    float local_var = 0.0f;"),
    sourceLine(122, "    float inv_std = rsqrtf(block_reduce_sum(local_var, smem) / cols + eps);", true),
    sourceLine(126, "        Y[row * cols + col] = norm * gamma[col] + beta[col];", true),
  ]),
  vectorRms: src("norm.cu:196-224", "RMSNorm vectorized kernel 把同一行 reinterpret 成 float4 数组，一次处理 4 个连续列。", [
    sourceLine(196, "__global__ void rmsnorm_vectorized_kernel(const float* __restrict__ X,", true),
    sourceLine(203, "    int vec_cols = cols / 4;", true),
    sourceLine(205, "    const float4* X4 = reinterpret_cast<const float4*>(X + row * cols);", true),
    sourceLine(207, "    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {", true),
    sourceLine(208, "        float4 x = X4[vec_col];", true),
    sourceLine(209, "        local_sum_sq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;", true),
    sourceLine(211, "    float inv_rms = rsqrtf(block_reduce_sum(local_sum_sq, smem) / cols + eps);"),
    sourceLine(213, "    float4* Y4 = reinterpret_cast<float4*>(Y + row * cols);", true),
    sourceLine(214, "    const float4* G4 = reinterpret_cast<const float4*>(gamma);", true),
    sourceLine(219, "        y.x = x.x * inv_rms * g.x;"),
    sourceLine(223, "        Y4[vec_col] = y;", true),
  ]),
  vectorLayer: src("norm.cu:151-193", "LayerNorm vectorized 版同样用 float4，只是 sum 和 variance 都要把 x/y/z/w 四个分量累进去。", [
    sourceLine(151, "__global__ void layernorm_vectorized_kernel(const float* __restrict__ X,", true),
    sourceLine(159, "    int vec_cols = cols / 4;", true),
    sourceLine(161, "    const float4* X4 = reinterpret_cast<const float4*>(X + row * cols);", true),
    sourceLine(163, "    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {", true),
    sourceLine(164, "        float4 x = X4[vec_col];", true),
    sourceLine(165, "        local_sum += x.x + x.y + x.z + x.w;", true),
    sourceLine(167, "    float mean = block_reduce_sum(local_sum, smem) / cols;"),
    sourceLine(176, "        local_var += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;", true),
    sourceLine(180, "    float4* Y4 = reinterpret_cast<float4*>(Y + row * cols);", true),
    sourceLine(188, "        y.x = (x.x - mean) * inv_std * g.x + b.x;", true),
    sourceLine(192, "        Y4[vec_col] = y;", true),
  ]),
};

const allCols = [0, 1, 2, 3, 4, 5, 6, 7];

function makeStep(base) {
  const defaults = {
    activeRow: [activeRow],
    activeCols: [],
    activeGamma: [],
    activeBeta: [],
    hostState: "准备",
    gridState: "rows 个 block",
    blockState: "256 threads",
    pipe: [],
    hostArrow: false,
    gridArrow: false,
    concept: 0,
    smem: null,
    smemActive: [],
    thread: "idle",
    output: null,
    showAllOutput: false,
    betaMode: "normal",
    stats: [],
  };
  return Object.assign({}, defaults, base);
}

const launcherSteps = [
  makeStep({
    label: "入口总览",
    title: "新版 norm.cu 暴露 6 个 norm API",
    calculation: "layernorm/rmsnorm × row/warp_reduce/vectorized",
    detail: "先从外层看：row 是基础版，warp_reduce 改规约方式，vectorized 在满足条件时走 float4，否则 fallback 到 warp_reduce。",
    hostState: "Python API",
    gridState: "未 launch",
    blockState: "等待选择",
    pipe: ["host"],
    source: sourceCatalog.rowLaunchers,
    stats: [
      ["LayerNorm", "row / warp / vectorized"],
      ["RMSNorm", "row / warp / vectorized"],
      ["共同输入", "X[rows, cols]"],
      ["输出", "Y same shape"],
    ],
  }),
  makeStep({
    label: "基础入口",
    title: "row 版本直接启动基础 kernel",
    calculation: "layernorm_row_kernel<<<rows, 256, smem>>>",
    detail: "这仍然是最好理解的一版：一个 block 处理一行，shared memory 保存每个线程的局部结果。",
    hostState: "检查 + 分配 Y",
    gridState: `${X.length} blocks`,
    blockState: "256 threads",
    pipe: ["host", "grid", "block"],
    hostArrow: true,
    gridArrow: true,
    concept: 1,
    source: sourceCatalog.rowLaunchers,
    thread: "row",
    smem: allCols.map((index) => `tid ${index}`),
    smemActive: allCols,
    stats: [
      ["函数", "layernorm_row / rmsnorm_row"],
      ["kernel", "基础 row kernel"],
      ["shared", "block floats"],
      ["难度", "最低"],
    ],
  }),
  makeStep({
    label: "warp 入口",
    title: "warp_reduce 版本先处理空矩阵",
    calculation: "if (rows == 0 || cols == 0) return Y",
    detail: "新版 wrapper 对空矩阵更稳：先分配空输出，再直接返回，避免 launch 一个没有实际工作的 kernel。",
    hostState: "空矩阵保护",
    gridState: "rows/cols checked",
    blockState: "等待 launch",
    pipe: ["host"],
    concept: 0,
    source: sourceCatalog.warpLaunchers,
    stats: [
      ["函数", "layernorm_warp_reduce"],
      ["函数", "rmsnorm_warp_reduce"],
      ["新增", "empty return"],
      ["规约", "block_reduce_sum"],
    ],
  }),
  makeStep({
    label: "向量化入口",
    title: "vectorized 版本先判断能不能 float4",
    calculation: "can_vectorize = cols % 4 == 0 && aligned_16(...)",
    detail: "float4 一次处理 4 个 float，要求列数按 4 分组，也要求相关指针 16 字节对齐。",
    hostState: "检查 can_vectorize",
    gridState: "rows 个 block",
    blockState: "float4 path?",
    pipe: ["host"],
    concept: 4,
    source: sourceCatalog.vectorLaunchers,
    thread: "vec",
    smem: ["vec0", "vec1", "", "", "", "", "", ""],
    smemActive: [0, 1],
    activeCols: allCols,
    stats: [
      ["cols % 4", cols % 4 === 0 ? "yes" : "no"],
      ["对齐", "16 bytes"],
      ["成功", "vectorized_kernel"],
      ["失败", "warp_reduce_kernel"],
    ],
  }),
  makeStep({
    label: "fallback",
    title: "不满足 float4 条件也能正确运行",
    calculation: "else → *_warp_reduce_kernel<<<rows, block, smem>>>",
    detail: "vectorized wrapper 的关键不是永远快，而是先尝试快路径，条件不合适时自动回到可靠的 warp-reduce 版本。",
    hostState: "选择 kernel",
    gridState: "launch selected",
    blockState: "warp fallback",
    pipe: ["host", "grid", "block"],
    hostArrow: true,
    gridArrow: true,
    concept: 4,
    source: sourceCatalog.vectorLaunchers,
    thread: "fallback",
    smem: ["fast path", "fallback", "", "", "", "", "", ""],
    smemActive: [0, 1],
    stats: [
      ["快路径", "float4"],
      ["回退", "warp reduce"],
      ["目标", "正确优先"],
      ["难度", "最高"],
    ],
  }),
];

const rowSteps = [
  makeStep({
    label: "行级任务",
    title: "基础版：一个 block 处理一行",
    calculation: "row = blockIdx.x；col = tid, tid + blockDim.x, ...",
    detail: "这是最容易理解的 kernel 版本。动画用 8 个 lane 表示真实代码里的 256 个线程。",
    hostState: "row kernel",
    gridState: `block ${activeRow} → row ${activeRow}`,
    blockState: "tid scans cols",
    pipe: ["grid", "block"],
    gridArrow: true,
    concept: 0,
    source: sourceCatalog.rowRmsMap,
    thread: "row",
    activeCols: allCols,
    stats: [
      ["并行单位", "1 block / row"],
      ["线程数", "256"],
      ["动画 lane", "8"],
      ["数据", "X[row, col]"],
    ],
  }),
  makeStep({
    label: "RMSNorm",
    title: "先看简单的 RMSNorm：统计 x²",
    calculation: "local_sum_sq += x * x",
    detail: "RMSNorm 不减 mean，只关心这一行的平方均值，所以基础版只需要一次规约。",
    hostState: "rmsnorm_row",
    gridState: "row blocks",
    blockState: "square sum",
    pipe: ["block"],
    concept: 1,
    source: sourceCatalog.rowRmsMap,
    thread: "square",
    activeCols: allCols,
    smem: demoSquares,
    smemActive: allCols,
    betaMode: "off",
    stats: [
      ["sum(x²)", fmt(sum(demoSquares))],
      ["mean(x²)", fmt(demoRms.rowMeanSq)],
      ["reduction", "1 次"],
      ["beta", "不用"],
    ],
  }),
  makeStep({
    label: "树形规约",
    title: "基础版用 smem[tid] 和 stride 合成总和",
    calculation: "if (tid < stride) smem[tid] += smem[tid + stride]",
    detail: "这就是最朴素的 block 内规约：每一轮同步一次，stride 每次减半。",
    hostState: "rmsnorm_row",
    gridState: "row blocks",
    blockState: "smem reduction",
    pipe: ["block"],
    concept: 2,
    source: sourceCatalog.rowRmsReduce,
    thread: "reduce",
    activeCols: allCols,
    smem: [sum(demoSquares), "", "", "", "", "", "", ""],
    smemActive: [0],
    betaMode: "off",
    stats: [
      ["smem[0]", fmt(sum(demoSquares))],
      ["inv_rms", fmt(demoRms.invRms)],
      ["同步", "每轮 __syncthreads"],
      ["输出", "x * inv_rms * gamma"],
    ],
  }),
  makeStep({
    label: "RMS 写回",
    title: "每个线程写回自己负责的列",
    calculation: "Y = X * inv_rms * gamma[col]",
    detail: "RMSNorm 的输出没有 beta。它的轻量感来自：少一次 mean，少一次 variance，少一次 shift。",
    hostState: "rmsnorm_row",
    gridState: "row blocks",
    blockState: "write Y",
    pipe: ["block"],
    concept: 2,
    source: sourceCatalog.rowRmsReduce,
    thread: "writeRms",
    activeCols: allCols,
    activeGamma: allCols,
    output: "rms",
    betaMode: "off",
    smem: [fmt(demoRms.invRms), "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["函数", "rmsnorm_row_kernel"],
      ["统计量", "mean(x²)"],
      ["公式", "x * inv_rms * gamma"],
      ["难度", "基础"],
    ],
  }),
  makeStep({
    label: "LayerNorm mean",
    title: "LayerNorm 多一步：先求 mean",
    calculation: "mean = sum(row) / cols",
    detail: "LayerNorm 的第一轮规约求原值总和。它和 RMS 的平方和规约形式一样，只是统计对象变了。",
    hostState: "layernorm_row",
    gridState: "row blocks",
    blockState: "sum X",
    pipe: ["block"],
    concept: 1,
    source: sourceCatalog.rowLayerMean,
    thread: "sum",
    activeCols: allCols,
    smem: demoRow,
    smemActive: allCols,
    stats: [
      ["sum(x)", fmt(sum(demoRow))],
      ["mean", fmt(demoLayer.rowMean)],
      ["reduction", "第 1 次"],
      ["下一步", "variance"],
    ],
  }),
  makeStep({
    label: "LayerNorm variance",
    title: "第二轮扫描：统计 (x - mean)²",
    calculation: "local_var += (x - mean) * (x - mean)",
    detail: "只有知道 mean 之后才能算 variance，所以 LayerNorm 比 RMSNorm 多一轮读 X 和一轮规约。",
    hostState: "layernorm_row",
    gridState: "row blocks",
    blockState: "variance",
    pipe: ["block"],
    concept: 2,
    source: sourceCatalog.rowLayerVar,
    thread: "variance",
    activeCols: allCols,
    smem: demoCenteredSquares,
    smemActive: allCols,
    stats: [
      ["mean", fmt(demoLayer.rowMean)],
      ["var", fmt(demoLayer.rowVar)],
      ["inv_std", fmt(demoLayer.invStd)],
      ["reduction", "第 2 次"],
    ],
  }),
  makeStep({
    label: "Affine 写回",
    title: "最后套 gamma 和 beta",
    calculation: "Y = (X - mean) * inv_std * gamma[col] + beta[col]",
    detail: "gamma 是缩放，beta 是平移；同一组 gamma/beta 会广播到每一行。",
    hostState: "layernorm_row",
    gridState: "row blocks",
    blockState: "write Y",
    pipe: ["block"],
    concept: 2,
    source: sourceCatalog.rowLayerVar,
    thread: "writeLayer",
    activeCols: allCols,
    activeGamma: allCols,
    activeBeta: allCols,
    output: "layer",
    smem: [fmt(demoLayer.invStd), "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["函数", "layernorm_row_kernel"],
      ["统计量", "mean + variance"],
      ["公式", "norm * gamma + beta"],
      ["成本", "2 次 reduction"],
    ],
  }),
];

const warpSteps = [
  makeStep({
    label: "为什么优化",
    title: "基础树形规约每轮都要 shared memory 和同步",
    calculation: "smem[tid] += smem[tid + stride]；__syncthreads()",
    detail: "warp-reduce 版本先把规约抽成 helper，然后用 warp shuffle 减少 shared memory 参与的次数。",
    hostState: "warp helper",
    gridState: "same rows",
    blockState: "reduce faster",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.blockReduceSum,
    thread: "reduce",
    activeCols: allCols,
    smem: ["warp0", "warp1", "warp2", "warp3", "warp4", "warp5", "warp6", "warp7"],
    smemActive: allCols,
    stats: [
      ["目标", "减少同步和 smem 压力"],
      ["工具", "warp shuffle"],
      ["封装", "block_reduce_sum"],
      ["输出", "smem[0]"],
    ],
  }),
  makeStep({
    label: "warp 内规约",
    title: "warp_reduce_sum 在寄存器之间交换值",
    calculation: "val += __shfl_down_sync(mask, val, offset)",
    detail: "同一个 warp 里的线程可以用 shuffle 直接取到其他 lane 的 val，不必每次都落到 shared memory。",
    hostState: "warp_reduce_sum",
    gridState: "warp lanes",
    blockState: "shuffle",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.warpReduceSum,
    thread: "shuffle",
    activeCols: allCols,
    smem: ["lane+16", "lane+8", "lane+4", "lane+2", "lane+1", "", "", ""],
    smemActive: [0, 1, 2, 3, 4],
    stats: [
      ["offset", "16, 8, 4, 2, 1"],
      ["范围", "一个 warp"],
      ["内存", "寄存器交换"],
      ["函数", "warp_reduce_sum"],
    ],
  }),
  makeStep({
    label: "block 规约",
    title: "每个 warp 的 lane0 写入 smem[warp_id]",
    calculation: "if (lane == 0) smem[warp_id] = val",
    detail: "block 里通常有多个 warp。第一阶段每个 warp 得到一个局部总和，再把这个总和交给 shared memory。",
    hostState: "block_reduce_sum",
    gridState: "warp groups",
    blockState: "warp sums",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.blockReduceSum,
    thread: "warpWrite",
    activeCols: allCols,
    smem: ["warp0 sum", "warp1 sum", "warp2 sum", "warp3 sum", "warp4 sum", "warp5 sum", "warp6 sum", "warp7 sum"],
    smemActive: allCols,
    stats: [
      ["lane", "threadIdx.x & 31"],
      ["warp_id", "threadIdx.x / 32"],
      ["smem", "每 warp 一个值"],
      ["同步", "跨 warp 前需要"],
    ],
  }),
  makeStep({
    label: "warp0 汇总",
    title: "warp 0 再把所有 warp 的结果合成 smem[0]",
    calculation: "if (warp_id == 0) val = warp_reduce_sum(val)",
    detail: "第二阶段只让 warp 0 工作。最后 thread 0 把 block 总和写进 smem[0]，供所有线程读取。",
    hostState: "block_reduce_sum",
    gridState: "warp 0",
    blockState: "final sum",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.blockReduceSum,
    thread: "warpFinal",
    activeCols: allCols,
    smem: ["block sum", "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["结果", "smem[0]"],
      ["返回", "return smem[0]"],
      ["同步", "写完后同步"],
      ["复用", "RMS + Layer"],
    ],
  }),
  makeStep({
    label: "RMS warp kernel",
    title: "RMSNorm 只调用一次 block_reduce_sum",
    calculation: "inv_rms = rsqrtf(block_reduce_sum(local_sum_sq, smem) / cols + eps)",
    detail: "数学不变，优化点只在规约实现：基础版显式写 stride 循环，warp 版交给 helper。",
    hostState: "rmsnorm_warp_reduce",
    gridState: "row blocks",
    blockState: "block_reduce_sum",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.warpRmsKernel,
    thread: "writeRms",
    activeCols: allCols,
    activeGamma: allCols,
    output: "rms",
    betaMode: "off",
    smem: [fmt(demoRms.invRms), "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["kernel", "rmsnorm_warp_reduce_kernel"],
      ["规约", "1 次 block_reduce_sum"],
      ["边界", "if (row >= rows)"],
      ["输出", "x * inv_rms * gamma"],
    ],
  }),
  makeStep({
    label: "Layer warp kernel",
    title: "LayerNorm 调两次 block_reduce_sum",
    calculation: "mean = reduce(sum) / cols；inv_std = rsqrtf(reduce(var) / cols + eps)",
    detail: "LayerNorm 的数据流仍然是 mean → variance → affine，只是两次规约都换成了 warp-aware helper。",
    hostState: "layernorm_warp_reduce",
    gridState: "row blocks",
    blockState: "two reductions",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.warpLayerKernel,
    thread: "writeLayer",
    activeCols: allCols,
    activeGamma: allCols,
    activeBeta: allCols,
    output: "layer",
    smem: [fmt(demoLayer.invStd), "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["kernel", "layernorm_warp_reduce_kernel"],
      ["规约", "2 次 block_reduce_sum"],
      ["边界", "if (row >= rows)"],
      ["输出", "norm * gamma + beta"],
    ],
  }),
];

const vectorSteps = [
  makeStep({
    label: "对齐判断",
    title: "float4 快路径要先过 can_vectorize",
    calculation: "cols % 4 == 0 && is_aligned_16(X/gamma/beta/Y)",
    detail: "float4 等于 4 个 float，也就是 16 字节。地址不对齐或 cols 不能被 4 整除时，不能安全地按 float4 读写。",
    hostState: "vectorized wrapper",
    gridState: "choose path",
    blockState: "alignment",
    pipe: ["host"],
    concept: 4,
    source: sourceCatalog.align,
    thread: "vec",
    activeCols: allCols,
    smem: ["cols%4", "align X", "align gamma", "align beta/Y", "", "", "", ""],
    smemActive: [0, 1, 2, 3],
    stats: [
      ["cols", `${cols}`],
      ["vec_cols", `${cols / 4}`],
      ["float4", "16 bytes"],
      ["fallback", "warp_reduce"],
    ],
  }),
  makeStep({
    label: "float4 分组",
    title: "8 列在动画里变成 2 个 float4",
    calculation: "vec_cols = cols / 4",
    detail: "真实数据按连续内存排布。reinterpret_cast 后，X[row, 0..3] 是 X4[0]，X[row, 4..7] 是 X4[1]。",
    hostState: "vectorized kernel",
    gridState: "row blocks",
    blockState: "vec_col",
    pipe: ["grid", "block"],
    gridArrow: true,
    concept: 4,
    source: sourceCatalog.vectorRms,
    thread: "vec",
    activeCols: allCols,
    smem: [`vec0: ${vecGroups[0].map(fmt).join(",")}`, `vec1: ${vecGroups[1].map(fmt).join(",")}`, "", "", "", "", "", ""],
    smemActive: [0, 1],
    stats: [
      ["vec0", "cols 0..3"],
      ["vec1", "cols 4..7"],
      ["读取", "float4 x"],
      ["线程分工", "vec_col += blockDim.x"],
    ],
  }),
  makeStep({
    label: "RMS float4",
    title: "RMSNorm 一次累加 4 个平方",
    calculation: "sum_sq += x.x² + x.y² + x.z² + x.w²",
    detail: "这和基础版逐元素累加的数学结果一样，但每次加载拿到四个连续元素。",
    hostState: "rmsnorm_vectorized",
    gridState: "row blocks",
    blockState: "float4 sum_sq",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.vectorRms,
    thread: "vecSquare",
    activeCols: allCols,
    smem: [vecSquareSums[0], vecSquareSums[1], "", "", "", "", "", ""],
    smemActive: [0, 1],
    betaMode: "off",
    stats: [
      ["vec0 x²", fmt(vecSquareSums[0])],
      ["vec1 x²", fmt(vecSquareSums[1])],
      ["reduce", "block_reduce_sum"],
      ["inv_rms", fmt(demoRms.invRms)],
    ],
  }),
  makeStep({
    label: "RMS float4 写回",
    title: "Y4[vec_col] 一次写 4 个输出",
    calculation: "y.x/y.y/y.z/y.w = x.* * inv_rms * g.*",
    detail: "gamma 也 reinterpret 成 float4，所以每个分量都有自己的缩放参数。",
    hostState: "rmsnorm_vectorized",
    gridState: "row blocks",
    blockState: "Y4 store",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.vectorRms,
    thread: "writeRmsVec",
    activeCols: allCols,
    activeGamma: allCols,
    output: "rms",
    betaMode: "off",
    smem: [fmt(demoRms.invRms), "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["X", "float4"],
      ["gamma", "float4"],
      ["Y", "float4"],
      ["输出", "4 cols / store"],
    ],
  }),
  makeStep({
    label: "Layer float4",
    title: "LayerNorm 的 float4 也要先求 mean",
    calculation: "local_sum += x.x + x.y + x.z + x.w",
    detail: "LayerNorm vectorized 版只是把每轮读取从单个 float 变成 float4；mean 和 variance 的顺序不变。",
    hostState: "layernorm_vectorized",
    gridState: "row blocks",
    blockState: "float4 sum",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.vectorLayer,
    thread: "vecSum",
    activeCols: allCols,
    smem: [vecSums[0], vecSums[1], "", "", "", "", "", ""],
    smemActive: [0, 1],
    stats: [
      ["vec0 sum", fmt(vecSums[0])],
      ["vec1 sum", fmt(vecSums[1])],
      ["mean", fmt(demoLayer.rowMean)],
      ["reduce", "block_reduce_sum"],
    ],
  }),
  makeStep({
    label: "Layer variance",
    title: "四个分量分别减 mean 后求平方",
    calculation: "local_var += v0² + v1² + v2² + v3²",
    detail: "float4 不改变 LayerNorm 的数学，只让连续列的加载和写回更粗粒度。",
    hostState: "layernorm_vectorized",
    gridState: "row blocks",
    blockState: "float4 var",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.vectorLayer,
    thread: "vecVar",
    activeCols: allCols,
    smem: [vecVarSums[0], vecVarSums[1], "", "", "", "", "", ""],
    smemActive: [0, 1],
    stats: [
      ["vec0 var sum", fmt(vecVarSums[0])],
      ["vec1 var sum", fmt(vecVarSums[1])],
      ["inv_std", fmt(demoLayer.invStd)],
      ["reduce", "block_reduce_sum"],
    ],
  }),
  makeStep({
    label: "Layer float4 写回",
    title: "X/Gamma/Beta/Y 都按 float4 读写",
    calculation: "Y4 = (X4 - mean) * inv_std * G4 + B4",
    detail: "这是当前 norm.cu 里最复杂的一层：有 wrapper 条件选择、warp 规约 helper，还有 float4 读写。",
    hostState: "layernorm_vectorized",
    gridState: "row blocks",
    blockState: "Y4 store",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.vectorLayer,
    thread: "writeLayerVec",
    activeCols: allCols,
    activeGamma: allCols,
    activeBeta: allCols,
    output: "layer",
    smem: [fmt(demoLayer.invStd), "", "", "", "", "", "", ""],
    smemActive: [0],
    stats: [
      ["X", "float4"],
      ["gamma/beta", "float4"],
      ["fallback", "warp_reduce"],
      ["难度", "最高"],
    ],
  }),
];

const modeConfig = {
  launcher: {
    title: "Host Launcher：从 Python API 到 kernel 选择",
    summary: "先看最容易的外层：Python 调用进入 C++，检查 tensor，分配输出，再选择基础版、warp-reduce 版或 float4 版。",
    stats: {
      function: "layernorm_* / rmsnorm_*",
      unit: "Python API → C++ launcher",
      action: "检查 + launch / fallback",
      formula: "row / warp / vectorized",
    },
    steps: launcherSteps,
  },
  row: {
    title: "Row Kernel：基础 shared-memory 规约",
    summary: "一个 block 处理一行。RMSNorm 做一次平方和规约，LayerNorm 做 mean 和 variance 两次规约。",
    stats: {
      function: "rmsnorm_row_kernel / layernorm_row_kernel",
      unit: "1 block / row",
      action: "smem[tid] + stride",
      formula: "RMS 1 次 / Layer 2 次",
    },
    steps: rowSteps,
  },
  warp: {
    title: "Warp Reduce：把规约抽成 shuffle helper",
    summary: "数学流程不变，规约方式升级：warp 内用 __shfl_down_sync，block 内只保存每个 warp 的局部结果。",
    stats: {
      function: "warp_reduce_sum / block_reduce_sum",
      unit: "warp → block",
      action: "shuffle + smem 汇总",
      formula: "return smem[0]",
    },
    steps: warpSteps,
  },
  vectorized: {
    title: "Float4 Vectorized：4 个连续 float 一组处理",
    summary: "在 cols 可被 4 整除且地址 16 字节对齐时，vectorized kernel 用 float4 加粗读取和写回；否则自动 fallback。",
    stats: {
      function: "layernorm_vectorized / rmsnorm_vectorized",
      unit: "4 cols / vec_col",
      action: "float4 load/store",
      formula: "fast path or fallback",
    },
    steps: vectorSteps,
  },
};

const dom = {
  modeTitle: document.querySelector("#mode-title"),
  modeSummary: document.querySelector("#mode-summary"),
  progressBar: document.querySelector("#progress-bar"),
  phaseLabel: document.querySelector("#phase-label"),
  phaseTitle: document.querySelector("#phase-title"),
  stepCount: document.querySelector("#step-count"),
  hostState: document.querySelector("#host-state"),
  gridState: document.querySelector("#grid-state"),
  blockState: document.querySelector("#block-state"),
  matrixX: document.querySelector("#matrix-x"),
  matrixY: document.querySelector("#matrix-y"),
  gamma: document.querySelector("#vector-gamma"),
  beta: document.querySelector("#vector-beta"),
  lanes: document.querySelector("#thread-lanes"),
  threadCaption: document.querySelector("#thread-caption"),
  smem: document.querySelector("#smem-grid"),
  statList: document.querySelector("#stat-list"),
  calculation: document.querySelector("#calculation"),
  detail: document.querySelector("#phase-detail"),
  sourceLocation: document.querySelector("#source-location"),
  sourceCode: document.querySelector("#source-code"),
  sourceExplanation: document.querySelector("#source-explanation"),
  sourceLink: document.querySelector("#source-link"),
  reset: document.querySelector("#reset-button"),
  prev: document.querySelector("#prev-button"),
  play: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  next: document.querySelector("#next-button"),
  speed: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
  statFunction: document.querySelector("#stat-function"),
  statUnit: document.querySelector("#stat-unit"),
  statAction: document.querySelector("#stat-action"),
  statFormula: document.querySelector("#stat-formula"),
  modeTabs: Array.from(document.querySelectorAll(".mode-tab")),
  comparisonCards: Array.from(document.querySelectorAll("[data-jump-mode]")),
  conceptItems: Array.from(document.querySelectorAll(".concept-panel li")),
  pipeNodes: Array.from(document.querySelectorAll(".pipe-node")),
  hostArrow: document.querySelector("#host-arrow"),
  gridArrow: document.querySelector("#grid-arrow"),
};

let currentMode = "launcher";
let currentStep = 0;
let playing = false;
let timer = null;

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function shouldShowOutput(step, rowIndex) {
  if (!step.output) return false;
  if (step.showAllOutput) return true;
  return rowIndex === activeRow;
}

function renderCells(container, matrix, step, outputMode) {
  const activeCols = new Set(step.activeCols || []);
  const activeRows = new Set(step.activeRow || [activeRow]);
  container.innerHTML = "";

  matrix.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const cell = document.createElement("div");
      cell.className = "cell";

      if (activeRows.has(rowIndex)) cell.classList.add("is-row");
      if (activeRows.has(rowIndex) && activeCols.has(colIndex)) {
        cell.classList.add(outputMode ? "is-active-y" : "is-active-x");
      }

      if (outputMode && shouldShowOutput(step, rowIndex)) {
        cell.textContent = fmt(value);
        cell.classList.add("is-written");
      } else if (outputMode) {
        cell.textContent = ".";
        cell.classList.add("is-muted");
      } else {
        cell.textContent = fmt(value);
      }

      container.appendChild(cell);
    });
  });
}

function renderVectors(step) {
  const activeGamma = new Set(step.activeGamma || []);
  const activeBeta = new Set(step.activeBeta || []);
  dom.gamma.innerHTML = "";
  dom.beta.innerHTML = "";

  gamma.forEach((value, index) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (activeGamma.has(index)) cell.classList.add("is-active-g");
    cell.textContent = fmt(value);
    dom.gamma.appendChild(cell);
  });

  beta.forEach((value, index) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (step.betaMode === "off") {
      cell.classList.add("is-muted");
      cell.textContent = "off";
    } else {
      if (activeBeta.has(index)) cell.classList.add("is-active-b");
      cell.textContent = fmt(value);
    }
    dom.beta.appendChild(cell);
  });
}

function laneText(step, index) {
  const x = demoRow[index];
  const vecId = Math.floor(index / 4);
  const component = ["x", "y", "z", "w"][index % 4];
  if (step.thread === "idle") return "等待 wrapper 选择路径";
  if (step.thread === "row") return `tid ${index} 读取 col ${index}`;
  if (step.thread === "square") return `${fmt(x)}² = ${fmt(demoSquares[index])}`;
  if (step.thread === "sum") return `local_sum += ${fmt(x)}`;
  if (step.thread === "variance") return `(x-mean)² = ${fmt(demoCenteredSquares[index])}`;
  if (step.thread === "reduce") return index === 0 ? "smem[0] 汇总" : "等待 stride 合并";
  if (step.thread === "writeRms") return `Y = ${fmt(rmsY[activeRow][index])}`;
  if (step.thread === "writeLayer") return `Y = ${fmt(layerY[activeRow][index])}`;
  if (step.thread === "shuffle") return `lane ${index} 用 shuffle 交换`;
  if (step.thread === "warpWrite") return index === 0 ? "lane0 写 smem[warp_id]" : "warp 内先规约";
  if (step.thread === "warpFinal") return index === 0 ? "thread0 写 smem[0]" : "warp0 汇总";
  if (step.thread === "fallback") return index < 4 ? "float4 快路径" : "fallback 到 warp";
  if (step.thread === "vec") return `vec${vecId}.${component}`;
  if (step.thread === "vecSquare") return `vec${vecId}.${component}²`;
  if (step.thread === "vecSum") return `vec${vecId}.${component} 加到 sum`;
  if (step.thread === "vecVar") return `vec${vecId}.${component} 减 mean`;
  if (step.thread === "writeRmsVec") return `Y4[${vecId}].${component}`;
  if (step.thread === "writeLayerVec") return `Y4[${vecId}].${component}`;
  return `col ${index}`;
}

function renderThreads(step) {
  dom.threadCaption.textContent = currentMode === "vectorized"
    ? "动画把 8 列压成 2 个 float4；真实代码仍然是 256 个线程跨步处理 vec_col"
    : "动画用 8 个 lane 表示真实代码中的 256 个线程";
  dom.lanes.innerHTML = "";

  for (let index = 0; index < cols; index += 1) {
    const lane = document.createElement("div");
    lane.className = "thread-lane";
    if (step.thread !== "idle") lane.classList.add("is-active");
    lane.innerHTML = `<strong>lane ${index}</strong><span>${escapeHtml(laneText(step, index))}</span>`;
    dom.lanes.appendChild(lane);
  }
}

function renderSmem(step) {
  dom.smem.innerHTML = "";
  const active = new Set(step.smemActive || []);
  const values = step.smem || Array.from({ length: cols }, () => "");

  for (let index = 0; index < cols; index += 1) {
    const cell = document.createElement("div");
    cell.className = "smem-cell";
    if (active.has(index)) cell.classList.add("is-active");
    const rawValue = values[index];
    const display = (rawValue || rawValue === 0) ? fmt(rawValue) : "empty";
    cell.innerHTML = `<span>smem[${index}]</span><br><strong>${escapeHtml(display)}</strong>`;
    dom.smem.appendChild(cell);
  }
}

function renderStats(step) {
  dom.statList.innerHTML = "";
  (step.stats || []).forEach(([label, value]) => {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`;
    dom.statList.appendChild(wrapper);
  });
}

function renderSource(step) {
  const source = step.source;
  dom.sourceLocation.textContent = source.location;
  dom.sourceExplanation.textContent = source.explanation;
  dom.sourceLink.href = "../../src/aiop4090/csrc/norm.cu";
  dom.sourceCode.innerHTML = source.lines.map((line) => {
    const activeClass = line.active ? " is-source-active" : "";
    return `<span class="source-line${activeClass}"><i>${line.number}</i><code>${escapeHtml(line.text)}</code></span>`;
  }).join("");
}

function renderPipeline(step) {
  const activeZones = new Set(step.pipe || []);
  dom.pipeNodes.forEach((node) => {
    node.classList.toggle("is-active", activeZones.has(node.dataset.zone));
  });
  dom.hostArrow.classList.toggle("is-active", Boolean(step.hostArrow));
  dom.gridArrow.classList.toggle("is-active", Boolean(step.gridArrow));
}

function renderConcept(step) {
  dom.conceptItems.forEach((item, index) => {
    item.classList.toggle("is-complete", index < step.concept);
    item.classList.toggle("is-current", index === step.concept);
  });
}

function renderModeStats(config) {
  dom.statFunction.textContent = config.stats.function;
  dom.statUnit.textContent = config.stats.unit;
  dom.statAction.textContent = config.stats.action;
  dom.statFormula.textContent = config.stats.formula;
}

function outputMatrix(step) {
  if (step.output === "rms") return rmsY;
  if (step.output === "layer") return layerY;
  return X.map((row) => row.map(() => ""));
}

function update() {
  const config = modeConfig[currentMode];
  const steps = config.steps;
  const step = steps[currentStep];
  const output = outputMatrix(step);

  dom.modeTitle.textContent = config.title;
  dom.modeSummary.textContent = config.summary;
  dom.phaseLabel.textContent = step.label;
  dom.phaseTitle.textContent = step.title;
  dom.stepCount.textContent = `${String(currentStep + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  dom.progressBar.style.width = `${((currentStep + 1) / steps.length) * 100}%`;
  dom.hostState.textContent = step.hostState;
  dom.gridState.textContent = step.gridState;
  dom.blockState.textContent = step.blockState;
  dom.calculation.textContent = step.calculation;
  dom.detail.textContent = step.detail;

  renderModeStats(config);
  renderPipeline(step);
  renderCells(dom.matrixX, X, step, null);
  renderCells(dom.matrixY, output, step, step.output);
  renderVectors(step);
  renderThreads(step);
  renderSmem(step);
  renderStats(step);
  renderSource(step);
  renderConcept(step);

  dom.modeTabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.mode === currentMode);
  });
  dom.playIcon.textContent = playing ? "Ⅱ" : "▶";
  updateSpeedLabel();
}

function stopPlayback() {
  playing = false;
  if (timer) window.clearTimeout(timer);
  timer = null;
}

function playbackInterval() {
  return Number(dom.speed.value);
}

function tick() {
  const steps = modeConfig[currentMode].steps;
  if (currentStep < steps.length - 1) {
    currentStep += 1;
  } else {
    const modes = Object.keys(modeConfig);
    const index = modes.indexOf(currentMode);
    currentMode = modes[(index + 1) % modes.length];
    currentStep = 0;
  }
  update();
  if (playing) timer = window.setTimeout(tick, playbackInterval());
}

function startPlayback() {
  if (playing) return;
  playing = true;
  update();
  timer = window.setTimeout(tick, playbackInterval());
}

function setMode(mode) {
  currentMode = mode;
  currentStep = 0;
  stopPlayback();
  update();
}

function updateSpeedLabel() {
  const speed = 1050 / playbackInterval();
  dom.speedOutput.textContent = `${speed.toFixed(1)}×`;
}

dom.reset.addEventListener("click", () => {
  currentStep = 0;
  stopPlayback();
  update();
});

dom.prev.addEventListener("click", () => {
  currentStep = Math.max(0, currentStep - 1);
  stopPlayback();
  update();
});

dom.next.addEventListener("click", () => {
  const steps = modeConfig[currentMode].steps;
  currentStep = Math.min(steps.length - 1, currentStep + 1);
  stopPlayback();
  update();
});

dom.play.addEventListener("click", () => {
  if (playing) {
    stopPlayback();
    update();
  } else {
    startPlayback();
  }
});

dom.speed.addEventListener("input", () => {
  updateSpeedLabel();
  if (playing) {
    window.clearTimeout(timer);
    timer = window.setTimeout(tick, playbackInterval());
  }
});

dom.modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

dom.comparisonCards.forEach((card) => {
  card.addEventListener("click", () => {
    setMode(card.dataset.jumpMode);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") dom.next.click();
  if (event.key === "ArrowLeft") dom.prev.click();
  if (event.key === " ") {
    event.preventDefault();
    dom.play.click();
  }
});

update();
