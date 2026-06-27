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
  const { rowMean, invStd } = layerStats(row);
  return row.map((value, col) => (value - rowMean) * invStd * gamma[col] + beta[col]);
}

function rmsOut(row) {
  const { invRms } = rmsStats(row);
  return row.map((value, col) => value * invRms * gamma[col]);
}

const demoRow = X[activeRow];
const demoLayer = layerStats(demoRow);
const demoRms = rmsStats(demoRow);
const layerY = X.map(layerOut);
const rmsY = X.map(rmsOut);
const demoSquares = demoRow.map((value) => value * value);
const demoCenteredSquares = demoRow.map((value) => (value - demoLayer.rowMean) ** 2);

function sourceLine(number, text, active = false) {
  return { number, text, active };
}

function src(location, explanation, lines) {
  return { location, explanation, lines };
}

const sourceCatalog = {
  launcherChecks: src("norm.cu:75-83, 92-98", "wrapper 是 CPU 侧入口：先确认 tensor 位于 CUDA、连续、float32，再检查 shape 是否符合 row-wise norm 的假设。", [
    sourceLine(75, "torch::Tensor layernorm_row(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, double eps) {", true),
    sourceLine(76, "    CHECK_INPUT(X);", true),
    sourceLine(77, "    CHECK_INPUT(gamma);", true),
    sourceLine(78, "    CHECK_INPUT(beta);", true),
    sourceLine(79, "    TORCH_CHECK(X.dim() == 2, \"X must be 2D [rows, cols]\");", true),
    sourceLine(80, "    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, \"gamma/beta must be 1D\");"),
    sourceLine(81, "    TORCH_CHECK(gamma.size(0) == X.size(1) && beta.size(0) == X.size(1), \"gamma/beta size mismatch\");"),
    sourceLine(92, "torch::Tensor rmsnorm_row(torch::Tensor X, torch::Tensor gamma, double eps) {"),
    sourceLine(93, "    CHECK_INPUT(X);"),
    sourceLine(94, "    CHECK_INPUT(gamma);"),
  ]),
  launcherShape: src("norm.cu:82-88", "rows 决定启动多少个 block，cols 决定每个 block 处理的一行有多长。", [
    sourceLine(82, "    int rows = static_cast<int>(X.size(0));", true),
    sourceLine(83, "    int cols = static_cast<int>(X.size(1));", true),
    sourceLine(84, "    auto Y = torch::empty_like(X);", true),
    sourceLine(85, "    int block = 256;", true),
    sourceLine(86, "    layernorm_row_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(87, "        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),"),
    sourceLine(88, "        Y.data_ptr<float>(), rows, cols, static_cast<float>(eps));"),
  ]),
  launcherRms: src("norm.cu:97-104", "RMSNorm 的 wrapper 少一个 beta，但 launch 结构一样：rows 个 block，每个 block 256 个线程，动态 shared memory 放 256 个 float。", [
    sourceLine(97, "    int rows = static_cast<int>(X.size(0));", true),
    sourceLine(98, "    int cols = static_cast<int>(X.size(1));", true),
    sourceLine(99, "    auto Y = torch::empty_like(X);", true),
    sourceLine(100, "    int block = 256;", true),
    sourceLine(101, "    rmsnorm_row_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(102, "        X.data_ptr<float>(), gamma.data_ptr<float>(), Y.data_ptr<float>(),"),
    sourceLine(103, "        rows, cols, static_cast<float>(eps));"),
    sourceLine(104, "    return Y;"),
  ]),
  rmsMap: src("norm.cu:51-58", "一个 block 负责一行；动画中的 8 个 lane 代表真实代码里的 256 个线程。", [
    sourceLine(51, "    extern __shared__ float smem[];", true),
    sourceLine(52, "    int row = blockIdx.x;", true),
    sourceLine(53, "    int tid = threadIdx.x;", true),
    sourceLine(55, "    float local_sum_sq = 0.0f;"),
    sourceLine(56, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(57, "        float v = X[row * cols + col];", true),
    sourceLine(58, "        local_sum_sq += v * v;", true),
  ]),
  rmsReduce: src("norm.cu:60-66", "每个线程先把自己的局部平方和写入 smem，然后通过 stride 逐半缩小的 reduction 合成总平方和。", [
    sourceLine(60, "    smem[tid] = local_sum_sq;", true),
    sourceLine(61, "    __syncthreads();", true),
    sourceLine(62, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(63, "        if (tid < stride) smem[tid] += smem[tid + stride];", true),
    sourceLine(64, "        __syncthreads();", true),
    sourceLine(66, "    float inv_rms = rsqrtf(smem[0] / cols + eps);"),
  ]),
  rmsScale: src("norm.cu:66-70", "RMSNorm 不减 mean；它直接用 inv_rms 缩放原值，再乘 gamma[col]。", [
    sourceLine(66, "    float inv_rms = rsqrtf(smem[0] / cols + eps);", true),
    sourceLine(68, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(69, "        Y[row * cols + col] = X[row * cols + col] * inv_rms * gamma[col];", true),
    sourceLine(70, "    }"),
  ]),
  layerMap: src("norm.cu:12-19", "LayerNorm 也采用 1 block / row。第一轮遍历先为 mean 准备每个线程的局部和。", [
    sourceLine(12, "    extern __shared__ float smem[];", true),
    sourceLine(13, "    int row = blockIdx.x;", true),
    sourceLine(14, "    int tid = threadIdx.x;", true),
    sourceLine(16, "    float local_sum = 0.0f;"),
    sourceLine(17, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(18, "        local_sum += X[row * cols + col];", true),
    sourceLine(19, "    }"),
  ]),
  layerMean: src("norm.cu:20-26", "mean 来自整行的总和：线程把局部和写到 smem，再在 block 内规约到 smem[0]。", [
    sourceLine(20, "    smem[tid] = local_sum;", true),
    sourceLine(21, "    __syncthreads();", true),
    sourceLine(22, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(23, "        if (tid < stride) smem[tid] += smem[tid + stride];", true),
    sourceLine(24, "        __syncthreads();", true),
    sourceLine(26, "    float mean = smem[0] / cols;", true),
  ]),
  layerVar: src("norm.cu:28-39", "有了 mean 后，第二轮遍历计算 (x - mean)^2，再做第二次 reduction 得到 variance。", [
    sourceLine(28, "    float local_var = 0.0f;", true),
    sourceLine(29, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(30, "        float v = X[row * cols + col] - mean;", true),
    sourceLine(31, "        local_var += v * v;", true),
    sourceLine(33, "    smem[tid] = local_var;"),
    sourceLine(35, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {"),
    sourceLine(39, "    float inv_std = rsqrtf(smem[0] / cols + eps);", true),
  ]),
  layerWrite: src("norm.cu:41-44", "最后每个线程回到自己负责的列，套用 affine：先标准化，再乘 gamma，加 beta。", [
    sourceLine(41, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(42, "        float norm = (X[row * cols + col] - mean) * inv_std;", true),
    sourceLine(43, "        Y[row * cols + col] = norm * gamma[col] + beta[col];", true),
    sourceLine(44, "    }"),
  ]),
};

const common = {
  activeRow: [activeRow],
  activeCols: [0, 1, 2, 3, 4, 5, 6, 7],
};

const launcherSteps = [
  {
    label: "最外层入口",
    title: "Python 调用进入 C++ wrapper",
    calculation: "ops.layernorm_row(x, gamma, beta) → _C.layernorm_row(...)",
    detail: "先不要急着看 GPU。norm.cu 的公开函数是 CPU 侧包装器，它负责把 Python tensor 变成 kernel 参数。",
    hostState: "接收 tensor",
    gridState: "尚未 launch",
    blockState: "等待配置",
    pipe: ["host"],
    concept: 0,
    source: sourceCatalog.launcherChecks,
    stats: [
      ["输入", "X, gamma, beta"],
      ["位置", "CPU host code"],
      ["职责", "参数检查"],
      ["下一步", "提取 rows/cols"],
    ],
    thread: "idle",
    smem: null,
  },
  {
    label: "检查形状",
    title: "X 是二维，gamma/beta 与 cols 对齐",
    calculation: "X[rows, cols]，gamma[cols]，beta[cols]",
    detail: "row-wise norm 的意思是：每一行独立归一化，gamma 和 beta 按列广播到每一行。",
    hostState: "检查 shape",
    gridState: "rows 未读取",
    blockState: "block 未配置",
    pipe: ["host"],
    concept: 0,
    source: sourceCatalog.launcherChecks,
    stats: [
      ["X", `${X.length} × ${cols}`],
      ["gamma", `${cols} 个 scale`],
      ["beta", `${cols} 个 shift`],
      ["dtype", "float32 CUDA"],
    ],
    thread: "idle",
    smem: null,
  },
  {
    label: "分配输出",
    title: "rows/cols 决定后，先创建 Y",
    calculation: "auto Y = torch::empty_like(X)",
    detail: "输出矩阵的形状和 X 完全一样。kernel 只负责把每个位置的值算出来并写进去。",
    hostState: "创建 Y",
    gridState: `${X.length} rows`,
    blockState: "准备 256 threads",
    pipe: ["host"],
    concept: 0,
    source: sourceCatalog.launcherShape,
    stats: [
      ["rows", `${X.length}`],
      ["cols", `${cols}`],
      ["Y", "empty_like(X)"],
      ["内存", "CUDA tensor"],
    ],
    thread: "idle",
    smem: null,
  },
  {
    label: "配置 kernel",
    title: "一行数据交给一个 CUDA block",
    calculation: "layernorm_row_kernel<<<rows, 256, 256*sizeof(float)>>>(...)",
    detail: "真实代码中 block=256。动画用 8 个 lane 缩小展示，但并行思想一样：线程分工扫同一行的 cols。",
    hostState: "发起 launch",
    gridState: "rows 个 block",
    blockState: "256 threads",
    pipe: ["host", "grid", "block"],
    hostArrow: true,
    gridArrow: true,
    concept: 1,
    source: sourceCatalog.launcherShape,
    stats: [
      ["grid", `${X.length} blocks`],
      ["block", "256 threads"],
      ["shared", "256 floats"],
      ["映射", "1 block / row"],
    ],
    thread: "row",
    smem: Array.from({ length: cols }, (_, index) => `lane ${index}`),
  },
  {
    label: "RMSNorm 启动",
    title: "RMSNorm wrapper 更短：少一个 beta",
    calculation: "rmsnorm_row_kernel<<<rows, block, block*sizeof(float)>>>(...)",
    detail: "RMSNorm 的 host 侧和 LayerNorm 几乎一样，只是参数少了 beta，GPU 内部也少一次 mean/variance 流程。",
    hostState: "启动 RMSNorm",
    gridState: "rows 个 block",
    blockState: "256 threads",
    pipe: ["host", "grid", "block"],
    hostArrow: true,
    gridArrow: true,
    concept: 1,
    source: sourceCatalog.launcherRms,
    stats: [
      ["函数", "rmsnorm_row"],
      ["参数", "X, gamma, eps"],
      ["少了", "beta"],
      ["kernel", "rmsnorm_row_kernel"],
    ],
    thread: "row",
    smem: Array.from({ length: cols }, (_, index) => `tid ${index}`),
  },
];

const rmsSteps = [
  {
    label: "行到 block",
    title: "blockIdx.x 选择当前行",
    calculation: "row = blockIdx.x，tid = threadIdx.x",
    detail: "每个 block 只管一行。不同 block 之间互不通信，所以每行的 RMSNorm 可以独立并行完成。",
    hostState: "已 launch",
    gridState: `block ${activeRow} → row ${activeRow}`,
    blockState: "8 lanes 展示",
    pipe: ["grid", "block"],
    gridArrow: true,
    concept: 0,
    source: sourceCatalog.rmsMap,
    stats: [
      ["row", `${activeRow}`],
      ["cols", `${cols}`],
      ["真实线程", "256"],
      ["动画 lane", "8"],
    ],
    thread: "row",
    smem: null,
  },
  {
    label: "平方局部和",
    title: "每个线程读取自己负责的列",
    calculation: "local_sum_sq += X[row, col] * X[row, col]",
    detail: "RMSNorm 不需要先减均值，所以第一件事就是统计这一行所有元素的平方和。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "threads read X",
    pipe: ["block"],
    concept: 1,
    source: sourceCatalog.rmsMap,
    stats: [
      ["X² 和", fmt(sum(demoSquares))],
      ["mean(X²)", fmt(demoRms.rowMeanSq)],
      ["eps", "1e-6"],
      ["下一步", "写入 smem"],
    ],
    thread: "square",
    smem: demoSquares,
    smemActive: common.activeCols,
    activeCols: common.activeCols,
  },
  {
    label: "写 shared",
    title: "局部平方和进入 smem[tid]",
    calculation: "smem[tid] = local_sum_sq；__syncthreads()",
    detail: "shared memory 属于同一个 block。先写进去，再同步，才能安全地让线程互相读取结果。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "smem ready",
    pipe: ["block"],
    concept: 2,
    source: sourceCatalog.rmsReduce,
    stats: [
      ["smem[0..7]", "local X²"],
      ["同步", "__syncthreads"],
      ["作用域", "同一 block"],
      ["下一步", "reduction"],
    ],
    thread: "smem",
    smem: demoSquares,
    smemActive: common.activeCols,
    activeCols: common.activeCols,
  },
  {
    label: "block reduction",
    title: "多个线程把 smem 合成一个总和",
    calculation: "for (stride >>= 1) smem[tid] += smem[tid + stride]",
    detail: "真实代码从 stride=128 开始逐半缩小。动画直接展示最终结果：整行平方和落在 smem[0]。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "reduce in smem",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.rmsReduce,
    stats: [
      ["sum(X²)", fmt(sum(demoSquares))],
      ["mean(X²)", fmt(demoRms.rowMeanSq)],
      ["smem[0]", "总平方和"],
      ["下一步", "rsqrt"],
    ],
    thread: "reduce",
    smem: [sum(demoSquares), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
  },
  {
    label: "得到 inv_rms",
    title: "rsqrtf 给出缩放系数",
    calculation: "inv_rms = rsqrtf(mean(X²) + eps)",
    detail: "rsqrtf(x) 就是 1 / sqrt(x)。后面每个元素都乘同一个 inv_rms，但 gamma[col] 仍然按列不同。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "compute inv_rms",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.rmsScale,
    stats: [
      ["mean(X²)", fmt(demoRms.rowMeanSq)],
      ["eps", "1e-6"],
      ["inv_rms", fmt(demoRms.invRms)],
      ["公式", "x * inv_rms"],
    ],
    thread: "inv",
    smem: [fmt(demoRms.invRms), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
  },
  {
    label: "写回 Y",
    title: "每个线程写回自己负责的输出列",
    calculation: "Y[row, col] = X[row, col] * inv_rms * gamma[col]",
    detail: "这一步没有 beta，也没有减 mean。RMSNorm 的轻量感就在这里：一次统计，一次缩放。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "write Y",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.rmsScale,
    stats: [
      ["输出", "Y[row, col]"],
      ["scale", "gamma[col]"],
      ["shift", "无 beta"],
      ["复杂度", "1 次 reduction"],
    ],
    thread: "writeRms",
    smem: [fmt(demoRms.invRms), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
    activeGamma: common.activeCols,
    output: "rms",
  },
  {
    label: "RMSNorm 总结",
    title: "RMSNorm = 统计均方根 + 按列缩放",
    calculation: "y = x / sqrt(mean(x²) + eps) * gamma",
    detail: "它比 LayerNorm 少了 mean 和 beta，所以常见于追求更轻量的 Transformer 结构。",
    hostState: "完成",
    gridState: "all rows done",
    blockState: "Y ready",
    pipe: ["grid", "block"],
    concept: 4,
    source: sourceCatalog.rmsScale,
    stats: [
      ["函数", "rmsnorm_row_kernel"],
      ["统计量", "mean(x²)"],
      ["reduction", "1 次"],
      ["输出", "x * inv_rms * gamma"],
    ],
    thread: "writeRms",
    smem: [fmt(demoRms.invRms), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
    activeGamma: common.activeCols,
    output: "rms",
    showAllOutput: true,
  },
];

const layerSteps = [
  {
    label: "行到 block",
    title: "LayerNorm 也从 1 block / row 开始",
    calculation: "row = blockIdx.x，tid = threadIdx.x",
    detail: "并行映射和 RMSNorm 一样。难点不是线程映射，而是多了 mean 和 variance 两轮统计。",
    hostState: "已 launch",
    gridState: `block ${activeRow} → row ${activeRow}`,
    blockState: "8 lanes 展示",
    pipe: ["grid", "block"],
    gridArrow: true,
    concept: 0,
    source: sourceCatalog.layerMap,
    stats: [
      ["row", `${activeRow}`],
      ["cols", `${cols}`],
      ["目标", "mean/var"],
      ["reduction", "2 次"],
    ],
    thread: "row",
    smem: null,
  },
  {
    label: "求局部和",
    title: "第一遍扫 X，准备求 mean",
    calculation: "local_sum += X[row, col]",
    detail: "每个线程负责一部分列。真实 cols 比 256 大时，同一个线程会 col += blockDim.x 继续处理后面的列。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "threads read X",
    pipe: ["block"],
    concept: 1,
    source: sourceCatalog.layerMap,
    stats: [
      ["sum(X)", fmt(sum(demoRow))],
      ["cols", `${cols}`],
      ["mean", "待归约"],
      ["下一步", "smem"],
    ],
    thread: "sum",
    smem: demoRow,
    smemActive: common.activeCols,
    activeCols: common.activeCols,
  },
  {
    label: "规约 mean",
    title: "smem[0] 得到整行总和",
    calculation: "mean = smem[0] / cols",
    detail: "和 RMSNorm 的 reduction 结构一样，只是这里统计的是原值总和，而不是平方和。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "reduce sum",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.layerMean,
    stats: [
      ["sum(X)", fmt(sum(demoRow))],
      ["mean", fmt(demoLayer.rowMean)],
      ["smem[0]", "总和"],
      ["下一步", "variance"],
    ],
    thread: "reduce",
    smem: [sum(demoRow), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
  },
  {
    label: "求方差",
    title: "第二遍扫 X，统计 (x - mean)²",
    calculation: "local_var += (X[row, col] - mean)²",
    detail: "方差必须先知道 mean，所以 LayerNorm 需要第二轮遍历和第二次 block reduction。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "threads read X again",
    pipe: ["block"],
    concept: 1,
    source: sourceCatalog.layerVar,
    stats: [
      ["mean", fmt(demoLayer.rowMean)],
      ["sum((x-mean)²)", fmt(sum(demoCenteredSquares))],
      ["variance", "待归约"],
      ["下一步", "rsqrt"],
    ],
    thread: "variance",
    smem: demoCenteredSquares,
    smemActive: common.activeCols,
    activeCols: common.activeCols,
  },
  {
    label: "规约 variance",
    title: "第二次 reduction 得到 inv_std",
    calculation: "inv_std = rsqrtf(variance + eps)",
    detail: "variance 是均方偏差。加 eps 是为了避免极小方差导致除零或数值不稳定。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "reduce variance",
    pipe: ["block"],
    concept: 3,
    source: sourceCatalog.layerVar,
    stats: [
      ["variance", fmt(demoLayer.rowVar)],
      ["eps", "1e-5"],
      ["inv_std", fmt(demoLayer.invStd)],
      ["下一步", "affine"],
    ],
    thread: "reduce",
    smem: [sum(demoCenteredSquares), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
  },
  {
    label: "标准化",
    title: "先把每个元素变成均值 0、方差 1 附近",
    calculation: "norm = (X[row, col] - mean) * inv_std",
    detail: "这一层还没有 gamma/beta，只是把这一行重新拉到统一尺度。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "normalize",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.layerWrite,
    stats: [
      ["mean", fmt(demoLayer.rowMean)],
      ["inv_std", fmt(demoLayer.invStd)],
      ["norm", "(x - mean) * inv_std"],
      ["下一步", "gamma/beta"],
    ],
    thread: "normalize",
    smem: [fmt(demoLayer.invStd), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
  },
  {
    label: "Affine 写回",
    title: "gamma 缩放，beta 平移，写入 Y",
    calculation: "Y[row, col] = norm * gamma[col] + beta[col]",
    detail: "gamma 和 beta 是按列使用的参数。每一行都会复用同一组 gamma/beta。",
    hostState: "kernel running",
    gridState: "row blocks",
    blockState: "write Y",
    pipe: ["block"],
    concept: 4,
    source: sourceCatalog.layerWrite,
    stats: [
      ["输出", "Y[row, col]"],
      ["scale", "gamma[col]"],
      ["shift", "beta[col]"],
      ["复杂度", "2 次 reduction"],
    ],
    thread: "writeLayer",
    smem: [fmt(demoLayer.invStd), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
    activeGamma: common.activeCols,
    activeBeta: common.activeCols,
    output: "layer",
  },
  {
    label: "LayerNorm 总结",
    title: "LayerNorm = mean + variance + affine",
    calculation: "y = (x - mean) / sqrt(var + eps) * gamma + beta",
    detail: "它比 RMSNorm 多减均值、多加 beta，也多一次统计方差的成本。",
    hostState: "完成",
    gridState: "all rows done",
    blockState: "Y ready",
    pipe: ["grid", "block"],
    concept: 4,
    source: sourceCatalog.layerWrite,
    stats: [
      ["函数", "layernorm_row_kernel"],
      ["统计量", "mean, variance"],
      ["reduction", "2 次"],
      ["输出", "norm * gamma + beta"],
    ],
    thread: "writeLayer",
    smem: [fmt(demoLayer.invStd), "", "", "", "", "", "", ""],
    smemActive: [0],
    activeCols: common.activeCols,
    activeGamma: common.activeCols,
    activeBeta: common.activeCols,
    output: "layer",
    showAllOutput: true,
  },
];

const modeConfig = {
  launcher: {
    title: "Host Launcher：检查输入并启动 kernel",
    summary: "先从 C++ 包装函数开始：检查 tensor、分配输出，然后用 rows 个 block 启动 GPU 代码。",
    stats: {
      function: "layernorm_row / rmsnorm_row",
      unit: "CPU host wrapper",
      action: "检查 + launch",
      formula: "<<<rows, 256, smem>>>",
    },
    steps: launcherSteps,
  },
  rmsnorm: {
    title: "RMSNorm：一次 reduction 得到缩放系数",
    summary: "一个 CUDA block 处理一行，只统计 mean(x²)，再把每列乘上 gamma[col]。",
    stats: {
      function: "rmsnorm_row_kernel",
      unit: "1 block / row",
      action: "一次 reduction",
      formula: "x * inv_rms * gamma",
    },
    steps: rmsSteps,
  },
  layernorm: {
    title: "LayerNorm：mean + variance + affine",
    summary: "一个 CUDA block 处理一行，block 内线程先求 mean，再求 variance，最后写回归一化结果。",
    stats: {
      function: "layernorm_row_kernel",
      unit: "1 block / row",
      action: "两次 reduction",
      formula: "norm * gamma + beta",
    },
    steps: layerSteps,
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
  modeTabs: [...document.querySelectorAll(".mode-tab")],
  comparisonCards: [...document.querySelectorAll("[data-jump-mode]")],
  conceptItems: [...document.querySelectorAll(".concept-panel li")],
  pipeNodes: [...document.querySelectorAll(".pipe-node")],
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

function renderCells(container, matrix, step, outputMode = null) {
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
        if (activeRows.has(rowIndex) || step.showAllOutput) {
          cell.classList.add("is-written");
        }
      } else if (outputMode) {
        cell.textContent = "·";
        cell.classList.add("is-muted");
      } else {
        cell.textContent = fmt(value);
      }

      container.appendChild(cell);
    });
  });
}

function shouldShowOutput(step, rowIndex) {
  if (!step.output) return false;
  if (step.showAllOutput) return true;
  return rowIndex === activeRow;
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
    if (currentMode === "rmsnorm") {
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
  if (step.thread === "idle") return "等待 CPU launch";
  if (step.thread === "row") return `block ${activeRow} 处理 row ${activeRow}`;
  if (step.thread === "sum") return `X[${activeRow},${index}] = ${fmt(x)}`;
  if (step.thread === "square") return `${fmt(x)}² = ${fmt(demoSquares[index])}`;
  if (step.thread === "smem") return `smem[${index}] ← ${fmt(step.smem && step.smem[index])}`;
  if (step.thread === "reduce") return index === 0 ? "smem[0] 收集总和" : "等待 stride 合并";
  if (step.thread === "inv") return index === 0 ? "计算 inv_rms" : "读取同一缩放系数";
  if (step.thread === "variance") return `(x-mean)² = ${fmt(demoCenteredSquares[index])}`;
  if (step.thread === "normalize") return `norm col ${index}`;
  if (step.thread === "writeRms") return `Y[${activeRow},${index}] = ${fmt(rmsY[activeRow][index])}`;
  if (step.thread === "writeLayer") return `Y[${activeRow},${index}] = ${fmt(layerY[activeRow][index])}`;
  return `col ${index}`;
}

function renderThreads(step) {
  dom.threadCaption.textContent = currentMode === "launcher"
    ? "动画用 8 个 lane 预演真实代码中的 256 个线程"
    : "动画用 8 个 lane 表示真实代码中的 256 个线程";
  dom.lanes.innerHTML = "";

  for (let index = 0; index < cols; index += 1) {
    const lane = document.createElement("div");
    lane.className = "thread-lane";
    if (step.thread !== "idle") lane.classList.add("is-active");
    lane.innerHTML = `<strong>tid ${index}</strong><span>${escapeHtml(laneText(step, index))}</span>`;
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
    const value = values[index];
    cell.innerHTML = `<span>smem[${index}]</span><br><strong>${escapeHtml(fmt(value)) || "empty"}</strong>`;
    if (!value && value !== 0) cell.classList.add("is-muted");
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
  renderCells(dom.matrixX, X, step);
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
