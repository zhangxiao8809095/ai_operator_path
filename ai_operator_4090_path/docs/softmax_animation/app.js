const matrixX = [
  [1.0, 2.0, 3.0, 1.0, 0.0, 2.5, -1.0, 4.0],
  [0.5, -0.5, 1.5, 2.0, 0.0, 3.0, 1.0, -1.0],
  [2.0, 2.0, 0.0, -2.0, 1.0, 4.0, 3.0, 0.5],
];

const activeRow = 0;
const lanes = Array.from({ length: 8 }, (_, index) => index);
const rowValues = matrixX[activeRow];
const maxValue = Math.max(...rowValues);
const expValues = rowValues.map((value) => Math.exp(value - maxValue));
const sumValue = expValues.reduce((sum, value) => sum + value, 0);
const probabilities = expValues.map((value) => value / sumValue);
const matrixY = matrixX.map((row, rowIndex) => {
  if (rowIndex !== activeRow) return Array.from({ length: row.length }, () => "");
  return probabilities;
});

function combineOnline(a, b) {
  if (a.sum === 0) return b;
  if (b.sum === 0) return a;
  const max = Math.max(a.max, b.max);
  const sum = a.sum * Math.exp(a.max - max) + b.sum * Math.exp(b.max - max);
  return { max, sum };
}

const onlinePrefixStates = [];
rowValues.reduce((state, value) => {
  const next = combineOnline(state, { max: value, sum: 1 });
  onlinePrefixStates.push(next);
  return next;
}, { max: -Infinity, sum: 0 });

function sourceLine(number, text, active = false) {
  return { number, text, active };
}

function source(location, explanation, lines) {
  return { location, explanation, lines };
}

const sourceCatalog = {
  intuition: source("softmax.cu:142-143, 191-192", "三个版本最终写出的公式都一样：exp(x - max) / sum。区别在于 max 和 sum 是怎么被更快、更稳地算出来。", [
    sourceLine(142, "for (int col = tid; col < cols; col += blockDim.x) {"),
    sourceLine(143, "    Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;", true),
    sourceLine(191, "for (int col = tid; col < cols; col += blockDim.x) {"),
    sourceLine(192, "    Y[row * cols + col] = expf(X[row * cols + col] - state.max_val) / state.sum_val;", true),
  ]),
  wrappers: source("softmax.cu:198-233", "文件现在暴露三个入口：基础版 softmax_row、warp-reduce 版 softmax_warp_reduce、online 版 softmax_online。", [
    sourceLine(198, "torch::Tensor softmax_row(torch::Tensor X) {", true),
    sourceLine(205, "    softmax_row_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(210, "torch::Tensor softmax_warp_reduce(torch::Tensor X) {", true),
    sourceLine(218, "    softmax_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(223, "torch::Tensor softmax_online(torch::Tensor X) {", true),
    sourceLine(231, "    softmax_online_kernel<<<rows, block, 64 * sizeof(float)>>>("),
  ]),
  launch: source("softmax.cu:204-231", "三个入口都保持 rows 个 block、每个 block 256 个线程。online 版只需要 64 个 float 的动态 shared memory，用来放 smem_max 和 smem_sum。", [
    sourceLine(204, "    int block = 256;", true),
    sourceLine(205, "    softmax_row_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(217, "    int block = 256;", true),
    sourceLine(218, "    softmax_warp_reduce_kernel<<<rows, block, block * sizeof(float)>>>("),
    sourceLine(230, "    int block = 256;", true),
    sourceLine(231, "    softmax_online_kernel<<<rows, block, 64 * sizeof(float)>>>("),
  ]),
  map: source("softmax.cu:112-118, 151-157, 179-185", "三个 kernel 的线程定位模型相同：blockIdx.x 选择 row，threadIdx.x 选择 tid，tid 再用 col += blockDim.x 跨步扫列。", [
    sourceLine(112, "int row = blockIdx.x;", true),
    sourceLine(113, "int tid = threadIdx.x;", true),
    sourceLine(117, "for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(151, "int row = blockIdx.x;"),
    sourceLine(156, "for (int col = tid; col < cols; col += blockDim.x) {"),
    sourceLine(179, "int row = blockIdx.x;"),
    sourceLine(184, "for (int col = tid; col < cols; col += blockDim.x) {"),
  ]),
  localMax: source("softmax.cu:116-121", "基础版先让每个线程求 local_max，然后把 256 个 local_max 写进 shared memory。", [
    sourceLine(116, "float local_max = -FLT_MAX;", true),
    sourceLine(117, "for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(118, "    local_max = fmaxf(local_max, X[row * cols + col]);", true),
    sourceLine(120, "smax[tid] = local_max;", true),
    sourceLine(121, "__syncthreads();", true),
  ]),
  treeReduce: source("softmax.cu:123-140", "基础版用 shared memory 做两轮树形归约：先 max，再 sum。每一轮 stride 都从 blockDim.x / 2 一直减半。", [
    sourceLine(123, "for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(124, "    if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);", true),
    sourceLine(127, "float max_val = smax[0];", true),
    sourceLine(136, "for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(137, "    if (tid < stride) ssum[tid] += ssum[tid + stride];", true),
    sourceLine(140, "float sum_val = ssum[0];", true),
  ]),
  rowWrite: source("softmax.cu:129-143", "有了 max_val 后再计算指数和，最后每个线程把自己负责的列写回 Y。", [
    sourceLine(129, "float local_sum = 0.0f;", true),
    sourceLine(131, "    local_sum += expf(X[row * cols + col] - max_val);", true),
    sourceLine(140, "float sum_val = ssum[0];", true),
    sourceLine(143, "    Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;", true),
  ]),
  warpHelpers: source("softmax.cu:8-19", "warp_reduce_sum/max 不走 shared memory 树，而是在一个 warp 内用 __shfl_down_sync 直接让 lane 之间交换数据。", [
    sourceLine(8, "__forceinline__ __device__ float warp_reduce_sum(float val) {", true),
    sourceLine(9, "    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {", true),
    sourceLine(10, "        val += __shfl_down_sync(0xffffffff, val, offset);", true),
    sourceLine(15, "__forceinline__ __device__ float warp_reduce_max(float val) {", true),
    sourceLine(17, "        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));", true),
  ]),
  blockReduce: source("softmax.cu:22-51", "block_reduce_* 分两层：每个 warp 先自己归约，lane 0 把结果放进 smem，最后 warp 0 再把这些 warp 结果合成 block 结果。", [
    sourceLine(23, "int lane = threadIdx.x & (warpSize - 1);", true),
    sourceLine(24, "int warp_id = threadIdx.x / warpSize;", true),
    sourceLine(27, "val = warp_reduce_sum(val);", true),
    sourceLine(28, "if (lane == 0) smem[warp_id] = val;", true),
    sourceLine(31, "val = (threadIdx.x < warp_count) ? smem[lane] : 0.0f;", true),
    sourceLine(32, "if (warp_id == 0) val = warp_reduce_sum(val);", true),
    sourceLine(43, "val = warp_reduce_max(val);", true),
    sourceLine(48, "if (warp_id == 0) val = warp_reduce_max(val);", true),
  ]),
  warpKernel: source("softmax.cu:147-169", "softmax_warp_reduce_kernel 的数学流程和基础版相同，但 max/sum 都交给 block_reduce_max/sum。", [
    sourceLine(147, "__global__ void softmax_warp_reduce_kernel(const float* __restrict__ X,", true),
    sourceLine(155, "float local_max = -FLT_MAX;"),
    sourceLine(159, "float max_val = block_reduce_max(local_max, smem);", true),
    sourceLine(161, "float local_sum = 0.0f;"),
    sourceLine(165, "float sum_val = block_reduce_sum(local_sum, smem);", true),
    sourceLine(168, "    Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;", true),
  ]),
  onlineState: source("softmax.cu:54-66, 183-188", "online 版把一个片段压成 (max_val, sum_val)。遇到新值时，先更新 max，再把旧 sum 按新 max 重新缩放。", [
    sourceLine(54, "struct OnlineSoftmaxState {", true),
    sourceLine(55, "    float max_val;", true),
    sourceLine(56, "    float sum_val;", true),
    sourceLine(63, "float max_val = fmaxf(a.max_val, b.max_val);", true),
    sourceLine(64, "float sum_val = a.sum_val * expf(a.max_val - max_val) +", true),
    sourceLine(65, "                b.sum_val * expf(b.max_val - max_val);", true),
    sourceLine(183, "OnlineSoftmaxState state{-FLT_MAX, 0.0f};", true),
    sourceLine(187, "    state = combine_online(state, item);", true),
  ]),
  onlineReduce: source("softmax.cu:69-102", "online 版的归约对象不再是单个 float，而是一对 (max_val, sum_val)。warp 内和 block 内都用 combine_online 合并状态。", [
    sourceLine(69, "__forceinline__ __device__ OnlineSoftmaxState warp_reduce_online(OnlineSoftmaxState state) {", true),
    sourceLine(72, "    __shfl_down_sync(0xffffffff, state.max_val, offset),", true),
    sourceLine(73, "    __shfl_down_sync(0xffffffff, state.sum_val, offset),", true),
    sourceLine(75, "state = combine_online(state, other);", true),
    sourceLine(87, "state = warp_reduce_online(state);", true),
    sourceLine(89, "smem_max[warp_id] = state.max_val;", true),
    sourceLine(90, "smem_sum[warp_id] = state.sum_val;", true),
  ]),
  onlineKernel: source("softmax.cu:172-193", "softmax_online_kernel 先在线程本地在线合并，再做 block_reduce_online，最后用 state.max_val/state.sum_val 写回。", [
    sourceLine(172, "__global__ void softmax_online_kernel(const float* __restrict__ X,", true),
    sourceLine(176, "float* smem_max = smem;", true),
    sourceLine(177, "float* smem_sum = smem + 32;", true),
    sourceLine(183, "OnlineSoftmaxState state{-FLT_MAX, 0.0f};", true),
    sourceLine(189, "state = block_reduce_online(state, smem_max, smem_sum);", true),
    sourceLine(192, "    Y[row * cols + col] = expf(X[row * cols + col] - state.max_val) / state.sum_val;", true),
  ]),
};

const steps = [
  {
    tab: 0,
    concept: 0,
    label: "直觉",
    title: "三个版本最后都在算同一个 softmax",
    lessonTitle: "先抓住不变的数学结果",
    summary: "softmax_row、softmax_warp_reduce、softmax_online 的输出公式一样，变化的是 max/sum 的归约方式。",
    detail: "输入最大值 4.0 会得到最大概率，但整行概率仍然加起来约等于 1。",
    calculation: "Y[i] = exp(X[i] - max_val) / sum(exp(X[j] - max_val))",
    source: sourceCatalog.intuition,
    zones: ["host"],
    arrows: [],
    laneMode: "idle",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "empty",
    stats: ["三个 softmax", "row-wise", "同一公式", "ΣY[row] ≈ 1"],
  },
  {
    tab: 1,
    concept: 1,
    label: "入口",
    title: "文件现在暴露三个 softmax 入口",
    lessonTitle: "先看 Python 能调用哪些函数",
    summary: "基础版用于理解，warp-reduce 版减少 shared-memory 归约成本，online 版把 max 和 sum 合成状态来归约。",
    detail: "三个 wrapper 都检查 CUDA/contiguous/float32/2D，创建 Y，然后发起对应 kernel。",
    calculation: "softmax_row / softmax_warp_reduce / softmax_online",
    source: sourceCatalog.wrappers,
    zones: ["host"],
    arrows: [],
    laneMode: "idle",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["host wrappers", "CPU host", "检查 + 分发", "X.dim() == 2"],
  },
  {
    tab: 1,
    concept: 1,
    label: "Launch",
    title: "三个版本仍然是一行一个 block",
    lessonTitle: "并行骨架保持稳定",
    summary: "每个 block 处理一行，每个 block 256 个线程。online 版只申请 64 个 float 的 smem 保存两组 warp 结果。",
    detail: "保持相同 grid/block 形状，可以更直接地比较三种归约策略的差异。",
    calculation: "kernel<<<rows, 256, shared_bytes>>>(X, Y, rows, cols)",
    source: sourceCatalog.launch,
    zones: ["host", "grid"],
    arrows: ["host-arrow"],
    laneMode: "idle",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["launch", "rows blocks", "启动 kernel", "block = 256"],
  },
  {
    tab: 2,
    concept: 2,
    label: "线程分工",
    title: "blockIdx.x 选行，threadIdx.x 选列起点",
    lessonTitle: "进入 kernel 后先定位",
    summary: "三个 kernel 都用同一套 row/tid/col 跨步扫描模型，只是后面的归约实现不同。",
    detail: "动画里用 8 个 lane 表示真实代码里的 256 个线程；真实 cols 更大时，每个线程会继续扫 tid + blockDim.x。",
    calculation: "row = blockIdx.x; col = tid, tid + blockDim.x, ...",
    source: sourceCatalog.map,
    zones: ["grid", "block"],
    arrows: ["grid-arrow"],
    laneMode: "map",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["all kernels", "1 block / row", "线程跨步", "tid → col"],
  },
  {
    tab: 2,
    concept: 3,
    label: "基础版",
    title: "softmax_row 先把 local_max 放进 smem",
    lessonTitle: "最直观的写法：先局部，再合并",
    summary: "每个线程看自己负责的列，得到 local_max，然后写入 smax[tid]。",
    detail: "这一步好懂但 shared memory 使用较多，后面的 warp-reduce 版就是在优化这一类归约。",
    calculation: "smax[tid] = local_max",
    source: sourceCatalog.localMax,
    zones: ["block"],
    arrows: [],
    laneMode: "max",
    smemMode: "maxValues",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["softmax_row_kernel", "thread local", "local max", "smax[tid]"],
  },
  {
    tab: 3,
    concept: 4,
    label: "Shared Tree",
    title: "基础版用 shared memory 做两次树形归约",
    lessonTitle: "第一次归约 max，第二次归约 sum",
    summary: "stride 从 128、64、32 一直减半，动画缩成 4、2、1；每轮都要 __syncthreads。",
    detail: "这一版最适合看懂 reduction 的形状：很多局部值最终被合并到 smem[0]。",
    calculation: "smax[0] = reduce_max(...); ssum[0] = reduce_sum(...)",
    source: sourceCatalog.treeReduce,
    zones: ["block"],
    arrows: [],
    laneMode: "reduceMax",
    smemMode: "maxReduced",
    reductionMode: "tree",
    yMode: "reserved",
    stats: ["softmax_row_kernel", "shared tree", "max + sum", "__syncthreads"],
  },
  {
    tab: 3,
    concept: 4,
    label: "基础写回",
    title: "基础版算 exp 和分母，再写回 Y",
    lessonTitle: "一行概率已经能正确落地",
    summary: "有了 max_val，再算 exp(x - max_val) 的总和，最后写出概率。",
    detail: "这一行输出加起来约等于 1。基础版到这里已经完整，只是归约方式还有优化空间。",
    calculation: "Y = exp(x - max_val) / sum_val",
    source: sourceCatalog.rowWrite,
    zones: ["block", "output"],
    arrows: ["block-arrow"],
    laneMode: "write",
    smemMode: "sumReduced",
    reductionMode: "tree",
    yMode: "written",
    stats: ["softmax_row_kernel", "thread write", "正确写回", `sum_val = ${format(sumValue)}`],
  },
  {
    tab: 4,
    concept: 5,
    label: "Warp Shuffle",
    title: "warp_reduce_sum/max 在 warp 内直接交换数据",
    lessonTitle: "第一层优化：少走 shared memory",
    summary: "__shfl_down_sync 让同一个 warp 内的 lane 直接读到彼此的值，先在 warp 内完成归约。",
    detail: "真实 warp 有 32 个 lane。动画仍用 8 个 lane 缩小展示 offset 减半的感觉。",
    calculation: "val += __shfl_down_sync(mask, val, offset)",
    source: sourceCatalog.warpHelpers,
    zones: ["block"],
    arrows: [],
    laneMode: "warpHelper",
    smemMode: "empty",
    reductionMode: "warp",
    yMode: "reserved",
    stats: ["warp_reduce_*", "warp local", "shuffle 归约", "offset >>= 1"],
  },
  {
    tab: 4,
    concept: 6,
    label: "Block Reduce",
    title: "block_reduce_* 把 warp 结果再合成 block 结果",
    lessonTitle: "第二层优化：warp 结果进 smem",
    summary: "每个 warp 的 lane 0 写一个结果到 smem，最后 warp 0 再把这些结果归约到 smem[0]。",
    detail: "这比 256 个线程全部在 shared memory 里树形归约更轻，通常同步和 shared-memory 访问更少。",
    calculation: "warp result → smem[warp_id] → warp 0 → smem[0]",
    source: sourceCatalog.blockReduce,
    zones: ["block"],
    arrows: [],
    laneMode: "warpBlock",
    smemMode: "warpBlockResult",
    reductionMode: "warpBlock",
    yMode: "reserved",
    stats: ["block_reduce_*", "block result", "warp + smem", "smem[warp_id]"],
  },
  {
    tab: 4,
    concept: 6,
    label: "Warp Kernel",
    title: "softmax_warp_reduce_kernel 替换两次归约实现",
    lessonTitle: "数学流程不变，归约实现换芯",
    summary: "仍然先 max 后 sum，仍然最后写回；只是 max/sum 都调用 block_reduce_max/sum。",
    detail: "这是从易到难的第一种优化：保持代码结构熟悉，只换掉归约组件。",
    calculation: "max_val = block_reduce_max(...); sum_val = block_reduce_sum(...)",
    source: sourceCatalog.warpKernel,
    zones: ["block", "output"],
    arrows: ["block-arrow"],
    laneMode: "warpKernel",
    smemMode: "warpBlockResult",
    reductionMode: "warpBlock",
    yMode: "written",
    stats: ["softmax_warp_reduce_kernel", "block reduce", "max + sum", "shuffle + smem"],
  },
  {
    tab: 5,
    concept: 7,
    label: "Online State",
    title: "online 版把片段压成 (max_val, sum_val)",
    lessonTitle: "更难一点：归约对象从 float 变成状态",
    summary: "每个线程边扫边维护状态。新 max 出现时，旧 sum 会按新 max 重新缩放。",
    detail: "这让 max 和指数和可以在一套合并规则里一起推进，是 online softmax 的核心。",
    calculation: "state = combine_online(state, {x, 1})",
    source: sourceCatalog.onlineState,
    zones: ["block"],
    arrows: [],
    laneMode: "onlineState",
    smemMode: "onlinePartial",
    reductionMode: "online",
    yMode: "reserved",
    stats: ["OnlineSoftmaxState", "thread state", "在线合并", "(max_val, sum_val)"],
  },
  {
    tab: 5,
    concept: 7,
    label: "Online Reduce",
    title: "online 版用 combine_online 归约状态",
    lessonTitle: "warp 和 block 都在合并状态",
    summary: "warp_reduce_online 用 shuffle 交换 max/sum，block_reduce_online 再用 smem_max 和 smem_sum 保存 warp 结果。",
    detail: "注意 online 版 shared memory 被分成两段：前 32 个 float 放 max，后 32 个 float 放 sum。",
    calculation: "smem_max[warp_id], smem_sum[warp_id]",
    source: sourceCatalog.onlineReduce,
    zones: ["block"],
    arrows: [],
    laneMode: "onlineCombine",
    smemMode: "onlineReduced",
    reductionMode: "onlineReduce",
    yMode: "reserved",
    stats: ["block_reduce_online", "state reduce", "max/sum 一起合并", "64 floats smem"],
  },
  {
    tab: 6,
    concept: 8,
    label: "Online 写回",
    title: "softmax_online_kernel 用最终 state 写出概率",
    lessonTitle: "最终仍回到熟悉的 softmax 公式",
    summary: "state.max_val 是整行最大值，state.sum_val 是按这个最大值缩放后的指数和。",
    detail: "online 版本难在归约过程，写回阶段反而和基础版非常像。",
    calculation: "Y = exp(x - state.max_val) / state.sum_val",
    source: sourceCatalog.onlineKernel,
    zones: ["block", "output"],
    arrows: ["block-arrow"],
    laneMode: "onlineWrite",
    smemMode: "onlineReduced",
    reductionMode: "onlineReduce",
    yMode: "written",
    stats: ["softmax_online_kernel", "online state", "最终写回", "state.sum_val"],
  },
];

const elements = {
  lessonTitle: document.getElementById("lesson-title"),
  lessonSummary: document.getElementById("lesson-summary"),
  progressBar: document.getElementById("progress-bar"),
  phaseLabel: document.getElementById("phase-label"),
  phaseTitle: document.getElementById("phase-title"),
  stepCount: document.getElementById("step-count"),
  hostState: document.getElementById("host-state"),
  gridState: document.getElementById("grid-state"),
  blockState: document.getElementById("block-state"),
  outputState: document.getElementById("output-state"),
  matrixX: document.getElementById("matrix-x"),
  matrixY: document.getElementById("matrix-y"),
  threadLanes: document.getElementById("thread-lanes"),
  threadCaption: document.getElementById("thread-caption"),
  smemGrid: document.getElementById("smem-grid"),
  reductionTrack: document.getElementById("reduction-track"),
  statList: document.getElementById("stat-list"),
  calculation: document.getElementById("calculation"),
  phaseDetail: document.getElementById("phase-detail"),
  sourceLocation: document.getElementById("source-location"),
  sourceCode: document.getElementById("source-code"),
  sourceExplanation: document.getElementById("source-explanation"),
  resetButton: document.getElementById("reset-button"),
  prevButton: document.getElementById("prev-button"),
  playButton: document.getElementById("play-button"),
  playIcon: document.getElementById("play-icon"),
  nextButton: document.getElementById("next-button"),
  speedInput: document.getElementById("speed-input"),
  speedOutput: document.getElementById("speed-output"),
  statFunction: document.getElementById("stat-function"),
  statUnit: document.getElementById("stat-unit"),
  statAction: document.getElementById("stat-action"),
  statFormula: document.getElementById("stat-formula"),
};

let currentStep = 0;
let isPlaying = false;
let playTimer = null;

function format(value) {
  if (value === "" || value === undefined) return "";
  const rounded = Math.abs(value) < 0.0005 ? 0 : value;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(3);
}

function formatOnlineState(state) {
  return `m=${format(state.max)}, s=${format(state.sum)}`;
}

function classToggle(element, className, condition) {
  element.classList.toggle(className, Boolean(condition));
}

function makeCell(value, row, col, matrix) {
  const cell = document.createElement("div");
  cell.className = "cell";
  cell.dataset.row = row;
  cell.dataset.col = col;
  cell.dataset.matrix = matrix;
  cell.textContent = value === "" ? "·" : format(value);
  return cell;
}

function renderMatrices() {
  elements.matrixX.replaceChildren();
  elements.matrixY.replaceChildren();

  matrixX.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      elements.matrixX.append(makeCell(value, rowIndex, colIndex, "x"));
    });
  });

  matrixY.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      elements.matrixY.append(makeCell(value, rowIndex, colIndex, "y"));
    });
  });
}

function renderThreadLanes() {
  elements.threadLanes.replaceChildren();
  lanes.forEach((lane) => {
    const laneNode = document.createElement("article");
    laneNode.className = "lane";
    laneNode.dataset.lane = lane;

    const top = document.createElement("div");
    top.className = "lane-top";
    const tid = document.createElement("span");
    tid.textContent = `tid ${lane}`;
    const col = document.createElement("span");
    col.textContent = `col ${lane}`;
    top.append(tid, col);

    const value = document.createElement("strong");
    value.dataset.role = "lane-value";
    value.textContent = "等待";

    const note = document.createElement("small");
    note.dataset.role = "lane-note";
    note.textContent = "等待 block 调度";

    laneNode.append(top, value, note);
    elements.threadLanes.append(laneNode);
  });
}

function renderSharedMemory() {
  elements.smemGrid.replaceChildren();
  lanes.forEach((lane) => {
    const cell = document.createElement("div");
    cell.className = "smem-cell";
    cell.dataset.lane = lane;
    cell.textContent = "·";
    elements.smemGrid.append(cell);
  });
}

function reductionItemsFor(mode) {
  if (mode === "tree") {
    return {
      kind: "tree",
      items: [
        ["stride 4", "0↔4, 1↔5, 2↔6, 3↔7"],
        ["stride 2", "0↔2, 1↔3"],
        ["stride 1", "0↔1, result → smem[0]"],
      ],
    };
  }
  if (mode === "warp" || mode === "warpBlock") {
    return {
      kind: "warp",
      items: [
        ["warp 内", "offset 16,8,4,2,1"],
        ["lane 0", "每个 warp 写 smem[warp_id]"],
        ["warp 0", "合并所有 warp 结果"],
      ],
    };
  }
  if (mode === "online" || mode === "onlineReduce") {
    return {
      kind: "online",
      items: [
        ["局部 state", "每个线程得到 (m,s)"],
        ["combine", "新 max 下重缩放 sum"],
        ["block state", "smem_max + smem_sum"],
      ],
    };
  }
  return {
    kind: "none",
    items: [
      ["准备", "等待线程扫描输入"],
      ["归约", "稍后合并局部结果"],
      ["写回", "最后写出 Y"],
    ],
  };
}

function renderReductionTrack(mode = "none") {
  const config = reductionItemsFor(mode);
  elements.reductionTrack.replaceChildren();
  config.items.forEach(([title, description]) => {
    const node = document.createElement("div");
    node.className = "reduce-step";
    if (config.kind !== "none") node.classList.add("is-active");
    if (config.kind === "warp") node.classList.add("is-warp");
    if (config.kind === "online") node.classList.add("is-online");
    const strong = document.createElement("strong");
    strong.textContent = title;
    const span = document.createElement("span");
    span.textContent = description;
    node.append(strong, span);
    elements.reductionTrack.append(node);
  });
}

function renderSource(sourceData) {
  elements.sourceLocation.textContent = sourceData.location;
  elements.sourceExplanation.textContent = sourceData.explanation;
  elements.sourceCode.replaceChildren();

  sourceData.lines.forEach((line) => {
    const lineNode = document.createElement("div");
    lineNode.className = "source-line";
    classToggle(lineNode, "is-active", line.active);

    const number = document.createElement("span");
    number.className = "source-line-number";
    number.textContent = line.number;

    const code = document.createElement("span");
    code.className = "source-line-code";
    code.textContent = line.text;

    lineNode.append(number, code);
    elements.sourceCode.append(lineNode);
  });
}

function updateMatrixClasses(stepData) {
  const xCells = elements.matrixX.querySelectorAll(".cell");
  const yCells = elements.matrixY.querySelectorAll(".cell");
  const maxIndex = rowValues.indexOf(maxValue);
  xCells.forEach((cell) => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    cell.className = "cell";
    if (row === activeRow) cell.classList.add("is-row");
    if (row === activeRow && stepData.laneMode === "max") cell.classList.add("is-hot");
    if (row === activeRow && col === maxIndex && stepData.concept >= 4) cell.classList.add("is-hot");
  });

  yCells.forEach((cell) => {
    const row = Number(cell.dataset.row);
    cell.className = "cell";
    if (row === activeRow && stepData.yMode !== "empty") cell.classList.add("is-output");
    if (row === activeRow && stepData.yMode === "written") cell.classList.add("is-written");
  });
}

function laneTextFor(mode, lane) {
  const x = rowValues[lane];
  const expValue = expValues[lane];
  const probability = probabilities[lane];
  const prefix = onlinePrefixStates[lane];

  if (mode === "map") return [`X[0,${lane}]`, `读取 ${format(x)}`];
  if (mode === "max") return [`local_max=${format(x)}`, "写入 smax[tid]"];
  if (mode === "reduceMax") return [lane === 0 ? `max=${format(maxValue)}` : "参与树形比较", "shared memory tree"];
  if (mode === "write") return [`Y=${format(probability)}`, "写回输出"];
  if (mode === "warpHelper") return [lane === 0 ? "lane 0 收结果" : `lane ${lane}`, "__shfl_down_sync"];
  if (mode === "warpBlock") return [lane === 0 ? "smem[0]" : `warp ${lane}`, "lane 0 写 warp 结果"];
  if (mode === "warpKernel") return [`Y=${format(probability)}`, "block_reduce 后写回"];
  if (mode === "onlineState") return [formatOnlineState(prefix), `读入 x=${format(x)}`];
  if (mode === "onlineCombine") return [lane === 0 ? formatOnlineState({ max: maxValue, sum: sumValue }) : "(m,s)", "combine_online"];
  if (mode === "onlineWrite") return [`Y=${format(probability)}`, "state 写回"];
  return ["等待", "等待 block 调度"];
}

function updateLanes(mode) {
  const laneNodes = elements.threadLanes.querySelectorAll(".lane");
  laneNodes.forEach((node) => {
    const lane = Number(node.dataset.lane);
    const [value, note] = laneTextFor(mode, lane);
    node.className = "lane";
    if (mode !== "idle") node.classList.add("is-active");
    if (mode === "max" || mode === "reduceMax") node.classList.add("is-max");
    if (mode === "write") node.classList.add("is-write");
    if (mode === "warpHelper" || mode === "warpBlock" || mode === "warpKernel") node.classList.add("is-warp");
    if (mode === "onlineState" || mode === "onlineCombine" || mode === "onlineWrite") node.classList.add("is-online");
    node.querySelector('[data-role="lane-value"]').textContent = value;
    node.querySelector('[data-role="lane-note"]').textContent = note;
  });
}

function updateSharedMemory(mode) {
  const cells = elements.smemGrid.querySelectorAll(".smem-cell");
  cells.forEach((cell) => {
    const lane = Number(cell.dataset.lane);
    cell.className = "smem-cell";
    let value = "·";

    if (mode === "maxValues") {
      value = format(rowValues[lane]);
      cell.classList.add("is-filled");
    }

    if (mode === "maxReduced") {
      value = lane === 0 ? format(maxValue) : lane < 4 ? "合并" : "·";
      cell.classList.add(lane === 0 ? "is-final" : "is-filled");
      if (lane > 3) cell.className = "smem-cell";
    }

    if (mode === "sumReduced") {
      value = lane === 0 ? format(sumValue) : lane < 4 ? "合并" : "·";
      cell.classList.add(lane === 0 ? "is-final" : "is-filled");
      if (lane > 3) cell.className = "smem-cell";
    }

    if (mode === "warpBlockResult") {
      value = lane === 0 ? `w0 ${format(maxValue)}` : `w${lane}`;
      cell.classList.add("is-warp");
      if (lane === 0) cell.classList.add("is-final");
    }

    if (mode === "onlinePartial") {
      const state = onlinePrefixStates[lane];
      value = `${format(state.max)}/${format(state.sum)}`;
      cell.classList.add("is-online");
    }

    if (mode === "onlineReduced") {
      value = lane === 0 ? `${format(maxValue)}/${format(sumValue)}` : lane < 4 ? "状态" : "·";
      cell.classList.add(lane === 0 ? "is-final" : "is-online");
      if (lane > 3) cell.className = "smem-cell";
    }

    cell.textContent = value;
  });
}

function updateStats(stepData) {
  const [fn, unit, action, formula] = stepData.stats;
  elements.statFunction.textContent = fn;
  elements.statUnit.textContent = unit;
  elements.statAction.textContent = action;
  elements.statFormula.textContent = formula;

  const showMax = stepData.concept >= 4 || stepData.smemMode !== "empty";
  const showSum = stepData.smemMode === "sumReduced" || stepData.smemMode === "onlineReduced" ||
    stepData.laneMode === "write" || stepData.laneMode === "warpKernel" || stepData.laneMode === "onlineWrite";

  const stats = [
    ["max_val", showMax ? format(maxValue) : "待计算"],
    ["sum_val", showSum ? format(sumValue) : "待计算"],
    ["active row", `row ${activeRow}`],
  ];

  elements.statList.replaceChildren();
  stats.forEach(([term, value]) => {
    const wrapper = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value;
    wrapper.append(dt, dd);
    elements.statList.append(wrapper);
  });
}

function updatePipeline(stepData) {
  document.querySelectorAll(".pipe-node").forEach((node) => {
    classToggle(node, "is-active", stepData.zones.includes(node.dataset.zone));
  });

  ["host-arrow", "grid-arrow", "block-arrow"].forEach((id) => {
    const arrow = document.getElementById(id);
    classToggle(arrow, "is-active", stepData.arrows.includes(id));
  });

  elements.hostState.textContent = currentStep <= 1 ? "检查输入" : "发起 kernel";
  elements.gridState.textContent = currentStep >= 2 ? "rows 个 block" : "等待启动";
  elements.blockState.textContent = currentStep >= 3 ? "row 0 / 256 threads" : "等待调度";
  elements.outputState.textContent = stepData.yMode === "written" ? "Y[row, col]" : "等待写回";
}

function updateTabs(stepData) {
  document.querySelectorAll(".mode-tab").forEach((tab, index) => {
    classToggle(tab, "is-active", index === stepData.tab);
  });
}

function updateConcepts(stepData) {
  document.querySelectorAll("#concept-list li").forEach((item, index) => {
    classToggle(item, "is-current", index === stepData.concept);
  });
}

function updateSpeedLabel() {
  const value = Number(elements.speedInput.value);
  const ratio = 1050 / value;
  elements.speedOutput.textContent = `${ratio.toFixed(1)}×`;
}

function showStep(index) {
  currentStep = (index + steps.length) % steps.length;
  const stepData = steps[currentStep];

  elements.lessonTitle.textContent = stepData.lessonTitle;
  elements.lessonSummary.textContent = stepData.summary;
  elements.progressBar.style.width = `${((currentStep + 1) / steps.length) * 100}%`;
  elements.phaseLabel.textContent = stepData.label;
  elements.phaseTitle.textContent = stepData.title;
  elements.stepCount.textContent = `${String(currentStep + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  elements.threadCaption.textContent = currentStep >= 3
    ? "动画用 8 个 lane 表示真实代码中的 256 个线程；warp 概念按比例缩小展示"
    : "线程在 kernel 启动后才开始执行";
  elements.calculation.textContent = stepData.calculation;
  elements.phaseDetail.textContent = stepData.detail;

  updateTabs(stepData);
  updateConcepts(stepData);
  updatePipeline(stepData);
  updateMatrixClasses(stepData);
  updateLanes(stepData.laneMode);
  updateSharedMemory(stepData.smemMode);
  renderReductionTrack(stepData.reductionMode);
  updateStats(stepData);
  renderSource(stepData.source);
}

function nextStep() {
  showStep(currentStep + 1);
}

function previousStep() {
  showStep(currentStep - 1);
}

function stopPlayback() {
  isPlaying = false;
  elements.playIcon.textContent = "▶";
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

function startPlayback() {
  isPlaying = true;
  elements.playIcon.textContent = "Ⅱ";
  playTimer = setInterval(nextStep, Number(elements.speedInput.value));
}

function togglePlayback() {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function resetAnimation() {
  stopPlayback();
  showStep(0);
}

function bindEvents() {
  elements.resetButton.addEventListener("click", resetAnimation);
  elements.prevButton.addEventListener("click", () => {
    stopPlayback();
    previousStep();
  });
  elements.nextButton.addEventListener("click", () => {
    stopPlayback();
    nextStep();
  });
  elements.playButton.addEventListener("click", togglePlayback);
  elements.speedInput.addEventListener("input", () => {
    updateSpeedLabel();
    if (isPlaying) {
      stopPlayback();
      startPlayback();
    }
  });

  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      stopPlayback();
      showStep(Number(tab.dataset.jump));
    });
  });
}

renderMatrices();
renderThreadLanes();
renderSharedMemory();
bindEvents();
updateSpeedLabel();
showStep(0);
