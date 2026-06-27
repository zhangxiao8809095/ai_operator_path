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

function sourceLine(number, text, active = false) {
  return { number, text, active };
}

function source(location, explanation, lines) {
  return { location, explanation, lines };
}

const sourceCatalog = {
  intuition: source("softmax.cu:32-47", "代码里的 softmax 分成三件事：先有 max_val，再有 sum_val，最后写出 exp(x - max_val) / sum_val。", [
    sourceLine(32, "float local_sum = 0.0f;"),
    sourceLine(34, "local_sum += expf(X[row * cols + col] - max_val);", true),
    sourceLine(43, "float sum_val = ssum[0];"),
    sourceLine(46, "Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;", true),
  ]),
  wrapper: source("softmax.cu:52-61", "softmax_row 是 host 侧入口：检查 X，读 rows/cols，创建 Y，然后启动 kernel。", [
    sourceLine(52, "torch::Tensor softmax_row(torch::Tensor X) {", true),
    sourceLine(53, "    CHECK_INPUT(X);", true),
    sourceLine(54, "    TORCH_CHECK(X.dim() == 2, \"X must be 2D [rows, cols]\");", true),
    sourceLine(55, "    int rows = static_cast<int>(X.size(0));"),
    sourceLine(56, "    int cols = static_cast<int>(X.size(1));"),
    sourceLine(57, "    auto Y = torch::empty_like(X);", true),
    sourceLine(61, "    return Y;"),
  ]),
  launch: source("softmax.cu:58-60", "启动参数写成 <<<rows, block, block * sizeof(float)>>>，意思是每一行一个 block，每个 block 有 256 个线程，并申请 256 个 float 的 shared memory。", [
    sourceLine(58, "    int block = 256;", true),
    sourceLine(59, "    softmax_row_kernel<<<rows, block, block * sizeof(float)>>>(" , true),
    sourceLine(60, "        X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);", true),
  ]),
  map: source("softmax.cu:15-22", "kernel 内部用 blockIdx.x 选择 row，用 threadIdx.x 选择线程编号；每个线程按 col += blockDim.x 跨步处理列。", [
    sourceLine(15, "    int row = blockIdx.x;", true),
    sourceLine(16, "    int tid = threadIdx.x;", true),
    sourceLine(17, "    if (row >= rows) return;"),
    sourceLine(20, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(21, "        local_max = fmaxf(local_max, X[row * cols + col]);"),
    sourceLine(22, "    }"),
  ]),
  localMax: source("softmax.cu:19-24", "每个线程先找自己负责列里的最大值，再把局部最大值放进 smem[tid]。", [
    sourceLine(19, "    float local_max = -FLT_MAX;", true),
    sourceLine(20, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(21, "        local_max = fmaxf(local_max, X[row * cols + col]);", true),
    sourceLine(23, "    smax[tid] = local_max;", true),
    sourceLine(24, "    __syncthreads();", true),
  ]),
  reduceMax: source("softmax.cu:26-30", "block 内归约把 256 个局部最大值一路合并到 smax[0]，动画里用 8 个 lane 缩小演示。", [
    sourceLine(26, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(27, "        if (tid < stride) smax[tid] = fmaxf(smax[tid], smax[tid + stride]);", true),
    sourceLine(28, "        __syncthreads();", true),
    sourceLine(30, "    float max_val = smax[0];", true),
  ]),
  localSum: source("softmax.cu:32-37", "有了 max_val 后，每个线程计算 exp(x - max_val) 的局部和。减 max 是为了数值稳定。", [
    sourceLine(32, "    float local_sum = 0.0f;", true),
    sourceLine(33, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(34, "        local_sum += expf(X[row * cols + col] - max_val);", true),
    sourceLine(36, "    ssum[tid] = local_sum;", true),
    sourceLine(37, "    __syncthreads();", true),
  ]),
  reduceSum: source("softmax.cu:39-43", "第二次归约把所有局部和加到 ssum[0]，它就是 softmax 分母。", [
    sourceLine(39, "    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {", true),
    sourceLine(40, "        if (tid < stride) ssum[tid] += ssum[tid + stride];", true),
    sourceLine(41, "        __syncthreads();", true),
    sourceLine(43, "    float sum_val = ssum[0];", true),
  ]),
  write: source("softmax.cu:45-47", "最后每个线程回到自己负责的列，把概率写入 Y[row, col]。", [
    sourceLine(45, "    for (int col = tid; col < cols; col += blockDim.x) {", true),
    sourceLine(46, "        Y[row * cols + col] = expf(X[row * cols + col] - max_val) / sum_val;", true),
    sourceLine(47, "    }", true),
  ]),
};

const steps = [
  {
    tab: 0,
    concept: 0,
    label: "直觉",
    title: "X 的每一行单独做 softmax",
    lessonTitle: "先把一行数变成概率",
    summary: "Softmax 会让每个值变成非负数，并且同一行的输出加起来接近 1。",
    detail: "先从数学直觉看：每一行会被转换成一组概率，行与行之间互不影响。",
    calculation: "softmax(row) = exp(row) / sum(exp(row))",
    source: sourceCatalog.intuition,
    zones: ["host"],
    arrows: [],
    laneMode: "idle",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "empty",
    stats: ["softmax_row", "row-wise", "理解输出", "sum(Y[row]) ≈ 1"],
  },
  {
    tab: 1,
    concept: 1,
    label: "入口函数",
    title: "softmax_row 先做输入检查和输出分配",
    lessonTitle: "最外层函数像调度员",
    summary: "它不直接算 softmax，而是确认 X 合法、创建 Y、准备启动 GPU kernel。",
    detail: "CHECK_INPUT 要求 X 是 CUDA Tensor、连续内存、float32；额外检查 X 必须是二维矩阵。",
    calculation: "Y = torch::empty_like(X)",
    source: sourceCatalog.wrapper,
    zones: ["host"],
    arrows: [],
    laneMode: "idle",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["softmax_row", "CPU host", "检查 + 分配", "X.dim() == 2"],
  },
  {
    tab: 1,
    concept: 1,
    label: "Kernel Launch",
    title: "rows 个 block 被派到 GPU",
    lessonTitle: "一个 block 对应一行",
    summary: "启动配置是 <<<rows, 256, 256*sizeof(float)>>>, 每行交给一个 block。",
    detail: "第三个启动参数给每个 block 申请动态 shared memory，这块内存后面会先当 smax 用，再当 ssum 用。",
    calculation: "softmax_row_kernel<<<rows, 256, 256 * sizeof(float)>>>(...)",
    source: sourceCatalog.launch,
    zones: ["host", "grid"],
    arrows: ["host-arrow"],
    laneMode: "idle",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["softmax_row", "rows blocks", "启动 kernel", "block = 256"],
  },
  {
    tab: 2,
    concept: 2,
    label: "定位",
    title: "blockIdx.x 选择 row，threadIdx.x 选择列起点",
    lessonTitle: "进入 kernel 后先定位",
    summary: "当前 block 负责第 0 行，线程 tid 从自己的列号开始跨步扫描。",
    detail: "动画里 cols=8，所以每个 lane 刚好拿到一个元素；真实代码里 cols 很大时会继续 col += blockDim.x。",
    calculation: "row = blockIdx.x; col = tid, tid + blockDim.x, ...",
    source: sourceCatalog.map,
    zones: ["grid", "block"],
    arrows: ["grid-arrow"],
    laneMode: "map",
    smemMode: "empty",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["softmax_row_kernel", "1 block / row", "线程跨步", "tid → col"],
  },
  {
    tab: 2,
    concept: 3,
    label: "局部最大值",
    title: "每个线程先找自己的 local_max",
    lessonTitle: "先找最大值，给稳定计算打底",
    summary: "每个线程只看自己负责的列，把局部最大值写入 shared memory。",
    detail: "local_max 初始为 -FLT_MAX，确保任何正常输入都能把它更新掉。",
    calculation: "local_max = max(local_max, X[row * cols + col])",
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
    label: "Max Reduction",
    title: "把所有 local_max 合成 max_val",
    lessonTitle: "第一次归约：求整行最大值",
    summary: "线程两两合作，stride 每轮减半，最后 smax[0] 留下整行最大值。",
    detail: "这一行的最大值是 4.0。之后所有 exp 都会先减去 4.0，避免指数过大。",
    calculation: "max_val = reduce_max(smax)",
    source: sourceCatalog.reduceMax,
    zones: ["block"],
    arrows: [],
    laneMode: "reduceMax",
    smemMode: "maxReduced",
    reductionMode: "max",
    yMode: "reserved",
    stats: ["softmax_row_kernel", "block reduction", "求 max_val", "max_val = 4.0"],
  },
  {
    tab: 3,
    concept: 4,
    label: "局部指数和",
    title: "每个线程计算 exp(x - max_val)",
    lessonTitle: "减最大值后再指数化",
    summary: "有了 max_val，每个线程计算自己负责元素的 exp(x - max_val)，再写入 ssum[tid]。",
    detail: "这里 ssum 和 smax 指向同一块 smem。max 阶段结束后，代码复用这块 shared memory 来存 sum。",
    calculation: "local_sum += expf(X[row * cols + col] - max_val)",
    source: sourceCatalog.localSum,
    zones: ["block"],
    arrows: [],
    laneMode: "sum",
    smemMode: "expValues",
    reductionMode: "none",
    yMode: "reserved",
    stats: ["softmax_row_kernel", "thread local", "local sum", "ssum[tid]"],
  },
  {
    tab: 3,
    concept: 4,
    label: "Sum Reduction",
    title: "把所有 local_sum 合成 sum_val",
    lessonTitle: "第二次归约：求 softmax 分母",
    summary: "和 max 归约很像，只是操作从 fmaxf 换成加法。",
    detail: "sum_val 是这一行所有 exp(x - max_val) 的总和，后面每个输出都要除以它。",
    calculation: "sum_val = reduce_sum(ssum)",
    source: sourceCatalog.reduceSum,
    zones: ["block"],
    arrows: [],
    laneMode: "reduceSum",
    smemMode: "sumReduced",
    reductionMode: "sum",
    yMode: "reserved",
    stats: ["softmax_row_kernel", "block reduction", "求 sum_val", `sum_val = ${format(sumValue)}`],
  },
  {
    tab: 4,
    concept: 5,
    label: "写回",
    title: "每个线程写出自己的 softmax 概率",
    lessonTitle: "最后把概率写回 Y",
    summary: "线程重新遍历自己负责的列，计算 exp(x - max_val) / sum_val 并写入输出矩阵。",
    detail: "这一行输出加起来约等于 1。最大输入 4.0 得到最大概率，但其他元素仍保留相对大小。",
    calculation: "Y[row * cols + col] = expf(x - max_val) / sum_val",
    source: sourceCatalog.write,
    zones: ["block", "output"],
    arrows: ["block-arrow"],
    laneMode: "write",
    smemMode: "sumReduced",
    reductionMode: "sum",
    yMode: "written",
    stats: ["softmax_row_kernel", "thread write", "写回 Y", "ΣY[row] ≈ 1"],
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

function renderReductionTrack() {
  elements.reductionTrack.replaceChildren();
  [
    ["stride 4", "0↔4, 1↔5, 2↔6, 3↔7"],
    ["stride 2", "0↔2, 1↔3"],
    ["stride 1", "0↔1, result → smem[0]"],
  ].forEach(([title, description]) => {
    const node = document.createElement("div");
    node.className = "reduce-step";
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

function updateMatrixClasses(step) {
  const xCells = elements.matrixX.querySelectorAll(".cell");
  const yCells = elements.matrixY.querySelectorAll(".cell");
  const maxIndex = rowValues.indexOf(maxValue);
  xCells.forEach((cell) => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    cell.className = "cell";
    if (row === activeRow && step >= 0) cell.classList.add("is-row");
    if (row === activeRow && (step === 4 || step === 6)) cell.classList.add("is-hot");
    if (row === activeRow && col === maxIndex && step >= 5) {
      cell.classList.add("is-hot");
    }
  });

  yCells.forEach((cell) => {
    const row = Number(cell.dataset.row);
    cell.className = "cell";
    if (row === activeRow && step >= 1) cell.classList.add("is-output");
    if (row === activeRow && step >= 8) cell.classList.add("is-written");
  });
}

function laneTextFor(mode, lane) {
  const x = rowValues[lane];
  const expValue = expValues[lane];
  const probability = probabilities[lane];

  if (mode === "map") return [`X[0,${lane}]`, `读取 ${format(x)}`];
  if (mode === "max") return [`local_max=${format(x)}`, "写入 smax[tid]"];
  if (mode === "reduceMax") return [lane === 0 ? `max=${format(maxValue)}` : "参与比较", "stride 减半归约"];
  if (mode === "sum") return [`exp=${format(expValue)}`, `${format(x)} - ${format(maxValue)}`];
  if (mode === "reduceSum") return [lane === 0 ? `sum=${format(sumValue)}` : "参与求和", "加到 ssum[0]"];
  if (mode === "write") return [`Y=${format(probability)}`, "写回输出"];
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
    if (mode === "sum" || mode === "reduceSum") node.classList.add("is-sum");
    if (mode === "write") node.classList.add("is-write");
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

    if (mode === "expValues") {
      value = format(expValues[lane]);
      cell.classList.add("is-filled");
      if (lane === rowValues.indexOf(maxValue)) cell.classList.add("is-hot");
    }

    if (mode === "sumReduced") {
      value = lane === 0 ? format(sumValue) : lane < 4 ? "合并" : "·";
      cell.classList.add(lane === 0 ? "is-final" : "is-filled");
      if (lane > 3) cell.className = "smem-cell";
    }

    cell.textContent = value;
  });
}

function updateReductionTrack(mode) {
  const nodes = elements.reductionTrack.querySelectorAll(".reduce-step");
  nodes.forEach((node) => {
    classToggle(node, "is-active", mode === "max" || mode === "sum");
  });
}

function updateStats(stepData) {
  const [fn, unit, action, formula] = stepData.stats;
  elements.statFunction.textContent = fn;
  elements.statUnit.textContent = unit;
  elements.statAction.textContent = action;
  elements.statFormula.textContent = formula;

  const stats = [
    ["max_val", stepData.smemMode === "empty" ? "待计算" : format(maxValue)],
    ["sum_val", stepData.smemMode === "sumReduced" ? format(sumValue) : "待计算"],
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
  elements.outputState.textContent = currentStep >= 8 ? "Y[row, col]" : "等待写回";
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
    ? "动画用 8 个 lane 表示真实代码中的 256 个线程"
    : "线程在 kernel 启动后才开始执行";
  elements.calculation.textContent = stepData.calculation;
  elements.phaseDetail.textContent = stepData.detail;

  updateTabs(stepData);
  updateConcepts(stepData);
  updatePipeline(stepData);
  updateMatrixClasses(currentStep);
  updateLanes(stepData.laneMode);
  updateSharedMemory(stepData.smemMode);
  updateReductionTrack(stepData.reductionMode);
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
renderReductionTrack();
bindEvents();
updateSpeedLabel();
showStep(0);
