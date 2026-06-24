const A = [
  [1, 2, 1, 0],
  [0, 1, 3, 1],
  [2, 0, 1, 2],
  [1, 1, 0, 2],
];

const B = [
  [1, 0, 2, 1],
  [2, 1, 0, 1],
  [1, 3, 1, 0],
  [0, 1, 2, 2],
];

const MATRIX = 4;
const TILE = 2;
const RM = 2;
const RN = 2;

const C = A.map((row) =>
  B[0].map((_, col) =>
    row.reduce((sum, value, k) => sum + value * B[k][col], 0)
  )
);

const threads = [
  { id: "t00", x: 0, y: 0, label: "T(0,0)", baseRow: 0, baseCol: 0 },
  { id: "t10", x: 1, y: 0, label: "T(1,0)", baseRow: 0, baseCol: 2 },
  { id: "t01", x: 0, y: 1, label: "T(0,1)", baseRow: 2, baseCol: 0 },
  { id: "t11", x: 1, y: 1, label: "T(1,1)", baseRow: 2, baseCol: 2 },
];

function sourceLine(number, text, active = false) {
  return { number, text, active };
}

const source = {
  map: {
    location: "gemm.cu:54-70",
    lines: [
      sourceLine(54, "constexpr int RT_TILE = 16;"),
      sourceLine(55, "constexpr int RM = 2;", true),
      sourceLine(56, "constexpr int RN = 2;", true),
      sourceLine(67, "int base_row = blockIdx.y * (RT_TILE * RM) + local_row * RM;", true),
      sourceLine(68, "int base_col = blockIdx.x * (RT_TILE * RN) + local_col * RN;", true),
      sourceLine(70, "float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;", true),
    ],
  },
  load: {
    location: "gemm.cu:72-83",
    lines: [
      sourceLine(72, "for (int t = 0; t < K; t += RT_TILE) {", true),
      sourceLine(74, "    for (int r = 0; r < RM; ++r) {"),
      sourceLine(77, "        As[local_row * RM + r][local_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;", true),
      sourceLine(80, "    for (int c = 0; c < RN; ++c) {"),
      sourceLine(83, "        Bs[local_row][local_col * RN + c] = (row < K && col < N) ? B[row * N + col] : 0.0f;", true),
    ],
  },
  sync: {
    location: "gemm.cu:85",
    lines: [
      sourceLine(77, "As[local_row * RM + r][local_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;"),
      sourceLine(83, "Bs[local_row][local_col * RN + c] = (row < K && col < N) ? B[row * N + col] : 0.0f;"),
      sourceLine(85, "__syncthreads();", true),
    ],
  },
  compute: {
    location: "gemm.cu:87-96",
    lines: [
      sourceLine(87, "#pragma unroll"),
      sourceLine(88, "for (int kk = 0; kk < RT_TILE; ++kk) {", true),
      sourceLine(89, "    float a0 = As[local_row * RM + 0][kk];", true),
      sourceLine(90, "    float a1 = As[local_row * RM + 1][kk];", true),
      sourceLine(91, "    float b0 = Bs[kk][local_col * RN + 0];", true),
      sourceLine(92, "    float b1 = Bs[kk][local_col * RN + 1];", true),
      sourceLine(93, "    acc00 += a0 * b0;", true),
      sourceLine(94, "    acc01 += a0 * b1;", true),
      sourceLine(95, "    acc10 += a1 * b0;", true),
      sourceLine(96, "    acc11 += a1 * b1;", true),
    ],
  },
  write: {
    location: "gemm.cu:101-104",
    lines: [
      sourceLine(101, "if (base_row + 0 < M && base_col + 0 < N) C[(base_row + 0) * N + base_col + 0] = acc00;", true),
      sourceLine(102, "if (base_row + 0 < M && base_col + 1 < N) C[(base_row + 0) * N + base_col + 1] = acc01;", true),
      sourceLine(103, "if (base_row + 1 < M && base_col + 0 < N) C[(base_row + 1) * N + base_col + 0] = acc10;", true),
      sourceLine(104, "if (base_row + 1 < M && base_col + 1 < N) C[(base_row + 1) * N + base_col + 1] = acc11;", true),
    ],
  },
};

const steps = [
  {
    label: "映射",
    title: "4 个线程，每个线程负责一个 2×2 输出块",
    summary: "教学版把 RT_TILE 从 16 缩小到 2：一个 2×2 线程块计算完整 4×4 的 C。",
    detail: "每个线程都有私有 acc00、acc01、acc10、acc11；此时 shared memory 还没有装入数据。",
    calculation: "acc00/01/10/11 = 0",
    source: source.map,
    tile: null,
    k: null,
    uptoK: -1,
  },
  {
    label: "Tile 0 搬运",
    title: "所有线程协作把 K=0..1 搬入 Shared Memory",
    summary: "A 的前两列进入 As[4,2]，B 的前两行进入 Bs[2,4]。",
    detail: "每个线程搬 2 个 A 元素和 2 个 B 元素。关键不是少数线程搬，而是每个 global 元素只搬入 shared 一次。",
    calculation: "Global A[:,0:2] → As；Global B[0:2,:] → Bs",
    source: source.load,
    tile: 0,
    k: null,
    uptoK: -1,
  },
  {
    label: "Tile 0 同步",
    title: "等待整个 Block 都完成搬运",
    summary: "__syncthreads() 保证 As 和 Bs 已经被所有线程填好。",
    detail: "后面的计算会读取其他线程写入的 shared memory，所以必须先让 block 内线程在这里对齐。",
    calculation: "__syncthreads();",
    source: source.sync,
    tile: 0,
    k: null,
    uptoK: -1,
  },
  {
    label: "kk = 0",
    title: "读取 Shared Memory 的第 0 列/行到寄存器",
    summary: "每个线程读取 a0、a1、b0、b1，并更新自己的四个 acc。",
    detail: "同一个 As 元素会被同一行方向的多个线程读取；同一个 Bs 元素会被同一列方向的多个线程读取。",
    calculation: "acc00 += a0*b0；acc01 += a0*b1；acc10 += a1*b0；acc11 += a1*b1",
    source: source.compute,
    tile: 0,
    k: 0,
    uptoK: 0,
  },
  {
    label: "kk = 1",
    title: "继续使用 Tile 0 中的第二组 K 数据",
    summary: "寄存器值继续累加，C 仍未写回 global memory。",
    detail: "注意 C 面板仍为空，因为 kernel 只在所有 K tile 都算完之后才写回 C。",
    calculation: "累加 k=1 的贡献，寄存器保存 k=0..1 的部分和",
    source: source.compute,
    tile: 0,
    k: 1,
    uptoK: 1,
  },
  {
    label: "Tile 1 搬运",
    title: "Shared Memory 被下一块 K 数据覆盖",
    summary: "A 的后两列进入 As，B 的后两行进入 Bs；寄存器中的部分和保留。",
    detail: "shared memory 是 block 的临时工作台，可以覆盖；acc 寄存器是每个线程自己的累加结果，继续保留。",
    calculation: "Global A[:,2:4] → As；Global B[2:4,:] → Bs",
    source: source.load,
    tile: 1,
    k: null,
    uptoK: 1,
  },
  {
    label: "Tile 1 同步",
    title: "第二次等待 Block 完成搬运",
    summary: "所有线程再次在 __syncthreads() 汇合。",
    detail: "没有这次同步，就可能有线程提前读取到尚未写完或正在覆盖的 As/Bs。",
    calculation: "__syncthreads();",
    source: source.sync,
    tile: 1,
    k: null,
    uptoK: 1,
  },
  {
    label: "kk = 2",
    title: "开始累加第二个 K tile",
    summary: "从新的 As/Bs 读取 global k=2 的数据。",
    detail: "shared memory 中显示的是 Tile 1；寄存器值已经包含 Tile 0 的部分和。",
    calculation: "累加 k=2 的贡献，寄存器保存 k=0..2 的部分和",
    source: source.compute,
    tile: 1,
    k: 2,
    uptoK: 2,
  },
  {
    label: "kk = 3",
    title: "完成最后一组乘加",
    summary: "四个线程的寄存器现在已经是最终 C 的 4 个 2×2 小块。",
    detail: "此时数学计算完成，但结果仍在寄存器里；下一步才会写回 C。",
    calculation: "累加 k=3 的贡献，寄存器保存 k=0..3 的最终值",
    source: source.compute,
    tile: 1,
    k: 3,
    uptoK: 3,
  },
  {
    label: "写回 C",
    title: "每个线程把自己的 2×2 结果写回 Global Memory",
    summary: "acc00/01/10/11 分别写到当前线程负责的四个 C 元素。",
    detail: "完整 C[4,4] 由 4 个线程的四组寄存器拼出来；真实代码里每条写回前都有边界检查。",
    calculation: "Registers → C[4,4]",
    source: source.write,
    tile: 1,
    k: null,
    uptoK: 3,
    writeback: true,
  },
];

const state = {
  step: 0,
  selectedThreadId: "t00",
  playing: false,
  intervalId: null,
};

const elements = {
  phaseTitle: document.querySelector("#phase-title"),
  phaseSummary: document.querySelector("#phase-summary"),
  phaseLabel: document.querySelector("#phase-label"),
  stageTitle: document.querySelector("#stage-title"),
  stepCount: document.querySelector("#step-count"),
  progressBar: document.querySelector("#progress-bar"),
  matrixA: document.querySelector("#matrix-a"),
  matrixB: document.querySelector("#matrix-b"),
  matrixC: document.querySelector("#matrix-c"),
  globalNote: document.querySelector("#global-note"),
  sharedTitle: document.querySelector("#shared-title"),
  sharedNote: document.querySelector("#shared-note"),
  sharedA: document.querySelector("#shared-a"),
  sharedB: document.querySelector("#shared-b"),
  registerNote: document.querySelector("#register-note"),
  threadGrid: document.querySelector("#thread-grid"),
  calculation: document.querySelector("#calculation"),
  phaseDetail: document.querySelector("#phase-detail"),
  selectedThreadTitle: document.querySelector("#selected-thread-title"),
  selectedThreadDetail: document.querySelector("#selected-thread-detail"),
  selectedOperands: document.querySelector("#selected-operands"),
  sourceLocation: document.querySelector("#source-location"),
  sourceCode: document.querySelector("#source-code"),
  resetButton: document.querySelector("#reset-button"),
  prevButton: document.querySelector("#prev-button"),
  nextButton: document.querySelector("#next-button"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  speedInput: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
  blockMap: document.querySelector("#block-map"),
  statTile: document.querySelector("#stat-tile"),
  statShared: document.querySelector("#stat-shared"),
  statRegisters: document.querySelector("#stat-registers"),
  statC: document.querySelector("#stat-c"),
  cNote: document.querySelector("#c-note"),
};

function key(row, col) {
  return `${row},${col}`;
}

function makeSet(coordinates) {
  return new Set(coordinates.map(([row, col]) => key(row, col)));
}

function getThread(id) {
  return threads.find((thread) => thread.id === id) || threads[0];
}

function computeAcc(thread, uptoK) {
  const acc = [
    [0, 0],
    [0, 0],
  ];
  if (uptoK < 0) return acc;

  for (let k = 0; k <= uptoK; k += 1) {
    acc[0][0] += A[thread.baseRow + 0][k] * B[k][thread.baseCol + 0];
    acc[0][1] += A[thread.baseRow + 0][k] * B[k][thread.baseCol + 1];
    acc[1][0] += A[thread.baseRow + 1][k] * B[k][thread.baseCol + 0];
    acc[1][1] += A[thread.baseRow + 1][k] * B[k][thread.baseCol + 1];
  }
  return acc;
}

function computeDelta(thread, k) {
  if (k === null || k === undefined) {
    return [
      [0, 0],
      [0, 0],
    ];
  }
  const a0 = A[thread.baseRow + 0][k];
  const a1 = A[thread.baseRow + 1][k];
  const b0 = B[k][thread.baseCol + 0];
  const b1 = B[k][thread.baseCol + 1];
  return [
    [a0 * b0, a0 * b1],
    [a1 * b0, a1 * b1],
  ];
}

function operandsFor(thread, k) {
  if (k === null || k === undefined) return null;
  return {
    a0: A[thread.baseRow + 0][k],
    a1: A[thread.baseRow + 1][k],
    b0: B[k][thread.baseCol + 0],
    b1: B[k][thread.baseCol + 1],
  };
}

function sharedForTile(tile) {
  if (tile === null || tile === undefined) return null;
  const t = tile * TILE;
  return {
    As: Array.from({ length: MATRIX }, (_, row) =>
      Array.from({ length: TILE }, (_, kk) => A[row][t + kk])
    ),
    Bs: Array.from({ length: TILE }, (_, kk) =>
      Array.from({ length: MATRIX }, (_, col) => B[t + kk][col])
    ),
  };
}

function getGlobalHighlights(step) {
  if (step.tile === null || step.tile === undefined || step.writeback) {
    return { a: new Set(), b: new Set() };
  }

  if (step.k !== null && step.k !== undefined) {
    return {
      a: makeSet(Array.from({ length: MATRIX }, (_, row) => [row, step.k])),
      b: makeSet(Array.from({ length: MATRIX }, (_, col) => [step.k, col])),
    };
  }

  const start = step.tile * TILE;
  return {
    a: makeSet(Array.from({ length: MATRIX * TILE }, (_, index) => [
      Math.floor(index / TILE),
      start + (index % TILE),
    ])),
    b: makeSet(Array.from({ length: MATRIX * TILE }, (_, index) => [
      start + Math.floor(index / MATRIX),
      index % MATRIX,
    ])),
  };
}

function getSharedHighlights(step) {
  if (step.tile === null || step.tile === undefined || step.writeback) {
    return { a: new Set(), b: new Set() };
  }

  if (step.k !== null && step.k !== undefined) {
    const kk = step.k % TILE;
    return {
      a: makeSet(Array.from({ length: MATRIX }, (_, row) => [row, kk])),
      b: makeSet(Array.from({ length: MATRIX }, (_, col) => [kk, col])),
    };
  }

  return {
    a: makeSet(Array.from({ length: MATRIX * TILE }, (_, index) => [
      Math.floor(index / TILE),
      index % TILE,
    ])),
    b: makeSet(Array.from({ length: MATRIX * TILE }, (_, index) => [
      Math.floor(index / MATRIX),
      index % MATRIX,
    ])),
  };
}

function renderMatrix(container, matrix, options = {}) {
  const {
    type = "a",
    activeCells = new Set(),
    showValues = true,
    written = false,
    dimInactive = false,
  } = options;
  container.replaceChildren();

  matrix.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const cell = document.createElement("div");
      const isActive = activeCells.has(key(rowIndex, colIndex));
      cell.className = "cell";
      if (type === "a" && isActive) cell.classList.add("is-a-active");
      if (type === "b" && isActive) cell.classList.add("is-b-active");
      if (type === "c" && written) cell.classList.add("is-c-written");
      if (dimInactive && !isActive) cell.classList.add("is-dim");
      cell.textContent = showValues ? value : "·";
      container.appendChild(cell);
    });
  });
}

function renderSharedGrid(container, matrix, activeCells, type, rows, cols) {
  container.replaceChildren();
  container.style.gridTemplateColumns = `repeat(${cols}, minmax(34px, 1fr))`;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const cell = document.createElement("div");
      const hasValue = matrix && matrix[row] && matrix[row][col] !== undefined;
      const isActive = activeCells.has(key(row, col));
      cell.className = "shared-cell";
      if (!hasValue) cell.classList.add("is-empty");
      if (type === "a" && isActive) cell.classList.add("is-a-active", "is-current");
      if (type === "b" && isActive) cell.classList.add("is-b-active", "is-current");
      cell.textContent = hasValue ? matrix[row][col] : "·";
      container.appendChild(cell);
    }
  }
}

function renderThreadCards(step) {
  elements.threadGrid.replaceChildren();
  threads.forEach((thread) => {
    const acc = computeAcc(thread, step.uptoK);
    const delta = computeDelta(thread, step.k);
    const operands = operandsFor(thread, step.k);
    const card = document.createElement("button");
    card.className = "thread-card";
    card.type = "button";
    card.dataset.threadId = thread.id;
    if (thread.id === state.selectedThreadId) card.classList.add("is-selected");

    const header = document.createElement("header");
    const title = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = thread.label;
    const range = document.createElement("small");
    range.textContent = `C[${thread.baseRow}:${thread.baseRow + 1}, ${thread.baseCol}:${thread.baseCol + 1}]`;
    title.append(name, range);
    const badge = document.createElement("span");
    badge.className = "operand-chip";
    badge.textContent = `(${thread.x},${thread.y})`;
    header.append(title, badge);

    const operandRow = document.createElement("div");
    operandRow.className = "operand-row";
    if (operands) {
      ["a0", "a1", "b0", "b1"].forEach((nameKey) => {
        const chip = document.createElement("span");
        chip.className = "operand-chip is-active";
        chip.textContent = `${nameKey}=${operands[nameKey]}`;
        operandRow.appendChild(chip);
      });
    } else {
      const chip = document.createElement("span");
      chip.className = "operand-chip";
      chip.textContent = step.writeback ? "write C" : "hold acc";
      operandRow.appendChild(chip);
    }

    const accGrid = document.createElement("div");
    accGrid.className = "acc-grid";
    [
      ["00", 0, 0],
      ["01", 0, 1],
      ["10", 1, 0],
      ["11", 1, 1],
    ].forEach(([label, row, col]) => {
      const cell = document.createElement("div");
      cell.className = "acc-cell";
      if (step.k !== null && step.k !== undefined && delta[row][col] !== 0) {
        cell.classList.add("is-updated");
      }
      if (step.writeback) cell.classList.add("is-final");
      cell.textContent = `${label}=${acc[row][col]}`;
      accGrid.appendChild(cell);
    });

    card.append(header, operandRow, accGrid);
    elements.threadGrid.appendChild(card);
  });
}

function renderBlockMap() {
  elements.blockMap.replaceChildren();
  threads.forEach((thread) => {
    const tile = document.createElement("button");
    tile.className = "thread-tile";
    tile.type = "button";
    tile.dataset.threadId = thread.id;
    if (thread.id === state.selectedThreadId) tile.classList.add("is-selected");
    const name = document.createElement("strong");
    name.textContent = thread.label;
    const detail = document.createElement("small");
    detail.textContent = `输出 C[${thread.baseRow}:${thread.baseRow + 1}, ${thread.baseCol}:${thread.baseCol + 1}]`;
    tile.append(name, detail);
    elements.blockMap.appendChild(tile);
  });
}

function renderSelectedThread(step) {
  const thread = getThread(state.selectedThreadId);
  const acc = computeAcc(thread, step.uptoK);
  const operands = operandsFor(thread, step.k);
  elements.selectedThreadTitle.textContent = thread.label;
  elements.selectedThreadDetail.textContent =
    `负责 C[${thread.baseRow}:${thread.baseRow + 1}, ${thread.baseCol}:${thread.baseCol + 1}]，当前寄存器为 [[${acc[0][0]}, ${acc[0][1]}], [${acc[1][0]}, ${acc[1][1]}]]。`;

  elements.selectedOperands.replaceChildren();
  if (operands) {
    [
      ["a0", operands.a0],
      ["a1", operands.a1],
      ["b0", operands.b0],
      ["b1", operands.b1],
    ].forEach(([name, value]) => {
      const chip = document.createElement("span");
      chip.className = "operand-chip is-active";
      chip.textContent = `${name}=${value}`;
      elements.selectedOperands.appendChild(chip);
    });
  } else {
    const chip = document.createElement("span");
    chip.className = "operand-chip";
    chip.textContent = step.writeback ? "寄存器写回 C" : "等待计算";
    elements.selectedOperands.appendChild(chip);
  }
}

function renderSource(step) {
  elements.sourceLocation.textContent = step.source.location;
  elements.sourceCode.replaceChildren();
  step.source.lines.forEach((line) => {
    const row = document.createElement("span");
    row.className = "source-line";
    if (line.active) row.classList.add("is-active");
    const number = document.createElement("i");
    number.textContent = String(line.number);
    const code = document.createElement("code");
    code.textContent = line.text;
    row.append(number, code);
    elements.sourceCode.appendChild(row);
  });
}

function renderStats(step) {
  if (step.tile === null || step.tile === undefined) {
    elements.statTile.textContent = "未开始";
  } else {
    const start = step.tile * TILE;
    elements.statTile.textContent = `t=${start}, K=${start}..${start + 1}`;
  }

  if (step.tile === null || step.tile === undefined) {
    elements.statShared.textContent = "空";
  } else if (step.k !== null && step.k !== undefined) {
    elements.statShared.textContent = `正在读 kk=${step.k % TILE}`;
  } else {
    elements.statShared.textContent = "已装入当前 tile";
  }

  elements.statRegisters.textContent =
    step.uptoK < 0 ? "全部为 0" : `已累计 k=0..${step.uptoK}`;
  elements.statC.textContent = step.writeback ? "完整写回" : "尚未发生";
  elements.cNote.textContent = step.writeback ? "最终结果" : "尚未写回";
}

function renderNotes(step) {
  if (step.tile === null || step.tile === undefined) {
    elements.globalNote.textContent = "等待把第一个 K tile 搬入 shared memory";
    elements.sharedTitle.textContent = "As[4,2] 与 Bs[2,4]";
    elements.sharedNote.textContent = "Shared memory 还没有装入数据";
  } else {
    const start = step.tile * TILE;
    elements.globalNote.textContent =
      step.k === null || step.k === undefined
        ? `当前搬运 K=${start}..${start + 1}`
        : `当前计算 global k=${step.k}`;
    elements.sharedTitle.textContent = `As/Bs 当前保存 K=${start}..${start + 1}`;
    elements.sharedNote.textContent =
      step.k === null || step.k === undefined
        ? "高亮的是整块将被多个线程复用的 tile"
        : "高亮的是本轮 kk 会被多个线程读取的 shared 元素";
  }

  elements.registerNote.textContent =
    step.k === null || step.k === undefined
      ? "点击线程卡片可查看该线程负责的 C 位置"
      : "黄色操作数来自 shared memory；粉色 acc 表示本轮发生了非零更新";
}

function render() {
  const step = steps[state.step];
  const globalHighlights = getGlobalHighlights(step);
  const sharedHighlights = getSharedHighlights(step);
  const shared = sharedForTile(step.tile);

  elements.phaseTitle.textContent = step.title;
  elements.phaseSummary.textContent = step.summary;
  elements.phaseLabel.textContent = step.label;
  elements.stageTitle.textContent = step.title;
  elements.stepCount.textContent =
    `${String(state.step + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  elements.progressBar.style.width = `${((state.step + 1) / steps.length) * 100}%`;
  elements.calculation.textContent = step.calculation;
  elements.phaseDetail.textContent = step.detail;

  renderMatrix(elements.matrixA, A, {
    type: "a",
    activeCells: globalHighlights.a,
    dimInactive: globalHighlights.a.size > 0,
  });
  renderMatrix(elements.matrixB, B, {
    type: "b",
    activeCells: globalHighlights.b,
    dimInactive: globalHighlights.b.size > 0,
  });
  renderMatrix(elements.matrixC, C, {
    type: "c",
    showValues: Boolean(step.writeback),
    written: Boolean(step.writeback),
  });

  renderSharedGrid(elements.sharedA, shared ? shared.As : null, sharedHighlights.a, "a", MATRIX, TILE);
  renderSharedGrid(elements.sharedB, shared ? shared.Bs : null, sharedHighlights.b, "b", TILE, MATRIX);
  renderThreadCards(step);
  renderBlockMap();
  renderSelectedThread(step);
  renderSource(step);
  renderStats(step);
  renderNotes(step);
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

function startPlayback() {
  stopPlayback();
  state.playing = true;
  elements.playIcon.textContent = "Ⅱ";
  elements.playButton.title = "暂停";
  elements.playButton.setAttribute("aria-label", "暂停");
  state.intervalId = window.setInterval(() => {
    if (state.step >= steps.length - 1) {
      stopPlayback();
      return;
    }
    state.step += 1;
    render();
  }, Number(elements.speedInput.value));
}

function updateSpeedLabel() {
  const speed = 1050 / Number(elements.speedInput.value);
  elements.speedOutput.textContent = `${speed.toFixed(1)}×`;
}

elements.nextButton.addEventListener("click", () => {
  stopPlayback();
  state.step = Math.min(steps.length - 1, state.step + 1);
  render();
});

elements.prevButton.addEventListener("click", () => {
  stopPlayback();
  state.step = Math.max(0, state.step - 1);
  render();
});

elements.resetButton.addEventListener("click", () => {
  stopPlayback();
  state.step = 0;
  render();
});

elements.playButton.addEventListener("click", () => {
  if (state.playing) {
    stopPlayback();
  } else {
    if (state.step >= steps.length - 1) state.step = 0;
    startPlayback();
  }
});

elements.speedInput.addEventListener("input", () => {
  updateSpeedLabel();
  if (state.playing) startPlayback();
});

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-thread-id]");
  if (!target) return;
  state.selectedThreadId = target.dataset.threadId;
  render();
});

updateSpeedLabel();
render();
