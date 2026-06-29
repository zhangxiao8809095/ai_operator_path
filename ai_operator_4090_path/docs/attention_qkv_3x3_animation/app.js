const B = 1;
const H = 3;
const S = 4;
const D = 3;
const scale = 1 / Math.sqrt(D);
const target = { b: 0, h: 1, i: 1, d: 1, bh: 1 };

const Q = [
  [
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 1, 0],
  ],
  [
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
  ],
  [
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
  ],
];

const K = [
  [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 1],
  ],
  [
    [1, 0, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 0],
  ],
  [
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
  ],
];

const V = [
  [
    [1, 2, 0],
    [2, 0, 1],
    [0, 1, 3],
    [1, 1, 1],
  ],
  [
    [2, 0, 1],
    [0, 3, 1],
    [1, 1, 2],
    [2, 2, 0],
  ],
  [
    [1, 1, 2],
    [3, 0, 1],
    [0, 2, 2],
    [2, 1, 0],
  ],
];

function computeHead(head) {
  const scores = Q[head].map((qRow) =>
    K[head].map((kRow) => qRow.reduce((sum, q, d) => sum + q * kRow[d], 0) * scale),
  );

  const probs = scores.map((row) => {
    const max = Math.max(...row);
    const exps = row.map((value) => Math.exp(value - max));
    const denom = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / denom);
  });

  const output = probs.map((probRow) =>
    Array.from({ length: D }, (_, d) =>
      probRow.reduce((sum, p, j) => sum + p * V[head][j][d], 0),
    ),
  );

  return { scores, probs, output };
}

const headResults = Array.from({ length: H }, (_, h) => computeHead(h));
const activeHead = headResults[target.h];
const scores = activeHead.scores;
const output = activeHead.output;
const targetScores = scores[target.i];
const targetMax = Math.max(...targetScores);
const targetExp = targetScores.map((value) => Math.exp(value - targetMax));
const targetDenom = targetExp.reduce((sum, value) => sum + value, 0);
const targetAcc = targetExp.map((p, j) => p * V[target.h][j][target.d]);
const targetAccTotal = targetAcc.reduce((sum, value) => sum + value, 0);
const targetValue = targetAccTotal / targetDenom;

const steps = [
  {
    title: "Grid 里有 3 × 4 × 3 个 block，每个 block 负责一个 O[h,i,d_out]",
    phase: "Kernel launch",
    summary: "这里 B=1、H=3、S=4、D=3，所以 grid(D,S,BH) 变成 grid(3,4,3)，一共 36 个 block。",
    formula: "dim3 grid(D, S, B * H) = dim3(3, 4, 3)",
    result: "36 个 block 拼出 3 个 head 的 O",
    focus: "grid",
    showGrid: true,
    showHeads: true,
  },
  {
    title: "grid.z 现在有 3 层：bh=0、bh=1、bh=2",
    phase: "Head layer",
    summary: "因为 B=1、H=3，所以 bh 实际上就是当前 head 编号；真实代码仍然用 bh/H 和 bh%H 还原 b/h。",
    formula: "bh = blockIdx.z；b = bh / H；h = bh % H",
    result: "bh=1 -> b=0, h=1",
    focus: "grid",
    showGrid: true,
    showHeads: true,
    activeHead: target.h,
  },
  {
    title: "当前 block 认领 O[h=1,i=1,d=1] 这个格子",
    phase: "Block mapping",
    summary: "blockIdx.x 给出 d_out，blockIdx.y 给出 query 位置 i，blockIdx.z 给出 batch/head 平面。",
    formula: "d_out=1；i=1；bh=1；b=1/3=0；h=1%3=1",
    result: "target = O[b=0,h=1,i=1,d=1]",
    focus: "block",
    showGrid: true,
    showHeads: true,
    activeHead: target.h,
    target: true,
  },
  {
    title: "固定 Q[h=1,i=1]，因为当前 block 已经选定 head 和 query",
    phase: "Q row",
    summary: "加入 H 之后，当前 block 不会混用其他 head；它只读同一个 h 里的 Q/K/V。",
    formula: "base = ((b * H + h) * S)；q_base = (base + i) * D",
    result: "Q[h=1,i=1] = [0, 1, 1]",
    focus: "block",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
  },
  {
    title: "block 内线程只扫描 head 1 里的所有 j",
    phase: "Thread scan",
    summary: "T0/T1/T2/T3 分别代表 j=0/1/2/3；它们读取的是 K[h=1,j] 和 V[h=1,j,d]。",
    formula: "for (int j = tid; j < S; j += blockDim.x)",
    result: "T0→j0，T1→j1，T2→j2，T3→j3",
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    kRows: [0, 1, 2, 3],
    activeLanes: [0, 1, 2, 3],
    scoreMode: "dot",
  },
  {
    title: "第一遍：每个线程计算自己的 score，并写入 smax",
    phase: "Pass 1: score/max",
    summary: "score 只在同一个 head 内计算：Q[h=1,i=1] · K[h=1,j] / sqrt(D)。",
    formula: "score(j) = Q[1,1] · K[1,j] / sqrt(3)",
    result: "local_max 分别来自 j=0、j=1、j=2、j=3",
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    kRows: [0, 1, 2, 3],
    activeLanes: [0, 1, 2, 3],
    scoreMode: "dot",
    smax: "local",
  },
  {
    title: "归约 smax，得到 head 1 这一行 softmax 需要的 max_val",
    phase: "Reduce max",
    summary: "不同 head 的 block 互不共享这个 max；当前 max 只属于 bh=1 这一个 block。",
    formula: "max_val = max(score[0], score[1], score[2], score[3])",
    result: `max_val = ${fmt(targetMax)}`,
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    kRows: [0, 1, 2, 3],
    scoreMode: "dot",
    smax: "final",
  },
  {
    title: "第二遍：重新算 score，变成 exp(score - max_val)",
    phase: "Pass 2: exp/sum",
    summary: "这里依然只处理 h=1 的那一行 attention 权重。",
    formula: "p = expf(score - max_val)；local_sum += p",
    result: `local p = [${targetExp.map(fmt).join(", ")}]`,
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    kRows: [0, 1, 2, 3],
    activeLanes: [0, 1, 2, 3],
    scoreMode: "exp",
    smax: "muted",
    ssum: "local",
  },
  {
    title: "归约 ssum，得到 softmax 分母 denom",
    phase: "Reduce sum",
    summary: "softmax 分母只覆盖当前 head 内的 j=0/1/2/3，不会跨 head 归一化。",
    formula: "denom = sum_j exp(score[j] - max_val)",
    result: `denom = ${fmt(targetDenom)}`,
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    kRows: [0, 1, 2, 3],
    scoreMode: "prob",
    ssum: "final",
  },
  {
    title: "同一遍里，每个线程还会累加 p × V[h=1,j,d_out]",
    phase: "Weighted V",
    summary: "当前 block 只算 O[h=1,i=1,d=1]，所以只取 V[h=1,j,1] 这一列。",
    formula: "local_acc += p * V[(base + j) * D + d_out]",
    result: `local_acc = [${targetAcc.map(fmt).join(", ")}]`,
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    kRows: [0, 1, 2, 3],
    vCol: target.d,
    activeLanes: [0, 1, 2, 3],
    scoreMode: "acc",
    ssum: "final",
    sacc: "local",
  },
  {
    title: "归约 sacc，得到 head 1 当前输出维度的加权 V 总和",
    phase: "Reduce acc",
    summary: "所有线程把自己负责的 j 贡献合并到 sacc[0]。",
    formula: "sacc[0] = sum_j exp(score[j] - max_val) * V[h,j,d_out]",
    result: `sacc[0] = ${fmt(targetAccTotal)}`,
    focus: "thread",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    vCol: target.d,
    scoreMode: "acc",
    ssum: "final",
    sacc: "final",
  },
  {
    title: "tid=0 写回 O[b=0,h=1,i=1,d=1]",
    phase: "Write O",
    summary: "最后除以 denom，写回的是四维 O 里的一个元素，而不是整个 O。",
    formula: "O[(base + i) * D + d_out] = sacc[0] / denom",
    result: `O[0,1,1,1] = ${fmt(targetAccTotal)} / ${fmt(targetDenom)} = ${fmt(targetValue)}`,
    focus: "write",
    showHeads: true,
    activeHead: target.h,
    target: true,
    qRow: target.i,
    vCol: target.d,
    writeTarget: true,
    ssum: "final",
    sacc: "final",
  },
  {
    title: "其他 35 个 block 重复同样流程，填满 3 个 head 的 O",
    phase: "Back to whole O",
    summary: "你现在看到的是 h=1 这一层的 4×3 O；h=0 和 h=2 也各有自己的 4×3 O。",
    formula: "所有 block: O[b,h,i,d] = sum_j softmax(Q[b,h,i]K[b,h,j]) * V[b,h,j,d]",
    result: "完整 O 形状是 [B,H,S,D] = [1,3,4,3]",
    focus: "grid",
    showHeads: true,
    activeHead: target.h,
    showFullO: true,
  },
];

const state = {
  step: 0,
  playing: false,
  intervalId: null,
  intervalMs: 1150,
};

const elements = {
  title: document.querySelector("#step-title"),
  summary: document.querySelector("#step-summary"),
  progress: document.querySelector("#progress-bar"),
  phaseLabel: document.querySelector("#phase-label"),
  phaseTitle: document.querySelector("#phase-title"),
  stepCount: document.querySelector("#step-count"),
  qHeadLabel: document.querySelector("#q-head-label"),
  kHeadLabel: document.querySelector("#k-head-label"),
  vHeadLabel: document.querySelector("#v-head-label"),
  oHeadLabel: document.querySelector("#o-head-label"),
  blockTitle: document.querySelector("#block-title"),
  mappingCode: document.querySelector("#mapping-code"),
  targetCode: document.querySelector("#target-code"),
  scoreStatus: document.querySelector("#score-status"),
  sharedTitle: document.querySelector("#shared-title"),
  formulaCode: document.querySelector("#formula-code"),
  formulaResult: document.querySelector("#formula-result"),
  takeaway: document.querySelector("#takeaway-text"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  speedInput: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
};

function fmt(value) {
  return Number(value).toFixed(2);
}

function dotFormula(i, j) {
  return Q[target.h][i].map((value, d) => `${value}*${K[target.h][j][d]}`).join(" + ");
}

function createMatrix(id, values, matrixName) {
  const container = document.querySelector(id);
  values.forEach((row, r) => {
    row.forEach((value, c) => {
      const cell = document.createElement("span");
      cell.className = "cell";
      cell.dataset.matrix = matrixName;
      cell.dataset.row = String(r);
      cell.dataset.col = String(c);
      cell.textContent = matrixName === "O" ? "·" : String(value);
      container.appendChild(cell);
    });
  });
}

function createHeadStrip() {
  const container = document.querySelector("#head-strip");
  for (let h = 0; h < H; h += 1) {
    const item = document.createElement("div");
    item.className = "head-layer";
    item.dataset.head = String(h);
    item.innerHTML = `<span>bh=${h}</span><strong>h=${h}</strong><small>4×3 O plane</small>`;
    container.appendChild(item);
  }
}

function createThreadLanes() {
  const container = document.querySelector("#thread-lanes");
  for (let tid = 0; tid < S; tid += 1) {
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.dataset.tid = String(tid);
    lane.innerHTML = `<span>T${tid}</span><strong>j=${tid}</strong><small>读取 K[h,${tid}] 和 V[h,${tid},d]</small>`;
    container.appendChild(lane);
  }
}

function createScoreRow() {
  const container = document.querySelector("#score-row");
  for (let j = 0; j < S; j += 1) {
    const item = document.createElement("div");
    item.className = "score-item";
    item.dataset.j = String(j);
    item.innerHTML = `<span>h=1, j=${j}</span><code>${dotFormula(target.i, j)}</code><strong>${fmt(targetScores[j])}</strong>`;
    container.appendChild(item);
  }
}

function createSharedRows() {
  ["smax", "ssum", "sacc"].forEach((name) => {
    const row = document.querySelector(`#${name}-row`);
    for (let index = 0; index < S; index += 1) {
      const cell = document.createElement("span");
      cell.className = "shared-cell";
      cell.dataset.index = String(index);
      cell.textContent = "·";
      row.appendChild(cell);
    }
  });
}

function resetClasses() {
  document.querySelectorAll(".cell").forEach((cell) => {
    cell.className = "cell";
  });
  document.querySelectorAll(".head-layer").forEach((item) => {
    item.className = "head-layer";
  });
  document.querySelectorAll(".lane").forEach((lane) => {
    lane.className = "lane";
  });
  document.querySelectorAll(".score-item").forEach((item) => {
    item.className = "score-item";
  });
  document.querySelectorAll(".shared-cell").forEach((cell) => {
    cell.className = "shared-cell";
  });
}

function matrixCell(matrixName, row, col) {
  return document.querySelector(`.cell[data-matrix="${matrixName}"][data-row="${row}"][data-col="${col}"]`);
}

function applyMatrixHighlights(step) {
  document.querySelectorAll('.cell[data-matrix="O"]').forEach((cell) => {
    const r = Number(cell.dataset.row);
    const c = Number(cell.dataset.col);
    cell.textContent = step.showFullO ? fmt(output[r][c]) : "·";
    if (step.showGrid) cell.classList.add("is-pending");
    if (step.showFullO) cell.classList.add("is-written");
  });

  if (step.qRow !== undefined) {
    for (let c = 0; c < D; c += 1) {
      matrixCell("Q", step.qRow, c).classList.add("is-q", "is-pulse");
    }
  }

  if (step.kRows) {
    step.kRows.forEach((r) => {
      for (let c = 0; c < D; c += 1) {
        matrixCell("K", r, c).classList.add("is-k");
      }
    });
  }

  if (step.vCol !== undefined) {
    for (let r = 0; r < S; r += 1) {
      matrixCell("V", r, step.vCol).classList.add("is-v", "is-pulse");
    }
  }

  if (step.target || step.writeTarget) {
    const targetCell = matrixCell("O", target.i, target.d);
    targetCell.textContent = step.writeTarget ? fmt(targetValue) : "block";
    targetCell.classList.add(step.writeTarget ? "is-written" : "is-o", "is-pulse");
  }

  if (!step.showFullO) {
    document.querySelectorAll('.cell[data-matrix="Q"], .cell[data-matrix="K"], .cell[data-matrix="V"]').forEach((cell) => {
      const active = cell.classList.contains("is-q") || cell.classList.contains("is-k") || cell.classList.contains("is-v");
      if ((step.qRow !== undefined || step.kRows || step.vCol !== undefined) && !active) {
        cell.classList.add("is-muted");
      }
    });
  }
}

function renderHeads(step) {
  document.querySelectorAll(".head-layer").forEach((item) => {
    const h = Number(item.dataset.head);
    if (step.showHeads) item.classList.add("is-visible");
    if (step.activeHead === h || ((step.target || step.writeTarget || step.showFullO) && h === target.h)) {
      item.classList.add("is-active");
    }
    if ((step.target || step.writeTarget) && h !== target.h) item.classList.add("is-muted");
    if (step.showFullO) item.classList.add("is-complete");
  });
}

function renderLanes(step) {
  const active = step.activeLanes || [];
  document.querySelectorAll(".lane").forEach((lane) => {
    const tid = Number(lane.dataset.tid);
    if (active.includes(tid)) lane.classList.add("is-active");
    if (active.length > 0 && !active.includes(tid)) lane.classList.add("is-muted");
  });
}

function renderScoreItems(step) {
  document.querySelectorAll(".score-item").forEach((item) => {
    const j = Number(item.dataset.j);
    const code = item.querySelector("code");
    const strong = item.querySelector("strong");

    if (!step.scoreMode) {
      item.classList.add("is-muted");
      code.textContent = "等待当前 block 扫描 j";
      strong.textContent = "·";
      return;
    }

    if (step.scoreMode === "dot") {
      code.textContent = `(${dotFormula(target.i, j)}) / sqrt(3)`;
      strong.textContent = `score=${fmt(targetScores[j])}`;
      item.classList.add("is-active");
    }

    if (step.scoreMode === "exp") {
      code.textContent = `exp(${fmt(targetScores[j])} - ${fmt(targetMax)})`;
      strong.textContent = `p=${fmt(targetExp[j])}`;
      item.classList.add("is-softmax");
    }

    if (step.scoreMode === "prob") {
      code.textContent = `${fmt(targetExp[j])} / ${fmt(targetDenom)}`;
      strong.textContent = `w=${fmt(targetExp[j] / targetDenom)}`;
      item.classList.add("is-softmax");
    }

    if (step.scoreMode === "acc") {
      code.textContent = `${fmt(targetExp[j])} * V[h=1,${j},${target.d}](${V[target.h][j][target.d]})`;
      strong.textContent = `acc=${fmt(targetAcc[j])}`;
      item.classList.add("is-active");
    }
  });
}

function setSharedRow(rowName, mode, values, finalValue) {
  const cells = document.querySelectorAll(`#${rowName}-row .shared-cell`);
  cells.forEach((cell, index) => {
    cell.textContent = "·";
    if (!mode) return;

    if (mode === "muted") {
      cell.classList.add("is-muted");
      return;
    }

    if (mode === "local") {
      cell.textContent = fmt(values[index]);
      cell.classList.add("is-active");
      return;
    }

    if (mode === "final") {
      if (index === 0) {
        cell.textContent = fmt(finalValue);
        cell.classList.add("is-final");
      } else {
        cell.classList.add("is-muted");
      }
    }
  });
}

function renderShared(step) {
  setSharedRow("smax", step.smax, targetScores, targetMax);
  setSharedRow("ssum", step.ssum, targetExp, targetDenom);
  setSharedRow("sacc", step.sacc, targetAcc, targetAccTotal);

  if (step.smax === "final") elements.sharedTitle.textContent = "smax[0] 保存 max_val";
  else if (step.ssum === "final") elements.sharedTitle.textContent = "ssum[0] 保存 denom";
  else if (step.sacc === "final") elements.sharedTitle.textContent = "sacc[0] 保存加权 V 总和";
  else elements.sharedTitle.textContent = "smax / ssum / sacc 复用同一块 smem";
}

function renderFocus(focus) {
  const order = ["grid", "block", "thread", "write"];
  const activeIndex = order.indexOf(focus);
  document.querySelectorAll(".focus-panel li").forEach((item) => {
    const index = order.indexOf(item.dataset.focus);
    item.classList.toggle("is-current", item.dataset.focus === focus);
    item.classList.toggle("is-complete", index >= 0 && activeIndex >= 0 && index < activeIndex);
  });
}

function render() {
  const step = steps[state.step];
  resetClasses();

  elements.title.textContent = step.title;
  elements.summary.textContent = step.summary;
  elements.phaseLabel.textContent = step.phase;
  elements.phaseTitle.textContent = step.title;
  elements.stepCount.textContent = `${String(state.step + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  elements.progress.style.width = `${((state.step + 1) / steps.length) * 100}%`;
  elements.qHeadLabel.textContent = `Input · head h=${target.h}`;
  elements.kHeadLabel.textContent = `Input · head h=${target.h}`;
  elements.vHeadLabel.textContent = `Input · head h=${target.h}`;
  elements.oHeadLabel.textContent = `Output · head h=${target.h}`;
  elements.formulaCode.textContent = step.formula;
  elements.formulaResult.textContent = step.result;
  elements.scoreStatus.textContent = step.scoreMode ? "当前 block 正在扫描 h=1 内的 j" : "等待当前 block 扫描 j";
  elements.blockTitle.textContent = step.target || step.writeTarget
    ? "blockIdx = (d_out=1, i=1, bh=1)"
    : "等待选择 block";
  elements.mappingCode.textContent = step.target || step.writeTarget
    ? "d_out=1；i=1；bh=1；b=0；h=1"
    : "d_out = blockIdx.x；i = blockIdx.y；bh = blockIdx.z";
  elements.targetCode.textContent = step.target || step.writeTarget
    ? "target: O[b=0,h=1,i=1,d=1]"
    : "target: 每个 block 各不相同";
  elements.takeaway.textContent = step.showFullO
    ? "H=3 后，完整 O 有 3 个 head 平面；一个 block 仍然只写其中一个 head 平面里的一个 O 元素。"
    : "当前动画始终站在一个 block 的视角：先用 bh 选 head，再固定一个 O 格子，扫描这个 head 内的所有 j。";

  applyMatrixHighlights(step);
  renderHeads(step);
  renderLanes(step);
  renderScoreItems(step);
  renderShared(step);
  renderFocus(step.focus);
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
  const next = state.step + direction;
  if (next >= steps.length) {
    state.step = state.playing ? 0 : steps.length - 1;
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

function updateSpeed(value) {
  state.intervalMs = 2200 - Number(value);
  elements.speedOutput.textContent = `${(1100 / state.intervalMs).toFixed(1)}x`;
  if (state.playing) startPlayback();
}

elements.playButton.addEventListener("click", () => {
  if (state.playing) stopPlayback();
  else startPlayback();
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

elements.speedInput.addEventListener("input", (event) => updateSpeed(event.target.value));

createMatrix("#q-matrix", Q[target.h], "Q");
createMatrix("#k-matrix", K[target.h], "K");
createMatrix("#v-matrix", V[target.h], "V");
createMatrix("#o-matrix", output, "O");
createHeadStrip();
createThreadLanes();
createScoreRow();
createSharedRows();
updateSpeed(elements.speedInput.value);
render();
