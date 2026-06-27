const chapterConfig = {
  host: {
    question: "Python 调用进来后，C++ 入口函数先做什么？",
    summary: "`attention_naive` 先检查 Q/K/V，再分配输出 O，最后配置 grid 和 block 启动 kernel。",
    mental: 0,
    factsTitle: "入口职责",
    facts: [
      ["输入", "Q/K/V"],
      ["形状", "[B,H,S,D]"],
      ["输出", "empty_like(Q)"],
      ["启动", "grid(D,S,BH)"],
    ],
    mistake: "入口函数运行在 host 侧，它不直接算每个 attention 值，而是负责把 GPU 任务组织起来。",
  },
  kernel: {
    question: "一个 CUDA block 到底在算哪一小块结果？",
    summary: "这个 naive kernel 的粒度很细：一个 block 只计算一个 `O[b,h,i,d_out]` 标量。",
    mental: 1,
    factsTitle: "Block 定位",
    facts: [
      ["blockIdx.x", "d_out"],
      ["blockIdx.y", "query i"],
      ["blockIdx.z", "b/h"],
      ["threadIdx.x", "分摊 key j"],
    ],
    mistake: "这里不是一个 block 算一整行 O，而是每个 d_out 都单独开一个 block，所以会重复计算同一批 QK score。",
  },
  reduce: {
    question: "softmax 为什么要两遍扫，并且为什么这个实现慢？",
    summary: "第一遍找最大值稳定 softmax，第二遍算分母和加权 V；写法清楚，但重复计算和访存复用都差。",
    mental: 2,
    factsTitle: "Softmax 路径",
    facts: [
      ["第一遍", "max(score)"],
      ["第二遍", "sum(exp)"],
      ["累加", "p × V"],
      ["写回", "sacc / denom"],
    ],
    mistake: "shared memory 在这里主要用于 block 内归约，不是用来缓存 Q/K/V tile；这也是它还不是 FlashAttention 的原因。",
  },
};

const hostSteps = [
  {
    label: "Python API",
    title: "ops.attention_naive(q, k, v, causal)",
    code: "return _C.attention_naive(q, k, v, bool(causal))",
    detail: "Python 包装层只把参数转交给 C++ 扩展，并把 causal 转成 bool。",
    python: true,
    tensors: [],
    axes: [],
  },
  {
    label: "输入检查",
    title: "CHECK_INPUT 确保输入能被这个 kernel 处理",
    code: "CHECK_CUDA + CHECK_CONTIGUOUS + CHECK_FLOAT32",
    detail: "这个 kernel 只处理 CUDA、连续内存、float32 的 Q/K/V。",
    arrow: true,
    hostPart: "check",
    tensors: ["Q", "K", "V"],
    axes: [],
  },
  {
    label: "形状约定",
    title: "Q/K/V 必须都是 [B,H,S,D]",
    code: "TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes())",
    detail: "B 是 batch，H 是 head，S 是序列长度，D 是每个 head 的向量维度。",
    hostPart: "shape",
    tensors: ["Q", "K", "V"],
    axes: [],
  },
  {
    label: "分配输出",
    title: "输出 O 和 Q 拥有相同形状",
    code: "auto O = torch::empty_like(Q);",
    detail: "kernel 稍后会把每个 `O[b,h,i,d_out]` 写进去。",
    hostPart: "output",
    tensors: ["O"],
    axes: [],
  },
  {
    label: "配置 Grid",
    title: "grid(D, S, B × H) 把输出空间铺开",
    code: "dim3 grid(D, S, B * H); block = 128;",
    detail: "x 轴对应输出维度 d_out，y 轴对应 query 位置 i，z 轴对应 batch/head。",
    hostPart: "launch",
    tensors: ["Q", "K", "V", "O"],
    axes: ["d", "s", "bh"],
  },
  {
    label: "启动 Kernel",
    title: "每个 block 进入 attention_naive_kernel",
    code: "attention_naive_kernel<<<grid, block, block * sizeof(float)>>>(...)",
    detail: "第三个 launch 参数是动态 shared memory 大小，这里给每个线程准备一个 float 槽位做归约。",
    arrow: true,
    hostPart: "launch",
    tensors: ["Q", "K", "V", "O"],
    axes: ["d", "s", "bh"],
  },
];

const kernelSteps = [
  {
    label: "选择输出",
    title: "先盯住一个输出标量",
    code: "O[b,h,i,d_out]",
    detail: "为了看清楚 naive 版本，动画只跟踪一个输出元素：`O[0,1,3,3]`。",
    scan: "选中 O[0,1,3,3]",
    output: true,
  },
  {
    label: "解析 blockIdx",
    title: "blockIdx 映射到 d_out、i、b/h",
    code: "d_out = blockIdx.x; i = blockIdx.y; bh = blockIdx.z;",
    detail: "`bh / H` 得到 batch，`bh % H` 得到 head。",
    scan: "blockIdx → 坐标",
    output: true,
    showFormula: true,
  },
  {
    label: "线程分工",
    title: "128 个线程一起扫 key 位置 j",
    code: "for (int j = tid; j < S; j += blockDim.x)",
    detail: "每个线程负责若干个 j。小 S 情况下，很多线程可能没有实际工作。",
    scan: "threadIdx.x 分摊 j",
    output: true,
    activeThreads: [0, 1, 2, 3, 4, 5],
  },
  {
    label: "QK 打分",
    title: "对当前 query i 和某个 key j 做点积",
    code: "score += Q[q_base + d] * K[k_base + d]; score *= rsqrtf(D);",
    detail: "这就是 `Q[i] · K[j] / sqrt(D)`，得到当前位置 i 对 key 位置 j 的注意力分数。",
    scan: "计算 score(i, j)",
    output: true,
    queryActive: true,
    keyActive: 1,
    activeThreads: [1],
    dot: "active",
  },
  {
    label: "Causal Mask",
    title: "causal 模式下不能看未来 token",
    code: "if (causal && j > i) continue;",
    detail: "当前 i = 3，所以 j = 4 和 j = 5 会被跳过，不参与 softmax。",
    scan: "跳过未来位置",
    output: true,
    queryActive: true,
    keyActive: 4,
    maskFuture: true,
    activeThreads: [4, 5],
    dot: "mask",
  },
  {
    label: "乘上 V",
    title: "softmax 权重最终会乘到 V[j,d_out]",
    code: "local_acc += p * V[(base + j) * D + d_out];",
    detail: "注意这里只取 V 的一个维度 d_out，因为这个 block 只负责一个输出标量。",
    scan: "累加 p × V[j,d_out]",
    output: true,
    queryActive: true,
    keyActive: 2,
    valueActive: 2,
    activeThreads: [2],
    dot: "active",
  },
];

const reduceSteps = [
  {
    label: "局部最大值",
    title: "每个线程先找自己负责 score 的最大值",
    code: "float local_max = -FLT_MAX;",
    detail: "softmax 先减最大值，可以避免 `exp(score)` 数值溢出。",
    reductionTitle: "先找每个 score 的最大值",
    laneMode: "active",
    rows: { smax: "active" },
  },
  {
    label: "Max reduction",
    title: "把所有 local_max 合成 block 最大值",
    code: "smax[tid] = fmaxf(smax[tid], smax[tid + stride]);",
    detail: "线程把局部结果写进 shared memory，然后用 stride 逐步减半归约。",
    reductionTitle: "shared memory 中做最大值归约",
    laneMode: "reduce",
    rows: { smax: "final" },
  },
  {
    label: "Softmax 分母",
    title: "第二遍扫描重新计算 score 并累加 exp",
    code: "p = expf(score - max_val); local_sum += p;",
    detail: "`denom = sum(exp(score - max_val))`，它是 softmax 的分母。",
    reductionTitle: "计算 softmax 的分母",
    laneMode: "active",
    rows: { ssum: "final", smax: "muted" },
  },
  {
    label: "加权 V",
    title: "同一遍扫描里累加 p × V",
    code: "local_acc += p * V[(base + j) * D + d_out];",
    detail: "这是输出标量的未归一化加权和，之后还要除以 denom。",
    reductionTitle: "累加当前 d_out 的 V",
    laneMode: "active",
    rows: { sacc: "final", ssum: "active" },
  },
  {
    label: "写回 O",
    title: "线程 0 写出最终结果",
    code: "O[(base + i) * D + d_out] = sacc[0] / denom;",
    detail: "到这里，一个 block 负责的那个输出标量才算完成。",
    reductionTitle: "输出一个 O 标量",
    laneMode: "write",
    rows: { sacc: "final", ssum: "final" },
    writeback: true,
  },
  {
    label: "瓶颈总结",
    title: "这个版本清楚，但故意很慢",
    code: "naive: repeated QK + two passes + little reuse",
    detail: "它适合作为正确性基线和 Nsight 观察对象，后续优化会走 tiled、online softmax、FlashAttention。",
    reductionTitle: "性能瓶颈暴露得很明显",
    laneMode: "muted",
    rows: { smax: "muted", ssum: "muted", sacc: "muted" },
    meters: ["repeat", "twopass", "memory"],
  },
];

const stepsByChapter = {
  host: hostSteps,
  kernel: kernelSteps,
  reduce: reduceSteps,
};

const state = {
  chapter: "host",
  step: 0,
  playing: false,
  intervalId: null,
  intervalMs: 1200,
};

const elements = {
  question: document.querySelector("#chapter-question"),
  summary: document.querySelector("#chapter-summary"),
  progress: document.querySelector("#progress-bar"),
  phaseLabel: document.querySelector("#phase-label"),
  phaseTitle: document.querySelector("#phase-title"),
  stepCount: document.querySelector("#step-count"),
  conceptCode: document.querySelector("#concept-code"),
  phaseDetail: document.querySelector("#phase-detail"),
  factsTitle: document.querySelector("#facts-title"),
  mistake: document.querySelector("#common-mistake"),
  pythonPanel: document.querySelector("#python-panel"),
  hostArrow: document.querySelector("#host-arrow"),
  blockIndexLabel: document.querySelector("#block-index-label"),
  blockFormulaCode: document.querySelector("#block-formula-code"),
  scanStatus: document.querySelector("#scan-status"),
  dotCore: document.querySelector("#dot-core"),
  reductionTitle: document.querySelector("#reduction-title"),
  formulaCode: document.querySelector("#formula-code"),
  formulaDetail: document.querySelector("#formula-detail"),
  writebackBox: document.querySelector("#writeback-box"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  speedInput: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
};

function createTensorStack() {
  const container = document.querySelector("#tensor-stack");
  ["Q", "K", "V", "O"].forEach((name) => {
    const slab = document.createElement("div");
    slab.className = "tensor-slab";
    slab.dataset.tensor = name;

    const label = document.createElement("div");
    label.className = "tensor-name";
    label.textContent = name;
    slab.appendChild(label);

    [
      ["B", "batch"],
      ["H", "head"],
      ["S", "seq"],
      ["D", "dim"],
    ].forEach(([dim, text]) => {
      const item = document.createElement("div");
      item.className = "tensor-dim";
      item.innerHTML = `<span>${text}</span><b>${dim}</b>`;
      slab.appendChild(item);
    });

    container.appendChild(slab);
  });
}

function createOutputMap() {
  const container = document.querySelector("#output-map");
  for (let i = 0; i < 4; i += 1) {
    for (let d = 0; d < 6; d += 1) {
      const cell = document.createElement("span");
      cell.className = "output-cell";
      cell.dataset.i = String(i);
      cell.dataset.d = String(d);
      cell.textContent = `${i},${d}`;
      container.appendChild(cell);
    }
  }
}

function createAttentionMatrix() {
  const query = document.querySelector("#query-cells");
  for (let d = 0; d < 8; d += 1) {
    const cell = document.createElement("span");
    cell.className = "query-cell";
    cell.dataset.d = String(d);
    cell.textContent = `q${d}`;
    query.appendChild(cell);
  }

  const keyStack = document.querySelector("#key-stack");
  const valueStack = document.querySelector("#value-stack");
  for (let j = 0; j < 6; j += 1) {
    const row = document.createElement("div");
    row.className = "key-row";
    row.dataset.j = String(j);
    const label = document.createElement("span");
    label.textContent = `K${j}`;
    row.appendChild(label);
    for (let d = 0; d < 8; d += 1) {
      const cell = document.createElement("span");
      cell.className = "key-cell";
      cell.textContent = `k${d}`;
      row.appendChild(cell);
    }
    keyStack.appendChild(row);

    const valueRow = document.createElement("div");
    valueRow.className = "value-row";
    valueRow.dataset.j = String(j);
    valueRow.innerHTML = `<span>V${j}</span><span class="value-cell">v${j},d</span>`;
    valueStack.appendChild(valueRow);
  }
}

function createThreadStrip() {
  const container = document.querySelector("#thread-strip");
  for (let tid = 0; tid < 16; tid += 1) {
    const token = document.createElement("span");
    token.className = "thread-token";
    token.dataset.tid = String(tid);
    token.textContent = tid < 6 ? `T${tid}→j${tid}` : `T${tid}`;
    container.appendChild(token);
  }
}

function createReductionLab() {
  const laneContainer = document.querySelector("#lane-lab");
  for (let tid = 0; tid < 16; tid += 1) {
    const token = document.createElement("span");
    token.className = "lane-token";
    token.dataset.tid = String(tid);
    token.textContent = `T${tid}`;
    laneContainer.appendChild(token);
  }

  ["smax", "ssum", "sacc"].forEach((rowName) => {
    const row = document.querySelector(`#${rowName}-row`);
    for (let index = 0; index < 8; index += 1) {
      const cell = document.createElement("span");
      cell.className = "shared-cell";
      cell.dataset.index = String(index);
      cell.textContent = index === 0 ? "0" : String(index);
      row.appendChild(cell);
    }
  });
}

function updateFacts(config) {
  elements.factsTitle.textContent = config.factsTitle;
  config.facts.forEach(([label, value], index) => {
    document.querySelector(`#fact-label-${index}`).textContent = label;
    document.querySelector(`#fact-value-${index}`).textContent = value;
  });
}

function updateMentalModel(activeIndex) {
  document.querySelectorAll(".mental-model li").forEach((item, index) => {
    item.classList.toggle("is-current", index === activeIndex);
    item.classList.toggle("is-complete", index < activeIndex);
  });
}

function renderHost(step) {
  elements.pythonPanel.classList.toggle("is-active", Boolean(step.python));
  elements.hostArrow.classList.toggle("is-active", Boolean(step.arrow));

  document.querySelectorAll(".launcher-step").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.hostPart === step.hostPart);
  });

  document.querySelectorAll(".tensor-slab").forEach((slab) => {
    slab.classList.toggle("is-active", step.tensors.includes(slab.dataset.tensor));
  });

  ["d", "s", "bh"].forEach((axis) => {
    document.querySelector(`#axis-${axis}`).classList.toggle("is-active", step.axes.includes(axis));
  });
}

function renderKernel(step) {
  elements.blockIndexLabel.textContent = step.showFormula ? "blockIdx = (3, 3, 1)" : "等待 block 定位";
  elements.blockFormulaCode.textContent = step.showFormula
    ? "bh = 1 → b = 0, h = 1；i = 3；d_out = 3"
    : "d_out = blockIdx.x；i = blockIdx.y；bh = blockIdx.z";
  elements.scanStatus.textContent = step.scan;

  document.querySelectorAll(".output-cell").forEach((cell) => {
    const isSelected = Number(cell.dataset.i) === 3 && Number(cell.dataset.d) === 3;
    cell.classList.toggle("is-visible", Boolean(step.output));
    cell.classList.toggle("is-selected", Boolean(step.output && isSelected));
  });

  document.querySelectorAll(".query-cell").forEach((cell) => {
    cell.classList.toggle("is-active", Boolean(step.queryActive || step.keyActive !== undefined));
  });

  document.querySelectorAll(".key-row").forEach((row) => {
    const j = Number(row.dataset.j);
    const active = j === step.keyActive;
    const masked = Boolean(step.maskFuture && j > 3);
    row.classList.toggle("is-active", active && !masked);
    row.classList.toggle("is-masked", masked);
  });

  document.querySelectorAll(".value-row").forEach((row) => {
    const j = Number(row.dataset.j);
    const active = j === step.valueActive;
    const masked = Boolean(step.maskFuture && j > 3);
    row.classList.toggle("is-active", active);
    row.classList.toggle("is-masked", masked);
  });

  elements.dotCore.classList.toggle("is-active", step.dot === "active");
  elements.dotCore.classList.toggle("is-mask", step.dot === "mask");

  const activeThreads = step.activeThreads || [];
  document.querySelectorAll(".thread-token").forEach((token) => {
    const tid = Number(token.dataset.tid);
    token.classList.toggle("is-active", activeThreads.includes(tid));
    token.classList.toggle("is-muted", activeThreads.length > 0 && !activeThreads.includes(tid));
  });
}

function setSharedRow(rowName, mode) {
  document.querySelectorAll(`#${rowName}-row .shared-cell`).forEach((cell) => {
    const index = Number(cell.dataset.index);
    cell.classList.toggle("is-active", mode === "active" || mode === "final");
    cell.classList.toggle("is-final", mode === "final" && index === 0);
    cell.classList.toggle("is-muted", mode === "muted");
  });
}

function renderReduce(step) {
  elements.reductionTitle.textContent = step.reductionTitle;
  elements.formulaCode.textContent = step.code;
  elements.formulaDetail.textContent = step.detail;
  elements.writebackBox.classList.toggle("is-active", Boolean(step.writeback));

  document.querySelectorAll(".lane-token").forEach((token) => {
    token.className = "lane-token";
    if (step.laneMode === "active") token.classList.add("is-active");
    if (step.laneMode === "reduce") token.classList.add(Number(token.dataset.tid) < 8 ? "is-reduce" : "is-muted");
    if (step.laneMode === "write") token.classList.add(Number(token.dataset.tid) === 0 ? "is-writing" : "is-muted");
    if (step.laneMode === "muted") token.classList.add("is-muted");
  });

  ["smax", "ssum", "sacc"].forEach((rowName) => {
    setSharedRow(rowName, step.rows[rowName] || "");
  });

  document.querySelectorAll(".meter").forEach((meter) => {
    meter.classList.toggle("is-hot", (step.meters || []).includes(meter.dataset.meter));
  });
}

function render() {
  const config = chapterConfig[state.chapter];
  const steps = stepsByChapter[state.chapter];
  const step = steps[state.step];

  elements.question.textContent = config.question;
  elements.summary.textContent = config.summary;
  elements.phaseLabel.textContent = step.label;
  elements.phaseTitle.textContent = step.title;
  elements.conceptCode.textContent = step.code;
  elements.phaseDetail.textContent = step.detail;
  elements.stepCount.textContent = `${String(state.step + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  elements.progress.style.width = `${((state.step + 1) / steps.length) * 100}%`;
  elements.mistake.textContent = config.mistake;

  document.querySelectorAll(".chapter-tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.chapter === state.chapter);
  });

  document.querySelectorAll(".chapter-view").forEach((view) => {
    view.classList.toggle("is-active", view.id === `${state.chapter}-view`);
  });

  updateFacts(config);
  updateMentalModel(config.mental);

  if (state.chapter === "host") renderHost(step);
  if (state.chapter === "kernel") renderKernel(step);
  if (state.chapter === "reduce") renderReduce(step);
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
  const steps = stepsByChapter[state.chapter];
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

function setChapter(chapter) {
  if (!stepsByChapter[chapter]) return;
  stopPlayback();
  state.chapter = chapter;
  state.step = 0;
  render();
}

function updateSpeed(value) {
  state.intervalMs = 2250 - Number(value);
  elements.speedOutput.textContent = `${(1200 / state.intervalMs).toFixed(1)}×`;
  if (state.playing) startPlayback();
}

document.querySelectorAll(".chapter-tab").forEach((tab) => {
  tab.addEventListener("click", () => setChapter(tab.dataset.chapter));
});

document.querySelectorAll("[data-jump-chapter]").forEach((card) => {
  card.addEventListener("click", () => {
    setChapter(card.dataset.jumpChapter);
    document.querySelector(".chapter-tabs").scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

const hostPartStep = {
  check: 1,
  shape: 2,
  output: 3,
  launch: 4,
};

document.querySelectorAll(".launcher-step").forEach((button) => {
  button.addEventListener("click", () => {
    stopPlayback();
    state.chapter = "host";
    state.step = hostPartStep[button.dataset.hostPart] || 0;
    render();
  });
});

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

createTensorStack();
createOutputMap();
createAttentionMatrix();
createThreadStrip();
createReductionLab();
updateSpeed(elements.speedInput.value);
render();
