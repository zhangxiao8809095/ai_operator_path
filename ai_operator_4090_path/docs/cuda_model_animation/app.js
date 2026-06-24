const chapterConfig = {
  launch: {
    question: "一个 CUDA kernel 如何产生大量线程？",
    summary: "程序员配置 Grid、Block、Thread；硬件再把 Block 内线程按 32 个组成 Warp 执行。",
    mental: 0,
    factsTitle: "层级关系",
    facts: [
      ["Grid", "包含多个 Block"],
      ["Block", "定义线程协作范围"],
      ["Warp", "硬件每 32 Thread 分组"],
      ["Thread", "执行一份 kernel 代码"],
    ],
    mistake: "Block 是程序员配置的协作范围；Warp 是硬件从 Block 中自动划分的 32 线程执行分组，两者不是同一层概念。",
  },
  warp: {
    question: "线程是逐个执行，还是成组执行？",
    summary: "程序员组织 Thread 和 Block；硬件把线程按 32 个一组组成 Warp，以 SIMT 方式调度。",
    mental: 1,
    factsTitle: "执行关系",
    facts: [
      ["Thread", "拥有私有状态"],
      ["Warp", "通常包含 32 个线程"],
      ["SIMT", "同指令、多线程数据"],
      ["Barrier", "只同步同一 Block"],
    ],
    mistake: "Warp 不是你在 kernel launch 中配置的维度。它是 GPU 对 Block 内线程进行硬件调度的基本分组。",
  },
  memory: {
    question: "数据应该放在 Register、Shared 还是 Global Memory？",
    summary: "内存位置决定访问速度、容量、共享范围和生命周期；优化常常就是重新安排数据流。",
    mental: 2,
    factsTitle: "内存归属",
    facts: [
      ["Register", "每个 Thread 私有"],
      ["Shared", "同一 Block 共享"],
      ["Global", "整个 Device 可访问"],
      ["同步", "共享数据前明确边界"],
    ],
    mistake: "Shared Memory 不是整个 GPU 共享。它只属于一个 Block，不同 Block 拥有不同的 Shared Memory 实例。",
  },
};

const launchSteps = [
  {
    label: "Kernel launch",
    title: "CPU 配置 Grid 和 Block",
    code: "kernel<<<dim3(3,2), dim3(8,8)>>>()",
    detail: "Host 指定 Grid 有多少个 Block，以及每个 Block 有多少个 Thread。",
    host: true,
    selectedBlock: [0, 0],
    selectedThread: [0, 0],
  },
  {
    label: "创建 Grid",
    title: "一次 launch 产生一个 Grid",
    code: "gridDim = (3, 2) → 6 Blocks",
    detail: "Grid 是本次 kernel 的全部并行任务。二维 Grid 常用来覆盖二维数据。",
    launchArrow: true,
    showBlocks: true,
    selectedBlock: [0, 0],
    selectedThread: [0, 0],
  },
  {
    label: "选择 Block",
    title: "Block (1,1) 是 Grid 中的一组线程",
    code: "blockIdx = (1, 1)",
    detail: "不同 Block 可以被调度到不同 SM，默认不能依赖彼此的执行先后顺序。",
    showBlocks: true,
    dimOtherBlocks: true,
    selectedBlock: [1, 1],
    selectedThread: [0, 0],
  },
  {
    label: "展开线程",
    title: "这个 Block 内含 8 × 8 = 64 个线程",
    code: "blockDim = (8, 8)",
    detail: "每个线程执行相同的 kernel 函数，但 threadIdx 不同，因此可以处理不同数据。",
    showBlocks: true,
    selectedBlock: [1, 1],
    showThreads: true,
    selectedThread: [0, 0],
  },
  {
    label: "组成 Warp",
    title: "64 个线程按线性编号分成两个 Warp",
    code: "Warp 0: tid 0..31；Warp 1: tid 32..63",
    detail: "Warp 是硬件执行分组，不是 kernel launch 的配置项。每个 Warp 通常包含连续的 32 个线程。",
    showBlocks: true,
    selectedBlock: [1, 1],
    showThreads: true,
    showWarps: true,
    selectedThread: [0, 0],
  },
  {
    label: "选择 Thread",
    title: "Thread (3,6) 属于 Warp 1 的 Lane 19",
    code: "linear_tid = 6 × 8 + 3 = 51",
    detail: "threadIdx 是二维局部坐标；硬件先转成线性编号，再用除以 32 得到 Warp 和 Lane。",
    showBlocks: true,
    selectedBlock: [1, 1],
    showThreads: true,
    showWarps: true,
    dimOtherWarps: true,
    dimOtherThreads: true,
    selectedThread: [3, 6],
  },
  {
    label: "计算全局坐标",
    title: "Block 坐标与 Thread 坐标组合成全局坐标",
    code: "global = blockIdx × blockDim + threadIdx",
    detail: "全局坐标才是线程在整个 Grid 中通常用来定位数组元素的位置。",
    showBlocks: true,
    selectedBlock: [1, 1],
    showThreads: true,
    showWarps: true,
    dimOtherWarps: true,
    dimOtherThreads: true,
    selectedThread: [3, 6],
  },
  {
    label: "线程模型总结",
    title: "Grid 决定总任务，Block 决定协作范围，Thread 执行具体工作",
    code: "Grid → Block → Thread → data[index]",
    detail: "写 kernel 时，第一件事通常是计算当前线程应该处理哪个 index。",
    showBlocks: true,
    selectedBlock: [1, 1],
    showThreads: true,
    showWarps: true,
    selectedThread: [3, 6],
  },
];

const warpSteps = [
  {
    label: "Warp 分组",
    title: "Block 内线程被硬件按 32 个组成 Warp",
    code: "warp_id = linear_thread_id / 32",
    detail: "本章聚焦前 32 个线程组成的 Warp 0；同一 Block 中的其余线程会继续组成 Warp 1、Warp 2 等。",
    laneMode: "group",
    instruction: "等待指令",
    status: "Warp 0 已形成",
    cycle: "0",
    active: "32 / 32",
  },
  {
    label: "SIMT 执行",
    title: "Warp 向 32 个 Lane 广播同一条指令",
    code: "y[lane] = x[lane] + 1",
    detail: "同一条指令处理不同线程的数据。每个 Lane 仍有自己的 x、y 和寄存器。",
    laneMode: "active",
    instruction: "y = x + 1",
    status: "同一指令",
    cycle: "1",
    active: "32 / 32",
  },
  {
    label: "合并访存",
    title: "相邻 Lane 读取相邻地址",
    code: "lane i → input[base + i]",
    detail: "连续访问模式更容易合并为较少的内存事务，是 CUDA 性能优化的重要基础。",
    laneMode: "memory",
    instruction: "x = input[base + lane]",
    status: "连续地址",
    cycle: "2",
    active: "32 / 32",
  },
  {
    label: "分支发散",
    title: "同一 Warp 的线程选择了不同分支",
    code: "if (lane % 2 == 0) A(); else B();",
    detail: "Warp 不能同时执行两条不同指令路径，因此需要分时执行路径 A 和路径 B。",
    laneMode: "split",
    instruction: "if (lane % 2 == 0)",
    status: "发生 Divergence",
    cycle: "3",
    active: "16 + 16",
  },
  {
    label: "执行路径 A",
    title: "偶数 Lane 执行 A，其他 Lane 暂时关闭",
    code: "active mask = 0x55555555",
    detail: "关闭并不代表线程消失，它们只是暂时不提交结果。",
    laneMode: "pathA",
    instruction: "A();",
    status: "路径 A",
    cycle: "4",
    active: "16 / 32",
  },
  {
    label: "执行路径 B",
    title: "奇数 Lane 执行 B，然后 Warp 汇合",
    code: "active mask = 0xAAAAAAAA",
    detail: "两条路径串行执行会降低有效吞吐，因此应尽量减少同一 Warp 内的复杂分支发散。",
    laneMode: "pathB",
    instruction: "B();",
    status: "路径 B",
    cycle: "5",
    active: "16 / 32",
  },
  {
    label: "Block 同步",
    title: "__syncthreads() 等待同一 Block 的所有线程",
    code: "__syncthreads();",
    detail: "它常用于确保 Shared Memory 已被所有线程写好，但无法同步不同 Block。",
    laneMode: "group",
    instruction: "__syncthreads()",
    status: "Block barrier",
    cycle: "6",
    active: "32 / 32",
    sync: true,
  },
];

const memorySteps = [
  {
    label: "Global Memory",
    title: "输入数据通常先位于容量最大的 Global Memory",
    code: "const float* input;  // device global memory",
    detail: "所有 Block 都能访问 Global Memory，但延迟较高，应重视访问模式和数据复用。",
    memoryMode: "global",
    action: "Global 数据可见",
    globalActive: [0, 1, 2, 3, 4, 5, 6, 7],
  },
  {
    label: "加载到 Register",
    title: "每个线程把自己的值加载到私有 Register",
    code: "float x = input[global_id];",
    detail: "Register 最靠近线程、速度快，但每个线程只能直接访问自己的寄存器。",
    memoryMode: "register",
    action: "Global → Register",
    globalActive: [0, 1, 2, 3],
    activeRegisters: [0, 1, 2, 3],
    bus: true,
  },
  {
    label: "协作加载",
    title: "同一 Block 的线程把一块数据搬进 Shared Memory",
    code: "tile[threadIdx.x] = input[global_id];",
    detail: "Shared Memory 属于 Block。线程各搬一部分，之后整个 Block 可以重复使用。",
    memoryMode: "shared-load",
    action: "Global → Shared",
    globalActive: [0, 1, 2, 3, 4, 5, 6, 7],
    sharedActive: [0, 1, 2, 3, 4, 5, 6, 7],
    bus: true,
  },
  {
    label: "同步屏障",
    title: "读取 Shared Memory 前先确认协作加载完成",
    code: "__syncthreads();",
    detail: "屏障保证同一 Block 中，所有线程都完成屏障前的 Shared Memory 写入。",
    memoryMode: "shared-sync",
    action: "Block 0 同步",
    sharedActive: [0, 1, 2, 3, 4, 5, 6, 7],
  },
  {
    label: "Block 内复用",
    title: "多个线程反复读取同一份 Shared Memory 数据",
    code: "x += tile[k];",
    detail: "用较少的 Global Memory 读取换取较多的 Shared Memory 访问，是 tiled 算子的核心思路。",
    memoryMode: "shared-reuse",
    action: "Shared → Threads",
    sharedActive: [1, 2, 3, 4],
    activeRegisters: [0, 1, 2, 3],
  },
  {
    label: "Register 累加",
    title: "线程把私有中间结果保存在 Register",
    code: "float acc = acc + x * w;",
    detail: "Register 适合频繁访问的标量和累加器，但使用过多可能降低 Occupancy。",
    memoryMode: "register-acc",
    action: "Thread 私有计算",
    activeRegisters: [0, 1, 2, 3],
  },
  {
    label: "写回结果",
    title: "最终结果从 Register 写回 Global Memory",
    code: "output[global_id] = acc;",
    detail: "kernel 结束后 Register 和 Shared Memory 生命周期结束，Global Memory 中的输出仍然存在。",
    memoryMode: "writeback",
    action: "Register → Global",
    globalActive: [8, 9, 10, 11],
    activeRegisters: [0, 1, 2, 3],
    bus: true,
  },
  {
    label: "内存模型总结",
    title: "越靠近线程越快越私有，越远容量越大共享范围越广",
    code: "Register → Shared → Global",
    detail: "算子优化通常同时考虑数据复用、访问连续性、同步成本和有限的片上资源。",
    memoryMode: "summary",
    action: "归属与生命周期",
    globalActive: [0, 1, 2, 3],
    sharedActive: [0, 1, 2, 3],
    activeRegisters: [0, 1, 2, 3],
  },
];

const stepsByChapter = {
  launch: launchSteps,
  warp: warpSteps,
  memory: memorySteps,
};

const state = {
  chapter: "launch",
  step: 0,
  playing: false,
  intervalId: null,
  intervalMs: 1050,
  selectedBlock: [0, 0],
  selectedThread: [0, 0],
  manualSelection: false,
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
  hostPanel: document.querySelector("#host-panel"),
  launchArrow: document.querySelector("#launch-arrow"),
  selectedBlockLabel: document.querySelector("#selected-block-label"),
  threadName: document.querySelector("#thread-name"),
  globalX: document.querySelector("#global-x-formula"),
  globalY: document.querySelector("#global-y-formula"),
  linearThread: document.querySelector("#linear-thread-formula"),
  warpLane: document.querySelector("#warp-lane-formula"),
  globalCoordinate: document.querySelector("#global-coordinate"),
  warpCoordinate: document.querySelector("#warp-coordinate"),
  warpInstruction: document.querySelector("#warp-instruction"),
  warpStatus: document.querySelector("#warp-status"),
  cycleNumber: document.querySelector("#cycle-number"),
  activeLanes: document.querySelector("#active-lanes"),
  warpExplanation: document.querySelector("#warp-explanation"),
  memoryAction: document.querySelector("#memory-action"),
  memoryBus: document.querySelector("#memory-bus"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  speedInput: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
};

function createGridMap() {
  const container = document.querySelector("#grid-map");
  for (let y = 0; y < 2; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      const block = document.createElement("button");
      block.type = "button";
      block.className = "grid-block";
      block.dataset.x = String(x);
      block.dataset.y = String(y);
      block.textContent = `B(${x},${y})`;
      block.setAttribute("aria-label", `Block (${x}, ${y})`);
      block.addEventListener("click", () => {
        state.selectedBlock = [x, y];
        state.manualSelection = true;
        state.step = Math.max(state.step, 2);
        stopPlayback();
        render();
      });
      container.appendChild(block);
    }
  }
}

function createThreadMap() {
  const container = document.querySelector("#thread-map");
  for (let warp = 0; warp < 2; warp += 1) {
    const group = document.createElement("section");
    group.className = "warp-group";
    group.dataset.warp = String(warp);

    const header = document.createElement("header");
    header.innerHTML = `<strong>Warp ${warp}</strong><span>linear thread ${warp * 32}–${warp * 32 + 31}</span>`;
    group.appendChild(header);

    const threads = document.createElement("div");
    threads.className = "warp-threads";
    for (let local = 0; local < 32; local += 1) {
      const linear = warp * 32 + local;
      const x = linear % 8;
      const y = Math.floor(linear / 8);
      const thread = document.createElement("button");
      thread.type = "button";
      thread.className = "thread-cell";
      thread.dataset.x = String(x);
      thread.dataset.y = String(y);
      thread.dataset.linear = String(linear);
      thread.dataset.warp = String(warp);
      thread.innerHTML = `<span>${x},${y}</span><small>L${local}</small>`;
      thread.setAttribute("aria-label", `Thread (${x}, ${y})`);
      thread.addEventListener("click", () => {
        state.selectedThread = [x, y];
        state.manualSelection = true;
        state.step = Math.max(state.step, 5);
        stopPlayback();
        render();
      });
      threads.appendChild(thread);
    }
    group.appendChild(threads);
    container.appendChild(group);
  }
}

function createLaneGrid() {
  const container = document.querySelector("#lane-grid");
  for (let lane = 0; lane < 32; lane += 1) {
    const item = document.createElement("span");
    item.className = "lane";
    item.dataset.lane = String(lane);
    item.textContent = `L${lane}`;
    container.appendChild(item);
  }
}

function createSyncThreads(containerId) {
  const container = document.querySelector(containerId);
  for (let index = 0; index < 8; index += 1) {
    const item = document.createElement("span");
    item.className = "sync-thread";
    item.dataset.index = String(index);
    container.appendChild(item);
  }
}

function createMemoryModel() {
  [0, 1].forEach((blockId) => {
    const registerRow = document.querySelector(`#register-row-${blockId}`);
    for (let thread = 0; thread < 4; thread += 1) {
      const register = document.createElement("span");
      register.className = "register-thread";
      register.dataset.thread = String(thread);
      register.dataset.block = String(blockId);
      register.textContent = `T${thread}: reg`;
      registerRow.appendChild(register);
    }

    const sharedCells = document.querySelector(`#shared-bank-${blockId} .shared-cells`);
    for (let index = 0; index < 8; index += 1) {
      const cell = document.createElement("span");
      cell.className = "memory-cell";
      cell.dataset.index = String(index);
      cell.textContent = String(index);
      sharedCells.appendChild(cell);
    }
  });

  const globalCells = document.querySelector("#global-cells");
  for (let index = 0; index < 16; index += 1) {
    const cell = document.createElement("span");
    cell.className = "memory-cell";
    cell.dataset.index = String(index);
    cell.textContent = String(index);
    globalCells.appendChild(cell);
  }
}

function createAddressRows() {
  const coalesced = document.querySelector("#coalesced-row");
  const strided = document.querySelector("#strided-row");
  for (let lane = 0; lane < 8; lane += 1) {
    const good = document.createElement("span");
    good.className = "address-cell";
    good.textContent = `L${lane}→${lane}`;
    coalesced.appendChild(good);

    const bad = document.createElement("span");
    bad.className = "address-cell";
    bad.textContent = `L${lane}→${lane * 4}`;
    strided.appendChild(bad);
  }
}

function updateCoordinates() {
  const [blockX, blockY] = state.selectedBlock;
  const [threadX, threadY] = state.selectedThread;
  const globalX = blockX * 8 + threadX;
  const globalY = blockY * 8 + threadY;
  const linearThread = threadY * 8 + threadX;
  const warpId = Math.floor(linearThread / 32);
  const laneId = linearThread % 32;
  elements.selectedBlockLabel.textContent = `Block (${blockX}, ${blockY})`;
  elements.threadName.textContent = `Thread (${threadX}, ${threadY})`;
  elements.globalX.textContent = `${blockX} × 8 + ${threadX} = ${globalX}`;
  elements.globalY.textContent = `${blockY} × 8 + ${threadY} = ${globalY}`;
  elements.linearThread.textContent = `${threadY} × 8 + ${threadX} = ${linearThread}`;
  elements.warpLane.textContent = `Warp ${warpId}，Lane ${laneId}`;
  elements.globalCoordinate.textContent = `(${globalX}, ${globalY})`;
  elements.warpCoordinate.textContent = `Warp ${warpId} / Lane ${laneId}`;
}

function renderLaunch(step) {
  elements.hostPanel.classList.toggle("is-active", Boolean(step.host));
  elements.launchArrow.classList.toggle("is-active", Boolean(step.launchArrow));

  if (!state.manualSelection) {
    if (step.selectedBlock) state.selectedBlock = [...step.selectedBlock];
    if (step.selectedThread) state.selectedThread = [...step.selectedThread];
  }

  document.querySelectorAll(".grid-block").forEach((block, index) => {
    const x = Number(block.dataset.x);
    const y = Number(block.dataset.y);
    const isSelected = x === state.selectedBlock[0] && y === state.selectedBlock[1];
    block.classList.toggle("is-created", Boolean(step.showBlocks));
    block.classList.toggle("is-selected", isSelected);
    block.classList.toggle("is-dimmed", Boolean(step.dimOtherBlocks && !isSelected));
    block.style.animationDelay = `${index * 45}ms`;
    block.style.visibility = step.showBlocks ? "visible" : "hidden";
  });

  document.querySelectorAll(".thread-cell").forEach((thread, index) => {
    const x = Number(thread.dataset.x);
    const y = Number(thread.dataset.y);
    const isSelected = x === state.selectedThread[0] && y === state.selectedThread[1];
    thread.classList.toggle("is-visible", Boolean(step.showThreads));
    thread.classList.toggle("is-selected", isSelected);
    thread.classList.toggle("is-dimmed", Boolean(step.dimOtherThreads && !isSelected));
    thread.style.animationDelay = `${index * 12}ms`;
    thread.style.visibility = step.showThreads ? "visible" : "hidden";
  });

  const selectedLinear = state.selectedThread[1] * 8 + state.selectedThread[0];
  const selectedWarp = Math.floor(selectedLinear / 32);
  document.querySelectorAll(".warp-group").forEach((group) => {
    const warp = Number(group.dataset.warp);
    group.classList.toggle("is-warp-visible", Boolean(step.showWarps));
    group.classList.toggle("is-selected-warp", Boolean(step.showWarps && warp === selectedWarp));
    group.classList.toggle("is-dimmed-warp", Boolean(step.dimOtherWarps && warp !== selectedWarp));
    group.style.visibility = step.showThreads ? "visible" : "hidden";
  });

  updateCoordinates();
}

function renderWarp(step) {
  elements.warpInstruction.textContent = step.instruction;
  elements.warpStatus.textContent = step.status;
  elements.cycleNumber.textContent = step.cycle;
  elements.activeLanes.textContent = step.active;
  elements.warpExplanation.textContent = step.detail;

  document.querySelectorAll(".lane").forEach((lane) => {
    const index = Number(lane.dataset.lane);
    lane.className = "lane";
    if (step.laneMode === "group") lane.classList.add("is-path-a");
    if (step.laneMode === "active") lane.classList.add("is-active");
    if (step.laneMode === "memory") lane.classList.add("is-memory-good");
    if (step.laneMode === "split") lane.classList.add(index % 2 === 0 ? "is-path-a" : "is-path-b");
    if (step.laneMode === "pathA") lane.classList.add(index % 2 === 0 ? "is-active" : "is-off");
    if (step.laneMode === "pathB") lane.classList.add(index % 2 === 1 ? "is-path-b" : "is-off");
  });

  document.querySelectorAll("#sync-block-0 .sync-thread").forEach((thread, index) => {
    thread.className = "sync-thread";
    if (step.sync) {
      thread.classList.add(index < 6 ? "is-arrived" : "is-waiting");
    }
  });
  document.querySelectorAll("#sync-block-1 .sync-thread").forEach((thread) => {
    thread.className = "sync-thread";
  });
}

function renderMemory(step) {
  elements.memoryAction.textContent = step.action;
  elements.memoryBus.classList.toggle("is-active", Boolean(step.bus));

  document.querySelectorAll(".register-thread").forEach((register) => {
    const thread = Number(register.dataset.thread);
    const block = Number(register.dataset.block);
    const active = block === 0 && (step.activeRegisters || []).includes(thread);
    register.classList.toggle("is-active", active);
  });

  document.querySelectorAll("#shared-bank-0 .memory-cell").forEach((cell) => {
    const index = Number(cell.dataset.index);
    cell.classList.toggle("is-loading", (step.sharedActive || []).includes(index));
  });
  document.querySelectorAll("#shared-bank-1 .memory-cell").forEach((cell) => {
    cell.classList.remove("is-loading");
  });

  document.querySelectorAll("#global-cells .memory-cell").forEach((cell) => {
    const index = Number(cell.dataset.index);
    cell.classList.toggle("is-global-active", (step.globalActive || []).includes(index));
  });

  document.querySelectorAll("#coalesced-row .address-cell").forEach((cell) => {
    cell.classList.toggle("is-good", ["global", "shared-load", "summary"].includes(step.memoryMode));
  });
  document.querySelectorAll("#strided-row .address-cell").forEach((cell) => {
    cell.classList.toggle("is-bad", step.memoryMode === "global");
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

  if (state.chapter === "launch") renderLaunch(step);
  if (state.chapter === "warp") renderWarp(step);
  if (state.chapter === "memory") renderMemory(step);
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
  state.manualSelection = false;
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
  state.manualSelection = false;
  render();
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
  state.manualSelection = false;
  render();
});

elements.speedInput.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  state.intervalMs = 2150 - value;
  elements.speedOutput.textContent = `${(1050 / state.intervalMs).toFixed(1)}×`;
  if (state.playing) startPlayback();
});

createGridMap();
createThreadMap();
createLaneGrid();
createSyncThreads("#sync-block-0");
createSyncThreads("#sync-block-1");
createMemoryModel();
createAddressRows();
render();
