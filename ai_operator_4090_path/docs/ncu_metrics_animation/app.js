const baseMetrics = {
  duration: 1.8,
  sm: 82,
  memory: 36,
  dram: 28,
  occupancy: 64,
};

const chapterConfig = {
  metrics: {
    question: "这五个指标分别回答什么问题？",
    summary: "Duration 是最终结果，其余指标用于解释时间花在了哪里，以及 GPU 哪类资源最接近上限。",
    mistake: "不要把 SM Throughput、Memory Throughput 和 DRAM Throughput 相加，它们不是同一份预算的分项百分比。",
  },
  scenarios: {
    question: "一组指标如何共同指向某种瓶颈？",
    summary: "不能只看一个百分比。要比较计算、内存和 DRAM 谁最接近上限，再结合 Occupancy 与 Duration 判断。",
    mistake: "某个 Throughput 很高不代表代码一定优秀，它也可能意味着这个资源已经成为限制性能的瓶颈。",
  },
  workflow: {
    question: "面对一页 NCU 报告，应该按什么顺序看？",
    summary: "先确认 Duration 是否改善，再找最高压力资源，最后用 Occupancy 和更细指标解释为什么没有吃满。",
    mistake: "Occupancy 不是越高越好。达到足以隐藏延迟的水平后，继续提高可能不会降低 Duration。",
  },
};

const metricSteps = [
  {
    metric: "duration",
    label: "最终结果",
    title: "Duration：一次 kernel 执行了多久",
    code: "Duration 是结果，不是瓶颈原因",
    detail: "先比较相同工作量下的执行时间，再用其他指标解释为什么快或慢。",
    question: "这个 kernel 最终花了多少时间？",
    meaning: "在工作量和输入 shape 相同的前提下，Duration 越低通常越好。",
    warning: "不要用不同 shape、不同算法或不同精度的 Duration 直接下结论。",
    work: "完整 kernel",
    computeBusy: 5,
    memoryBusy: 1,
  },
  {
    metric: "sm",
    label: "计算资源",
    title: "SM Throughput：计算侧有多接近理论峰值",
    code: "高 SM Throughput → 计算资源压力大",
    detail: "它是 Speed Of Light 视角下的高层计算吞吐利用率，用于观察 SM 相关资源是否接近上限。",
    question: "SM 上的计算资源有多忙？",
    meaning: "接近 100% 表示计算侧某些资源接近理论最大吞吐，可能是 compute-bound。",
    warning: "高数值不代表每条指令都有效，也可能存在重复计算或低效指令。",
    work: "算术指令",
    computeBusy: 8,
    memoryBusy: 1,
  },
  {
    metric: "memory",
    label: "内存系统",
    title: "Memory Throughput：内存路径总体有多忙",
    code: "Memory ≠ 只有 DRAM",
    detail: "这是高层内存资源压力视角，可能涉及缓存、共享内存、内存管线和带宽等子资源。",
    question: "GPU 内存系统中，最忙的相关资源接近上限了吗？",
    meaning: "高数值说明内存侧压力大，应继续展开 Memory Workload Analysis 确认具体层级。",
    warning: "不要直接把它理解成显存带宽；DRAM 只是内存系统的一部分。",
    work: "内存请求",
    computeBusy: 3,
    memoryBusy: 3,
  },
  {
    metric: "dram",
    label: "片外显存",
    title: "DRAM Throughput：显存接口有多接近峰值",
    code: "高 DRAM → 片外带宽可能成为瓶颈",
    detail: "它聚焦 GPU 芯片外部的 device memory 流量，比 Memory Throughput 的范围更具体。",
    question: "数据往返显存的带宽是否接近硬件上限？",
    meaning: "高 DRAM 常见于读取/写回数据量大、复用不足或缓存命中较低的 kernel。",
    warning: "DRAM 低也不一定好，可能是访问零散、延迟受限或工作量太小。",
    work: "DRAM 数据",
    computeBusy: 2,
    memoryBusy: 3,
    dramBusy: true,
  },
  {
    metric: "occupancy",
    label: "Warp 驻留",
    title: "Achieved Occupancy：SM 实际驻留了多少活跃 Warp",
    code: "active warps ÷ maximum possible active warps",
    detail: "更多驻留 Warp 可以在一部分 Warp 等待数据时切换执行，从而隐藏延迟。",
    question: "硬件可容纳的 Warp 能力，有多少在运行时真正处于活跃状态？",
    meaning: "低 Occupancy 会降低隐藏延迟的能力，常受寄存器、Shared Memory、Block 配置或工作量限制。",
    warning: "高 Occupancy 不保证高性能；当计算或带宽已饱和时，再提高它未必有用。",
    work: "活跃 Warp",
    computeBusy: 6,
    memoryBusy: 2,
  },
  {
    metric: "all",
    label: "关系总结",
    title: "Duration 是结果，四类资源指标负责解释",
    code: "先看时间 → 再找最高压力 → 最后分析原因",
    detail: "单个百分比无法证明瓶颈。比较同一 kernel、同一输入下的指标组合和优化前后变化。",
    question: "为什么这个 kernel 花了这些时间？",
    meaning: "SM、Memory、DRAM 谁最高，提供第一条瓶颈线索；Occupancy 帮助判断延迟能否被隐藏。",
    warning: "若所有吞吐都低，继续看 Warp Stall、并行度、负载均衡和 kernel 是否太小。",
    work: "完整执行",
    computeBusy: 6,
    memoryBusy: 2,
  },
];

const scenarios = {
  compute: {
    title: "计算流水线接近饱和",
    tag: "Compute-bound",
    duration: 2.4,
    sm: 92,
    memory: 43,
    dram: 31,
    occupancy: 68,
    diagnosis: "主要受计算资源限制",
    detail: "SM Throughput 明显最高，Memory 和 DRAM 尚有余量。继续减少显存读取可能帮助有限。",
    chain: ["计算指令多", "SM 接近峰值", "Duration 受计算吞吐约束"],
    actions: ["使用 Tensor Core / 更合适的数据类型", "减少重复计算和低效指令", "检查指令流水线与依赖"],
    relationship: "高 SM Throughput + 较低 Memory/DRAM，通常说明算术资源比内存带宽更接近上限。",
  },
  dram: {
    title: "片外显存带宽接近饱和",
    tag: "DRAM-bound",
    duration: 3.1,
    sm: 47,
    memory: 91,
    dram: 94,
    occupancy: 72,
    diagnosis: "主要受 DRAM 带宽限制",
    detail: "Memory 和 DRAM 都很高，而 SM 明显较低。计算单元经常在等待片外数据。",
    chain: ["传输字节数大", "DRAM 接近峰值", "SM 等待数据，Duration 增长"],
    actions: ["使用 Shared Memory / Cache 提高复用", "确保合并访问并减少跨步访问", "融合算子、减少中间结果写回"],
    relationship: "Memory 与 DRAM 同时很高，通常说明压力确实到达片外显存，而不仅是片上缓存或共享内存。",
  },
  latency: {
    title: "资源没有吃满，但 Warp 不够隐藏等待",
    tag: "Latency-bound",
    duration: 4.2,
    sm: 34,
    memory: 39,
    dram: 27,
    occupancy: 22,
    diagnosis: "低 Occupancy 或并行度导致延迟暴露",
    detail: "所有 Throughput 都不高，Occupancy 又明显偏低，GPU 可能缺少可切换的 Warp。",
    chain: ["寄存器/Shared 用量大或 Grid 小", "活跃 Warp 少", "等待无法隐藏，资源空闲"],
    actions: ["检查 Registers per Thread 和 Shared Memory per Block", "调整 block size 或增加并行工作", "继续查看 Warp Stall 与负载均衡"],
    relationship: "低 Throughput + 低 Occupancy 更像延迟或并行度问题，而不是计算或带宽已经饱和。",
  },
  optimized: {
    title: "数据复用改善，时间明显下降",
    tag: "Optimized",
    duration: 1.25,
    sm: 78,
    memory: 66,
    dram: 48,
    occupancy: 61,
    diagnosis: "资源利用更均衡，Duration 下降",
    detail: "DRAM 压力下降、SM 利用增加，说明更多时间用于有效计算，且最终执行时间更短。",
    chain: ["减少 DRAM 往返", "片上复用增加", "有效计算比例提高，Duration 降低"],
    actions: ["确认 correctness 与不同 shape 的稳定性", "比较优化前后的实际字节数和指令数", "继续检查新的最高压力资源"],
    relationship: "优化成功的最终证据是 Duration 下降；Throughput 的变化用于解释数据流和资源利用为何改善。",
  },
};

const workflowSteps = [
  {
    label: "性能结果",
    title: "第一步：确认 Duration 是否真的下降",
    code: "same work + lower Duration = real speedup",
    detail: "比较时保持输入 shape、dtype、算法语义和测量条件一致。",
    question: "相同输入下，新版本的 Duration 是否下降？",
    explanation: "其他指标都是解释变量。若 Duration 没下降，就不能只凭某个百分比更高宣称优化成功。",
    decision: "1.80 ms → 1.25 ms：优化有效",
    before: "1.80 ms",
    after: "1.25 ms",
  },
  {
    label: "资源分类",
    title: "第二步：比较 SM Throughput 和 Memory Throughput",
    code: "max(SM, Memory) → first bottleneck clue",
    detail: "更接近峰值的一侧通常更值得优先展开分析。",
    question: "计算侧和内存侧，谁更接近硬件上限？",
    explanation: "SM 高倾向 compute-bound；Memory 高倾向 memory-bound；两者都低则需要调查延迟、并行度和调度。",
    decision: "SM 92% > Memory 43%：先查计算侧",
    before: "SM 92%",
    after: "Mem 43%",
  },
  {
    label: "内存拆解",
    title: "第三步：用 DRAM 判断压力是否来自片外显存",
    code: "Memory high + DRAM high → off-chip bandwidth",
    detail: "Memory 高但 DRAM 不高时，瓶颈可能位于 L1/L2、Shared Memory 或内存指令管线。",
    question: "Memory Throughput 高，是不是 DRAM 带宽造成的？",
    explanation: "DRAM 是更具体的片外层级。它能帮助避免把所有内存问题都归结为显存带宽。",
    decision: "Memory 91% + DRAM 94%：DRAM-bound 线索强",
    before: "Mem 91%",
    after: "DRAM 94%",
  },
  {
    label: "延迟隐藏",
    title: "第四步：用 Occupancy 解释为何资源没有吃满",
    code: "low throughput + low occupancy → latency exposed",
    detail: "Occupancy 低时，SM 缺少其他 Warp 来覆盖访存或指令等待。",
    question: "Throughput 都不高，是否因为活跃 Warp 太少？",
    explanation: "检查寄存器、Shared Memory、Block 大小和 Grid 规模；但不要把提高 Occupancy 当成最终目标。",
    decision: "Occupancy 22%：继续查资源限制与 Warp Stall",
    before: "Occ 22%",
    after: "目标：足以隐藏延迟",
  },
];

const stepsByChapter = {
  metrics: metricSteps,
  scenarios: Object.keys(scenarios).map((key) => ({ scenario: key })),
  workflow: workflowSteps,
};

const state = {
  chapter: "metrics",
  step: 0,
  scenario: "compute",
  playing: false,
  intervalId: null,
  intervalMs: 1050,
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
  mistake: document.querySelector("#common-mistake"),
  durationValue: document.querySelector("#duration-value"),
  durationBar: document.querySelector("#duration-bar"),
  smValue: document.querySelector("#sm-value"),
  memoryValue: document.querySelector("#memory-value"),
  dramValue: document.querySelector("#dram-value"),
  occupancyValue: document.querySelector("#occupancy-value"),
  workLabel: document.querySelector("#work-label"),
  storyArrow: document.querySelector("#story-arrow"),
  metricQuestion: document.querySelector("#metric-question"),
  metricMeaning: document.querySelector("#metric-meaning"),
  metricWarning: document.querySelector("#metric-warning"),
  scenarioTitle: document.querySelector("#scenario-title"),
  scenarioTag: document.querySelector("#scenario-tag"),
  scenarioSm: document.querySelector("#scenario-sm"),
  scenarioMemory: document.querySelector("#scenario-memory"),
  scenarioDram: document.querySelector("#scenario-dram"),
  scenarioOccupancy: document.querySelector("#scenario-occupancy"),
  scenarioDuration: document.querySelector("#scenario-duration"),
  diagnosisTitle: document.querySelector("#diagnosis-title"),
  diagnosisDetail: document.querySelector("#diagnosis-detail"),
  chainCause: document.querySelector("#chain-cause"),
  chainPressure: document.querySelector("#chain-pressure"),
  chainResult: document.querySelector("#chain-result"),
  scenarioActions: document.querySelector("#scenario-actions"),
  relationship: document.querySelector("#relationship-text"),
  workflowQuestion: document.querySelector("#workflow-question"),
  workflowExplanation: document.querySelector("#workflow-explanation"),
  workflowDecision: document.querySelector("#workflow-decision"),
  beforeValue: document.querySelector("#before-value"),
  afterValue: document.querySelector("#after-value"),
  playButton: document.querySelector("#play-button"),
  playIcon: document.querySelector("#play-icon"),
  speedInput: document.querySelector("#speed-input"),
  speedOutput: document.querySelector("#speed-output"),
};

function createStaticVisuals() {
  const workItems = document.querySelector("#work-items");
  for (let index = 0; index < 16; index += 1) {
    const item = document.createElement("span");
    item.className = "work-item";
    item.dataset.index = String(index);
    workItems.appendChild(item);
  }

  const computeLanes = document.querySelector("#compute-lanes");
  for (let index = 0; index < 8; index += 1) {
    const lane = document.createElement("span");
    lane.className = "unit-lane";
    lane.dataset.index = String(index);
    computeLanes.appendChild(lane);
  }

  const warps = document.querySelector("#scenario-warps");
  for (let index = 0; index < 32; index += 1) {
    const warp = document.createElement("span");
    warp.className = "warp-cell";
    warp.dataset.index = String(index);
    warps.appendChild(warp);
  }
}

function setGauge(name, value) {
  const gauge = document.querySelector(`[data-gauge="${name}"]`);
  gauge.style.setProperty("--value", `${value}%`);
  elements[`${name}Value`].textContent = `${value}%`;
}

function setDashboardValues(values) {
  elements.durationValue.textContent = values.duration.toFixed(2);
  elements.durationBar.style.width = `${Math.min(100, values.duration / 3.5 * 100)}%`;
  setGauge("sm", values.sm);
  setGauge("memory", values.memory);
  setGauge("dram", values.dram);
  setGauge("occupancy", values.occupancy);
}

function renderMetricStory(step) {
  document.querySelectorAll(".metric-card").forEach((card) => {
    card.classList.toggle("is-focus", step.metric === "all" || card.dataset.metric === step.metric);
  });

  elements.metricQuestion.textContent = step.question;
  elements.metricMeaning.textContent = step.meaning;
  elements.metricWarning.textContent = step.warning;
  elements.workLabel.textContent = step.work;
  elements.storyArrow.classList.add("is-active");

  document.querySelectorAll(".work-item").forEach((item, index) => {
    item.classList.toggle("is-active", index < 10);
  });
  document.querySelectorAll(".unit-lane").forEach((lane, index) => {
    lane.classList.toggle("is-busy", index < step.computeBusy);
  });
  document.querySelectorAll(".memory-pipe span").forEach((pipe, index) => {
    pipe.className = "";
    if (index < step.memoryBusy) pipe.classList.add("is-busy");
    if (step.dramBusy && index === 2) pipe.classList.add("is-dram-busy");
  });
  setDashboardValues(baseMetrics);
}

function renderScenario(scenarioKey) {
  const scenario = scenarios[scenarioKey];
  state.scenario = scenarioKey;
  elements.scenarioTitle.textContent = scenario.title;
  elements.scenarioTag.textContent = scenario.tag;
  elements.scenarioSm.textContent = `${scenario.sm}%`;
  elements.scenarioMemory.textContent = `${scenario.memory}%`;
  elements.scenarioDram.textContent = `${scenario.dram}%`;
  elements.scenarioOccupancy.textContent = `${scenario.occupancy}%`;
  elements.scenarioDuration.textContent = scenario.duration.toFixed(2);
  elements.diagnosisTitle.textContent = scenario.diagnosis;
  elements.diagnosisDetail.textContent = scenario.detail;
  elements.chainCause.textContent = scenario.chain[0];
  elements.chainPressure.textContent = scenario.chain[1];
  elements.chainResult.textContent = scenario.chain[2];
  elements.relationship.textContent = scenario.relationship;

  document.querySelector("#compute-runner").style.width = `${scenario.sm}%`;
  document.querySelector("#memory-runner").style.width = `${scenario.memory}%`;
  document.querySelector("#dram-runner").style.width = `${scenario.dram}%`;

  document.querySelectorAll(".warp-cell").forEach((warp, index) => {
    warp.classList.toggle("is-active", index < Math.round(scenario.occupancy / 100 * 32));
  });

  elements.scenarioActions.replaceChildren();
  scenario.actions.forEach((action) => {
    const item = document.createElement("li");
    item.textContent = action;
    elements.scenarioActions.appendChild(item);
  });

  document.querySelectorAll(".scenario-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.scenario === scenarioKey);
  });
}

function renderWorkflow(step, index) {
  elements.workflowQuestion.textContent = step.question;
  elements.workflowExplanation.textContent = step.explanation;
  elements.workflowDecision.textContent = step.decision;
  elements.beforeValue.textContent = step.before;
  elements.afterValue.textContent = step.after;

  document.querySelectorAll(".workflow-step").forEach((item, itemIndex) => {
    item.classList.toggle("is-active", itemIndex === index);
    item.classList.toggle("is-complete", itemIndex < index);
  });
}

function render() {
  const config = chapterConfig[state.chapter];
  const steps = stepsByChapter[state.chapter];
  const step = steps[state.step];

  elements.question.textContent = config.question;
  elements.summary.textContent = config.summary;
  elements.mistake.textContent = config.mistake;
  elements.stepCount.textContent = `${String(state.step + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
  elements.progress.style.width = `${((state.step + 1) / steps.length) * 100}%`;

  document.querySelectorAll(".chapter-tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.chapter === state.chapter);
  });
  document.querySelectorAll(".chapter-view").forEach((view) => {
    view.classList.toggle("is-active", view.id === `${state.chapter}-view`);
  });

  if (state.chapter === "metrics") {
    elements.phaseLabel.textContent = step.label;
    elements.phaseTitle.textContent = step.title;
    elements.conceptCode.textContent = step.code;
    elements.phaseDetail.textContent = step.detail;
    renderMetricStory(step);
  }

  if (state.chapter === "scenarios") {
    const scenarioKey = step.scenario;
    const scenario = scenarios[scenarioKey];
    elements.phaseLabel.textContent = "典型场景";
    elements.phaseTitle.textContent = scenario.title;
    elements.conceptCode.textContent = `${scenario.tag}: SM ${scenario.sm}% / Memory ${scenario.memory}% / DRAM ${scenario.dram}%`;
    elements.phaseDetail.textContent = scenario.detail;
    renderScenario(scenarioKey);
  }

  if (state.chapter === "workflow") {
    elements.phaseLabel.textContent = step.label;
    elements.phaseTitle.textContent = step.title;
    elements.conceptCode.textContent = step.code;
    elements.phaseDetail.textContent = step.detail;
    renderWorkflow(step, state.step);
  }
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

document.querySelectorAll(".chapter-tab").forEach((tab) => {
  tab.addEventListener("click", () => setChapter(tab.dataset.chapter));
});

document.querySelectorAll(".scenario-button").forEach((button) => {
  button.addEventListener("click", () => {
    stopPlayback();
    const keys = Object.keys(scenarios);
    state.step = keys.indexOf(button.dataset.scenario);
    render();
  });
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
  render();
});

elements.speedInput.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  state.intervalMs = 2150 - value;
  elements.speedOutput.textContent = `${(1050 / state.intervalMs).toFixed(1)}×`;
  if (state.playing) startPlayback();
});

createStaticVisuals();
render();
