const baseMetrics = {
  duration: 1.8,
  sm: 82,
  memory: 36,
  dram: 28,
  occupancy: 64,
  scheduler: 38,
  roofline: "Mem side",
  launch: "Regs",
};

const chapterConfig = {
  metrics: {
    question: "这八个关键指标分别回答什么问题？",
    summary: "Duration 是结果；其余七个指标分别解释计算、内存、并行度、调度、算术强度和启动配置。",
    mistake: "不要把八个指标当成互斥分类。它们是从不同层级观察同一次 kernel 执行。",
  },
  map: {
    question: "这些指标之间如何互相影响？",
    summary: "把报告里的八个指标放到同一张关系图里：启动配置影响并行度，资源压力影响 Duration，Roofline 决定下一步方向。",
    mistake: "不要看到某个百分比低就立刻优化它。低值可能是另一个资源先卡住，也可能是 kernel 太小或并行度不足。",
  },
  scenarios: {
    question: "一组指标如何共同指向某种瓶颈？",
    summary: "不能只看一个百分比。要比较计算、内存和 DRAM 谁最接近上限，再结合 Occupancy 与 Duration 判断。",
    mistake: "某个 Throughput 很高不代表代码一定优秀，它也可能意味着这个资源已经成为限制性能的瓶颈。",
  },
  workflow: {
    question: "面对一页 NCU 报告，应该按什么顺序看？",
    summary: "按八个指标逐层排查：结果、启动配置、计算、内存、DRAM、Occupancy、Scheduler/Stall、Roofline。",
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
    metric: "scheduler",
    label: "调度健康",
    title: "Scheduler / Stall：Warp 是否真的能发射指令",
    code: "eligible warps + stall reasons → issue health",
    detail: "Scheduler 每周期从 eligible warps 中选择 warp 发射指令；eligible 少或 stall 高时，资源可能空着等。",
    question: "活跃 Warp 里，有多少已经准备好发射下一条指令？",
    meaning: "Eligible Warps 偏低、Issue Slots 空着、某类 Stall 偏高时，继续用 Warp State / Source Counters 找等待来源。",
    warning: "只在 scheduler 发不满时优先看 Stall；如果 issue 已经很满，某个 Stall 高未必是主瓶颈。",
    work: "调度槽",
    computeBusy: 3,
    memoryBusy: 2,
  },
  {
    metric: "roofline",
    label: "理论边界",
    title: "Roofline：算术强度把 kernel 推向哪一侧",
    code: "arithmetic intensity = FLOPs / bytes",
    detail: "Roofline 把 achieved FLOP/s 和 FLOPs/byte 放到同一张图里，判断更像 memory-bound 还是 compute-bound。",
    question: "当前性能点离内存带宽边界近，还是离计算峰值边界近？",
    meaning: "Memory side 优先提高数据复用、减少写回；Compute side 优先看 Tensor Core、数据类型、指令 mix 和依赖。",
    warning: "Roofline 给优化方向，不替代真实 Duration、正确性测试和源码级瓶颈定位。",
    work: "FLOPs / bytes",
    computeBusy: 5,
    memoryBusy: 3,
    dramBusy: true,
  },
  {
    metric: "launch",
    label: "启动配置",
    title: "Launch Stats：grid、block 和资源用量设定上限",
    code: "grid/block + registers + shared memory → occupancy limit",
    detail: "Launch Stats 解释 kernel 的并行规模和每个 block 消耗的硬件资源，决定理论 occupancy 和可用并行度上限。",
    question: "这次 kernel launch 是否给了 GPU 足够工作量，且没有被寄存器或 Shared Memory 卡住？",
    meaning: "Grid 太小会让 GPU 吃不满；寄存器或 Shared Memory 太多会减少每个 SM 能驻留的 block/warp。",
    warning: "不要只调 block size。每次修改都要回看 Duration、Occupancy、Scheduler 和新的最高压力资源。",
    work: "Grid / Block",
    computeBusy: 4,
    memoryBusy: 1,
  },
];

const mapSteps = [
  {
    reportRow: "duration",
    label: "报告入口",
    title: "Duration 是最终账单",
    tag: "Result",
    code: "gpu__time_duration.sum -> 先确认快慢",
    detail: "Duration 只回答最终花了多久；它不解释原因，但所有优化判断都要回到它。",
    signal: "先把 Duration 当成最终结果",
    inference: "相同输入、相同 dtype、相同语义下，Duration 下降才是有效优化。其他百分比负责解释为什么。",
    next: "Launch Stats 是否限制并行度",
    trap: "吞吐百分比不能相加，也不能脱离 Duration 单独评价。",
    nodes: ["duration"],
    primaryNodes: ["duration"],
    edges: [],
    metricName: "gpu__time_duration.sum",
    tokens: [
      ["unit", "gpu"],
      ["counter", "time_duration"],
      ["rollup", "sum"],
      ["meaning", "kernel elapsed time"],
    ],
  },
  {
    reportRow: "launch",
    label: "启动配置",
    title: "Launch Stats 先解释并行度上限",
    tag: "Launch",
    code: "launch__registers_per_thread / launch__shared_mem_per_block",
    detail: "Launch Stats 给出 grid、block、寄存器和 Shared Memory 等静态配置，先判断是否存在并行规模或资源上限问题。",
    signal: "寄存器或 Shared Memory 限制理论 Occupancy",
    inference: "如果 launch 本身限制并行度，后面的吞吐指标可能全部偏低。",
    next: "Theoretical Occupancy 与 Achieved Occupancy",
    trap: "不要只看 block size；per-thread registers 和 per-block shared memory 同样会改变驻留能力。",
    nodes: ["launch", "occupancy", "duration"],
    primaryNodes: ["launch"],
    edges: ["launch-occupancy"],
    metricName: "launch__registers_per_thread / launch__shared_mem_per_block",
    tokens: [
      ["prefix", "launch"],
      ["resource", "registers / shared memory"],
      ["scope", "per thread / per block"],
      ["effect", "occupancy limit"],
    ],
  },
  {
    reportRow: "sm",
    label: "Speed Of Light",
    title: "SM Throughput 找计算侧压力",
    tag: "Compute SOL",
    code: "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    detail: "Speed Of Light 的百分比表示相对理论峰值的利用率。SM 高于 Memory 时，先怀疑计算流水线、指令 mix 或依赖。",
    signal: "SM Throughput 明显高于 Memory",
    inference: "计算资源更接近上限，继续减少 DRAM 访问可能不是第一优先级。",
    next: "Compute Workload Analysis / Instruction Stats",
    trap: "SM 高不等于代码高效；重复计算、低效数据类型和管线偏科也会把 SM 打满。",
    nodes: ["instructions", "sm", "compute-bound", "duration"],
    primaryNodes: ["sm"],
    edges: ["instructions-sm", "sm-duration", "compute-bound-duration"],
    metricName: "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    tokens: [
      ["unit", "sm"],
      ["throughput", "throughput"],
      ["rollup", "avg"],
      ["submetric", "pct_of_peak_sustained_elapsed"],
    ],
  },
  {
    reportRow: "memory",
    label: "内存系统",
    title: "Memory Throughput 找内存侧总体压力",
    tag: "Memory SOL",
    code: "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    detail: "Memory Throughput 是内存系统的高层压力信号。它先告诉你内存侧是否值得展开，而不是直接等于 DRAM 带宽。",
    signal: "Memory 高于 SM 或接近峰值",
    inference: "先展开 Memory Workload Analysis，再区分压力来自 L1/TEX、L2、Shared、DRAM 或内存指令管线。",
    next: "DRAM Throughput 与 Memory Chart",
    trap: "不要把 Memory Throughput 直接解释成显存带宽；DRAM 只是其中一层。",
    nodes: ["requests", "hierarchy", "dram", "duration"],
    primaryNodes: ["hierarchy"],
    edges: ["requests-hierarchy", "hierarchy-dram", "dram-duration"],
    metricName: "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    tokens: [
      ["unit", "gpu"],
      ["quantity", "compute_memory_throughput"],
      ["rollup", "avg"],
      ["submetric", "pct_of_peak_sustained_elapsed"],
    ],
  },
  {
    reportRow: "dram",
    label: "内存层级",
    title: "DRAM Throughput 定位片外显存压力",
    tag: "Memory Path",
    code: "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    detail: "Memory Throughput 高时继续拆 L1/TEX、L2、Shared 和 DRAM。只有 DRAM 也高，才更像片外显存带宽瓶颈。",
    signal: "Memory 与 DRAM 同时偏高",
    inference: "数据已经冲到片外显存上限附近，优先减少字节数、提升复用和合并访问。",
    next: "Memory Workload Analysis / Memory Chart",
    trap: "Memory 高但 DRAM 不高时，瓶颈可能在 L1/L2、Shared Memory、内存指令管线或访问形态。",
    nodes: ["requests", "hierarchy", "dram", "duration"],
    primaryNodes: ["dram"],
    edges: ["requests-hierarchy", "hierarchy-dram", "dram-duration"],
    metricName: "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    tokens: [
      ["unit", "dram"],
      ["throughput", "throughput"],
      ["rollup", "avg"],
      ["submetric", "pct_of_peak_sustained_elapsed"],
    ],
  },
  {
    reportRow: "occupancy",
    label: "延迟隐藏",
    title: "Occupancy 解释为什么资源没吃满",
    tag: "Latency Hiding",
    code: "sm__warps_active.avg.pct_of_peak_sustained_active",
    detail: "低 Occupancy 会减少可切换 Warp，访存或依赖等待更容易暴露；但 Occupancy 足够后继续提高未必降低 Duration。",
    signal: "Throughput 都不高，同时 Occupancy 低",
    inference: "资源不是被算力或带宽打满，而是缺少足够并行 Warp 来覆盖等待。",
    next: "Launch Stats: registers / smem / block size / grid size",
    trap: "不要把 Occupancy 当 KPI。高 Occupancy 可能仍然慢，低 Occupancy 也可能已经足够。",
    nodes: ["launch", "occupancy", "scheduler", "duration"],
    primaryNodes: ["occupancy"],
    edges: ["launch-occupancy", "occupancy-scheduler", "scheduler-duration"],
    metricName: "sm__warps_active.avg.pct_of_peak_sustained_active",
    tokens: [
      ["unit", "sm"],
      ["counter", "warps_active"],
      ["rollup", "avg"],
      ["submetric", "pct_of_peak_sustained_active"],
    ],
  },
  {
    reportRow: "scheduler",
    label: "调度原因",
    title: "Warp Stall 告诉你等待卡在哪里",
    tag: "Warp Stall",
    code: "smsp__warpidsamp_warps_issue_stalled_long_scoreboard_not_issued",
    detail: "只有当 scheduler 发不出指令时，stall 才是优先线索。long scoreboard 常指向等待 L1TEX 相关数据依赖。",
    signal: "Eligible Warps 少，long scoreboard 高",
    inference: "SM 不是没有能力，而是在等数据或依赖。继续查产生等待的源码行和内存访问形态。",
    next: "Source Counters / Warp State Stats / Memory tables",
    trap: "看到某个 stall 高时，先确认 scheduler 是否真的 issue 不满；否则它可能不是主瓶颈。",
    nodes: ["launch", "occupancy", "scheduler", "requests", "hierarchy", "duration"],
    primaryNodes: ["scheduler"],
    edges: ["launch-occupancy", "occupancy-scheduler", "scheduler-duration", "requests-hierarchy"],
    metricName: "smsp__warpidsamp_warps_issue_stalled_long_scoreboard_not_issued",
    tokens: [
      ["unit", "smsp"],
      ["sampler", "warpidsamp"],
      ["state", "long_scoreboard"],
      ["scope", "not_issued"],
    ],
  },
  {
    reportRow: "roofline",
    label: "Roofline",
    title: "Roofline 把优化方向压成一张图",
    tag: "Bound Model",
    code: "arithmetic intensity = FLOPs / bytes",
    detail: "Roofline 通过算术强度判断当前点更靠近内存带宽边界还是计算峰值边界，再决定提高复用还是提升计算吞吐。",
    signal: "点落在 memory side 或 compute side",
    inference: "memory side 先提高复用、减少写回；compute side 先看 Tensor Core、指令 mix 和流水线依赖。",
    next: "优化后回到 Duration 与新最高压力资源",
    trap: "Roofline 给方向，不替代 correctness、实际 Duration 和具体源码级诊断。",
    nodes: ["bytes", "roofline", "action", "duration"],
    primaryNodes: ["roofline"],
    edges: ["bytes-roofline", "roofline-action", "action-duration"],
    metricName: "ncu roofline: FLOPs / bytes + achieved FLOP/s",
    tokens: [
      ["x-axis", "arithmetic intensity"],
      ["y-axis", "achieved performance"],
      ["memory", "bandwidth boundary"],
      ["compute", "peak performance boundary"],
    ],
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
    launch: "enough grid",
    scheduler: "math pipe",
    roofline: "compute side",
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
    launch: "enough grid",
    scheduler: "long scoreboard",
    roofline: "memory side",
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
    launch: "regs limit",
    scheduler: "eligible low",
    roofline: "below roofs",
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
    launch: "balanced",
    scheduler: "issue healthier",
    roofline: "closer roof",
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
    label: "启动配置",
    title: "第二步：用 Launch Stats 确认并行规模",
    code: "grid/block + regs/smem → theoretical occupancy",
    detail: "先排除工作量太小、block 配置不合适、寄存器或 Shared Memory 限制并行度的问题。",
    question: "这次 launch 是否足够大，且没有被 per-block 资源卡住？",
    explanation: "如果 grid 太小或每个 block 消耗资源太多，后面的吞吐指标可能都偏低。",
    decision: "Registers 是 occupancy limiter：先记录上限",
    before: "block 256",
    after: "limit: regs",
  },
  {
    label: "计算压力",
    title: "第三步：看 SM Throughput",
    code: "sm__throughput → compute pressure",
    detail: "SM Throughput 接近峰值时，计算侧资源更可能是当前限制因素。",
    question: "SM 相关资源是否已经接近理论峰值？",
    explanation: "SM 高时继续展开 Compute Workload Analysis、Instruction Stats、Tensor Core 使用和依赖链。",
    decision: "SM 92%：计算侧线索强",
    before: "SM 92%",
    after: "Compute",
  },
  {
    label: "内存压力",
    title: "第四步：看 Memory Throughput",
    code: "gpu__compute_memory_throughput → memory pressure",
    detail: "Memory Throughput 是高层内存系统压力，不等同于 DRAM。",
    question: "内存系统总体是否比计算侧更接近上限？",
    explanation: "Memory 高时继续拆 L1/TEX、L2、Shared Memory、DRAM 和内存管线。",
    decision: "Memory 43%：不是第一瓶颈",
    before: "SM 92%",
    after: "Mem 43%",
  },
  {
    label: "内存拆解",
    title: "第五步：用 DRAM 判断是否来自片外显存",
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
    title: "第六步：用 Occupancy 解释低吞吐",
    code: "low throughput + low occupancy → latency exposed",
    detail: "Occupancy 低时，SM 缺少其他 Warp 来覆盖访存或指令等待。",
    question: "Throughput 都不高，是否因为活跃 Warp 太少？",
    explanation: "检查寄存器、Shared Memory、Block 大小和 Grid 规模；但不要把提高 Occupancy 当成最终目标。",
    decision: "Occupancy 22%：继续查资源限制与 Warp Stall",
    before: "Occ 22%",
    after: "目标：足以隐藏延迟",
  },
  {
    label: "调度原因",
    title: "第七步：看 Scheduler / Stall",
    code: "eligible warps + stall reasons → issue health",
    detail: "当 scheduler 发不满时，Stall reasons 才是定位等待来源的优先线索。",
    question: "Warp 是不够 eligible，还是被某类依赖长期卡住？",
    explanation: "Long scoreboard 指向等待 L1TEX 相关数据依赖；barrier、mio throttle、math pipe throttle 指向不同源码改法。",
    decision: "Eligible 少 + long scoreboard 高：查访存依赖",
    before: "eligible low",
    after: "long scoreboard",
  },
  {
    label: "优化方向",
    title: "第八步：用 Roofline 决定下一刀",
    code: "FLOPs / bytes + achieved FLOP/s → bound side",
    detail: "Roofline 把性能点放在内存带宽边界和计算峰值边界之间，用来选择下一类优化动作。",
    question: "应该优先提高数据复用、减少字节数，还是提高计算吞吐？",
    explanation: "Memory side 先做复用和合并访问；Compute side 先看 Tensor Core、数据类型、指令 mix 和依赖。",
    decision: "Memory side：先提升复用并减少写回",
    before: "AI 2.1",
    after: "Mem side",
  },
];

const stepsByChapter = {
  metrics: metricSteps,
  map: mapSteps,
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
  schedulerValue: document.querySelector("#scheduler-value"),
  rooflineValue: document.querySelector("#roofline-value"),
  launchValue: document.querySelector("#launch-value"),
  workLabel: document.querySelector("#work-label"),
  storyArrow: document.querySelector("#story-arrow"),
  metricQuestion: document.querySelector("#metric-question"),
  metricMeaning: document.querySelector("#metric-meaning"),
  metricWarning: document.querySelector("#metric-warning"),
  mapTitle: document.querySelector("#map-title"),
  mapTag: document.querySelector("#map-tag"),
  mapSignal: document.querySelector("#map-signal"),
  mapInference: document.querySelector("#map-inference"),
  mapNext: document.querySelector("#map-next"),
  mapTrap: document.querySelector("#map-trap"),
  metricName: document.querySelector("#metric-name"),
  metricTokenRow: document.querySelector("#metric-token-row"),
  scenarioTitle: document.querySelector("#scenario-title"),
  scenarioTag: document.querySelector("#scenario-tag"),
  scenarioSm: document.querySelector("#scenario-sm"),
  scenarioMemory: document.querySelector("#scenario-memory"),
  scenarioDram: document.querySelector("#scenario-dram"),
  scenarioOccupancy: document.querySelector("#scenario-occupancy"),
  scenarioLaunch: document.querySelector("#scenario-launch"),
  scenarioScheduler: document.querySelector("#scenario-scheduler"),
  scenarioRoofline: document.querySelector("#scenario-roofline"),
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
  setGauge("scheduler", values.scheduler);
  elements.rooflineValue.textContent = values.roofline;
  elements.launchValue.textContent = values.launch;
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

function renderMap(step) {
  elements.mapTitle.textContent = step.title;
  elements.mapTag.textContent = step.tag;
  elements.mapSignal.textContent = step.signal;
  elements.mapInference.textContent = step.inference;
  elements.mapNext.textContent = step.next;
  elements.mapTrap.textContent = step.trap;
  elements.metricName.textContent = step.metricName;

  document.querySelectorAll(".report-row").forEach((row) => {
    row.classList.toggle("is-active", row.dataset.reportRow === step.reportRow);
  });

  document.querySelectorAll(".relation-node").forEach((node) => {
    const isActive = step.nodes.includes(node.dataset.node);
    const isPrimary = step.primaryNodes.includes(node.dataset.node);
    node.classList.toggle("is-active", isActive);
    node.classList.toggle("is-primary", isPrimary);
  });

  document.querySelectorAll(".relation-arrow").forEach((edge) => {
    edge.classList.toggle("is-active", step.edges.includes(edge.dataset.edge));
  });

  elements.metricTokenRow.replaceChildren();
  step.tokens.forEach(([label, value]) => {
    const token = document.createElement("div");
    token.className = "metric-token";

    const tokenLabel = document.createElement("span");
    tokenLabel.textContent = label;

    const tokenValue = document.createElement("small");
    tokenValue.textContent = value;

    token.append(tokenLabel, tokenValue);
    elements.metricTokenRow.appendChild(token);
  });
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
  elements.scenarioLaunch.textContent = scenario.launch;
  elements.scenarioScheduler.textContent = scenario.scheduler;
  elements.scenarioRoofline.textContent = scenario.roofline;
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

  if (state.chapter === "map") {
    elements.phaseLabel.textContent = step.label;
    elements.phaseTitle.textContent = step.title;
    elements.conceptCode.textContent = step.code;
    elements.phaseDetail.textContent = step.detail;
    renderMap(step);
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

document.querySelectorAll(".report-row").forEach((row) => {
  row.addEventListener("click", () => {
    stopPlayback();
    state.chapter = "map";
    const nextIndex = mapSteps.findIndex((step) => step.reportRow === row.dataset.reportRow);
    state.step = Math.max(0, nextIndex);
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
