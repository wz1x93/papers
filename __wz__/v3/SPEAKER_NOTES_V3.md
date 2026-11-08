# V3｜逐页讲稿

19 页主讲 + 6 页附录；新增行业与竞争背景在第 3–7 页。主讲约 18 分钟，可按评审时长略讲技术附录。

## 01｜精度与位置， 一起决定推理效率

建议 40 秒。

这不是三个模块分别压缩后再做分层。本项目把精度与驻留作为耦合决策，在相同资源、模型质量与服务时限下，最大化有效吞吐。今天申请一个六个月的孵化验证窗口，检验相对强基线能否获得超过 15% 的净增益。


## 02｜把内存组合，变成可验证的系统优势

建议 50 秒。

评审首先要看到投资命题。海力士已经有跨层产品能力，我们要验证的是连接这些产品的工作负载感知决策。15% 不是市场宣传数字，也不是把不同论文的加速比相加，而是同约束强基线上的项目目标。

- [S02] SK hynix 2026 Future Forum · 2026-09-09
  https://news.skhynix.com/en/future-forum-2026/
- [S03] SK hynix at OCP Global Summit 2026 · 2026-10-16
  https://news.skhynix.com/en/ocp-global-summit-2026/

## 03｜行业变化：模型能力扩展，带来更多状态与权重管理

建议 65 秒。

这一页提供行业背景，不预测市场份额或所有模型的路线。NVIDIA 的 Nemotron 3 Ultra 报告了 1M 上下文、550B 总参数与 55B 激活参数。它说明计算激活规模和需要保存的模型容量并不相同。Mooncake 与 Dynamo 则把状态复用和跨层移动提升到服务系统层面。数字属于特定模型，不能外推为行业平均。

- [S12] Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning · 2026-06-12
  https://arxiv.org/abs/2606.15007
- [S21] NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models · 2025-03-18
  https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- [S22] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving · v1 2024-06-24；v5 2026-10-08
  https://arxiv.org/abs/2407.00079

## 04｜跨层状态管理，已经进入服务架构与产品交付

建议 65 秒。

这三条不是可直接比较的性能测试，而是不同层次的行业落地信号。Mooncake 以生产服务背景描述多层 KV；NVIDIA 官方博客把 KV 管理和 NIXL 纳入 Dynamo；美光 2026 年 9 月在产品要点中自述 SSD 已向 KV-cache 客户出货。平台和客户已经在处理跨层状态，项目应验证如何在已有生态上进一步联合精度与驻留。

- [S20] Micron Technology, Inc. Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results · 2026-09-30
  https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results
- [S21] NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models · 2025-03-18
  https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- [S22] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving · v1 2024-06-24；v5 2026-10-08
  https://arxiv.org/abs/2407.00079

## 05｜三星：软件、器件、认证与协同研发连续推进

建议 75 秒。

三星是本次重点竞争对手。2021 年 SMDK 软件公告包含智能分层引擎，当前公开仓库提供了代码层证据；2023 年 CXL 2.0 为历史开发里程碑；2024 年三星宣布 Red Hat 认证的 CXL 基础设施与 CMM-D 验证。2026 年 zHBM 文章明确写 currently under development。它们证明了投入方向，不证明三星已经实现与本提案相同的三对象联合优化，也不支持任何现成的跨厂商性能排名。

- [S14] Samsung Introduces Industry’s First Open-source Software Solution for CXL Memory Platform · 2021-10-07
  https://news.samsungsemiconductor.com/global/samsung-introduces-industrys-first-open-source-software-solution-for-cxl-memory-platform/
- [S15] Samsung Electronics Announces Successful Building of Industry-First CXL Infrastructure Certified by Red Hat · 2024-06-25
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-successful-building-of-industry-first-cxl-infrastructure-certified-by-red-hat/
- [S16] Part 2. zHBM: Next-Generation Memory for the Data Movement Era · 2026-10-29
  https://news.samsungsemiconductor.com/global/part-2-zhbm-next-generation-memory-for-the-data-movement-era/
- [S17] Samsung Electronics Announces Third Quarter 2026 Results · 2026-10-29
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-third-quarter-2026-results/
- [S23] Samsung Develops Industry’s First CXL DRAM Supporting CXL 2.0 · 2023-05-12
  https://news.samsungsemiconductor.com/global/samsung-develops-industrys-first-cxl-dram-supporting-cxl-2-0/

## 06｜竞争维度扩展：产品组合与系统协同都值得比较

建议 70 秒。

这页并列公开行动，不能用单个公告打出领先落后的分数。美光 3 月官方宣布 HBM4、Gen6 SSD 和 SOCAMM2 高量产，并联系 Vera Rubin 与 BlueField-4 STX；9 月产品要点自述 KV-cache SSD 出货。海力士 10 月展示 MARS、CXL池化及混合内存。三星组合证据包括软件认证与协同研发。三家都在建设系统能力，因此提案需要以端到端验证和设备接口建立自身价值，不能把硬件产品齐全本身当作护城河。

- [S02] SK hynix 2026 Future Forum · 2026-09-09
  https://news.skhynix.com/en/future-forum-2026/
- [S03] SK hynix at OCP Global Summit 2026 · 2026-10-16
  https://news.skhynix.com/en/ocp-global-summit-2026/
- [S15] Samsung Electronics Announces Successful Building of Industry-First CXL Infrastructure Certified by Red Hat · 2024-06-25
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-successful-building-of-industry-first-cxl-infrastructure-certified-by-red-hat/
- [S16] Part 2. zHBM: Next-Generation Memory for the Data Movement Era · 2026-10-29
  https://news.samsungsemiconductor.com/global/part-2-zhbm-next-generation-memory-for-the-data-movement-era/
- [S17] Samsung Electronics Announces Third Quarter 2026 Results · 2026-10-29
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-third-quarter-2026-results/
- [S18] OpenMPDK / SMDK — Scalable Memory Development Kit, README · 访问 2026-11-08；持续更新仓库
  https://github.com/OpenMPDK/SMDK
- [S19] Micron in High-Volume Production of HBM4 Designed for NVIDIA Vera Rubin, PCIe Gen6 SSD and SOCAMM2 · 2026-03-16
  https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx
- [S20] Micron Technology, Inc. Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results · 2026-09-30
  https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results

## 07｜项目必要性：把更多硬件选择，变成客户可测的收益

建议 65 秒。

必要性来自可验证的经营与技术问题。需求与硬件选项越来越多，如果没有工作负载和质量成本画像，就难以向客户量化不同配置的价值。竞争公开投入软件和验证能力，是应加快试验的信号，但不意味着这个具体优化必然有收益。右侧属于提案的战略判断，最终仍由六个月强基线实验决定继续与否。

- [S03] SK hynix at OCP Global Summit 2026 · 2026-10-16
  https://news.skhynix.com/en/ocp-global-summit-2026/
- [S12] Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning · 2026-06-12
  https://arxiv.org/abs/2606.15007
- [S15] Samsung Electronics Announces Successful Building of Industry-First CXL Infrastructure Certified by Red Hat · 2024-06-25
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-successful-building-of-industry-first-cxl-infrastructure-certified-by-red-hat/
- [S16] Part 2. zHBM: Next-Generation Memory for the Data Movement Era · 2026-10-29
  https://news.samsungsemiconductor.com/global/part-2-zhbm-next-generation-memory-for-the-data-movement-era/
- [S18] OpenMPDK / SMDK — Scalable Memory Development Kit, README · 访问 2026-11-08；持续更新仓库
  https://github.com/OpenMPDK/SMDK
- [S19] Micron in High-Volume Production of HBM4 Designed for NVIDIA Vera Rubin, PCIe Gen6 SSD and SOCAMM2 · 2026-03-16
  https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx
- [S20] Micron Technology, Inc. Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results · 2026-09-30
  https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results
- [S21] NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models · 2025-03-18
  https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- [S22] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving · v1 2024-06-24；v5 2026-10-08
  https://arxiv.org/abs/2407.00079

## 08｜三类数据，三种访问规律，一个 HBM 预算

建议 55 秒。

三类对象对应模型不同模块，但共享同一硬件预算。我们保留它们的语义差异：旧 KV 不等于冷数据，状态量化误差不能简单恢复，专家热度也不等于质量敏感性。统一优化的是稀缺资源如何分配，而不是用同一规则处理所有对象。

- [S01] NOVA: Technology-Architecture Co-Design of Near-Memory Processing for Attention-SSM-MoE Hybrid LLM Inference · 2026-08-23
  https://arxiv.org/abs/2608.22613
- [S10] ReplaySSM: Cache SSM Inputs, Not State · 2026-10-09
  https://arxiv.org/abs/2610.13778
- [S13] PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States · 2026-10-21
  https://arxiv.org/abs/2610.29545

## 09｜两次局部最优，不保证组合后的系统最优

建议 55 秒。

本提案的关键不是执行顺序换一下，而是改变优化目标。先确定精度再做分层，会把精度释放的容量和其他对象的驻留收益割裂。强交替优化可能弥补一部分，因此它必须是主要基线。最终差异能否保留是本项目要证伪或支持的假设。


## 10｜两条非线性，共同决定可行与最优的边界

建议 55 秒。

双重非线性具体指质量响应和容量性能阈值。左图说明位宽下降可能先影响很小，再越过质量边界。右图说明装下一个工作集后可能减少整组缺失。两个概念图不携带数据，也没有优化器跑出来的曲面。正式模型可以用离散查表和分段近似，不必预先承诺非线性求解器。


## 11｜压缩 KV 或状态，收益可能出现在专家执行

建议 55 秒。

这个算例用同一 80GB 预算展示资源替代。上下两行总量相同，只是状态与 KV 假设各释放 5GB，使专家驻留增加 10GB。这里不能推断吞吐或无损质量。要测的正是驻留改变带来的收益，能否覆盖量化与执行的全部成本。


## 12｜用一个资源管理器，连接模型需求与真实数据路径

建议 60 秒。

架构图明确了普通路径。GPU 与 HBM在一侧，主机与 DDR 中转、CXL 和 SSD 在另一侧。具体平台支持 GPU DMA 或直接访问时可以替换路径，但不能从 CXL 标准推断 GPU 一定可直接 load。离线做质量和平台成本画像，在线管理器执行轻量的预算、驻留、迁移与预取。


## 13｜目标只有一个：约束内的有效吞吐最大化

建议 55 秒。

Goodput 定义为满足事先确定的质量和时限要求的输出 token 每秒。q 是精度，p 包括放置和副本，π 处理运行时实现。副本、元数据与临时缓冲都计入预算。质量误差不预设可按对象相加，最终必须端到端验证。能耗和成本是副指标，不能混用 15%。


## 14｜离线找边界，在线做有界开销的联合决策

建议 55 秒。

这里是拟议算法路线，不是已经完成的算法。先用小规模离线优化得到可解释的空间，再选查表、贪心或有界搜索等在线近似。在线不能每个 token 重做大规模优化。重规划要有最小驻留期和收益门槛，防止负载漂移导致反复搬迁。控制和数据转换成本完整计入。


## 15｜已有局部组合；待证明的是跨对象的额外价值

建议 60 秒。

新颖性需要建立在可验证机制上。专家、KV和跨层精度已有直接先例。我们要证明统一模型与跨对象分配确实比强交替优化有额外收益，并能实现。若额外收益不存在，就不能坚持强创新主张。这里的选取不是对全部文献的穷尽调查。

- [S04] HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference · 2024-11
  https://arxiv.org/abs/2411.01433
- [S05] TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization · 2025-05
  https://arxiv.org/abs/2505.19586
- [S06] Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems · 2025-12-04
  https://arxiv.org/abs/2512.04476
- [S07] SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference · v1 2025-12; v5 2026-09
  https://arxiv.org/abs/2512.12990
- [S08] ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories · 2026-06
  https://arxiv.org/abs/2606.12556
- [S09] A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache) · 2026-07; revised 2026-08
  https://arxiv.org/abs/2607.18141
- [S10] ReplaySSM: Cache SSM Inputs, Not State · 2026-10-09
  https://arxiv.org/abs/2610.13778
- [S11] StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference · 2026-11-05
  https://arxiv.org/abs/2611.08363

## 16｜15%+，相对最强可复现基线，而不是静态分层

建议 60 秒。

15% 相对最强可复现的匹配基线。至少包含只优化精度、只优化放置、双顺序、充分调优的交替优化以及相关方法的适用组合。模型与内核不兼容的差异需要公开。建议主测试集合覆盖预登记的容量受限场景，逐场景配对比较，报告几何均值及 95% CI，完整报告其他场景。


## 17｜六个月，用三道证据门决定是否继续

建议 55 秒。

G0判断问题与空间是否成立；上界不足时应及时调整。G1验证机制与原型，并与强基线比较。G2上真实设备和动态负载。主目标不成立时可以收缩为局部专项，不以其他产出来替代主 KPI。通过后再申请更大规模产品协同投入。


## 18｜三年沉淀：从优化策略，走向产品组合的系统能力

建议 50 秒。

战略资产不是某个模型专用的一组参数，而是可校准的质量成本模型、设备接口和客户验证能力。第一年验证机制，第二年协同接口，第三年形成参考方案。客户减少 HBM 使用并不自动等于海力士利润增加，企业收益需要看采用率、组合收入和维护成本。

- [S02] SK hynix 2026 Future Forum · 2026-09-09
  https://news.skhynix.com/en/future-forum-2026/
- [S03] SK hynix at OCP Global Summit 2026 · 2026-10-16
  https://news.skhynix.com/en/ocp-global-summit-2026/

## 19｜批准六个月孵化，验证联合优化的系统价值

建议 40 秒。

最后明确本轮申请：六个月，建议四至五人，有限真实测试平台与业务技术对接。资源配置是提案，不是已批准预算。项目坚持在质量、服务与资源约束下验证净增益，并通过阶段门决定继续、收缩或停止。请评审批准验证窗口，而不是要求提前相信尚未产生的实验结果。


## 20｜附录 / 实验协议与归因设计

建议 55 秒。

协议在 G0 锁定，避免事后选择有利场景。模型需要验证内核与上下文支持。建议均值至少 15%，并对收益方向提供统计证据；CI 下界高于 1.00 不代表真实收益的下界超过 15%。质量阈值按任务预登记，完整报告未通过情形。


## 21｜附录 / 哪些配置必须从可行域排除

建议 50 秒。

统一优化首先需要正确的可行域。SSM 不可逆误差、CXL 可达性和 SSD 访问时限决定某些组合根本不能使用。量化副本也可能抵消容量节省，不能只按一个理想低位宽副本算账。

- [S05] TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization · 2025-05
  https://arxiv.org/abs/2505.19586
- [S06] Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems · 2025-12-04
  https://arxiv.org/abs/2512.04476
- [S07] SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference · v1 2025-12; v5 2026-09
  https://arxiv.org/abs/2512.12990
- [S10] ReplaySSM: Cache SSM Inputs, Not State · 2026-10-09
  https://arxiv.org/abs/2610.13778
- [S13] PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States · 2026-10-21
  https://arxiv.org/abs/2610.29545

## 22｜附录 / 超过 15% 的必要条件是什么

建议 45 秒。

f 是未被计算重叠掩盖的搬运等待。两个例子分别得 17% 和 2%，说明优化空间必须先测量。延迟、吞吐和生产有效吞吐不能直接等同。单请求固定串行场景下 17% 延迟下降可换算约 20.5% 速率提升，但这里不拿它预测服务 goodput。


## 23｜附录 / 研究依据与证据边界

建议 25 秒。

这些公开来源支持战略匹配和研究背景。企业展示不证明本项目已获认可；仿真或原型的论文结果也不直接外推。完整来源索引沿用 SOURCES.md，创新性仍需系统查新和实验证明。

- [S01] NOVA: Technology-Architecture Co-Design of Near-Memory Processing for Attention-SSM-MoE Hybrid LLM Inference · 2026-08-23
  https://arxiv.org/abs/2608.22613
- [S02] SK hynix 2026 Future Forum · 2026-09-09
  https://news.skhynix.com/en/future-forum-2026/
- [S03] SK hynix at OCP Global Summit 2026 · 2026-10-16
  https://news.skhynix.com/en/ocp-global-summit-2026/
- [S04] HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference · 2024-11
  https://arxiv.org/abs/2411.01433
- [S05] TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization · 2025-05
  https://arxiv.org/abs/2505.19586
- [S06] Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems · 2025-12-04
  https://arxiv.org/abs/2512.04476
- [S07] SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference · v1 2025-12; v5 2026-09
  https://arxiv.org/abs/2512.12990
- [S08] ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories · 2026-06
  https://arxiv.org/abs/2606.12556
- [S09] A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache) · 2026-07; revised 2026-08
  https://arxiv.org/abs/2607.18141
- [S10] ReplaySSM: Cache SSM Inputs, Not State · 2026-10-09
  https://arxiv.org/abs/2610.13778
- [S11] StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference · 2026-11-05
  https://arxiv.org/abs/2611.08363
- [S12] Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning · 2026-06-12
  https://arxiv.org/abs/2606.15007
- [S13] PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States · 2026-10-21
  https://arxiv.org/abs/2610.29545

## 24｜附录 / 行业与竞争证据索引 1/2

建议 25 秒。

引用均来自已核查的官方页面、公开代码或论文。点击来源可访问原始网页；来源文档还保留原文节选及证据边界。竞争公告不是独立横向测试，当前未检索到的内容不意味着竞争对手没有该能力。

- [S14] Samsung Introduces Industry’s First Open-source Software Solution for CXL Memory Platform · 2021-10-07
  https://news.samsungsemiconductor.com/global/samsung-introduces-industrys-first-open-source-software-solution-for-cxl-memory-platform/
- [S15] Samsung Electronics Announces Successful Building of Industry-First CXL Infrastructure Certified by Red Hat · 2024-06-25
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-successful-building-of-industry-first-cxl-infrastructure-certified-by-red-hat/
- [S16] Part 2. zHBM: Next-Generation Memory for the Data Movement Era · 2026-10-29
  https://news.samsungsemiconductor.com/global/part-2-zhbm-next-generation-memory-for-the-data-movement-era/
- [S18] OpenMPDK / SMDK — Scalable Memory Development Kit, README · 访问 2026-11-08；持续更新仓库
  https://github.com/OpenMPDK/SMDK
- [S23] Samsung Develops Industry’s First CXL DRAM Supporting CXL 2.0 · 2023-05-12
  https://news.samsungsemiconductor.com/global/samsung-develops-industrys-first-cxl-dram-supporting-cxl-2-0/

## 25｜附录 / 行业与竞争证据索引 2/2

建议 25 秒。

引用均来自已核查的官方页面、公开代码或论文。点击来源可访问原始网页；来源文档还保留原文节选及证据边界。竞争公告不是独立横向测试，当前未检索到的内容不意味着竞争对手没有该能力。

- [S17] Samsung Electronics Announces Third Quarter 2026 Results · 2026-10-29
  https://news.samsungsemiconductor.com/global/samsung-electronics-announces-third-quarter-2026-results/
- [S19] Micron in High-Volume Production of HBM4 Designed for NVIDIA Vera Rubin, PCIe Gen6 SSD and SOCAMM2 · 2026-03-16
  https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx
- [S20] Micron Technology, Inc. Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results · 2026-09-30
  https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results
- [S21] NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models · 2025-03-18
  https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- [S22] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving · v1 2024-06-24；v5 2026-10-08
  https://arxiv.org/abs/2407.00079

