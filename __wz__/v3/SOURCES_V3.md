# V3｜行业、竞争与技术来源

核查日期：2026-11-08。S01–S13 为既有技术来源；S14–S23 为本轮新增核查。

## 引用规范

每项行业判断都区分公开事实、状态及提案分析。公司公告按公司自述标注，研发不视作量产，认证 / 演示不视作特定 LLM 性能验证。未使用未核实的市场份额、CAGR、财务数字或领先排名。

## S14｜Samsung Introduces Industry’s First Open-source Software Solution for CXL Memory Platform

- 发布日期：2021-10-07
- 核查日期：2026-11-08
- 原始来源：https://news.samsungsemiconductor.com/global/samsung-introduces-industrys-first-open-source-software-solution-for-cxl-memory-platform/
- 使用范围与限制：三星 SMDK 发布公告，含 Intelligent Tiering Engine、库与 API。该历史公告当时为有限测试，开源计划不能按当时已经全面部署描述；当前仓库由 S18 补充。

原文节选：

> Leveraging a proprietary Intelligent Tiering Engine, the SMDK can identify and configure the memory type, capacity and bandwidth that are most appropriate for each use case.

## S15｜Samsung Electronics Announces Successful Building of Industry-First CXL Infrastructure Certified by Red Hat

- 发布日期：2024-06-25
- 核查日期：2026-11-08
- 原始来源：https://news.samsungsemiconductor.com/global/samsung-electronics-announces-successful-building-of-industry-first-cxl-infrastructure-certified-by-red-hat/
- 使用范围与限制：三星 SMRC 的 Red Hat 认证 CXL 基础设施、CMM-D 验证与 SMDK memory interleaving 演示。认证与 DLRM 演示不等于完成混合 LLM 联合量化。

原文节选：

> Elements that configure servers, from CXL-related products to software, can now be directly verified at the Samsung Memory Research Center (SMRC)

## S16｜Part 2. zHBM: Next-Generation Memory for the Data Movement Era

- 发布日期：2026-10-29
- 核查日期：2026-11-08
- 原始来源：https://news.samsungsemiconductor.com/global/part-2-zhbm-next-generation-memory-for-the-data-movement-era/
- 使用范围与限制：三星明确称 zHBM currently under development，讨论直接 3D 堆叠、Distributed I/O 和系统协同。是研发方向，非量产公告；未用其推断本项目收益。

原文节选：

> Samsung zHBM, currently under development, is a next-generation AI memory architecture designed around this challenge.

## S17｜Samsung Electronics Announces Third Quarter 2026 Results

- 发布日期：2026-10-29
- 核查日期：2026-11-08
- 原始来源：https://news.samsungsemiconductor.com/global/samsung-electronics-announces-third-quarter-2026-results/
- 使用范围与限制：仅引用 Memory Business 对 HBM4 销售扩展、服务器 SSD 需求和 AI 产品组合的定性自述。不使用财务数据、份额或厂商自称领先来比较技术能力。

原文节选：

> It significantly scaled up sales of HBM4 on the back of industry-leading product competitiveness while proactively addressing growing demand for server SSDs.

## S18｜OpenMPDK / SMDK — Scalable Memory Development Kit, README

- 发布日期：访问 2026-11-08；持续更新仓库
- 核查日期：2026-11-08
- 原始来源：https://github.com/OpenMPDK/SMDK
- 使用范围与限制：README 描述 SMDK v2.2、Intelligent Tiering Engine、兼容与优化 API。代码公开性可验证，不证明任何特定模型性能或本提案所需 GPU/CXL 路径。

原文节选：

> The SMDK is developed for the Samsung CXL (Compute Express Link) Memory Expander devices to enable a full-stack Software-Defined Memory system.

## S19｜Micron in High-Volume Production of HBM4 Designed for NVIDIA Vera Rubin, PCIe Gen6 SSD and SOCAMM2

- 发布日期：2026-03-16
- 核查日期：2026-11-08
- 原始来源：https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx
- 使用范围与限制：美光官方称 HBM4 36GB 12H、9650 Gen6 SSD、192GB SOCAMM2 高量产，并讨论 Vera Rubin 与 BlueField-4 STX。量产状态是公司公告，不是本项目实测或独立横评。

原文节选：

> HBM4 36GB 12H in high-volume production ... / 192GB SOCAMM2 in high-volume production [节选新闻要点]

## S20｜Micron Technology, Inc. Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results

- 发布日期：2026-09-30
- 核查日期：2026-11-08
- 原始来源：https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results
- 使用范围与限制：仅使用 Product Highlights：7600 PCIe Gen5 和 9650 PCIe Gen6 SSD 已向头部客户的 KV-cache 应用出货。未使用财务数字或将出货表述为 LLM 端到端提速证据。

原文节选：

> Our 7600 PCIe Gen 5 and 9650 PCIe Gen 6 SSD high-performance products are now shipping to market-leading customers for KV-cache applications.

## S21｜NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models

- 发布日期：2025-03-18
- 核查日期：2026-11-08
- 原始来源：https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- 使用范围与限制：官方技术博客正文：PD 分离、KV-aware routing、多层 KV 管理和 NIXL。只引用架构功能，不使用其中混有预测条件的 30x 数字。NVIDIA 是平台与生态方，不在本提案中作为内存器件竞争对手排名。

原文节选：

> KV cache offloading across different memory hierarchies to increase system throughput

## S22｜Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- 发布日期：v1 2024-06-24；v5 2026-10-08
- 核查日期：2026-11-08
- 原始来源：https://arxiv.org/abs/2407.00079
- 正文：https://arxiv.org/html/2407.00079v5
- 使用范围与限制：论文描述 Kimi 的 KV-centric PD 分离架构，使用 CPU、DRAM、SSD 缓存并优化 SLO 下有效吞吐。仅用架构作为行业信号，不外推其 reported throughput 到本项目。

原文节选：

> It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache.

## S23｜Samsung Develops Industry’s First CXL DRAM Supporting CXL 2.0

- 发布日期：2023-05-12
- 核查日期：2026-11-08
- 原始来源：https://news.samsungsemiconductor.com/global/samsung-develops-industrys-first-cxl-dram-supporting-cxl-2-0/
- 使用范围与限制：三星 128GB CXL 2.0 DRAM 开发及内存池支持的历史里程碑，当年量产为计划；不能把它当作 2026 年最新代际或现状排名。

原文节选：

> For the first time ever, CXL 2.0 supports memory pooling

---

# 既有技术来源与假设边界（S01–S13）

# 参考来源与证据边界

公开来源核查日期：2026-11-08。来源基于本次提案前的网络检索与正文核查。链接保留原始地址，便于评审追溯。

性能结果均属于对应论文的实验设置，未将其相加或当作本提案结果。本文献列表为定向查新，不能证明不存在其他相近工作，不能代替专利查新。

## S01｜NOVA: Technology-Architecture Co-Design of Near-Memory Processing for Attention-SSM-MoE Hybrid LLM Inference

- 日期：2026-08-23
- 原始来源：https://arxiv.org/abs/2608.22613
- 正文：https://arxiv.org/html/2608.22613v1
- 支持内容与边界：混合负载的内存特征；近存计算背景。其架构收益来自仿真，不是本提案结果。

## S02｜SK hynix 2026 Future Forum

- 日期：2026-09-09
- 原始来源：https://news.skhynix.com/en/future-forum-2026/
- 支持内容与边界：海力士公开提出 Full-Stack AI Memory 和系统协同设计；企业战略信号，不代表本项目已获认可。

## S03｜SK hynix at OCP Global Summit 2026

- 日期：2026-10-16
- 原始来源：https://news.skhynix.com/en/ocp-global-summit-2026/
- 支持内容与边界：CXL 内存池、DRAM+SSD 混合内存、KV 复用和 CMM-Ax 的公开展示；展示不等同于全部功能量产。

## S04｜HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

- 日期：2024-11
- 原始来源：https://arxiv.org/abs/2411.01433
- 支持内容与边界：混合精度专家卸载、预取和缓存的已有工作。

## S05｜TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

- 日期：2025-05
- 原始来源：https://arxiv.org/abs/2505.19586
- 支持内容与边界：KV 量化与卸载相结合；不能将该组合本身声明为首创。

## S06｜Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- 日期：2025-12-04
- 原始来源：https://arxiv.org/abs/2512.04476
- 正文：https://arxiv.org/html/2512.04476v1
- 支持内容与边界：专家 HBM/CXL-NDP 放置与位宽选择；最接近的 MoE 子问题。性能评估采用模拟器，使用多个精度副本。

## S07｜SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference

- 日期：v1 2025-12; v5 2026-09
- 原始来源：https://arxiv.org/abs/2512.12990
- 支持内容与边界：按位切片的混合精度专家缓存；精度切片不能作为未经区分的原创主张。

## S08｜ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories

- 日期：2026-06
- 原始来源：https://arxiv.org/abs/2606.12556
- 支持内容与边界：基于海力士 CMM 和 PCIe Gen5 SSD 的潜力评估及 FPGA 功能原型；最高 35.7% 是其特定基线下的报告值。

## S09｜A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache)

- 日期：2026-07; revised 2026-08
- 原始来源：https://arxiv.org/abs/2607.18141
- 支持内容与边界：真实 CXL-HM 原型，面向可复用 KV 的 DRAM/SSD 管理；不可直接外推到活跃 SSM 状态。

## S10｜ReplaySSM: Cache SSM Inputs, Not State

- 日期：2026-10-09
- 原始来源：https://arxiv.org/abs/2610.13778
- 支持内容与边界：SSM 检查点与输入缓冲；读写和精度的专门机制。其报告的 1.20–1.47 倍加速不是本项目收益。

## S11｜StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference

- 日期：2026-11-05
- 原始来源：https://arxiv.org/abs/2611.08363
- 支持内容与边界：低位宽权重/KV 与 RAM、SSD 精度补充流；截至检索日为近期预印本。

## S12｜Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

- 日期：2026-06-12
- 原始来源：https://arxiv.org/abs/2606.15007
- 支持内容与边界：Attention/Mamba/MoE 混合路线的模型实例；不足以证明未来模型全部采用这一架构。

## S13｜PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States

- 日期：2026-10-21
- 原始来源：https://arxiv.org/abs/2610.29545
- 支持内容与边界：递归状态的周期量化及混合位宽，强调误差传播和更新频率的重要性。

## 本提案的独立假设与验收边界

- 至少 15% 指相对同等资源、质量和服务要求下强基线的有效吞吐增益，尚未验证。
- 80 GB 容量条形图为算例：20+10+20+30=80；20+5+15+40=80。未指向具体模型或已测收益。
- 附录的 17% 延迟下降由假设 0.4×0.5−0.03 得到；约 20.5% 速率提升只适用于固定串行服务假设。
- 六个月、4–5 人和 2027–2029 年交付物是拟议研发计划，未引用海力士内部预算或量产承诺。
- CXL 是互连协议；CXL 内存层在本提案中指附加 DRAM 或内存池。普通 CXL 扩展、SSD-backed CXL Hybrid Memory 和 CXL-NDP 应分别建模。
- 活跃 SSM 状态或稠密注意力 KV 的 SSD 放置并非预设可行；须满足访问时限、质量和带宽要求。
- 降精度造成的信息损失不能通过简单升精度逆转；权重主副本、残差、状态检查点均计入容量与执行成本。
- 新颖性主张是待验证的跨对象边际收益模型、联合分配和低开销实现。量化+分层、精度切片、专家冷热划分均有先例。
- 不承诺 15% 性能、15% 能效和 15% TCO 同时成立；主 KPI 与副指标分开。


## V3 补充边界

- 1M token、550B 总参数、55B 激活参数是 Nemotron 3 Ultra 论文中的模型实例，非行业平均值。
- Samsung 2021/2023/2024 事件表示历史发展路径，不作为 2026 年最新 CXL 产品代际排名。
- zHBM 在 2026-10-29 的原文中明确为 currently under development。
- Micron 量产与出货状态按发布方表述记录，不代替独立出货核验或 LLM 性能横评。
- NVIDIA 为平台 / 软件生态方；未将其作为与 DRAM 厂商相同类别的竞争主体排序。
- “需要孵化联合优化”是提案战略推论；并非引用文献已经证明本项目必要或必然成功。
