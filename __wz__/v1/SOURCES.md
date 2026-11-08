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
