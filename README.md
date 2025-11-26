# papers
Reading Papers on Artificial Intelligence

# LLM related
## 1. 基础架构与模型设计
- 《Attention Is All You Need》（2017）
  - 摘要：提出Transformer架构，用自注意力机制替代RNN/CNN，奠定大模型基础
  - 推荐原因：理解现代LLM的底层设计，掌握位置编码、多头注意力等核心概念。
- 《Language Models are Few-Shot Learners》（GPT-3, 2020）
  - 摘要：验证大规模预训练模型（175B参数）的上下文学习能力。
  - 推荐原因：学习数据缩放定律（Scaling Laws）与Prompt工程基础。
- 《GShard: Scaling Giant Models with Conditional Computation》（2020）
  - 摘要：提出混合专家模型（MoE）实现万亿参数级别扩展。
  - 推荐原因：掌握稀疏化训练与分布式计算优化方法。
- 《Switch Transformers: Scaling to Trillion Parameter Models》（2021）
  - 摘要：改进MoE架构，单模型参数突破1.6万亿。
  - 推荐原因：学习动态路由与专家并行技术。

## 2. 训练优化与扩展
- 《Training Compute-Optimal Large Language Models》（Chinchilla, 2022）
  - 摘要：证明模型参数与训练数据的均衡缩放法则（70B参数+1.4T tokens最优）。
  - 推荐原因：指导模型训练的资源分配策略。
- 《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》（2020）
  - 摘要：提出三级内存优化策略，降低显存占用。
  - 推荐原因：掌握分布式训练核心技术。
- 《Mixed Precision Training》（2018）
  - 摘要：使用FP16/FP32混合精度加速训练并保持稳定性。
  - 推荐原因：学习显存与计算效率优化基础。
- 《LoRA: Low-Rank Adaptation of Large Language Models》（2021）
  - 摘要：通过低秩矩阵微调实现参数高效迁移。
  - 推荐原因：掌握轻量化微调的核心方法。

## 3. 多模态与跨领域
- 《Learning Transferable Visual Models From Natural Language Supervision》（CLIP, 2021）
  - 摘要：通过图文对比学习实现跨模态对齐。
  - 推荐原因：理解多模态预训练的基础范式
- 《Flamingo: a Visual Language Model for Few-Shot Learning》（2022）
  - 摘要：融合视觉编码器与LLM，支持多模态上下文学习。
  - 推荐原因：掌握跨模态信息融合架构设计。
- 《PaLM-E: An Embodied Multimodal Language Model》（2023）
  - 摘要：将机器人传感器数据融入语言模型，实现物理世界推理。
  - 推荐原因：学习具身智能（Embodied AI）的实现路径。

## 4. 推理与知识增强
- 《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》（2022）
  - 摘要：通过思维链（CoT）提示激发模型分步推理能力。
  - 推荐原因：掌握复杂问题求解的Prompt设计方法。
- 《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》（RAG, 2020）
  - 摘要：结合检索系统与生成模型解决知识密集型任务。
  - 推荐原因：学习外部知识增强的核心技术。
- 《Toolformer: Language Models Can Teach Themselves to Use Tools》（2023）
  - 摘要：让LLM自主学习调用计算器、搜索引擎等外部工具。
  - 推荐原因：掌握工具增强型Agent的开发逻辑。

## 5. 安全与对齐
- 《Training Language Models to Follow Instructions with Human Feedback》（InstructGPT, 2022）
  - 摘要：基于人类反馈的强化学习（RLHF）实现模型对齐。
  - 推荐原因：理解价值观对齐的核心技术。
- 《Red Teaming Language Models with Language Models》（2022）
  - 摘要：用AI生成对抗性Prompt测试模型安全漏洞。
  - 推荐原因：学习模型红队测试方法论。
- 《Constitutional AI: Harmlessness from AI Feedback》（2022） 
  - 摘要：通过AI自我监督替代人类标注，实现安全对齐。
  - 推荐原因：掌握低成本价值观对齐方案。

## 6. 应用与Agent系统
- 《AutoGPT: Autonomous GPT with Memory and Tools》（2023）
  - 摘要：构建具备长期记忆与工具调用能力的自主Agent框架。
  - 推荐原因：学习复杂任务分解与循环控制机制。
- 《ReAct: Synergizing Reasoning and Acting in Language Models》（2023）
  - 摘要：提出推理（Reasoning）与行动（Action）协同的Agent架构。
  - 推荐原因：掌握动态环境交互的关键设计。
- 《Voyager: An Open-Ended Embodied Agent with Large Language Models》（2023）
  - 摘要：在《我的世界》中实现终身学习的自主探索Agent。
  - 推荐原因：理解开放世界Agent的迭代学习机制。




