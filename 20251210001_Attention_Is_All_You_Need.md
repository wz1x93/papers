# 《Attention Is All You Need》
- https://arxiv.org/pdf/1706.03762

---

## 一、《Attention Is All You Need》核心知识点梳理

### 1. 背景与动机
- 传统序列建模（如 RNN、LSTM、GRU）存在**顺序依赖**，难以并行化，训练慢。
- CNN-based 模型虽可并行，但需多层堆叠才能捕获长距离依赖。
- 论文提出 **Transformer** 架构：**完全基于注意力机制**，摒弃循环和卷积结构，实现高度并行化，显著提升训练速度和效果。

---

### 2. 核心组件

#### (1) Scaled Dot-Product Attention

公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q \in \mathbb{R}^{n \times d_k}$: queries  
- $K \in \mathbb{R}^{m \times d_k}$: keys  
- $V \in \mathbb{R}^{m \times d_v}$: values  
- $\sqrt{d_k}$ 用于缩放，防止点积过大导致 softmax 梯度消失。

#### (2) Multi-Head Attention

将 attention 分成多个“头”，分别学习不同子空间的表示：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

其中每个 head 为：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K$, $W_i^V$ 为可学习参数
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$
- 默认设置：$h=8$, $d_{\text{model}}=512$, $d_k = d_v = 64$

#### (3) Positional Encoding

由于 Transformer 无位置信息，需显式加入位置编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- $pos$：token 位置索引
- $i$：维度索引
- 也可用可学习的位置嵌入（learnable positional embedding）

#### (4) Encoder-Decoder 架构

- **Encoder**: 6 层堆叠，每层包含：
  - Multi-Head Self-Attention
  - Position-wise Feed-Forward Network（FFN）：两层全连接 + ReLU
  - 残差连接 + Layer Normalization

- **Decoder**: 6 层堆叠，每层包含：
  - Masked Multi-Head Self-Attention（防止看到未来 token）
  - Multi-Head Encoder-Decoder Attention（Query 来自 decoder，Key/Value 来自 encoder）
  - FFN + 残差 + LayerNorm

#### (5) 其他关键技术
- **残差连接** 和 **Layer Normalization**：稳定训练
- **Label Smoothing**：正则化技巧，提升泛化
- **Adam 优化器** with $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$，配合 warmup 和 decay 的学习率调度

---

### 3. 创新与影响
- 首次提出纯注意力架构，成为 NLP 领域里程碑
- 奠定 BERT、GPT、T5 等大模型基础
- 启发 Vision Transformer（ViT）等跨模态应用

---

## 二、面试官提问（模拟）

现在我将以**面试官**身份，围绕该论文向你提出以下问题。请思考如何回答：

---

### 🔹 算法原理类

1. **为什么 Transformer 使用 $\sqrt{d_k}$ 对点积进行缩放？如果不缩放会有什么问题？**
2. **Multi-Head Attention 中的“多头”机制为何有效？能否用数学或直观方式解释其优势？**
3. **Decoder 中的 masked self-attention 是如何实现的？mask 的具体形式是什么？**
4. **Positional Encoding 为什么使用正弦和余弦函数？相比可学习的位置嵌入，各有什么优劣？**

---

### 🔹 应用与设计类

5. **Transformer 最初用于机器翻译，但如今广泛用于各种任务（如文本生成、图像分类）。你认为它成功泛化的关键原因是什么？**
6. **在低资源语言翻译场景下，直接使用标准 Transformer 可能面临什么挑战？有哪些改进策略？**
7. **如果让你用 Transformer 处理长文本（如 10k tokens），你会遇到哪些问题？如何解决？**

---

### 🔹 工程与移植难点类

8. **Transformer 的计算复杂度是多少？与 RNN 相比，在长序列下的内存和计算开销如何？**
9. **在移动端或嵌入式设备上部署 Transformer 模型时，主要瓶颈是什么？有哪些压缩或加速手段？**
10. **训练一个大型 Transformer 模型时，常见的数值不稳定现象有哪些？如何通过初始化、归一化或优化器设置缓解？**

---
