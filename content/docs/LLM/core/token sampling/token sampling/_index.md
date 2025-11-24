---
title: (翻译)token sampling
weight: 1
---



# Token 采样方法（Token Sampling Methods）

---

## 概述

- 生成式大语言模型（LLM）将输入和输出文本理解为“token”序列；这些 token 可以是单词，也可以是标点符号或单词的一部分。
- LLM 提供若干 token 选择参数，用以控制推理/运行时输出的随机性。选择输出 token 的方法（具体称为 **token 采样方法** 或 **解码策略**），是语言模型文本生成中的一个核心概念。
- 从技术底层来看，token 采样的核心是：模型不断生成一个称为**概率分布**的数学函数，用于决定下一个 token（例如单词）——这一决策会考虑所有先前已输出的 token。简单来说，LLM 在生成文本时执行的是**采样**：即根据条件概率分布**随机选择**下一个单词。
- 以 OpenAI 托管的系统（例如 ChatGPT）为例：在生成概率分布后，OpenAI 的服务器会根据该分布进行 token 采样。该过程存在一定随机性，因此相同的输入提示可能产生不同的输出。
- 本指南将介绍不同的 token 采样方法及相关概念，包括：温度（Temperature）、贪心解码、穷举搜索解码、束搜索、Top-$k$、Top-$p$（核心采样）以及 Min-$p$。

---

## 背景

### 自回归解码（Autoregressive Decoding）

- 在使用语言模型生成文本序列时，我们通常从一段文本前缀（即提示 prompt）开始，然后按以下步骤循环：
  1. 使用语言模型预测下一个 token；
  2. 将该 token 加入当前输入序列；
  3. 重复上述过程。
- 通过这种持续生成下一个 token 的方式（即 **自回归解码**），我们可以生成整个文本序列（见下图；来源）。

![自回归解码示意图](https://aman.ai/primers/ai/assets/token-sampling/ar.jpg)

### Token 概率

- 那么，我们该如何选择/预测下一个 token（即上述第 1 步）？
- 语言模型并不直接输出下一个 token，而是输出一个**所有可能 token 的概率分布**。简言之，LLM 本质上是在词汇表（所有唯一 token 的集合）上进行分类任务的神经网络。
- 基于该概率分布，我们可以采用多种策略来选择下一个 token。例如，后文将介绍的**贪心解码**（greedy decoding）直接选择概率最高的 token 作为下一个输出。

### Logits 与 Softmax

- LLM 通过 logits 向量 $\mathbf{z} = (z_1, \dots, z_n)$ 表示类别打分，并使用 **softmax 函数** 将其转化为概率向量 $\mathbf{q} = (q_1, \dots, q_n)$：
  $$
  q_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
  $$
- Softmax 函数通过对 logits 取指数并归一化，使得模型在每个时间步的输出均落在 $[0, 1]$ 区间，且总和为 1，从而便于将输出解释为概率（见下图；来源）。

![Softmax 示意图](https://aman.ai/primers/ai/assets/token-sampling/Softmax.jpg)

---

## 相关概念：温度（Temperature）

- 尽管温度本身并非一种 token 采样方法，但它显著影响采样过程，因此本篇纳入讨论。
- 温度参数允许我们调整 token 的概率分布。它作为 softmax 变换中的一个超参数（见下图；来源），在应用 softmax 前对 logits 进行缩放，从而控制预测的随机性。

![温度对 softmax 的影响](/primers/ai/assets/token-sampling/T.jpg)

- 例如在 TensorFlow 的 Magenta 项目中（LSTM 实现），温度参数控制 logits 在 softmax 前被缩放（或除以）的程度。

### 温度在 Softmax 中的作用

- 标准 softmax 引入温度超参数 $T$ 后的形式为：
  $$
  q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
  $$
  其中 $T$ 为温度（默认为 1）。

- 当 $T=1$，即直接对 logits 计算 softmax；  
  若 $T=0.6$，则对 $\frac{\text{logits}}{0.6}$ 计算 softmax —— 此时数值被放大，softmax 结果更“尖锐”；  
  → 模型更**自信**（更少输入即可激活输出层），但也更**保守**（不太可能采样低概率候选）。

### 不同温度范围及其影响

#### 低温（$T \approx 0.0$–$0.5$）

- **特征**：
  - 强烈偏好高概率 token；
  - 输出确定性强、随机性低；
  - 文本常重复，多样性差。
- **适用场景**：
  - 需要高精度或高置信度的场景（例如生成事实性回答）。
- **局限性**：
  - 可能过于保守，陷入重复环（repetitive loops）。

#### 中温（$T \approx 0.6$–$1.0$）

- **特征**：
  - 在多样性与连贯性之间取得平衡；
  - 允许探索较低概率选项，但不显著损伤文本合理性。
- **适用场景**：
  - 生成类人语句、代码补全、音乐作曲等需“有创意但合理”的任务。
- **局限性**：
  - 仍会偏向高概率 token，抑制极低概率创意探索。

#### 高温（$T > 1.0$）

- **特征**：
  - 生成更“平缓”的概率分布；
  - 输出更随机、多样；更倾向低概率选项；
  - 有助于跳出重复环，探索更广空间。
- **适用场景**：
  - 头脑风暴、高度创意内容生成（如艺术、故事、诗歌）。
- **局限性**：
  - 易出现逻辑错误或语义混乱；
  - 采样到不合理 token 的风险增大。

### Softmax 函数的深层理解（引自维基百科）

> 当温度 $\tau \to \infty$ 时，所有样本概率趋近相等；温度越低（$\tau \to 0^+$），预期奖励最高的样本概率趋近于 1。

### 温度影响总结

- **低温** → 更自信、更保守 → 适合确定性任务；  
- **中温** → 平衡随机性与连贯性 → 通用推荐区间；  
- **高温** → 更富创意、更随机 → 适合探索性任务。  
- 通过调节温度，可依任务需求灵活调整模型行为，极大增强 LLM 的适用性。

---

## 贪心解码（Greedy Decoding）

- 贪心解码在每一步都使用 `argmax` 选择**当前概率最高**的 token 作为输出。
- **问题**：它无法回溯修正之前生成的 token。  
  举例：输入法语 “il a m’entarté”（他用派砸了我），若贪心解码已生成 “he hit a”，即使后续发现应为 “me”，也无法回头修改。
- 模型逐 token 生成序列，每步仅考虑当前最优——不评估该选择对未来步骤的影响。
- 解码通常持续至生成 `<END>` token 为止。例如：  
  `<START>` he hit me with a pie `<END>`（来源）
- 优点：计算高效、实现简单；  
  缺点：不保证全局最优输出序列。  
- 改进方向：采用穷举搜索或束搜索（beam search）。

![贪心解码示意图](https://aman.ai/primers/ai/assets/token-sampling/2.png)

---

## 穷举搜索解码（Exhaustive Search Decoding）

- 顾名思义，穷举搜索考察**所有可能的输出序列组合**，并选出评分最高的那个。
- 在序列到序列任务（如神经机器翻译）中，这意味着生成所有可能的译文，再用评分函数评估其与目标的匹配度。
- **问题**：计算复杂度极高——候选数量随输出长度呈指数级增长。
- 时间复杂度为 $O(V^T)$，其中 $V$ 为词表大小，$T$ 为输出长度；实际中几乎不可行。
- 尽管理论上可得最优解，但因其高昂开销，极少用于真实场景。

---

## 束搜索（Beam Search）

- 束搜索是机器翻译等任务中常用的搜索算法，用于高效生成**最可能的词序列**。
- 核心思想：在每步解码时，仅保留 $k$ 个最高分的**部分候选序列**（partial hypotheses），$k$ 即为**束宽（beam size）**，通常取 5–10。
- 具体流程（见下图，束宽=2）：
  - 每步计算若干候选项及其累积得分（通常为对数概率之和）；
  - 保留 top-$k$ 路径继续扩展；
  - 后续通过回溯获得完整输出。

![束搜索示意图](https://aman.ai/primers/ai/assets/token-sampling/1.png)

- 不同候选可能在不同时间步生成 `<END>`：
  - 一旦某候选产出 `<END>`，视为完成，暂存；
  - 继续扩展其余候选，直至：
    - 达到预设最大长度 $T$，或  
    - 已获得足够多（如 $n$ 个）完成候选。
- **得分归一化问题**：较长序列通常累积得分更低（因每步概率 <1，连乘/求和后更小）→ 需按长度归一化（如使用平均对数概率）后再比较。
- 注意：束搜索**不保证全局最优**，但远优于穷举，兼顾质量与效率。  
  详见：D2L.ai《动手学深度学习》— 束搜索章节。

---

## 约束束搜索（Constrained Beam Search）

- 适用场景：需**强制输出中包含特定词/短语**（如机器翻译中必须包含某术语）。
- 基本思想：在束搜索过程中**加入硬性约束条件**，仅保留满足约束的候选路径。
- 实现方式：
  - 修改评分函数，或
  - 在每步生成后剔除违反约束的候选，
  - 或引入惩罚项降低违规路径得分，
  - 或用独立模块动态反馈约束满足情况。
- 示例：生成句子时需包含短语 “is fast”；  
  除常规高概率词（如 “dog”、“nice”）外，**强制加入 “is”** 以推进约束达成（见下图）。

![约束束搜索 Step 1](https://aman.ai/primers/ai/assets/token-sampling/5.png)

### 银行机制（Banking）

- 强制插入 token 是否会导致荒谬输出？**银行机制**可解决此问题：
  - 将候选按**满足约束的程度**分为多个“银行”（Bank）；
  - Bank 2：已满足全部约束；  
    Bank 1：接近满足；  
    Bank 0：尚未开始满足。
  - 采用**轮询选择**（round-robin）：依次从 Bank 2、1、0 中各选最高分候选，再从 Bank 2、1、0 选次高……  
    （例：若用 3 束，则选出：`["The is fast", "The dog is", "The dog and"]`）

![银行机制示意图](https://aman.ai/primers/ai/assets/token-sampling/6.png)

- 这样既保证约束逐步满足，又维持高概率合理序列的竞争力。
- 下图为全流程结果：

![约束束搜索全流程](https://aman.ai/primers/ai/assets/token-sampling/7.png)

---

## Top-$k$ 采样

- 核心思想：每步从**概率最高的 $k$ 个 token**中采样，而非仅选最大者。
- 采样方式可为：
  - **均匀采样**：top-$k$ 内各 token 等概率 → 提升多样性；
  - **按原概率采样**：保持分布权重 → 提升连贯性。
- $k=1$ 时退化为贪心解码。
- $k$ 越小 → 选择越窄 → 多样性↓、控制性↑；  
  $k$ 越大 → 选择越宽 → 多样性↑、控制性↓。
- 适用于需平衡多样性与可控性的任务（如对话生成）。

![Top-$k$ 示意图（$k=3$）](https://aman.ai/primers/ai/assets/token-sampling/3.png)

---

## Top-$p$（核心采样 / Nucleus Sampling）

- 动机：Top-$k$ 中 $k$ 难以选取 → 需动态调整候选集大小。
- Top-$p$ 方法：
  - 按概率降序排列所有 token；
  - 取**最小的前缀子集**，使其**累积概率 ≥ $p$**（如 $p=0.9$）；
  - 重新归一化该子集概率（使其和为 1）；
  - 从中按新概率采样。
- 与 Top-$k$ 关键区别：  
  Top-$k$ 固定数量，Top-$p$ 固定**概率质量**；  
  后者可根据分布“自适应”调整候选数。

![核心采样示意图](https://aman.ai/primers/ai/assets/token-sampling/nucleus.jpg)

![Top-$p=0.15$ 示例](https://aman.ai/primers/ai/assets/token-sampling/4.png)

### 实用价值

- 适合需精细调控多样性与流畅度的任务（如语言建模、摘要生成）；
- 实际中 $p$ 常设为 0.75 左右，以过滤长尾低概率噪声 token；
- 特殊情形：
  - 若某 token 概率 > $p$，则必然被选（退化为贪心）；
  - 若概率分布平坦，则候选集变大 → 更富创意。
- 注意：Top-$k$ 与 Top-$p$ 可**联用**（先取 top-$k$，再在其中做 top-$p$），但 $p$ 作用于 $k$ 之后。

### 与温度参数的关系

- OpenAI GPT-3 API 提示：**温度与 top-$p$ 互斥**（见下图）；  
  二者是**不同且互斥**的随机性控制机制。

![GPT-3 API 参数说明](https://aman.ai/primers/ai/assets/token-sampling/API.jpg)

---

## 贪心 vs. Top-$k$ 与 Top-$p$

| 对比维度         | 贪心解码                 | Top-$k$/Top-$p$            |
| ---------------- | ------------------------ | -------------------------- |
| **确定性**       | 确定性（总是选最高概率） | 随机性（引入采样）         |
| **采样方式**     | 无（直接 argmax）        | 可均匀或按概率             |
| **文本风格倾向** | 安全、保守、缺乏创意     | 更新颖、多样，但可能不连贯 |

---

## Min-$p$ 采样

- Min-$p$ 是 Hugging Face Transformers 库引入的**新型解码策略**，旨在改进 Top-$k$ 与 Top-$p$ 的不足。

### 现有方法局限回顾

- **Top-$k$**：
  - 固定截断 → 可能丢弃高质量低频 token → 降低词汇多样性。
- **Top-$p$**：
  - 包含极低概率 token → 可能破坏连贯性。

### Min-$p$ 核心思想

- 引入**动态阈值**：  
  设定最小概率因子 `min_p`（如 0.05），  
  计算阈值 = `min_p × 最高概率 token 的概率`；  
  → 仅保留概率 ≥ 该阈值的 token。
- 优势：
  - 当存在**绝对主导 token**时 → 严格过滤，保证聚焦与连贯；
  - 当概率分布**较平坦**时 → 宽松保留，支持创意发散。
- **推荐配置**（尤其适合创意生成）：
  - `min_p ∈ [0.05, 0.1]`  
  - 配合**高温**（$T > 1$）→ 充分激发创造力；
  - 可**减少甚至无需**使用“重复惩罚（repetition penalty）”等补丁技巧。

![Min-$p$ 示意图](https://aman.ai/primers/ai/assets/token-sampling/min-p.jpeg)

- 简言之，Min-$p$ 通过**自适应概率截断**，在多样性与连贯性间取得更优平衡，有望成为新一代解码标准。

---

## 参考文献

- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. “Distilling the knowledge in a neural network.” *arXiv preprint arXiv:1503.02531* (2015).  
- What is Temperature in LSTM (and neural networks generally)?  
- Stanford CS224n  
- Ketan Doshi, *Foundations of NLP Explained Visually*: Beam Search, How it Works  
- Cohere: Top-k and Top-p  
- HuggingFace: Constrained Beam Search  

---

如需 PDF 或 Markdown 格式文档，我可为您整理输出。


# 参考
[Token Sampling Methods](https://aman.ai/primers/ai/token-sampling/)   qwen 翻译





# 总结（Kimi）

| 方法        | 是否随机 | 控制方式        | 优点           | 缺点             |
| ----------- | -------- | --------------- | -------------- | ---------------- |
| Greedy      | 否       | argmax          | 快速、确定     | 易重复、局部最优 |
| Beam Search | 否       | 保留 top-k 序列 | 比贪心更优     | 不保证全局最优   |
| Top-k       | 是       | 固定数量        | 简单有效       | 可能包含低质量词 |
| Top-p       | 是       | 动态累积概率    | 更灵活         | 可能引入低概率词 |
| Min-p       | 是       | 动态阈值        | 高温度下更稳定 | 新方法，需调参   |
