---
title: Kimi1.5
type: docs
weight: 1
---

# 论文
[KIMI K1.5:SCALING REINFORCEMENT LEARNING WITH LLMS](https://arxiv.org/pdf/2501.12599)

[git](https://github.com/MoonshotAI/Kimi-k1.5)



# Kimi K1.5 技术报告深度解读 (Kimi K1.5 Paper Reading Notes)
### 0. 背景 (Background)



- **发布 (Release):** 2025年1月20日 (与 DeepSeek-R1 同时期)。
- **效果 (Achievement):** 达到了与 OpenAI-O1 模型相似的效果。
- **价值 (Significance):** 报告中包含更多可供算法工程师和研究人员参考的**算法处理细节**，特别是关于如何处理数据以增强推理能力。

------



### 1. 整体架构 (Overall Architecture)



- 遵循与 ChatGPT 相似的标准流程：
  1. **预训练 (Pre-training)**
  2. **SFT 训练 (Supervised Fine-Tuning)**
  3. **强化学习 (RL)**

------



### 2. 预训练 (Pre-training)



- **基础模型 (Base Model):** 在多样化、高质量的多模态语料库上训练 (包括：英、中、代码、数学、知识、图像描述等)。
- **训练阶段 (Phases):** 分三个阶段进行
  1. **视觉-语言预训练 (Vision-Language Pre-training):** 先训练 LLM，然后逐步整合多模态能力 (Vision Tower 独立训练，然后加入图文交织数据并更新 LLM 参数)。
  2. **冷却阶段 (Cooling Down):** 利用精选和合成数据巩固模型能力，尤其针对推理和知识任务 (使用**拒绝采样**保证质量)。
  3. **长上下文激活阶段 (Long-Context Activation):**
     - **目标:** 将序列处理能力扩展到 **131,072 Token**。
     - **方法:**
       - 过采样 Long-Context 数据 (40% 全注意力，60% 部分注意力)。
       - 逐步训练：4k -> 32k -> 128k。

------



### 3. SFT 训练 (SFT Training)





#### 3.1. 常规的 SFT (Standard SFT)



- **数据构建 (Data Construction):** 通过人工标注构建种子集，用种子模型生成回复，再进行排序和优化 (对推理任务使用基于规则/奖励模型的**拒绝采样**扩充)。
- **数据分布 (Data Distribution):** 总计约 100 万样本，涵盖通用问答、编码、数学/科学、长上下文任务及文本-视觉示例。



#### 3.2. Long-CoT SFT (重点 1)



- **目的 (Purpose):** 获得长链思维 (Long-CoT) 能力。
- **数据特点 (Data Characteristics):** 答案部分较长，且蕴含**人类思考过程** (规划、评估、反思、探索)。

------



### 4. 强化学习 (RL)（重点 2）





#### 4.1. RL 数据集构建 (Dataset Construction)



- **衡量属性 (Key Properties):**
  - **多样性 (Diversity):** 保证数据来源和领域丰富，通过**标签系统**打标。
  - **难度平衡 (Difficulty Balance):** 对问题进行难度分级 (基于高温度回复通过率)。采用**课程学习 (Curriculum Learning)**，先学简单，再学复杂。
  - **精确评估 (Accurate Evaluation):** 移除难以精确评估的问题 (如多选题、对错题、证明题)，以及容易被 **Hack** 的 Prompt。



#### 4.2. 问题定义 (Problem Definition)



- 核心是将优化问题转化为最大化**最终答案** (y) 和**思考过程** (Z) 的奖励 (r) 的期望。



#### 4.3. 策略优化 (Policy Optimization)



- **优化目标 (Optimization Goal):** 将 RL 期望最大化转化为 Loss 函数进行优化 (通过梯度上升)。
- **关键点 (Key Details):**
  - <u>移除了 **Value Model** (与 DeepSeek-R1 相似)，认为错误路径也有助益。</u>
  - <u>设计了**长度惩罚奖励** (Length Penalty) 来减少模型过度思考 (越长奖励越小)。</u>



#### 4.4. 采样策略 (Sampling Strategy)



- 采用 Off-policy 训练，使用：
  - **课程采样 (Curriculum Sampling):** 从简单任务过渡到复杂任务。
  - **优先采样策略 (Priority Sampling):** 重点训练模型表现不佳的问题 (按 1−si 比例采样，si 为成功率)。



#### 4.5. Long2short (重点 3)



- **目的 (Purpose):** 实现长链推理 (Long-CoT) 到短链推理 (Short-CoT) 的高效迁移，提升响应速度。
- **方法 (Methods):**
  - <u>**模型融合 (Model Fusion):** Long/Short 模型权重直接融合。</u>
  - **最短拒绝采样 (Shortest Refusal Sampling):** 生成多条样本，选最短且正确的。
  - **长短样本的 DPO (DPO):** 短而正确的作为正样本，错误或 1.5 倍长于短样本的作为负样本。
  - <u>**Long2short 强化学习:** 在一阶段 RL 后，使用长度惩罚减少生成长度。</u>



#### 4.6. 其他细节 (Other Details)



- **代码 (Code):** 自动生成测试用例作为奖励 (使用 CYaRon，并对生成的测试用例进行严格筛选验证)。
- **数学 (Math):** 采用**思维链奖励模型 (Chain-of-Thought RM)** 提高评估不同答案书写形式的准确性。
- **视觉数据 (Vision Data):** Vision RL 数据来自真实世界、合成视觉推理和文本渲染数据 (如将文本、代码转为图像)。

------



### 5. 实验结论 (Experimental Conclusions)



- Kimi-K1.5 在主要数据集上表现出色，是首个追平 OpenAI-O1 的**多模态大模型**。
- **自我进化 (Self-Evolution):** 随着训练，模型自发输出更长的 CoT，且效果更好。
- **课程学习 (Curriculum Learning):** 课程采样策略能有效提升模型能力，防止过早饱和。
- **负样本梯度 (Negative Sample Gradient):** 实验证明，使用负样本 (奖励低于平均值的样本) 对模型能力提升有益。





# 参考
[深度解读 Kimi-K1.5，真正了解 RL 数据是怎么筛选的](https://yuanchaofa.com/post/kimi-k1.5-paper-reading-notes.html) ***    
[细节之王 Kimi K1.5，大模型算法工程师复现推理模型必读文章之一](https://mp.weixin.qq.com/s/E6_5_e2Td35h3j11c1V84Q)    
由 Gemimi 辅助生成  
