---
title: Survey
date: 2024-07-08 18:49:43
weight: 1
---


# 论文
[Reinforcement Learning Foundations for Deep Research Systems: A Survey](https://arxiv.org/pdf/2509.06733)



# Method

| 方法 | 优化目标   | 数据形式    | 关键短板              |
| :--- | :--------- | :---------- | :-------------------- |
| SFT  | 模仿单步   | (q, a) 对   | 暴露偏差、无法纠错    |
| DPO  | 偏好排序   | (q, a⁺, a⁻) | 无状态、信用分配短视  |
| RL   | 最大化回报 | (q, τ, r)   | 需可验证奖励+探索策略 |



# RL METHODS FOR AGENTIC RESEARCH

### TRAINING REGIME AND OPTIMIZATION STRUCTURE

![TRAINING REGIME AND OPTIMIZATION STRUCTURE](./images/640.webp)



### REWARD DESIGN AND CREDIT ASSIGNMENT

![REWARD DESIGN AND CREDIT ASSIGNMENT](./images/640-1.webp)


+ 结果奖励（Outcome-only）


+ 步骤奖励（Step-level）


# framework
![framework](./images/640-framework.webp)



# 参考

[2篇最新论文，把Deep Research讲透了~](https://mp.weixin.qq.com/s/SbPF7zAulPok2Xz3wU2ncw)