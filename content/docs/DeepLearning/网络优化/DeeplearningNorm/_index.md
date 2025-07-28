---
title: 规范化 Norm
date: 2024-01-08 19:40:39
weight: 3
tags:
  - deepLearning
categories: 
  - AIGC
  - deepLearning   
---

<p></p>
<!-- more -->



# Norm 作用[1]

dnn 的标准组件，稳定和加速训练过程

# Batch Norm[1]

- reduce cross **batch size**
- **mini-batch dimension**

一般用于图像，不涉及到padding的问题；

# Layer Norm[1]

- reduce cross **hidden dim**
- reduce across the **feature dimension**.

一般用于序列，一个 batch size 内存在 padding；

- RMSNorm: 对 LN 的一种变体，llama

---

<aside> 💡

- https://spaces.ac.cn/archives/9009
- Pre LN: `llama`
- Post LN: `attention is all you need`

llama在工程上使用Pre LN

</aside>

---

1. [[pytorch\] BN、LN、RMSNorm 及 pre LN vs. post LN 对比，标准化](https://www.bilibili.com/video/BV13q49eaERj/)  v ***

​	[normalization.ipynb](https://github.com/chunhuizhang/llm_aigc/blob/main/tutorials/nn_basics/tricks_norms/normalization.ipynb)

​	[[pytorch\] BN、LN、RMSNorm 及 pre LN vs. post LN 对比，标准化 ](https://www.notion.so/pytorch-BN-LN-RMSNorm-pre-LN-vs-post-LN-177bfe2110848088830cfea3d5a33d3e?pvs=21)

1xx. [Batch Normalization, Layer Normalization and Root Mean Square Layer Normalization: A Comprehensive Guide with Python Implementations](https://afterhoursresearch.hashnode.dev/batch-normalization-layer-normalization-and-root-mean-square-layer-normalization-a-comprehensive-guide-with-python-implementations)

todo

- [7.5 逐层规范化](https://www.notion.so/7-5-174bfe2110848045bc6cff467363d471?pvs=21) 百度邱  有代码

  https://aistudio.baidu.com/education/lessonvideo/3048901
