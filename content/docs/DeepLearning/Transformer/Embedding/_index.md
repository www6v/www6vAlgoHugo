---
title: (原理)Embedding
date: 2023-04-18 06:29:28
weight: 5
tags:
  - Embedding
categories:
  - AIGC  
  - Embedding
---

<p></p>
<!-- more -->

## 目录
<!-- toc -->

 

# example [1]
  - **降维**:   t-SNE  
  - K-Means 聚类
  - 文本搜索  相似度搜索

# Embedding 价值 [2]
  - **降维**
    将这些高维数据映射到一个低维空间，大大减少了模型的复杂度。
  - 捕捉语义信息 
    Embedding不仅仅是降维，更重要的是，它能够捕捉到数据的语义信息。
  - 泛化能力
    由于Embedding能够捕捉到数据的一些内在规律，因此对于这些未见过的数据，Embedding仍然能够给出合理的表示

# 应用 [2]
  - 语义表示和语义相似度
  - 词语关系和类比推理
  - 上下文理解
  - 文本分类和情感分析
  - 机器翻译和生成模型

# 天梯榜
  [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

# example[3]
+ m3e模型
+ bge模型

# 参考
1. [embedding](https://github.com/www6v/openai-quickstart/blob/main/openai_api/embedding.ipynb) git

2. 《AI 大模型应用开发实战营》 03-大模型开发基础：Embedding

3. [一文通透Text Embedding模型：从text2vec、openai-ada-002到m3e、bge](https://blog.csdn.net/v_JULY_v/article/details/135311471) 

1xx. [如何选取RAG中的embedding模型](https://www.bilibili.com/video/BV1Hk4y1X7aG/) v ***
   [huggingface embedding模型排行榜](https://huggingface.co/spaces/mteb/leaderboard)
   [Sentence Bert](https://arxiv.org/pdf/1908.10084.pdf)  
   [Demo Repo](https://github.com/blackinkkkxi/RAG_langchain)  git

1xx. [引入任务Instruction指令的句子向量化方案：Instructor的实现思路及训练数据集构造方案](https://mp.weixin.qq.com/s/qIh07eU8_lYL2gBVzTFzKA)
   [Repo](https://github.com/xlang-ai/instructor-embedding) git

1xx. [也看利用大模型进行RAG文本嵌入训练数据生成：兼看面向NLP任务的开源指令微调数据集 ](https://mp.weixin.qq.com/s?__biz=MzAxMjc3MjkyMg==&mid=2648406715&idx=1&sn=a680597afdb7d5439a11302c7911795f)        《Improving Text Embeddings with Large Language Models》

1xx. [如何提高LLMs的文本表征(Text Embedding)能力?](https://zhuanlan.zhihu.com/p/676589001)
    《Improving Text Embeddings with Large Language Models》
    
1xx. [文本转向量教程s2——认识文本转向量方法（sbert本质和推理加速）](https://www.bilibili.com/video/BV1ex4y1S7u5/?p=2)   V 