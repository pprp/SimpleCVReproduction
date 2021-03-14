# NAS Paper Reading List



## 综述

 AutoML: A Survey of the State-of-the-Art  https://arxiv.org/pdf/1908.00709.pdf



## 分类

NAS的搜索空间分类

- entire-structured 
- cell-based
- hierarchical
- morphism-based

Architecture Optimization分类：

- Reinforcement Learning : RL
- Evolution-Based algorithm :EA
- Gradient descent : GD
- Surrogate Model-Based Optimization: SMBO
- hybrid AO methods
- Random search

Model Estimation分类：

- low-fidelity
- Early-stopping
- Surrogate Model 代理模型
- Resource-aware
- Weight-sharing



## 数据准备

数据准备可以分为三个角度：数据搜集、数据清洗、数据增强。



## 网络架构搜索论文



| 题目                                                       | 评价                                                 | 链接                                 | 代码       | 分类 | CIFAR10 Error rate |
| ---------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------ | ---------- | ---- | ------------------ |
| Neural Architecture Search with Reinforcement Learning     | 谷歌最初提出NAS的文章 12,800, 强化学习搜索网络架构。 | https://arxiv.org/abs/1611.01578     | [NAS-RL]() |      | 3.65%-5.50%        |
| Efficient neural architecture search via parameter sharing | 提出权重共享的方法，提升效率，提出cell-based方法     | https://arxiv.org/pdf/1802.03268.pdf | [ENAS]()   |      | 2.89%-4.23%        |
|                                                            |                                                      |                                      | DARTS      |      |                    |
|                                                            |                                                      |                                      | P-DARTS    |      |                    |
|                                                            |                                                      |                                      | HierNAS    |      |                    |
|                                                            |                                                      |                                      | PNAS       |      |                    |
|                                                            |                                                      |                                      |            |      |                    |





