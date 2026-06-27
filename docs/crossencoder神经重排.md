# CrossEncoder 神经重排详解（面试学习笔记）

本文档详细讲解 retrieve 节点中的 CrossEncoder 神经重排技术，包括核心概念、三步工作流程、项目配置和面试要点。

---

## 一、为什么需要 CrossEncoder 重排？

前面的 BM25 + 向量检索已经召回了 Top-K 候选文档，但这些文档只是"可能相关"——

**问题**：向量检索是用**独立编码**的方式（query 转向量、document 转向量、再算相似度），这种方式有一个根本缺陷：**编码时看不到对方的信息**。

**示例**：
- query：`"苹果手机怎么拍照？"` → 向量 A
- document：`"iPhone 拍照技巧：打开相机，点击屏幕对焦..."` → 向量 B  
- document：`"苹果公司发布新款手机，拍照功能大幅升级..."` → 向量 C

向量相似度可能认为 B 和 C 差不多，但实际上 B 才是真正相关的。

**CrossEncoder 的核心优势**：它能看到 query 和 document 的完整信息，做**交叉注意力（cross-attention）**，判断"这段文字是否真的能回答这个问题"。

---

## 二、Bi-encoder vs CrossEncoder

理解两者的区别是关键：

### Bi-encoder（向量检索用的）

**做法**：
1. Query 和 Document **分别独立**编码成向量
2. 用余弦相似度算距离

**优点**：快——Document 向量可以预先算好存起来，查询时只需算一次 query 向量。

**缺点**：编码时看不到对方，丢失了精细匹配信息。

**示意图**：
```
Query → [Encoder] → Vector Q
Document → [Encoder] → Vector D
          ↓
     Cosine(Q, D) → 相似度分数
```

### CrossEncoder（重排用的）

**做法**：
1. 把 Query 和 Document **拼在一起**输入模型
2. 模型做交叉注意力，直接输出一个相关性分数

**优点**：精度高——能看到完整上下文，理解语义关系。

**缺点**：慢——每个(query, document)对都要单独算一次。

**示意图**：
```
[Query + [SEP] + Document] → [CrossEncoder] → 相关性分数
```

---

## 三、为什么只对 Top-K 做重排？

假设文档库里有 100,000 个文档：

| 方案 | 计算量 | 耗时 |
|------|-------|------|
| 对全库做 CrossEncoder | 100,000 次模型推理 | 几十分钟（不可接受） |
| 先召回 Top-50，再做重排 | 50 次模型推理 | 几秒（可接受） |

**两阶段架构**：
1. **召回阶段（Recall）**：用 BM25 + 向量检索快速捞回候选（Top-50~100）
2. **精排阶段（Rerank）**：用 CrossEncoder 对候选做神经网络精排（Top-K）

---

## 四、CrossEncoder 的工作流程（三步骤）

### 步骤 1：输入拼接

把 query 和 document 用分隔符拼在一起：

```
输入 = query + "[SEP]" + document
```

**示例**：
```
"LangGraph 的 StateGraph 怎么用？[SEP]LangGraph 的 StateGraph 是一个状态图框架，用于构建有状态的工作流..."
```

**为什么需要分隔符？**

模型需要知道哪部分是 query，哪部分是 document。`[SEP]` 是特殊 token，告诉模型两个片段的边界。

---

### 步骤 2：模型推理

用预训练的 CrossEncoder 模型处理这个拼接文本，输出一个相关性分数。

**本项目使用的模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2`

| 属性 | 说明 |
|------|------|
| 基础模型 | BERT-base 变体（6 层 Transformer） |
| 训练数据 | MS MARCO 数据集（微软大规模问答数据集） |
| 输出范围 | [-∞, +∞]，分数越高越相关 |
| 模型大小 | ~100MB |
| 推理速度 | 每个 pair 几毫秒 |

**推理过程**：
1. 模型对拼接文本做 tokenize（分词）
2. 通过 Transformer 编码器，query 和 document 之间做交叉注意力
3. 最后一层的 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token（句子开始标记）输出一个向量
4. 经过一个线性层，输出最终的相关性分数

**为什么精度高？**

交叉注意力让模型能：
- 理解 query 中的代词指代 document 中的哪个实体
- 匹配句法结构（比如"怎么用"对应 document 中的步骤描述）
- 判断语义关系（比如"拍照"不只是匹配"照片"这个词）

---

### 步骤 3：排序输出

对所有候选文档的分数排序，取前 K 个作为最终检索结果。

**示例**：

| 文档 | CrossEncoder 分数 |
|------|------------------|
| doc_A | 8.2 |
| doc_B | 6.5 |
| doc_C | 2.1 |
| doc_D | -1.3 |

**最终排名**：doc_A > doc_B > doc_C > doc_D

**本项目配置**：保留前 4 个（top_n=4）

---

## 五、项目中的实际配置

看 [retriever.py](file:///Users/scm/code/CORTEX-AI-SUPER-RAG/src/retrieval/retriever.py) 中的代码：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 创建 CrossEncoder 重排器
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=4  # 重排后保留前 4 个
)

# 把重排器包装到检索器上
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever  # 先用 EnsembleRetriever 召回
)
```

**完整流程**：
1. `ensemble_retriever` 先召回 Top-N（比如 Top-10）文档
2. `CrossEncoderReranker` 对这 10 个文档逐一打分
3. 按分数排序，保留前 4 个

---

## 六、为什么选 ms-marco-MiniLM-L-6-v2？

| 维度 | ms-marco-MiniLM-L-6-v2 | 更大的模型（如 MultiBERT） |
|------|------------------------|--------------------------|
| 模型大小 | 小（~100MB） | 大（~400MB+） |
| 推理速度 | 快（每对几毫秒） | 慢（每对几十毫秒） |
| 精度 | 高 | 更高 |
| 资源需求 | 低（CPU 即可） | 高（推荐 GPU） |

**选择理由**：在个人知识库场景下，精度已经足够，速度更重要。

---

## 七、面试怎么讲？（30秒版）

> CrossEncoder 是用来做神经重排的——它把 query 和 document 拼在一起输入模型，做交叉注意力来判断相关性。和向量检索用的 bi-encoder 不同，CrossEncoder 编码时能看到完整上下文，所以精度更高，但计算代价大，必须先做召回再做精排。我们用的是 ms-marco-MiniLM-L-6-v2，先通过 BM25+向量召回 Top-N 候选，再用 CrossEncoder 对这些候选逐一打分排序，保留前 4 个。这样既保证了召回率，又提升了最终排序的准确性。

---

## 八、面试高频追问

### Q1：为什么不用 CrossEncoder 直接做全库检索？

因为慢。假设全库有 100,000 个文档，每个文档都要和 query 拼在一起做一次模型推理，这在本地环境下根本不可接受。必须先做快速召回（BM25+向量），把候选集缩小到几十到几百个，再用 CrossEncoder 精排。

### Q2：CrossEncoder 和 Bi-encoder 的本质区别？

Bi-encoder 是"先分别编码，再算相似度"——编码时看不到对方的信息；CrossEncoder 是"拼在一起编码"——能看到完整上下文，做交叉注意力。这就像考试时：Bi-encoder 是闭卷（只看自己的材料），CrossEncoder 是开卷（能看到题目和材料）。

### Q3：为什么不一直用 CrossEncoder？

精度和速度的 trade-off。在召回阶段需要的是"快速捞回可能相关的文档"，精度要求不高；在精排阶段需要的是"从候选中挑出最相关的"，精度要求高。两阶段结合是最优方案。

### Q4：如果文档很长怎么办？

CrossEncoder 有长度限制（通常 512 或 1024 token）。如果文档太长，会被截断。但在 RAG 场景下，我们已经把文档切成了 chunk（通常 512 token），正好符合模型的输入要求。

### Q5：和 Rerank 相关的还有哪些方法？

- **ColBERT**：用 token-level 的交互，介于 Bi-encoder 和 CrossEncoder 之间，精度高且支持高效检索
- **Dense Passage Retrieval (DPR)**：纯 bi-encoder 检索，就是向量检索的一种
- **SPLADE**：把 sparse 和 dense 结合，让 BM25 也能学到语义信息

---

## 九、核心概念速记

| 术语 | 一句话解释 |
|------|-----------|
| CrossEncoder | 把 query 和 document 拼在一起编码，做交叉注意力 |
| Bi-encoder | 分别编码 query 和 document，再算相似度 |
| 交叉注意力 | 模型能看到两个片段的完整信息并做交互 |
| 召回阶段 | BM25+向量快速捞回候选 |
| 精排阶段 | CrossEncoder 对候选逐一打分排序 |
| ms-marco-MiniLM-L-6-v2 | 常用的轻量级 CrossEncoder 模型 |

---

## 十、三步骤总结

| 步骤 | 名称 | 核心操作 | 为什么重要 |
|------|------|---------|-----------|
| 1 | 输入拼接 | query + "[SEP]" + document | 让模型知道边界，做交叉注意力 |
| 2 | 模型推理 | CrossEncoder 输出相关性分数 | 精度高，能理解语义关系 |
| 3 | 排序输出 | 按分数排序，保留前 K | 挑出最相关的文档 |

**一句话记忆**：CrossEncoder = 开卷考试（能看到题目和材料），三步骤：拼接 → 推理 → 排序，只在精排阶段用。
