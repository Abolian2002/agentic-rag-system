# Agentic RAG 工作流详解（面试学习笔记）

本文档从"用户输入"到"最终输出"完整走一遍系统的 LangGraph 工作流，作为面试准备的核心参考。

---

## 一、整体概述

系统将整个 RAG 流程建模为 **LangGraph 驱动的循环状态机（Cyclic State Graph）**，而非传统的线性函数链。每个节点承担单一职责，接收状态快照并返回**部分字段的增量更新**，条件边（Conditional Edge）根据中间结果（缓存是否命中、路由类型、检索置信度等）动态决定下一节点。

**State（状态）** 是贯穿单次推理链路的**类型化结构化数据载体**（基于 TypedDict 定义），采用**黑板模式（Blackboard Pattern）**——所有节点通过读写 State 交换数据，节点之间不直接耦合。State 存储了完整流程的输入、中间产物与最终输出，包括：原始输入（query、chat_history）、中间决策（route、cache_hit、confidence、retrieval_attempts）、中间产物（transformed_queries、documents、thinking、generation）以及可观测性数据（steps、timeline）。每个节点读取当前 State 快照，计算后仅返回自己修改的字段，由运行时自动合并回全局 State。

**运行时（Runtime）** = 编译后的图执行器 + MemorySaver 检查点机制 + 通过 config 注入的外部资源（检索器、模型客户端、缓存等）。Runtime 负责调度节点、沿条件边路由、合并状态增量、持久化检查点，是 State 的管理者和执行者。

### 四种可能的执行路径

| 场景 | 执行路径 | 节点数 |
|------|---------|--------|
| 语义缓存命中（重复/近义问题） | route → check_cache → finalize | 3 |
| 闲聊/无检索路径 | route → check_cache → generate → verify → finalize | 5 |
| 正常 RAG 问答（一次检索成功） | route → check_cache → transform → retrieve → generate → verify → finalize | 7 |
| CRAG 低置信度重试（首次检索失败） | route → check_cache → transform → retrieve → **transform → retrieve** → generate → verify → finalize | 最多 9 |

### 工作流拓扑图

```
                         ┌──────────────┐
                         │    START     │
                         └──────┬───────┘
                                ▼
                         ┌──────────────┐
                         │    route     │  查询意图路由
                         └──────┬───────┘
                                ▼
                         ┌──────────────┐
                    ┌───►│  check_cache │──┐
                    │    └──────────────┘  │
               HIT  │                      │ MISS (chat)
                    │                      ▼
                    │               ┌──────────────┐
                    │               │   generate   │  直接回答（无检索）
                    │               └──────┬───────┘
                    │                      │
                    │               ┌──────┴───────┐
                    │               │    verify    │  引用核查
                    │               └──────┬───────┘
                    │                      │
                    │    ┌─────────────────┘
                    │    │
                    │    ▼
                    │ ┌──────────────┐
                    │ │   finalize   │  写入缓存 · 返回结果
                    │ └──────┬───────┘
                    │        │
                    │        ▼
                    │   ┌─────────┐
                    └──►│   END   │
                        └─────────┘

   MISS (rag/summary)
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │  transform   │────►│   retrieve   │  查询改写 / 多路检索
  └──────▲───────┘     └──────┬───────┘
         │  低置信度           │
         └────────────────────┘         CRAG 循环（最多重试1次）
```

---

## 二、七大节点详解

### 节点 1：route — 查询意图路由

**职责**：判断用户问题属于哪一类，决定整条链路走向。

**核心逻辑**：
- 若当前没有加载任何文档 → 强制路由为 `chat`，因为没有可检索的知识库。
- 否则调用 LLM 做意图分类，将 query 分为三类：
  - **rag**：需要基于文档回答的知识型问题（如"RAG 和微调的区别是什么"）。
  - **chat**：问候、闲聊、元问题或可由模型参数化知识直接回答的问题（如"你好"、"你是谁"）。
  - **summary**：要求总结文档的请求（如"总结一下这份报告"）。
- 路由结果及原因写入执行时间线（steps），用于 UI 展示和调试。

**为什么需要路由**：不是所有问题都需要跑检索。闲聊类问题强行做检索反而会引入噪声上下文，浪费时间且可能误导生成。路由让"是否检索"成为一个显式决策点。

**关键词**：意图识别、query classification、LLM router、intent classification。

---

### 节点 2：check_cache — 三级语义缓存闸门

**职责**：在做任何昂贵计算之前，检查缓存中是否已有答案，命中则直接短路返回。

**三级缓存架构**：

- **第一级：Embedding 缓存**（精确匹配，key = 文本 hash）
  存储已经计算过的 query 向量。同一次请求的不同节点（缓存检查需要 embedding、写回缓存又需要同一个 embedding）以及跨会话的相同字符串，都可直接复用，避免重复调用本地 embedding 模型。

- **第二级：检索结果缓存**（精确匹配，key = 文本 hash）
  缓存"query → 命中的 chunk 列表"映射。"用户精确重复同一问题时"，直接返回上次检索到的文档chunk，跳过 BM25 + 向量 + GraphRAG + Rerank 整条检索管线。

- **第三级：答案缓存**（语义匹配，余弦相似度）
  存储完整的答案、思维链、来源列表。与前两级不同，它不是按字符串精确匹配，而是将新 query 的 embedding 与缓存中所有历史 query 的 embedding 逐一计算余弦相似度，若相似度 ≥ 0.92，则视为语义等价（如"LangGraph 是什么"和"解释一下 LangGraph"会命中同一答案），整条后续链路直接跳过。

**未命中后的分支**：
- cache_hit = False，根据 route 类型决定走向：`chat` → 直接去 generate；`rag/summary` → 去 transform 做查询改写。

**关键词**：semantic cache、cosine similarity、embedding memoization、short-circuit、write-through、tiered cache。

---

### 节点 3：transform — 查询改写与扩展

**职责**：把原始 query 改写成更适合检索的形式，缓解 query 与文档之间的词汇鸿沟（lexical gap）。

**三种改写策略**（根据配置和是否重试决定）：

- **HyDE（Hypothetical Document Embeddings，假设文档嵌入）**
  让 LLM 先"幻想"一篇能够回答该问题的理想文档段落，用这段假设性文字代替原始 query 去做检索。因为假设文档的词汇分布更接近真实文档，能显著缩小短问题与长文档之间的词汇差异。

- **RAG-Fusion（多路查询融合）**
  让 LLM 从不同角度生成 N（通常为 3）个查询变体，每个变体独立检索，之后通过 **Reciprocal Rank Fusion（RRF，k=60）** 融合多路排序结果。降低单一 query 表述带来的召回脆弱性。

- **重试扩展（Retry Expansion）**
  如果是从 retrieve 节点循环回来的第二次尝试（CRAG 判断首次检索低置信度），使用更宽泛的改写（如加前缀 "Tell me about:"），扩大召回范围。

改写后的查询列表（transformed_queries）写入状态，供 retrieve 节点使用。

**关键词**：HyDE、query rewriting、query expansion、RAG-Fusion、Reciprocal Rank Fusion (RRF)、lexical gap。

**RRF** 是 Reciprocal Rank Fusion，用于融合多个检索器的排序结果。它不管文档的绝对分数，只看排名位置，公式是 Σ 1/(k + rank(d))。k=60 是平滑常数，避免除零并控制排名差异的权重。
一句话记忆 ：RRF = 只看排名不看分数，1/(k+rank（d）) 加总，多检索器共识越多的文档排名越靠前。
---

### 节点 4：retrieve — 多路检索 + 重排 + CRAG 相关性分级

**职责**：核心检索引擎，分阶段执行"召回 → 融合 → 重排 → 过滤"的完整管线，在保证召回率的前提下提升精度。

**六阶段管线**：

1. **BM25 稀疏检索**
   基于词频-逆文档频率（TF-IDF 家族）的经典词汇匹配算法。擅长精确关键词、专有名词、代码片段、数字等的匹配。

2. **Chroma 稠密向量检索**
   基于 embedding 语义相似度的近似最近邻搜索（ANN）。擅长处理同义词、释义、概念级匹配，弥补 BM25 的语义盲区。

3. **Ensemble 加权融合**
   通过 EnsembleRetriever 将 BM25（权重 0.35）和向量检索（权重 0.65）的结果加权合并，平衡词汇精度和语义召回。

4. **GraphRAG（可选）**
   基于 NetworkX 构建实体共现图：在同一滑动窗口内共同出现的实体之间建立边。从 query 中提取实体后沿图游走，召回与问题实体在文中紧密关联的 chunk，补充纯向量搜索容易遗漏的跨段关联信息。

5. **CrossEncoder 神经重排（可选）**
   使用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 对 Top-K 候选 chunk 重新打分。CrossEncoder 对 query 和 chunk 做完整交叉注意力（cross-attention），相比 bi-encoder（向量独立编码比较）精度显著更高，但计算代价大，因此只对短候选列表使用。

6. **CRAG（Corrective RAG）相关性分级（可选）**
   调用 LLM 对剩余每个 chunk 逐一进行相关性评判：相关则保留，不相关则丢弃。若**全部** chunk 被判定为不相关，置 confidence=0.2、needs_retry=True，条件边触发循环回到 transform 节点，用更宽泛的查询做第二次尝试。最多重试 1 次，避免死循环。若为重试，max_contexts 会从 4 扩展到 6，召回更多文档。

最终保留的 chunk、置信度分数、调试元数据写入状态，流向生成节点。

**关键词**：hybrid search、sparse retrieval、dense retrieval、ANN、EnsembleRetriever、GraphRAG、entity co-occurrence graph、NetworkX、CrossEncoder reranking、bi-encoder vs cross-encoder、CRAG (Corrective RAG)。

---

### 节点 5：generate — 流式答案生成

**职责**：根据路由类型组装 Prompt，调用 Ollama 流式生成答案，并实时解析思维链标签。

**三种 Prompt 模板**：

- **summary 模式**：要求对提供的文档做结构化要点总结。
- **rag 模式**：
  - 注入最近 6 轮聊天历史，支持多轮对话。
  - 将检索到的 chunk 按 `[Source 1] ... [Source 2] ...` 格式拼入 context。
  - 明确指令：必须基于 context 回答、引用 `[Source N]`、找不到答案时明确说不知道。
  - 若 confidence < 0.3（CRAG 判定相关性不足但仍生成），追加不确定性提示。
  - 启用 `<think>...</think>` 思维链（Chain-of-Thought）指令，要求模型在正式回答前先在 think 标签内分步推理。
- **chat 模式**：仅携带聊天历史和用户问题，不注入任何检索文档。

**实时流式标签解析**：
逐 token 接收 Ollama 的 streaming 响应，维护一个小缓冲区，实时识别 `<think>` 和 `</think>` 分隔符：标签内的 token 归入 `thinking`（思维链）字段，标签外的归入 `answer`（最终答案）字段。前端据此分别渲染推理过程和最终答案，最终答案通过打字机效果逐字呈现。

**关键词**：streaming generation、Chain-of-Thought (CoT)、`<think>` tag parsing、prompt engineering、citation prompting、context assembly、typewriter effect。

---

### 节点 6：verify — 引用核查

**职责**：生成后的事实一致性校验，聚焦于**引用完整性**。

以 LLM-as-Judge 的方式，将答案中的 `[Source N]` 引用与真实检索到的 source chunk 逐一比对，验证两点：
- 每个被引用的来源确实支持其对应的断言（不张冠李戴）。
- 没有凭空编造不存在的来源（引用幻觉 / citation hallucination）。

若未启用该功能或无 sources（chat 模式），直接跳过，视为通过。核查结果（通过/失败 + 具体问题）记录到执行时间线。

**为什么引用核查必要**：LLM 非常擅长编造看起来合理但与原文不符的引用编号，这一步是 post-hoc（事后）质量兜底，是 Agentic 系统"自我反思"能力的体现。

**关键词**：citation verification、LLM-as-judge、hallucination detection、post-hoc verification、self-reflection。

---

### 节点 7：finalize — 缓存写回与结果收尾

**职责**：终端节点，把本次结果持久化并输出最终响应。

- 若本次结果**不是**从缓存命中得到（cache_hit=False），则将 query、其 embedding、生成的 answer、thinking、sources 写入答案缓存，同时补齐 embedding 缓存和检索结果缓存，供未来语义相似问题直接复用。
- 追加最后一步 step 到执行时间线。
- 条件边路由至 END，状态机终止，最终结果（answer、thinking、sources、confidence、route、steps）返回给调用方（Streamlit UI）。

**关键词**：cache write-through、persistence、result assembly。

---

## 三、关键面试 Q&A 速查

### Q1：为什么用 LangGraph 而不是普通 LangChain Chain（LCEL）？

LCEL 链是 DAG（有向无环图），天然不支持循环。CRAG 低置信度重试需要从 retrieve 回到 transform，这是一个**回边（back-edge）**，只有状态机模型（StateGraph）能原生表达。此外 LangGraph 提供：显式类型化 State、条件边路由、内置 MemorySaver 会话检查点、以及每步可观测可中断的执行模型。

### Q2：为什么 Chroma 代替 FAISS？

FAISS 是一个相似度搜索库，不是数据库；Chroma 是持久化向量数据库，支持文档 CRUD、元数据过滤、集合管理，是实现增量索引（基于文件 Hash 只更新变动文档）的必要条件。

### Q3：三级缓存命中率最高的是哪一层？为什么？

第三层（答案级语义缓存）。前两层是字符串精确匹配，用户几乎不会一字不差重复问题；第三层基于余弦相似度，同义不同表述的问题都能命中，体验上是"智能复用"。

### Q4：HyDE 和 RAG-Fusion 的核心区别？

- HyDE 生成**一篇假设答案**，用它 embedding 去检索——核心是"用接近文档分布的文本替代短 query"。
- RAG-Fusion 生成**多个 query 变体**，并行检索后用 RRF 融合排序——核心是"多角度投票，降低单 query 脆弱性"。

### Q5：为什么 CrossEncoder 只重排 Top-K 而不直接用于全量检索？

CrossEncoder 需要把 query 和每个 chunk 拼在一起做一次完整的 Transformer 前向推理，时间复杂度 O(N)，N 为全库 chunk 数时完全不可接受。先用 Bi-Encoder（向量检索）+ BM25 快速召回 Top-50~100 候选，再用 CrossEncoder 精排，是经典的"召回粗排 → 精排"两阶段架构。

### Q6：GraphRAG 补充了什么纯向量检索做不到的事？

向量检索基于语义相似度，但"相关"不等于"共现实体关联"。当问题涉及多个实体之间的关系（如"A 算法和 B 框架如何集成"），相关信息可能分散在不同 chunk 中通过实体相互引用。GraphRAG 通过实体共现图把这种"跨 chunk 实体链路"显式建模，召回向量搜索容易漏掉的关联 chunk。我们用的是轻量级 NetworkX 共现图，相比微软的 GraphRAG（需要社区检测、摘要生成、大量 LLM 预算）更适合个人知识库场景。

### Q7：CRAG 为什么只重试一次？为什么不无限重试？

检索问题本身具有边界——如果两次扩展后仍找不到相关内容，说明文档中确实没有答案，继续重试只会浪费 token 和时间并产生幻觉。设上限是"优雅失败"的工程化体现：低置信度答案会在 prompt 中注入不确定性提示，让模型明确告知用户信息不足，而不是编造。

### Q8：MemorySaver 是什么？为什么需要它？

LangGraph 的 Checkpointer 实现，用于在图执行过程中持久化每一步的 State 快照。它让多轮对话中跨请求的状态得以保留，也支持中断、恢复、人工审批等高级 Agent 模式。本项目用 thread_id 隔离不同会话的状态。
