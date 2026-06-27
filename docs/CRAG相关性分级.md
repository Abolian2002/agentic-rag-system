# CRAG 相关性分级详解（面试学习笔记）

本文档详细讲解 retrieve 节点中的 CRAG（Corrective RAG）相关性分级技术，包括核心概念、四步工作流程、项目配置和面试要点。

---

## 一、什么是 CRAG？

**CRAG = Corrective RAG（纠正性 RAG）**

核心思想：**不是所有检索结果都能用**。如果检索回来的文档和问题不相关，直接送给 LLM 生成答案，就会出现"幻觉"——LLM 硬编一个不靠谱的回答。

CRAG 的做法是：
1. **先判断检索结果是否相关**
2. 如果不相关 → **触发纠正机制**（二次检索、查询扩展）
3. 如果相关 → 才送给生成节点

类比：考试时如果资料不对，先换一本正确的参考书，再回答问题。

---

## 二、为什么需要相关性分级？

### 传统 RAG 的问题

传统 RAG 假设"检索回来的文档就是答案"——但这往往是错的：

| 场景 | 检索结果 | 问题 |
|------|---------|------|
| 问"LangGraph 怎么用" | 检索回一篇 LangChain 的教程 | 内容不相关，LLM 会编答案 |
| 问"Python 如何处理 JSON" | 检索回一篇 Java 的 JSON 解析文章 | 语言不对，答案错误 |
| 问"报销流程" | 检索回 10 篇文档，其中 2 篇相关 | 相关信息被噪音淹没 |

**CRAG 的解决方案**：在送给 LLM 之前，先判断"这些文档真的能回答这个问题吗？"

---

## 三、CRAG 的工作流程（四步骤）

### 步骤 1：文档逐个判断（LLM-as-Judge）

把 query 和每个文档拼接成 prompt，让 LLM 判断这个文档是否相关。

**判断 prompt 结构**：
```
你是一个相关性判断助手。

用户问题：{query}

检索到的文档：
{document}

请判断这个文档是否能回答用户的问题。
输出：relevant / ambiguous / irrelevant
```

**三种相关性等级**：

| 等级 | 英文 | 含义 | 示例 |
|------|------|------|------|
| **相关** | `relevant` | 文档内容能直接回答问题 | 问 LangGraph 用法，文档讲 StateGraph |
| **模糊** | `ambiguous` | 文档部分相关，但不够清晰 | 问 LangGraph 用法，文档讲 LangChain 基础 |
| **不相关** | `irrelevant` | 文档内容和问题无关 | 问 LangGraph 用法，文档讲 PyTorch |

**判断方式**：LLM-as-Judge——用 LLM 自己做裁判，判断文档是否相关。

---

### 步骤 2：计算整体 confidence 分数

统计所有文档的判断结果，计算整体相关性分数：

```
confidence = relevant 文档数 / 总文档数
```

**示例计算**：

| 场景 | 文档 1 | 文档 2 | 文档 3 | 文档 4 | confidence |
|------|--------|--------|--------|--------|-----------|
| 全相关 | relevant | relevant | relevant | relevant | 4/4 = 1.0 |
| 大部分相关 | relevant | relevant | relevant | ambiguous | 3/4 = 0.75 |
| 一半相关 | relevant | relevant | irrelevant | irrelevant | 2/4 = 0.5 |
| 全不相关 | irrelevant | irrelevant | irrelevant | irrelevant | 0/4 = 0 |

---

### 步骤 3：阈值判断，决定是否重试

用 confidence 和阈值比较，决定下一步：

**本项目配置**：
- **confidence ≥ 0.7** → 相关性足够，正常生成答案
- **confidence < 0.7** → 触发二次检索（CRAG 循环）

**阈值为什么是 0.7？**

| 阈值 | 效果 | 问题 |
|------|------|------|
| 太低（0.4） | 很少重试 | 容易把不相关文档送进去生成，幻觉多 |
| 太高（0.9） | 几乎每次都重试 | 效率太低，增加延迟 |
| **0.7（推荐）** | 平衡质量和效率 | 经验最优值 |

---

### 步骤 4：触发纠正机制（二次检索）

当 confidence < 0.7 时，CRAG 触发纠正机制，做三件事：

#### 4.1 查询扩展

把原来的 query 扩展成更具体、更多角度的问题，提升召回率。

**示例**：
- 原问题："LangGraph 怎么用"
- 扩展后：
  - "LangGraph 的基本概念是什么"
  - "LangGraph StateGraph 的使用方法"
  - "LangGraph 的节点和边如何定义"

本项目用的是更宽泛的改写：`"Tell me about: {query}"`

#### 4.2 增加检索数量

原来只检索 Top-4 个文档，二次检索时扩大到 Top-6，捞回更多候选，避免漏掉相关文档。

#### 4.3 重试计数 + 1

记录这是第几次重试，最多重试 1 次，避免无限循环。

**重试流程示意图**：

```
第一次检索 → confidence < 0.7 → 查询扩展 + 增加数量 → 第二次检索
                                                  ↓
                                             confidence < 0.7？
                                                  ↓
                                          是 → 标记低置信度，生成答案
                                          否 → 正常生成答案
```

**为什么最多重试 1 次？**

- 每次重试都要调用 LLM 做判断 + 重新检索，成本增加
- 如果第一次检索就不相关，说明问题本身可能有问题，或者知识库缺失这个领域的内容
- 重试多次后可能还是不相关，浪费计算

---

## 四、项目中的实际实现

CRAG 在 LangGraph 工作流中通过**条件边 + 循环**实现：

```
START → route → check_cache → transform → retrieve
                                         ↓
                              confidence < 0.7 且重试次数 < 1？
                                         ↓
                                    是 → 回到 transform（循环）
                                    否 → generate → verify → finalize → END
```

看 [nodes.py](file:///Users/scm/code/CORTEX-AI-SUPER-RAG/src/graph/nodes.py) 中的 `node_retrieve` 函数：

```python
# 在 retrieve 节点中调用 CRAG
confidence = crag_judge(query, documents)

# 根据 confidence 决定下一步
if confidence < 0.7 and retrieval_attempts < 1:
    # 触发二次检索
    return {
        "confidence": confidence,
        "route": "retry",
        "retrieval_attempts": retrieval_attempts + 1
    }
else:
    # 正常生成
    return {
        "documents": documents,
        "confidence": confidence,
        "route": "generate"
    }
```

**LangGraph 的条件边实现循环**：
- `retrieve` → `transform`（低置信度时）
- `transform` → `retrieve`（二次检索）
- 形成一个循环，最多执行一次

---

## 五、CRAG 和 Self-RAG 的区别

| 维度 | CRAG | Self-RAG |
|------|------|---------|
| 判断时机 | 检索后，生成前 | 生成过程中实时判断 |
| 判断内容 | 文档相关性 | 答案片段是否有支撑 |
| 纠正方式 | 二次检索 | 重新生成 / 补充检索 |
| 复杂度 | 低 | 高 |
| 适用场景 | 大多数 RAG 系统 | 需要精细质量控制的场景 |

Self-RAG 更细粒度，但更复杂；CRAG 更简单实用，适合大多数场景。

---

## 六、面试怎么讲？（30秒版）

> CRAG 是 Corrective RAG 的简称，核心是在生成答案前先判断检索结果的相关性。我们用 LLM-as-Judge 的方式，把每个文档和 query 一起送给 LLM，输出 relevant/ambiguous/irrelevant 三种判断，然后计算 confidence 分数。如果 confidence < 0.7，说明检索结果不够好，就触发二次检索——扩展查询、增加检索数量。我们限制了最多重试一次，避免无限循环。这样就能避免把不相关的文档送给 LLM 硬编答案，减少幻觉问题。

---

## 七、面试高频追问

### Q1：为什么用 LLM-as-Judge 而不是人工规则？

人工规则（比如关键词匹配）太粗糙——无法判断语义相关性。LLM 能理解"这段文档是否真的能回答这个问题"，判断更准确。当然，LLM 判断也有成本，所以只对检索后的候选文档做判断，数量有限。

### Q2：如果二次检索还是不相关怎么办？

标记低置信度，但还是会送入生成节点——不过答案里会提示"可能不准确"。这样不会让用户等太久，同时也提醒用户答案可能有问题。

### Q3：CRAG 会增加多少延迟？

每次判断要调用 LLM（假设 4 个文档，每次约 50-100ms），加上二次检索（约 300-500ms），整体增加约 500ms。但相比生成错误答案再重来，这个成本是值得的。

### Q4：如果知识库本身就没有相关内容怎么办？

CRAG 会判断所有文档都不相关（confidence = 0），然后可以：
- 拒绝回答（提示用户"知识库中没有相关内容"）
- 转外部搜索（Web Search）
- 或标记低置信度生成一个回答

这取决于产品设计。我们的做法是标记低置信度，提示用户答案可能不准确。

### Q5：阈值 0.7 是怎么定的？

经验值。我们在个人知识库场景下做过测试：
- 0.5 以下：太多不相关文档混进去，幻觉明显
- 0.7 左右：既能过滤掉大部分噪音，又不会过度重试
- 0.8 以上：几乎每次都重试，延迟太高

如果换一个场景（比如企业知识库），这个阈值可能需要调整。

---

## 八、核心概念速记

| 术语 | 一句话解释 |
|------|-----------|
| CRAG | Corrective RAG，检索后判断相关性再生成 |
| LLM-as-Judge | 用 LLM 判断文档是否相关 |
| confidence | 相关性分数 = relevant 文档数 / 总文档数 |
| 相关性等级 | relevant / ambiguous / irrelevant |
| 二次检索 | confidence < 阈值时触发纠正机制 |
| 重试阈值 | 0.7（经验值） |
| 重试次数 | 最多 1 次 |

---

## 九、四步骤总结

| 步骤 | 名称 | 核心操作 | 为什么重要 |
|------|------|---------|-----------|
| 1 | 文档逐个判断 | LLM-as-Judge 输出 relevant/ambiguous/irrelevant | 区分相关和不相关文档 |
| 2 | 计算 confidence | relevant 文档数 / 总文档数 | 量化整体相关性 |
| 3 | 阈值判断 | ≥0.7 正常生成，<0.7 触发重试 | 平衡质量和效率 |
| 4 | 触发纠正机制 | 查询扩展 + 增加检索数量 + 重试计数 +1 | 提升召回率，避免无限循环 |

**一句话记忆**：CRAG = 判断 → 打分 → 比较 → 纠正，四步流程，用 LLM-as-Judge 做相关性分级，最多重试一次。
