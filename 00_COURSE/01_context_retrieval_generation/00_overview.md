# 上下文检索与生成
## 从静态提示词到动态知识编排

> **模块 01** | *上下文工程课程：从基础到前沿系统*
>
> 基于[上下文工程综述](https://arxiv.org/pdf/2507.13334) | 推进软件3.0范式

---

## 学习目标

在本模块结束时，你将理解并实现：

- **高级提示词工程**：从基础提示词到复杂推理模板
- **外部知识集成**：RAG基础和动态知识检索
- **动态上下文组装**：多源信息的实时组合
- **策略性上下文编排**：优化信息载荷以实现最大模型效能

---

## 概念演进：从静态文本到智能知识编排

将上下文生成想象成我们向解决问题的人提供信息的演变过程——从递给他们一份文档，到组织一个研究图书馆，再到拥有一个智能研究助手，它准确知道要收集什么信息以及如何呈现它。

### 阶段 1：静态提示词工程
```
"解决这个问题：[问题描述]"
```
**上下文**：就像给某人一张指令单。简单直接，但受限于你能装进一份文档的内容。

### 阶段 2：增强提示词模式
```
"让我们一步一步思考：
1. 首先，理解问题...
2. 然后考虑方法...
3. 最后实现解决方案..."
```
**上下文**：就像提供一个结构化的方法论。更有效，因为它引导思考过程，但仍然受限于静态内容。

### 阶段 3：外部知识集成
```
[从知识库检索相关信息]
"给定以下上下文：[外部知识]
现在解决：[问题]"
```
**上下文**：就像拥有一个研究图书馆的访问权限。更强大得多，因为它可以包含超出工作记忆容量的专业、最新信息。

### 阶段 4：动态上下文组装
```
Context = Assemble(
    task_instructions +
    relevant_retrieved_knowledge +
    user_history +
    domain_expertise +
    real_time_data
)
```
**上下文**：就像拥有一个研究助手，他从多个来源准确收集正确的信息，并为你的特定任务进行最优组织。

### 阶段 5：智能上下文编排
```
自适应上下文系统：
- 理解你的目标和约束
- 监控你的进度并调整信息流
- 从结果中学习以改进未来的上下文组装
- 平衡相关性、完整性和认知负荷
```
**上下文**：就像拥有一个AI研究伙伴，它不仅理解你需要知道什么，还理解你如何思考和学习，持续优化信息环境以实现最大效能。

---

## 数学基础

### 上下文形式化框架
从我们的核心数学基础：
```
C = A(cinstr, cknow, ctools, cmem, cstate, cquery)
```

在本模块中，我们主要关注 **cknow**（外部知识）和组装函数 **A**，特别是：

```
cknow = R(cquery, K)
```

其中：
- **R** 是检索函数
- **cquery** 是用户的即时请求
- **K** 是外部知识库

### 信息论优化
最优检索函数最大化相关信息：
```
R* = arg max_R I(Y*; cknow | cquery)
```

其中 **I(Y*; cknow | cquery)** 是目标响应 **Y*** 与检索到的知识 **cknow** 之间的互信息，给定查询 **cquery**。

**直观解释**：我们想要检索能告诉我们最多关于正确答案应该是什么的信息。这就像一位熟练的图书管理员，他不仅仅是找到关于你主题的书籍，而是找到包含你所需确切见解的特定书籍。

### 动态组装优化
```
A*(cinstr, cknow, cmem, cquery) = arg max_A P(Y* | A(...)) × Efficiency(A)
```

受以下约束：
- `|A(...)| ≤ Lmax`（上下文窗口限制）
- `Quality(cknow) ≥ threshold`（信息质量阈值）
- `Relevance(cknow, cquery) ≥ min_relevance`（相关性阈值）

**直观解释**：组装函数就像一位大师级编辑，他知道如何将不同的信息片段组合成一个连贯、有效的简报，在实际限制内最大化获得优秀响应的机会。

---

## 可视化架构：上下文工程栈

```
┌─────────────────────────────────────────────────────────────┐
│                    上下文组装层                            │
│  ┌─────────────────┬────────────────┬─────────────────────┐ │
│  │     指令        │      知识      │        编排         │ │
│  │                 │                │                     │ │
│  │  • 任务规格     │  • 检索到的    │  • 组装逻辑         │ │
│  │  • 约束条件     │    文档        │  • 优先级排序       │ │
│  │  • 示例         │  • 实时        │  • 格式化           │ │
│  │  • 格式规则     │    数据        │  • 长度管理         │ │
│  └─────────────────┴────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│                   知识检索层                               │
│  ┌─────────────────┬────────────────┬─────────────────────┐ │
│  │   查询处理      │     检索       │      知识库         │ │
│  │                 │                │                     │ │
│  │  • 查询分析     │  • 向量        │  • 文档             │ │
│  │  • 意图提取     │    搜索        │  • 数据库           │ │
│  │  • 扩展         │  • 语义        │  • API              │ │
│  │  • 过滤         │    匹配        │  • 实时数据         │ │
│  └─────────────────┴────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│                    提示词工程层                            │
│  ┌─────────────────┬────────────────┬─────────────────────┐ │
│  │  基础提示词     │     模板       │     推理链          │ │
│  │                 │                │                     │ │
│  │  • 直接指令     │  • 可复用      │  • 思维链           │ │
│  │  • 少样本       │    模式        │  • 思维树           │ │
│  │  • 零样本       │  • 领域        │  • 自洽性           │ │
│  │  • 基于角色     │    特定        │  • 反思             │ │
│  └─────────────────┴────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**自底向上解释**：这个栈展示了上下文工程如何从基础提示词构建到复杂的信息编排。每一层都增加了能力：
- **底层**：核心提示词工程——如何与LLM有效沟通
- **中层**：知识检索——如何查找和访问相关外部信息
- **顶层**：上下文组装——如何最优地组合一切

---

## 软件3.0范式1：提示词（策略性模板）

上下文工程中的提示词超越了简单的指令，成为信息收集和推理的策略性模板。

### 高级推理模板
```markdown
# 思维链推理框架

## 上下文评估
你的任务是[特定任务]，需要深入分析和逐步推理。
考虑复杂性、可用信息和推理要求。

## 信息清单
**可用上下文**：{上下文摘要}
**缺失信息**：{信息缺口}
**所需假设**：{必要假设}
**推理类型**：{演绎|归纳|溯因|类比}

## 结构化推理过程

### 步骤 1：问题分解
将主要问题分解为子问题：
1. {子问题_1}
2. {子问题_2}
3. {子问题_3}

### 步骤 2：证据分析
对于每个子问题，分析可用证据：
- **支持性证据**：[列出相关支持信息]
- **矛盾性证据**：[列出冲突信息]
- **证据质量**：[评估可靠性和相关性]
- **证据缺口**：[识别缺失的关键信息]

### 步骤 3：推理链构建
在证据和结论之间建立逻辑连接：

前提 1：[带有证据的陈述]
    ├─ 支持细节 A
    ├─ 支持细节 B
    └─ 置信水平：[高/中/低]

前提 2：[带有证据的陈述]
    ├─ 支持细节 C
    ├─ 支持细节 D
    └─ 置信水平：[高/中/低]

中间结论：[从前提推导的逻辑推论]
    └─ 推理：[解释逻辑连接]


### 步骤 4：替代假设考虑
还有哪些其他可能的解释或解决方案？
- **替代方案 1**：[不同的解释/方法]
  - 优势：[支持这个替代方案的内容]
  - 劣势：[反对它的内容]
- **替代方案 2**：[另一个解释/方法]
  - 优势：[支持因素]
  - 劣势：[限制因素]

### 步骤 5：综合与结论
**主要结论**：[主要答案/解决方案]
**置信水平**：[百分比或定性评估]
**关键推理**：[导致这个结论的最关键逻辑步骤]
**局限性**：[什么可能使这个结论错误]
**下一步**：[哪些额外信息将加强结论]

## 质量保证
- [ ] 我是否解决了所有子问题？
- [ ] 我的逻辑连接是否明确且有效？
- [ ] 我是否考虑了主要的替代解释？
- [ ] 我的置信度评估是否现实？
- [ ] 其他人能否跟随我的推理链？
```

**自底向上解释**：这个模板将简单的"让我们一步一步思考"方法转变为一个全面的推理方法论。它就像有一位逻辑大师指导你的思维过程，确保你考虑所有角度，建立明确的连接，并评估你自己的推理质量。

### 动态知识集成模板
```xml
<knowledge_integration_template>
  <intent>系统性地将外部知识与用户查询集成以获得最优响应</intent>

  <context_analysis>
    <user_query>
      <main_intent>{主要用户目标}</main_intent>
      <sub_intents>
        <intent priority="high">{关键子目标}</intent>
        <intent priority="medium">{重要子目标}</intent>
        <intent priority="low">{可选子目标}</intent>
      </sub_intents>
      <complexity_level>{简单|中等|复杂|专家}</complexity_level>
      <domain_context>{特定领域或通用}</domain_context>
    </user_query>

    <information_needs>
      <critical_info>准确响应绝对需要的信息</critical_info>
      <supporting_info>能提高响应质量的信息</supporting_info>
      <contextual_info>提供有用背景的信息</contextual_info>
    </information_needs>
  </context_analysis>

  <knowledge_retrieval_strategy>
    <search_approach>
      <primary_search>{最可能找到关键信息}</primary_search>
      <secondary_search>{全面覆盖的备用方法}</secondary_search>
      <tertiary_search>{专业或边缘情况覆盖}</tertiary_search>
    </search_approach>

    <quality_filters>
      <relevance_threshold>信息必须与查询意图匹配的紧密程度</relevance_threshold>
      <credibility_threshold>最低来源可靠性标准</credibility_threshold>
      <recency_weight>最新信息与权威信息的优先级权重</recency_weight>
    </quality_filters>
  </knowledge_retrieval_strategy>

  <context_assembly>
    <information_hierarchy>
      <tier_1>直接回答主要问题的核心事实</tier_1>
      <tier_2>支持性证据和解释</tier_2>
      <tier_3>背景上下文和相关信息</tier_3>
    </information_hierarchy>

    <assembly_constraints>
      <max_context_length>{令牌限制考虑}</max_context_length>
      <cognitive_load_limit>用户理解的最大信息复杂度</cognitive_load_limit>
      <coherence_requirement>信息应该如何逻辑连接</coherence_requirement>
    </assembly_constraints>

    <assembly_process>
      <step name="prioritize">按相关性和重要性对检索到的信息进行排名</step>
      <step name="filter">删除冗余、过时或低质量的信息</step>
      <step name="structure">组织信息以获得逻辑流程和理解</step>
      <step name="integrate">将信息编织成解决用户查询的连贯叙述</step>
      <step name="validate">确保组装的上下文支持准确、有用的响应</step>
    </assembly_process>
  </context_assembly>

  <response_optimization>
    <tailoring>
      <user_expertise_level>适当调整技术深度</user_expertise_level>
      <communication_style>匹配用户偏好的交互模式</communication_style>
      <information_density>平衡全面性与清晰度</information_density>
    </tailoring>

    <quality_assurance>
      <accuracy_check>验证信息正确性和上下文对齐</accuracy_check>
      <completeness_check>确保所有关键用户需求都得到解决</completeness_check>
      <coherence_check>确认逻辑流程和清晰沟通</coherence_check>
    </quality_assurance>
  </response_optimization>
</knowledge_integration_template>
```

**自底向上解释**：这个XML模板结构化了查找和集成外部知识的复杂过程。它就像拥有一个研究方法论，确保你不仅找到相关信息，而且以对特定用户和任务最有效的方式组织和呈现它。

---

## 软件3.0范式2：编程（检索算法）

编程为智能上下文检索和组装提供了计算机制。

### 语义检索引擎

```python
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sqlite3
import json
from datetime import datetime, timedelta

@dataclass
class RetrievalCandidate:
    """可能与查询相关的一条信息"""
    content: str
    source: str
    relevance_score: float
    credibility_score: float
    recency_score: float
    content_type: str  # 'fact', 'procedure', 'example', 'definition'
    metadata: Dict

class KnowledgeRetriever(ABC):
    """不同知识检索策略的抽象基类"""

    @abstractmethod
    def retrieve(self, query: str, max_results: int = 10) -> List[RetrievalCandidate]:
        """为给定查询检索相关知识"""
        pass

    @abstractmethod
    def update_relevance_feedback(self, query: str, candidate: RetrievalCandidate,
                                 helpful: bool):
        """从用户关于检索质量的反馈中学习"""
        pass

class SemanticVectorRetriever(KnowledgeRetriever):
    """使用嵌入的语义相似度检索"""

    def __init__(self, embedding_model, vector_database):
        self.embedding_model = embedding_model
        self.vector_db = vector_database
        self.feedback_history = []

    def retrieve(self, query: str, max_results: int = 10) -> List[RetrievalCandidate]:
        """检索语义相似的内容"""

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode(query)

        # 搜索向量数据库
        raw_results = self.vector_db.similarity_search(
            query_embedding,
            top_k=max_results * 2  # 获取更多候选进行过滤
        )

        # 转换为带评分的RetrievalCandidate
        candidates = []
        for result in raw_results:
            candidate = RetrievalCandidate(
                content=result.content,
                source=result.source,
                relevance_score=self._calculate_relevance_score(query, result),
                credibility_score=self._calculate_credibility_score(result),
                recency_score=self._calculate_recency_score(result),
                content_type=self._classify_content_type(result.content),
                metadata=result.metadata
            )
            candidates.append(candidate)

        # 应用从反馈历史中学到的知识
        candidates = self._apply_feedback_learning(query, candidates)

        # 排名和过滤
        ranked_candidates = self._rank_candidates(candidates)

        return ranked_candidates[:max_results]

    def _calculate_relevance_score(self, query: str, result) -> float:
        """计算内容与查询的相关程度"""

        # 基础语义相似度
        base_score = result.similarity_score

        # 根据内容类型匹配进行调整
        content_type_bonus = self._get_content_type_bonus(query, result.content)

        # 根据查询特异性进行调整
        specificity_factor = self._calculate_query_specificity_factor(query, result)

        # 组合因素
        relevance_score = base_score * (1 + content_type_bonus) * specificity_factor

        return min(1.0, max(0.0, relevance_score))

    def _calculate_credibility_score(self, result) -> float:
        """评估来源可信度和信息质量"""

        # 来源权威性（学术、政府、成熟组织）
        source_authority = self._get_source_authority_score(result.source)

        # 内容质量指标（长度、结构、引用）
        content_quality = self._assess_content_quality(result.content)

        # 交叉引用验证（与其他来源的对齐程度）
        cross_reference_score = self._calculate_cross_reference_score(result)

        # 组合因素
        credibility = (source_authority * 0.4 +
                      content_quality * 0.3 +
                      cross_reference_score * 0.3)

        return credibility

    def _calculate_recency_score(self, result) -> float:
        """基于信息新近度评分（越新 = 分数越高）"""
        if 'date' not in result.metadata:
            return 0.5  # 无日期内容的中性分数

        content_date = datetime.fromisoformat(result.metadata['date'])
        days_old = (datetime.now() - content_date).days

        # 指数衰减：随着内容变旧，分数降低
        # 半衰期为365天（信息相关性每年减半）
        half_life = 365
        recency_score = 0.5 ** (days_old / half_life)

        return recency_score

    def _classify_content_type(self, content: str) -> str:
        """将内容分类为事实、程序、示例或定义"""

        # 简单的启发式分类（实际中使用ML分类器）
        content_lower = content.lower()

        if any(phrase in content_lower for phrase in ['step', 'first', 'then', 'finally', 'procedure']):
            return 'procedure'
        elif any(phrase in content_lower for phrase in ['for example', 'such as', 'instance']):
            return 'example'
        elif any(phrase in content_lower for phrase in ['is defined as', 'refers to', 'means']):
            return 'definition'
        else:
            return 'fact'

    def _rank_candidates(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """使用综合评分对候选进行排名"""

        for candidate in candidates:
            # 平衡多个因素的综合分数
            candidate.composite_score = (
                candidate.relevance_score * 0.5 +      # 相关性最重要
                candidate.credibility_score * 0.3 +    # 可信度非常重要
                candidate.recency_score * 0.2          # 新近度重要但较次
            )

        # 按综合分数排序
        ranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)

        return ranked

    def update_relevance_feedback(self, query: str, candidate: RetrievalCandidate,
                                 helpful: bool):
        """从反馈中学习以改进未来的检索"""

        feedback_entry = {
            'query': query,
            'candidate_source': candidate.source,
            'candidate_type': candidate.content_type,
            'helpful': helpful,
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_history.append(feedback_entry)

        # 基于反馈模式更新检索参数
        self._update_retrieval_parameters()

    def _apply_feedback_learning(self, query: str, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """基于学习到的反馈模式调整候选分数"""

        if not self.feedback_history:
            return candidates

        # 分析反馈模式
        feedback_patterns = self._analyze_feedback_patterns(query)

        # 基于模式调整分数
        for candidate in candidates:
            adjustment = self._calculate_feedback_adjustment(candidate, feedback_patterns)
            candidate.relevance_score = min(1.0, max(0.0, candidate.relevance_score + adjustment))

        return candidates

    def _analyze_feedback_patterns(self, query: str) -> Dict:
        """分析历史反馈以识别有用的模式"""

        patterns = {
            'helpful_sources': [],
            'helpful_content_types': [],
            'unhelpful_sources': [],
            'unhelpful_content_types': []
        }

        # 按有用性分组反馈
        for feedback in self.feedback_history[-100:]:  # 最近的反馈
            if self._is_similar_query(query, feedback['query']):
                if feedback['helpful']:
                    patterns['helpful_sources'].append(feedback['candidate_source'])
                    patterns['helpful_content_types'].append(feedback['candidate_type'])
                else:
                    patterns['unhelpful_sources'].append(feedback['candidate_source'])
                    patterns['unhelpful_content_types'].append(feedback['candidate_type'])

        return patterns

class HybridKnowledgeRetriever(KnowledgeRetriever):
    """结合多种检索策略以获得全面结果"""

    def __init__(self, retrievers: List[KnowledgeRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.performance_history = {i: [] for i in range(len(retrievers))}

    def retrieve(self, query: str, max_results: int = 10) -> List[RetrievalCandidate]:
        """从多个来源检索并智能地组合结果"""

        all_candidates = []

        # 从每个策略检索
        for i, retriever in enumerate(self.retrievers):
            try:
                candidates = retriever.retrieve(query, max_results)

                # 根据检索器性能对候选进行加权
                weight = self.weights[i] * self._get_dynamic_weight(i, query)

                for candidate in candidates:
                    candidate.composite_score *= weight
                    candidate.metadata['retriever_id'] = i

                all_candidates.extend(candidates)

            except Exception as e:
                print(f"检索器 {i} 失败: {e}")
                continue

        # 删除重复项并合并相似内容
        unique_candidates = self._deduplicate_candidates(all_candidates)

        # 对最终候选进行排名
        final_candidates = self._rank_hybrid_candidates(unique_candidates)

        return final_candidates[:max_results]

    def _get_dynamic_weight(self, retriever_id: int, query: str) -> float:
        """基于检索器对相似查询的性能计算动态权重"""

        if not self.performance_history[retriever_id]:
            return 1.0  # 新检索器的默认权重

        # 计算最近的性能平均值
        recent_performance = self.performance_history[retriever_id][-10:]  # 最后10个查询
        avg_performance = sum(recent_performance) / len(recent_performance)

        # 基于性能的动态权重（表现更好的获得更高权重）
        return max(0.1, min(2.0, avg_performance))

    def _deduplicate_candidates(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """删除重复和非常相似的候选"""

        unique_candidates = []
        content_hashes = set()

        for candidate in sorted(candidates, key=lambda c: c.composite_score, reverse=True):
            # 基于内容相似度的简单去重
            content_hash = hash(candidate.content[:200])  # 对前200个字符进行哈希

            if content_hash not in content_hashes:
                content_hashes.add(content_hash)
                unique_candidates.append(candidate)

        return unique_candidates

    def update_relevance_feedback(self, query: str, candidate: RetrievalCandidate, helpful: bool):
        """为提供此候选的特定检索器更新反馈"""

        retriever_id = candidate.metadata.get('retriever_id')
        if retriever_id is not None:
            # 更新性能历史
            performance_score = 1.0 if helpful else 0.0
            self.performance_history[retriever_id].append(performance_score)

            # 将反馈转发给特定检索器
            self.retrievers[retriever_id].update_relevance_feedback(query, candidate, helpful)

class DynamicContextAssembler:
    """从检索到的知识和其他来源组装最优上下文"""

    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.assembly_history = []

    def assemble_context(self, query: str, retrieved_candidates: List[RetrievalCandidate],
                        instructions: str = "", user_context: str = "",
                        task_type: str = "general") -> str:
        """从可用信息动态组装最优上下文"""

        # 分析查询以理解信息需求
        info_needs = self._analyze_information_needs(query, task_type)

        # 选择候选的最优子集
        selected_candidates = self._select_optimal_candidates(
            retrieved_candidates, info_needs, self.max_context_length
        )

        # 结构化和格式化上下文
        assembled_context = self._structure_context(
            instructions, selected_candidates, user_context, query, info_needs
        )

        # 验证和优化最终上下文
        optimized_context = self._optimize_context(assembled_context, query)

        return optimized_context

    def _analyze_information_needs(self, query: str, task_type: str) -> Dict:
        """分析此查询需要哪些类型的信息"""

        needs = {
            'definitions': 0.0,
            'facts': 0.0,
            'procedures': 0.0,
            'examples': 0.0,
            'background': 0.0
        }

        query_lower = query.lower()

        # 信息需求的启发式分析
        if any(word in query_lower for word in ['what is', 'define', 'meaning', 'definition']):
            needs['definitions'] = 1.0
            needs['examples'] = 0.7

        elif any(word in query_lower for word in ['how to', 'steps', 'procedure', 'process']):
            needs['procedures'] = 1.0
            needs['examples'] = 0.8

        elif any(word in query_lower for word in ['why', 'explain', 'reason', 'cause']):
            needs['facts'] = 1.0
            needs['background'] = 0.8

        elif 'example' in query_lower:
            needs['examples'] = 1.0
            needs['procedures'] = 0.5

        else:
            # 通用查询 - 平衡的信息需求
            for key in needs:
                needs[key] = 0.6

        # 根据任务类型调整
        if task_type == "analytical":
            needs['facts'] *= 1.3
            needs['background'] *= 1.2
        elif task_type == "practical":
            needs['procedures'] *= 1.3
            needs['examples'] *= 1.2
        elif task_type == "creative":
            needs['examples'] *= 1.2
            needs['background'] *= 1.1

        return needs

    def _select_optimal_candidates(self, candidates: List[RetrievalCandidate],
                                  info_needs: Dict, max_length: int) -> List[RetrievalCandidate]:
        """基于信息需求和长度约束选择候选的最优子集"""

        # 根据信息需求对齐对候选进行评分
        for candidate in candidates:
            content_type_score = info_needs.get(candidate.content_type, 0.5)
            candidate.need_alignment_score = (
                candidate.composite_score * 0.7 +
                content_type_score * 0.3
            )

        # 使用贪心背包式选择
        selected = []
        total_length = 0
        remaining_candidates = sorted(candidates, key=lambda c: c.need_alignment_score, reverse=True)

        for candidate in remaining_candidates:
            candidate_length = len(candidate.content)

            if total_length + candidate_length <= max_length * 0.8:  # 保留20%用于格式化
                selected.append(candidate)
                total_length += candidate_length
            elif len(selected) < 2:  # 确保我们至少有2个候选
                # 截断内容以适应
                available_space = max_length * 0.8 - total_length
                if available_space > 100:  # 仅当我们可以适应有意义的内容时
                    truncated_candidate = RetrievalCandidate(
                        content=candidate.content[:int(available_space)],
                        source=candidate.source,
                        relevance_score=candidate.relevance_score,
                        credibility_score=candidate.credibility_score,
                        recency_score=candidate.recency_score,
                        content_type=candidate.content_type,
                        metadata=candidate.metadata
                    )
                    selected.append(truncated_candidate)
                    break

        return selected

    def _structure_context(self, instructions: str, candidates: List[RetrievalCandidate],
                          user_context: str, query: str, info_needs: Dict) -> str:
        """为最优理解和实用性构建上下文结构"""

        context_parts = []

        # 如果提供了指令则添加
        if instructions.strip():
            context_parts.append(f"## 指令\n{instructions}\n")

        # 如果提供了用户上下文则添加
        if user_context.strip():
            context_parts.append(f"## 上下文\n{user_context}\n")

        # 按类型分组候选以获得更好的组织
        candidates_by_type = {}
        for candidate in candidates:
            if candidate.content_type not in candidates_by_type:
                candidates_by_type[candidate.content_type] = []
            candidates_by_type[candidate.content_type].append(candidate)

        # 按逻辑顺序添加检索到的信息
        type_order = ['definitions', 'facts', 'procedures', 'examples']
        type_labels = {
            'definition': '关键定义',
            'fact': '相关信息',
            'procedure': '程序和方法',
            'example': '示例和案例研究'
        }

        context_parts.append("## 检索到的知识\n")

        for content_type in type_order:
            if content_type in candidates_by_type:
                candidates_of_type = candidates_by_type[content_type]
                section_label = type_labels.get(content_type, content_type.title())

                context_parts.append(f"### {section_label}\n")

                for i, candidate in enumerate(candidates_of_type, 1):
                    source_note = f" (来源: {candidate.source})" if candidate.source else ""
                    context_parts.append(f"{i}. {candidate.content.strip()}{source_note}\n")

                context_parts.append("")  # 添加间距

        # 添加用户的特定查询
        context_parts.append(f"## 当前查询\n{query}\n")

        return "\n".join(context_parts)

    def _optimize_context(self, context: str, query: str) -> str:
        """组装上下文的最终优化"""

        # 删除过多的空白
        optimized = "\n".join(line.strip() for line in context.split("\n"))

        # 删除重复信息（简单方法）
        lines = optimized.split("\n")
        unique_lines = []
        seen_content = set()

        for line in lines:
            if line.strip():
                # 检查实质性重复（不仅仅是标题）
                line_content = line.lower().strip()
                if len(line_content) > 20:  # 仅检查实质性行
                    if line_content not in seen_content:
                        seen_content.add(line_content)
                        unique_lines.append(line)
                else:
                    unique_lines.append(line)
            else:
                unique_lines.append(line)

        return "\n".join(unique_lines)

# 演示完整检索和组装管道的示例用法
class ContextGenerationDemo:
    """完整上下文生成管道的演示"""

    def __init__(self):
        # 初始化检索器（用于演示的模拟实现）
        self.semantic_retriever = SemanticVectorRetriever(
            embedding_model=MockEmbeddingModel(),
            vector_database=MockVectorDatabase()
        )

        self.hybrid_retriever = HybridKnowledgeRetriever([
            self.semantic_retriever,
            # 根据需要添加其他检索器
        ])

        self.context_assembler = DynamicContextAssembler(max_context_length=4000)

    def generate_context(self, query: str, instructions: str = "",
                        user_context: str = "", task_type: str = "general") -> str:
        """完整的上下文生成管道"""

        print(f"为查询生成上下文: '{query}'")

        # 步骤 1：检索相关知识
        print("步骤 1: 检索知识...")
        candidates = self.hybrid_retriever.retrieve(query, max_results=10)
        print(f"检索到 {len(candidates)} 个候选")

        # 步骤 2：组装最优上下文
        print("步骤 2: 组装上下文...")
        context = self.context_assembler.assemble_context(
            query, candidates, instructions, user_context, task_type
        )

        print(f"步骤 3: 生成的上下文 ({len(context)} 个字符)")

        return context

# 用于演示的模拟类
class MockEmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        # 简化的模拟嵌入
        return np.random.rand(384)

class MockVectorDatabase:
    def __init__(self):
        self.mock_results = [
            MockResult("机器学习是人工智能的一个子集...", "wikipedia.org", 0.85),
            MockResult("要实现一个神经网络: 1. 定义架构...", "tutorial.com", 0.78),
            MockResult("例如，一个简单的分类模型...", "examples.org", 0.72)
        ]

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 10):
        return self.mock_results[:top_k]

@dataclass
class MockResult:
    content: str
    source: str
    similarity_score: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"date": "2024-01-01"}
```

**自底向上解释**：这个检索系统就像拥有多个具有不同专业的研究助手，加上一位知道如何将他们的发现结合成完美简报文档的总编辑。`HybridKnowledgeRetriever` 从多个来源获取输入，`DynamicContextAssembler` 最优地组织一切，系统从反馈中学习以随时间改进。

---

## 软件3.0范式3：协议（自适应组装外壳）

协议提供基于有效性演变的自修改上下文生成模式。

### 自适应上下文生成协议

```
/context.generate.adaptive{
    intent="通过从使用模式中学习并调整组装策略来动态生成最优上下文",

    input={
        user_query=<即时用户请求>,
        task_context={
            domain=<主题领域或字段>,
            complexity_level=<简单|中等|复杂|专家>,
            user_expertise=<新手|中级|高级|专家>,
            time_constraints=<可用处理时间>,
            quality_requirements=<准确性_完整性_特异性需求>
        },
        available_sources={
            knowledge_bases=<可访问的信息库>,
            real_time_data=<当前信息流>,
            user_history=<相关的过去交互>,
            domain_expertise=<专业知识来源>
        }
    },

    process=[
        /analyze.information_needs{
            action="深入分析获得最优响应所需的信息",
            method="多维需求评估与学习集成",
            analysis_dimensions=[
                {factual_requirements="需要哪些事实、数据或证据？"},
                {conceptual_requirements="需要哪些概念、定义或框架？"},
                {procedural_requirements="需要哪些过程、方法或步骤？"},
                {contextual_requirements="需要哪些背景或情境信息？"},
                {example_requirements="需要哪些插图、案例或演示？"}
            ],
            learning_integration="应用从类似成功查询上下文中学到的模式",
            output="具有优先级加权的综合信息需求规范"
        },

        /orchestrate.multi_source_retrieval{
            action="智能协调从多个信息源的检索",
            method="带有战略源选择和结果融合的并行检索",
            retrieval_strategies=[
                {semantic_search="针对知识嵌入的向量相似度匹配"},
                {keyword_expansion="使用领域特定术语的查询扩展"},
                {contextual_filtering="按与用户上下文和专业水平的相关性过滤"},
                {temporal_prioritization="适当权衡最新与权威信息"},
                {cross_reference_validation="验证跨多个来源的一致性"}
            ],
            fusion_algorithm="使用去重和相关性排名智能组合结果",
            output="高质量信息候选的排名集合"
        },

        /optimize.context_assembly{
            action="将检索到的信息组装成最优上下文结构",
            method="带有认知负荷管理的动态组装优化",
            assembly_strategies=[
                {information_hierarchy="将信息从最关键到最不关键进行结构化"},
                {cognitive_chunking="分组相关信息以减少认知负荷"},
                {logical_flow="在自然推理进程中组织信息"},
                {length_optimization="在上下文窗口约束内最大化信息价值"},
                {user_customization="使呈现风格适应用户专业知识和偏好"}
            ],
            optimization_criteria=[
                {relevance_maximization="确保每条信息都服务于用户的目标"},
                {coherence_enhancement="在信息片段之间创建逻辑连接"},
                {clarity_optimization="以适当的复杂度级别呈现信息"},
                {actionability_focus="强调能使用户采取行动的信息"}
            ],
            output="为模型消费做好准备的最优结构化上下文"
        },

        /monitor.effectiveness{
            action="跟踪上下文生成有效性并识别改进机会",
            method="带有学习集成的多指标有效性评估",
            effectiveness_metrics=[
                {response_quality="生成的上下文在多大程度上能够产生高质量响应？"},
                {user_satisfaction="用户对从此上下文生成的响应的满意度如何？"},
                {task_completion="上下文在多大程度上有效地促进任务完成？"},
                {efficiency_measures="上下文生成速度和资源利用"},
                {learning_indicators="随时间改进性能的证据"}
            ],
            feedback_integration=[
                {explicit_feedback="对响应质量的直接用户评分和评论"},
                {implicit_feedback="表明满意/不满意的用户行为模式"},
                {outcome_tracking="涉及生成上下文的任务的长期成功指标"},
                {comparative_analysis="与替代上下文生成方法的性能比较"}
            ],
            output="具有具体改进建议的综合有效性评估"
        }
    ],

    output={
        generated_context={
            assembled_information=<为模型做好准备的最优结构化上下文>,
            information_sources=<归属和可信度信息>,
            assembly_rationale=<上下文构建决策的解释>,
            quality_indicators=<置信度分数和完整性度量>
        },

        optimization_metadata={
            retrieval_performance=<信息收集有效性的指标>,
            assembly_efficiency=<上下文构建性能的指标>,
            predicted_effectiveness=<生成上下文的估计质量>,
            alternative_approaches=<考虑的其他上下文生成策略>
        },

        learning_updates={
            pattern_discoveries=<识别的新有效模式>,
            strategy_refinements=<对现有方法的改进>,
            feedback_integration=<用户反馈如何影响上下文生成>,
            knowledge_base_updates=<对底层信息源的改进>
        }
    },

    // 自我改进机制
    adaptation_triggers=[
        {condition="user_satisfaction < 0.7", action="analyze_context_assembly_weaknesses"},
        {condition="response_quality_decline_detected", action="audit_information_source_quality"},
        {condition="new_domain_patterns_identified", action="integrate_domain_specific_optimizations"},
        {condition="efficiency_below_threshold", action="optimize_retrieval_and_assembly_performance"}
    ],

    meta={
        context_generation_version="adaptive_v2.1",
        learning_integration_level="advanced",
        adaptation_frequency="continuous_with_batch_updates",
        quality_assurance="multi_dimensional_effectiveness_monitoring"
    }
}
```

**自底向上解释**：这个协议创建了一个自我改进的上下文生成系统。就像拥有一个研究团队，每次进行项目工作时都会变得更擅长查找和组织信息，学习哪些类型的信息对不同类型的问题和用户最有价值。

---

## 集成和实际应用

### 案例研究：医疗诊断支持上下文生成

```python
def medical_diagnosis_context_example():
    """演示用于医疗诊断支持的上下文生成"""

    # 模拟医疗查询
    query = "患者出现胸痛、呼吸短促和肌钙蛋白水平升高。鉴别诊断和推荐的诊断工作是什么？"

    # 医疗特定的上下文生成
    medical_context_generator = ContextGenerationDemo()

    # 生成专业医疗上下文
    context = medical_context_generator.generate_context(
        query=query,
        instructions="""
        您正在提供医疗决策支持。重点关注：
        1. 基于证据的鉴别诊断
        2. 适当的诊断工作建议
        3. 风险分层考虑
        4. 最新的临床指南和协议

        始终强调需要临床判断和直接患者评估。
        """,
        user_context="急诊科环境，成年患者，无已知过敏",
        task_type="analytical"
    )

    print("医疗诊断支持上下文:")
    print("=" * 50)
    print(context)

    return context
```

### 性能评估框架

```python
class ContextGenerationEvaluator:
    """上下文生成有效性的综合评估"""

    def __init__(self):
        self.evaluation_metrics = {
            'relevance': self._evaluate_relevance,
            'completeness': self._evaluate_completeness,
            'clarity': self._evaluate_clarity,
            'efficiency': self._evaluate_efficiency,
            'adaptability': self._evaluate_adaptability
        }

    def evaluate_context_generation(self, query: str, generated_context: str,
                                   response_quality: float, user_feedback: Dict) -> Dict:
        """上下文生成性能的综合评估"""

        results = {}
        for metric_name, metric_function in self.evaluation_metrics.items():
            score = metric_function(query, generated_context, response_quality, user_feedback)
            results[metric_name] = score

        # 计算整体有效性
        results['overall_effectiveness'] = self._calculate_overall_effectiveness(results)

        # 生成改进建议
        results['improvement_recommendations'] = self._generate_improvement_recommendations(results)

        return results

    def _evaluate_relevance(self, query: str, context: str, response_quality: float, feedback: Dict) -> float:
        """评估生成的上下文与查询的相关性"""

        # 分析查询和上下文之间的语义对齐
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())

        term_overlap = len(query_terms.intersection(context_terms)) / len(query_terms.union(context_terms))

        # 将响应质量作为上下文相关性的指标
        relevance_score = (term_overlap * 0.3 + response_quality * 0.7)

        return min(1.0, max(0.0, relevance_score))

    def _evaluate_completeness(self, query: str, context: str, response_quality: float, feedback: Dict) -> float:
        """评估上下文是否包含所有必要信息"""

        # 简单的启发式：较长的上下文通常更完整
        # 但也考虑用户关于缺失信息的反馈

        context_length_score = min(1.0, len(context) / 2000)  # 标准化到合理长度

        # 检查反馈中的缺失信息指标
        missing_info_penalty = 0.0
        if feedback.get('missing_information', False):
            missing_info_penalty = 0.3

        completeness_score = max(0.0, context_length_score - missing_info_penalty)

        return completeness_score

    def _calculate_overall_effectiveness(self, metric_scores: Dict) -> float:
        """计算加权整体有效性分数"""

        weights = {
            'relevance': 0.30,
            'completeness': 0.25,
            'clarity': 0.20,
            'efficiency': 0.15,
            'adaptability': 0.10
        }

        overall = sum(metric_scores[metric] * weight
                     for metric, weight in weights.items()
                     if metric in metric_scores)

        return overall
```

**自底向上解释**：这个评估框架就像拥有一个综合的质量控制系统，从多个角度审视上下文生成——不仅仅是它是否有效，还包括它有多有效以及如何改进。

---

## 实践练习和下一步

### 练习 1：构建您自己的检索系统
**目标**：实现一个基本的语义检索系统

```python
# 您的实现模板
class BasicRetriever:
    def __init__(self):
        # TODO: 初始化您的检索系统
        self.knowledge_base = {}
        self.embedding_cache = {}

    def add_document(self, doc_id: str, content: str):
        # TODO: 将文档添加到知识库
        pass

    def retrieve(self, query: str, max_results: int = 5) -> List[str]:
        # TODO: 实现检索逻辑
        pass

# 测试您的检索器
retriever = BasicRetriever()
# 添加一些测试文档
# 用不同的查询测试检索
```

### 练习 2：上下文组装优化
**目标**：创建一个优化信息组织的上下文组装器

```python
class ContextOptimizer:
    def __init__(self, max_length: int = 2000):
        # TODO: 初始化上下文优化器
        self.max_length = max_length

    def optimize_context(self, information_pieces: List[str], query: str) -> str:
        # TODO: 实现最优上下文组装
        pass
```

---

## 总结和下一步

**掌握的核心概念**：
- 从静态提示词到动态上下文编排的演变
- 知识检索的信息论优化
- 多源检索策略和结果融合
- 带有学习集成的自适应上下文组装
- 上下文生成有效性的综合评估

**软件3.0集成**：
- **提示词**：用于推理和知识集成的策略性模板
- **编程**：复杂的检索和组装算法
- **协议**：自我改进的上下文生成系统

**实现技能**：
- 使用嵌入和向量数据库的语义检索
- 带有认知负荷优化的动态上下文组装
- 多源信息融合和去重
- 有效性评估和持续改进系统

**研究基础**：直接实现上下文生成研究（§4.1），并在自适应组装、多源融合和自我改进的上下文编排方面进行新的扩展。

**下一个模块**：[01_prompt_engineering.md](01_prompt_engineering.md) - 深入探讨高级提示词技术，在上下文生成基础上构建，掌握LLM通信的艺术和科学。

---

*本模块为智能上下文工程奠定了基础，将"提示词"的简单概念转变为动态知识编排和最优信息组装的复杂系统。*
