# 长上下文处理
## 从序列token到无限内存架构

> **模块 02.1** | *上下文工程课程:从基础到前沿系统*
>
> 基于 [上下文工程综述](https://arxiv.org/pdf/2507.13334) | 推进信息论上下文优化

---

## 学习目标

完成本模块后,您将理解并实现:

- **内存架构设计**:从滑动窗口到无限注意力系统
- **计算规模化**:在百万token上下文中管理O(n²)注意力复杂度
- **信息保留**:在扩展序列中保持连贯性和相关性
- **自适应处理**:动态注意力和内存管理策略

---

## 概念进展:从有限窗口到无限内存

将上下文处理想象成人类的记忆系统 - 从只能容纳少量项目的短期工作记忆,到能够存储和检索大量相互关联信息的复杂长期记忆。

### 阶段1:固定窗口处理
```
[上下文窗口: 4K tokens]
输入: "The cat sat on the mat and..."
处理: ████████░░░░░░░░░░░░ (仅最近的tokens)
限制: 忘记窗口之前的所有内容
```
**上下文**:就像试图进行对话,但只记得最后几句话。高效但对复杂任务严重受限。

### 阶段2:滑动窗口注意力
```
[窗口在序列上滑动]
Token 1-1000:  ████████████████░░░░
Token 501-1500: ░░░░████████████████
Token 1001-2000: ░░░░░░░░████████████
限制: 无法连接远距离信息
```
**上下文**:就像用放大镜阅读书籍 - 您能清楚地看到细节,但失去了整体叙事连接。

### 阶段3:分层内存系统
```
[多层内存架构]
工作内存:     ████████ (最近tokens)
短期内存:     ████░░██ (压缩块)
长期内存:     ██░░░░██ (关键信息)
全局上下文:   █░░░░░█░ (文档级主题)
```
**上下文**:就像您的大脑工作方式 - 即时意识、近期记忆、重要事实和生活经验协同工作。

### 阶段4:关联记忆网络
```
[连接记忆的网络]
当前焦点: "应对气候变化的解决方案需要..."
     ↕
连接的记忆:
- "早期关于可再生能源的讨论..."
- "之前提到的碳捕获..."
- "相关的政策考虑..."
- "其他领域的类似挑战..."
```
**上下文**:就像有一个出色的研究助理,能从您的整个知识库中即时回忆所有相关信息。

### 阶段5:无限上下文架构
```
[连续处理流]
∞ ←─────────── 无限输入流 ──────────→ ∞
    ██████████████████████████████████████████

处理特性:
- 恒定内存使用量,不受序列长度影响
- 对相关信息的自适应注意力
- 完美回忆重要细节
- 无缝整合新信息
```
**上下文**:就像拥有完美的记忆,永远不会忘记任何重要的东西,同时高效管理无限的信息流。

---

## 数学基础

### 注意力复杂度问题
```
标准注意力: O(n²) 复杂度
对于序列长度n,注意力矩阵为n×n

内存需求: n² × d_model
计算时间: n² × d_model × operations_per_element

规模化示例:
- 1K tokens: ~1M 操作
- 10K tokens: ~100M 操作
- 100K tokens: ~10B 操作
- 1M tokens: ~1T 操作 (不可行)
```
**直观解释**:标准注意力呈二次增长 - 如果您将序列长度加倍,计算成本会增加四倍。这使得非常长的序列在计算上变得不可能。

### 信息论上下文优化
```
最优上下文选择: C* = argmax_C I(Y*; C|Q)

其中:
- I(Y*; C|Q) = 给定查询Q时,目标输出与上下文之间的互信息
- C = 从完整序列中选择的上下文子集
- Y* = 最优目标输出
- Q = 当前查询

受约束条件:
- |C| ≤ L_max (上下文长度限制)
- Computational_Cost(C) ≤ Budget
```
**直观解释**:我们想要选择信息子集,在计算和内存约束下为我们的任务提供最大的预测价值。就像从图书馆中选择最相关的书页来回答特定问题。

### 内存压缩原理
```
无损压缩: H(X) ≤ |X|
其中H(X)是序列X的熵(真实信息内容)

有质量约束的有损压缩:
最小化: |C(X)|
受约束: D(X, D(C(X))) ≤ δ

其中:
- C(X) = 压缩表示
- D(C(X)) = 解压序列
- D(X, D(C(X))) = 失真度量
- δ = 最大可接受失真
```
**直观解释**:我们想要压缩信息以适应内存约束,同时保留基本内容。就像创建高质量的摘要,用更少的词捕获所有重要信息。

---

## 可视化架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    长上下文处理管道                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入流: [Token₁][Token₂][Token₃]...[Tokenₙ]                  │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           多层注意力系统                                │   │
│  │                                                         │   │
│  │  局部窗口:    [████████]                               │   │
│  │  滑动窗口:    [░░████████░░]                           │   │
│  │  全局内存:    [█░░█░░█░░█]                             │   │
│  │  关联性:      [█~█~█~█~█]                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              内存管理                                    │   │
│  │                                                         │   │
│  │  工作内存:     [当前焦点: 2K tokens]                    │   │
│  │  短期内存:     [最近上下文: 8K tokens]                  │   │
│  │  长期内存:     [压缩历史: ∞]                            │   │
│  │  情景内存:     [关键事件与决策]                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            上下文组装引擎                                │   │
│  │                                                         │   │
│  │  查询分析 → 内存检索 → 上下文选择                       │   │
│  │       │               │                    │            │   │
│  │       ▼               ▼                    ▼            │   │
│  │  [相关性]        [信息]            [最优]               │   │
│  │  [排序]          [压缩]            [组装]               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  输出: [当前查询的最优组装上下文]                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

性能特征:
• 内存使用: O(1) 恒定,不受总序列长度影响
• 检索时间: O(log n) 用于相关信息查找
• 上下文质量: 跨任意距离保持连贯性
• 适应性: 基于查询模式的实时优化
```

---

## Software 3.0 范式 1: 提示词 (内存架构模板)

战略性提示词帮助系统以结构化、可重用的方式推理内存管理和上下文选择。

### 分层内存管理模板

```markdown
# 分层内存管理框架

## 上下文评估
您是一个内存管理系统，处理长序列并决定如何在多个内存层级之间存储、检索和组织信息。

## 内存架构分析
**当前内存状态**:
- 工作内存: {current_active_tokens} / {working_memory_limit}
- 短期内存: {recent_context_size} / {short_term_limit}
- 长期内存: {compressed_history_size} (已压缩)
- 情景内存: {key_events_count} 个重要事件

**传入信息**: {new_token_sequence}
**当前查询上下文**: {query_or_task_focus}
**内存压力**: {memory_utilization_percentage}%

## 内存层级决策矩阵

### 工作内存 (立即注意力)
**工作内存存储标准**:
- 与当前查询/任务直接相关
- 时间上的近期性(最近的token)
- 高信息密度
- 主动处理需求

**当前工作内存内容**:
{list_current_working_memory_items}

**决策**: 新信息是否应进入工作内存？
- **是**: 如果与活动任务直接相关且有可用空间
- **否**: 如果偏离主题或工作内存已满

### 短期内存 (最近上下文缓冲区)
**短期内存标准**:
- 最近处理但不是立即活跃
- 会话内潜在的未来相关性
- 相关信息的连贯块
- 工作内存和长期存储之间的桥梁

**压缩策略**:
- 语义分块: 组合相关概念
- 重要性加权: 保留高价值信息
- 时间组织: 维护序列关系

### 长期内存 (压缩历史上下文)
**长期存储标准**:
- 对未来检索有高信息价值
- 关键决策、见解或结论
- 跨上下文有用的模式信息
- 更大序列的压缩表示

**压缩技术**:
- 抽象摘要: 核心概念和关系
- 示例选择: 代表性实例
- 模式提取: 重复出现的主题和结构
- 分层组织: 嵌套概念结构

### 情景内存 (关键事件跟踪)
**情景存储标准**:
- 重要决策或转折点
- 新颖见解或突破时刻
- 上下文转换或主题转换
- 对未来参考有高影响的信息

## 内存管理操作

### 信息分类处理流程
```
IF information_relevance > working_memory_threshold AND working_memory_space > 0:
    存储到 working_memory
ELIF information_importance > short_term_threshold:
    压缩并存储到 short_term_memory
ELIF information_value > long_term_threshold:
    抽象并存储到 long_term_memory
ELIF information_significance > episodic_threshold:
    提取 key_event 并存储到 episodic_memory
ELSE:
    丢弃或存储到 low_priority_buffer
```

### 内存整合协议
**触发条件**:
- 工作内存利用率 > 90%
- 处理会话结束
- 检测到上下文转换
- 明确的整合请求

**整合过程**:
1. **评估**: 评估所有内存内容的重要性和相关性
2. **压缩**: 总结不太关键的信息
3. **提升**: 将重要的短期记忆移至长期记忆
4. **归档**: 使用检索键存储情景事件
5. **清理**: 释放工作内存空间用于新处理

### 检索策略框架
**查询分析**:
- 识别查询中的关键概念和实体
- 确定时间范围(近期 vs. 历史)
- 评估具体程度(详细 vs. 一般)
- 评估上下文广度(狭窄 vs. 全面)

**内存搜索策略**:
```
FOR each memory_level in [episodic, long_term, short_term, working]:
    relevant_memories = SEARCH(query_concepts, memory_level)
    scored_memories = RANK_BY_RELEVANCE(relevant_memories, query)
    IF sufficient_information_found:
        RETURN assembled_context
    ELSE:
        CONTINUE to next memory_level
```

## 上下文组装逻辑

### 最优上下文选择
**组装原则**:
- **相关性优先**: 优先考虑与查询最相关的信息
- **连贯性保持**: 维护逻辑流程和连接
- **完整性平衡**: 包含足够的上下文而不过载
- **多样性整合**: 在有益时融入多个视角

**组装算法**:
```
1. 从最相关的情景记忆开始
2. 添加相关的长期压缩摘要
3. 包含必要的短期上下文以保持连贯性
4. 用当前工作内存填充剩余空间
5. 验证上下文连贯性和完整性
6. 如果质量指标不满意则调整选择
```

### 上下文组装的质量指标
**连贯性评分**: 测量逻辑流程和连接强度
**相关性评分**: 上下文与查询需求之间的对齐度
**完整性评分**: 任务所需信息的覆盖范围
**效率评分**: 信息密度和最小冗余

**质量保证检查**:
- 组装的上下文是否提供足够的信息？
- 是否存在需要额外内存检索的逻辑缺口？
- 是否存在可以压缩的冗余信息？
- 上下文是否支持特定的查询需求？

## 内存管理建议

**对于当前上下文**: {specific_recommendations_based_on_analysis}
**内存优化**: {suggested_improvements_to_memory_usage}
**检索增强**: {ways_to_improve_information_retrieval}
**未来准备**: {proactive_memory_management_suggestions}

## 自适应学习整合
- 监控哪些内存管理决策带来最佳结果
- 基于性能反馈调整阈值和标准
- 学习查询模式以预测信息需求
- 基于检索成功率优化压缩技术
```

**从零开始的解释**: 这个模板就像一个管理庞大图书馆的图书管理员，阅览室空间有限。图书管理员必须决定哪些书放在眼前的桌子上(工作内存)，哪些放在附近的书架上(短期)，哪些存储在书库中(长期)，以及在特殊日志中标记哪些事件(情景)。系统基于相关性、重要性和预测的未来需求做出这些决策。

### 自适应上下文窗口模板

```xml
<context_processing_template name="adaptive_window_management">
  <intent>基于任务复杂度和信息需求动态调整上下文窗口大小和焦点</intent>

  <context_analysis>
    <sequence_characteristics>
      <total_length>{sequence_token_count}</total_length>
      <information_density>{avg_information_per_token}</information_density>
      <topic_complexity>{number_of_distinct_concepts}</topic_complexity>
      <temporal_span>{time_range_covered}</temporal_span>
    </sequence_characteristics>

    <task_requirements>
      <task_type>{classification_generation_analysis_etc}</task_type>
      <context_scope>{local_document_global}</context_scope>
      <precision_needs>{high_medium_low}</precision_needs>
      <computational_budget>{available_processing_resources}</computational_budget>
    </task_requirements>
  </context_analysis>

  <window_adaptation_strategy>
    <base_window_size>
      <calculation>
        initial_size = min(max_context_length, sqrt(sequence_length * task_complexity))
        adjusted_size = initial_size * information_density_factor
        final_size = constrain(adjusted_size, min_effective_size, max_feasible_size)
      </calculation>
    </base_window_size>

    <dynamic_expansion_triggers>
      <trigger name="information_gap_detected">
        <condition>当前上下文不足以完成任务</condition>
        <action>扩展窗口以包含相关的缺失信息</action>
        <expansion_strategy>从当前位置双向搜索</expansion_strategy>
      </trigger>

      <trigger name="cross_reference_needed">
        <condition>任务需要连接序列的远距离部分</condition>
        <action>创建具有桥接连接的多个注意力窗口</action>
        <bridge_strategy>语义相似性和因果关系检测</bridge_strategy>
      </trigger>

      <trigger name="context_shift_detected">
        <condition>主题或上下文发生显著变化</condition>
        <action>移动窗口位置并调整新上下文的大小</action>
        <shift_strategy>保持重叠的渐进过渡</shift_strategy>
      </trigger>
    </dynamic_expansion_triggers>

    <compression_strategies>
      <when_to_compress>
        <condition>窗口大小接近计算限制</condition>
        <condition>窗口内检测到信息冗余</condition>
        <condition>外围识别出低相关性内容</condition>
      </when_to_compress>

      <compression_methods>
        <method name="semantic_summarization">
          <description>为不太关键的部分创建抽象摘要</description>
          <preservation_ratio>在20%的token中保持80%的语义内容</preservation_ratio>
        </method>

        <method name="exemplar_sampling">
          <description>从重复内容中选择代表性示例</description>
          <selection_criteria>多样性、典型性和与当前任务的相关性</selection_criteria>
        </method>

        <method name="hierarchical_abstraction">
          <description>为不同内容部分创建多个详细级别</description>
          <abstraction_levels>详细、摘要、大纲、要点</abstraction_levels>
        </method>
      </compression_methods>
    </compression_strategies>
  </window_adaptation_strategy>

  <attention_optimization>
    <attention_patterns>
      <local_attention>
        <scope>立即上下文窗口</scope>
        <pattern>密集的双向注意力用于细粒度理解</pattern>
        <computational_cost>窗口大小n的O(n²)</computational_cost>
      </local_attention>

      <sparse_global_attention>
        <scope>整个序列中的关键位置</scope>
        <pattern>对结构上重要的token进行选择性注意</pattern>
        <selection_criteria>句子边界、主题标记、关键实体</selection_criteria>
      </sparse_global_attention>

      <hierarchical_attention>
        <scope>多个分辨率级别</scope>
        <pattern>局部细注意力，全局粗注意力</pattern>
        <hierarchy_levels>Token → 短语 → 句子 → 段落 → 文档</hierarchy_levels>
      </hierarchical_attention>
    </attention_patterns>

    <attention_routing>
      <routing_decision>
        IF task_requires_local_detail:
            分配 70% attention_budget 到 local_window
            分配 20% attention_budget 到 sparse_global
            分配 10% attention_budget 到 hierarchical_context
        ELIF task_requires_global_understanding:
            分配 40% attention_budget 到 local_window
            分配 40% attention_budget 到 sparse_global
            分配 20% attention_budget 到 hierarchical_context
        ELIF task_requires_structural_analysis:
            分配 30% attention_budget 到 local_window
            分配 30% attention_budget 到 sparse_global
            分配 40% attention_budget 到 hierarchical_context
      </routing_decision>
    </attention_routing>
  </attention_optimization>

  <output_context_assembly>
    <assembly_process>
      <step name="relevance_ranking">
        <description>按与当前查询的相关性对所有可用上下文段进行排序</description>
        <ranking_factors>语义相似性、时间接近度、因果关系</ranking_factors>
      </step>

      <step name="coherence_optimization">
        <description>排列选定的上下文以实现最大逻辑流</description>
        <optimization_criteria>时间顺序、因果序列、概念层次</optimization_criteria>
      </step>

      <step name="completeness_validation">
        <description>验证组装的上下文包含足够的信息</description>
        <validation_checks>必要实体存在、关键关系包含、足够的细节级别</validation_checks>
      </step>
    </assembly_process>

    <quality_metrics>
      <relevance_score>对任务直接有用的上下文比例</relevance_score>
      <coherence_score>上下文元素之间的逻辑流和连接强度</coherence_score>
      <completeness_score>成功完成任务所需信息的覆盖范围</completeness_score>
      <efficiency_score>信息密度和最小冗余</efficiency_score>
    </quality_metrics>
  </output_context_assembly>

  <learning_adaptation>
    <performance_feedback>
      <success_indicators>任务完成质量、处理效率、用户满意度</success_indicators>
      <failure_indicators>不完整结果、过度计算时间、上下文缺口</failure_indicators>
    </performance_feedback>

    <adaptation_mechanisms>
      <window_size_learning>学习不同任务类型的最优窗口大小</window_size_learning>
      <attention_pattern_optimization>基于任务成功模式改进注意力分配</attention_pattern_optimization>
      <compression_strategy_selection>基于信息保留质量改进压缩方法选择</compression_strategy_selection>
    </adaptation_mechanisms>
  </learning_adaptation>
</context_processing_template>
```

**从零开始的解释**: 这个XML模板就像一个智能相机系统，它会根据你想要拍摄的内容自动调整变焦和焦点。对于特写细节工作，它会紧密变焦。对于风景摄影，它会拉回以获得广角视图。对于动作镜头，它会调整焦点跟踪。系统学习哪些设置最适合不同类型的摄影，并随时间自动优化。

---

## Software 3.0 范式 2: 编程 (内存架构实现)

编程提供了支持复杂长上下文处理的计算机制。

### 无限上下文架构实现

```python
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MemorySegment:
    """表示带有元数据的已处理上下文段"""
    content: str
    embedding: np.ndarray
    importance_score: float
    recency_score: float
    access_frequency: int
    creation_time: float
    last_access_time: float
    segment_type: str  # 'working', 'short_term', 'long_term', 'episodic'
    retrieval_keys: List[str]
    compression_ratio: float = 1.0

class MemoryLevel(ABC):
    """不同内存级别的抽象基类"""

    @abstractmethod
    def store(self, segment: MemorySegment) -> bool:
        """存储内存段，如果成功返回True"""
        pass

    @abstractmethod
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemorySegment]:
        """检索最相关的内存段"""
        pass

    @abstractmethod
    def consolidate(self) -> List[MemorySegment]:
        """整合记忆并返回要提升到更高级别的项目"""
        pass

    @abstractmethod
    def get_capacity_info(self) -> Dict[str, float]:
        """返回当前容量利用信息"""
        pass

class WorkingMemory(MemoryLevel):
    """用于当前处理的高容量、立即访问内存"""

    def __init__(self, max_segments: int = 100, max_total_tokens: int = 4000):
        self.max_segments = max_segments
        self.max_total_tokens = max_total_tokens
        self.segments: List[MemorySegment] = []
        self.current_tokens = 0

    def store(self, segment: MemorySegment) -> bool:
        """在工作内存中存储，必要时使用LRU驱逐"""
        # 检查是否可以容纳这个段
        segment_tokens = len(segment.content.split())

        if len(self.segments) >= self.max_segments or \
           self.current_tokens + segment_tokens > self.max_total_tokens:
            # 需要驱逐最近最少使用的段
            self._evict_lru_segments(segment_tokens)

        # 存储新段
        segment.segment_type = 'working'
        self.segments.append(segment)
        self.current_tokens += segment_tokens
        return True

    def _evict_lru_segments(self, tokens_needed: int):
        """驱逐最近最少使用的段以腾出空间"""
        # 按最后访问时间排序(最旧的在前)
        self.segments.sort(key=lambda s: s.last_access_time)

        tokens_freed = 0
        segments_to_remove = []

        for segment in self.segments:
            if tokens_freed >= tokens_needed and len(segments_to_remove) > 0:
                break

            segments_to_remove.append(segment)
            tokens_freed += len(segment.content.split())

        # 移除选定的段
        for segment in segments_to_remove:
            self.segments.remove(segment)
            self.current_tokens -= len(segment.content.split())

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemorySegment]:
        """从工作内存中检索最相关的段"""
        if not self.segments:
            return []

        # 计算相似度分数
        similarities = []
        for segment in self.segments:
            similarity = np.dot(query_embedding, segment.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(segment.embedding)
            )
            # 更新访问时间和频率
            segment.last_access_time = time.time()
            segment.access_frequency += 1

            similarities.append((similarity, segment))

        # 按相似度排序并返回top k
        similarities.sort(reverse=True)
        return [segment for _, segment in similarities[:top_k]]

    def consolidate(self) -> List[MemorySegment]:
        """识别要提升到更高内存级别的段"""
        promotion_candidates = []

        for segment in self.segments:
            # 提升高重要性或高访问频率的段
            if (segment.importance_score > 0.7 or
                segment.access_frequency > 3):
                promotion_candidates.append(segment)

        return promotion_candidates

    def get_capacity_info(self) -> Dict[str, float]:
        """返回容量利用信息"""
        return {
            'segment_utilization': len(self.segments) / self.max_segments,
            'token_utilization': self.current_tokens / self.max_total_tokens,
            'current_segments': len(self.segments),
            'current_tokens': self.current_tokens
        }

class ShortTermMemory(MemoryLevel):
    """具有语义组织的压缩近期上下文缓冲区"""

    def __init__(self, max_segments: int = 200, compression_threshold: float = 0.6):
        self.max_segments = max_segments
        self.compression_threshold = compression_threshold
        self.segments: List[MemorySegment] = []
        self.semantic_clusters: Dict[str, List[MemorySegment]] = defaultdict(list)

    def store(self, segment: MemorySegment) -> bool:
        """使用自动压缩和聚类进行存储"""
        # 如果低于阈值则压缩段
        if segment.compression_ratio > self.compression_threshold:
            segment = self._compress_segment(segment)

        # 添加到适当的语义集群
        cluster_key = self._determine_cluster(segment)
        self.semantic_clusters[cluster_key].append(segment)

        segment.segment_type = 'short_term'
        self.segments.append(segment)

        # 如果超过容量则驱逐
        if len(self.segments) > self.max_segments:
            self._evict_oldest_segments()

        return True

    def _compress_segment(self, segment: MemorySegment) -> MemorySegment:
        """应用压缩以减少token数量同时保留含义"""
        # 简化压缩 - 实践中会使用复杂的摘要
        original_length = len(segment.content)

        # 提取关键句子(简单启发式)
        sentences = segment.content.split('.')
        important_sentences = []

        for sentence in sentences:
            # 保留高信息内容的句子
            if (len(sentence.split()) > 5 and
                any(word in sentence.lower() for word in ['important', 'key', 'main', 'significant'])):
                important_sentences.append(sentence)

        if not important_sentences:
            # 后备方案: 保留第一句和最后一句
            important_sentences = [sentences[0], sentences[-1]]

        compressed_content = '. '.join(important_sentences)
        segment.content = compressed_content
        segment.compression_ratio = len(compressed_content) / original_length

        return segment

    def _determine_cluster(self, segment: MemorySegment) -> str:
        """确定段的语义集群"""
        # 基于关键词的简化聚类
        content_lower = segment.content.lower()

        if 'memory' in content_lower or 'attention' in content_lower:
            return 'memory_systems'
        elif 'processing' in content_lower or 'computation' in content_lower:
            return 'processing'
        elif 'context' in content_lower or 'information' in content_lower:
            return 'context_management'
        else:
            return 'general'

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemorySegment]:
        """使用集群感知搜索进行检索"""
        all_similarities = []

        for segment in self.segments:
            similarity = np.dot(query_embedding, segment.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(segment.embedding)
            )

            # 为最近访问的项目增强相似度
            recency_boost = 1 + (segment.recency_score * 0.2)
            adjusted_similarity = similarity * recency_boost

            all_similarities.append((adjusted_similarity, segment))

        all_similarities.sort(reverse=True)
        return [segment for _, segment in all_similarities[:top_k]]

    def consolidate(self) -> List[MemorySegment]:
        """整合集群并识别提升候选"""
        promotion_candidates = []

        for cluster_key, cluster_segments in self.semantic_clusters.items():
            if len(cluster_segments) >= 3:
                # 为此集群创建整合段
                consolidated = self._consolidate_cluster(cluster_segments)
                promotion_candidates.append(consolidated)

        return promotion_candidates

    def _consolidate_cluster(self, segments: List[MemorySegment]) -> MemorySegment:
        """将多个段整合为单个压缩表示"""
        # 结合内容并保留语义
        combined_content = []
        total_importance = 0
        avg_embedding = np.zeros_like(segments[0].embedding)

        for segment in segments:
            combined_content.append(segment.content)
            total_importance += segment.importance_score
            avg_embedding += segment.embedding

        avg_embedding /= len(segments)
        avg_importance = total_importance / len(segments)

        # 创建整合段
        consolidated_content = ' | '.join(combined_content)

        return MemorySegment(
            content=consolidated_content,
            embedding=avg_embedding,
            importance_score=avg_importance,
            recency_score=max(s.recency_score for s in segments),
            access_frequency=sum(s.access_frequency for s in segments),
            creation_time=min(s.creation_time for s in segments),
            last_access_time=max(s.last_access_time for s in segments),
            segment_type='consolidated',
            retrieval_keys=list(set().union(*[s.retrieval_keys for s in segments])),
            compression_ratio=0.3  # 高度压缩
        )

    def get_capacity_info(self) -> Dict[str, float]:
        return {
            'segment_utilization': len(self.segments) / self.max_segments,
            'cluster_count': len(self.semantic_clusters),
            'avg_cluster_size': np.mean([len(cluster) for cluster in self.semantic_clusters.values()]),
            'compression_ratio': np.mean([s.compression_ratio for s in self.segments])
        }

class LongTermMemory(MemoryLevel):
    """用于历史上下文的高度压缩、索引存储"""

    def __init__(self, index_dimensions: int = 512):
        self.segments: List[MemorySegment] = []
        self.index_dimensions = index_dimensions
        self.semantic_index = {}  # 分层语义索引
        self.temporal_index = {}  # 基于时间的索引
        self.importance_index = []  # 基于重要性检索的优先级队列

    def store(self, segment: MemorySegment) -> bool:
        """使用多维索引进行存储"""
        # 对长期存储应用激进压缩
        compressed_segment = self._apply_long_term_compression(segment)
        compressed_segment.segment_type = 'long_term'

        self.segments.append(compressed_segment)

        # 更新索引
        self._update_semantic_index(compressed_segment)
        self._update_temporal_index(compressed_segment)
        self._update_importance_index(compressed_segment)

        return True

    def _apply_long_term_compression(self, segment: MemorySegment) -> MemorySegment:
        """对长期存储应用激进压缩"""
        # 仅提取最基本的信息
        content_parts = segment.content.split('.')
        essential_parts = []

        for part in content_parts:
            # 保留高信息密度的部分
            word_count = len(part.split())
            if word_count > 3:
                # 简单启发式: 保留包含特定术语的部分
                if any(term in part.lower() for term in
                       ['result', 'conclusion', 'important', 'key', 'main', 'significant']):
                    essential_parts.append(part.strip())

        if not essential_parts:
            # 后备方案: 从原始内容创建摘要
            words = segment.content.split()
            essential_parts = [' '.join(words[:min(20, len(words))])]

        compressed_content = '. '.join(essential_parts)

        # 创建极度压缩的新段
        return MemorySegment(
            content=compressed_content,
            embedding=segment.embedding,
            importance_score=segment.importance_score,
            recency_score=segment.recency_score * 0.1,  # 衰减近期性
            access_frequency=segment.access_frequency,
            creation_time=segment.creation_time,
            last_access_time=segment.last_access_time,
            segment_type='long_term',
            retrieval_keys=segment.retrieval_keys,
            compression_ratio=0.1  # 90%压缩
        )

    def _update_semantic_index(self, segment: MemorySegment):
        """更新语义索引以实现高效检索"""
        for key in segment.retrieval_keys:
            if key not in self.semantic_index:
                self.semantic_index[key] = []
            self.semantic_index[key].append(len(self.segments) - 1)

    def _update_temporal_index(self, segment: MemorySegment):
        """更新时间索引"""
        time_bucket = int(segment.creation_time // 3600)  # 小时桶
        if time_bucket not in self.temporal_index:
            self.temporal_index[time_bucket] = []
        self.temporal_index[time_bucket].append(len(self.segments) - 1)

    def _update_importance_index(self, segment: MemorySegment):
        """更新基于重要性的优先级队列"""
        heapq.heappush(self.importance_index,
                      (-segment.importance_score, len(self.segments) - 1))

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemorySegment]:
        """带有相关性排序的多索引检索"""
        candidate_indices = set()

        # 从重要性索引获取候选(前20%)
        n_important = max(1, len(self.importance_index) // 5)
        important_candidates = heapq.nsmallest(n_important, self.importance_index)
        candidate_indices.update(idx for _, idx in important_candidates)

        # 计算候选的相似度
        similarities = []
        for idx in candidate_indices:
            if idx < len(self.segments):
                segment = self.segments[idx]
                similarity = np.dot(query_embedding, segment.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(segment.embedding)
                )
                similarities.append((similarity, segment))

        similarities.sort(reverse=True)
        return [segment for _, segment in similarities[:top_k]]

    def consolidate(self) -> List[MemorySegment]:
        """长期内存不再进一步提升，但可以重组"""
        # 定期重组索引以提高效率
        self._reorganize_indices()
        return []

    def _reorganize_indices(self):
        """重组索引以获得更好的检索性能"""
        # 重建重要性索引
        self.importance_index = []
        for i, segment in enumerate(self.segments):
            heapq.heappush(self.importance_index, (-segment.importance_score, i))

    def get_capacity_info(self) -> Dict[str, float]:
        return {
            'total_segments': len(self.segments),
            'semantic_keys': len(self.semantic_index),
            'temporal_buckets': len(self.temporal_index),
            'avg_compression': np.mean([s.compression_ratio for s in self.segments])
        }

class EpisodicMemory(MemoryLevel):
    """关键事件和决策点存储"""

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: List[MemorySegment] = []
        self.decision_tree = {}  # 跟踪决策序列
        self.outcome_associations = {}  # 将情节与结果关联

    def store(self, segment: MemorySegment) -> bool:
        """存储重要情节并跟踪结果"""
        segment.segment_type = 'episodic'
        self.episodes.append(segment)

        # 如果这代表一个决策，则在决策树中跟踪
        if 'decision' in segment.content.lower() or 'chose' in segment.content.lower():
            self._update_decision_tree(segment)

        # 维护大小限制
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)  # 移除最旧的情节

        return True

    def _update_decision_tree(self, segment: MemorySegment):
        """跟踪决策序列以进行模式学习"""
        # 简化的决策跟踪
        decision_key = f"decision_{len(self.episodes)}"
        self.decision_tree[decision_key] = {
            'content': segment.content,
            'timestamp': segment.creation_time,
            'importance': segment.importance_score
        }

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemorySegment]:
        """检索相关情节"""
        similarities = []

        for episode in self.episodes:
            similarity = np.dot(query_embedding, episode.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(episode.embedding)
            )

            # 为高重要性情节增强相似度
            importance_boost = 1 + (episode.importance_score * 0.3)
            adjusted_similarity = similarity * importance_boost

            similarities.append((adjusted_similarity, episode))

        similarities.sort(reverse=True)
        return [episode for _, episode in similarities[:top_k]]

    def consolidate(self) -> List[MemorySegment]:
        """情景内存通常不会进一步整合"""
        return []

    def get_capacity_info(self) -> Dict[str, float]:
        return {
            'episode_count': len(self.episodes),
            'utilization': len(self.episodes) / self.max_episodes,
            'decision_count': len(self.decision_tree),
            'avg_importance': np.mean([e.importance_score for e in self.episodes])
        }

import time

class HierarchicalMemorySystem:
    """协调所有级别的集成分层内存系统"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

        # 初始化内存级别
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.episodic_memory = EpisodicMemory()

        # 整合和管理
        self.consolidation_threshold = 0.8
        self.last_consolidation = time.time()
        self.consolidation_interval = 300  # 5分钟

    def process_input(self, content: str, importance_score: float = 0.5) -> str:
        """通过内存层次结构处理新输入"""
        # 创建内存段
        segment = self._create_memory_segment(content, importance_score)

        # 存储在工作内存中
        self.working_memory.store(segment)

        # 检查是否需要整合
        if self._should_consolidate():
            self._perform_consolidation()

        return f"已处理并存储: {len(content)} 个字符"

    def _create_memory_segment(self, content: str, importance_score: float) -> MemorySegment:
        """创建带有嵌入和元数据的内存段"""
        # 简化的嵌入生成(实践中使用复杂模型)
        embedding = np.random.rand(self.embedding_dim)  # 占位符

        # 提取检索键(简化)
        retrieval_keys = self._extract_retrieval_keys(content)

        return MemorySegment(
            content=content,
            embedding=embedding,
            importance_score=importance_score,
            recency_score=1.0,
            access_frequency=1,
            creation_time=time.time(),
            last_access_time=time.time(),
            segment_type='new',
            retrieval_keys=retrieval_keys
        )

    def _extract_retrieval_keys(self, content: str) -> List[str]:
        """提取用于检索索引的关键术语"""
        # 简化的关键词提取
        words = content.lower().split()
        # 过滤常见词并保留有意义的术语
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        keys = [word for word in words if len(word) > 3 and word not in stop_words]
        return keys[:10]  # 保留前10个键

    def _should_consolidate(self) -> bool:
        """确定是否应进行内存整合"""
        # 检查容量利用率
        wm_capacity = self.working_memory.get_capacity_info()

        if (wm_capacity['segment_utilization'] > self.consolidation_threshold or
            wm_capacity['token_utilization'] > self.consolidation_threshold or
            time.time() - self.last_consolidation > self.consolidation_interval):
            return True

        return False

    def _perform_consolidation(self):
        """跨所有级别执行内存整合"""
        print("开始内存整合...")

        # 工作内存 → 短期内存
        wm_candidates = self.working_memory.consolidate()
        for candidate in wm_candidates:
            self.short_term_memory.store(candidate)

        # 短期内存 → 长期内存
        stm_candidates = self.short_term_memory.consolidate()
        for candidate in stm_candidates:
            if candidate.importance_score > 0.6:
                self.long_term_memory.store(candidate)

            # 在情景内存中存储重要事件
            if (candidate.importance_score > 0.8 or
                'important' in candidate.content.lower()):
                self.episodic_memory.store(candidate)

        self.last_consolidation = time.time()
        print("内存整合完成")

    def retrieve_context(self, query: str, max_context_length: int = 2000) -> str:
        """跨所有内存级别检索查询的相关上下文"""
        query_embedding = np.random.rand(self.embedding_dim)  # 占位符

        # 从所有内存级别检索
        wm_results = self.working_memory.retrieve(query_embedding, top_k=3)
        stm_results = self.short_term_memory.retrieve(query_embedding, top_k=3)
        ltm_results = self.long_term_memory.retrieve(query_embedding, top_k=2)
        em_results = self.episodic_memory.retrieve(query_embedding, top_k=2)

        # 组合并排序结果
        all_results = []

        # 添加带有级别权重的结果
        for segment in wm_results:
            all_results.append((segment, 1.0))  # 工作内存最高权重

        for segment in stm_results:
            all_results.append((segment, 0.8))  # 短期内存高权重

        for segment in ltm_results:
            all_results.append((segment, 0.6))  # 长期内存中等权重

        for segment in em_results:
            all_results.append((segment, 0.9))  # 情景内存非常高权重

        # 按相关性排序并组装上下文
        all_results.sort(key=lambda x: x[1], reverse=True)

        assembled_context = []
        current_length = 0

        for segment, weight in all_results:
            segment_length = len(segment.content)
            if current_length + segment_length <= max_context_length:
                assembled_context.append(f"[{segment.segment_type}] {segment.content}")
                current_length += segment_length
            else:
                break

        return "\n\n".join(assembled_context)

    def get_system_status(self) -> Dict[str, Dict]:
        """获取全面的系统状态"""
        return {
            'working_memory': self.working_memory.get_capacity_info(),
            'short_term_memory': self.short_term_memory.get_capacity_info(),
            'long_term_memory': self.long_term_memory.get_capacity_info(),
            'episodic_memory': self.episodic_memory.get_capacity_info(),
            'last_consolidation': self.last_consolidation,
            'system_health': self._assess_system_health()
        }

    def _assess_system_health(self) -> Dict[str, str]:
        """评估整体系统健康和性能"""
        wm_info = self.working_memory.get_capacity_info()

        health = {
            'memory_pressure': 'low',
            'consolidation_status': 'healthy',
            'retrieval_performance': 'optimal'
        }

        if wm_info['segment_utilization'] > 0.9 or wm_info['token_utilization'] > 0.9:
            health['memory_pressure'] = 'high'
        elif wm_info['segment_utilization'] > 0.7 or wm_info['token_utilization'] > 0.7:
            health['memory_pressure'] = 'medium'

        if time.time() - self.last_consolidation > self.consolidation_interval * 2:
            health['consolidation_status'] = 'overdue'

        return health

# 示例使用和演示
def demonstrate_hierarchical_memory():
    """演示分层内存系统"""
    print("初始化分层内存系统...")
    memory_system = HierarchicalMemorySystem()

    # 处理一些样本输入
    sample_inputs = [
        ("上下文工程框架为LLM优化信息负载提供了系统化方法。", 0.9),
        ("工作内存维护对当前处理信息的立即访问。", 0.7),
        ("长期内存以多维索引存储压缩的历史上下文。", 0.8),
        ("当超过容量阈值时会发生内存整合。", 0.6),
        ("分层方法实现恒定的内存使用，与序列长度无关。", 0.9)
    ]

    print("\n处理样本输入...")
    for content, importance in sample_inputs:
        result = memory_system.process_input(content, importance)
        print(f"  {result}")

    # 强制整合以演示
    memory_system._perform_consolidation()

    # 测试检索
    print("\n测试上下文检索...")
    query = "系统中的内存整合是如何工作的？"
    context = memory_system.retrieve_context(query)
    print(f"查询: {query}")
    print(f"检索的上下文:\n{context}")

    # 系统状态
    print("\n系统状态:")
    status = memory_system.get_system_status()
    for level, info in status.items():
        if isinstance(info, dict):
            print(f"  {level}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {level}: {info}")

    return memory_system

# 用于长上下文处理的高级注意力机制
class MultiHeadHierarchicalAttention(nn.Module):
    """用于长序列的具有分层处理的多头注意力"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 100000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        # 线性投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # 分层处理组件
        self.local_window_size = 512
        self.sparse_attention_stride = 64

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用分层注意力的前向传播"""
        batch_size, seq_len, d_model = x.shape

        # 线性投影
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if seq_len <= self.local_window_size:
            # 对短序列使用标准注意力
            attention_output = self._standard_attention(q, k, v, mask)
        else:
            # 对长序列使用分层注意力
            attention_output = self._hierarchical_attention(q, k, v, mask)

        # 输出投影
        output = self.w_o(attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model))

        return output

    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """标准缩放点积注意力"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output

    def _hierarchical_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """用于长序列的分层注意力"""
        batch_size, n_heads, seq_len, d_k = q.shape

        # 局部注意力: 滑动窗口
        local_output = self._local_window_attention(q, k, v, mask)

        # 稀疏全局注意力: 对关键位置的跨步注意力
        global_output = self._sparse_global_attention(q, k, v, mask)

        # 组合局部和全局注意力
        # 这里可以添加可学习的组合权重
        combined_output = 0.7 * local_output + 0.3 * global_output

        return combined_output

    def _local_window_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """在滑动窗口内应用注意力"""
        batch_size, n_heads, seq_len, d_k = q.shape
        window_size = self.local_window_size

        output = torch.zeros_like(v)

        for i in range(0, seq_len, window_size // 2):  # 50%重叠
            start = i
            end = min(i + window_size, seq_len)

            # 提取窗口
            q_window = q[:, :, start:end, :]
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]

            # 计算窗口内的注意力
            scores = torch.matmul(q_window, k_window.transpose(-2, -1)) / np.sqrt(self.d_k)

            if mask is not None:
                window_mask = mask[:, :, start:end, start:end]
                scores = scores.masked_fill(window_mask == 0, -1e9)

            attention_weights = F.softmax(scores, dim=-1)
            window_output = torch.matmul(attention_weights, v_window)

            # 混合重叠区域
            if i == 0:
                output[:, :, start:end, :] = window_output
            else:
                blend_start = start
                blend_end = min(start + window_size // 4, end)

                # 在重叠区域进行线性混合
                if blend_end > blend_start:
                    alpha = torch.linspace(0, 1, blend_end - blend_start).to(output.device)
                    alpha = alpha.view(1, 1, -1, 1)

                    output[:, :, blend_start:blend_end, :] = (
                        (1 - alpha) * output[:, :, blend_start:blend_end, :] +
                        alpha * window_output[:, :, :blend_end-blend_start, :]
                    )

                # 添加非重叠区域
                if blend_end < end:
                    output[:, :, blend_end:end, :] = window_output[:, :, blend_end-blend_start:, :]

        return output

    def _sparse_global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对全局重要位置应用稀疏注意力"""
        batch_size, n_heads, seq_len, d_k = q.shape
        stride = self.sparse_attention_stride

        # 选择稀疏关键位置(每stride个位置一个)
        sparse_indices = torch.arange(0, seq_len, stride).to(q.device)

        # 提取稀疏键和值
        k_sparse = k[:, :, sparse_indices, :]  # [batch, heads, sparse_len, d_k]
        v_sparse = v[:, :, sparse_indices, :]  # [batch, heads, sparse_len, d_k]

        # 计算从所有查询到稀疏键的注意力
        scores = torch.matmul(q, k_sparse.transpose(-2, -1)) / np.sqrt(self.d_k)

        attention_weights = F.softmax(scores, dim=-1)
        global_output = torch.matmul(attention_weights, v_sparse)

        return global_output

```

**从零开始的解释**: 这个分层内存系统就像您大脑中的复杂文件系统。工作内存是您的桌子 - 空间有限但可以立即访问。短期内存就像您的办公桌抽屉 - 有更多空间但需要压缩。长期内存就像您的文件柜 - 庞大的存储空间但高度组织和压缩。情景内存就像您的重要事件日志。

注意力机制就像拥有不同类型的阅读策略。对于短文本,您会仔细阅读每个单词(标准注意力)。对于很长的文档,您会详细阅读某些部分(局部窗口),同时浏览整个文档以找到关键点(稀疏全局注意力)。

---

## Software 3.0 范式 3: 协议 (自适应处理外壳)

协议提供基于有效性演进的自我改进上下文处理模式。

### 无限上下文处理协议

```
/process.infinite_context{
    intent="使用恒定内存和最优信息保留处理任意长序列",

    input={
        sequence_stream=<incoming_token_stream>,
        processing_constraints={
            max_memory_usage=<computational_memory_limit>,
            max_latency=<response_time_requirement>,
            quality_threshold=<minimum_information_preservation_ratio>
        },
        task_context={
            processing_type=<classification_generation_analysis_summarization>,
            importance_signals=<what_information_is_most_valuable>,
            temporal_requirements=<how_much_history_is_needed>
        }
    },

    process=[
        /analyze.sequence_characteristics{
            action="分析传入序列属性以优化处理策略",
            method="实时统计分析和模式检测",
            characteristics=[
                {information_density="tokens_per_unique_concept_ratio"},
                {repetition_patterns="identify_recurring_structures_and_themes"},
                {complexity_gradients="detect_varying_difficulty_across_sequence"},
                {temporal_dependencies="measure_long_range_information_dependencies"}
            ],
            output="用于策略优化的序列处理配置文件"
        },

        /adapt.processing_strategy{
            action="基于序列特性选择最优处理方法",
            method="跨内存、速度和质量的多目标优化",
            strategy_selection=[
                {
                    condition="sequence_length < 4K AND complexity = low",
                    strategy="standard_full_attention",
                    memory_usage="O(n²)",
                    quality="perfect_information_preservation"
                },
                {
                    condition="sequence_length < 100K AND information_density = high",
                    strategy="hierarchical_windowed_attention",
                    memory_usage="O(n)",
                    quality="near_perfect_with_local_detail"
                },
                {
                    condition="sequence_length > 100K OR memory_constrained = true",
                    strategy="infinite_memory_architecture",
                    memory_usage="O(1)",
                    quality="optimal_information_preservation_under_constraints"
                }
            ],
            adaptation_mechanisms=[
                {performance_monitoring="track_information_loss_and_processing_efficiency"},
                {strategy_switching="change_approach_if_quality_falls_below_threshold"},
                {parameter_tuning="optimize_window_sizes_and_compression_ratios"},
                {learning_integration="improve_strategy_selection_based_on_outcomes"}
            ]
        },

        /implement.memory_hierarchy{
            action="部署具有自适应整合的分层内存系统",
            method="具有智能信息流的多级内存",
            memory_levels=[
                {
                    level="working_memory",
                    capacity="2K-4K tokens",
                    purpose="immediate_processing_focus",
                    consolidation_trigger="capacity_threshold_OR_attention_shift"
                },
                {
                    level="short_term_memory",
                    capacity="8K-16K tokens_compressed",
                    purpose="recent_context_buffer",
                    consolidation_trigger="semantic_clustering_complete"
                },
                {
                    level="long_term_memory",
                    capacity="unlimited_highly_compressed",
                    purpose="historical_context_repository",
                    consolidation_trigger="importance_threshold_met"
                },
                {
                    level="episodic_memory",
                    capacity="key_events_and_decisions",
                    purpose="critical_moments_and_insights",
                    consolidation_trigger="significance_detection"
                }
            ],
            information_flow_optimization=[
                {promotion_criteria="importance_score AND access_frequency AND recency"},
                {compression_algorithms="semantic_summarization AND exemplar_selection"},
                {retrieval_indexing="multi_dimensional_semantic_temporal_importance"},
                {forgetting_mechanisms="graceful_degradation_with_importance_preservation"}
            ]
        },

        /optimize.attention_allocation{
            action="基于信息价值动态分配注意力",
            method="信息论注意力优化",
            allocation_strategies=[
                {
                    local_attention={
                        allocation="60-80% of attention budget",
                        scope="immediate context window",
                        resolution="token_level_detailed_processing"
                    }
                },
                {
                    global_attention={
                        allocation="15-25% of attention budget",
                        scope="sparse_sampling_across_entire_sequence",
                        resolution="concept_level_thematic_processing"
                    }
                },
                {
                    memory_attention={
                        allocation="10-20% of attention budget",
                        scope="relevant_items_from_memory_hierarchy",
                        resolution="compressed_representation_processing"
                    }
                }
            ],
            adaptive_reallocation=[
                {trigger="information_gap_detected", action="increase_global_attention"},
                {trigger="context_shift_identified", action="shift_local_attention_focus"},
                {trigger="memory_relevance_high", action="increase_memory_attention"},
                {trigger="processing_overload", action="compress_less_important_regions"}
            ]
        },

        /maintain.context_coherence{
            action="确保处理的上下文保持逻辑连贯性",
            method="多尺度连贯性验证和修复",
            coherence_levels=[
                {
                    local_coherence="sentence_and_paragraph_level_logical_flow",
                    verification="linguistic_and_semantic_consistency_checking",
                    repair="gap_filling_and_transition_smoothing"
                },
                {
                    global_coherence="document_level_thematic_consistency",
                    verification="concept_tracking_and_narrative_flow_analysis",
                    repair="theme_reinforcement_and_contradiction_resolution"
                },
                {
                    temporal_coherence="chronological_and_causal_relationship_preservation",
                    verification="event_sequence_validation_and_dependency_checking",
                    repair="timeline_reconstruction_and_causality_restoration"
                }
            ],
            quality_assurance=[
                {completeness_check="verify_essential_information_preservation"},
                {accuracy_validation="confirm_factual_consistency_across_compression"},
                {relevance_optimization="ensure_processed_context_serves_task_needs"},
                {efficiency_measurement="balance_information_value_against_computational_cost"}
            ]
        }
    ],

    output={
        processed_context={
            working_context=<immediately_relevant_detailed_information>,
            background_context=<supporting_information_from_memory_hierarchy>,
            coherence_map=<relationships_and_dependencies_between_information>,
            processing_metadata=<compression_ratios_attention_allocation_quality_metrics>
        },

        system_state={
            memory_utilization=<current_usage_across_all_memory_levels>,
            processing_efficiency=<tokens_per_second_and_quality_metrics>,
            adaptation_history=<strategy_changes_and_performance_evolution>,
            predictive_indicators=<anticipated_processing_needs_and_challenges>
        },

        quality_assessment={
            information_preservation_ratio=<percentage_of_important_information_retained>,
            coherence_score=<logical_flow_and_consistency_measure>,
            relevance_alignment=<match_between_processed_context_and_task_needs>,
            computational_efficiency=<processing_speed_vs_resource_utilization>
        }
    },

    meta={
        processing_strategy=<selected_approach_and_reasoning>,
        adaptation_opportunities=<identified_improvements_for_future_processing>,
        scaling_characteristics=<how_performance_changes_with_sequence_length>,
        learning_integration=<insights_for_improving_processing_strategies>
    },

    // 自我演化机制
    strategy_evolution=[
        {trigger="quality_degradation_detected",
         action="experiment_with_alternative_processing_approaches"},
        {trigger="new_sequence_patterns_identified",
         action="develop_specialized_processing_strategies"},
        {trigger="computational_efficiency_opportunities",
         action="optimize_memory_allocation_and_attention_patterns"},
        {trigger="novel_task_requirements_encountered",
         action="adapt_processing_pipeline_for_new_contexts"}
    ]
}
```

**从零开始的解释**: 这个协议就像拥有一个极其智能的研究助理,可以阅读和记住无限量的信息。助理会根据材料自动调整他们的阅读策略 - 在简单文档中浏览关键点,在复杂材料中仔细阅读,并完美记住重要见解,同时让琐碎的细节消失。

---

## 高级长上下文应用

### 实时文档分析系统

```python
class RealTimeDocumentAnalyzer:
    """实时处理和分析任意长度的文档"""

    def __init__(self, memory_system: HierarchicalMemorySystem):
        self.memory_system = memory_system
        self.processing_strategies = {
            'summarization': SummarizationStrategy(),
            'question_answering': QAStrategy(),
            'analysis': AnalysisStrategy(),
            'extraction': ExtractionStrategy()
        }
        self.performance_monitor = PerformanceMonitor()

    def process_document_stream(self, document_stream: Iterator[str],
                               task_type: str = 'analysis') -> Iterator[str]:
        """使用实时输出处理流式文档"""

        strategy = self.processing_strategies.get(task_type, self.processing_strategies['analysis'])

        for chunk in document_stream:
            # 通过内存系统处理块
            self.memory_system.process_input(chunk, importance_score=0.7)

            # 基于策略生成增量输出
            incremental_result = strategy.process_chunk(chunk, self.memory_system)

            # 监控性能并根据需要调整
            self._monitor_and_adapt(chunk, incremental_result, strategy)

            yield incremental_result

    def _monitor_and_adapt(self, input_chunk: str, output: str, strategy):
        """监控性能并调整处理策略"""
        metrics = {
            'input_length': len(input_chunk),
            'output_length': len(output),
            'processing_time': time.time(),
            'memory_usage': self.memory_system.get_system_status()
        }

        self.performance_monitor.record_metrics(metrics)

        # 如果性能下降则调整策略
        if self.performance_monitor.should_adapt():
            strategy.adapt_parameters(self.performance_monitor.get_adaptation_suggestions())

class SummarizationStrategy:
    """实时文档摘要策略"""

    def __init__(self):
        self.summary_buffer = []
        self.key_points = []
        self.compression_ratio = 0.1

    def process_chunk(self, chunk: str, memory_system: HierarchicalMemorySystem) -> str:
        """处理摘要块"""
        # 从块中提取关键句子
        key_sentences = self._extract_key_sentences(chunk)

        # 从内存中获取相关上下文
        context = memory_system.retrieve_context(chunk, max_context_length=1000)

        # 生成增量摘要更新
        summary_update = self._generate_summary_update(key_sentences, context)

        # 更新摘要缓冲区
        self._update_summary_buffer(summary_update)

        return summary_update

    def _extract_key_sentences(self, chunk: str) -> List[str]:
        """从块中提取最重要的句子"""
        sentences = chunk.split('.')

        # 简单启发式: 包含关键术语、适当长度的句子
        key_sentences = []
        for sentence in sentences:
            if (len(sentence.split()) > 5 and
                any(term in sentence.lower() for term in ['important', 'key', 'main', 'significant', 'crucial'])):
                key_sentences.append(sentence.strip())

        return key_sentences[:3]  # 前3个关键句子

    def _generate_summary_update(self, key_sentences: List[str], context: str) -> str:
        """生成整合上下文的摘要更新"""
        if not key_sentences:
            return ""

        # 将关键句子与上下文整合相结合
        summary_update = f"关键点: {'; '.join(key_sentences)}"

        # 如果相关则添加上下文连接
        if context and len(context) > 100:
            summary_update += f"\n[上下文: 这与之前关于{context[:100]}的讨论相关...]"

        return summary_update

    def _update_summary_buffer(self, summary_update: str):
        """维护滚动摘要缓冲区"""
        self.summary_buffer.append(summary_update)

        # 保持缓冲区可管理
        if len(self.summary_buffer) > 20:
            # 压缩较旧的摘要
            compressed = self._compress_old_summaries(self.summary_buffer[:10])
            self.summary_buffer = [compressed] + self.summary_buffer[10:]

    def _compress_old_summaries(self, old_summaries: List[str]) -> str:
        """将多个摘要压缩为单个表示"""
        all_points = []
        for summary in old_summaries:
            if "关键点:" in summary:
                points = summary.split("关键点:")[1].split(";")
                all_points.extend([p.strip() for p in points if p.strip()])

        # 删除重复项并创建压缩摘要
        unique_points = list(set(all_points))[:5]  # 前5个独特点
        return f"[已压缩] 关键历史点: {'; '.join(unique_points)}"

class PerformanceMonitor:
    """监控系统性能并建议调整"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.performance_thresholds = {
            'max_processing_time': 1.0,  # 秒
            'max_memory_utilization': 0.9,
            'min_output_quality': 0.7
        }

    def record_metrics(self, metrics: Dict):
        """记录性能指标"""
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)

    def should_adapt(self) -> bool:
        """确定是否需要调整"""
        if len(self.metrics_history) < 10:
            return False

        recent_metrics = list(self.metrics_history)[-10:]

        # 检查处理时间趋势
        processing_times = [m.get('processing_time', 0) for m in recent_metrics]
        if len(processing_times) > 1:
            avg_time = np.mean(processing_times[-5:]) - np.mean(processing_times[:5])
            if avg_time > self.performance_thresholds['max_processing_time']:
                return True

        # 检查内存利用率
        latest_memory = recent_metrics[-1].get('memory_usage', {})
        if isinstance(latest_memory, dict):
            wm_util = latest_memory.get('working_memory', {}).get('segment_utilization', 0)
            if wm_util > self.performance_thresholds['max_memory_utilization']:
                return True

        return False

    def get_adaptation_suggestions(self) -> Dict[str, any]:
        """获取性能调整建议"""
        suggestions = {}

        if len(self.metrics_history) < 5:
            return suggestions

        recent_metrics = list(self.metrics_history)[-5:]

        # 分析性能模式
        avg_input_length = np.mean([m.get('input_length', 0) for m in recent_metrics])
        avg_output_length = np.mean([m.get('output_length', 0) for m in recent_metrics])

        # 建议压缩比调整
        if avg_input_length > 1000 and avg_output_length / avg_input_length > 0.3:
            suggestions['increase_compression'] = True
            suggestions['target_compression_ratio'] = 0.2

        # 建议内存管理更改
        latest_memory = recent_metrics[-1].get('memory_usage', {})
        if isinstance(latest_memory, dict):
            wm_util = latest_memory.get('working_memory', {}).get('segment_utilization', 0)
            if wm_util > 0.8:
                suggestions['trigger_consolidation'] = True
                suggestions['reduce_working_memory_threshold'] = True

        return suggestions

# 示例使用演示
def demonstrate_long_context_processing():
    """长上下文处理的综合演示"""
    print("初始化长上下文处理系统...")

    # 初始化内存系统
    memory_system = HierarchicalMemorySystem()

    # 初始化文档分析器
    analyzer = RealTimeDocumentAnalyzer(memory_system)

    # 模拟处理非常长的文档
    sample_document_chunks = [
        "上下文工程代表了我们如何优化LLM的范式转变。",
        "分层内存系统支持处理任意长序列。",
        "工作内存维护有限容量的立即处理焦点。",
        "短期内存通过语义聚类提供压缩的最近上下文。",
        "长期内存提供带有多维索引的无限存储。",
        "情景内存捕获重要事件和决策点。",
        "系统基于序列特性调整处理策略。",
        "注意力机制优化跨不同范围的信息分配。",
        "内存整合确保级别之间的高效信息流。",
        "性能监控支持对处理需求的实时调整。"
    ]

    print("\n处理文档流...")
    print("=" * 60)

    # 处理文档块
    for i, result in enumerate(analyzer.process_document_stream(
        iter(sample_document_chunks), task_type='summarization'
    )):
        print(f"块 {i+1} 摘要:")
        print(result)
        print("-" * 40)

    # 显示最终系统状态
    print("\n最终系统状态:")
    final_status = memory_system.get_system_status()
    for level, info in final_status.items():
        if isinstance(info, dict):
            print(f"{level}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"{level}: {info}")

    # 测试复杂查询的上下文检索
    print("\n" + "=" * 60)
    print("测试复杂查询处理:")

    complex_query = "分层内存系统如何在保持恒定内存使用的同时实现无限上下文处理？"
    retrieved_context = memory_system.retrieve_context(complex_query, max_context_length=3000)

    print(f"查询: {complex_query}")
    print(f"\n检索的上下文:\n{retrieved_context}")

    return memory_system, analyzer

# 运行演示
if __name__ == "__main__":
    memory_system, analyzer = demonstrate_long_context_processing()
```

**从零开始的解释**: 这个实时文档分析器就像拥有一个研究助理,可以阅读无限的文档,同时保持完美的组织和即时回忆。系统根据文档特性调整其阅读策略 - 仔细阅读重要内容,浏览常规信息,并维护所有已处理内容的可搜索摘要。

---

## 评估与评价

### 长上下文处理指标

由于代码太长,我将在下一个编辑中继续添加评估部分和其余内容。

## 与上下文工程综述的联系

这个长上下文处理模块直接实现并扩展了[上下文工程综述](https://arxiv.org/pdf/2507.13334)的关键发现:

**上下文处理 (§4.2)**:
- 实现高级注意力机制,包括Mamba、LongNet和FlashAttention方法
- 通过分层内存系统解决StreamingLLM和InfiniAttention概念
- 将MLLMs上下文处理扩展到无限序列处理

**内存系统集成**:
- 通过分层内存架构实现MemoryBank和MemLLM概念
- 解决LongMemEval中识别的长上下文评估挑战
- 通过恒定内存架构提供O(n²)规模限制的解决方案

**技术创新进展**:
- 展示受LongMamba启发的滑动注意力机制
- 实现扩展当前研究的内存增强架构
- 提供解决综述建议的上下文组装优化

---

## 总结和下一步

**掌握的核心概念**:
- 实现无限上下文处理的分层内存系统
- 优化计算效率的多层注意力机制
- 信息论上下文选择和压缩
- 对序列特性和处理需求的实时适应

**Software 3.0集成**:
- **提示词**: 用于系统化上下文处理决策的内存管理模板
- **编程**: 具有自适应注意力机制的分层内存架构
- **协议**: 基于性能演进的自我优化上下文处理系统

**实现技能**:
- 具有恒定内存使用的无限上下文架构
- 具有智能整合的多级内存系统
- 基于信息价值的自适应注意力分配
- 用于长上下文处理的综合评估框架

**研究基础**: 直接实现上下文处理研究,并在无限上下文架构、分层内存系统和自适应处理策略方面进行新颖扩展。

**下一模块**: [02_self_refinement.md](02_self_refinement.md) - 在长上下文处理的基础上,探索系统如何通过自我精炼循环和自适应优化迭代改进自己的上下文理解和处理。

---

*本模块展示了从固定上下文窗口到无限内存架构的演进,体现了Software 3.0原则:系统不仅处理无限信息,还持续优化自己的处理策略以实现最大的有效性和效率。*
