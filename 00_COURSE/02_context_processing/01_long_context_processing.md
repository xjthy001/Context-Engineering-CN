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

## 与上下文工程综述的联系

这个长上下文处理模块直接实现并扩展了[上下文工程综述](https://arxiv.org/pdf/2507.13334)的关键发现。

---

## 总结和下一步

**掌握的核心概念**:
- 实现无限上下文处理的分层内存系统
- 优化计算效率的多层注意力机制
- 信息论上下文选择和压缩
- 对序列特性和处理需求的实时适应

**下一模块**: [02_self_refinement.md](02_self_refinement.md) - 在长上下文处理的基础上,探索系统如何通过自我精炼循环和自适应优化迭代改进自己的上下文理解和处理。

---

*本模块展示了从固定上下文窗口到无限内存架构的演进,体现了Software 3.0原则:系统不仅处理无限信息,还持续优化自己的处理策略以实现最大的有效性和效率。*
