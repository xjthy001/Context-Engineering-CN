# 信息论：量化上下文质量和相关性
## 从直觉相关性到数学精确性

> **模块 00.3** | *上下文工程课程：从基础到前沿系统*
>
> *"信息是不确定性的消解" — Claude Shannon*

---

## 从猜测到信息科学

你已经学会了形式化上下文和优化组装函数。现在来了一个基本问题：**我们如何衡量上下文组件的信息价值？**

### 通用信息挑战

考虑这些常见的信息场景：

**通信中的信号与噪声**：
```
清晰的电话通话：高信息内容，低噪声
有杂音的通话：相同信息，但更难提取（低信噪比）
```

**相关与不相关的搜索结果**：
```
精准搜索：结果直接回答你的问题（高相关性）
广泛搜索：很多结果，但很少真正有帮助（低信息密度）
```

**上下文工程信息问题**：
```
高质量上下文：在token约束内的最大相关信息
低质量上下文：相关和不相关信息的混合（低效）
```

**模式**：在每种情况下，我们都想最大化有用信息，同时最小化噪声、不相关性或冗余。

---

## 信息论的数学基础

### 核心信息概念

#### 信息内容（惊奇度）
```
I(x) = -log₂(P(x))

其中：
I(x) = 事件x的信息内容（以比特为单位）
P(x) = 事件x发生的概率

直觉：罕见事件比常见事件包含更多信息
```

**视觉理解**：
```
    信息内容
       ↑
    10 │████ "AI系统变得有意识" (非常罕见，高信息量)
       │
     5 │██ "今天下雨" (有点罕见，中等信息量)
       │
     1 │▌ "今天早上太阳升起" (非常常见，低信息量)
       │
     0 └─────────────────────────────────────►
        0    0.5    1.0     事件概率
```

#### 熵（平均信息）
```
H(X) = -Σ P(x) × log₂(P(x))

其中：
H(X) = 随机变量X的熵（平均信息内容）
P(x) = 每个可能结果x的概率

直觉：熵衡量不确定性 - 我们平均期望多少信息
```

#### 互信息（共享信息）
```
I(X;Y) = H(X) + H(Y) - H(X,Y)

其中：
I(X;Y) = X和Y之间的互信息
H(X,Y) = X和Y的联合熵

直觉：知道Y能告诉我们关于X多少信息（反之亦然）
```

**从零开始的解释**：信息论提供了衡量信息内容的数学工具，就像物理学提供了衡量能量的工具一样。熵衡量某物平均包含多少信息，而互信息衡量两条信息的重叠或相关程度。

---

## Software 3.0 范式一：提示词（信息评估模板）

提示词提供了系统化框架，用于分析和优化上下文组件中的信息内容。

### 信息相关性评估模板

<pre>
```markdown
# 信息相关性分析框架

## 相关性量化策略
**目标**：系统化地衡量每条信息与用户查询的相关程度
**方法**：具有数学精确性的多维相关性评分

## 语义相关性分析

### 1. 直接相关性（主要维度）
**定义**：这条信息在多大程度上直接解决核心查询？
**测量框架**：

Direct_Relevance(info, query) = Semantic_Similarity(info_embedding, query_embedding)

其中：
- Semantic_Similarity 使用嵌入向量之间的余弦相似度
- 范围：[0, 1]，其中1 = 完美语义匹配
- 包含阈值：通常 ≥ 0.6


**评估问题**：
- 这条信息是否直接回答用户的问题？
- 移除这条信息会使响应不完整吗？
- 这条信息对查询的核心意图有多重要？

**评分标准**：
- **0.9-1.0**：信息直接回答查询
- **0.7-0.9**：信息强力支持回答查询
- **0.5-0.7**：信息为查询提供有用的上下文
- **0.3-0.5**：信息与查询有切线关系
- **0.0-0.3**：信息与查询不相关

### 2. 上下文相关性（次要维度）
**定义**：这条信息与所需的更广泛上下文和背景有何关系？
**测量框架**：

Contextual_Relevance(info, context) =
    α × Background_Importance(info) +
    β × Dependency_Strength(info, other_components) +
    γ × Completeness_Contribution(info)

其中 α + β + γ = 1（加权组合）


**评估标准**：
- **背景重要性**：理解所必需的 vs. 有了更好的
- **依赖强度**：其他信息对此的依赖程度
- **完整性贡献**：这条信息对整体完整性的贡献

### 3. 信息效率分析
**定义**：每个token这个组件提供多少有价值的信息？
**测量框架**：

Information_Efficiency(component) =
    Information_Value(component) / Token_Count(component)

其中 Information_Value 结合了：
- Relevance_Score × Importance_Weight
- Uniqueness_Factor（对冗余信息的惩罚）
- Credibility_Multiplier（对权威来源的加成）


**效率优化问题**：
- 这条信息能否在不损失价值的情况下更简洁地表达？
- 是否存在可以消除的与其他组件的冗余？
- 有效传达这条信息所需的最少token数是多少？

## 信息价值计算

### 综合信息评分

Total_Information_Value(component) =
    w₁ × Direct_Relevance(component) +
    w₂ × Contextual_Relevance(component) +
    w₃ × Information_Efficiency(component) +
    w₄ × Source_Credibility(component) +
    w₅ × Information_Freshness(component)

其中：w₁ + w₂ + w₃ + w₄ + w₅ = 1


## 冗余检测框架

### 信息重叠分析

Redundancy_Score(component_A, component_B) =
    Mutual_Information(A, B) / min(H(A), H(B))

其中：
- 高冗余 (>0.8)：考虑合并或删除一个组件
- 中等冗余 (0.4-0.8)：寻找要保留的互补方面
- 低冗余 (<0.4)：两个组件都提供独特价值


### 多样性优化策略

目标：在最小化冗余的同时最大化信息覆盖

Optimal_Component_Set = arg max[Σ Information_Value(cᵢ) - λ × Σᵢⱼ Redundancy(cᵢ, cⱼ)]

其中 λ 控制冗余信息的惩罚
```
</pre>

**从零开始的解释**：这个模板提供了一种系统化的方法来衡量信息价值，就像有一个精确的秤来称量不同信息片段的有用性。它帮助你识别什么增加了真正的价值，什么只是占用空间。

### 互信息优化模板

```xml
<mutual_information_optimization>
  <objective>最大化上下文组件与用户查询之间的互信息</objective>

  <mutual_information_framework>
    <definition>
      I(Context; Query) = H(Query) - H(Query|Context)

      解释：
      - H(Query)：没有上下文时关于查询的不确定性
      - H(Query|Context)：给定上下文时关于查询的不确定性
      - I(Context; Query)：上下文提供的关于查询的信息
    </definition>

    <optimization_target>
      最大化：I(Context; Query) = Σᵢ I(component_i; Query) - Redundancy_Penalty

      约束条件：Token_Budget_Constraint
    </optimization_target>
  </mutual_information_framework>

  <component_selection_strategy>
    <greedy_approach>
      <step_1>为所有可用组件计算 I(component; Query)</step_1>
      <step_2>选择具有最高互信息的组件</step_2>
      <step_3>对于剩余组件，计算条件互信息：
        I(component; Query | already_selected_components)</step_3>
      <step_4>重复直到token预算耗尽</step_4>
    </greedy_approach>

    <optimal_approach>
      <description>找到全局最优的组件子集</description>
      <formulation>
        max Σᵢ∈S I(componentᵢ; Query) - λ × Σᵢ,ⱼ∈S I(componentᵢ; componentⱼ)

        约束条件：Σᵢ∈S tokens(componentᵢ) ≤ Budget
      </formulation>
      <solution_method>动态规划或整数线性规划</solution_method>
    </optimal_approach>
  </component_selection_strategy>

  <practical_implementation>
    <embedding_based_approximation>
      <mutual_information_estimate>
        I(component; query) ≈ 1 - JS_Divergence(P_component, P_query)

        其中 JS_Divergence 是从嵌入向量派生的概率分布之间的
        Jensen-Shannon散度
      </mutual_information_estimate>

      <conditional_mutual_information>
        I(component; query | context) ≈
          I(component; query) - α × max_j I(component; context_component_j)

        其中 α 控制冗余惩罚强度
      </conditional_mutual_information>
    </embedding_based_approximation>

    <frequency_based_approximation>
      <term_overlap_method>
        I(component; query) ≈
          |Unique_Terms(component) ∩ Terms(query)| / |Terms(query)|
      </term_overlap_method>

      <semantic_term_expansion>
        用同义词和相关概念扩展查询词
        计算与扩展词集的重叠
      </semantic_term_expansion>
    </frequency_based_approximation>
  </practical_implementation>

  <quality_validation>
    <information_coverage_check>
      确保选定的组件覆盖查询的所有主要方面：
      Coverage(components, query) = |Query_Aspects_Covered| / |Total_Query_Aspects|
    </information_coverage_check>

    <diminishing_returns_analysis>
      监控添加组件时的边际信息增益：
      如果 Marginal_I(new_component) < threshold，考虑停止选择
    </diminishing_returns_analysis>

    <coherence_validation>
      确保选定的组件形成连贯的信息集：
      Coherence = Average_Mutual_Information(component_pairs) - Conflict_Penalty
    </coherence_validation>
  </quality_validation>
</mutual_information_optimization>
```

**从零开始的解释**：这个XML模板提供了一种系统化的方法来选择能最大化与用户查询的互信息的信息组件，就像从图书馆中选择最相关的书籍来回答特定的研究问题。

### 信息压缩策略模板

```yaml
# 信息压缩策略模板
compression_optimization:

  objective: "在保留基本内容的同时最大化信息密度"

  compression_dimensions:
    semantic_compression:
      description: "在保留含义的同时减少冗余"
      techniques:
        - synonym_replacement: "用简洁的等价词替换冗长的短语"
        - redundancy_elimination: "删除重复信息"
        - concept_consolidation: "将相关概念合并为统一描述"

      measurement:
        compression_ratio: "原始tokens / 压缩后tokens"
        information_preservation: "semantic_similarity(原始, 压缩后)"
        target_preservation: ">= 0.95"

    syntactic_compression:
      description: "优化句子结构和用词"
      techniques:
        - passive_to_active_voice: "将被动结构转换为主动"
        - unnecessary_qualifier_removal: "删除模糊词和填充短语"
        - sentence_combination: "合并相关句子以提高简洁性"

      measurement:
        readability_preservation: "阅读难度评分比较"
        clarity_maintenance: "信息可访问性评估"

    structural_compression:
      description: "优化信息组织和呈现"
      techniques:
        - hierarchical_organization: "将相关信息分组"
        - bullet_point_conversion: "在适当时将散文转换为结构化列表"
        - example_consolidation: "将多个示例减少为最具说明性的"

  compression_strategies:
    lossy_compression:
      description: "删除被认为不太重要的信息"
      decision_criteria:
        - relevance_threshold: "删除低于相关性阈值的组件"
        - importance_ranking: "首先保留最高价值的信息"
        - user_priority_alignment: "保留用户明确优先考虑的信息"

      quality_control:
        - essential_information_preservation: "永远不要压缩关键事实"
        - accuracy_maintenance: "确保压缩不引入错误"
        - completeness_thresholds: "保持最低完整性水平"

    lossless_compression:
      description: "在不丢失信息内容的情况下减少tokens"
      techniques:
        - format_optimization: "使用更紧凑的表示格式"
        - reference_consolidation: "有效使用代词和引用"
        - abbreviation_standardization: "一致使用公认的缩写"

      validation:
        - information_equivalence: "验证压缩版本包含相同信息"
        - reconstructability: "确保可以恢复原始含义"
        - error_detection: "检查压缩引起的歧义"

  adaptive_compression:
    context_aware_compression:
      high_relevance_preservation: "对高度相关的内容应用最小压缩"
      background_information_compression: "对支持细节进行更激进的压缩"
      user_expertise_adjustment: "对专家用户更多压缩基本概念"

    token_budget_adaptation:
      emergency_compression: "严重超出token预算时的激进压缩"
      optimal_compression: "正常token压力下的平衡压缩"
      minimal_compression: "在预算范围内时的轻度压缩"

    quality_feedback_integration:
      user_satisfaction_monitoring: "跟踪用户对压缩内容的满意度"
      compression_strategy_adjustment: "根据反馈修改压缩"
      iterative_improvement: "随时间改进压缩算法"

  implementation_guidelines:
    compression_pipeline:
      step_1: "通过信息分析识别压缩机会"
      step_2: "根据内容类型应用适当的压缩技术"
      step_3: "验证压缩质量和信息保留"
      step_4: "根据token预算和质量要求调整压缩级别"

    quality_assurance:
      - pre_compression_analysis: "压缩前评估信息价值"
      - compression_impact_measurement: "量化压缩决策的影响"
      - post_compression_validation: "验证压缩内容符合质量标准"
      - user_feedback_integration: "将用户偏好纳入压缩策略"

    compression_monitoring:
      - compression_effectiveness_tracking: "监控压缩比与质量权衡"
      - user_satisfaction_correlation: "跟踪压缩与用户满意度的关系"
      - continuous_improvement: "基于经验数据改进压缩策略"
```

**从零开始的解释**：这个YAML模板提供了信息压缩的系统化方法，就像拥有专业的编辑技巧，在减少长度的同时保留含义。它在效率和质量保留之间取得平衡。

---

## Software 3.0 范式二：编程（信息算法）

编程提供了计算方法来测量、优化和管理上下文组件中的信息内容。

### 信息论实现

```python
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

@dataclass
class InformationMetrics:
    """信息论测量的容器"""
    entropy: float
    mutual_information: float
    conditional_entropy: float
    information_gain: float
    redundancy_score: float
    efficiency_ratio: float

class InformationAnalyzer:
    """用于上下文组件的综合信息论分析"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.vocabulary_stats = {}

    def calculate_entropy(self, text: str) -> float:
        """
        基于字符/单词频率计算文本的香农熵

        参数：
            text: 要分析的输入文本

        返回：
            熵值（比特）
        """

        if not text or len(text.strip()) == 0:
            return 0.0

        # 计算字符级熵
        char_counts = Counter(text.lower())
        total_chars = len(text)

        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy += -probability * math.log2(probability)

        return entropy

    def calculate_word_entropy(self, text: str) -> float:
        """基于单词频率计算熵"""

        words = text.lower().split()
        if not words:
            return 0.0

        word_counts = Counter(words)
        total_words = len(words)

        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            entropy += -probability * math.log2(probability)

        return entropy

    def calculate_mutual_information(self, text1: str, text2: str) -> float:
        """
        计算两个文本组件之间的互信息

        使用TF-IDF向量和熵计算来估计互信息
        """

        try:
            # 创建TF-IDF向量
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # 计算联合分布近似
            vec1 = tfidf_matrix[0].toarray().flatten()
            vec2 = tfidf_matrix[1].toarray().flatten()

            # 归一化以创建概率分布
            vec1_norm = vec1 / (np.sum(vec1) + 1e-10)
            vec2_norm = vec2 / (np.sum(vec2) + 1e-10)

            # 计算互信息近似
            joint_prob = np.outer(vec1_norm, vec2_norm)

            # 计算边际熵
            h1 = -np.sum(vec1_norm * np.log2(vec1_norm + 1e-10))
            h2 = -np.sum(vec2_norm * np.log2(vec2_norm + 1e-10))

            # 计算联合熵
            joint_prob_flat = joint_prob.flatten()
            h_joint = -np.sum(joint_prob_flat * np.log2(joint_prob_flat + 1e-10))

            # 互信息 = H(X) + H(Y) - H(X,Y)
            mutual_info = h1 + h2 - h_joint

            return max(0.0, mutual_info)

        except Exception as e:
            # 回退到更简单的基于重叠的度量
            return self._calculate_overlap_based_mi(text1, text2)

    def _calculate_overlap_based_mi(self, text1: str, text2: str) -> float:
        """基于单词重叠的回退互信息计算"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        # Jaccard相似度作为MI近似
        jaccard = overlap / union

        # 转换为互信息尺度（粗略近似）
        return -math.log2(1 - jaccard + 1e-10)

    def calculate_conditional_entropy(self, text_y: str, text_x: str) -> float:
        """
        计算 H(Y|X) - 给定X时Y的熵

        近似为 H(Y) - I(X;Y)
        """

        h_y = self.calculate_word_entropy(text_y)
        mi_xy = self.calculate_mutual_information(text_x, text_y)

        return max(0.0, h_y - mi_xy)

    def calculate_information_gain(self, text_before: str, additional_text: str) -> float:
        """
        计算添加额外文本的信息增益

        IG = H(before) - H(before|additional)
        """

        h_before = self.calculate_word_entropy(text_before)
        h_conditional = self.calculate_conditional_entropy(text_before, additional_text)

        return h_before - h_conditional

    def analyze_component_information(self, component_text: str,
                                    query_text: str) -> InformationMetrics:
        """
        上下文组件的综合信息分析

        参数：
            component_text: 组件的文本内容
            query_text: 用于相关性评估的用户查询

        返回：
            包含所有计算度量的InformationMetrics
        """

        # 计算基本信息度量
        entropy = self.calculate_word_entropy(component_text)
        mutual_info = self.calculate_mutual_information(component_text, query_text)
        conditional_entropy = self.calculate_conditional_entropy(query_text, component_text)
        information_gain = self.calculate_information_gain(query_text, component_text)

        # 计算冗余（自相似度度量）
        sentences = component_text.split('.')
        if len(sentences) > 1:
            redundancy_scores = []
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    if sentences[i].strip() and sentences[j].strip():
                        redundancy = self.calculate_mutual_information(
                            sentences[i], sentences[j]
                        )
                        redundancy_scores.append(redundancy)

            redundancy_score = np.mean(redundancy_scores) if redundancy_scores else 0.0
        else:
            redundancy_score = 0.0

        # 计算效率（每个token的信息）
        token_count = len(component_text.split())
        efficiency_ratio = mutual_info / (token_count + 1) if token_count > 0 else 0.0

        return InformationMetrics(
            entropy=entropy,
            mutual_information=mutual_info,
            conditional_entropy=conditional_entropy,
            information_gain=information_gain,
            redundancy_score=redundancy_score,
            efficiency_ratio=efficiency_ratio
        )

class InformationOptimizer:
    """基于信息论原理优化上下文组件"""

    def __init__(self):
        self.analyzer = InformationAnalyzer()
        self.optimization_history = []

    def optimize_component_selection(self, candidate_components: List[str],
                                   query: str, token_budget: int) -> List[str]:
        """
        选择最优组件子集以最大化与查询的互信息

        参数：
            candidate_components: 候选文本组件列表
            query: 用户查询
            token_budget: 允许的最大tokens

        返回：
            最优选择的组件
        """

        # 计算所有组件的信息度量
        component_metrics = []
        for i, component in enumerate(candidate_components):
            metrics = self.analyzer.analyze_component_information(component, query)
            token_count = len(component.split())

            component_metrics.append({
                'index': i,
                'component': component,
                'metrics': metrics,
                'token_count': token_count,
                'efficiency': metrics.mutual_information / (token_count + 1)
            })

        # 按效率排序（每个token的互信息）
        component_metrics.sort(key=lambda x: x['efficiency'], reverse=True)

        # 带冗余惩罚的贪心选择
        selected_components = []
        selected_indices = set()
        total_tokens = 0

        for comp_data in component_metrics:
            if comp_data['token_count'] + total_tokens <= token_budget:
                # 检查与已选择组件的冗余
                redundancy_penalty = 0.0

                for selected_comp in selected_components:
                    redundancy = self.analyzer.calculate_mutual_information(
                        comp_data['component'], selected_comp
                    )
                    redundancy_penalty += redundancy

                # 考虑冗余的调整分数
                adjusted_score = (comp_data['metrics'].mutual_information -
                                0.5 * redundancy_penalty)

                if adjusted_score > 0.1:  # 最小阈值
                    selected_components.append(comp_data['component'])
                    selected_indices.add(comp_data['index'])
                    total_tokens += comp_data['token_count']

        return selected_components

    def optimize_component_order(self, components: List[str], query: str) -> List[str]:
        """
        优化组件顺序以最大化信息流

        将与查询具有最高互信息的组件放在前面，
        然后是提供互补信息的组件
        """

        if len(components) <= 1:
            return components

        # 计算每个组件与查询的互信息
        mi_scores = []
        for comp in components:
            mi = self.analyzer.calculate_mutual_information(comp, query)
            mi_scores.append(mi)

        # 按与查询的互信息排序（降序）
        component_mi_pairs = list(zip(components, mi_scores))
        component_mi_pairs.sort(key=lambda x: x[1], reverse=True)

        return [comp for comp, _ in component_mi_pairs]

    def compress_component(self, component: str, target_compression: float = 0.8) -> str:
        """
        在保留最大信息内容的同时压缩组件

        参数：
            component: 原始组件文本
            target_compression: 目标长度占原始的比例（0.8 = 原始的80%）

        返回：
            压缩后的组件文本
        """

        sentences = [s.strip() for s in component.split('.') if s.strip()]

        if len(sentences) <= 1:
            return component  # 无法有意义地压缩单个句子

        # 计算每个句子的信息价值
        sentence_scores = []

        for sentence in sentences:
            # 基于熵和唯一性评分
            entropy = self.analyzer.calculate_word_entropy(sentence)

            # 对与其他句子的冗余进行惩罚
            redundancy_penalty = 0.0
            for other_sentence in sentences:
                if sentence != other_sentence:
                    redundancy = self.analyzer.calculate_mutual_information(
                        sentence, other_sentence
                    )
                    redundancy_penalty += redundancy

            score = entropy - 0.3 * redundancy_penalty
            sentence_scores.append((sentence, score))

        # 按分数排序并选择顶部句子以达到压缩目标
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        target_length = int(len(sentences) * target_compression)
        target_length = max(1, target_length)  # 至少保留一个句子

        selected_sentences = [s for s, _ in sentence_scores[:target_length]]

        # 保持逻辑顺序重构组件
        compressed_component = '. '.join(selected_sentences)

        return compressed_component

class MutualInformationMaximizer:
    """用于在上下文组装中最大化互信息的专用优化器"""

    def __init__(self, token_budget: int):
        self.token_budget = token_budget
        self.analyzer = InformationAnalyzer()

    def maximize_mutual_information(self, knowledge_base: List[str],
                                  query: str) -> Dict:
        """
        找到知识组件的最优组合以最大化 I(Context; Query)

        使用带前瞻的贪心算法来近似最优解
        """

        # 阶段1：计算个体互信息分数
        component_scores = []
        for i, component in enumerate(knowledge_base):
            mi_score = self.analyzer.calculate_mutual_information(component, query)
            token_count = len(component.split())

            component_scores.append({
                'index': i,
                'component': component,
                'mi_score': mi_score,
                'token_count': token_count,
                'efficiency': mi_score / (token_count + 1)
            })

        # 阶段2：考虑冗余的贪心选择
        selected = []
        remaining = component_scores.copy()
        total_tokens = 0
        total_mi = 0.0

        while remaining and total_tokens < self.token_budget:
            best_addition = None
            best_marginal_mi = 0.0

            for candidate in remaining:
                # 检查是否适合预算
                if total_tokens + candidate['token_count'] > self.token_budget:
                    continue

                # 计算边际互信息
                marginal_mi = candidate['mi_score']

                # 减去与已选组件的冗余
                for selected_comp in selected:
                    redundancy = self.analyzer.calculate_mutual_information(
                        candidate['component'], selected_comp['component']
                    )
                    marginal_mi -= 0.5 * redundancy  # 冗余惩罚

                if marginal_mi > best_marginal_mi:
                    best_marginal_mi = marginal_mi
                    best_addition = candidate

            if best_addition and best_marginal_mi > 0.01:  # 最小增益阈值
                selected.append(best_addition)
                remaining.remove(best_addition)
                total_tokens += best_addition['token_count']
                total_mi += best_marginal_mi
            else:
                break  # 没有更多有益的添加

        return {
            'selected_components': [comp['component'] for comp in selected],
            'total_mutual_information': total_mi,
            'token_utilization': total_tokens / self.token_budget,
            'selection_metadata': {
                'num_selected': len(selected),
                'efficiency_score': total_mi / (total_tokens + 1),
                'coverage_score': len(selected) / len(knowledge_base)
            }
        }

# 示例使用和演示
def demonstrate_information_theory():
    """演示信息论在上下文工程中的应用"""

    # 示例组件和查询
    query = "机器学习如何改善商业决策？"

    candidate_components = [
        "机器学习算法可以分析大型数据集以识别人类可能错过的模式和趋势，从而实现更多数据驱动的商业决策。",
        "使用机器学习的预测分析可以预测市场趋势、客户行为和运营需求，使企业能够做出主动决策。",
        "自动化决策系统可以比人类更快地处理信息，实现对变化的商业条件的实时响应。",
        "机器学习可以通过依赖客观数据分析而不是主观判断来减少决策中的人类偏见。",
        "今天天气晴朗，最高温度为75度，非常适合户外活动和海滩游览。",
        "机器学习模型需要仔细验证和测试，以确保它们为商业决策过程提供可靠的见解。",
        "将机器学习与现有商业智能工具集成可以增强整个组织的决策能力。"
    ]

    # 初始化分析器
    analyzer = InformationAnalyzer()
    optimizer = InformationOptimizer()
    mi_maximizer = MutualInformationMaximizer(token_budget=150)

    print("=== 信息论演示 ===")
    print(f"查询：{query}")
    print(f"候选组件：{len(candidate_components)}")

    # 分析每个组件
    print("\n=== 组件分析 ===")
    for i, component in enumerate(candidate_components):
        metrics = analyzer.analyze_component_information(component, query)
        print(f"\n组件 {i+1}:")
        print(f"  互信息：{metrics.mutual_information:.3f}")
        print(f"  熵：{metrics.entropy:.3f}")
        print(f"  效率比：{metrics.efficiency_ratio:.3f}")
        print(f"  冗余分数：{metrics.redundancy_score:.3f}")

    # 优化组件选择
    print("\n=== 优化结果 ===")
    selected_components = optimizer.optimize_component_selection(
        candidate_components, query, token_budget=150
    )

    print(f"选择了 {len(selected_components)} 个组件：")
    for i, component in enumerate(selected_components):
        print(f"  {i+1}. {component[:80]}...")

    # 最大化互信息
    mi_results = mi_maximizer.maximize_mutual_information(candidate_components, query)

    print(f"\n互信息优化：")
    print(f"  总MI：{mi_results['total_mutual_information']:.3f}")
    print(f"  Token利用率：{mi_results['token_utilization']:.1%}")
    print(f"  效率分数：{mi_results['selection_metadata']['efficiency_score']:.3f}")

    return {
        'selected_components': selected_components,
        'mi_results': mi_results,
        'component_analysis': [
            analyzer.analyze_component_information(comp, query)
            for comp in candidate_components
        ]
    }

# 运行演示
if __name__ == "__main__":
    results = demonstrate_information_theory()
```

**从零开始的解释**：这个编程框架将信息论概念实现为可工作的算法。就像拥有能够精确测量信息内容的科学仪器一样，它量化每条信息对回答用户问题的贡献价值。

---

## Software 3.0 范式三：协议（自适应信息演化）

协议提供自我改进的信息系统，根据有效性反馈学习最优信息选择和组织策略。

### 自适应信息优化协议

```
/information.optimize.adaptive{
    intent="通过信息论学习持续改进信息选择和组织",

    input={
        information_landscape={
            available_knowledge=<综合知识来源>,
            query_context=<用户查询和意图分析>,
            information_constraints=<token预算质量要求>,
            user_preferences=<信息密度风格偏好>
        },

        information_theory_context={
            historical_mi_performance=<互信息优化结果>,
            entropy_patterns=<信息内容分布分析>,
            redundancy_detection_history=<过去冗余识别成功>,
            compression_effectiveness=<信息压缩质量指标>
        },

        adaptation_parameters={
            information_learning_rate=<信息策略适应速度>,
            exploration_vs_exploitation=<平衡新与已证明的信息源>,
            quality_vs_efficiency_preference=<完整性与简洁性之间的权衡>,
            user_feedback_sensitivity=<对用户信息偏好的响应性>
        }
    },

    process=[
        /analyze.information.landscape{
            action="使用信息论原理系统化分析可用信息",
            method="多维信息分析，包含熵和互信息评估",
            analysis_dimensions=[
                {entropy_assessment="计算信息内容和不确定性降低潜力"},
                {mutual_information_calculation="测量信息源之间的相关性和重叠"},
                {redundancy_detection="识别重复或高度相似的信息内容"},
                {information_efficiency_evaluation="评估每个token或处理成本的信息价值"}
            ],
            pattern_recognition=[
                {high_value_information_characteristics="识别最有效信息中的模式"},
                {redundancy_sources="识别信息重复的常见来源"},
                {information_gaps="检测会增加互信息的缺失信息"},
                {optimal_information_density="学习细节与简洁性的理想平衡"}
            ],
            output="包含优化机会的综合信息景观分析"
        },

        /optimize.information.selection{
            action="选择最优信息子集以最大化与查询的互信息",
            method="带冗余惩罚和前瞻启发式的贪心优化",
            selection_algorithms=[
                {greedy_mutual_information="选择具有最高I(component; query)的组件"},
                {redundancy_penalized_selection="对I(component_i; component_j)应用惩罚"},
                {marginal_information_gain="选择具有最高边际信息的组件"},
                {diversity_maximization="确保信息覆盖查询的不同方面"}
            ],
            optimization_strategies=[
                {token_budget_optimization="在约束内最大化每个token的信息"},
                {quality_threshold_maintenance="确保最低信息质量标准"},
                {user_preference_integration="根据用户偏好加权信息类型"},
                {dynamic_threshold_adjustment="根据可用信息调整选择标准"}
            ],
            output="具有最大互信息的最优选择信息组件"
        },

        /compress.information.intelligently{
            action="应用信息论压缩以最大化信息密度",
            method="保持熵的压缩，维护语义连贯性",
            compression_techniques=[
                {entropy_based_sentence_selection="保留信息内容最高的句子"},
                {redundancy_elimination="删除重复或高度重叠的信息"},
                {semantic_compression="在保留含义的同时使用更紧凑的表示"},
                {hierarchical_information_organization="组织信息以获得最大清晰度"}
            ],
            quality_preservation=[
                {semantic_similarity_maintenance="确保压缩内容保留原始含义"},
                {mutual_information_preservation="通过压缩保持与查询的相关性"},
                {readability_optimization="保持压缩内容易于理解"},
                {critical_information_protection="永远不要压缩基本事实或关键见解"}
            ],
            output="保留语义价值的信息密集压缩内容"
        },

        /validate.information.quality{
            action="使用多个信息论度量评估信息质量",
            method="综合质量评估，整合用户反馈",
            quality_dimensions=[
                {relevance_assessment="测量I(selected_information; query)"},
                {completeness_evaluation="评估查询方面的覆盖范围"},
                {efficiency_measurement="计算使用的每个token的信息价值"},
                {coherence_analysis="评估信息的逻辑流和一致性"}
            ],
            validation_metrics=[
                {mutual_information_achievement="比较实现的与理论最大MI"},
                {redundancy_minimization="验证成功消除重复内容"},
                {user_satisfaction_correlation="跟踪MI分数与用户反馈的关系"},
                {compression_fidelity="测量通过压缩的信息保留"}
            ],
            output="包含改进建议的综合信息质量评估"
        },

        /learn.information.patterns{
            action="从信息优化经验中提取模式和见解",
            method="关于信息选择和组织有效性的元学习",
            learning_mechanisms=[
                {information_type_effectiveness="学习哪些类型的信息对不同查询最有效"},
                {compression_strategy_optimization="识别最有效的压缩技术"},
                {redundancy_pattern_recognition="理解信息重复的常见来源"},
                {user_preference_modeling="构建用户信息偏好和需求的模型"}
            ],
            knowledge_integration=[
                {selection_strategy_refinement="改进信息选择算法"},
                {compression_algorithm_tuning="优化压缩技术以获得更好的结果"},
                {mutual_information_prediction="构建模型来预测信息价值"},
                {adaptive_threshold_learning="学习最优质量和选择阈值"}
            ],
            output="使用改进策略更新的信息优化知识"
        }
    ],

    output={
        optimized_information={
            selected_components=<最大化互信息的信息组件>,
            information_organization=<信息呈现的最优结构>,
            compression_results=<智能压缩的高密度信息>,
            quality_metrics=<信息论质量测量>
        },

        optimization_insights={
            mutual_information_achieved=<完成的总I_context_query>,
            redundancy_eliminated=<删除的重复信息量>,
            compression_efficiency=<信息密度改进比>,
            selection_effectiveness=<信息组件选择的质量>
        },

        learning_outcomes={
            information_strategy_improvements=<选择算法的增强>,
            pattern_discoveries=<关于有效信息组织的新见解>,
            user_preference_updates=<对用户信息需求的精炼理解>,
            predictive_model_improvements=<信息价值预测的更好模型>
        }
    },

    meta={
        information_optimization_approach=<使用的具体算法和技术>,
        learning_integration_level=<实现的自适应改进程度>,
        theoretical_grounding=<与信息论原理的联系>,
        practical_effectiveness=<真实世界性能和用户满意度>
    },

    // 信息优化改进的自演化机制
    information_evolution=[
        {trigger="实现的互信息低",
         action="尝试替代信息选择策略"},
        {trigger="选择后检测到高冗余",
         action="改进冗余检测和消除算法"},
        {trigger="用户反馈表明信息缺口",
         action="增强完整性评估和缺口检测"},
        {trigger="压缩导致信息丢失",
         action="改进压缩技术以更好地保留"}
    ]
}
```

**从零开始的解释**：这个协议创建了一个信息优化系统，通过学习过去有效的经验，不断学习如何更有效地选择和组织信息，就像一个图书管理员通过学习哪些资源有效而越来越擅长找到完全正确的资源。

---

## 研究联系和未来方向

### 与上下文工程调查的联系

这个信息论模块直接实现和扩展了[上下文工程调查](https://arxiv.org/pdf/2507.13334)的基础概念：

**信息论上下文优化（§4.1 & §4.2）**：
- 通过互信息最大化实现上下文生成的系统化方法
- 通过基于熵的组件选择扩展动态组装概念
- 通过数学冗余检测解决信息冗余挑战

**上下文处理和管理（§4.2 & §4.3）**：
- 通过信息论压缩策略解决上下文压缩
- 通过熵和互信息指标解决上下文质量评估
- 基于信息价值量化实现智能上下文过滤

**基础研究应用（§7.1）**：
- 演示上下文优化的信息论基础
- 通过信息组件分析实现组合理解
- 为上下文质量测量和优化提供数学基础

### 超越当前研究的新贡献

**上下文工程的数学信息框架**：虽然调查涵盖了上下文技术，但我们对上下文组件选择系统应用香农信息论（熵、互信息、条件熵）代表了对上下文质量测量的严格信息论基础的新研究。

**冗余感知优化**：我们通过互信息计算整合冗余检测和消除，超越了当前方法，提供了识别和删除重复信息同时保留独特价值的数学框架。

**保留语义的信息压缩**：开发在最大化信息密度的同时保持语义连贯性的压缩技术，代表了超越简单token减少的进步，转向智能信息提炼。

**自适应信息学习**：我们的自我改进信息选择系统通过经验学习最优信息模式，代表了元信息优化的前沿研究。

### 未来研究方向

**量子信息论应用**：探索量子信息概念，如量子熵和量子互信息用于上下文工程，可能实现更复杂的信息关系和信息相关性的叠加状态。

**多模态信息集成**：研究文本、视觉、音频和时间信息的统一信息论框架，开发跨不同模态测量互信息的数学方法。

**因果信息论**：使用有向信息和传递熵研究信息组件之间的因果关系，使上下文系统不仅理解相关性，还理解信息流中的因果关系。

**信息论上下文安全**：开发信息论在上下文隐私和安全中的应用，使用差分隐私和信息论安全等概念来保护敏感信息，同时保持上下文效用。

**时间信息动态**：研究时间依赖的信息论，其中信息价值、熵和互信息随时间演化，需要用于时间上下文优化的动态数学框架。

**分布式信息优化**：研究信息论在分布式上下文工程中的应用，其中信息组件分布在多个系统中，同时保持全局信息优化。

**元信息论**：研究关于信息的信息 - 开发用于推理信息选择策略本身的信息内容的数学框架。

**人类信息论集成**：开发考虑人类认知处理、注意力限制和信息理解模式的信息论模型，用于上下文优化。

---

## 实践练习和项目

### 练习1：互信息计算器
**目标**：实现文本组件的互信息计算

```python
# 你的实现模板
class MutualInformationCalculator:
    def __init__(self):
        # TODO: 初始化文本处理组件
        pass

    def calculate_mi(self, text1: str, text2: str) -> float:
        # TODO: 实现互信息计算
        # 考虑单词级和字符级方法
        pass

    def calculate_conditional_entropy(self, text_y: str, text_x: str) -> float:
        # TODO: 计算 H(Y|X) = H(Y) - I(X;Y)
        pass

# 测试你的实现
calculator = MutualInformationCalculator()
# 在这里添加测试用例
```

### 练习2：基于信息论的组件选择器
**目标**：构建使用信息论选择最优组件的系统

```python
class InformationBasedSelector:
    def __init__(self, token_budget: int):
        self.token_budget = token_budget

    def select_components(self, candidates: List[str],
                         query: str) -> List[str]:
        # TODO: 实现最大化互信息的贪心选择
        # TODO: 包含冗余惩罚
        # TODO: 遵守token预算约束
        pass

    def calculate_selection_quality(self, selected: List[str],
                                   query: str) -> Dict[str, float]:
        # TODO: 返回综合质量指标
        pass

# 测试你的选择器
selector = InformationBasedSelector(token_budget=200)
```

### 练习3：自适应信息压缩
**目标**：创建保留最大信息的压缩系统

```python
class InformationPreservingCompressor:
    def __init__(self):
        # TODO: 初始化压缩算法
        pass

    def compress_with_entropy_preservation(self, text: str,
                                         compression_ratio: float) -> str:
        # TODO: 在保留最高熵内容的同时压缩
        pass

    def measure_compression_quality(self, original: str,
                                   compressed: str) -> Dict[str, float]:
        # TODO: 计算信息保留指标
        pass

# 测试压缩系统
compressor = InformationPreservingCompressor()
```

---

## 总结和下一步

### 掌握的关键概念

**信息论基础**：
- 香农熵：H(X) = -Σ P(x) × log₂(P(x))
- 互信息：I(X;Y) = H(X) + H(Y) - H(X,Y)
- 条件熵和信息增益计算
- 冗余检测和消除策略

**三范式集成**：
- **提示词**：信息评估和优化的战略模板
- **编程**：信息论计算的计算算法
- **协议**：学习最优信息选择模式的自适应系统

**高级信息应用**：
- 基于互信息最大化的组件选择
- 保留语义和信息内容的智能压缩
- 具有数学精确性的冗余消除
- 使用多个指标的信息质量评估

### 实现的实践掌握

你现在可以：
1. **量化信息价值**，使用数学信息论
2. **优化组件选择**，以最大化与查询的互信息
3. **消除冗余**，同时保留独特的信息内容
4. **智能压缩信息**，保持语义连贯性
5. **构建自适应系统**，学习最优信息模式

### 与课程进度的联系

这个信息论基础使能够：
- **贝叶斯推理**（模块04）：关于信息不确定性的概率推理
- **高级上下文系统**：实际应用中的信息论优化
- **研究应用**：为信息论上下文工程研究做贡献

你在这里掌握的信息测量中的数学精确性为关于包含什么信息、排除什么信息以及如何最有效地组织信息的最优决策提供了定量基础。

**下一个模块**：[04_bayesian_inference.md](04_bayesian_inference.md) - 我们将学习推理上下文选择中的不确定性，并根据概率反馈调整上下文策略。

---

## 快速参考：信息论公式

| 概念 | 公式 | 应用 |
|---------|---------|-------------|
| **熵** | H(X) = -Σ P(x)log₂(P(x)) | 测量信息内容 |
| **互信息** | I(X;Y) = H(X) + H(Y) - H(X,Y) | 测量相关性/重叠 |
| **条件熵** | H(Y\|X) = H(Y) - I(X;Y) | 剩余不确定性 |
| **信息增益** | IG = H(before) - H(after) | 额外信息的价值 |
| **冗余** | R = I(X;Y) / min(H(X),H(Y)) | 重复信息 |

这种信息论掌握将上下文工程从直觉相关性评估转变为基于信息科学基本原理的数学精确信息优化。
