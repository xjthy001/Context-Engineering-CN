# 结构化上下文处理
## 用于上下文工程的图与关系数据集成

> **模块 02.4** | *上下文工程课程:从基础到前沿系统*
>
> 基于 [上下文工程综述](https://arxiv.org/pdf/2507.13334) | 推进知识图谱增强的上下文系统

---

## 学习目标

完成本模块后,你将理解并实现:

- **基于图的上下文表示**:将复杂关系建模为连接的知识结构
- **关系推理系统**:理解实体和关系如何创造意义
- **知识图谱集成**:将结构化知识融入上下文组装
- **层次信息组织**:管理嵌套和递归数据结构以实现最优上下文

---

## 概念演进:从线性文本到网络智能

将结构化上下文处理想象成阅读字典(线性的、按字母顺序排列)与理解一个生态系统(网络化的、关联的、相互依存的)之间的区别。

### 阶段1:线性信息处理
```
文本:"Alice 在 Google 工作。Google 是一家科技公司。科技公司开发软件。"

处理: Alice → 在工作 → Google → 是一个 → 科技公司 → 开发 → 软件

理解: 顺序的,连接有限
```
**上下文**:就像从教科书中逐个阅读事实。你获得了信息,但错过了创造更深层次理解的丰富关系网络。

**局限性**:
- 信息被孤立处理
- 关系未被明确建模
- 难以推理连接
- 没有层次结构组织

### 阶段2:简单实体-关系识别
```
实体: [Alice, Google, 科技公司, 软件]
关系: [在工作(Alice, Google), 是一个(Google, 科技公司), 开发(科技公司, 软件)]

基本图:
Alice --在工作--> Google --是一个--> 科技公司 --开发--> 软件
```
**上下文**:就像创建一个简单的组织结构图或家谱。你可以看到直接连接,但复杂模式仍然隐藏。

**改进**:
- 明确识别实体和关系
- 出现基本图结构
- 可以回答简单的关系查询

**仍存在的问题**:
- 扁平的关系结构
- 没有推理或推断
- 有限的上下文传播

### 阶段3:知识图谱集成
```
丰富的知识图谱:

    Alice (人)
      ├─ 在工作 → Google (公司)
      ├─ 技能 → [编程, AI]
      └─ 位置 → 山景城

    Google (公司)
      ├─ 是一个 → 科技公司
      ├─ 成立于 → 1998
      ├─ 总部 → 山景城
      ├─ 开发 → [搜索, Android, AI]
      ├─ 员工数 → 150000
      └─ 竞争对手 → [Apple, Microsoft]

    科技公司 (类别)
      ├─ 特征 → [创新, 软件, 数字化]
      └─ 例子 → [Google, Apple, Microsoft]
```
**上下文**:就像能够访问维基百科的整个知识网络。支持复杂推理和推断的丰富互联信息。

**能力**:
- 跨关系的多跳推理
- 层次分类和继承
- 通过图遍历进行上下文丰富
- 支持复杂查询和推断

### 阶段4:动态层次上下文网络
```
┌─────────────────────────────────────────────────────────────────┐
│                    层次上下文网络                                 │
│                                                                 │
│  领域层:科技行业                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  公司层: Google                                          │   │
│  │  ├─ 商业模式:广告、云、硬件                               │   │
│  │  ├─ 核心技术: AI、搜索、移动                              │   │
│  │  └─ 市场定位:搜索领域领导者,在AI领域增长                  │   │
│  │                                                         │   │
│  │    个人层: Alice                                         │   │
│  │    ├─ 角色上下文: AI研究员                               │   │
│  │    ├─ 技能上下文:机器学习、Python                         │   │
│  │    └─ 项目上下文:大语言模型                               │   │
│  │                                                         │   │
│  │      任务层:当前任务                                      │   │
│  │      ├─ 目标:改进模型安全性                              │   │
│  │      ├─ 方法: Constitutional AI, RLHF                   │   │
│  │      └─ 时间线: 2024年第三-第四季度                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  跨层连接:                                                       │
│  • 行业趋势影响公司战略                                          │
│  • 公司资源支持个人项目                                          │
│  • 个人专业知识塑造项目方法                                      │
│  • 项目成果影响公司定位                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
**上下文**:就像拥有一位战略大师,能同时理解个人行动如何与团队动态、组织目标和行业趋势相联系。

### 阶段5:具有涌现结构发现的自适应图智能
```
┌─────────────────────────────────────────────────────────────────┐
│                   自适应图智能系统                                │
│                                                                 │
│  自组织知识网络:                                                 │
│                                                                 │
│  模式识别引擎:                                                   │
│    • 发现数据中的隐式关系                                        │
│    • 识别重复出现的结构模式                                      │
│    • 学习最优图组织策略                                          │
│                                                                 │
│  涌现结构形成:                                                   │
│    • 创建原始数据中不存在的新关系类型                             │
│    • 在关系模式之间形成元关系                                     │
│    • 自动开发层次抽象                                            │
│                                                                 │
│  动态上下文适应:                                                 │
│    • 基于查询模式重构图                                          │
│    • 优化不同推理类型的信息路径                                   │
│    • 基于使用和反馈演进表示                                      │
│                                                                 │
│  实时推理和推断:                                                 │
│    • 跨复杂关系链的多跳推理                                      │
│    • 相似图模式之间的类比推理                                     │
│    • 从结构关系进行因果推断                                      │
│    • 关于关系演化的时间推理                                      │
│                                                                 │
│  自我改进机制:                                                   │
│    • 学习更好的图构建策略                                        │
│    • 改进关系提取和分类                                          │
│    • 基于结果增强推理算法                                        │
│    • 优化计算效率的结构                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
**上下文**:就像拥有一位AI科学家,不仅理解现有的知识网络,还能发现新模式、创建新颖的组织结构,并持续改进自己的理解和推理能力。

---

## 数学基础

### 基于图的上下文表示
```
知识图谱: G = (E, R, T)
其中:
- E = 实体集合 {e₁, e₂, ..., eₙ}
- R = 关系类型集合 {r₁, r₂, ..., rₖ}
- T = 表示事实的三元组集合 {(eᵢ, rⱼ, eₖ)}

从图中组装上下文:
C(q, G) = TraversePath(q, G, depth=d, strategy=s)

其中:
- q = 查询或信息需求
- G = 知识图谱
- d = 最大遍历深度
- s = 遍历策略 (BFS, DFS, 相关性引导)
```
**直观解释**:知识图谱就像一张信息地图,其中实体是位置,关系是它们之间的路径。上下文组装变成一个导航问题——通过知识网络找到从查询到答案的最相关路径。

### 数学基础

### 层次信息编码
```
层次上下文树: H = (N, P, C)
其中:
- N = 表示信息单元的节点集合
- P = 父子关系(分类结构)
- C = 交叉链接(关联关系)

信息传播:
I(n) = Local(n) + α·∑ᵢ Parent(i)·w(i→n) + β·∑ⱼ Child(j)·w(n→j) + γ·∑ₖ CrossLink(k)·w(n↔k)

其中:
- Local(n) = 直接在节点n的信息
- α, β, γ = 不同关系类型的传播权重
- w(·) = 关系强度权重
```
**直观解释**:层次结构中的信息不仅存在于单个节点——它在各层之间流动。一个概念从其父节点(它所属的类别)、子节点(具体实例)和交叉链接(相关概念)继承意义。就像你对"狗"的理解来自"动物"(父节点)、"金毛猎犬"(子节点)和"伴侣"(交叉链接)。

### 关系推理优化
```
多跳路径推理:
P(答案 | 查询, 图) = ∑ paths π P(答案 | π) · P(π | 查询, 图)

其中路径 π = (e₀, r₁, e₁, r₂, e₂, ..., rₙ, eₙ)

路径概率:
P(π | 查询, 图) = ∏ᵢ P(rᵢ₊₁ | eᵢ, 查询) · P(eᵢ₊₁ | eᵢ, rᵢ₊₁, 查询)

优化遍历:
π* = argmax_π P(π | 查询, 图) 受约束于 |π| ≤ max_hops
```
**直观解释**:在知识图谱中推理时,从问题到答案有许多可能的路径。我们想找到将查询连接到相关信息的最可能路径,同时考虑每个关系的可能性和整体路径的连贯性。

---

## Software 3.0 范式1:提示(结构化推理模板)

### 知识图谱推理模板

```markdown
# 知识图谱推理框架

## 图上下文分析
你正在通过表示为知识图谱的结构化信息进行推理。使用系统化的遍历和关系分析来建立全面理解。

## 图结构评估
**可用实体**: {当前图中的实体}
**关系类型**: {关系类型及其含义}
**图深度**: {最大关系链长度}
**查询上下文**: {特定问题或推理目标}

### 实体分析
对于推理路径中的每个相关实体:

**实体**: {实体名称}
- **类型/类别**: {实体分类}
- **直接属性**: {直接与实体相关的属性}
- **出向关系**: {实体作为主语的关系}
- **入向关系**: {实体作为宾语的关系}
- **层次上下文**: {分类法中的父实体和子实体}

### 关系链构建

#### 单跳推理
**直接连接**: {实体1} --{关系}--> {实体2}
- **关系强度**: {关系的置信度或权重}
- **上下文相关性**: {与当前查询的相关程度}
- **信息内容**: {这个关系告诉我们什么}

#### 多跳推理路径
**路径1**: {实体1} --{关系1}--> {实体2} --{关系2}--> {实体3} --{关系3}--> {目标}
- **路径连贯性**: {这个链条的逻辑一致性如何}
- **累积证据**: {沿路径的证据强度}
- **替代解释**: {理解这条路径的其他方式}

**路径2**: {替代推理路径}
**路径3**: {如果相关,额外的推理路径}

### 推理策略选择

#### 自底向上推理(从具体到一般)
```
IF 查询需要泛化:
    从 具体实例 开始
    识别 共同模式和属性
    通过 层次关系 向上遍历
    综合 一般原则或类别
```

#### 自顶向下推理(从一般到具体)
```
IF 查询需要具体信息:
    从 一般类别或原则 开始
    通过 专门化关系 向下遍历
    识别 相关的具体实例
    提取 关于实例的详细信息
```

#### 横向推理(跨同一层级)
```
IF 查询需要比较或类比:
    识别 相似层次级别的实体
    遍历 交叉链接和关联关系
    比较 属性和关系模式
    识别 相似性和差异性
```

### 层次上下文集成

#### 局部上下文(直接邻域)
- **直接属性**: {焦点实体的属性}
- **直接关系**: {一跳关系}
- **局部约束**: {直接上下文中的规则或约束}

#### 中间上下文(2-3跳)
- **扩展关系**: {多跳连接}
- **模式识别**: {扩展邻域中的重复结构}
- **上下文修饰符**: {中间上下文如何影响解释}

#### 全局上下文(完整图视角)
- **领域级模式**: {大规模结构和模式}
- **跨领域连接**: {跨越不同知识领域的关系}
- **系统级约束**: {全局规则或原则}

### 推理执行

#### 演绎推理
**给定事实**: {图中的显式关系和属性}
**逻辑规则**: {可以应用的如果-那么规则}
**结论**: {可以逻辑推导出什么}

示例:
```
IF Alice 在Google工作 AND Google 是一个科技公司
THEN Alice 在一个科技公司工作(雇佣和分类的传递性)
```

#### 归纳推理
**观察到的模式**: {图中重复出现的结构或关系}
**泛化规则**: {可能更广泛适用的模式}
**置信水平**: {我们对这些泛化有多确定}

#### 溯因推理(最佳解释)
**观察到的证据**: {需要解释的事实}
**候选解释**: {观察证据的可能原因}
**最佳解释**: {给定图结构最可能的解释}

### 上下文组装策略

#### 查询驱动组装
1. **解析查询**:识别提到的关键实体和关系
2. **种子选择**:在图中选择起始点
3. **扩展策略**:决定如何从种子扩展上下文
4. **相关性过滤**:保留最相关的信息,修剪无关的
5. **连贯性验证**:确保组装的上下文形成连贯叙述

#### 结构驱动组装
1. **识别关键结构**:找到重要的子图或模式
2. **提取层次**:建立分类和部分-整体关系
3. **映射交叉链接**:包含重要的关联关系
4. **上下文分层**:按抽象层次组织信息
5. **集成综合**:结合不同的结构视图

### 质量评估

#### 完整性检查
- **必需信息覆盖率**: {包含的必要信息百分比}
- **关键关系覆盖率**: {表示的重要关系}
- **层次完整性**: {跨不同抽象层次的覆盖}

#### 连贯性验证
- **逻辑一致性**: {组装上下文中没有矛盾}
- **关系有效性**: {所有关系都是有意义和正确的}
- **叙述流畅性**: {信息从前提到结论逻辑流动}

#### 相关性优化
- **查询对齐**: {上下文如何很好地解决原始查询}
- **信息密度**: {有用信息与总信息的比率}
- **焦点适当性**: {查询类型的正确详细程度}

## 结构化上下文输出

**主要推理路径**: {最有信心的推理链}
**支持证据**: {支持结论的额外关系}
**替代解释**: {理解信息的其他可能方式}
**不确定性因素**: {推理置信度较低的领域}

**层次总结**:
- **高层概念**: {一般类别和原则}
- **中层关系**: {具体连接和模式}
- **详细事实**: {具体属性和实例}

**交叉引用**: {提供额外上下文的相关信息}
```

**从零开始的解释**:这个模板就像侦探通过一个互联线索网络调查案件。侦探不只是查看单个证据,而是绘制它们如何连接的地图,建立从线索到线索的推理链,并在得出结论前考虑多种可能的解释。

---

## Software 3.0 范式2:编程(结构化上下文实现)

### 知识图谱上下文引擎

```python
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import json

class RelationType(Enum):
    """Types of relationships in knowledge graph"""
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    INSTANCE_OF = "instance_of"
    HAS_PROPERTY = "has_property"
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    CAUSES = "causes"
    ENABLES = "enables"
    SIMILAR_TO = "similar_to"

@dataclass
class Entity:
    """Knowledge graph entity with properties"""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    confidence: float = 1.0

@dataclass
class Relationship:
    """Knowledge graph relationship"""
    subject: str
    predicate: RelationType
    object: str
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningPath:
    """Path through knowledge graph for reasoning"""
    entities: List[str]
    relationships: List[Relationship]
    path_score: float
    reasoning_type: str
    evidence_strength: float

class KnowledgeGraph:
    """Core knowledge graph representation and operations"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.graph = nx.MultiDiGraph()
        self.entity_types: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[RelationType, List[Relationship]] = defaultdict(list)

    def add_entity(self, entity: Entity):
        """Add entity to knowledge graph"""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **entity.properties)
        self.entity_types[entity.entity_type].add(entity.id)

    def add_relationship(self, relationship: Relationship):
        """Add relationship to knowledge graph"""
        self.relationships.append(relationship)
        self.graph.add_edge(
            relationship.subject,
            relationship.object,
            predicate=relationship.predicate,
            weight=relationship.weight,
            confidence=relationship.confidence
        )
        self.relation_index[relationship.predicate].append(relationship)

    def get_neighbors(self, entity_id: str, relation_type: Optional[RelationType] = None,
                     direction: str = "outgoing") -> List[Tuple[str, Relationship]]:
        """Get neighboring entities connected by specific relationship type"""
        neighbors = []

        if direction in ["outgoing", "both"]:
            for target in self.graph.successors(entity_id):
                edges = self.graph[entity_id][target]
                for edge_data in edges.values():
                    if relation_type is None or edge_data['predicate'] == relation_type:
                        rel = Relationship(
                            subject=entity_id,
                            predicate=edge_data['predicate'],
                            object=target,
                            weight=edge_data['weight'],
                            confidence=edge_data['confidence']
                        )
                        neighbors.append((target, rel))

        if direction in ["incoming", "both"]:
            for source in self.graph.predecessors(entity_id):
                edges = self.graph[source][entity_id]
                for edge_data in edges.values():
                    if relation_type is None or edge_data['predicate'] == relation_type:
                        rel = Relationship(
                            subject=source,
                            predicate=edge_data['predicate'],
                            object=entity_id,
                            weight=edge_data['weight'],
                            confidence=edge_data['confidence']
                        )
                        neighbors.append((source, rel))

        return neighbors

    def find_paths(self, start_entity: str, end_entity: str,
                   max_depth: int = 3) -> List[ReasoningPath]:
        """Find reasoning paths between two entities"""
        paths = []

        try:
            # Find all simple paths up to max_depth
            nx_paths = nx.all_simple_paths(self.graph, start_entity, end_entity, cutoff=max_depth)

            for path in nx_paths:
                reasoning_path = self._convert_to_reasoning_path(path)
                if reasoning_path:
                    paths.append(reasoning_path)

        except nx.NetworkXNoPath:
            pass  # No path exists

        # Sort by path score
        paths.sort(key=lambda p: p.path_score, reverse=True)
        return paths[:10]  # Return top 10 paths

    def _convert_to_reasoning_path(self, node_path: List[str]) -> Optional[ReasoningPath]:
        """Convert networkx path to reasoning path"""
        if len(node_path) < 2:
            return None

        relationships = []
        path_score = 1.0

        for i in range(len(node_path) - 1):
            source, target = node_path[i], node_path[i + 1]

            # Find the relationship between these nodes
            edges = self.graph[source][target]
            if not edges:
                return None

            # Take the edge with highest confidence
            best_edge = max(edges.values(), key=lambda e: e['confidence'])

            rel = Relationship(
                subject=source,
                predicate=best_edge['predicate'],
                object=target,
                weight=best_edge['weight'],
                confidence=best_edge['confidence']
            )
            relationships.append(rel)

            # Update path score based on relationship confidence
            path_score *= rel.confidence

        return ReasoningPath(
            entities=node_path,
            relationships=relationships,
            path_score=path_score,
            reasoning_type="multi_hop",
            evidence_strength=path_score
        )

    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get rich context for an entity including neighbors at specified depth"""
        if entity_id not in self.entities:
            return {}

        context = {
            'entity': self.entities[entity_id],
            'immediate_neighbors': {},
            'extended_context': {},
            'hierarchical_context': {}
        }

        # Get immediate neighbors (depth 1)
        immediate = self.get_neighbors(entity_id, direction="both")
        context['immediate_neighbors'] = {
            'outgoing': [(target, rel) for target, rel in immediate if rel.subject == entity_id],
            'incoming': [(source, rel) for source, rel in immediate if rel.object == entity_id]
        }

        # Get extended context (depth 2+)
        if depth > 1:
            extended_entities = set()
            queue = deque([(entity_id, 0)])
            visited = {entity_id}

            while queue:
                current_entity, current_depth = queue.popleft()

                if current_depth >= depth:
                    continue

                neighbors = self.get_neighbors(current_entity, direction="both")
                for neighbor_id, rel in neighbors:
                    if neighbor_id not in visited:
                        extended_entities.add(neighbor_id)
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, current_depth + 1))

            context['extended_context'] = {
                eid: self.entities[eid] for eid in extended_entities if eid in self.entities
            }

        # Get hierarchical context (is_a relationships)
        hierarchical = self._get_hierarchical_context(entity_id)
        context['hierarchical_context'] = hierarchical

        return context

    def _get_hierarchical_context(self, entity_id: str) -> Dict[str, List[str]]:
        """Get hierarchical context (parents and children in taxonomy)"""
        parents = []
        children = []

        # Find parents (things this entity is_a instance of)
        parent_rels = self.get_neighbors(entity_id, RelationType.IS_A, "outgoing")
        parents.extend([target for target, _ in parent_rels])

        instance_rels = self.get_neighbors(entity_id, RelationType.INSTANCE_OF, "outgoing")
        parents.extend([target for target, _ in instance_rels])

        # Find children (things that are instances of this entity)
        child_rels = self.get_neighbors(entity_id, RelationType.IS_A, "incoming")
        children.extend([source for source, _ in child_rels])

        instance_child_rels = self.get_neighbors(entity_id, RelationType.INSTANCE_OF, "incoming")
        children.extend([source for source, _ in instance_child_rels])

        return {
            'parents': parents,
            'children': children
        }

class StructuredContextAssembler:
    """Assembles context from structured knowledge representations"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.reasoning_strategies = {
            'deductive': self._deductive_reasoning,
            'inductive': self._inductive_reasoning,
            'abductive': self._abductive_reasoning,
            'analogical': self._analogical_reasoning
        }

    def assemble_context(self, query: str, entities: List[str],
                        max_context_size: int = 2000,
                        reasoning_strategy: str = "deductive") -> Dict[str, Any]:
        """Main context assembly process"""

        print(f"Assembling structured context for query: {query}")
        print(f"Starting entities: {entities}")

        # Extract key information from query
        query_analysis = self._analyze_query(query)

        # Collect relevant subgraphs around seed entities
        relevant_subgraphs = []
        for entity_id in entities:
            if entity_id in self.kg.entities:
                subgraph = self._extract_relevant_subgraph(entity_id, query_analysis, depth=3)
                relevant_subgraphs.append(subgraph)

        # Apply reasoning strategy
        reasoning_results = self.reasoning_strategies[reasoning_strategy](
            query_analysis, relevant_subgraphs
        )

        # Assemble final context
        assembled_context = self._integrate_reasoning_results(
            query, query_analysis, reasoning_results, max_context_size
        )

        return assembled_context

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand information needs"""
        query_lower = query.lower()

        analysis = {
            'query_text': query,
            'query_type': 'factual',  # Default
            'entities_mentioned': [],
            'relationships_implied': [],
            'reasoning_depth': 'shallow',
            'answer_type': 'descriptive'
        }

        # Determine query type
        if any(word in query_lower for word in ['why', 'because', 'cause', 'reason']):
            analysis['query_type'] = 'causal'
            analysis['reasoning_depth'] = 'deep'
        elif any(word in query_lower for word in ['how', 'process', 'method', 'way']):
            analysis['query_type'] = 'procedural'
        elif any(word in query_lower for word in ['compare', 'difference', 'similar', 'versus']):
            analysis['query_type'] = 'comparative'
            analysis['reasoning_depth'] = 'medium'
        elif any(word in query_lower for word in ['what is', 'define', 'definition']):
            analysis['query_type'] = 'definitional'

        # Extract mentioned entities (simplified)
        for entity_id, entity in self.kg.entities.items():
            if entity.name.lower() in query_lower:
                analysis['entities_mentioned'].append(entity_id)

        # Infer required relationships
        if analysis['query_type'] == 'causal':
            analysis['relationships_implied'].append(RelationType.CAUSES)
        elif analysis['query_type'] == 'comparative':
            analysis['relationships_implied'].append(RelationType.SIMILAR_TO)

        return analysis

    def _extract_relevant_subgraph(self, start_entity: str, query_analysis: Dict,
                                 depth: int = 3) -> Dict[str, Any]:
        """Extract relevant subgraph around an entity"""

        # Start with entity context
        entity_context = self.kg.get_entity_context(start_entity, depth=depth)

        # Score relevance of different parts
        relevance_scores = self._score_context_relevance(entity_context, query_analysis)

        # Filter based on relevance
        filtered_context = self._filter_by_relevance(entity_context, relevance_scores, threshold=0.3)

        return {
            'root_entity': start_entity,
            'context': filtered_context,
            'relevance_scores': relevance_scores,
            'subgraph_summary': self._summarize_subgraph(filtered_context)
        }

    def _score_context_relevance(self, context: Dict, query_analysis: Dict) -> Dict[str, float]:
        """Score relevance of different context elements to query"""
        scores = {}

        # Score immediate neighbors
        for direction in ['outgoing', 'incoming']:
            for target_id, rel in context['immediate_neighbors'][direction]:
                score = 0.5  # Base score

                # Boost score if relationship type is implied by query
                if rel.predicate in query_analysis['relationships_implied']:
                    score += 0.3

                # Boost score if target entity is mentioned in query
                if target_id in query_analysis['entities_mentioned']:
                    score += 0.4

                scores[f"{direction}_{target_id}"] = score

        # Score extended context entities
        for entity_id, entity in context['extended_context'].items():
            score = 0.3  # Lower base score for extended context

            if entity_id in query_analysis['entities_mentioned']:
                score += 0.4

            # Boost based on entity type relevance
            if entity.entity_type in query_analysis.get('relevant_types', []):
                score += 0.2

            scores[f"extended_{entity_id}"] = score

        # Score hierarchical context
        for parent_id in context['hierarchical_context']['parents']:
            scores[f"parent_{parent_id}"] = 0.4

        for child_id in context['hierarchical_context']['children']:
            scores[f"child_{child_id}"] = 0.3

        return scores

    def _filter_by_relevance(self, context: Dict, relevance_scores: Dict,
                           threshold: float) -> Dict[str, Any]:
        """Filter context based on relevance scores"""
        filtered = {
            'entity': context['entity'],
            'immediate_neighbors': {'outgoing': [], 'incoming': []},
            'extended_context': {},
            'hierarchical_context': {'parents': [], 'children': []}
        }

        # Filter immediate neighbors
        for direction in ['outgoing', 'incoming']:
            for target_id, rel in context['immediate_neighbors'][direction]:
                score_key = f"{direction}_{target_id}"
                if relevance_scores.get(score_key, 0) >= threshold:
                    filtered['immediate_neighbors'][direction].append((target_id, rel))

        # Filter extended context
        for entity_id, entity in context['extended_context'].items():
            score_key = f"extended_{entity_id}"
            if relevance_scores.get(score_key, 0) >= threshold:
                filtered['extended_context'][entity_id] = entity

        # Filter hierarchical context
        for parent_id in context['hierarchical_context']['parents']:
            if relevance_scores.get(f"parent_{parent_id}", 0) >= threshold:
                filtered['hierarchical_context']['parents'].append(parent_id)

        for child_id in context['hierarchical_context']['children']:
            if relevance_scores.get(f"child_{child_id}", 0) >= threshold:
                filtered['hierarchical_context']['children'].append(child_id)

        return filtered

    def _summarize_subgraph(self, context: Dict) -> str:
        """Create summary of subgraph structure"""
        entity = context['entity']

        summary_parts = [f"Entity: {entity.name} ({entity.entity_type})"]

        # Count connections
        outgoing_count = len(context['immediate_neighbors']['outgoing'])
        incoming_count = len(context['immediate_neighbors']['incoming'])
        extended_count = len(context['extended_context'])

        summary_parts.append(f"Direct connections: {outgoing_count + incoming_count}")
        summary_parts.append(f"Extended network: {extended_count} entities")

        # Hierarchical position
        parent_count = len(context['hierarchical_context']['parents'])
        child_count = len(context['hierarchical_context']['children'])

        if parent_count > 0 or child_count > 0:
            summary_parts.append(f"Hierarchical: {parent_count} parents, {child_count} children")

        return "; ".join(summary_parts)

    def _deductive_reasoning(self, query_analysis: Dict, subgraphs: List[Dict]) -> Dict[str, Any]:
        """Apply deductive reasoning to extract logical conclusions"""

        reasoning_chains = []

        for subgraph in subgraphs:
            context = subgraph['context']
            root_entity = subgraph['root_entity']

            # Find logical inference chains
            chains = self._find_inference_chains(context, query_analysis)
            reasoning_chains.extend(chains)

        # Rank reasoning chains by strength
        reasoning_chains.sort(key=lambda c: c['confidence'], reverse=True)

        return {
            'reasoning_type': 'deductive',
            'chains': reasoning_chains[:5],  # Top 5 chains
            'conclusions': [chain['conclusion'] for chain in reasoning_chains[:3]],
            'confidence': np.mean([chain['confidence'] for chain in reasoning_chains[:3]]) if reasoning_chains else 0
        }

    def _find_inference_chains(self, context: Dict, query_analysis: Dict) -> List[Dict]:
        """Find logical inference chains in context"""
        chains = []

        # Simple transitivity chains
        entity = context['entity']

        # For each outgoing relationship, see if we can chain it
        for target_id, rel1 in context['immediate_neighbors']['outgoing']:
            if target_id in context['extended_context']:
                # Look for relationships from this target
                target_context = self.kg.get_entity_context(target_id, depth=1)

                for final_target, rel2 in target_context['immediate_neighbors']['outgoing']:
                    # Check if this creates a meaningful chain
                    if self._is_valid_inference_chain(rel1, rel2):
                        chains.append({
                            'premises': [f"{entity.name} {rel1.predicate.value} {target_id}",
                                       f"{target_id} {rel2.predicate.value} {final_target}"],
                            'conclusion': f"{entity.name} (transitively) {rel2.predicate.value} {final_target}",
                            'confidence': rel1.confidence * rel2.confidence,
                            'chain_length': 2
                        })

        return chains

    def _is_valid_inference_chain(self, rel1: Relationship, rel2: Relationship) -> bool:
        """Check if two relationships can form valid inference chain"""
        # Valid transitivity patterns
        valid_patterns = [
            (RelationType.IS_A, RelationType.IS_A),
            (RelationType.PART_OF, RelationType.PART_OF),
            (RelationType.LOCATED_IN, RelationType.LOCATED_IN),
            (RelationType.WORKS_AT, RelationType.LOCATED_IN)
        ]

        return (rel1.predicate, rel2.predicate) in valid_patterns

    def _inductive_reasoning(self, query_analysis: Dict, subgraphs: List[Dict]) -> Dict[str, Any]:
        """Apply inductive reasoning to identify patterns"""

        patterns = []

        # Look for recurring relationship patterns across subgraphs
        for subgraph in subgraphs:
            context = subgraph['context']
            local_patterns = self._identify_local_patterns(context)
            patterns.extend(local_patterns)

        # Generalize patterns
        generalized_patterns = self._generalize_patterns(patterns)

        return {
            'reasoning_type': 'inductive',
            'patterns': generalized_patterns,
            'generalizations': [p['generalization'] for p in generalized_patterns],
            'confidence': np.mean([p['support'] for p in generalized_patterns]) if generalized_patterns else 0
        }

    def _identify_local_patterns(self, context: Dict) -> List[Dict]:
        """Identify patterns in local context"""
        patterns = []

        # Pattern: entities of same type often have similar relationships
        entity_type = context['entity'].entity_type

        for target_id, rel in context['immediate_neighbors']['outgoing']:
            if target_id in context['extended_context']:
                target_entity = context['extended_context'][target_id]
                patterns.append({
                    'pattern_type': 'entity_type_relationship',
                    'entity_type': entity_type,
                    'relationship': rel.predicate,
                    'target_type': target_entity.entity_type,
                    'instance': f"{entity_type} entities often have {rel.predicate.value} relationships with {target_entity.entity_type} entities"
                })

        return patterns

    def _generalize_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Generalize patterns across multiple instances"""
        pattern_counts = defaultdict(list)

        # Group similar patterns
        for pattern in patterns:
            if pattern['pattern_type'] == 'entity_type_relationship':
                key = (pattern['entity_type'], pattern['relationship'], pattern['target_type'])
                pattern_counts[key].append(pattern)

        # Create generalizations
        generalizations = []
        for key, instances in pattern_counts.items():
            if len(instances) >= 2:  # Need at least 2 instances to generalize
                entity_type, relationship, target_type = key
                generalizations.append({
                    'generalization': f"{entity_type} entities typically have {relationship.value} relationships with {target_type} entities",
                    'support': len(instances) / len(patterns),
                    'instances': len(instances),
                    'confidence': min(1.0, len(instances) / 5)  # More instances = higher confidence
                })

        return generalizations

    def _abductive_reasoning(self, query_analysis: Dict, subgraphs: List[Dict]) -> Dict[str, Any]:
        """Apply abductive reasoning to find best explanations"""

        # Look for phenomena that need explanation
        phenomena = self._identify_phenomena(query_analysis, subgraphs)

        # Generate candidate explanations
        explanations = []
        for phenomenon in phenomena:
            candidates = self._generate_explanations(phenomenon, subgraphs)
            explanations.extend(candidates)

        # Rank explanations by plausibility
        explanations.sort(key=lambda e: e['plausibility'], reverse=True)

        return {
            'reasoning_type': 'abductive',
            'phenomena': phenomena,
            'explanations': explanations[:3],  # Top 3 explanations
            'best_explanation': explanations[0] if explanations else None,
            'confidence': explanations[0]['plausibility'] if explanations else 0
        }

    def _identify_phenomena(self, query_analysis: Dict, subgraphs: List[Dict]) -> List[Dict]:
        """Identify phenomena that need explanation"""
        phenomena = []

        # Look for unusual patterns or relationships
        for subgraph in subgraphs:
            context = subgraph['context']

            # Phenomenon: entity has unusually many relationships of one type
            outgoing_rels = context['immediate_neighbors']['outgoing']
            rel_counts = defaultdict(int)
            for _, rel in outgoing_rels:
                rel_counts[rel.predicate] += 1

            for rel_type, count in rel_counts.items():
                if count > 3:  # Arbitrary threshold
                    phenomena.append({
                        'type': 'high_relationship_count',
                        'entity': context['entity'].name,
                        'relationship_type': rel_type,
                        'count': count,
                        'description': f"{context['entity'].name} has {count} {rel_type.value} relationships"
                    })

        return phenomena

    def _generate_explanations(self, phenomenon: Dict, subgraphs: List[Dict]) -> List[Dict]:
        """Generate candidate explanations for a phenomenon"""
        explanations = []

        if phenomenon['type'] == 'high_relationship_count':
            entity_name = phenomenon['entity']
            rel_type = phenomenon['relationship_type']
            count = phenomenon['count']

            # Find the entity in subgraphs
            entity_context = None
            for subgraph in subgraphs:
                if subgraph['context']['entity'].name == entity_name:
                    entity_context = subgraph['context']
                    break

            if entity_context:
                entity_type = entity_context['entity'].entity_type

                # Generate explanations based on entity type
                if entity_type == 'Company' and rel_type == RelationType.HAS_PROPERTY:
                    explanations.append({
                        'explanation': f"{entity_name} is a large company with many diverse attributes",
                        'plausibility': 0.8,
                        'evidence': f"Companies typically have many properties; {count} is reasonable for a major company"
                    })

                if entity_type == 'Person' and rel_type == RelationType.WORKS_AT:
                    explanations.append({
                        'explanation': f"{entity_name} may have had multiple jobs or consulting roles",
                        'plausibility': 0.6,
                        'evidence': f"People can work at multiple organizations throughout their career"
                    })

        return explanations

    def _analogical_reasoning(self, query_analysis: Dict, subgraphs: List[Dict]) -> Dict[str, Any]:
        """Apply analogical reasoning to find similar patterns"""

        analogies = []

        # Compare subgraphs to find structural similarities
        for i, subgraph1 in enumerate(subgraphs):
            for j, subgraph2 in enumerate(subgraphs[i+1:], i+1):
                analogy = self._find_structural_analogy(subgraph1, subgraph2)
                if analogy:
                    analogies.append(analogy)

        return {
            'reasoning_type': 'analogical',
            'analogies': analogies,
            'insights': [a['insight'] for a in analogies],
            'confidence': np.mean([a['similarity'] for a in analogies]) if analogies else 0
        }

    def _find_structural_analogy(self, subgraph1: Dict, subgraph2: Dict) -> Optional[Dict]:
        """Find structural analogy between two subgraphs"""
        context1 = subgraph1['context']
        context2 = subgraph2['context']

        entity1 = context1['entity']
        entity2 = context2['entity']

        # Skip if same entity
        if entity1.id == entity2.id:
            return None

        # Compare relationship patterns
        rels1 = [rel.predicate for _, rel in context1['immediate_neighbors']['outgoing']]
        rels2 = [rel.predicate for _, rel in context2['immediate_neighbors']['outgoing']]

        # Calculate similarity
        common_rels = set(rels1) & set(rels2)
        total_rels = set(rels1) | set(rels2)

        if total_rels:
            similarity = len(common_rels) / len(total_rels)

            if similarity > 0.5:  # Threshold for considering analogy
                return {
                    'entity1': entity1.name,
                    'entity2': entity2.name,
                    'similarity': similarity,
                    'common_patterns': list(common_rels),
                    'insight': f"{entity1.name} and {entity2.name} have similar relationship patterns, suggesting they may belong to the same category or serve similar roles"
                }

        return None

    def _integrate_reasoning_results(self, query: str, query_analysis: Dict,
                                   reasoning_results: Dict, max_size: int) -> Dict[str, Any]:
        """Integrate reasoning results into final context"""

        # Start with reasoning conclusions
        context_parts = []

        if reasoning_results['reasoning_type'] == 'deductive':
            context_parts.append("Deductive reasoning conclusions:")
            for conclusion in reasoning_results['conclusions']:
                context_parts.append(f"• {conclusion}")

        elif reasoning_results['reasoning_type'] == 'inductive':
            context_parts.append("Identified patterns:")
            for generalization in reasoning_results['generalizations']:
                context_parts.append(f"• {generalization}")

        elif reasoning_results['reasoning_type'] == 'abductive':
            if reasoning_results['best_explanation']:
                context_parts.append("Best explanation:")
                context_parts.append(f"• {reasoning_results['best_explanation']['explanation']}")

        elif reasoning_results['reasoning_type'] == 'analogical':
            context_parts.append("Analogical insights:")
            for insight in reasoning_results['insights']:
                context_parts.append(f"• {insight}")

        # Assemble final context
        integrated_context = "\n".join(context_parts)

        # Truncate if too long
        if len(integrated_context) > max_size:
            integrated_context = integrated_context[:max_size] + "..."

        return {
            'query': query,
            'reasoning_type': reasoning_results['reasoning_type'],
            'context': integrated_context,
            'confidence': reasoning_results.get('confidence', 0),
            'reasoning_details': reasoning_results,
            'query_analysis': query_analysis
        }

# Example usage and demonstration
def create_sample_knowledge_graph() -> KnowledgeGraph:
    """Create sample knowledge graph for demonstration"""
    kg = KnowledgeGraph()

    # Add entities
    entities = [
        Entity("alice", "Alice", "Person", {"age": 30, "location": "San Francisco"}),
        Entity("google", "Google", "Company", {"founded": 1998, "employees": 150000}),
        Entity("tech_company", "Technology Company", "Category", {"industry": "Technology"}),
        Entity("ai_researcher", "AI Researcher", "Role", {"field": "Artificial Intelligence"}),
        Entity("machine_learning", "Machine Learning", "Field", {"domain": "Computer Science"}),
        Entity("python", "Python", "Programming Language", {"type": "interpreted"}),
        Entity("san_francisco", "San Francisco", "City", {"state": "California"})
    ]

    for entity in entities:
        kg.add_entity(entity)

    # Add relationships
    relationships = [
        Relationship("alice", RelationType.WORKS_AT, "google", weight=1.0, confidence=0.95),
        Relationship("alice", RelationType.IS_A, "ai_researcher", weight=1.0, confidence=0.9),
        Relationship("alice", RelationType.LOCATED_IN, "san_francisco", weight=1.0, confidence=0.85),
        Relationship("google", RelationType.IS_A, "tech_company", weight=1.0, confidence=1.0),
        Relationship("google", RelationType.LOCATED_IN, "san_francisco", weight=1.0, confidence=1.0),
        Relationship("ai_researcher", RelationType.RELATED_TO, "machine_learning", weight=0.8, confidence=0.8),
        Relationship("machine_learning", RelationType.ENABLES, "python", weight=0.7, confidence=0.7),
        Relationship("tech_company", RelationType.HAS_PROPERTY, "machine_learning", weight=0.6, confidence=0.6)
    ]

    for rel in relationships:
        kg.add_relationship(rel)

    return kg

def demonstrate_structured_context():
    """Demonstrate structured context processing"""
    print("Structured Context Processing Demonstration")
    print("=" * 50)

    # Create knowledge graph
    kg = create_sample_knowledge_graph()

    print(f"Knowledge Graph created with {len(kg.entities)} entities and {len(kg.relationships)} relationships")

    # Create context assembler
    assembler = StructuredContextAssembler(kg)

    # Test queries
    test_queries = [
        ("What can you tell me about Alice?", ["alice"]),
        ("How is Google related to technology?", ["google", "tech_company"]),
        ("What is the connection between Alice and machine learning?", ["alice", "machine_learning"])
    ]

    for query, seed_entities in test_queries:
        print(f"\nQuery: {query}")
        print(f"Seed entities: {seed_entities}")
        print("-" * 40)

        # Test different reasoning strategies
        for strategy in ['deductive', 'inductive', 'abductive', 'analogical']:
            print(f"\n{strategy.upper()} REASONING:")

            result = assembler.assemble_context(query, seed_entities, reasoning_strategy=strategy)

            print(f"Context: {result['context']}")
            print(f"Confidence: {result['confidence']:.3f}")

            if result['reasoning_details']:
                details = result['reasoning_details']
                if strategy == 'deductive' and 'chains' in details:
                    print(f"Reasoning chains found: {len(details['chains'])}")
                elif strategy == 'inductive' and 'patterns' in details:
                    print(f"Patterns identified: {len(details['patterns'])}")
                elif strategy == 'abductive' and 'explanations' in details:
                    print(f"Explanations generated: {len(details['explanations'])}")
                elif strategy == 'analogical' and 'analogies' in details:
                    print(f"Analogies found: {len(details['analogies'])}")

    # Demonstrate graph traversal
    print(f"\n" + "=" * 50)
    print("GRAPH TRAVERSAL DEMONSTRATION")
    print("=" * 50)

    # Find paths between entities
    paths = kg.find_paths("alice", "machine_learning", max_depth=3)
    print(f"\nPaths from Alice to Machine Learning:")
    for i, path in enumerate(paths[:3]):
        print(f"Path {i+1}: {' -> '.join(path.entities)}")
        print(f"  Relationships: {[rel.predicate.value for rel in path.relationships]}")
        print(f"  Score: {path.path_score:.3f}")

    # Show entity context
    print(f"\nAlice's Context:")
    alice_context = kg.get_entity_context("alice", depth=2)
    print(f"Entity: {alice_context['entity'].name} ({alice_context['entity'].entity_type})")
    print(f"Immediate connections: {len(alice_context['immediate_neighbors']['outgoing']) + len(alice_context['immediate_neighbors']['incoming'])}")
    print(f"Extended network: {len(alice_context['extended_context'])} entities")
    print(f"Hierarchical: {len(alice_context['hierarchical_context']['parents'])} parents, {len(alice_context['hierarchical_context']['children'])} children")

    return kg, assembler

# Run demonstration
if __name__ == "__main__":
    kg, assembler = demonstrate_structured_context()
```

**从零开始的解释**:这个结构化上下文系统就像一位研究图书管理员,不仅知道信息存储在哪里,还理解不同知识片段如何相互连接。系统可以通过多个步骤追踪关系,识别不同领域的模式,并应用各种推理策略来提取数据中未明确说明的见解。

---

## 研究联系和未来方向

### 与上下文工程综述的联系

本结构化上下文模块直接实现并扩展了[上下文工程综述](https://arxiv.org/pdf/2507.13334)的关键概念:

**知识图谱集成(通篇引用)**:
- 实现了StructGPT和GraphFormers的结构化数据处理方法
- 将知识图谱集成概念扩展到全面的上下文组装
- 通过系统化的图推理解决结构化上下文挑战

**上下文处理创新(§4.2)**:
- 将上下文处理原则应用于图结构信息
- 将自我改进概念扩展到知识图谱优化
- 实现针对关系数据的结构化上下文方法

**新颖的研究贡献**:
- **多策略推理**:系统整合演绎、归纳、溯因和类比推理
- **层次上下文网络**:跨多个抽象层次动态组织信息
- **自适应图智能**:优化自身知识表示的自我改进系统

### 未来研究方向

**时序知识图谱**:将静态知识图谱扩展为捕获关系和实体如何随时间演化,实现时序推理和预测。

**概率图推理**:将不确定性和概率推断纳入知识图谱推理,以实现更稳健的上下文组装。

**多模态知识图谱**:将前一模块的多模态处理与结构化知识表示集成,以获得更丰富、更全面的上下文。

**涌现关系发现**:自动发现未明确编程的新关系类型和模式的系统,超越当前知识图谱的局限性。

---

## 总结和下一步

**掌握的核心概念**:
- 基于图的上下文表示和遍历算法
- 多策略推理系统(演绎、归纳、溯因、类比)
- 层次信息组织和传播
- 用于上下文组装的知识图谱集成

**Software 3.0 集成**:
- **提示**:用于系统化图遍历的结构化推理模板
- **编程**:具有多策略推理能力的知识图谱引擎
- **协议**:优化自身推理的自适应图智能系统

**实现技能**:
- 知识图谱构建和管理系统
- 多跳推理和路径查找算法
- 带有相关性过滤的结构化上下文组装
- 全面的推理策略实现

**研究基础**:直接实现知识图谱研究,并在多策略推理、层次上下文网络和自适应图智能系统方面进行创新扩展。

**下一模块**:长上下文处理实验室——通过交互式编程练习实践注意力机制、记忆系统和层次处理架构。

---

*本模块展示了从线性信息处理到网络智能的演进,体现了Software 3.0原则:系统不仅存储和检索信息,还理解和推理创造意义和产生洞察的复杂关系。*
