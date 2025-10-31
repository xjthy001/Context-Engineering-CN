# Context-Engineering – 结构概览
_下一代 LLM 编排的实用、第一性原理手册_

> **本仓库存在的原因**
> 提示工程 = 思考**你说什么**。
> **上下文工程** = 思考模型看到的**其他一切**。
> 我们的目标是从基础开始教授"其他一切"，保持谦逊，偏向简单、可运行的代码。
>
> 随着模型演变，我们的方法也在演变：从离散 token 到连续场，从静态提示到共振模式。

---

## 1. 领域地图

| 文件夹 | 角色（通俗说法） | 第一性原理隐喻 |
|--------|----------------------|---------------------------|
| `00_foundations` | 理论与直觉。小型、自包含的读物。 | 原子 → 分子 → 细胞 → 器官 → 神经系统 → 场 |
| `10_guides_zero_to_hero` | 可以**运行**、调整、破坏的互动指南。 | 化学实验套装 |
| `20_templates` | 可复制/粘贴的即插即用代码片段。 | 乐高积木 |
| `30_examples` | 端到端的迷你应用，每个都比上一个更难。 | 模式生物 |
| `40_reference` | 深入探讨和评估指南。 | 教科书附录 |
| `50_contrib` | 社区拉取请求空间。 | 开放实验台 |
| `60_protocols` | 协议壳、模式和框架。 | DNA 序列 |
| `70_agents` | 使用协议的自包含代理演示。 | 干细胞培养 |
| `80_field_integration` | 使用场协议的端到端项目。 | 完整生物体 |
| `cognitive-tools` | 高级认知框架和架构。 | 扩展神经系统 |

---

## 2. 学习路径（0 → Zero → Hero）

### 基础（理解基础知识）

1. **浏览 `README.md`（2 分钟）**
   了解"上下文"在提示之外的含义。

2. **阅读 `00_foundations/01_atoms_prompting.md`（5 分钟）**
   *原子*：单个指令/示例。
   为什么仅有原子通常表现不佳。

3. **继续学习生物隐喻链：**
   - `02_molecules_context.md`：少样本组合
   - `03_cells_memory.md`：内存和日志
   - `04_organs_applications.md`：多步骤控制流
   - `05_cognitive_tools.md`：心智模型扩展
   - `06_advanced_applications.md`：实际实现
   - `07_prompt_programming.md`：类代码推理模式
   - `08_neural_fields_foundations.md`：作为连续场的上下文
   - `09_persistence_and_resonance.md`：场动力学和吸引子
   - `10_field_orchestration.md`：协调多个场

### 实践操作（边做边学）

4. **打开 `10_guides_zero_to_hero/01_min_prompt.ipynb`**
   运行、修改、观察 token 计数。
   笔记本单元格突出显示每行额外内容**为什么**有帮助（或有害）。

5. **通过渐进式笔记本进行实验：**
   - 基本上下文操作
   - 控制流和推理模式
   - 检索增强策略
   - 提示编程技术
   - 模式设计原则
   - 递归上下文模式
   - 神经场实现

### 应用技能（构建真实解决方案）

6. **从 `20_templates/` 复制模板**
   用作项目的起点：
   - `minimal_context.yaml` 用于基本项目
   - `control_loop.py` 用于交互式系统
   - `scoring_functions.py` 用于评估
   - `prompt_program_template.py` 用于推理任务
   - `schema_template.yaml` 用于结构化数据
   - `recursive_framework.py` 用于自我改进系统
   - `neural_field_context.yaml` 用于基于场的方法

7. **研究 `30_examples/` 中的示例**
   查看逐渐复杂系统的完整实现：
   - 基本对话代理
   - 数据标注系统
   - 多代理编排
   - 认知助手
   - RAG 实现
   - 神经场编排器

### 高级主题（掌握技艺）

8. **探索认知工具和协议：**
   - `cognitive-tools/` 中的高级推理框架
   - `60_protocols/` 中的协议壳和模式
   - `70_agents/` 中的代理演示
   - `80_field_integration/` 中的完整场集成项目

9. **回馈社区：**
   - 查看 `50_contrib/README.md` 中的贡献指南
   - 检查 `40_reference/eval_checklist.md` 中的评估标准
   - 通过您的改进或扩展开启 PR

---

## 3. 生物隐喻演变

我们的仓库围绕一个扩展的生物隐喻组织，这有助于使抽象概念具体化，并展示简单组件如何构建成复杂系统：

```
                                   ┌───────────────────┐
                                   │  神经场           │  08_neural_fields_foundations.md
                                   │  (连续            │  09_persistence_and_resonance.md
                                   │   上下文介质)     │  10_field_orchestration.md
                                   └───────┬───────────┘
                                           │
                                           ▲
                                           │
                                   ┌───────┴───────────┐
                                   │ 神经生物学        │  05_cognitive_tools.md
                                   │ 系统              │  06_advanced_applications.md
                                   │ (认知工具)        │  07_prompt_programming.md
                                   └───────┬───────────┘
                                           │
                                           ▲
                                           │
                             ┌─────────────┴─────────────┐
                             │         器官              │  04_organs_applications.md
                             │  (多代理系统)             │
                             └─────────────┬─────────────┘
                                           │
                                           ▲
                                           │
                             ┌─────────────┴─────────────┐
                             │         细胞              │  03_cells_memory.md
                             │   (记忆系统)              │
                             └─────────────┬─────────────┘
                                           │
                                           ▲
                                           │
                             ┌─────────────┴─────────────┐
                             │       分子                │  02_molecules_context.md
                             │   (少样本示例)            │
                             └─────────────┬─────────────┘
                                           │
                                           ▲
                                           │
                             ┌─────────────┴─────────────┐
                             │         原子              │  01_atoms_prompting.md
                             │    (单个提示)             │
                             └───────────────────────────┘
```

这种演变遵循生物系统中复杂性的自然进展，并反映了日益复杂的上下文工程方法的发展。

---

## 4. 高级上下文框架

### 协议壳框架

协议提供结构化壳，用于编排复杂的上下文操作。在 `60_protocols/` 目录中找到：

```
/recursive.field{
    intent="定义场属性和操作",
    input={
        field_state=<当前状态>,
        new_information=<传入数据>
    },
    process=[
        /field.measure{resonance, coherence, entropy},
        /pattern.detect{across="field_state"},
        /attractor.form{where="pattern_strength > threshold"},
        /field.evolve{with="new_information"}
    ],
    output={
        updated_field=<新状态>,
        metrics={resonance_score, coherence_delta}
    }
}
```

这些协议壳实现了：
- 上下文操作的声明式定义
- 递归自我改进模式
- 基于场的上下文操作
- 通过显式过程步骤的可审计性

### 认知工具框架

认知工具提供可重用的推理模式，扩展模型能力。在 `cognitive-tools/` 目录中找到：

```
cognitive-tools/
├── cognitive-templates/     # 不同推理模式的模式模板
├── cognitive-programs/      # 具有类代码模式的结构化提示程序
├── cognitive-schemas/       # 知识表示格式
├── cognitive-architectures/ # 完整推理系统
└── integration/            # 与其他组件集成的指南
```

该框架支持：
- 模块化推理组件
- 领域特定推理模式
- 与检索和记忆系统集成
- 推理质量的评估指标

### 神经场框架

神经场将上下文表示为连续介质，而不是离散 token。跨以下实现：

```
00_foundations/08_neural_fields_foundations.md  # 概念基础
00_foundations/09_persistence_and_resonance.md  # 场动力学
00_foundations/10_field_orchestration.md        # 多场协调
20_templates/neural_field_context.yaml          # 实现模板
30_examples/05_neural_field_orchestrator/       # 完整示例
```

关键概念包括：
- 作为连续语义场的上下文
- 通过共振的信息持久性
- 吸引子形成和动力学
- 用于复杂任务的场编排

---

## 5. Quiet Karpathy 指南（风格 DNA）

*保持原子化 → 构建起来。*
1. **最小优先** – 从最小可行上下文开始。
2. **迭代添加** – 仅添加模型明显缺少的内容。
3. **测量一切** – token 成本、延迟、质量分数、场共振。
4. **无情删除** – 修剪胜过填充。
5. **代码 > 幻灯片** – 每个概念都有可运行的单元格。
6. **递归思考** – 自我演化的上下文。

---

## 6. 仓库结构详细信息

```
Context-Engineering/
├── LICENSE                          # MIT 许可证
├── README.md                        # 快速入门概览
├── structure.md                     # 此结构地图
├── context.json                     # 原始模式配置
├── context_v2.json                  # 带场协议的扩展模式
│
├── 00_foundations/                  # 第一性原理理论
│   ├── 01_atoms_prompting.md        # 原子指令单元
│   ├── 02_molecules_context.md      # 少样本示例/上下文
│   ├── 03_cells_memory.md           # 有状态对话层
│   ├── 04_organs_applications.md    # 多步骤控制流
│   ├── 05_cognitive_tools.md        # 心智模型扩展
│   ├── 06_advanced_applications.md  # 实际实现
│   ├── 07_prompt_programming.md     # 类代码推理模式
│   ├── 08_neural_fields_foundations.md # 作为连续场的上下文
│   ├── 09_persistence_and_resonance.md # 场动力学和吸引子
│   └── 10_field_orchestration.md    # 协调多个场
│
├── 10_guides_zero_to_hero/          # 实践教程
│   ├── 01_min_prompt.ipynb          # 最小提示实验
│   ├── 02_expand_context.ipynb      # 上下文扩展技术
│   ├── 03_control_loops.ipynb       # 流控制机制
│   ├── 04_rag_recipes.ipynb         # 检索增强模式
│   ├── 05_prompt_programs.ipynb     # 结构化推理程序
│   ├── 06_schema_design.ipynb       # 模式创建模式
│   ├── 07_recursive_patterns.ipynb  # 自引用上下文
│   └── 08_neural_fields.ipynb       # 使用基于场的上下文
│
├── 20_templates/                    # 可重用组件
│   ├── minimal_context.yaml         # 基础上下文结构
│   ├── control_loop.py              # 编排模板
│   ├── scoring_functions.py         # 评估指标
│   ├── prompt_program_template.py   # 程序结构模板
│   ├── schema_template.yaml         # 模式定义模板
│   ├── recursive_framework.py       # 递归上下文模板
│   ├── neural_field_context.yaml    # 基于场的上下文模板
│   ├── field_resonance_measure.py   # 场属性测量
│   └── context_audit.py             # 上下文分析工具
│
├── 30_examples/                     # 实际实现
│   ├── 00_toy_chatbot/              # 简单对话代理
│   ├── 01_data_annotator/           # 数据标注系统
│   ├── 02_multi_agent_orchestrator/ # 代理协作系统
│   ├── 03_cognitive_assistant/      # 高级推理助手
│   ├── 04_rag_minimal/              # 最小 RAG 实现
│   └── 05_neural_field_orchestrator/ # 基于场的编排
│
├── 40_reference/                    # 深入文档
│   ├── token_budgeting.md           # Token 优化策略
│   ├── retrieval_indexing.md        # 检索系统设计
│   ├── eval_checklist.md            # PR 评估标准
│   ├── cognitive_patterns.md        # 推理模式目录
│   ├── schema_cookbook.md           # 模式模式集合
│   ├── neural_field_theory.md       # 综合场理论
│   ├── symbolic_residue_guide.md    # 残留追踪指南
│   └── protocol_reference.md        # 协议壳参考
│
├── 50_contrib/                      # 社区贡献
│   └── README.md                    # 贡献指南
│
├── 60_protocols/                    # 协议壳和框架
│   ├── README.md                    # 协议概览
│   ├── shells/                      # 协议壳定义
│   │   ├── attractor.co.emerge.shell      # 吸引子共同涌现
│   │   ├── recursive.emergence.shell      # 递归场涌现
│   │   ├── recursive.memory.attractor.shell # 记忆持久性
│   │   └── field.resonance.scaffold.shell  # 场共振
│   ├── digests/                     # 简化协议文档
│   └── schemas/                     # 协议模式
│       ├── fractalRepoContext.v1.json     # 仓库上下文
│       ├── fractalConsciousnessField.v1.json # 场模式
│       └── protocolShell.v1.json           # 壳模式
│
├── 70_agents/                       # 代理演示
│   ├── README.md                    # 代理概览
│   ├── 01_residue_scanner/          # 符号残留检测
│   └── 02_self_repair_loop/         # 自修复协议
│
├── 80_field_integration/            # 完整场项目
│   ├── README.md                    # 集成概览
│   ├── 00_protocol_ide_helper/      # 协议开发工具
│   └── 01_context_engineering_assistant/ # 基于场的助手
│
├── cognitive-tools/                 # 高级认知框架
│   ├── README.md                    # 概览和快速入门指南
│   ├── cognitive-templates/         # 推理模板
│   │   ├── understanding.md         # 理解操作
│   │   ├── reasoning.md             # 分析操作
│   │   ├── verification.md          # 检查和验证
│   │   └── composition.md           # 组合多个工具
│   │
│   ├── cognitive-programs/          # 结构化提示程序
│   │   ├── basic-programs.md        # 基础程序结构
│   │   ├── advanced-programs.md     # 复杂程序架构
│   │   ├── program-library.py       # Python 实现
│   │   └── program-examples.ipynb   # 交互式示例
│   │
│   ├── cognitive-schemas/           # 知识表示
│   │   ├── user-schemas.md          # 用户信息模式
│   │   ├── domain-schemas.md        # 领域知识模式
│   │   ├── task-schemas.md          # 推理任务模式
│   │   └── schema-library.yaml      # 可重用模式库
│   │
│   ├── cognitive-architectures/     # 完整推理系统
│   │   ├── solver-architecture.md   # 问题解决系统
│   │   ├── tutor-architecture.md    # 教育系统
│   │   ├── research-architecture.md # 信息综合
│   │   └── architecture-examples.py # 实现示例
│   │
│   └── integration/                 # 集成模式
│       ├── with-rag.md              # 与检索集成
│       ├── with-memory.md           # 与记忆集成
│       ├── with-agents.md           # 与代理集成
│       └── evaluation-metrics.md    # 有效性测量
│
└── .github/                         # GitHub 配置
    ├── CONTRIBUTING.md              # 贡献指南
    ├── workflows/ci.yml             # CI 管道配置
    ├── workflows/eval.yml           # 评估自动化
    └── workflows/protocol_tests.yml # 协议测试
```

---

## 7. 如何贡献

在 `50_contrib/` 中开启 PR。
检查清单位于 `40_reference/eval_checklist.md` — 提交前运行它。

贡献时：
1. 遵循 Karpathy 风格指南
2. 包含可运行的代码示例
3. 测量 token 使用和性能
4. 保持生物隐喻一致性
5. 为任何新功能添加测试

---

## 8. 许可证和归属

MIT。无门槛：复制、混搭、重新分发。
尊敬地向 Andrej Karpathy 致意，感谢他创造了这个框架。
所有错误都是我们的；欢迎改进。
