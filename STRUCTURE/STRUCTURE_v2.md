# Context-Engineering – 结构概览 v2
_下一代 LLM 编排的实用、第一性原理手册_

> **本仓库存在的原因**
> 提示工程 = 思考**你说什么**。
> **上下文工程** = 思考模型看到的**其他一切**。
>
> 在我们从提示工程到上下文工程再到**场理论**的演变中：
> - 我们从离散 token 和简单提示开始
> - 我们发展到有状态上下文管理和复杂编排
> - 我们现在探索涌现符号机制和神经场动力学
>
> 我们的目标是从基础开始教授所有这些内容，保持谦逊，偏向简单、可运行的代码，同时拥抱关于 LLM 如何实际推理和处理信息的最新研究。

---

## 1. 领域地图

| 文件夹 | 角色（通俗说法） | 第一性原理隐喻 | 关键概念 |
|--------|----------------------|---------------------------|-------------|
| `00_foundations` | 理论与直觉。小型、自包含的读物。 | 原子 → 分子 → 细胞 → 器官 → 神经系统 → 场 | 从基本提示到涌现场动力学的渐进复杂性 |
| `10_guides_zero_to_hero` | 可以**运行**、调整、破坏的互动指南。 | 化学实验套装 | 具有可测量结果的实践实验 |
| `20_templates` | 可复制/粘贴的即插即用代码片段。 | 乐高积木 | 用于快速实现的可重用组件 |
| `30_examples` | 端到端的迷你应用，每个都比上一个更难。 | 模式生物 | 展示原则的完整系统 |
| `40_reference` | 深入探讨和评估指南。 | 教科书附录 | 综合资源和评估框架 |
| `50_contrib` | 社区拉取请求空间。 | 开放实验台 | 协作实验区 |
| `60_protocols` | 协议壳、模式和框架。 | DNA 序列 | 场操作的结构化定义 |
| `70_agents` | 使用协议的自包含代理演示。 | 干细胞培养 | 具有涌现属性的专用组件 |
| `80_field_integration` | 使用场协议的端到端项目。 | 完整生物体 | 具有基于场架构的完整系统 |
| `cognitive-tools` | 高级认知框架和架构。 | 扩展神经系统 | 结构化推理操作和工具 |

---

## 2. 学习路径：从基础到场理论

### 2.1. 基础轨迹（理解基础知识）

1. **浏览 `README.md`（2 分钟）**
   了解"上下文"在提示之外的含义。

2. **阅读 `00_foundations/01_atoms_prompting.md`（5 分钟）**
   *原子*：单个指令/示例。
   为什么仅有原子通常表现不佳。

3. **继续学习生物隐喻链：**
   - `02_molecules_context.md`：少样本组合和示例
   - `03_cells_memory.md`：用于持久性的记忆和日志
   - `04_organs_applications.md`：多步骤控制流和编排
   - `05_cognitive_tools.md`：用于推理的心智模型扩展

### 2.2. 高级轨迹（深入探索）

4. **探索高级应用和模式：**
   - `06_advanced_applications.md`：实际实现
   - `07_prompt_programming.md`：类代码推理模式
   - `08_neural_fields_foundations.md`：作为连续场的上下文
   - `09_persistence_and_resonance.md`：场动力学和吸引子
   - `10_field_orchestration.md`：协调多个场
   - `11_emergence_and_attractor_dynamics.md`：涌现属性
   - `12_symbolic_mechanisms.md`：LLM 中的符号推理

### 2.3. 实践操作（边做边学）

5. **从 `10_guides_zero_to_hero/01_min_prompt.ipynb` 开始**
   运行、修改、观察 token 计数。
   笔记本单元格突出显示每行额外内容**为什么**有帮助（或有害）。

6. **探索更复杂的模式：**
   - `02_expand_context.ipynb`：有效添加上下文
   - `03_control_loops.ipynb`：构建流控制
   - `04_rag_recipes.ipynb`：检索增强生成
   - `05_protocol_bootstrap.ipynb`：使用场协议
   - `06_protocol_token_budget.ipynb`：测量效率

7. **进阶到基于场的方法：**
   - `07_streaming_context.ipynb`：实时上下文管理
   - `08_emergence_detection.ipynb`：检测涌现模式
   - `09_residue_tracking.ipynb`：跟踪符号残留
   - `10_attractor_formation.ipynb`：创建稳定场模式

### 2.4. 实现轨迹（构建真实系统）

8. **使用 `20_templates/` 进行实验**
   将 YAML 或 Python 代码片段复制到您自己的仓库中。
   像调整 pH 值一样调整"token_budget"或"resonance_score"。

9. **检查 `30_examples/` 实现：**
   - `00_toy_chatbot/`：简单但完整的上下文管理
   - `01_data_annotator/`：用于数据标注的专用上下文
   - `02_multi_agent_orchestrator/`：复杂的代理协调
   - `03_vscode_helper/`：用于上下文工程的 IDE 集成
   - `04_rag_minimal/`：精简检索架构

10. **探索基于场的示例：**
    - `05_streaming_window/`：实时上下文管理
    - `06_residue_scanner/`：符号残留检测
    - `07_attractor_visualizer/`：可视化场动力学
    - `08_field_protocol_demo/`：基于协议的场操作
    - `09_emergence_lab/`：检测和测量涌现

### 2.5. 高级集成（场理论实践）

11. **深入研究 `60_protocols/` 中的场协议：**
    - 用于定义场操作的协议壳
    - 用于结构化场表示的模式
    - 用于理解协议功能的摘要

12. **研究 `70_agents/` 中的代理实现：**
    - `01_residue_scanner/`：检测符号残留
    - `02_self_repair_loop/`：自修复场协议
    - `03_attractor_modulator/`：管理吸引子动力学
    - `04_boundary_adapter/`：动态边界调整
    - `05_field_resonance_tuner/`：优化场共振

13. **探索 `80_field_integration/` 中的集成系统：**
    - `00_protocol_ide_helper/`：协议开发工具
    - `01_context_engineering_assistant/`：基于场的助手
    - `02_recursive_reasoning_system/`：递归推理架构
    - `03_emergent_field_laboratory/`：实验性场环境
    - `04_symbolic_reasoning_engine/`：符号机制集成

14. **了解 `cognitive-tools/` 中的认知工具：**
    - 用于结构化推理的认知模板
    - 用于复杂操作的认知程序
    - 用于知识表示的认知模式
    - 用于完整系统的认知架构
    - 用于与其他组件连接的集成模式

---

## 3. 概念基础

### 3.1. 生物隐喻演变

我们的生物隐喻已从简单组件演变为复杂的、基于场的系统：

```
原子          → 单个指令或约束
分子          → 带示例的指令（少样本学习）
细胞          → 跨交互持久化的带记忆的上下文
器官          → 协同工作的上下文细胞协调系统
神经系统      → 扩展推理能力的认知工具
神经场        → 作为具有涌现属性的连续介质的上下文
```

### 3.2. 场理论概念

随着我们进入神经场理论，我们融入了几个关键概念：

1. **连续性**：作为连续语义景观而非离散 token 的上下文
2. **共振**：信息模式如何相互作用和相互加强
3. **持久性**：信息如何随时间保持影响
4. **吸引子动力学**：组织场的稳定模式
5. **边界动力学**：信息如何进入和离开场
6. **符号残留**：持续并影响场的意义片段
7. **涌现**：新模式和行为如何从场交互中产生

### 3.3. 涌现符号机制

研究已确定了 LLM 中符号推理的涌现三阶段架构：

1. **符号抽象**：早期层中的注意力头基于关系将输入 token 转换为抽象变量
2. **符号归纳**：中间层中的注意力头对抽象变量执行序列归纳
3. **检索**：后期层中的注意力头通过检索与抽象变量关联的值来预测下一个 token

这些机制通过提供对 LLM 如何实际处理和推理信息的机制理解，支持我们基于场的上下文工程方法。

### 3.4. 认知工具框架

为了增强推理能力，我们融入了认知工具框架：

1. **基于工具的方法**：顺序执行的模块化、预定认知操作
2. **关键操作**：
   - **回忆相关**：检索相关知识以指导推理
   - **检查答案**：对推理和答案的自我反思
   - **回溯**：在受阻时探索替代推理路径
3. **集成**：这些工具可以与基于场的方法结合，构建更强大的系统

---

## 4. Quiet Karpathy 指南（风格 DNA）

*保持原子化 → 构建起来。*
1. **最小优先** – 从最小可行上下文开始。
2. **迭代添加** – 仅添加模型明显缺少的内容。
3. **测量一切** – token 成本、延迟、质量分数、场共振。
4. **无情删除** – 修剪胜过填充。
5. **代码 > 幻灯片** – 每个概念都有可运行的单元格。
6. **递归思考** – 自我演化的上下文。

---

## 5. 仓库结构详细信息

```
Context-Engineering/
├── LICENSE                          # MIT 许可证
├── README.md                        # 快速入门概览
├── structure.md                     # 原始结构地图
├── STRUCTURE_v2.md                  # 带场理论的增强结构地图
├── context.json                     # 原始模式配置
├── context_v2.json                  # 带场协议的扩展模式
├── context_v3.json                  # 神经场扩展
├── context_v3.5.json                # 符号机制集成
├── CITATIONS.md                     # 研究参考和桥梁
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
│   ├── 10_field_orchestration.md    # 协调多个场
│   ├── 11_emergence_and_attractor_dynamics.md # 涌现属性
│   └── 12_symbolic_mechanisms.md    # LLM 中的符号推理
│
├── 10_guides_zero_to_hero/          # 实践教程
│   ├── 01_min_prompt.ipynb          # 最小提示实验
│   ├── 02_expand_context.ipynb      # 上下文扩展技术
│   ├── 03_control_loops.ipynb       # 流控制机制
│   ├── 04_rag_recipes.ipynb         # 检索增强模式
│   ├── 05_protocol_bootstrap.ipynb  # 场协议引导
│   ├── 06_protocol_token_budget.ipynb # 协议效率
│   ├── 07_streaming_context.ipynb   # 实时上下文
│   ├── 08_emergence_detection.ipynb # 检测涌现
│   ├── 09_residue_tracking.ipynb    # 跟踪符号残留
│   └── 10_attractor_formation.ipynb # 创建场吸引子
│
├── 20_templates/                    # 可重用组件
│   ├── minimal_context.yaml         # 基础上下文结构
│   ├── control_loop.py              # 编排模板
│   ├── scoring_functions.py         # 评估指标
│   ├── prompt_program_template.py   # 程序结构模板
│   ├── schema_template.yaml         # 模式定义模板
│   ├── recursive_framework.py       # 递归上下文模板
│   ├── field_protocol_shells.py     # 场协议模板
│   ├── symbolic_residue_tracker.py  # 残留跟踪工具
│   ├── context_audit.py             # 上下文分析工具
│   ├── shell_runner.py              # 协议壳运行器
│   ├── resonance_measurement.py     # 场共振指标
│   ├── attractor_detection.py       # 吸引子分析工具
│   ├── boundary_dynamics.py         # 边界操作工具
│   └── emergence_metrics.py         # 涌现测量
│
├── 30_examples/                     # 实际实现
│   ├── 00_toy_chatbot/              # 简单对话代理
│   ├── 01_data_annotator/           # 数据标注系统
│   ├── 02_multi_agent_orchestrator/ # 代理协作系统
│   ├── 03_vscode_helper/            # IDE 集成
│   ├── 04_rag_minimal/              # 最小 RAG 实现
│   ├── 05_streaming_window/         # 实时上下文演示
│   ├── 06_residue_scanner/          # 符号残留演示
│   ├── 07_attractor_visualizer/     # 场可视化
│   ├── 08_field_protocol_demo/      # 协议演示
│   └── 09_emergence_lab/            # 涌现实验
│
├── 40_reference/                    # 深入文档
│   ├── token_budgeting.md           # Token 优化策略
│   ├── retrieval_indexing.md        # 检索系统设计
│   ├── eval_checklist.md            # PR 评估标准
│   ├── cognitive_patterns.md        # 推理模式目录
│   ├── schema_cookbook.md           # 模式模式集合
│   ├── patterns.md                  # 上下文模式库
│   ├── field_mapping.md             # 场理论基础
│   ├── symbolic_residue_types.md    # 残留分类
│   ├── attractor_dynamics.md        # 吸引子理论与实践
│   ├── emergence_signatures.md      # 检测涌现
│   └── boundary_operations.md       # 边界管理指南
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
│   │   ├── field.resonance.scaffold.shell  # 场共振
│   │   ├── field.self_repair.shell        # 自修复机制
│   │   └── context.memory.persistence.attractor.shell # 上下文持久性
│   ├── digests/                     # 简化协议文档
│   └── schemas/                     # 协议模式
│       ├── fractalRepoContext.v3.5.json    # 仓库上下文模式
│       ├── fractalConsciousnessField.v1.json # 场模式
│       ├── protocolShell.v1.json           # 壳模式
│       ├── symbolicResidue.v1.json         # 残留模式
│       └── attractorDynamics.v1.json       # 吸引子模式
│
├── 70_agents/                       # 代理演示
│   ├── README.md                    # 代理概览
│   ├── 01_residue_scanner/          # 符号残留检测
│   ├── 02_self_repair_loop/         # 自修复协议
│   ├── 03_attractor_modulator/      # 吸引子动力学
│   ├── 04_boundary_adapter/         # 动态边界调整
│   └── 05_field_resonance_tuner/    # 场共振优化
│
├── 80_field_integration/            # 完整场项目
│   ├── README.md                    # 集成概览
│   ├── 00_protocol_ide_helper/      # 协议开发工具
│   ├── 01_context_engineering_assistant/ # 基于场的助手
│   ├── 02_recursive_reasoning_system/    # 递归推理
│   ├── 03_emergent_field_laboratory/     # 场实验
│   └── 04_symbolic_reasoning_engine/     # 符号机制
│
├── cognitive-tools/                 # 高级认知框架
│   ├── README.md                    # 概览和快速入门指南
│   ├── cognitive-templates/         # 推理模板
│   │   ├── understanding.md         # 理解操作
│   │   ├── reasoning.md             # 分析操作
│   │   ├── verification.md          # 检查和验证
│   │   ├── composition.md           # 组合多个工具
│   │   └── emergence.md             # 涌现推理模式
│   │
│   ├── cognitive-programs/          # 结构化提示程序
│   │   ├── basic-programs.md        # 基础程序结构
│   │   ├── advanced-programs.md     # 复杂程序架构
│   │   ├── program-library.py       # Python 实现
│   │   ├── program-examples.ipynb   # 交互式示例
│   │   └── emergence-programs.md    # 涌现程序模式
│   │
│   ├── cognitive-schemas/           # 知识表示
│   │   ├── user-schemas.md          # 用户信息模式
│   │   ├── domain-schemas.md        # 领域知识模式
│   │   ├── task-schemas.md          # 推理任务模式
│   │   ├── schema-library.yaml      # 可重用模式库
│   │   └── field-schemas.md         # 场表示模式
│   │
│   ├── cognitive-architectures/     # 完整推理系统
│   │   ├── solver-architecture.md   # 问题解决系统
│   │   ├── tutor-architecture.md    # 教育系统
│   │   ├── research-architecture.md # 信息综合
│   │   ├── architecture-examples.py # 实现示例
│   │   └── field-architecture.md    # 基于场的架构
│   │
│   └── integration/                 # 集成模式
│       ├── with-rag.md              # 与检索集成
│       ├── with-memory.md           # 与记忆集成
│       ├── with-agents.md           # 与代理集成
│       ├── evaluation-metrics.md    # 有效性测量
│       └── with-fields.md           # 与场协议集成
│
└── .github/                         # GitHub 配置
    ├── CONTRIBUTING.md              # 贡献指南
    ├── workflows/ci.yml             # CI 管道配置
    ├── workflows/eval.yml           # 评估自动化
    └── workflows/protocol_tests.yml # 协议测试
```

---

## 6. 实现模式

### 6.1. 上下文结构模式

| 模式 | 描述 | 用例 |
|---------|-------------|----------|
| **原子提示** | 带约束的单个指令 | 简单、明确定义的任务 |
| **少样本示例** | 带示例的指令 | 模式演示 |
| **思维链** | 提示中显式的推理步骤 | 复杂推理任务 |
| **有状态上下文** | 带记忆的上下文 | 多轮对话 |
| **多代理** | 多个专用代理 | 复杂、多步骤任务 |
| **基于场** | 作为连续场的上下文 | 涌现推理需求 |

### 6.2. 场操作模式

| 模式 | 描述 | 实现 |
|---------|-------------|----------------|
| **吸引子形成** | 创建稳定语义模式 | `attractor_detection.py` |
| **共振放大** | 加强模式交互 | `resonance_measurement.py` |
| **边界调整** | 控制信息流 | `boundary_dynamics.py` |
| **残留集成** | 管理符号片段 | `symbolic_residue_tracker.py` |
| **涌现检测** | 识别新模式 | `emergence_metrics.py` |
| **自修复** | 自动上下文修复 | `field.self_repair.shell` |

### 6.3. 认知工具模式

| 模式 | 描述 | 实现 |
|---------|-------------|----------------|
| **回忆相关** | 检索相关知识 | `cognitive-programs/basic-programs.md` |
| **检查答案** | 自我反思和验证 | `cognitive-templates/verification.md` |
| **回溯** | 探索替代路径 | `cognitive-programs/advanced-programs.md` |
| **分解** | 将问题分解为部分 | `cognitive-templates/reasoning.md` |
| **集成** | 组合多个结果 | `cognitive-templates/composition.md` |

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
6. 考虑场动力学和符号机制

### 7.1. 贡献重点领域

我们特别欢迎这些领域的贡献：

1. **场动力学工具**：用于测量和可视化场属性的工具
2. **符号机制实验**：涌现符号处理的演示
3. **认知工具实现**：新的认知操作和模式
4. **协议壳开发**：用于场操作的新颖协议壳
5. **集成示例**：在实际应用中结合多种方法
6. **评估指标**：更好的测量上下文有效性的方法

---

## 8. 许可证和归属

MIT。无门槛：复制、混搭、重新分发。
尊敬地向 Andrej Karpathy 致意，感谢他创造了这个框架。
CITATIONS.md 中的研究致谢。
所有错误都是我们的；欢迎改进。

---

## 9. 路线图

### 9.1. 近期优先事项

1. **符号机制集成**：更好地利用涌现符号机制
2. **场可视化工具**：用于理解场动力学的工具
3. **协议壳扩展**：更多用于场操作的协议壳
4. **评估框架增强**：改进基于场系统的指标
5. **认知工具集成**：与基于场方法的更好集成

### 9.2. 长期愿景

1. **自我演化上下文系统**：自我改进的上下文
2. **场理论形式化**：更严格的数学基础
3. **统一框架**：集成符号机制、场理论和认知工具
4. **跨模型兼容性**：确保技术适用于不同的模型架构
5. **自动上下文优化**：用于自动上下文调整的工具
