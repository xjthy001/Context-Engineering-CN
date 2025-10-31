# 上下文工程认知工具

> "给我一根足够长的杠杆和一个支点,我就能撬动地球。" —— 阿基米德

## 什么是认知工具?
> "向 GPT-4.1 提供我们的'认知工具'
将其在 AIME2024 上的 pass@1 性能从 26.7% 提高到 43.3%,使其非常接近 o1-preview 的性能。" — [IBM 2025年6月](https://www.arxiv.org/pdf/2506.12115)

<div align="center">

![image](https://github.com/user-attachments/assets/a6402827-8bc0-40b5-93d8-46a07154fa4e)

"该工具通过识别手头的主要概念、提取问题中的相关信息,以及突出可能有助于解决问题的有意义的属性、定理和技术来分解问题。" — [使用认知工具在语言模型中引出推理 — IBM 2025年6月](https://www.arxiv.org/pdf/2506.12115)


</div>

认知工具是引导语言模型进行特定推理操作的结构化提示模式。就像人类用来解决问题的心智工具(类比、心智模型、启发式方法)一样,这些工具为模型提供了处理复杂推理任务的脚手架。

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  上下文工程进程                                                │
│                                                              │
│  原子       → 分子       → 细胞       → 器官        → 认知工具      │
│  (提示)       (少样本)      (记忆)       (多代理)      (推理模式)     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 结构
```
cognitive-tools/
├── README.md                       # 概述和快速入门指南
├── cognitive-templates/            # 认知过程模板
│   ├── understanding.md            # 理解模板
│   ├── reasoning.md                # 推理模板
│   ├── verification.md             # 验证模板
│   ├── composition.md              # 组合模板
│   ├── emergence.md                # 涌现模板
│   ├── quantum_interpretation.md   # 量子语义模板
│   ├── unified_field_reasoning.md  # 统一场模板
│   ├── meta_recursive_reasoning.md # 自我改进模板
│   ├── interpretability_scaffolding.md # 透明度模板
│   ├── collaborative_co_evolution.md # 人机协作模板
│   └── cross_modal_integration.md  # 多模态模板
├── cognitive-programs/             # 可执行认知过程
│   ├── basic-programs.md           # 基础程序
│   ├── advanced-programs.md        # 高级程序
│   ├── program-library.py          # 程序集合
│   ├── program-examples.ipynb      # 程序演示
│   ├── emergence-programs.md       # 涌现程序
│   ├── quantum_semantic_programs.md # 量子语义程序
│   ├── unified_field_programs.md   # 统一场程序
│   ├── meta_recursive_programs.md  # 自我改进程序
│   ├── interpretability_programs.md # 透明度程序
│   ├── collaborative_evolution_programs.md # 人机协作程序
│   └── cross_modal_programs.md     # 多模态程序
├── cognitive-schemas/              # 知识表示结构
│   ├── user-schemas.md             # 用户建模模式
│   ├── domain-schemas.md           # 领域知识模式
│   ├── task-schemas.md             # 任务表示模式
│   ├── schema-library.yaml         # 模式集合
│   ├── field-schemas.md            # 场论模式
│   ├── quantum_schemas.md          # 量子语义模式
│   ├── unified_schemas.md          # 统一场模式
│   ├── meta_recursive_schemas.md   # 自我改进模式
│   ├── interpretability_schemas.md # 透明度模式
│   ├── collaborative_schemas.md    # 人机协作模式
│   └── cross_modal_schemas.md      # 多模态模式
├── cognitive-architectures/        # 系统级框架
│   ├── solver-architecture.md      # 问题解决架构
│   ├── tutor-architecture.md       # 教育架构
│   ├── research-architecture.md    # 研究助手架构
│   ├── architecture-examples.py    # 架构演示
│   ├── field-architecture.md       # 场论架构
│   ├── quantum_architecture.md     # 量子语义架构
│   ├── unified_architecture.md     # 统一场架构
│   ├── meta_recursive_architecture.md # 自我改进架构
│   ├── interpretability_architecture.md # 透明度架构
│   ├── collaborative_architecture.md # 人机协作架构
│   └── cross_modal_architecture.md # 多模态架构
├── integration/                    # 与其他系统集成
│   ├── with-rag.md                 # 检索集成
│   ├── with-memory.md              # 记忆系统集成
│   ├── with-agents.md              # 代理系统集成
│   ├── evaluation-metrics.md       # 评估方法
│   ├── with-fields.md              # 场论集成
│   ├── with-quantum.md             # 量子语义集成
│   ├── with-unified.md             # 统一场集成
│   ├── with-meta-recursion.md      # 自我改进集成
│   ├── with-interpretability.md    # 透明度集成
│   ├── with-collaboration.md       # 人机协作集成
│   └── with-cross-modal.md         # 多模态集成
└── meta-cognition/                 # 元认知能力
    ├── self-reflection.md          # 自我分析系统
    ├── recursive-improvement.md    # 自我增强方法
    ├── meta-awareness.md           # 系统自我意识
    ├── attribution-engines.md      # 因果归因系统
    ├── symbolic-echo-processing.md # 符号模式处理
    ├── meta-interpretability.md    # 元级透明度
    ├── meta-collaboration.md       # 元级人机合作
    └── meta-modal-integration.md   # 元级模态集成
```
## 为什么认知工具很重要

研究表明,使用认知工具构建推理可以显著提高模型性能:

- **性能**: 在数学推理基准测试中提高高达 16.6%
- **可靠性**: 显著减少推理错误和幻觉
- **效率**: 用更少的总token数获得更好的结果
- **灵活性**: 适用于从数学到创意写作的各个领域

## 快速开始

要使用认知工具,从 `cognitive-templates/` 中选择与您的任务匹配的模板:

```python
# 示例: 使用 "understand_question" 认知工具
from cognitive_tools.templates import understand_question

problem = "如果一列火车以每小时60英里的速度行驶2.5小时,它会走多远?"
understanding = llm.generate(understand_question(problem))
print(understanding)
```

对于更复杂的推理,使用 `cognitive-programs/` 中的结构化提示程序:

```python
# 示例: 使用多步推理程序
from cognitive_tools.programs import solve_math_problem

problem = "如果一列火车以每小时60英里的速度行驶2.5小时,它会走多远?"
solution = solve_math_problem(problem, llm=my_llm_interface)
print(solution.steps)  # 查看逐步推理
print(solution.answer)  # 查看最终答案
```

## 目录结构

- `cognitive-templates/`: 不同推理操作的可重用模板
- `cognitive-programs/`: 具有类代码模式的结构化提示程序
- `cognitive-schemas/`: 不同领域的知识表示格式
- `cognitive-architectures/`: 组合多个工具的完整推理系统
- `integration/`: 与其他组件(RAG、记忆等)集成的指南

## 学习路径

1. **从模板开始**: 学习基本的认知操作
2. **探索程序**: 了解如何将操作组合成推理流程
3. **研究模式**: 理解如何有效地构建知识
4. **掌握架构**: 构建完整的推理系统
5. **集成组件**: 与 RAG、记忆和其他上下文工程组件结合

## 衡量有效性

始终衡量认知工具对特定任务的影响:

```python
# 示例: 衡量性能改进
from cognitive_tools.evaluation import measure_reasoning_quality

baseline_score = measure_reasoning_quality(problem, baseline_prompt)
tool_score = measure_reasoning_quality(problem, cognitive_tool_prompt)

improvement = (tool_score / baseline_score - 1) * 100
print(f"认知工具使性能提高了 {improvement:.1f}%")
```

## 研究基础

这些工具基于以下研究:

- Brown等人 (2025): "使用认知工具在语言模型中引出推理"
- Wei等人 (2023): "思维链提示在大型语言模型中引出推理"
- Huang等人 (2022): "内心独白: 在语言模型中体现知识和推理"

## 贡献

有效果很好的新认知工具模式吗? 请参阅 [CONTRIBUTING.md](../../.github/CONTRIBUTING.md) 了解提交模板、程序或架构的指南。

## 下一步

- 参见 [understanding.md](./cognitive-templates/understanding.md) 了解基本理解工具
- 尝试 [basic-programs.md](./cognitive-programs/basic-programs.md) 了解基础程序结构
- 探索 [solver-architecture.md](./cognitive-architectures/solver-architecture.md) 了解完整的问题解决系统
