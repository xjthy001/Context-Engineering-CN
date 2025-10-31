# 模板组合

> "整体大于部分之和。" — 亚里士多德

## 概述

模板组合涉及结合多个认知模板来处理需要多个推理阶段的复杂问题。通过战略性地排列模板顺序,我们可以创建复杂的认知工作流,引导语言模型通过复杂任务,同时保持结构性和清晰性。

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  模板组合                                                             │
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│  │             │     │             │     │             │             │
│  │  模板 A     │────►│  模板 B     │────►│  模板 C     │─────► ...   │
│  │             │     │             │     │             │             │
│  └─────────────┘     └─────────────┘     └─────────────┘             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 基础组合模式

### 1. 线性序列

最简单的组合模式按固定顺序链接模板。

```markdown
# 线性序列模板

任务:通过结构化的多阶段方法解决以下复杂问题。

问题:{{problem}}

## 阶段 1:理解问题
{{understanding_template}}

## 阶段 2:规划解决方案
{{reasoning_template}}

## 阶段 3:执行计划
{{step_by_step_template}}

## 阶段 4:验证解决方案
{{verification_template}}

## 阶段 5:最终答案
基于以上分析和验证,提供您对原始问题的最终答案。
```

**Token 数量**:根据组件模板而变化

**使用示例**:
- 用于数学问题解决
- 处理复杂推理任务时
- 用于任何多阶段问题解决过程

### 2. 条件分支

此模式引入决策点,确定应用下一个模板。

```markdown
# 条件分支模板

任务:根据问题特征,使用适当方法分析并解决以下问题。

问题:{{problem}}

## 阶段 1:问题分析
{{understanding_template}}

## 阶段 2:方法选择
根据您的分析,确定以下哪种方法最合适:

A) 如果这主要是数学计算问题:
   {{mathematical_reasoning_template}}

B) 如果这主要是逻辑推理问题:
   {{logical_reasoning_template}}

C) 如果这主要是数据分析问题:
   {{data_analysis_template}}

## 阶段 3:解决方案验证
{{verification_template}}

## 阶段 4:最终答案
提供您对原始问题的最终答案。
```

**Token 数量**:根据组件模板而变化

**使用示例**:
- 用于可能需要不同方法的问题
- 当问题类型最初不清楚时
- 用于处理多样化查询类型的系统

### 3. 迭代优化

此模式重复应用模板,直到获得令人满意的结果。

```markdown
# 迭代优化模板

任务:迭代开发和完善以下问题的解决方案。

问题:{{problem}}

## 迭代 1:初始解决方案
{{reasoning_template}}

## 迭代 1 的评估
{{evaluation_template}}

## 迭代 2:改进解决方案
基于对第一次尝试的评估,提供改进的解决方案。
{{reasoning_template}}

## 迭代 2 的评估
{{evaluation_template}}

## 迭代 3:最终解决方案
基于对第二次尝试的评估,提供您的最终解决方案。
{{reasoning_template}}

## 最终验证
{{verification_template}}

## 最终答案
提供您对原始问题的最终答案。
```

**Token 数量**:根据组件模板和迭代次数而变化

**使用示例**:
- 用于受益于优化的创意任务
- 处理困难问题时
- 用于生成高质量内容

## 高级组合模式

### 4. 分而治之

此模式将复杂问题分解为子问题,独立解决每个问题,然后合并结果。

```markdown
# 分而治之模板

任务:通过将以下复杂问题分解为可管理的子问题来解决。

问题:{{problem}}

## 阶段 1:问题分解
{{decomposition_template}}

## 阶段 2:解决子问题
对于上面识别的每个子问题:

### 子问题 1:
{{reasoning_template}}

### 子问题 2:
{{reasoning_template}}

### 子问题 3:
{{reasoning_template}}
(根据需要添加其他子问题)

## 阶段 3:解决方案整合
{{integration_template}}

## 阶段 4:验证
{{verification_template}}

## 阶段 5:最终答案
提供您对原始问题的最终答案。
```

**Token 数量**:根据组件模板和子问题数量而变化

**使用示例**:
- 用于具有不同组件的复杂问题
- 处理具有多个交互部分的系统时
- 用于需要多种类型分析的项目

### 5. 辩证推理

此模式探索对立观点以达成微妙的结论。

```markdown
# 辩证推理模板

任务:通过辩证方法分析以下问题以达成微妙的结论。

问题:{{issue}}

## 阶段 1:问题分析
{{understanding_template}}

## 阶段 2:正题(立场 A)
{{argument_template}}

## 阶段 3:反题(立场 B)
{{argument_template}}

## 阶段 4:合题
{{synthesis_template}}

## 阶段 5:验证
{{verification_template}}

## 阶段 6:结论
提供您对该问题的最终结论。
```

**Token 数量**:根据组件模板而变化

**使用示例**:
- 用于有争议或复杂的主题
- 当存在多个有效观点时
- 用于哲学或伦理问题

### 6. 多代理模拟

此模式通过不同的"代理"模拟不同的专业知识或观点。

```markdown
# 多代理模拟模板

任务:从多个专家视角分析以下问题,以达成全面的解决方案。

问题:{{problem}}

## 阶段 1:问题分析
{{understanding_template}}

## 阶段 2:专家视角

### 视角 1:{{expert_1}}(例如,"数学家")
{{reasoning_template}}

### 视角 2:{{expert_2}}(例如,"经济学家")
{{reasoning_template}}

### 视角 3:{{expert_3}}(例如,"历史学家")
{{reasoning_template}}
(根据需要添加其他视角)

## 阶段 3:协作整合
{{integration_template}}

## 阶段 4:验证
{{verification_template}}

## 阶段 5:最终解决方案
提供您对问题的最终解决方案,整合所有视角的见解。
```

**Token 数量**:根据组件模板和视角数量而变化

**使用示例**:
- 用于跨学科问题
- 当多样化专业知识有价值时
- 用于复杂情况的全面分析

## 实现模式

以下是实现基本线性序列组合的 Python 函数:

```python
def linear_sequence(problem, templates):
    """
    创建一个以线性序列组合多个模板的提示。

    Args:
        problem (str): 要解决的问题
        templates (dict): 按阶段名称键控的模板函数字典

    Returns:
        str: 格式化的线性序列模板提示
    """
    prompt = f"""
任务:通过结构化的多阶段方法解决以下复杂问题。

问题:{problem}
"""

    for i, (stage_name, template_func) in enumerate(templates.items()):
        prompt += f"\n## 阶段 {i+1}:{stage_name}\n"

        # 对于每个模板,我们仅包含指令,而不是再次包含问题陈述
        template_content = template_func(problem)
        # 仅提取指令,假设问题陈述在开头
        instructions = "\n".join(template_content.split("\n")[3:])

        prompt += instructions

    prompt += """
## 最终答案
基于以上分析,提供您对原始问题的最终答案。
"""

    return prompt

# 使用示例
from cognitive_templates import understanding, step_by_step_reasoning, verify_solution

templates = {
    "理解问题": understanding,
    "逐步解决": step_by_step_reasoning,
    "验证解决方案": verify_solution
}

problem = "如果一列火车以 60 英里/小时的速度行驶 2.5 小时,它行驶多远?"
composed_prompt = linear_sequence(problem, templates)
```

## 模板组合策略

组合模板时,考虑这些策略以获得最佳结果:

### 1. 状态管理

确保信息在模板之间正确流动:

```python
def managed_sequence(problem, llm):
    """
    执行具有显式状态管理的模板序列。

    Args:
        problem (str): 要解决的问题
        llm: 用于生成响应的 LLM 接口

    Returns:
        dict: 包含中间结果的完整解决方案
    """
    # 初始化状态
    state = {"problem": problem, "stages": {}}

    # 阶段 1:理解
    understanding_prompt = understanding(problem)
    understanding_result = llm.generate(understanding_prompt)
    state["stages"]["understanding"] = understanding_result

    # 阶段 2:基于理解的上下文进行规划
    planning_prompt = f"""
任务:基于此问题分析规划解决方案方法。

问题:{problem}

问题分析:
{understanding_result}

请概述解决此问题的逐步方法。
"""
    planning_result = llm.generate(planning_prompt)
    state["stages"]["planning"] = planning_result

    # 阶段 3:基于规划的上下文进行执行
    execution_prompt = f"""
任务:执行此问题的解决方案计划。

问题:{problem}

问题分析:
{understanding_result}

解决方案计划:
{planning_result}

请逐步实施此计划以解决问题。
"""
    execution_result = llm.generate(execution_prompt)
    state["stages"]["execution"] = execution_result

    # 阶段 4:基于执行的上下文进行验证
    verification_prompt = verify_solution(problem, execution_result)
    verification_result = llm.generate(verification_prompt)
    state["stages"]["verification"] = verification_result

    # 返回包含所有中间阶段的完整解决方案
    return state
```

### 2. 自适应选择

基于问题特征动态选择模板:

```python
def adaptive_composition(problem, llm):
    """
    根据问题特征自适应地选择和组合模板。

    Args:
        problem (str): 要解决的问题
        llm: 用于生成响应的 LLM 接口

    Returns:
        dict: 包含模板选择理由的完整解决方案
    """
    # 阶段 1:问题分类
    classification_prompt = f"""
任务:对以下问题进行分类,以确定最合适的解决方法。

问题:{problem}

请将此问题分类为以下类别之一:
1. 数学计算
2. 逻辑推理
3. 数据分析
4. 创意写作
5. 决策制定

提供您的分类和简要解释您的推理。
"""
    classification_result = llm.generate(classification_prompt)

    # 解析分类(在实际实现中,使用更健壮的解析)
    problem_type = "未知"
    for category in ["数学", "逻辑", "数据", "创意", "决策"]:
        if category in classification_result:
            problem_type = category
            break

    # 基于问题类型选择模板
    if "数学" in problem_type:
        templates = {
            "理解": understanding,
            "解决方案": step_by_step_reasoning,
            "验证": verify_solution
        }
    elif "逻辑" in problem_type:
        templates = {
            "理解": understanding,
            "论证分析": lambda p: logical_argument_template(p),
            "验证": verify_solution
        }
    # 为其他问题类型添加更多条件

    # 执行选定的模板序列
    result = {
        "problem": problem,
        "classification": classification_result,
        "selected_approach": problem_type,
        "stages": {}
    }

    for stage_name, template_func in templates.items():
        prompt = template_func(problem)
        response = llm.generate(prompt)
        result["stages"][stage_name] = response

    return result
```

### 3. 反馈驱动优化

使用评估结果指导模板选择和优化:

```python
def feedback_driven_composition(problem, llm, max_iterations=3):
    """
    使用反馈驱动模板选择和优化。

    Args:
        problem (str): 要解决的问题
        llm: 用于生成响应的 LLM 接口
        max_iterations (int): 最大优化迭代次数

    Returns:
        dict: 包含优化历史的完整解决方案
    """
    # 初始化状态
    state = {
        "problem": problem,
        "iterations": [],
        "final_solution": None,
        "quality_score": 0
    }

    # 初始解决方案
    solution = llm.generate(step_by_step_reasoning(problem))

    for i in range(max_iterations):
        # 评估当前解决方案
        evaluation_prompt = f"""
任务:评估此解决方案的质量和正确性。

问题:{problem}

提议的解决方案:
{solution}

请在 1-10 的范围内评估此解决方案:
1. 正确性(答案是否正确?)
2. 清晰度(推理是否清晰?)
3. 完整性(是否解决了所有方面?)

对于每个标准,提供分数和简要解释。
然后建议可以进行的具体改进。
"""
        evaluation = llm.generate(evaluation_prompt)

        # 提取质量分数(在实际实现中,使用更健壮的解析)
        quality_score = 0
        for line in evaluation.split("\n"):
            if "正确性" in line and ":" in line:
                try:
                    quality_score += int(line.split(":")[1].strip().split("/")[0])
                except:
                    pass
            if "清晰度" in line and ":" in line:
                try:
                    quality_score += int(line.split(":")[1].strip().split("/")[0])
                except:
                    pass
            if "完整性" in line and ":" in line:
                try:
                    quality_score += int(line.split(":")[1].strip().split("/")[0])
                except:
                    pass

        quality_score = quality_score / 3  # 平均分数

        # 记录此迭代
        state["iterations"].append({
            "solution": solution,
            "evaluation": evaluation,
            "quality_score": quality_score
        })

        # 检查质量是否令人满意
        if quality_score >= 8:
            break

        # 基于评估选择改进模板
        if "正确性" in evaluation and "清晰" not in evaluation.lower():
            # 如果正确性是主要问题,专注于验证
            improvement_template = verify_solution
        elif "清晰" in evaluation.lower():
            # 如果清晰度是主要问题,专注于解释
            improvement_template = lambda p: step_by_step_reasoning(p, steps=["理解", "计划", "用清晰的解释执行", "验证", "结论"])
        else:
            # 默认为一般改进
            improvement_template = step_by_step_reasoning

        # 生成改进的解决方案
        improvement_prompt = f"""
任务:基于此评估反馈改进以下解决方案。

问题:{problem}

当前解决方案:
{solution}

评估:
{evaluation}

请提供改进的解决方案,解决评估中识别的问题。
"""
        solution = llm.generate(improvement_prompt)

    # 基于质量分数选择最佳解决方案
    best_iteration = max(state["iterations"], key=lambda x: x["quality_score"])
    state["final_solution"] = best_iteration["solution"]
    state["quality_score"] = best_iteration["quality_score"]

    return state
```

## 测量组合有效性

使用模板组合时,通过以下方式测量其有效性:

1. **端到端准确性**:完整组合是否产生正确结果?
2. **阶段贡献**:每个模板对最终质量的贡献有多大?
3. **信息流**:重要上下文是否在模板之间保留?
4. **效率**:与更简单的方法相比,组合的 token 开销是多少?
5. **适应性**:组合如何处理不同的问题变化?

## 有效组合的技巧

1. **从简单开始**:在尝试更复杂的模式之前,从线性序列开始
2. **最小化冗余**:避免跨模板重复指令
3. **保留上下文**:确保关键信息在模板之间流动
4. **平衡结构与灵活性**:过于严格的组合限制了模型的优势
5. **用变化进行测试**:验证您的组合在问题变化中是否有效
6. **包含自我纠正**:内置验证和优化机会

## 下一步

- 了解这些组合模式如何在 [../cognitive-programs/program-library.py](../cognitive-programs/program-library.py) 中实现
- 在 [../cognitive-architectures/solver-architecture.md](../cognitive-architectures/solver-architecture.md) 中探索完整的认知架构
- 在 [../integration/with-rag.md](../integration/with-rag.md) 和 [../integration/with-memory.md](../integration/with-memory.md) 中了解如何将这些组合与检索和记忆集成

---

## 深入探讨:模板元编程

高级从业者可以创建动态生成模板的系统:

```python
def generate_specialized_template(domain, complexity, llm):
    """
    为特定领域和复杂度级别生成专门化模板。

    Args:
        domain (str): 领域区域(例如,"数学","法律")
        complexity (str): 复杂度级别(例如,"基础","高级")
        llm: 用于生成模板的 LLM 接口

    Returns:
        function: 生成的模板函数
    """
    prompt = f"""
任务:创建一个专门化的认知模板,用于解决 {domain} 领域的 {complexity} 问题。

模板应该:
1. 包含适当的领域特定术语和概念
2. 将推理过程分解为清晰的步骤
3. 包含领域特定的验证检查
4. 针对 {complexity} 复杂度级别进行校准

将模板格式化为 markdown 文档,包含:
1. 清晰的任务描述
2. 在此领域解决问题的结构化步骤
3. 每个步骤的领域特定指导
4. 此领域特定的验证标准

请生成完整的模板文本。
"""

    template_text = llm.generate(prompt)

    # 创建应用此模板的函数
    def specialized_template(problem):
        return f"""
任务:使用专门化方法解决以下 {complexity} {domain} 问题。

问题:{problem}

{template_text}
"""

    return specialized_template

# 使用示例
legal_reasoning_template = generate_specialized_template("法律", "高级", llm)
math_template = generate_specialized_template("数学", "中级", llm)

# 应用生成的模板
legal_problem = "分析本合同条款中的责任影响..."
legal_prompt = legal_reasoning_template(legal_problem)
```

这种元级别方法能够创建针对特定领域和复杂度级别定制的高度专业化模板。
