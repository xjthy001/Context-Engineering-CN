# 原子：提示工程的基本单元

> "如果你想从零开始制作苹果派，你必须先创造宇宙。" — 卡尔·萨根

## 原子：单一指令

在我们探索上下文工程的旅程中，我们从最基本的单元开始：**原子** — 一条发送给大语言模型的单一、独立的指令。

```
┌───────────────────────────────────────────────┐
│                                               │
│  "Write a poem about the ocean in 4 lines."   │
│                                               │
└───────────────────────────────────────────────┘
```

这是最纯粹形式的提示工程：一个人类，一条指令，一个模型响应。简单、直接、原子化。

## 原子提示的解剖结构

让我们分解一下有效原子提示的构成要素：

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ATOMIC PROMPT = [TASK] + [CONSTRAINTS] + [OUTPUT FORMAT]   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

例如：

```
┌─────────────────────┬────────────────────────┬────────────────────┐
│        TASK         │      CONSTRAINTS       │   OUTPUT FORMAT    │
├─────────────────────┼────────────────────────┼────────────────────┤
│ "Write a poem       │ "about the ocean       │ "in 4 lines."      │
│  about space."      │  using only words      │                    │
│                     │  with 5 letters        │                    │
│                     │  or less."             │                    │
└─────────────────────┴────────────────────────┴────────────────────┘
```

## 原子的局限性

虽然原子提示是大语言模型交互的基石，但它们很快就会暴露出根本性的局限：

```
┌──────────────────────────────────────┐
│ LIMITATIONS OF ATOMIC PROMPTS        │
├──────────────────────────────────────┤
│ ✗ No memory across interactions      │
│ ✗ Limited demonstration capability   │
│ ✗ No complex reasoning scaffolds     │
│ ✗ Prone to ambiguity                 │
│ ✗ High variance in outputs           │
└──────────────────────────────────────┘
```

让我们通过一个简单的实验来实证地衡量这一点：

```python
# A basic atomic prompt
atomic_prompt = "List 5 symptoms of diabetes."

# Send to LLM multiple times
responses = [llm.generate(atomic_prompt) for _ in range(5)]

# Measure variability
unique_symptoms = set()
for response in responses:
    symptoms = extract_symptoms(response)
    unique_symptoms.update(symptoms)

print(f"Found {len(unique_symptoms)} unique symptoms across 5 identical prompts")
# Typically outputs far more than just 5 unique symptoms
```

问题在哪？当给定最少的上下文时，模型很难保持一致性。

## 单原子基线：有用但受限

尽管有局限性，原子提示建立了我们的基线。它们帮助我们：

1. 衡量词元效率（最小开销）
2. 基准测试响应质量
3. 为实验建立对照组

```
                     [Response Quality]
                            ▲
                            │
                            │               ⭐ Context
                            │                 Engineering
                            │
                            │
                            │       ⭐ Advanced
                            │         Prompting
                            │
                            │   ⭐ Basic Prompting
                            │
                            │
                            └────────────────────────►
                                  [Complexity]
```

## 未言明的上下文：模型已经"知道"什么

即使使用原子提示，大语言模型也会利用其训练中的大量隐式上下文：

```
┌───────────────────────────────────────────────────────────────┐
│ IMPLICIT CONTEXT IN MODELS                                    │
├───────────────────────────────────────────────────────────────┤
│ ✓ Language rules and grammar                                  │
│ ✓ Common knowledge facts                                      │
│ ✓ Format conventions (lists, paragraphs, etc.)                │
│ ✓ Domain-specific knowledge (varies by model)                 │
│ ✓ Learned interaction patterns                                │
└───────────────────────────────────────────────────────────────┘
```

这种隐式知识为我们提供了基础，但它并不可靠，并且在不同的模型和版本之间会有所不同。

## 幂律：词元-质量曲线

对于许多任务，我们观察到上下文词元与输出质量之间存在幂律关系：

```
Quality
      ▲
      │                        •
      │                    •       •
      │                •               •
      │            •                       •
      │        •                               •
      │    •
      │•
      └───────────────────────────────────────────► Tokens
          [Poor Start]  [Maximum ROI]  [Diminishing Returns]
```

关键见解：存在一个"最大投资回报率区间"，在这个区间，仅添加几个词元就能带来显著的质量提升，同时也存在"收益递减"区间，在这个区间添加更多词元反而会降低性能。

## [阅读更多关于上下文衰退的内容](https://research.trychroma.com/context-rot)

## 从原子到分子：对更多上下文的需求

原子的局限性自然地引导我们进入下一步：**分子**，即结合指令与示例、额外上下文和结构化格式的多部分提示。

这是基本的转变：

```
┌──────────────────────────┐         ┌──────────────────────────┐
│                          │         │ "Here's an example:      │
│ "Write a limerick about  │    →    │  There once was a...     │
│  a programmer."          │         │                          │
│                          │         │  Now write a limerick    │
└──────────────────────────┘         │  about a programmer."    │
                                     └──────────────────────────┘
    [Atomic Prompt]                       [Molecular Prompt]
```

通过添加示例和结构，我们开始有意识地塑造上下文窗口——这是迈向上下文工程的第一步。

## 衡量原子效率：你的第一个任务

在继续之前，尝试这个简单的练习：

1. 选择一个你会给大语言模型的基本任务
2. 创建三个不同的原子提示版本
3. 衡量使用的词元数量和主观质量
4. 绘制效率边界

```
┌─────────────────────────────────────────────────────────────┐
│ Task: Summarize a news article                              │
├─────────┬───────────────────────────────┬────────┬──────────┤
│ Version │ Prompt                        │ Tokens │ Quality  │
├─────────┼───────────────────────────────┼────────┼──────────┤
│ A       │ "Summarize this article."     │ 4      │ 2/10     │
├─────────┼───────────────────────────────┼────────┼──────────┤
│ B       │ "Provide a concise summary    │ 14     │ 6/10     │
│         │  of this article in 3         │        │          │
│         │  sentences."                  │        │          │
├─────────┼───────────────────────────────┼────────┼──────────┤
│ C       │ "Write a summary of the key   │ 27     │ 8/10     │
│         │  points in this article,      │        │          │
│         │  highlighting the main        │        │          │
│         │  people and events."          │        │          │
└─────────┴───────────────────────────────┴────────┴──────────┘
```

## 关键要点

1. **原子提示**是大语言模型交互的基本单元
2. 它们遵循基本结构：任务 + 约束 + 输出格式
3. 它们具有固有的局限性：没有记忆、示例或推理支架
4. 即使是简单的原子提示也利用了模型的隐式知识
5. 上下文词元与质量之间存在幂律关系
6. 超越原子是迈向上下文工程的第一步

## 下一步

在下一节中，我们将探讨如何将原子组合成**分子** — 显著提高可靠性和可控性的少样本学习模式。

[继续阅读 02_molecules_context.md →](02_molecules_context.md)

---

## 深入探讨：提示模板

对于那些想要更多地尝试原子提示的人，这里有一些可以尝试的模板：

```
# Basic instruction
{task}

# Persona-based
As a {persona}, {task}

# Format-specific
{task}
Format: {format_specification}

# Constraint-based
{task}
Constraints:
- {constraint_1}
- {constraint_2}
- {constraint_3}

# Step-by-step guided
{task}
Please follow these steps:
1. {step_1}
2. {step_2}
3. {step_3}
```

尝试测量将每个模板应用于相同任务时的词元数量和质量！
