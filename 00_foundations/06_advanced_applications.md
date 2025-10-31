# 高级应用：将上下文工程付诸实践

> "理论上，理论和实践是一样的。实践中，它们并不相同。" —— 阿尔伯特·爱因斯坦

## 超越基础：应用上下文工程

我们已经建立了上下文工程概念的坚实基础，从原子提示到认知工具。现在是时候看看这些原则如何应用于突破 LLM 可能性边界的实际挑战了。

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │     │              │
│    原子      │────►│    分子      │────►│    细胞      │────►│    器官      │
│   (提示)     │     │  (少样本)    │     │   (记忆)     │     │  (多智能体)  │
│              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                   │                    │
       │                    │                   │                    │
       │                    │                   │                    │
       ▼                    ▼                   ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                            高级应用                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 应用领域：长篇内容创建

创建长篇、连贯的内容突破了上下文管理的极限。让我们看看我们的原则如何应用：

```
┌───────────────────────────────────────────────────────────────────────────┐
│                       长篇内容创建                                         │
│                                                                           │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│  │                 │     │                 │     │                 │      │
│  │   内容          │────►│   章节          │────►│   渐进式        │      │
│  │   规划          │     │   生成          │     │   整合          │      │
│  │                 │     │                 │     │                 │      │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │             │         │             │         │             │          │
│  │   大纲      │         │   章节      │         │   连贯性    │          │
│  │   架构      │         │   模板      │         │   验证      │          │
│  │             │         │             │         │             │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 实现：文档生成系统

```python
class LongFormGenerator:
    """用于生成连贯长篇内容的系统。"""

    def __init__(self, llm_service):
        self.llm = llm_service
        self.document_state = {
            "title": "",
            "outline": [],
            "sections": {},
            "current_section": "",
            "theme_keywords": [],
            "style_guide": {},
            "completed_sections": []
        }

    def create_outline(self, topic, length="medium", style="informative"):
        """为文档生成结构化大纲。"""
        # 大纲生成的提示程序示例
        outline_prompt = f"""
        任务：为关于 {topic} 的 {length} {style} 文档创建详细大纲。

        过程：
        1. 确定 3-5 个全面覆盖主题的主要部分
        2. 对于每个主要部分，确定 2-4 个小节
        3. 添加简短描述（1-2 句话）说明每个部分将涵盖的内容
        4. 包括各部分之间的建议过渡

        格式：
        标题：[建议标题]

        主要部分：
        1. [部分标题]
           - 描述：[简短描述]
           - 小节：
             a. [小节标题]
             b. [小节标题]
           - 过渡：[流向下一部分的建议]

        2. [继续模式...]

        主题关键词：[5-7 个保持一致性的关键术语]
        语气指南：[3-4 个文体建议]
        """

        outline_response = self.llm.generate(outline_prompt)
        self._parse_outline(outline_response)
        return self.document_state["outline"]

    def _parse_outline(self, outline_text):
        """将大纲响应解析为结构化格式。"""
        # 在实际实现中，这将提取结构化大纲
        # 为简单起见，我们将使用占位符实现
        self.document_state["title"] = "示例文档标题"
        self.document_state["outline"] = [
            {"title": "引言", "subsections": ["背景", "重要性"]},
            {"title": "主要部分 1", "subsections": ["子主题 A", "子主题 B"]},
            {"title": "主要部分 2", "subsections": ["子主题 C", "子主题 D"]},
            {"title": "结论", "subsections": ["总结", "未来方向"]}
        ]
        self.document_state["theme_keywords"] = ["关键词1", "关键词2", "关键词3"]
        self.document_state["style_guide"] = {
            "tone": "informative",
            "perspective": "third person",
            "style_notes": "使用具体示例"
        }

    def generate_section(self, section_index):
        """为特定部分生成内容。"""
        section = self.document_state["outline"][section_index]
        self.document_state["current_section"] = section["title"]

        # 创建上下文感知的部分提示
        context = self._build_section_context(section_index)

        section_prompt = f"""
        任务：为标题为 "{self.document_state["title"]}" 的文档编写 "{section["title"]}" 部分。

        上下文：
        {context}

        指南：
        - 与文档的主题和前面的部分保持一致
        - 处理所有小节：{", ".join(section["subsections"])}
        - 保持 {self.document_state["style_guide"]["tone"]} 的语气
        - 从 {self.document_state["style_guide"]["perspective"]} 的角度写作
        - {self.document_state["style_guide"]["style_notes"]}

        格式：
        ## {section["title"]}

        [处理所有小节的内容，大约 300-500 字]
        """

        section_content = self.llm.generate(section_prompt)
        self.document_state["sections"][section["title"]] = section_content
        self.document_state["completed_sections"].append(section["title"])

        return section_content

    def _build_section_context(self, section_index):
        """为生成部分构建相关上下文。"""
        context = "前面的部分：\n"

        # 包括先前编写部分的摘要以提供上下文
        for title in self.document_state["completed_sections"]:
            # 实践中，您会包括摘要而不是全文以节省令牌
            content = self.document_state["sections"].get(title, "")
            summary = content[:100] + "..." if len(content) > 100 else content
            context += f"- {title}：{summary}\n"

        # 包括主题关键词以保持一致性
        context += "\n主题关键词：" + ", ".join(self.document_state["theme_keywords"])

        # 位置信息（开头、中间、结尾）
        total_sections = len(self.document_state["outline"])
        if section_index == 0:
            context += "\n这是文档的开头部分。"
        elif section_index == total_sections - 1:
            context += "\n这是文档的结尾部分。"
        else:
            context += f"\n这是 {total_sections} 个部分中的第 {section_index + 1} 个。"

        return context

    def verify_coherence(self, section_index):
        """验证和改进与前面部分的连贯性。"""
        if section_index == 0:
            return "第一部分 - 不需要连贯性检查。"

        section = self.document_state["outline"][section_index]
        previous_section = self.document_state["outline"][section_index - 1]

        current_content = self.document_state["sections"][section["title"]]
        previous_content = self.document_state["sections"][previous_section["title"]]

        # 使用专门的提示程序进行连贯性验证
        coherence_prompt = f"""
        任务：验证和改进两个连续文档部分之间的连贯性。

        前一部分：{previous_section["title"]}
        {previous_content[-200:]}  # 前一部分的最后部分

        当前部分：{section["title"]}
        {current_content[:200]}  # 当前部分的开头部分

        过程：
        1. 识别任何主题或逻辑不连贯
        2. 检查重复或矛盾
        3. 验证过渡是否流畅
        4. 确保术语和风格一致

        格式：
        连贯性评估：[良好/需要改进]

        发现的问题：
        1. [问题 1（如果有）]
        2. [问题 2（如果有）]

        改进建议：
        [改进连接的具体建议]
        """

        assessment = self.llm.generate(coherence_prompt)

        # 在完整实现中，您将解析评估并应用改进
        return assessment

    def generate_complete_document(self):
        """逐部分生成完整文档。"""
        # 首先，确保我们有大纲
        if not self.document_state["outline"]:
            raise ValueError("必须先创建大纲")

        # 按顺序生成每个部分
        all_content = [f"# {self.document_state['title']}\n\n"]

        for i in range(len(self.document_state["outline"])):
            section_content = self.generate_section(i)

            # 对于第一个之后的部分，验证连贯性
            if i > 0:
                coherence_check = self.verify_coherence(i)
                # 实践中，您将使用它来改进该部分

            all_content.append(section_content)

        # 合并所有部分
        return "\n\n".join(all_content)
```

此实现演示了：
1. **结构化内容规划**使用提示程序
2. **渐进式上下文构建**随着部分的生成
3. **连贯性验证**在相邻部分之间
4. **状态管理**贯穿整个文档创建过程

## 应用领域：带记忆的复杂推理

复杂推理通常需要在多个步骤中跟踪状态，同时保留关键见解：

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         复杂推理系统                                       │
│                                                                           │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│  │                 │     │                 │     │                 │      │
│  │   问题          │────►│   解决方案      │────►│   验证与        │      │
│  │   分析          │     │   生成          │     │   改进          │      │
│  │                 │     │                 │     │                 │      │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │             │         │             │         │             │          │
│  │   结构化    │         │   思维链    │         │   自我      │          │
│  │   问题      │         │   模板      │         │   纠正      │          │
│  │   架构      │         │             │         │   循环      │          │
│  │             │         │             │         │             │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 实现：数学问题求解器

```python
class MathProblemSolver:
    """用于逐步解决复杂数学问题的系统。"""

    def __init__(self, llm_service):
        self.llm = llm_service
        self.problem_state = {
            "original_problem": "",
            "parsed_problem": {},
            "solution_steps": [],
            "current_step": 0,
            "verification_results": [],
            "final_answer": ""
        }

    def parse_problem(self, problem_text):
        """解析和结构化数学问题。"""
        # 基于架构的问题解析
        parse_prompt = f"""
        任务：分析和结构化以下数学问题。

        问题：{problem_text}

        过程：
        1. 识别问题类型（代数、微积分、几何等）
        2. 提取相关变量及其关系
        3. 识别约束和条件
        4. 确定要求什么

        格式：
        问题类型：[类型]

        变量：
        - [变量 1]：[描述]
        - [变量 2]：[描述]

        关系：
        - [等式或关系 1]
        - [等式或关系 2]

        约束：
        - [约束 1]
        - [约束 2]

        目标：[需要找到什么]

        建议方法：[解决方法的简要概述]
        """

        parse_result = self.llm.generate(parse_prompt)
        self.problem_state["original_problem"] = problem_text

        # 实践中，您将解析结构化输出
        # 为简单起见，我们将使用占位符实现
        self.problem_state["parsed_problem"] = {
            "type": "代数",
            "variables": {"x": "未知值", "y": "因变量值"},
            "relationships": ["y = 2x + 3"],
            "constraints": ["x > 0"],
            "goal": "当 y = 15 时求 x",
            "approach": "代入 y = 15 并求解 x"
        }

        return self.problem_state["parsed_problem"]

    def generate_solution_step(self):
        """生成解决过程中的下一步。"""
        # 从先前的步骤构建上下文
        context = self._build_step_context()

        step_prompt = f"""
        任务：生成解决此数学问题的下一步。

        原始问题：{self.problem_state["original_problem"]}

        问题分析：
        类型：{self.problem_state["parsed_problem"]["type"]}
        目标：{self.problem_state["parsed_problem"]["goal"]}

        先前的步骤：
        {context}

        过程：
        1. 考虑先前步骤中已完成的内容
        2. 确定下一个逻辑操作
        3. 执行该操作，显示所有工作
        4. 解释数学推理

        格式：
        步骤 {self.problem_state["current_step"] + 1}：[简要描述]

        操作：[执行的数学操作]

        工作：
        [逐步计算]

        解释：
        [为什么这一步是必要的以及它完成了什么]

        状态：[完成/需要更多步骤]
        """

        step_result = self.llm.generate(step_prompt)
        self.problem_state["solution_steps"].append(step_result)
        self.problem_state["current_step"] += 1

        # 检查此步骤是否包含最终答案
        if "状态：完成" in step_result:
            # 提取最终答案（实践中，您会更仔细地解析这个）
            self.problem_state["final_answer"] = "x = 6"

        return step_result

    def _build_step_context(self):
        """从先前的解决步骤构建上下文。"""
        if not self.problem_state["solution_steps"]:
            return "没有先前的步骤。这是解决方案的开始。"

        # 在上下文中包括所有先前的步骤
        # 实践中，您可能需要为令牌限制进行总结或截断
        context = "先前的步骤：\n"
        for i, step in enumerate(self.problem_state["solution_steps"]):
            context += f"步骤 {i+1}：{step[:200]}...\n"

        return context

    def verify_step(self, step_index):
        """验证特定解决步骤的正确性。"""
        if step_index >= len(self.problem_state["solution_steps"]):
            return "步骤索引超出范围"

        step = self.problem_state["solution_steps"][step_index]

        # 使用专门的提示进行验证
        verify_prompt = f"""
        任务：验证此数学解决步骤的正确性。

        原始问题：{self.problem_state["original_problem"]}

        要验证的步骤：
        {step}

        过程：
        1. 检查数学操作的准确性
        2. 验证逻辑是否遵循先前的步骤
        3. 确保解释与显示的工作相匹配
        4. 寻找常见错误或误解

        格式：
        正确性：[正确/不正确/部分正确]

        发现的问题：
        - [问题 1（如果有）]
        - [问题 2（如果有）]

        建议的更正：
        [如何修复识别的任何问题]
        """

        verification = self.llm.generate(verify_prompt)
        self.problem_state["verification_results"].append(verification)

        return verification

    def solve_complete_problem(self, problem_text, max_steps=10):
        """通过验证逐步解决完整问题。"""
        # 解析问题
        self.parse_problem(problem_text)

        # 生成并验证步骤，直到解决方案完成
        while self.problem_state["final_answer"] == "" and self.problem_state["current_step"] < max_steps:
            # 生成下一步
            step = self.generate_solution_step()

            # 验证该步骤
            verification = self.verify_step(self.problem_state["current_step"] - 1)

            # 如果验证发现问题，您可能会重新生成该步骤
            # 这是一个简化的实现
            if "正确性：不正确" in verification:
                # 实践中，您将使用反馈来改进该步骤
                print(f"步骤 {self.problem_state['current_step']} 有问题。为本示例继续。")

        # 返回完整解决方案
        return {
            "problem": self.problem_state["original_problem"],
            "steps": self.problem_state["solution_steps"],
            "verifications": self.problem_state["verification_results"],
            "final_answer": self.problem_state["final_answer"]
        }
```

此实现演示了：
1. **结构化问题解析**使用基于架构的方法
2. **逐步推理**具有显式的中间状态
3. **自我验证**在每个阶段检查工作
4. **记忆管理**在整个解决过程中保持上下文

## 应用领域：知识综合

从多个来源综合信息需要复杂的上下文管理：

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         知识综合系统                                       │
│                                                                           │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│  │                 │     │                 │     │                 │      │
│  │   信息          │────►│   概念          │────►│   整合与        │      │
│  │   检索          │     │   提取          │     │   综合          │      │
│  │                 │     │                 │     │                 │      │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │             │         │             │         │             │          │
│  │   检索      │         │   知识      │         │   比较      │          │
│  │   查询      │         │   图谱      │         │   矩阵      │          │
│  │   模板      │         │   架构      │         │   模板      │          │
│  │             │         │             │         │             │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 实现：研究助手

```python
class ResearchAssistant:
    """用于从多个来源综合信息的系统。"""

    def __init__(self, llm_service, retrieval_service):
        self.llm = llm_service
        self.retrieval = retrieval_service
        self.research_state = {
            "topic": "",
            "query_results": [],
            "extracted_concepts": {},
            "concept_relationships": [],
            "synthesis": "",
            "knowledge_gaps": []
        }

    def set_research_topic(self, topic):
        """设置研究主题并生成初始查询。"""
        self.research_state["topic"] = topic

        # 使用提示程序生成结构化查询
        query_prompt = f"""
        任务：为研究主题生成有效的搜索查询："{topic}"

        过程：
        1. 将主题分解为其核心组成部分
        2. 对于每个组成部分，生成具体的搜索查询
        3. 包括关于主题不同观点的查询
        4. 添加背景/基础信息的查询

        格式：
        核心组成部分：
        - [组成部分 1]
        - [组成部分 2]

        推荐查询：
        1. [具体查询 1]
        2. [具体查询 2]
        3. [具体查询 3]

        观点查询：
        1. [观点 1 的查询]
        2. [观点 2 的查询]

        背景查询：
        1. [背景 1 的查询]
        2. [背景 2 的查询]
        """

        query_suggestions = self.llm.generate(query_prompt)

        # 实践中，您将解析结构化输出
        # 对于本示例，我们将使用占位符查询
        return ["query1", "query2", "query3"]

    def retrieve_information(self, queries):
        """使用生成的查询检索信息。"""
        # 在实际实现中，这将调用实际的检索服务
        # 对于本示例，我们将使用占位符结果
        for query in queries:
            # 模拟检索结果
            results = [
                {"title": f"Result 1 for {query}", "content": "示例内容 1", "source": "来源 A"},
                {"title": f"Result 2 for {query}", "content": "示例内容 2", "source": "来源 B"}
            ]
            self.research_state["query_results"].extend(results)

        return self.research_state["query_results"]

    def extract_concepts(self):
        """从检索的信息中提取关键概念。"""
        # 从检索结果构建上下文
        context = self._build_retrieval_context()

        # 使用基于架构的提示进行概念提取
        concept_prompt = f"""
        任务：从以下研究信息中提取关键概念。

        研究主题：{self.research_state["topic"]}

        信息来源：
        {context}

        过程：
        1. 识别多个来源中提到的关键概念
        2. 对于每个概念，提取相关细节和定义
        3. 注意概念描述方式的变化或分歧
        4. 为每个概念分配相关性分数（1-10）

        格式：
        概念：[概念名称 1]
        定义：[综合定义]
        关键属性：
        - [属性 1]
        - [属性 2]
        来源变化：
        - [来源 A]：[此来源如何描述它]
        - [来源 B]：[此来源如何描述它]
        相关性分数：[1-10]

        概念：[概念名称 2]
        ...
        """

        extraction_results = self.llm.generate(concept_prompt)

        # 实践中，您将解析结构化输出
        # 对于本示例，我们将使用占位符概念
        self.research_state["extracted_concepts"] = {
            "concept1": {
                "definition": "概念1的定义",
                "properties": ["属性1", "属性2"],
                "source_variations": {
                    "来源 A": "来自 A 的描述",
                    "来源 B": "来自 B 的描述"
                },
                "relevance": 8
            },
            "concept2": {
                "definition": "概念2的定义",
                "properties": ["属性1", "属性2"],
                "source_variations": {
                    "来源 A": "来自 A 的描述",
                    "来源 B": "来自 B 的描述"
                },
                "relevance": 7
            }
        }

        return self.research_state["extracted_concepts"]

    def _build_retrieval_context(self):
        """从检索结果构建上下文。"""
        if not self.research_state["query_results"]:
            return "尚未检索到信息。"

        # 包括检索信息的样本
        # 实践中，您可能需要为令牌限制进行总结或选择
        context = ""
        for i, result in enumerate(self.research_state["query_results"][:5]):
            context += f"来源 {i+1}：{result['title']}\n"
            context += f"内容：{result['content'][:200]}...\n"
            context += f"来源：{result['source']}\n\n"

        return context

    def analyze_relationships(self):
        """分析提取的概念之间的关系。"""
        if not self.research_state["extracted_concepts"]:
            return "尚未提取概念。"

        # 获取概念名称列表
        concepts = list(self.research_state["extracted_concepts"].keys())

        # 使用比较矩阵模板进行关系分析
        relationship_prompt = f"""
        任务：分析研究主题中关键概念之间的关系。

        研究主题：{self.research_state["topic"]}

        要分析的概念：
        {", ".join(concepts)}

        过程：
        1. 创建所有概念之间的关系矩阵
        2. 对于每对，确定关系类型
        3. 注意每种关系的强度（1-5）
        4. 识别任何冲突或互补关系

        格式：
        关系矩阵：

        | 概念 | {" | ".join(concepts)} |
        |---------|{"-|" * len(concepts)}
        """

        # 为每个概念添加行
        for concept in concepts:
            relationship_prompt += f"| {concept} |"
            for other in concepts:
                if concept == other:
                    relationship_prompt += " X |"
                else:
                    relationship_prompt += " ? |"
            relationship_prompt += "\n"

        relationship_prompt += """

        详细关系：

        [概念 A] → [概念 B]
        类型：[因果/层次/相关等]
        强度：[1-5]
        描述：[它们如何相关的简要描述]

        [继续其他相关对...]
        """

        relationship_results = self.llm.generate(relationship_prompt)

        # 实践中，您将解析结构化输出
        # 对于本示例，我们将使用占位符关系
        self.research_state["concept_relationships"] = [
            {
                "source": "concept1",
                "target": "concept2",
                "type": "因果",
                "strength": 4,
                "description": "概念1直接影响概念2"
            }
        ]

        return self.research_state["concept_relationships"]

    def synthesize_research(self):
        """综合全面的研究摘要。"""
        # 确保我们已提取概念和关系
        if not self.research_state["extracted_concepts"]:
            self.extract_concepts()

        if not self.research_state["concept_relationships"]:
            self.analyze_relationships()

        # 从概念和关系构建上下文
        concepts_str = json.dumps(self.research_state["extracted_concepts"], indent=2)
        relationships_str = json.dumps(self.research_state["concept_relationships"], indent=2)

        synthesis_prompt = f"""
        任务：综合关于该主题的全面研究摘要。

        研究主题：{self.research_state["topic"]}

        关键概念：
        {concepts_str}

        概念关系：
        {relationships_str}

        过程：
        1. 创建一个整合关键概念的连贯叙述
        2. 突出各来源之间的共识领域
        3. 注意重要的分歧或矛盾
        4. 识别知识差距或需要进一步研究的领域
        5. 总结最重要的发现

        格式：
        # 研究综合：[主题]

        ## 关键发现
        [最重要见解的摘要]

        ## 概念整合
        [连接概念及其关系的叙述]

        ## 共识领域
        [来源一致的观点]

        ## 分歧领域
        [来源不一致或矛盾的观点]

        ## 知识差距
        [需要更多研究的领域]

        ## 结论
        [对知识现状的总体评估]
        """

        synthesis = self.llm.generate(synthesis_prompt)
        self.research_state["synthesis"] = synthesis

        # 提取知识差距（实践中，您将从综合中解析这些）
        self.research_state["knowledge_gaps"] = [
            "差距 1：需要对 X 进行更多研究",
            "差距 2：Y 和 Z 之间的关系不清楚"
        ]

        return synthesis

    def complete_research_cycle(self, topic):
        """从主题到综合运行完整的研究周期。"""
        # 设置研究主题并生成查询
        queries = self.set_research_topic(topic)

        # 检索信息
        self.retrieve_information(queries)

        # 提取和分析概念
        self.extract_concepts()
        self.analyze_relationships()

        # 综合研究发现
        synthesis = self.synthesize_research()

        return {
            "topic": topic,
            "synthesis": synthesis,
            "concepts": self.research_state["extracted_concepts"],
            "relationships": self.research_state["concept_relationships"],
            "knowledge_gaps": self.research_state["knowledge_gaps"]
        }
```

此实现演示了：
1. **结构化查询生成**以检索相关信息
2. **基于架构的概念提取**以识别关键思想
3. **关系分析**使用比较矩阵方法
4. **知识综合**将概念整合成连贯的叙述

## 应用领域：自适应学习系统

个性化学习需要跟踪用户知识状态并相应地调整内容：

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         自适应学习系统                                     │
│                                                                           │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│  │                 │     │                 │     │                 │      │
│  │   知识          │────►│   内容          │────►│   评估与        │      │
│  │   建模          │     │   选择          │     │   反馈          │      │
│  │                 │     │                 │     │                 │      │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│  ┌─────────────┐         ┌─────────────┐         ┌───────────────┐        │
│  │             │         │             │         │               │        │
│  │   用户模型  │         │   自适应    │         │   误解        │        │
│  │   架构      │         │   挑战      │         │   检测        │        │
│  │             │         │   模板      │         │               │        │
│  └─────────────┘         └─────────────┘         └───────────────┘        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 实现：个性化导师

```python
class PersonalizedTutor:
    """基于用户知识个性化内容的自适应学习系统。"""

    def __init__(self, llm_service):
        self.llm = llm_service
        self.learning_state = {
            "subject": "",
            "user_profile": {
                "name": "",
                "skill_level": "",  # beginner, intermediate, advanced
                "learning_style": "",  # visual, auditory, kinesthetic, etc.
                "known_concepts": [],
                "struggling_concepts": [],
                "mastered_concepts": []
            },
            "domain_model": {
                "concepts": {},
                "concept_dependencies": []
            },
            "session_history": [],
            "current_concept": "",
            "next_concepts": []
        }

    def initialize_user_profile(self, name, subject, initial_assessment=None):
        """初始化用户配置文件和知识状态。"""
        self.learning_state["subject"] = subject
        self.learning_state["user_profile"]["name"] = name

        if initial_assessment:
            # 解析评估结果
            self._parse_assessment(initial_assessment)
        else:
            # 生成初始评估
            self._generate_initial_assessment()

        # 初始化领域模型
        self._initialize_domain_model()

        return self.learning_state["user_profile"]

    def _parse_assessment(self, assessment_results):
        """解析初始评估的结果。"""
        # 实践中，这将解析实际的评估数据
        # 对于本示例，我们将使用占位符数据
        self.learning_state["user_profile"]["skill_level"] = "中级"
        self.learning_state["user_profile"]["learning_style"] = "视觉"
        self.learning_state["user_profile"]["known_concepts"] = ["concept1", "concept2"]
        self.learning_state["user_profile"]["struggling_concepts"] = ["concept3"]
        self.learning_state["user_profile"]["mastered_concepts"] = []

    def _generate_initial_assessment(self):
        """生成用户知识的初始评估。"""
        # 在实际实现中，这将生成问题以评估用户知识
        # 为简单起见，我们将使用占位符数据
        self.learning_state["user_profile"]["skill_level"] = "初学者"
        self.learning_state["user_profile"]["learning_style"] = "视觉"
        self.learning_state["user_profile"]["known_concepts"] = []
        self.learning_state["user_profile"]["struggling_concepts"] = []
        self.learning_state["user_profile"]["mastered_concepts"] = []

    def _initialize_domain_model(self):
        """初始化学科的领域模型。"""
        # 使用基于架构的提示来建模领域
        domain_prompt = f"""
        任务：为学科创建结构化知识模型：{self.learning_state["subject"]}

        过程：
        1. 识别该学科中的核心概念
        2. 对于每个概念，提供简要定义
        3. 指定每个概念的先决条件
        4. 识别常见误解
        5. 确定适当的难度级别

        格式：
        概念：[概念名称 1]
        定义：[简要定义]
        先决条件：[先决条件概念列表，如果有]
        误解：[常见误解]
        难度：[初学者/中级/高级]

        概念：[概念名称 2]
        ...

        依赖关系图：
        [概念 A] → [概念 B]（表示 B 依赖于理解 A）
        [概念 B] → [概念 C, 概念 D]
        ...
        """

        domain_model = self.llm.generate(domain_prompt)

        # 实践中，您将解析结构化输出
        # 对于本示例，我们将使用占位符数据
        self.learning_state["domain_model"]["concepts"] = {
            "concept1": {
                "definition": "概念1的定义",
                "prerequisites": [],
                "misconceptions": ["常见误解 1"],
                "difficulty": "初学者"
            },
            "concept2": {
                "definition": "概念2的定义",
                "prerequisites": ["concept1"],
                "misconceptions": ["常见误解 2"],
                "difficulty": "初学者"
            },
            "concept3": {
                "definition": "概念3的定义",
                "prerequisites": ["concept1", "concept2"],
                "misconceptions": ["常见误解 3"],
                "difficulty": "中级"
            }
        }

        self.learning_state["domain_model"]["concept_dependencies"] = [
            {"source": "concept1", "target": "concept2"},
            {"source": "concept1", "target": "concept3"},
            {"source": "concept2", "target": "concept3"}
        ]

        return self.learning_state["domain_model"]

    def select_next_concept(self):
        """根据用户状态选择下一个要教的概念。"""
        # 从用户配置文件和领域模型构建上下文
        user_profile = self.learning_state["user_profile"]
        domain_model = self.learning_state["domain_model"]

        # 使用上下文感知提示选择下一个概念
        selection_prompt = f"""
        任务：选择最合适的下一个要教的概念。

        用户配置文件：
        姓名：{user_profile["name"]}
        技能水平：{user_profile["skill_level"]}
        学习风格：{user_profile["learning_style"]}
        已知概念：{", ".join(user_profile["known_concepts"])}
        困难概念：{", ".join(user_profile["struggling_concepts"])}
        掌握概念：{", ".join(user_profile["mastered_concepts"])}

        领域模型：
        {json.dumps(domain_model["concepts"], indent=2)}

        概念依赖关系：
        {json.dumps(domain_model["concept_dependencies"], indent=2)}

        过程：
        1. 识别满足先决条件的概念
        2. 考虑用户的技能水平和困难概念
        3. 优先考虑建立在掌握内容基础上的概念
        4. 避免对当前状态来说过于高级的概念

        格式：
        选定概念：[概念名称]

        理由：
        [为什么此概念合适的解释]

        备选概念：
        1. [备选 1]：[简要原因]
        2. [备选 2]：[简要原因]
        """

        selection_result = self.llm.generate(selection_prompt)

        # 实践中，您将从输出中解析概念
        # 对于本示例，我们将使用占位符
        selected_concept = "concept2"
        self.learning_state["current_concept"] = selected_concept

        return selected_concept

    def generate_learning_content(self):
        """为当前概念生成个性化学习内容。"""
        # 确保我们有当前概念
        if not self.learning_state["current_concept"]:
            self.select_next_concept()

        current_concept = self.learning_state["current_concept"]
        concept_data = self.learning_state["domain_model"]["concepts"][current_concept]
        user_profile = self.learning_state["user_profile"]

        # 使用自适应模板生成个性化内容
        content_prompt = f"""
        任务：为概念创建个性化学习内容：{current_concept}

        用户配置文件：
        姓名：{user_profile["name"]}
        技能水平：{user_profile["skill_level"]}
        学习风格：{user_profile["learning_style"]}
        已知概念：{", ".join(user_profile["known_concepts"])}

        概念信息：
        定义：{concept_data["definition"]}
        常见误解：{", ".join(concept_data["misconceptions"])}

        过程：
        1. 使解释适应用户的技能水平
        2. 使用建立在已知概念基础上的示例
        3. 明确解决常见误解
        4. 根据用户的学习风格定制呈现方式
        5. 包括练习问题以巩固理解

        格式：
        # 学习模块：{current_concept}

        ## 引言
        [适合技能水平的简短、引人入胜的介绍]

        ## 核心解释
        [主要解释，适应学习风格]

        ## 示例
        [建立在已知概念基础上的示例]

        ## 常见误解
        [直接解决误解]

        ## 练习问题
        1. [问题 1]
        2. [问题 2]
        3. [问题 3]

        ## 总结
        [关键要点的简要回顾]
        """

        learning_content = self.llm.generate(content_prompt)

        # 添加到会话历史
        self.learning_state["session_history"].append({
            "concept": current_concept,
            "content": learning_content,
            "timestamp": time.time()
        })

        return learning_content

    def process_user_response(self, concept, user_response):
        """处理和评估用户对练习问题的响应。"""
        # 从概念和领域模型构建上下文
        concept_data = self.learning_state["domain_model"]["concepts"][concept]

        # 使用专门的提示进行响应评估
        eval_prompt = f"""
        任务：根据用户的响应评估其理解。

        概念：{concept}
        定义：{concept_data["definition"]}
        常见误解：{", ".join(concept_data["misconceptions"])}

        用户响应：
        {user_response}

        过程：
        1. 评估响应的准确性
        2. 识别存在的任何误解
        3. 确定理解水平
        4. 生成建设性反馈
        5. 如果需要，创建后续问题

        格式：
        理解水平：[完全/部分/最低]

        优势：
        - [用户正确理解的内容]

        差距：
        - [用户遗漏或误解的内容]

        检测到的误解：
        - [识别的任何具体误解]

        反馈：
        [建设性、鼓励性的反馈]

        后续问题：
        1. [解决特定差距的问题]
        2. [确认理解的问题]
        """

        evaluation = self.llm.generate(eval_prompt)

        # 根据评估更新用户配置文件
        # 实践中，您将更仔细地解析评估
        if "理解水平：完全" in evaluation:
            if concept in self.learning_state["user_profile"]["struggling_concepts"]:
                self.learning_state["user_profile"]["struggling_concepts"].remove(concept)
            if concept not in self.learning_state["user_profile"]["mastered_concepts"]:
                self.learning_state["user_profile"]["mastered_concepts"].append(concept)
        elif "理解水平：最低" in evaluation:
            if concept not in self.learning_state["user_profile"]["struggling_concepts"]:
                self.learning_state["user_profile"]["struggling_concepts"].append(concept)

        # 确保概念在已知概念中
        if concept not in self.learning_state["user_profile"]["known_concepts"]:
            self.learning_state["user_profile"]["known_concepts"].append(concept)

        return evaluation

    def run_learning_session(self, num_concepts=3):
        """运行涵盖多个概念的完整学习会话。"""
        session_results = []

        for i in range(num_concepts):
            # 选择下一个概念
            concept = self.select_next_concept()

            # 生成学习内容
            content = self.generate_learning_content()

            # 在实际应用中，您将在此处收集和处理用户响应
            # 对于本示例，我们将模拟用户响应
            simulated_response = f"对 {concept} 的模拟响应"
            evaluation = self.process_user_response(concept, simulated_response)

            session_results.append({
                "concept": concept,
                "content": content,
                "evaluation": evaluation
            })

        return {
            "user_profile": self.learning_state["user_profile"],
            "concepts_covered": [r["concept"] for r in session_results],
            "session_results": session_results
        }
```

此实现演示了：
1. **用户知识建模**使用基于架构的方法
2. **自适应内容选择**基于先决条件和用户状态
3. **个性化内容生成**根据学习风格和知识定制
4. **响应评估**具有误解检测

## 高级应用的关键模式

在这些不同的应用中，我们可以识别出增强上下文工程有效性的共同模式：

```
┌───────────────────────────────────────────────────────────────────┐
│ 高级上下文工程模式                                                 │
├───────────────────────────────────────────────────────────────────┤
│ ◆ 状态管理：跨交互跟踪复杂状态                                     │
│ ◆ 渐进式上下文：逐步构建上下文                                     │
│ ◆ 验证循环：质量和准确性的自我检查                                 │
│ ◆ 结构化架构：系统地组织信息                                       │
│ ◆ 模板程序：用于特定任务的可重用提示模式                           │
│ ◆ 个性化：适应用户需求和上下文                                     │
│ ◆ 多步处理：将复杂任务分解为阶段                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 测量应用性能

与更简单的上下文结构一样，测量对于高级应用仍然至关重要：

```
┌───────────────────────────────────────────────────────────────────┐
│ 高级应用的测量维度                                                 │
├──────────────────────────────┬────────────────────────────────────┤
│ 维度                         │ 指标                               │
├──────────────────────────────┼────────────────────────────────────┤
│ 端到端质量                   │ 准确性、正确性、连贯性             │
├──────────────────────────────┼────────────────────────────────────┤
│ 效率                         │ 总令牌数、完成时间                 │
├──────────────────────────────┼────────────────────────────────────┤
│ 鲁棒性                       │ 错误恢复率、边缘情况               │
│                              │ 处理                               │
├──────────────────────────────┼────────────────────────────────────┤
│ 用户满意度                   │ 相关性、个性化准确性               │
├──────────────────────────────┼────────────────────────────────────┤
│ 自我改进                     │ 随时间减少错误                     │
└──────────────────────────────┴────────────────────────────────────┘
```

## 关键要点

1. **高级应用**建立在上下文工程的基本原则之上
2. **状态管理**对于复杂应用变得越来越重要
3. **基于架构的方法**为处理复杂信息提供结构
4. **多步处理**将复杂任务分解为可管理的部分
5. **自我验证**提高可靠性和准确性
6. **测量仍然至关重要**用于优化应用性能

## 实践练习

1. 使用附加功能扩展其中一个示例实现
2. 在您的领域中实现应用的简化版本
3. 为您使用的特定类型信息设计架构
4. 为您的应用创建测量框架

## 下一步

在下一节中，我们将探索提示编程——一种将编程结构与提示灵活性相结合的强大方法，以创建更复杂的上下文工程解决方案。

[继续到 07_prompt_programming.md →](07_prompt_programming.md)

---

## 深入探讨：工程权衡

高级应用需要平衡几个竞争因素：

```
┌──────────────────────────────────────────────────────────────────┐
│ 上下文工程权衡                                                    │
├──────────────────────────────────────────────────────────────────┤
│ ◆ 复杂性 vs. 可维护性                                             │
│   更复杂的系统可能更难调试和维护                                   │
│                                                                  │
│ ◆ 令牌使用 vs. 质量                                               │
│   更多上下文通常提高质量但增加成本                                 │
│                                                                  │
│ ◆ 专用 vs. 通用                                                   │
│   专用组件效果更好但可重用性较差                                   │
│                                                                  │
│ ◆ 严格结构 vs. 灵活性                                             │
│   结构化架构提高一致性但降低适应性                                 │
└──────────────────────────────────────────────────────────────────┘
```

为您的特定应用找到正确的平衡是高级上下文工程的关键部分。
