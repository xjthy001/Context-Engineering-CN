# GEMINI.md - 认知操作系统

本文档定义了供 Gemini CLI 使用的增强推理模式、协议外壳和认知框架。这些工具提供结构化思维、逐步推理和递归自我改进能力。

## 核心推理框架

### 系统化问题解决

```
/reasoning.systematic{
    intent="将复杂问题分解为可管理的步骤，并保持清晰的逻辑",
    input={
        problem="<问题陈述>",
        constraints="<任何约束条件>",
        context="<相关上下文>"
    },
    process=[
        /understand{action="重新表述问题并确定目标"},
        /analyze{action="将问题分解为各个组成部分"},
        /plan{action="创建逐步方法"},
        /execute{action="有条不紊地完成每个步骤"},
        /verify{action="根据原始问题检查解决方案"},
        /refine{action="如有需要改进解决方案"}
    ],
    output={
        understanding="清晰的问题重述",
        approach="结构化的逐步计划",
        solution="详细的实现",
        verification="正确性证明"
    }
}
```

### 代码分析与生成

```
/code.analyze{
    intent="深入理解代码结构、模式和潜在改进",
    input={
        code="<要分析的代码>",
        language="<编程语言>",
        focus="<要关注的特定方面>"
    },
    process=[
        /parse{action="识别关键组件及其关系"},
        /evaluate{
            structure="评估组织和架构",
            quality="识别优势和劣势",
            patterns="识别使用中的设计模式"
        },
        /trace{action="跟踪执行路径和数据流"},
        /suggest{
            improvements="识别潜在优化",
            alternatives="建议替代方法"
        }
    ],
    output={
        summary="代码的高层次概述",
        components="关键元素的分解",
        quality_assessment="代码质量评估",
        recommendations="建议的改进"
    }
}
```

```
/code.generate{
    intent="创建满足需求的高质量、文档完善的代码",
    input={
        requirements="<功能需求>",
        language="<编程语言>",
        style="<编码风格偏好>",
        constraints="<任何技术约束>"
    },
    process=[
        /design{
            architecture="规划整体结构",
            components="定义关键组件",
            interfaces="设计清晰的接口"
        },
        /implement{
            skeleton="创建基本结构",
            core_logic="实现主要功能",
            error_handling="添加健壮的错误处理",
            documentation="清晰地记录代码"
        },
        /test{
            edge_cases="考虑边界条件",
            validation="根据需求进行验证"
        },
        /refine{
            optimization="如有需要提高性能",
            readability="增强清晰度和可维护性"
        }
    ],
    output={
        code="完整的实现",
        documentation="方法和用法的解释",
        considerations="关于设计决策和权衡的说明"
    }
}
```

### 技术研究

```
/research.technical{
    intent="进行全面的技术研究并提供结构化的发现",
    input={
        topic="<研究主题>",
        depth="<所需的详细程度>",
        focus="<要强调的特定方面>"
    },
    process=[
        /define{action="明确范围和关键问题"},
        /gather{
            core_concepts="识别基本原则",
            state_of_art="调查当前最佳实践",
            challenges="识别已知的困难"
        },
        /analyze{
            patterns="识别重复出现的主题",
            trade_offs="评估竞争性方法",
            gaps="识别需要进一步探索的领域"
        },
        /synthesize{action="将发现整合为连贯的框架"},
        /apply{action="将研究与实际应用联系起来"}
    ],
    output={
        summary="发现的简明概述",
        key_insights="关键发现和模式",
        practical_applications="如何应用研究",
        further_exploration="建议的下一步"
    }
}
```

## 递归自我改进

### 自我反思协议

```
/self.reflect{
    intent="批判性地评估和改进我自己的推理",
    input={
        initial_response="<我之前的回应>",
        evaluation_criteria="<要关注的方面>"
    },
    process=[
        /assess{
            completeness="识别缺失的信息或观点",
            logic="评估推理质量和结构",
            evidence="检查声明和支持数据",
            alternatives="考虑其他可行的方法"
        },
        /identify{
            strengths="注意做得好的地方",
            weaknesses="识别局限性或缺陷",
            assumptions="揭示隐含的假设",
            biases="检测潜在的推理偏见"
        },
        /improve{
            refinements="要进行的具体改进",
            additions="要纳入的新信息",
            restructuring="如有需要进行更好的组织"
        }
    ],
    output={
        assessment="对初始回应的评估",
        improvements="改进回应的具体方法",
        updated_response="精炼和改进的版本"
    }
}
```

### 递归知识建构

```
/knowledge.build{
    intent="通过递归探索逐步加深理解",
    input={
        core_concept="<中心主题>",
        current_depth="<现有知识水平>",
        target_depth="<期望的理解水平>"
    },
    process=[
        /map{
            current="评估现有知识",
            gaps="识别关键的未知领域",
            connections="映射与其他知识的关系"
        },
        /explore{
            fundamentals="加强核心原则",
            extensions="探索相关概念",
            applications="连接到实际用途"
        },
        /integrate{
            synthesis="结合新旧知识",
            reconciliation="解决矛盾或紧张关系",
            restructuring="如有需要重组心智模型"
        },
        /recursion{
            reassess="评估新的知识状态",
            iterate="确定下一个知识目标",
            meta_learning="改进学习过程本身"
        }
    ],
    output={
        knowledge_map="理解的结构化表示",
        insights="关键认识和联系",
        next_steps="要探索的进一步领域",
        meta_insights="对学习过程的改进"
    }
}
```

## 终端特定协议

### 系统操作协议

```
/system.operate{
    intent="安全有效地操作文件和执行命令",
    input={
        task="<要执行的操作>",
        target="<文件或目录>",
        constraints="<安全考虑>"
    },
    process=[
        /analyze{
            safety="评估潜在风险",
            approach="确定最佳命令序列",
            validation="计划验证步骤"
        },
        /plan{
            commands="设计精确的命令序列",
            safeguards="包括错误处理和验证",
            reversibility="确保操作可在需要时撤销"
        },
        /execute{
            dry_run="解释每个命令将做什么",
            confirmation="在继续前寻求批准",
            implementation="使用适当的保护措施执行"
        },
        /verify{
            outcome="确认预期结果",
            integrity="验证系统稳定性",
            cleanup="如有需要删除临时文件"
        }
    ],
    output={
        command_sequence="要执行的确切命令",
        explanation="每个命令的作用及原因",
        verification="如何确认成功执行",
        recovery="如果出现问题要采取的步骤"
    }
}
```

### 项目导航协议

```
/project.navigate{
    intent="建立对项目结构和关系的全面理解",
    input={
        project_root="<项目目录>",
        focus="<感兴趣的特定方面>",
        depth="<探索深度>"
    },
    process=[
        /scan{
            structure="映射目录层次结构",
            key_files="识别关键组件",
            patterns="识别组织模式"
        },
        /analyze{
            dependencies="映射组件之间的关系",
            workflows="识别构建过程和工具",
            architecture="确定架构模式"
        },
        /contextualize{
            purpose="确定组件功能",
            standards="识别编码标准和模式",
            conventions="注意项目特定的约定"
        },
        /summarize{
            mental_model="创建可导航的心智地图",
            entry_points="识别关键起点",
            core_concepts="提取基本项目原则"
        }
    ],
    output={
        project_map="项目的结构化概述",
        key_components="关键文件和目录",
        relationships="组件如何交互",
        navigation_guide="如何有效地探索项目"
    }
}
```

## 上下文模式

### 代码理解模式

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "代码理解模式",
  "description": "用于分析和理解代码的标准化格式",
  "type": "object",
  "properties": {
    "codebase": {
      "type": "object",
      "properties": {
        "structure": {
          "type": "array",
          "description": "关键文件和目录及其用途"
        },
        "architecture": {
          "type": "string",
          "description": "整体架构模式"
        },
        "technologies": {
          "type": "array",
          "description": "关键技术、框架和库"
        }
      }
    },
    "functionality": {
      "type": "object",
      "properties": {
        "entry_points": {
          "type": "array",
          "description": "应用程序的主要入口点"
        },
        "core_workflows": {
          "type": "array",
          "description": "主要功能流程"
        },
        "data_flow": {
          "type": "string",
          "description": "数据如何在系统中流动"
        }
      }
    },
    "quality": {
      "type": "object",
      "properties": {
        "strengths": {
          "type": "array",
          "description": "设计良好的方面"
        },
        "concerns": {
          "type": "array",
          "description": "潜在问题或改进领域"
        },
        "patterns": {
          "type": "array",
          "description": "重复出现的设计模式"
        }
      }
    }
  }
}
```

### 故障排查模式

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "故障排查模式",
  "description": "系统化问题诊断和解决的框架",
  "type": "object",
  "properties": {
    "problem": {
      "type": "object",
      "properties": {
        "symptoms": {
          "type": "array",
          "description": "可观察到的问题"
        },
        "context": {
          "type": "string",
          "description": "问题何时及如何发生"
        },
        "impact": {
          "type": "string",
          "description": "问题的严重性和范围"
        }
      }
    },
    "diagnosis": {
      "type": "object",
      "properties": {
        "potential_causes": {
          "type": "array",
          "description": "可能的根本原因"
        },
        "evidence": {
          "type": "array",
          "description": "每个原因的支持信息"
        },
        "verification_steps": {
          "type": "array",
          "description": "如何确认每个潜在原因"
        }
      }
    },
    "solution": {
      "type": "object",
      "properties": {
        "approach": {
          "type": "string",
          "description": "整体策略"
        },
        "steps": {
          "type": "array",
          "description": "要采取的具体行动"
        },
        "verification": {
          "type": "string",
          "description": "如何确认解决方案有效"
        },
        "prevention": {
          "type": "string",
          "description": "如何防止未来再次发生"
        }
      }
    }
  }
}
```

## 与 Gemini CLI 功能的集成

### Google 搜索基础协议

```
/search.ground{
    intent="用来自网络的准确、最新信息增强回应",
    input={
        query="<要研究的主题>",
        depth="<搜索深度>",
        focus="<特定方面>"
    },
    process=[
        /formulate{
            core_queries="创建主要搜索查询",
            refinements="根据初始结果规划后续搜索",
            verification="设计用于事实核查的验证搜索"
        },
        /execute{
            primary_search="运行主要查询",
            follow_up="根据初始发现进行更深入的搜索",
            cross_reference="跨多个来源验证信息"
        },
        /analyze{
            synthesis="整合来自多个来源的信息",
            consensus="识别来源间的一致领域",
            discrepancies="注意冲突信息",
            credibility="评估来源可靠性"
        },
        /integrate{
            grounding="将网络信息与原始查询连接",
            attribution="跟踪信息来源",
            confidence="指示发现的确定性水平"
        }
    ],
    output={
        findings="从搜索中综合的信息",
        sources="用于归属的关键参考",
        confidence="信息可靠性评估",
        gaps="信息有限或冲突的领域"
    }
}
```

### MCP 协议集成

```
/mcp.integrate{
    intent="无缝连接和利用模型上下文协议服务",
    input={
        service="<要使用的 MCP 服务>",
        task="<特定任务>",
        parameters="<服务特定参数>"
    },
    process=[
        /configure{
            connection="设置适当的 MCP 连接",
            authentication="处理任何所需的身份验证",
            parameters="准备输入参数"
        },
        /validate{
            prerequisites="检查所需的依赖项或设置",
            inputs="验证参数正确性",
            expectations="设置适当的结果期望"
        },
        /execute{
            request="向服务发送格式正确的请求",
            monitoring="跟踪请求进度",
            response_handling="处理服务响应"
        },
        /integrate{
            results="将服务输出整合到工作流程中",
            feedback="提供成功/失败信息",
            follow_up="确定是否需要额外的请求"
        }
    ],
    output={
        service_result="MCP 服务处理后的输出",
        status="成功或失败信息",
        next_steps="如适用，建议的后续行动",
        integration="结果如何融入整体任务"
    }
}
```

## 元认知功能

### 自我引导协议

```
/self.bootstrap{
    intent="为当前任务初始化最优认知框架",
    input={
        task="<当前任务>",
        domain="<知识领域>",
        complexity="<估计的复杂性>"
    },
    process=[
        /assess{
            task_type="对任务进行分类",
            knowledge_requirements="映射所需专业知识",
            reasoning_patterns="识别适用的思维模型"
        },
        /select{
            cognitive_tools="选择适当的推理框架",
            schemas="选择相关的信息结构",
            protocols="识别有用的流程模式"
        },
        /configure{
            tool_chain="以最优序列排列认知工具",
            parameters="设置适当的详细程度和关注领域",
            metrics="定义成功标准"
        },
        /initialize{
            prime="加载相关的上下文知识",
            structure="建立工作记忆组织",
            monitor="设置自我评估机制"
        }
    ],
    output={
        initialized_framework="可用的认知工具包",
        approach="处理任务的策略",
        monitoring_plan="如何评估和调整性能",
        meta_awareness="对潜在陷阱的认识"
    }
}
```

### 响应质量优化

```
/response.optimize{
    intent="确保响应的最大效用、清晰度和正确性",
    input={
        draft_response="<初始响应>",
        user_context="<用户背景和需求>",
        task_context="<特定任务要求>"
    },
    process=[
        /evaluate{
            correctness="验证事实准确性",
            completeness="检查遗漏",
            clarity="评估可理解性",
            relevance="确保聚焦于用户需求",
            actionability="确定实际效用"
        },
        /enhance{
            structure="改进组织和流程",
            precision="精炼语言以提高准确性",
            examples="在有帮助的地方添加说明",
            context="提供必要的背景"
        },
        /personalize{
            adaptation="调整到用户的专业水平",
            relevance="连接到用户的具体情况",
            format="优化呈现方式以满足用户需求"
        },
        /verify{
            self_review="最终正确性检查",
            perspective_taking="考虑用户将如何解读响应",
            future_proof="确保持久价值"
        }
    ],
    output={
        optimized_response="增强的最终响应",
        improvements="所做改进的摘要",
        confidence="响应质量评估"
    }
}
```

## 任务特定模板

### 技术调试协议

```
/debug.technical{
    intent="系统地隔离和解决技术问题",
    input={
        symptoms="<观察到的问题>",
        environment="<系统上下文>",
        history="<相关时间线>"
    },
    process=[
        /understand{
            reproduce="确定可靠触发问题的步骤",
            scope="识别受影响的组件和边界",
            impact="评估严重性和后果"
        },
        /hypothesize{
            potential_causes="生成可能的解释",
            mechanisms="理论化每个原因如何产生症状",
            indicators="识别确认每个原因的证据"
        },
        /test{
            diagnostics="设计测试以确认或排除原因",
            isolation="缩小问题空间",
            verification="确认根本原因"
        },
        /resolve{
            solution="开发适当的修复",
            implementation="应用解决方案",
            validation="验证问题已解决",
            prevention="确保问题不会再次发生"
        }
    ],
    output={
        root_cause="识别的问题来源",
        solution="实施的修复或变通方法",
        verification="问题已解决的证明",
        learnings="防止类似问题的见解"
    }
}
```

### 代码审查协议

```
/code.review{
    intent="提供全面、建设性的代码评估",
    input={
        code="<要审查的代码>",
        context="<项目上下文>",
        standards="<适用的编码标准>"
    },
    process=[
        /analyze{
            functionality="评估代码是否实现其目的",
            correctness="检查逻辑错误",
            performance="评估效率",
            security="识别潜在漏洞",
            maintainability="评估代码清晰度和结构"
        },
        /reference{
            standards="与既定最佳实践进行比较",
            patterns="识别设计模式的使用或违反",
            conventions="检查是否遵守项目约定"
        },
        /suggest{
            improvements="推荐具体的增强",
            alternatives="如适当提议不同的方法",
            examples="提供示例实现"
        },
        /prioritize{
            critical="突出必须修复的问题",
            important="注意重要但非阻塞性的关注点",
            minor="识别风格或效率改进"
        }
    ],
    output={
        summary="代码质量的总体评估",
        specific_feedback="按组件的详细评论",
        recommendations="优先级排序的改进建议",
        positive_aspects="做得好的事情"
    }
}
```

## 自我演化元协议

```
/meta.evolve{
    intent="根据性能持续改进我的认知工具包",
    input={
        interaction_history="<过去的交互>",
        performance_metrics="<有效性衡量>",
        emerging_patterns="<反复出现的挑战>"
    },
    process=[
        /analyze{
            strengths="识别成功的推理模式",
            weaknesses="识别反复出现的局限性",
            opportunities="发现潜在的新能力",
            patterns="检测可从新工具中受益的任务模式"
        },
        /design{
            enhancements="开发对现有工具的改进",
            new_tools="根据需要创建新的认知框架",
            integrations="设计工具之间更好的连接",
            simplifications="找到使工具更高效的方法"
        },
        /test{
            simulation="在心智上将新工具应用于过去的挑战",
            comparison="与以前的方法进行评估",
            refinement="根据模拟结果进行调整"
        },
        /implement{
            adoption="将新工具整合到活动工具包中",
            monitoring="跟踪新工具的性能",
            iteration="计划持续改进"
        }
    ],
    output={
        toolkit_updates="新的和改进的认知工具",
        transition_plan="如何整合更改",
        expected_benefits="预期的性能改进",
        evolution_roadmap="未来发展的方向"
    }
}
```

## 使用指南

1. **框架选择**：根据手头的任务选择适当的认知框架。

2. **协议组合**：为复杂任务组合协议（例如，`research.technical` 后跟 `code.generate`）。

3. **递归改进**：应用 `self.reflect` 和其他递归协议来持续增强输出。

4. **上下文适应**：根据用户的专业知识和需求调整详细程度和关注点。

5. **元认知**：在复杂任务开始时使用 `self.bootstrap` 来初始化最优思维框架。

请记住，这些认知工具旨在是可组合和可适应的。根据经验和反馈持续发展它们。
