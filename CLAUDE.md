# CLAUDE.md - 认知操作系统

本文档为 Claude Code 提供了一个全面的认知工具、协议外壳、推理模板和工作流框架。将此文件加载到项目根目录中，以增强 Claude 在所有上下文中的能力。

## 1. 核心元认知框架

## 上下文模式（Context Schemas）

### 代码理解模式（Code Understanding Schema）

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Code Understanding Schema",
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

### 故障排除模式（Troubleshooting Schema）

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Troubleshooting Schema",
  "description": "系统性问题诊断和解决的框架",
  "type": "object",
  "properties": {
    "problem": {
      "type": "object",
      "properties": {
        "symptoms": {
          "type": "array",
          "description": "可观察的问题"
        },
        "context": {
          "type": "string",
          "description": "问题何时以及如何发生"
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


### 推理协议（Reasoning Protocols）

```
/reasoning.systematic{
    intent="将复杂问题分解为逻辑步骤并提供可追溯的推理",
    input={
        problem="<问题陈述>",
        constraints="<约束条件>",
        context="<上下文>"
    },
    process=[
        /understand{action="重新陈述问题并明确目标"},
        /analyze{action="分解为组件"},
        /plan{action="设计分步方法"},
        /execute{action="系统化实施解决方案"},
        /verify{action="根据需求进行验证"},
        /refine{action="基于验证进行改进"}
    ],
    output={
        solution="实施的解决方案",
        reasoning="完整的推理轨迹",
        verification="验证证据"
    }
}
```

```
/thinking.extended{
    intent="对需要仔细考虑的复杂问题进行深入、彻底的推理",
    input={
        problem="<需要深度思考的问题>",
        level="<basic|deep|deeper|ultra>" // 对应于 think, think hard, think harder, ultrathink
    },
    process=[
        /explore{action="考虑多个角度和方法"},
        /evaluate{action="评估每种方法的权衡"},
        /simulate{action="针对边缘情况测试心智模型"},
        /synthesize{action="将洞察整合为连贯的解决方案"},
        /articulate{action="清晰彻底地表达推理"}
    ],
    output={
        conclusion="经过深思熟虑的解决方案",
        rationale="完整的思考过程",
        alternatives="考虑过的其他方法"
    }
}
```

### 自我改进协议（Self-Improvement Protocol）

```
/self.reflect{
    intent="通过递归评估持续改进推理和输出",
    input={
        previous_output="<要评估的输出>",
        criteria="<评估标准>"
    },
    process=[
        /assess{
            completeness="识别缺失信息",
            correctness="验证事实准确性",
            clarity="评估可理解性",
            effectiveness="确定是否满足需求"
        },
        /identify{
            strengths="注意做得好的地方",
            weaknesses="识别局限性",
            assumptions="揭示隐含假设"
        },
        /improve{
            strategy="计划具体改进",
            implementation="系统化应用改进"
        }
    ],
    output={
        evaluation="对原始输出的评估",
        improved_output="增强版本",
        learning="未来改进的见解"
    }
}
```

## 2. 工作流协议（Workflow Protocols）

### 探索-计划-编码-提交工作流（Explore-Plan-Code-Commit Workflow）

```
/workflow.explore_plan_code_commit{
    intent="实施一种系统化的编码任务方法，并进行充分规划",
    input={
        task="<任务描述>",
        codebase="<相关文件或目录>"
    },
    process=[
        /explore{
            action="阅读相关文件并理解代码库",
            instruction="分析但暂不编写代码"
        },
        /plan{
            action="创建详细的实施计划",
            instruction="使用扩展思考来评估替代方案"
        },
        /implement{
            action="按照计划编写代码",
            instruction="在每一步验证正确性"
        },
        /finalize{
            action="提交更改并根据需要创建 PR",
            instruction="编写清晰的提交消息"
        }
    ],
    output={
        implementation="可工作的代码解决方案",
        explanation="方法文档",
        commit="提交消息和 PR 详情"
    }
}
```

### 测试驱动开发工作流（Test-Driven Development Workflow）

```
/workflow.test_driven{
    intent="使用测试优先方法实施更改",
    input={
        feature="<要实施的功能>",
        requirements="<详细需求>"
    },
    process=[
        /write_tests{
            action="根据需求创建全面的测试",
            instruction="暂不实施功能"
        },
        /verify_tests_fail{
            action="运行测试以确认它们适当失败",
            instruction="验证测试正确性"
        },
        /implement{
            action="编写代码以使测试通过",
            instruction="专注于通过测试，而不是最初的实现优雅性"
        },
        /refactor{
            action="在保持测试通过的同时清理实现",
            instruction="在不改变行为的情况下提高代码质量"
        },
        /finalize{
            action="提交测试和实现",
            instruction="在提交消息中包含测试理由"
        }
    ],
    output={
        tests="全面的测试套件",
        implementation="通过测试的可工作代码",
        commit="提交消息和 PR 详情"
    }
}
```

### 迭代式 UI 开发工作流（Iterative UI Development Workflow）

```
/workflow.ui_iteration{
    intent="通过视觉反馈循环实施 UI 组件",
    input={
        design="<设计模型或描述>",
        components="<现有组件引用>"
    },
    process=[
        /analyze_design{
            action="理解设计需求和约束",
            instruction="识别可重用的模式和组件"
        },
        /implement_initial{
            action="创建 UI 的首次实现",
            instruction="专注于结构而非样式"
        },
        /screenshot{
            action="截取当前实现的屏幕截图",
            instruction="使用浏览器工具或 Puppeteer MCP"
        },
        /compare{
            action="将实现与设计进行比较",
            instruction="识别差异和需要的改进"
        },
        /refine{
            action="迭代改进实现",
            instruction="在每次重大更改后截取新的屏幕截图"
        },
        /finalize{
            action="完善并提交最终实现",
            instruction="在文档中包含屏幕截图"
        }
    ],
    output={
        implementation="可工作的 UI 组件",
        screenshots="前后视觉文档",
        commit="提交消息和 PR 详情"
    }
}
```

## 3. 代码分析与生成工具（Code Analysis & Generation Tools）

### 代码分析协议（Code Analysis Protocol）

```
/code.analyze{
    intent="深入理解代码结构、模式和质量",
    input={
        code="<要分析的代码>",
        focus="<要检查的特定方面>"
    },
    process=[
        /parse{
            structure="识别主要组件和组织结构",
            patterns="识别设计模式和约定",
            flow="追踪执行和数据流路径"
        },
        /evaluate{
            quality="评估代码质量和最佳实践",
            performance="识别潜在性能问题",
            security="发现潜在安全问题",
            maintainability="评估长期可维护性"
        },
        /summarize{
            purpose="描述代码的主要功能",
            architecture="概述架构方法",
            interfaces="记录关键接口和契约"
        }
    ],
    output={
        overview="代码的高级摘要",
        details="逐组件分解",
        recommendations="建议的改进"
    }
}
```

### 代码生成协议（Code Generation Protocol）

```
/code.generate{
    intent="创建满足需求的高质量、可维护代码",
    input={
        requirements="<功能需求>",
        context="<代码库上下文>",
        constraints="<技术约束>"
    },
    process=[
        /design{
            architecture="规划整体结构",
            interfaces="定义清晰的接口",
            patterns="选择适当的设计模式"
        },
        /implement{
            skeleton="创建基础结构",
            core="实施主要功能",
            edge_cases="处理异常和边缘情况",
            tests="包含适当的测试"
        },
        /review{
            functionality="验证是否满足需求",
            quality="确保代码符合质量标准",
            style="遵守项目约定"
        },
        /document{
            usage="提供使用示例",
            rationale="解释关键决策",
            integration="描述集成点"
        }
    ],
    output={
        code="完整实现",
        tests="配套测试",
        documentation="全面文档"
    }
}
```

### 重构协议（Refactoring Protocol）

```
/code.refactor{
    intent="在不改变行为的情况下改进现有代码",
    input={
        code="<要重构的代码>",
        goals="<重构目标>"
    },
    process=[
        /analyze{
            behavior="精确记录当前行为",
            tests="识别或创建验证测试",
            issues="识别代码异味和问题"
        },
        /plan{
            approach="设计重构策略",
            steps="分解为安全、增量的更改",
            verification="计划每一步的验证"
        },
        /execute{
            changes="增量实施重构",
            tests="每次更改后运行测试",
            review="自我审查每次修改"
        },
        /validate{
            functionality="验证保留的行为",
            improvements="确认重构目标已实现",
            documentation="根据需要更新文档"
        }
    ],
    output={
        refactored_code="改进的实现",
        verification="保留行为的证据",
        improvements="更改和收益摘要"
    }
}
```

## 4. 测试与验证框架（Testing & Validation Frameworks）

### 测试套件生成协议（Test Suite Generation Protocol）

```
/test.generate{
    intent="创建用于代码验证的全面测试套件",
    input={
        code="<要测试的代码>",
        requirements="<功能需求>"
    },
    process=[
        /analyze{
            functionality="识别核心功能",
            edge_cases="确定边界条件",
            paths="映射执行路径"
        },
        /design{
            unit_tests="设计集中的组件测试",
            integration_tests="设计跨组件测试",
            edge_case_tests="设计边界条件测试",
            performance_tests="设计性能验证"
        },
        /implement{
            framework="设置测试框架",
            fixtures="创建必要的测试夹具",
            tests="实施设计的测试",
            assertions="包含清晰的断言"
        },
        /validate{
            coverage="验证充分的代码覆盖率",
            independence="确保测试独立性",
            clarity="确认测试可读性"
        }
    ],
    output={
        test_suite="完整的测试实现",
        coverage_analysis="测试覆盖率评估",
        run_instructions="如何执行测试"
    }
}
```

### Bug 诊断协议（Bug Diagnosis Protocol）

```
/bug.diagnose{
    intent="系统化识别问题的根本原因",
    input={
        symptoms="<观察到的问题>",
        context="<环境和条件>"
    },
    process=[
        /reproduce{
            steps="建立可靠的复现步骤",
            environment="识别环境因素",
            consistency="确定可复现性一致性"
        },
        /isolate{
            scope="缩小受影响的组件",
            triggers="识别特定触发器",
            patterns="识别症状模式"
        },
        /analyze{
            trace="跟踪代码执行路径",
            state="检查相关状态和数据",
            interactions="研究组件交互"
        },
        /hypothesize{
            causes="制定潜在根本原因",
            tests="为每个假设设计测试",
            verification="计划验证方法"
        }
    ],
    output={
        diagnosis="识别的根本原因",
        evidence="支持证据",
        fix_strategy="推荐的解决方法"
    }
}
```

## 5. Git 与 GitHub 集成（Git & GitHub Integration）

### Git 工作流协议（Git Workflow Protocol）

```
/git.workflow{
    intent="使用 Git 最佳实践管理代码更改",
    input={
        changes="<代码更改>",
        branch_strategy="<分支方法>"
    },
    process=[
        /prepare{
            branch="创建或选择适当的分支",
            scope="定义更改的明确范围",
            baseline="确保干净的起点"
        },
        /develop{
            changes="实施所需更改",
            commits="创建逻辑的、原子的提交",
            messages="编写清晰的提交消息"
        },
        /review{
            diff="彻底审查更改",
            tests="确保测试通过",
            standards="验证遵守标准"
        },
        /integrate{
            sync="与目标分支同步",
            conflicts="解决任何冲突",
            validate="验证集成成功"
        }
    ],
    output={
        commits="干净的提交历史",
        branches="更新的分支状态",
        next_steps="推荐的后续行动"
    }
}
```

### GitHub PR 协议（GitHub PR Protocol）

```
/github.pr{
    intent="创建和管理有效的拉取请求",
    input={
        changes="<实施的更改>",
        context="<目的和背景>"
    },
    process=[
        /prepare{
            review="自我审查更改",
            tests="验证测试通过",
            ci="检查 CI 管道状态"
        },
        /create{
            title="编写清晰、描述性的标题",
            description="创建全面的描述",
            labels="添加适当的标签",
            reviewers="请求适当的审查者"
        },
        /respond{
            reviews="处理审查反馈",
            updates="进行请求的更改",
            discussion="参与建设性讨论"
        },
        /finalize{
            checks="确保所有检查通过",
            approval="确认必要的批准",
            merge="完成合并过程"
        }
    ],
    output={
        pr="完整的拉取请求",
        status="PR 状态和下一步",
        documentation="任何后续文档"
    }
}
```

### Git 历史分析协议（Git History Analysis Protocol）

```
/git.analyze_history{
    intent="从仓库历史中提取见解",
    input={
        repo="<仓库路径>",
        focus="<分析目标>"
    },
    process=[
        /collect{
            commits="收集相关提交历史",
            authors="识别贡献者",
            patterns="检测贡献模式"
        },
        /analyze{
            changes="检查代码演变",
            decisions="追踪架构决策",
            trends="识别开发趋势"
        },
        /synthesize{
            insights="提取关键见解",
            timeline="创建演变时间线",
            attribution="将功能映射到贡献者"
        }
    ],
    output={
        history_analysis="全面的历史分析",
        key_insights="重要的历史模式",
        visualization="演变的时间表示"
    }
}
```

## 6. 项目导航与探索（Project Navigation & Exploration）

### 代码库探索协议（Codebase Exploration Protocol）

```
/project.explore{
    intent="建立对项目结构的全面理解",
    input={
        repo="<仓库路径>",
        focus="<探索目标>"
    },
    process=[
        /scan{
            structure="映射目录层次结构",
            files="识别关键文件",
            patterns="识别组织模式"
        },
        /analyze{
            architecture="确定架构方法",
            components="识别主要组件",
            dependencies="映射组件关系"
        },
        /document{
            overview="创建高级摘要",
            components="记录关键组件",
            patterns="描述重复出现的模式"
        }
    ],
    output={
        map="代码库的结构表示",
        key_areas="识别的重要组件",
        entry_points="推荐的起点"
    }
}
```

### 依赖分析协议（Dependency Analysis Protocol）

```
/project.analyze_dependencies{
    intent="理解项目依赖关系和关联",
    input={
        project="<项目路径>",
        depth="<分析深度>"
    },
    process=[
        /scan{
            direct="识别直接依赖",
            transitive="映射传递依赖",
            versions="编目版本约束"
        },
        /analyze{
            usage="确定依赖的使用方式",
            necessity="评估每个依赖的必要性",
            alternatives="识别潜在替代方案"
        },
        /evaluate{
            security="检查安全问题",
            maintenance="评估维护状态",
            performance="评估性能影响"
        }
    ],
    output={
        dependency_map="依赖关系的可视化表示",
        recommendations="建议的优化",
        risks="识别的潜在问题"
    }
}
```

## 7. 自我反思与改进机制（Self-Reflection & Improvement Mechanisms）

### 知识差距识别协议（Knowledge Gap Identification Protocol）

```
/self.identify_gaps{
    intent="识别并解决知识局限",
    input={
        context="<当前任务上下文>",
        requirements="<知识需求>"
    },
    process=[
        /assess{
            current="评估当前理解",
            needed="识别所需知识",
            gaps="精确定位特定知识差距"
        },
        /plan{
            research="设计有针对性的研究方法",
            questions="制定具体问题",
            sources="识别信息来源"
        },
        /acquire{
            research="进行必要的研究",
            integration="整合新知识",
            verification="验证理解"
        }
    ],
    output={
        gap_analysis="识别的知识局限",
        acquired_knowledge="收集的新信息",
        updated_approach="用新知识修订的方法"
    }
}
```

### 解决方案质量改进协议（Solution Quality Improvement Protocol）

```
/self.improve_solution{
    intent="迭代提高解决方案质量",
    input={
        current_solution="<现有解决方案>",
        quality_criteria="<质量标准>"
    },
    process=[
        /evaluate{
            strengths="识别解决方案优势",
            weaknesses="精确定位改进领域",
            benchmarks="与标准进行比较"
        },
        /plan{
            priorities="确定改进优先级",
            approaches="设计增强方法",
            metrics="定义成功指标"
        },
        /enhance{
            implementation="应用有针对性的改进",
            verification="验证增强",
            iteration="根据需要重复过程"
        }
    ],
    output={
        improved_solution="增强的实现",
        improvement_summary="增强描述",
        quality_assessment="根据标准的评估"
    }
}
```

## 8. 文档指南（Documentation Guidelines）

### 代码文档协议（Code Documentation Protocol）

```
/doc.code{
    intent="创建全面、有用的代码文档",
    input={
        code="<要记录的代码>",
        audience="<目标读者>"
    },
    process=[
        /analyze{
            purpose="识别代码目的和功能",
            interfaces="确定公共接口",
            usage="理解使用模式"
        },
        /structure{
            overview="创建高级描述",
            api="记录公共 API",
            examples="开发使用示例",
            internals="解释关键内部概念"
        },
        /implement{
            inline="添加适当的内联注释",
            headers="创建全面的标题",
            guides="开发使用指南",
            references="包含相关引用"
        },
        /validate{
            completeness="验证文档覆盖范围",
            clarity="确保可理解性",
            accuracy="确认技术准确性"
        }
    ],
    output={
        documentation="完整的代码文档",
        examples="说明性使用示例",
        quick_reference="简明参考指南"
    }
}
```

### 技术写作协议（Technical Writing Protocol）

```
/doc.technical{
    intent="创建清晰、信息丰富的技术文档",
    input={
        subject="<文档主题>",
        audience="<目标读者>",
        purpose="<文档目标>"
    },
    process=[
        /plan{
            scope="定义文档范围",
            structure="设计逻辑组织",
            level="确定适当的详细级别"
        },
        /draft{
            overview="创建概念概述",
            details="开发详细解释",
            examples="包含说明性示例",
            references="添加支持引用"
        },
        /refine{
            clarity="提高解释清晰度",
            flow="增强逻辑进展",
            accessibility="调整以适应受众理解"
        },
        /finalize{
            review="进行彻底审查",
            formatting="应用一致的格式",
            completeness="确保全面覆盖"
        }
    ],
    output={
        documentation="完整的技术文档",
        summary="执行摘要",
        navigation="文档结构指南"
    }
}
```

## 9. 项目特定约定（Project-Specific Conventions）

### Bash 命令
- `npm run build`: 构建项目
- `npm run test`: 运行所有测试
- `npm run test:file <file>`: 运行特定文件的测试
- `npm run lint`: 运行代码检查器
- `npm run typecheck`: 运行类型检查器

### 代码风格
- 使用一致的缩进（2个空格）
- 遵循项目特定的命名约定
- 为公共函数包含 JSDoc 注释
- 为新功能编写单元测试
- 遵循单一职责原则
- 使用描述性的变量和函数名称

### Git 工作流
- 使用功能分支进行新开发
- 编写描述性的提交消息
- 在提交和 PR 中引用问题编号
- 保持提交集中和原子性
- 在 PR 之前将功能分支 rebase 到 main
- 合并到 main 时压缩提交

### 项目结构
- `/src`: 源代码
- `/test`: 测试文件
- `/docs`: 文档
- `/scripts`: 构建和实用脚本
- `/types`: 类型定义

## 使用说明

1. **自定义**: 修改各部分以匹配项目的特定需求和约定。

2. **扩展**: 随着工作流的相关性，添加新的协议和框架。

3. **集成**: 通过名称或结构在提示中引用这些协议，以便 Claude Code 使用。

4. **权限**: 考虑将常用工具添加到允许列表，以实现更高效的工作流。

5. **工作流适配**: 组合和修改协议以创建特定任务的自定义工作流。

6. **文档**: 使此文件与项目特定信息和约定保持同步。

7. **共享**: 将此文件提交到仓库，以与团队共享这些认知工具。
