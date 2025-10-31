# 上下文管理:软件3.0革命
> "受过教育的头脑的标志是能够思考一个想法而不接受它。"
>
> — [亚里士多德](https://www.goodreads.com/quotes/1629-it-is-the-mark-of-an-educated-mind-to-be)
>
## 转变:从代码到上下文
> [**软件正在改变(再次)——Andrej Karpathy在YC AI创业学校的演讲**](https://www.youtube.com/watch?v=LCEmiRjPEtQ)

我们正在见证[**软件3.0**](https://x.com/karpathy/status/1935518272667217925)的出现——一个创新的新时代,在这个时代,结构化提示成为编程,上下文工程成为新的软件架构。这代表了我们构建智能系统方式的根本性转变。

<img width="1917" height="360" alt="image" src="https://github.com/user-attachments/assets/91457d09-434c-4476-a0ed-2d78a19c4154" />


```
软件1.0:手动编程
├─ 编写显式指令
├─ 手动处理所有边缘情况
└─ 刚性、确定性执行

软件2.0:机器学习
├─ 从数据模式中训练
├─ 学习隐式关系
└─ 统计的、概率性输出

软件3.0:上下文工程
├─ 结构化提示作为编程
├─ 协议作为可重用程序模块
└─ 动态的、上下文感知的执行
```




## 三大支柱:初学者指南

### 这三个东西是什么?

**想象建造一座房子:**
- **提示(PROMPTS)** = 与建筑师交谈(沟通)
- **编程(PROGRAMMING)** = 建筑工具和技术(实现)
- **协议(PROTOCOLS)** = 协调一切的完整蓝图(编排)

### 支柱1:提示模板 - 通信层

**什么是提示模板?**
提示模板是与AI系统通信的可重用模式。与其每次都编写独特的提示,你可以创建带有占位符的模板,这些占位符可以被填充。

**简单示例:**
```
基础提示:"分析这段代码的错误。"

模板版本:
"分析以下{LANGUAGE}代码的{ANALYSIS_TYPE}:
关注:{FOCUS_AREAS}
输出格式:{OUTPUT_FORMAT}

代码:
{CODE_BLOCK}
"
```

**带结构的高级模板:**
```
CONTEXT_ANALYSIS_TEMPLATE = """
# 上下文分析请求

## 目标信息
- 领域:{domain}
- 范围:{scope}
- 优先级:{priority_level}

## 分析参数
- 深度:{analysis_depth}
- 视角:{viewpoint}
- 约束:{limitations}

## 输入数据
{input_content}

## 预期输出格式
{output_specification}

请根据这些参数分析提供的信息,并按照指定的格式提供见解。
"""
```

**模板为什么重要:**
- **一致性**:每次都是相同的格式
- **可重用性**:可跨不同项目使用
- **可扩展性**:易于修改和扩展
- **质量**:减少错误和遗漏

### 支柱2:编程 - 实现层

编程提供支持上下文管理的计算基础设施。

**传统的上下文管理代码:**
```python
class ContextManager:
    """传统的上下文管理编程方法"""

    def __init__(self, max_context_size=10000):
        self.context_buffer = []
        self.max_size = max_context_size
        self.compression_ratio = 0.7

    def add_context(self, new_info, priority=1):
        """将信息添加到上下文中,带有优先级权重"""
        context_item = {
            'content': new_info,
            'priority': priority,
            'timestamp': time.now(),
            'token_count': self.estimate_tokens(new_info)
        }

        self.context_buffer.append(context_item)

        if self.get_total_tokens() > self.max_size:
            self.compress_context()

    def compress_context(self):
        """减少上下文大小,同时保留重要信息"""
        # 按优先级和最近度排序
        sorted_context = sorted(
            self.context_buffer,
            key=lambda x: (x['priority'], x['timestamp']),
            reverse=True
        )

        # 保留高优先级项目,压缩或删除低优先级项目
        compressed = []
        total_tokens = 0

        for item in sorted_context:
            if total_tokens + item['token_count'] <= self.max_size:
                compressed.append(item)
                total_tokens += item['token_count']
            elif item['priority'] > 0.8:  # 关键信息
                # 压缩而不是删除
                compressed_item = self.compress_item(item)
                compressed.append(compressed_item)
                total_tokens += compressed_item['token_count']

        self.context_buffer = compressed

    def retrieve_relevant_context(self, query, max_items=5):
        """检索给定查询的最相关上下文"""
        relevance_scores = []

        for item in self.context_buffer:
            score = self.calculate_relevance(query, item['content'])
            relevance_scores.append((score, item))

        # 按相关性排序并返回顶部项目
        relevant_items = sorted(
            relevance_scores,
            key=lambda x: x[0],
            reverse=True
        )[:max_items]

        return [item[1] for item in relevant_items]
```

**与提示模板的集成:**
```python
def generate_contextual_prompt(self, base_template, query, context_items):
    """将模板与相关上下文结合"""

    # 格式化上下文以便包含
    formatted_context = self.format_context_items(context_items)

    # 用动态值填充模板
    prompt = base_template.format(
        domain=self.detect_domain(query),
        context_information=formatted_context,
        user_query=query,
        output_format=self.determine_output_format(query)
    )

    return prompt
```

### 支柱3:协议 - 编排层

**什么是协议?(简单解释)**

协议就像一个**会思考的菜谱**。就像烹饪菜谱告诉你:
- 你需要什么配料(输入)
- 要遵循什么步骤(过程)
- 你应该得到什么(输出)

协议告诉AI系统:
- 要收集什么信息(输入)
- 如何处理该信息(步骤)
- 如何格式化和交付结果(输出)

**但与简单的菜谱不同,协议是:**
- **自适应的**:它们可以根据条件改变
- **递归的**:它们可以调用自己或其他协议
- **上下文感知的**:它们考虑当前情况
- **可组合的**:它们可以与其他协议组合

**基础协议示例:**

```
/analyze.text{
    intent="系统地分析文本内容以获取见解",

    input={
        text_content="<要分析的文本>",
        analysis_type="<情感|主题|结构|质量>",
        depth_level="<表面|中等|深入>"
    },

    process=[
        /understand{
            action="阅读和理解文本",
            output="basic_understanding"
        },
        /categorize{
            action="根据分析类型识别关键类别",
            depends_on="basic_understanding",
            output="category_structure"
        },
        /analyze{
            action="在每个类别内执行详细分析",
            depends_on="category_structure",
            output="detailed_findings"
        },
        /synthesize{
            action="将发现综合成连贯的见解",
            depends_on="detailed_findings",
            output="synthesis_results"
        }
    ],

    output={
        analysis_report="结构化的发现和见解",
        confidence_metrics="可靠性指标",
        recommendations="建议的下一步"
    }
}
```

**高级上下文管理协议:**

```
/context.orchestration{
    intent="跨多个信息源和处理阶段动态管理上下文",

    input={
        primary_query="<用户的主要请求>",
        available_sources=["<信息源列表>"],
        constraints={
            max_tokens="<令牌限制>",
            processing_time="<时间限制>",
            priority_areas="<重点领域>"
        },
        current_context_state="<现有上下文信息>"
    },

    process=[
        /context.assessment{
            action="评估当前上下文的完整性和相关性",
            evaluate=[
                "information_gaps",
                "redundancy_levels",
                "relevance_scores",
                "temporal_currency"
            ],
            output="context_assessment_report"
        },

        /source.prioritization{
            action="按相关性和可靠性对信息源进行排名",
            consider=[
                "source_authority",
                "information_freshness",
                "alignment_with_query",
                "processing_cost"
            ],
            depends_on="context_assessment_report",
            output="prioritized_source_list"
        },

        /adaptive.retrieval{
            action="基于优先级和约束检索信息",
            strategy="dynamic_allocation",
            process=[
                /high_priority{
                    sources="top_3_sources",
                    allocation="60%_of_token_budget"
                },
                /medium_priority{
                    sources="next_5_sources",
                    allocation="30%_of_token_budget"
                },
                /background{
                    sources="remaining_sources",
                    allocation="10%_of_token_budget"
                }
            ],
            depends_on="prioritized_source_list",
            output="retrieved_information_package"
        },

        /context.synthesis{
            action="智能地将检索到的信息与现有上下文结合",
            methods=[
                /deduplication{action="删除冗余信息"},
                /hierarchical_organization{action="按重要性和关系进行结构化"},
                /compression{action="优化信息密度"},
                /coherence_check{action="确保逻辑一致性"}
            ],
            depends_on="retrieved_information_package",
            output="synthesized_context_structure"
        },

        /response.generation{
            action="使用优化的上下文生成响应",
            approach="template_plus_dynamic_content",
            template_selection="based_on_query_type_and_context_complexity",
            depends_on="synthesized_context_structure",
            output="contextually_informed_response"
        }
    ],

    output={
        final_response="用户查询的完整答案",
        context_utilization_report="上下文如何被使用",
        efficiency_metrics={
            token_usage="实际与预算",
            processing_time="持续时间细分",
            information_coverage="完整性评估"
        },
        improvement_suggestions="对未来类似查询的建议"
    },

    meta={
        protocol_version="v1.2.0",
        execution_timestamp="<运行时>",
        resource_consumption="<指标>",
        adaptation_log="<协议在执行期间如何调整>"
    }
}
```

## 集成:三者如何协同工作

### 真实世界示例:代码审查系统

让我们构建一个综合的代码审查系统,展示三大支柱如何协同工作。

**1. 提示模板(通信层):**

```python
CODE_REVIEW_TEMPLATES = {
    'security_focus': """
    # 以安全为重点的代码审查

    ## 要审查的代码
    语言:{language}
    框架:{framework}
    安全上下文:{security_requirements}

    ```{language}
    {code_content}
    ```

    ## 审查要求
    - 识别潜在的安全漏洞
    - 检查常见的攻击向量:{attack_vectors}
    - 验证输入清理和输出编码
    - 审查身份验证和授权逻辑
    - 评估加密实现

    ## 输出格式
    以JSON格式提供结果,包括严重性级别和补救指导。
    """,

    'performance_focus': """
    # 以性能为重点的代码审查

    ## 代码分析目标
    {code_content}

    ## 性能标准
    - 时间复杂度:{max_time_complexity}
    - 空间复杂度:{max_space_complexity}
    - 可扩展性要求:{scale_requirements}

    关注:{performance_areas}
    """,

    'maintainability_focus': """
    # 可维护性代码审查

    分析:
    - 代码清晰度和可读性
    - 文档完整性
    - 设计模式使用
    - 技术债务指标

    代码:
    {code_content}
    """
}
```

**2. 编程(实现层):**

```python
class CodeReviewOrchestrator:
    """管理代码审查过程的编程层"""

    def __init__(self):
        self.templates = CODE_REVIEW_TEMPLATES
        self.context_manager = ContextManager(max_tokens=50000)
        self.review_history = []

    def analyze_code(self, code_content, review_type='comprehensive'):
        """编排代码审查的主要方法"""

        # 步骤1:分析代码特征
        code_metadata = self.extract_code_metadata(code_content)

        # 步骤2:构建上下文
        relevant_context = self.build_review_context(
            code_metadata,
            review_type
        )

        # 步骤3:选择并自定义模板
        template = self.select_template(review_type, code_metadata)
        customized_prompt = self.customize_template(
            template,
            code_content,
            code_metadata,
            relevant_context
        )

        # 步骤4:执行审查协议
        review_results = self.execute_review_protocol(
            customized_prompt,
            code_content,
            review_type
        )

        # 步骤5:后处理和格式化结果
        formatted_results = self.format_review_results(review_results)

        # 步骤6:更新上下文以供未来审查
        self.update_review_context(code_content, formatted_results)

        return formatted_results

    def extract_code_metadata(self, code):
        """提取有关代码结构和特征的信息"""
        return {
            'language': self.detect_language(code),
            'framework': self.detect_framework(code),
            'complexity_score': self.calculate_complexity(code),
            'size_metrics': self.get_size_metrics(code),
            'dependency_analysis': self.analyze_dependencies(code),
            'pattern_usage': self.detect_patterns(code)
        }

    def build_review_context(self, metadata, review_type):
        """为审查构建相关上下文"""
        context_elements = []

        # 添加相关的历史审查
        similar_reviews = self.find_similar_reviews(metadata)
        context_elements.extend(similar_reviews)

        # 添加框架特定的指南
        if metadata['framework']:
            guidelines = self.get_framework_guidelines(metadata['framework'])
            context_elements.append(guidelines)

        # 如果是安全审查,添加安全模式
        if 'security' in review_type:
            security_patterns = self.get_security_patterns(metadata['language'])
            context_elements.append(security_patterns)

        return self.context_manager.optimize_context(context_elements)
```

**3. 协议(编排层):**

```
/code.review.comprehensive{
    intent="基于代码特征进行自适应焦点的全面、多维度代码审查",

    input={
        source_code="<要审查的代码>",
        review_scope="<安全|性能|可维护性|综合>",
        project_context="<项目信息和要求>",
        constraints={
            time_budget="<可用审查时间>",
            expertise_level="<审查者专业水平>",
            priority_areas="<特定关注领域>"
        }
    },

    process=[
        /code.analysis.initial{
            action="执行初步代码分析以了解结构和特征",
            analyze=[
                "language_and_framework_detection",
                "architectural_pattern_identification",
                "complexity_assessment",
                "dependency_mapping",
                "surface_level_issue_detection"
            ],
            output="code_analysis_profile"
        },

        /context.preparation{
            action="基于代码分析准备相关上下文",
            context_sources=[
                /historical_reviews{
                    source="similar_code_reviews_from_history",
                    relevance_threshold=0.7
                },
                /framework_guidelines{
                    source="best_practices_for_detected_framework",
                    priority="high"
                },
                /security_patterns{
                    source="known_vulnerability_patterns_for_language",
                    condition="security_review_requested"
                },
                /performance_benchmarks{
                    source="performance_standards_for_code_type",
                    condition="performance_review_requested"
                }
            ],
            depends_on="code_analysis_profile",
            output="review_context_package"
        },

        /adaptive.review.strategy{
            action="根据代码特征和约束确定最优审查方法",
            strategy_selection=[
                /comprehensive_approach{
                    condition="sufficient_time_and_simple_code",
                    coverage="all_dimensions_equally"
                },
                /focused_approach{
                    condition="time_constraints_or_complex_code",
                    coverage="prioritize_by_risk_and_impact"
                },
                /iterative_approach{
                    condition="very_large_codebase",
                    coverage="review_in_phases_with_feedback_loops"
                }
            ],
            depends_on=["code_analysis_profile", "review_context_package"],
            output="review_execution_plan"
        },

        /multi.dimensional.analysis{
            action="跨多个维度同时执行审查",
            dimensions=[
                /security.analysis{
                    focus="vulnerability_detection_and_threat_modeling",
                    methods=["static_analysis_patterns", "attack_vector_mapping", "data_flow_security"],
                    output="security_findings"
                },
                /performance.analysis{
                    focus="efficiency_and_scalability_assessment",
                    methods=["complexity_analysis", "resource_usage_patterns", "bottleneck_identification"],
                    output="performance_findings"
                },
                /maintainability.analysis{
                    focus="code_quality_and_long_term_sustainability",
                    methods=["readability_assessment", "design_pattern_usage", "technical_debt_identification"],
                    output="maintainability_findings"
                },
                /correctness.analysis{
                    focus="logical_accuracy_and_requirement_alignment",
                    methods=["logic_flow_verification", "edge_case_identification", "requirement_traceability"],
                    output="correctness_findings"
                }
            ],
            parallel_execution=true,
            depends_on="review_execution_plan",
            output="multi_dimensional_findings"
        },

        /synthesis.and.prioritization{
            action="跨维度组合发现并按影响优先排序",
            synthesis_methods=[
                /cross_dimensional_correlation{
                    action="identify_issues_that_span_multiple_dimensions",
                    example="security_vulnerability_that_also_impacts_performance"
                },
                /impact_assessment{
                    action="evaluate_business_and_technical_impact_of_each_finding",
                    factors=["severity", "likelihood", "fix_complexity", "business_criticality"]
                },
                /priority_ranking{
                    action="rank_all_findings_by_overall_priority",
                    algorithm="weighted_impact_urgency_matrix"
                }
            ],
            depends_on="multi_dimensional_findings",
            output="prioritized_comprehensive_report"
        },

        /actionable.recommendations{
            action="为每个发现生成具体的、可操作的建议",
            recommendation_types=[
                /immediate_fixes{
                    description="issues_that_should_be_addressed_immediately",
                    include_code_examples=true
                },
                /refactoring_suggestions{
                    description="structural_improvements_for_long_term_benefit",
                    include_before_after_examples=true
                },
                /process_improvements{
                    description="development_process_changes_to_prevent_similar_issues",
                    include_implementation_guidance=true
                }
            ],
            depends_on="prioritized_comprehensive_report",
            output="actionable_improvement_plan"
        }
    ],

    output={
        executive_summary="代码质量和关键发现的高级概述",
        detailed_findings="按维度和优先级组织的完整分析结果",
        improvement_roadmap="解决已识别问题的分阶段计划",
        code_quality_metrics="定量评估和基准测试",
        recommendations={
            immediate_actions="需要紧急关注的关键问题",
            short_term_improvements="下一个开发周期的增强",
            long_term_strategic="架构和流程改进"
        },
        context_for_future_reviews="经验教训和未来使用的模式"
    },

    meta={
        review_methodology="具有自适应优先级的综合多维分析",
        tools_used="静态分析、模式匹配、上下文评估",
        confidence_levels="每个发现类别的可靠性指标",
        execution_metrics={
            time_consumed="实际与预算时间",
            coverage_achieved="每个维度分析的代码百分比",
            context_utilization="可用上下文的有效使用程度"
        }
    }
}
```

**4. 完整集成:**

```python
# 这是三大支柱在实践中如何协同工作:

class Software3CodeReviewer:
    """提示、编程和协议的完整集成"""

    def __init__(self):
        # 编程层
        self.context_manager = ContextManager()
        self.template_engine = TemplateEngine(CODE_REVIEW_TEMPLATES)
        self.protocol_executor = ProtocolExecutor()

    def review_code(self, code_content, requirements=None):
        """展示集成的主要方法"""

        # 1. 协议确定总体策略
        review_protocol = self.protocol_executor.load_protocol("code.review.comprehensive")

        # 2. 编程处理计算方面
        code_metadata = self.extract_metadata(code_content)
        relevant_context = self.context_manager.build_context(code_metadata, requirements)

        # 3. 提示模板提供通信结构
        selected_template = self.template_engine.select_optimal_template(
            code_metadata,
            requirements
        )

        # 4. 协议编排执行
        review_results = self.protocol_executor.execute(
            protocol=review_protocol,
            inputs={
                'source_code': code_content,
                'review_scope': requirements.get('scope', 'comprehensive'),
                'project_context': relevant_context,
                'constraints': requirements.get('constraints', {})
            },
            template_engine=self.template_engine,
            context_manager=self.context_manager
        )

        return review_results

# 使用示例:
reviewer = Software3CodeReviewer()

result = reviewer.review_code(
    code_content=my_python_code,
    requirements={
        'scope': 'security_and_performance',
        'constraints': {
            'time_budget': '30_minutes',
            'priority_areas': ['authentication', 'data_validation']
        }
    }
)
```

## 为什么这种集成很重要

### 传统方法的问题:
- **刚性**:每次都是相同的分析
- **低效**:大量冗余工作
- **有限**:单一视角
- **难以扩展**:需要手动定制

### 软件3.0解决方案的优势:
- **自适应**:根据上下文和要求变化
- **高效**:智能地重用模板和上下文
- **全面**:系统地集成多个视角
- **可扩展**:易于扩展和定制新场景

## 初学者的关键原则

### 1. 从简单开始,逐步构建复杂性
```
级别1:基础提示模板
├─ 带占位符的固定模板
└─ 简单替换逻辑

级别2:编程集成
├─ 动态模板选择
├─ 上下文感知定制
└─ 计算预处理

级别3:协议编排
├─ 多步骤工作流
├─ 条件逻辑和适应
└─ 跨系统集成
```

### 2. 分层思考
- **通信层**:如何与AI对话(提示/模板)
- **逻辑层**:如何处理信息(编程)
- **编排层**:如何协调一切(协议)

### 3. 专注于可重用性
- 模板应该在类似场景中工作
- 代码应该是模块化和可组合的
- 协议应该能够适应不同的上下文

### 4. 优化上下文
- 一切都应该是上下文感知的
- 信息应该在层之间高效流动
- 系统应该根据可用资源和约束进行调整

## 本课程的下一步

以下部分将深入探讨:
- **基本约束**:计算限制如何塑造我们的方法
- **内存层次结构**:多级存储和检索策略
- **压缩技术**:优化信息密度
- **优化策略**:性能和效率改进

每个部分都将演示提示、编程和协议的完整集成,展示软件3.0原则如何应用于特定的上下文管理挑战。

---

*本概述为理解提示、编程和协议如何协同工作以创建复杂、自适应和高效的上下文管理系统奠定了基础。这三大支柱的集成代表了软件3.0范式的核心。*
