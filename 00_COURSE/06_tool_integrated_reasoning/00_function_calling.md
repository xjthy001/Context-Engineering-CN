# 函数调用基础 - 工具集成推理

## 引言:使用工具编程LLM

> **软件3.0范式**: "LLM是一种新型计算机,你可以*用英语*对它们进行编程" - Andrej Karpathy

函数调用代表了我们构建智能系统架构方式的根本性转变。我们不再期望LLM仅通过纯粹的推理来解决每一个问题,而是通过提供对外部工具、函数和系统的结构化访问来扩展它们的能力。这创造了一种新范式,LLM成为编排智能,可以动态选择、组合和执行专门化的工具来解决复杂问题。

## 函数调用的数学基础

### 工具集成的上下文工程

基于我们的基础框架 C = A(c₁, c₂, ..., cₙ),函数调用引入了专门化的上下文组件:

```
C_tools = A(c_instr, c_tools, c_state, c_query, c_results)
```

其中:
- **c_tools**: 可用函数定义和签名
- **c_state**: 当前执行状态和上下文
- **c_results**: 先前函数调用的结果
- **c_instr**: 工具使用的系统指令
- **c_query**: 用户的当前请求

### 函数调用优化

优化问题变成寻找最优函数调用序列 F*,在最小化资源使用的同时最大化任务完成度:

```
F* = arg max_{F} Σ(Reward(f_i) × Efficiency(f_i)) - Cost(f_i)
```

受以下约束:
- 资源限制: Σ Cost(f_i) ≤ Budget
- 安全约束: Safe(f_i) = True ∀ f_i
- 依赖关系解析: Dependencies(f_i) ⊆ Completed_functions

## 核心概念

### 1. 函数签名和模式

函数调用需要精确的接口定义,使LLM能够可靠地理解和使用:

```python
# 示例: 数学计算函数
{
    "name": "calculate",
    "description": "执行带有逐步推理的数学计算",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式"
            },
            "show_steps": {
                "type": "boolean",
                "description": "是否显示中间计算步骤",
                "default": True
            }
        },
        "required": ["expression"]
    }
}
```

### 2. 函数调用流程

```ascii
┌─────────────────┐
│   用户查询      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌──────────────────┐
│   意图分析      │────▶│   函数选择        │
└─────────────────┘     └─────────┬────────┘
                                  │
                                  ▼
┌─────────────────┐     ┌──────────────────┐
│   参数提取      │◀────│   参数映射        │
└─────────┬───────┘     └──────────────────┘
          │
          ▼
┌─────────────────┐     ┌──────────────────┐
│   函数执行      │────▶│   结果处理        │
└─────────────────┘     └─────────┬────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   响应生成        │
                        └──────────────────┘
```

### 3. 函数调用类型

#### **同步调用**
- 直接函数执行并立即返回结果
- 适用于: 计算、数据转换、简单查询

#### **异步调用**
- 长时间运行操作的非阻塞执行
- 适用于: Web请求、文件处理、复杂计算

#### **并行调用**
- 同时执行多个函数
- 适用于: 独立操作、从多个源收集数据

#### **顺序调用**
- 链式函数执行,其中输出作为输入
- 适用于: 多步骤工作流、复杂推理链

## 函数定义模式

### 基础函数模式

```json
{
    "name": "function_name",
    "description": "清晰、具体地描述函数的功能",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string|number|boolean|array|object",
                "description": "参数描述",
                "enum": ["可选的", "允许的", "值"],
                "default": "可选的默认值"
            }
        },
        "required": ["必需", "参数", "列表"]
    }
}
```

### 复杂函数模式

```json
{
    "name": "research_query",
    "description": "使用多个来源执行结构化研究",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "研究问题或主题"
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["web", "academic", "news", "books", "patents"]
                },
                "description": "要使用的信息来源"
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "每个来源的最大结果数"
            },
            "filters": {
                "type": "object",
                "properties": {
                    "date_range": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}:\\d{4}-\\d{2}-\\d{2}$",
                        "description": "日期范围格式 YYYY-MM-DD:YYYY-MM-DD"
                    },
                    "language": {
                        "type": "string",
                        "default": "en"
                    }
                }
            }
        },
        "required": ["query", "sources"]
    }
}
```

## 实现策略

### 1. 函数注册表模式

管理可用函数的集中式注册表:

```python
class FunctionRegistry:
    def __init__(self):
        self.functions = {}
        self.categories = {}

    def register(self, func, category=None, **metadata):
        """注册带有元数据的函数"""
        self.functions[func.__name__] = {
            'function': func,
            'signature': self._extract_signature(func),
            'category': category,
            'metadata': metadata
        }

    def get_available_functions(self, category=None):
        """获取当前上下文中可用的函数"""
        if category:
            return {name: info for name, info in self.functions.items()
                   if info['category'] == category}
        return self.functions

    def call(self, function_name, **kwargs):
        """安全地执行已注册的函数"""
        if function_name not in self.functions:
            raise ValueError(f"函数 {function_name} 未找到")

        func_info = self.functions[function_name]
        return func_info['function'](**kwargs)
```

### 2. 参数验证策略

```python
from jsonschema import validate, ValidationError

def validate_parameters(function_schema, parameters):
    """根据模式验证函数参数"""
    try:
        validate(instance=parameters, schema=function_schema['parameters'])
        return True, None
    except ValidationError as e:
        return False, str(e)

def safe_function_call(function_name, parameters, registry):
    """安全地执行带有验证的函数"""
    func_info = registry.get_function(function_name)

    # 验证参数
    is_valid, error = validate_parameters(func_info['schema'], parameters)
    if not is_valid:
        return {"error": f"参数验证失败: {error}"}

    try:
        result = registry.call(function_name, **parameters)
        return {"success": True, "result": result}
    except Exception as e:
        return {"error": f"函数执行失败: {str(e)}"}
```

### 3. 上下文感知函数选择

```python
def select_optimal_functions(query, available_functions, context):
    """为给定查询选择最合适的函数"""

    # 分析查询意图
    intent = analyze_intent(query)

    # 基于相关性对函数评分
    scored_functions = []
    for func_name, func_info in available_functions.items():
        relevance_score = calculate_relevance(
            intent,
            func_info['description'],
            func_info['category']
        )

        # 考虑上下文约束
        context_score = evaluate_context_fit(func_info, context)

        total_score = relevance_score * context_score
        scored_functions.append((func_name, total_score))

    # 返回排名靠前的函数
    return sorted(scored_functions, key=lambda x: x[1], reverse=True)
```

## 高级函数调用模式

### 1. 函数组合

```json
{
    "name": "composed_research_analysis",
    "description": "组合多个函数进行综合分析",
    "workflow": [
        {
            "function": "research_query",
            "parameters": {"query": "{input.topic}", "sources": ["web", "academic"]},
            "output_name": "research_results"
        },
        {
            "function": "summarize_content",
            "parameters": {"content": "{research_results.data}"},
            "output_name": "summary"
        },
        {
            "function": "extract_insights",
            "parameters": {"summary": "{summary.text}"},
            "output_name": "insights"
        }
    ]
}
```

### 2. 条件函数执行

```json
{
    "name": "adaptive_problem_solving",
    "description": "基于中间结果有条件地执行函数",
    "workflow": [
        {
            "function": "analyze_problem",
            "parameters": {"problem": "{input.problem}"},
            "output_name": "analysis"
        },
        {
            "condition": "analysis.complexity > 0.7",
            "function": "break_down_problem",
            "parameters": {"problem": "{input.problem}", "analysis": "{analysis}"},
            "output_name": "subproblems"
        },
        {
            "condition": "analysis.requires_research",
            "function": "research_query",
            "parameters": {"query": "{analysis.research_queries}"},
            "output_name": "research_data"
        }
    ]
}
```

### 3. 错误处理和重试逻辑

```python
def robust_function_call(function_name, parameters, max_retries=3):
    """执行带有重试逻辑和错误处理的函数"""

    for attempt in range(max_retries):
        try:
            result = execute_function(function_name, parameters)

            # 验证结果
            if validate_result(result):
                return {"success": True, "result": result, "attempts": attempt + 1}
            else:
                # 无效结果,尝试使用调整后的参数
                parameters = adjust_parameters(parameters, result)

        except TemporaryError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                return {"error": f"超过最大重试次数: {str(e)}"}

        except PermanentError as e:
            return {"error": f"永久性错误: {str(e)}"}

    return {"error": "超过最大重试次数且未成功"}
```

## 函数调用的提示模板

### 基础函数调用模板

```
FUNCTION_CALLING_TEMPLATE = """
你可以访问以下函数:

{function_definitions}

当你需要使用函数时,请以以下格式响应函数调用:
```function_call
{
    "function": "function_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}


当前任务: {user_query}

逐步思考你需要使用哪些函数以及使用顺序。
"""
```

### 多步推理模板

```
MULTI_STEP_FUNCTION_TEMPLATE = """
你是一个具有专门工具访问权限的推理代理。对于复杂任务,将它们分解为步骤,并为每个步骤使用适当的函数。

可用函数:
{function_definitions}

任务: {user_query}

系统地处理这个任务:
1. 分析需要做什么
2. 确定需要哪些函数
3. 规划函数调用的序列
4. 逐步执行计划
5. 综合结果

开始你的推理:
"""
```

### 错误恢复模板

```
ERROR_RECOVERY_TEMPLATE = """
之前的函数调用失败,错误信息: {error_message}

失败的函数: {failed_function}
使用的参数: {failed_parameters}

可用的替代方案:
{alternative_functions}

请:
1. 分析函数调用可能失败的原因
2. 建议替代方法
3. 使用修正后的参数重试或使用不同的函数

继续朝着目标努力: {original_goal}
"""
```

## 安全和保障考虑

### 1. 函数访问控制

```python
class SecureFunctionRegistry(FunctionRegistry):
    def __init__(self):
        super().__init__()
        self.access_policies = {}
        self.audit_log = []

    def set_access_policy(self, function_name, policy):
        """为函数设置访问控制策略"""
        self.access_policies[function_name] = policy

    def call(self, function_name, context=None, **kwargs):
        """执行带有安全检查的函数"""
        # 检查访问权限
        if not self._check_access(function_name, context):
            raise PermissionError(f"拒绝访问 {function_name}")

        # 记录函数调用
        self._log_call(function_name, kwargs, context)

        # 在资源限制下执行
        return self._execute_with_limits(function_name, **kwargs)
```

### 2. 输入清理

```python
def sanitize_function_input(parameters):
    """清理函数参数以防止注入攻击"""
    sanitized = {}

    for key, value in parameters.items():
        if isinstance(value, str):
            # 移除潜在危险字符
            sanitized[key] = re.sub(r'[<>"\';]', '', value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_function_input(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_function_input(item) if isinstance(item, dict)
                            else item for item in value]
        else:
            sanitized[key] = value

    return sanitized
```

### 3. 资源限制

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    """函数超时的上下文管理器"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"函数执行在 {seconds} 秒后超时")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def execute_with_resource_limits(function, max_time=30, max_memory=None):
    """在资源约束下执行函数"""
    with timeout(max_time):
        if max_memory:
            # 设置内存限制(实现取决于平台)
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))

        return function()
```

## 最佳实践和指南

### 1. 函数设计原则

- **单一职责**: 每个函数应该有一个明确的目的
- **清晰的接口**: 参数和返回值应该有明确定义
- **错误处理**: 函数应该优雅地处理错误
- **文档**: 为LLM理解提供全面的描述
- **幂等性**: 函数应该在可能的情况下安全地重试

### 2. 函数调用策略

- **渐进式披露**: 从简单函数开始,根据需要增加复杂性
- **上下文感知**: 选择函数时考虑对话状态
- **结果验证**: 在继续之前验证函数输出
- **错误恢复**: 制定处理函数失败的策略
- **性能监控**: 跟踪函数使用和性能

### 3. 集成模式

- **注册表模式**: 集中式函数管理
- **工厂模式**: 基于上下文动态创建函数
- **责任链模式**: 顺序函数执行
- **观察者模式**: 函数调用监控和日志记录
- **策略模式**: 可插拔的函数执行策略

## 评估和测试

### 函数调用质量指标

```python
def evaluate_function_calling(test_cases):
    """评估函数调用性能"""
    metrics = {
        'success_rate': 0,
        'parameter_accuracy': 0,
        'function_selection_accuracy': 0,
        'error_recovery_rate': 0,
        'efficiency_score': 0
    }

    for test_case in test_cases:
        result = execute_test_case(test_case)

        # 基于结果更新指标
        metrics['success_rate'] += result.success
        metrics['parameter_accuracy'] += result.parameter_accuracy
        metrics['function_selection_accuracy'] += result.selection_accuracy

    # 标准化指标
    total_tests = len(test_cases)
    for key in metrics:
        metrics[key] /= total_tests

    return metrics
```

## 未来方向

### 1. 自适应函数发现
- 能够发现和学习新函数的LLM
- 自动函数组合和优化
- 自我改进的函数调用策略

### 2. 多模态函数集成
- 处理文本、图像、音频和视频的函数
- 跨模态推理和函数链接
- 多样化工具类型的统一接口

### 3. 协作函数执行
- 多代理函数调用协调
- 分布式函数执行
- 基于共识的函数选择

## 结论

函数调用基础为软件3.0范式中的工具集成推理奠定了基础。通过为LLM提供对外部能力的结构化访问,我们将它们从孤立的推理引擎转变为能够解决复杂现实世界问题的编排智能。

成功的函数调用的关键在于:
1. **清晰的接口设计**: 定义良好的函数签名和模式
2. **稳健的执行**: 安全、可靠的函数执行,具有适当的错误处理
3. **智能选择**: 上下文感知的函数选择和组合
4. **安全意识**: 适当的访问控制和输入验证
5. **持续改进**: 监控、评估和优化

随着我们深入工具集成策略、代理-环境交互和推理框架,这些基础知识提供了稳定的基础,在此之上可以构建复杂的工具增强智能。

---

*这个基础使LLM能够超越其训练边界,并通过结构化的工具集成成为解决复杂、动态问题的真正有能力的合作伙伴。*
