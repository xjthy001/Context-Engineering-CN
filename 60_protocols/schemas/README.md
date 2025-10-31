# 协议架构 (Protocol Schemas)

本目录包含上下文工程协议的形式化架构定义。

## 什么是协议架构?

协议架构提供了协议的形式化、结构化定义,包括:

- **类型定义**: 输入和输出的数据类型
- **约束**: 参数的验证规则
- **接口**: 协议之间的交互方式
- **形式化规范**: 数学或逻辑定义
- **验证规则**: 确保正确性的规则

## 架构格式

协议架构使用多种格式定义:

### 1. JSON Schema

用于数据结构的形式化定义:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "协议名称",
  "type": "object",
  "properties": {
    "input": { "type": "object" },
    "output": { "type": "object" }
  }
}
```

### 2. Pareto-lang

用于协议逻辑的声明性定义:

```
/protocol.name {
  intent: "描述",
  input: { ... },
  process: [ ... ],
  output: { ... }
}
```

### 3. Python 类型注解

用于实现的类型安全:

```python
from typing import Dict, List, Optional

class ProtocolInput:
    field_state: FieldState
    parameters: Dict[str, Any]
```

## 可用架构

### 核心协议架构

1. **吸引子共现架构**
   - 定义吸引子交互
   - 指定共振参数
   - 约束涌现行为

2. **递归涌现架构**
   - 定义递归结构
   - 指定终止条件
   - 约束自主性级别

3. **记忆吸引子架构**
   - 定义记忆表示
   - 指定检索机制
   - 约束持久性规则

4. **场共振架构**
   - 定义共振结构
   - 指定放大参数
   - 约束场动力学

## 使用架构

### 验证

架构可用于验证协议输入和输出:

```python
from jsonschema import validate

# 根据架构验证输入
validate(instance=protocol_input, schema=input_schema)
```

### 生成

架构可用于生成文档和代码:

```python
# 从架构生成文档
docs = generate_documentation(schema)

# 从架构生成类型
types = generate_types(schema)
```

### 测试

架构可用于生成测试用例:

```python
# 从架构生成测试
tests = generate_tests(schema)
```

## 架构开发

### 1. 定义核心类型

从基本类型开始:

```python
class FieldState:
    """场状态的表示"""
    dimensions: int
    attractors: List[Attractor]
    coherence: float
```

### 2. 指定约束

添加验证规则:

```python
def validate_field_state(state: FieldState) -> bool:
    assert state.dimensions > 0
    assert 0 <= state.coherence <= 1
    return True
```

### 3. 文档化接口

清晰地记录所有接口:

```python
def execute(self, input: ProtocolInput) -> ProtocolOutput:
    """
    执行协议。

    参数:
        input: 协议输入数据

    返回:
        协议输出结果

    引发:
        ValidationError: 如果输入无效
    """
```

### 4. 创建示例

提供有效输入的示例:

```python
example_input = {
    "current_field_state": example_field,
    "parameters": {
        "threshold": 0.5,
        "iterations": 10
    }
}
```

## 架构版本控制

架构使用语义版本控制:

- **主版本**: 破坏性更改
- **次版本**: 向后兼容的添加
- **修订版本**: 向后兼容的修复

示例:
```json
{
  "version": "1.2.3",
  "schema_version": "http://json-schema.org/draft-07/schema#"
}
```

## 架构验证

所有架构应该:

1. **语法有效**: 通过架构验证器
2. **语义一致**: 逻辑上连贯
3. **完整**: 覆盖所有用例
4. **文档化**: 清晰解释
5. **测试**: 有验证测试

## 最佳实践

### 1. 保持简单

从最简单的有效架构开始:

```json
{
  "type": "object",
  "required": ["input"],
  "properties": {
    "input": { "type": "object" }
  }
}
```

### 2. 逐步完善

根据需要添加约束:

```json
{
  "type": "object",
  "required": ["input"],
  "properties": {
    "input": {
      "type": "object",
      "required": ["field_state"],
      "properties": {
        "field_state": { "$ref": "#/definitions/FieldState" }
      }
    }
  }
}
```

### 3. 使用引用

避免重复,使用引用:

```json
{
  "definitions": {
    "FieldState": { ... }
  },
  "properties": {
    "input_field": { "$ref": "#/definitions/FieldState" },
    "output_field": { "$ref": "#/definitions/FieldState" }
  }
}
```

### 4. 记录决策

解释架构选择:

```json
{
  "description": "使用 0-1 范围表示连贯性以简化计算",
  "type": "number",
  "minimum": 0,
  "maximum": 1
}
```

## 贡献

欢迎贡献架构:

1. 遵循现有约定
2. 提供完整文档
3. 包含验证测试
4. 附带示例

## 工具

用于处理架构的推荐工具:

- **jsonschema**: Python 中的架构验证
- **dataclasses**: Python 中的数据类
- **pydantic**: Python 中的数据验证
- **typescript**: JavaScript/TypeScript 中的类型系统

## 进一步阅读

- [JSON Schema 规范](https://json-schema.org/)
- [Python 类型提示](https://docs.python.org/3/library/typing.html)
- [Pydantic 文档](https://pydantic-docs.helpmanual.io/)
