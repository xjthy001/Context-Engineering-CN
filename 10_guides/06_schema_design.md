# 06. 架构设计（Schema Design）

## 概述

本指南介绍如何设计结构化的数据架构和提示架构，以实现更可靠和可维护的AI系统。

## 核心概念

1. **结构化输出** - 使用模式定义输出格式
2. **数据验证** - 确保输出符合预期的架构
3. **类型安全** - 使用类型系统确保数据一致性
4. **可组合性** - 设计可以组合的架构单元
5. **可扩展性** - 创建易于扩展的架构

## 架构基础

### JSON 架构示例

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "内容标题"
    },
    "content": {
      "type": "string",
      "description": "主要内容"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "author": { "type": "string" },
        "date": { "type": "string", "format": "date" },
        "tags": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    }
  },
  "required": ["title", "content"]
}
```

## 设计模式

### 模式 1: 分层架构

**目的**: 将复杂的数据组织成多个层级

**结构**:
```
顶层
├─ 第二层
│  ├─ 第三层
│  └─ 第三层
└─ 第二层
   └─ 第三层
```

**示例**:
```json
{
  "document": {
    "metadata": {
      "title": "...",
      "author": "..."
    },
    "sections": [
      {
        "heading": "...",
        "paragraphs": ["...", "..."]
      }
    ]
  }
}
```

**适用场景**:
- 复杂的嵌套数据
- 清晰的数据层次结构
- 递归结构

### 模式 2: 扁平架构

**目的**: 简化数据结构

**结构**:
```json
{
  "field1": "value1",
  "field2": "value2",
  "field3": "value3"
}
```

**适用场景**:
- 简单的数据
- 快速处理
- 易于序列化

### 模式 3: 枚举架构

**目的**: 限制值的选择范围

```json
{
  "status": {
    "enum": ["pending", "active", "completed", "failed"],
    "type": "string"
  },
  "priority": {
    "enum": ["low", "medium", "high"],
    "type": "string"
  }
}
```

**适用场景**:
- 固定选项
- 状态机
- 分类系统

### 模式 4: 条件架构

**目的**: 根据某个字段的值改变其他字段

```json
{
  "if": { "properties": { "type": { "const": "person" } } },
  "then": {
    "properties": {
      "firstName": { "type": "string" },
      "lastName": { "type": "string" }
    }
  },
  "else": {
    "properties": {
      "companyName": { "type": "string" }
    }
  }
}
```

**适用场景**:
- 多态结构
- 条件验证
- 灵活的架构

### 模式 5: 模板架构

**目的**: 定义可重用的架构片段

```json
{
  "definitions": {
    "person": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" }
      }
    }
  },
  "type": "object",
  "properties": {
    "owner": { "$ref": "#/definitions/person" },
    "author": { "$ref": "#/definitions/person" }
  }
}
```

**适用场景**:
- 重复的结构
- 大型架构
- 模块化设计

## 类型系统

### 基本类型

| 类型 | 描述 | 示例 |
|------|------|------|
| string | 字符串 | "hello" |
| number | 数字 | 42, 3.14 |
| integer | 整数 | 42 |
| boolean | 布尔值 | true, false |
| null | 空值 | null |
| array | 数组 | [1, 2, 3] |
| object | 对象 | {"key": "value"} |

### 复杂类型

```json
// 数组类型
{
  "type": "array",
  "items": { "type": "string" },
  "minItems": 1,
  "maxItems": 10
}

// 对象类型
{
  "type": "object",
  "properties": {
    "name": { "type": "string" }
  },
  "required": ["name"]
}

// 组合类型
{
  "oneOf": [
    { "type": "string" },
    { "type": "number" }
  ]
}
```

## 验证和约束

### 字符串约束

```json
{
  "type": "string",
  "minLength": 1,
  "maxLength": 100,
  "pattern": "^[A-Za-z]+$",
  "enum": ["value1", "value2", "value3"]
}
```

### 数字约束

```json
{
  "type": "number",
  "minimum": 0,
  "maximum": 100,
  "exclusiveMinimum": true,
  "multipleOf": 5
}
```

### 数组约束

```json
{
  "type": "array",
  "items": { "type": "string" },
  "minItems": 1,
  "maxItems": 10,
  "uniqueItems": true
}
```

## 实际架构示例

### 例子 1: 任务管理系统

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "title": { "type": "string" },
          "description": { "type": "string" },
          "status": {
            "enum": ["todo", "in_progress", "done"]
          },
          "priority": {
            "enum": ["low", "medium", "high"]
          },
          "assignee": { "type": "string" },
          "dueDate": { "type": "string", "format": "date" },
          "subtasks": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": { "type": "string" },
                "title": { "type": "string" },
                "completed": { "type": "boolean" }
              }
            }
          }
        },
        "required": ["id", "title", "status"]
      }
    }
  }
}
```

### 例子 2: 产品目录

```json
{
  "type": "object",
  "properties": {
    "products": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "description": { "type": "string" },
          "price": { "type": "number", "minimum": 0 },
          "currency": { "enum": ["USD", "EUR", "CNY"] },
          "stock": { "type": "integer", "minimum": 0 },
          "categories": {
            "type": "array",
            "items": { "type": "string" }
          },
          "specifications": {
            "type": "object",
            "additionalProperties": { "type": "string" }
          }
        },
        "required": ["id", "name", "price"]
      }
    }
  }
}
```

### 例子 3: 用户反馈系统

```json
{
  "type": "object",
  "properties": {
    "feedback": {
      "type": "object",
      "properties": {
        "userId": { "type": "string" },
        "rating": {
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        },
        "comment": { "type": "string" },
        "categories": {
          "type": "array",
          "items": {
            "enum": ["product", "service", "delivery", "other"]
          }
        },
        "timestamp": { "type": "string", "format": "date-time" }
      },
      "required": ["userId", "rating"]
    }
  }
}
```

## 使用架构的好处

### 1. 数据验证

```python
import jsonschema

schema = {...}
data = {...}

try:
    jsonschema.validate(instance=data, schema=schema)
    print("数据有效")
except jsonschema.ValidationError as e:
    print(f"数据无效: {e.message}")
```

### 2. 类型提示

```python
from typing import TypedDict

class Task(TypedDict):
    id: str
    title: str
    status: str
    priority: str
```

### 3. 文档生成

使用架构自动生成API文档和客户端代码

### 4. IDE 支持

架构定义提供IDE的自动完成和类型检查

## 最佳实践

1. **清晰的命名** - 使用描述性的字段名
2. **完整的文档** - 为每个字段添加描述
3. **类型严格** - 明确定义类型和约束
4. **版本管理** - 管理架构的版本变化
5. **渐进演化** - 向后兼容地演化架构
6. **验证** - 总是验证输入数据
7. **测试** - 为架构编写测试用例

## 常见错误

- ❌ 过度复杂的架构
- ❌ 不充分的约束
- ❌ 缺乏版本控制
- ❌ 忽视向后兼容性
- ❌ 不验证数据

## 高级主题

- **OpenAPI 规范** - REST API 架构定义
- **GraphQL 架构** - 查询语言架构
- **Protocol Buffers** - 二进制序列化格式
- **Apache Avro** - 数据序列化框架
- **AsyncAPI** - 异步API规范

## 下一步

- 学习 `07_recursive_patterns.md` - 递归模式设计
- 探索 `05_prompt_programs.md` - 在提示程序中使用架构
- 参考 `04_rag_recipes.md` - RAG 系统的架构设计

## 相关资源

- JSON Schema 官方文档
- OpenAPI 规范
- 数据架构最佳实践
- 类型系统入门
