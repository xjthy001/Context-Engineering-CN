# 07. 递归模式（Recursive Patterns）

## 概述

本指南介绍如何在提示工程和AI系统中使用递归模式，实现自相似的处理和自我改进的系统。

## 核心概念

1. **递归结构** - 问题的递归分解
2. **自相似性** - 模式在不同层级的重复
3. **基础情形** - 递归终止条件
4. **递归深度** - 控制递归的层级数
5. **记忆化** - 缓存递归调用的结果

## 递归基础

### 递归的三个要素

```
1. 基础情形（Base Case）
   - 何时停止递归
   - 直接返回结果

2. 递归情形（Recursive Case）
   - 如何简化问题
   - 调用自身解决更小的问题

3. 进步条件（Progress Condition）
   - 确保向基础情形前进
   - 避免无限递归
```

## 递归模式

### 模式 1: 分而治之递归

**结构**:
```
问题(n)
├─ 分解为子问题
├─ 递归解决各个子问题
├─ 合并结果
└─ 返回解决方案
```

**示例**:
```
分析复杂文档
├─ 分割为章节
├─ 每章节递归分析
│  ├─ 分割为段落
│  └─ 每段落递归分析
├─ 合并各级分析
└─ 返回整体分析
```

**代码框架**:
```python
def analyze_recursive(document, depth=0, max_depth=3):
    if depth >= max_depth or is_leaf(document):
        # 基础情形：直接分析
        return analyze_leaf(document)

    # 递归情形：分解并递归
    parts = split(document)
    results = []

    for part in parts:
        result = analyze_recursive(part, depth + 1, max_depth)
        results.append(result)

    # 合并结果
    return merge_results(results)
```

### 模式 2: 自我改进递归

**目的**: 通过多轮迭代逐步改进输出

**结构**:
```
初始输出 → 评估 → 改进 → 新输出 → 重复
                         ↓
                  满足条件时停止
```

**示例**:
```
代码生成
├─ 第1轮：生成基础代码
├─ 评估：检查错误
├─ 第2轮：修复错误
├─ 评估：检查完整性
├─ 第3轮：添加缺失功能
├─ 评估：检查优化
└─ 返回最终代码
```

**代码框架**:
```python
def improve_recursive(content, max_iterations=3, iteration=0):
    if iteration >= max_iterations:
        return content

    # 评估当前内容
    quality_score = evaluate(content)

    if quality_score >= TARGET_QUALITY:
        return content

    # 识别改进点
    improvements = identify_improvements(content)

    # 执行改进
    improved_content = apply_improvements(content, improvements)

    # 递归改进
    return improve_recursive(improved_content, max_iterations, iteration + 1)
```

### 模式 3: 层级化递归

**目的**: 从高层级逐步细化到低层级

**结构**:
```
抽象层级
    ↓ 细化
具体层级
```

**示例**:
```
产品需求规范
├─ 一级细化：主要功能模块
│  ├─ 二级细化：各模块的子功能
│  │  ├─ 三级细化：具体实现细节
│  │  │  └─ 四级细化：代码级别说明
│  │  └─ ...
│  └─ ...
└─ ...
```

**应用场景**:
- 系统设计
- 需求分解
- 项目规划

### 模式 4: 树形递归

**目的**: 生成和处理树形结构

**结构**:
```
根节点
├─ 子树1
│  ├─ 子树1.1
│  └─ 子树1.2
├─ 子树2
│  └─ 子树2.1
└─ 子树3
```

**示例**:
```python
def process_tree(node):
    if is_leaf(node):
        return process_leaf(node)

    # 处理当前节点
    current = process_node(node)

    # 递归处理子节点
    children_results = []
    for child in node.children:
        result = process_tree(child)
        children_results.append(result)

    # 合并
    return combine(current, children_results)
```

**应用**:
- 目录树处理
- 组织结构分析
- XML/JSON 处理

### 模式 5: 背包递归

**目的**: 在多个约束下优化选择

**应用场景**:
- Token 预算约束下的内容选择
- 成本约束下的功能选择
- 时间约束下的任务优先级

**框架**:
```python
def optimize_recursive(items, budget, index=0, memo=None):
    if memo is None:
        memo = {}

    # 基础情形
    if index >= len(items) or budget <= 0:
        return (0, [])

    # 记忆化
    if (index, budget) in memo:
        return memo[(index, budget)]

    # 选择当前项目
    include = 0
    included_items = []
    if items[index].cost <= budget:
        value, rest = optimize_recursive(
            items,
            budget - items[index].cost,
            index + 1,
            memo
        )
        include = items[index].value + value
        included_items = [items[index]] + rest

    # 不选择当前项目
    exclude, excluded_items = optimize_recursive(
        items, budget, index + 1, memo
    )

    # 比较选择
    if include >= exclude:
        result = (include, included_items)
    else:
        result = (exclude, excluded_items)

    memo[(index, budget)] = result
    return result
```

## 递归的关键要素

### 1. 基础情形定义

```python
# 良好的基础情形
if len(text) < MIN_LENGTH:
    return analyze_directly(text)

# 避免
if True:  # 没有合适的终止条件
    return analyze_directly(text)
```

### 2. 进度保证

```python
# 确保向基础情形前进
def process_recursive(data, depth=0):
    if depth >= MAX_DEPTH:  # 深度增加，向基础情形前进
        return process_base(data)

    # 处理
    result = recursive_process(data, depth + 1)  # 明确递归参数变化
    return result
```

### 3. 记忆化优化

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 4. 深度限制

```python
def recursive_call(data, max_depth=5, current_depth=0):
    if current_depth >= max_depth:
        # 达到深度限制，返回中间结果
        return get_partial_result(data)

    # 继续递归
    return recursive_call(data, max_depth, current_depth + 1)
```

## 实际应用示例

### 例子 1: 文档理解的递归分析

```
问题：理解一份长文档

第1层：分割为段落
  第2层：分析每个段落
    第3层：提取关键概念
      第4层：建立概念关系

结果：多层级的理解和分析
```

### 例子 2: 问题求解的递归改进

```
初始问题
  ↓ 第1次尝试
输出1（有缺陷）
  ↓ 评估和识别问题
改进策略1
  ↓ 第2次尝试
输出2（改进）
  ↓ 评估
改进策略2
  ↓ 第3次尝试
输出3（最终）
```

### 例子 3: 复杂决策的递归分解

```
主决策问题
├─ 子决策1
│  ├─ 子决策1.1
│  └─ 子决策1.2
├─ 子决策2
│  ├─ 子决策2.1
│  ├─ 子决策2.2
│  └─ 子决策2.3
└─ 子决策3

综合各层决策 → 最终决策
```

## 性能考虑

### 时间复杂度

| 模式 | 时间复杂度 | 说明 |
|------|----------|------|
| 线性递归 | O(n) | 每层一次调用 |
| 二分递归 | O(log n) | 每层分成两半 |
| 树形递归 | O(2^n) | 每层调用数倍增 |
| 记忆化 | O(n) | 避免重复计算 |

### 优化策略

1. **记忆化** - 缓存已计算的结果
2. **动态规划** - 自底向上计算
3. **尾递归优化** - 编译器优化
4. **深度限制** - 避免栈溢出
5. **迭代替换** - 使用循环代替递归

## 常见陷阱

- ❌ 无限递归 - 缺少或错误的基础情形
- ❌ 栈溢出 - 递归深度过大
- ❌ 低效率 - 重复计算相同的子问题
- ❌ 难以调试 - 复杂的递归逻辑
- ❌ Token 成本高 - 未考虑成本

## 最佳实践

1. **明确的基础情形** - 清晰定义何时停止
2. **明确的递归关系** - 明确一般情形如何处理
3. **进度保证** - 确保向基础情形前进
4. **记忆化** - 避免重复计算
5. **深度限制** - 保护系统稳定性
6. **清晰的文档** - 记录递归逻辑
7. **充分的测试** - 测试各种输入情况

## 调试技巧

```python
# 添加日志跟踪递归
def recursive_debug(data, depth=0):
    indent = "  " * depth
    print(f"{indent}→ 处理: {data}")

    if is_base_case(data):
        result = base_result(data)
        print(f"{indent}← 返回: {result}")
        return result

    result = recursive_debug(data, depth + 1)
    print(f"{indent}← 返回: {result}")
    return result
```

## 下一步

- 回顾 `05_prompt_programs.md` - 在程序中使用递归
- 学习 `06_schema_design.md` - 为递归结构定义架构
- 探索 `04_rag_recipes.md` - 递归检索模式

## 高级主题

- **互相递归** - 两个函数相互调用
- **间接递归** - 通过其他函数的递归调用
- **计算复杂性** - 递归算法的复杂性分析
- **并行递归** - 并行执行递归分支

## 相关资源

- 算法和数据结构教科书
- 递归模式文献
- 动态规划指南
- 函数式编程概念
