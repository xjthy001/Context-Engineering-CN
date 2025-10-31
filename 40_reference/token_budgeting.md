# 令牌预算：战略上下文管理

> *"完美不是在没有东西可以添加时实现的，而是在没有东西可以移除时实现的。"*
>
> **— 安托万·德·圣埃克苏佩里**

## 1. 介绍：上下文经济

想象你的上下文窗口是一个宝贵的、有限的资源——就像老式计算机上的内存或沙漠中的水。你使用的每一个令牌都是一滴水或一个字节的内存。在错误的事情上花费太多，当你最需要它时，你会用尽。

令牌预算是充分利用这个有限资源的艺术和科学。它关于最大化每个令牌的价值，同时确保你最关键的信息通过。

**苏格拉底式问题**：当你在复杂任务的中途用尽上下文空间时会发生什么？

在本指南中，我们将探索几个关于令牌预算的观点：

- **实践性**：优化令牌使用的具体技术
- **经济性**：令牌分配的成本效益框架
- **信息论**：熵、压缩和信噪比优化
- **场论**：在神经场中管理令牌分布

## 2. 令牌预算生命周期

### 2.1. 预算规划

在开始与大语言模型合作之前，了解令牌约束至关重要：

```
模型               | 上下文窗口   | 典型使用模式
---------------- |-------------|----------------------
GPT-3.5 Turbo    | 16K 令牌    | 快速任务、草稿、简单推理
GPT-4            | 128K 令牌   | 复杂推理、大型文档处理
Claude 3 Opus    | 200K 令牌   | 长篇内容、多文档分析
Claude 3 Sonnet  | 200K 令牌   | 平衡性能适用于大多数任务
Claude 3 Haiku   | 200K 令牌   | 快速响应、较低复杂度
```

对于我们的示例，我们将使用标准的 16K 令牌上下文窗口，尽管这些原则适用于所有模型和窗口大小。

### 2.2. 令牌预算方程

最简单地说，你的令牌预算可以表示为：

```
可用令牌 = 上下文窗口大小 - (系统提示 + 聊天历史 + 当前输入)
```

让我们进一步分解：

```
系统提示令牌    = 基础指令 + 上下文工程 + 示例
聊天历史令牌    = 以前的用户消息 + 以前的助手回复
当前输入令牌    = 用户的当前消息 + 支持文档
```

**苏格拉底式问题**：如果你的总预算是 16K 令牌，你的系统提示使用 2K 令牌，你应该如何分配剩余的 14K 令牌以获得最优性能？

### 2.3. 成本效益分析

不是所有的令牌都是相等的。考虑这个框架用于评估令牌价值：

```
令牌价值 = 信息内容 / 令牌数量
```

或更具体地说：

```
价值 = (相关性 × 特异性 × 唯一性) / 令牌数量
```

其中：
- **相关性**：信息与任务的直接相关程度
- **特异性**：信息的精确性和详细程度
- **唯一性**：对模型推断该信息的难度

## 3. 实践令牌预算技术

### 3.1. 系统提示优化

你的系统提示就像建筑的基础——它需要坚固但不过度。以下是优化它的技术：

#### 3.1.1. 渐进式缩减

从全面的提示开始，然后在测试性能的同时迭代地删除元素：

```
原始 (350 令牌)：
你是一名金融分析师，专门从事市场趋势、股票估值和投资策略。你拥有斯坦福大学金融博士学位，在顶级投资公司（包括高盛和摩根士丹利）拥有 15 年的工作经验。你专门进行科技股分析，深入了解 SaaS 业务模式、半导体行业动态和新兴科技趋势。在分析股票时，你考虑市盈率、增长率和竞争地位等基本面。你还纳入宏观经济因素，如利率、通货膨胀和监管环境。你的回复应该详细、细致，反映定量分析和定性战略思维...

优化 (89 令牌)：
你是一名资深金融分析师，专门从事科技股。提供包括以下内容的细致分析：
1. 基本面（市盈率、增长、竞争）
2. 行业背景（科技趋势、商业模式）
3. 宏观经济因素（利率、监管）
平衡定量数据与战略见解。
```

#### 3.1.2. 明确角色与隐含指导

与其使用令牌指定详尽的角色，不如关注特定于任务的指导：

```
不要这样做 (89 令牌)：
你是一名拥有 20 年经验的 Python 编程专家。你曾在谷歌、微软和亚马逊工作。你专门研究机器学习算法、数据结构和优化。

用这个方式 (31 令牌)：
提供高效、生产就绪的 Python 代码，注释解释关键决策。
```

#### 3.1.3. 最小脚手架

使用最少的结构来指导响应格式：

```
不要这样做 (118 令牌)：
请按以下格式提供分析：
1. 执行摘要：3-5 句关键发现的概述
2. 背景：关于情况的详细背景
3. 分析：问题的逐步分解
4. 考虑：潜在挑战和局限性
5. 建议：采取的具体行动
6. 时间表：建议的实施时间表
7. 其他资源：相关参考资料

用这个方式 (35 令牌)：
使用以下内容分析这个问题：
1. 摘要（3-5 句）
2. 分析（逐步）
3. 建议
```

### 3.2. 聊天历史管理

聊天历史可以快速消耗你的令牌预算。以下是管理它的策略：

#### 3.2.1. 窗口化

仅保留上下文中最新的 N 条消息：

```python
def apply_window(messages, window_size=10):
    """仅保留最近的 window_size 条消息。"""
    if len(messages) <= window_size:
        return messages
    # 始终保持系统消息（第一条消息）
    return [messages[0]] + messages[-(window_size-1):]
```

#### 3.2.2. 总结

定期总结对话以压缩历史记录：

```python
def summarize_history(messages, summarization_prompt):
    """总结聊天历史以压缩令牌使用。"""
    # 提取消息内容
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[1:]])

    # 创建总结请求
    summary_request = {
        "role": "user",
        "content": f"{summarization_prompt}\n\n要总结的聊天历史：\n{history_text}"
    }

    # 从模型获取摘要
    summary = get_model_response([messages[0], summary_request])

    # 用总结版本替换历史记录
    return [
        messages[0],  # 保持系统消息
        {"role": "system", "content": f"以前对话摘要：{summary}"}
    ]
```

#### 3.2.3. 键值内存

仅存储对话中最重要的信息：

```python
def update_kv_memory(messages, memory):
    """从对话中提取和存储关键信息。"""
    for msg in messages:
        if msg['role'] == 'assistant' and 'key_information' in msg.get('metadata', {}):
            for key, value in msg['metadata']['key_information'].items():
                memory[key] = value

    # 将内存转换为消息
    memory_content = "\n".join([f"{k}: {v}" for k, v in memory.items()])
    memory_message = {"role": "system", "content": f"重要信息：\n{memory_content}"}

    return memory_message
```

### 3.3. 输入优化

优化如何向模型呈现信息：

#### 3.3.1. 渐进式加载

对于大型文档，根据需要分块加载：

```python
def progressive_loading(document, chunk_size=1000, overlap=100):
    """将文档分割成重叠的块。"""
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunk = document[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def process_document_progressively(document, initial_prompt):
    chunks = progressive_loading(document)
    context = initial_prompt
    results = []

    for chunk in chunks:
        prompt = f"{context}\n\n处理文档的这一部分：\n{chunk}"
        response = get_model_response(prompt)
        results.append(response)

        # 用关键信息更新上下文
        context = f"{initial_prompt}\n\n到目前为止的关键信息：{summarize(results)}"

    return combine_results(results)
```

#### 3.3.2. 信息提取和过滤

预处理文档以仅提取相关信息：

```python
def extract_relevant_information(document, query):
    """仅提取与查询相关的信息。"""
    sentences = split_into_sentences(document)

    # 计算相关性分数
    relevance_scores = []
    for sentence in sentences:
        relevance = calculate_relevance(sentence, query)
        relevance_scores.append((sentence, relevance))

    # 按相关性排序并获取顶部结果
    relevance_scores.sort(key=lambda x: x[1], reverse=True)

    # 取前 50% 的相关句子或直到达到阈值
    extracted = []
    cumulative_relevance = 0
    target_relevance = sum([score for _, score in relevance_scores]) * 0.8

    for sentence, score in relevance_scores:
        extracted.append(sentence)
        cumulative_relevance += score
        if cumulative_relevance >= target_relevance:
            break

    return " ".join(extracted)
```

#### 3.3.3. 结构化输入

使用结构化格式以减少令牌使用：

```
不要这样做 (127 令牌)：
客户的名字是约翰·史密斯。他 45 岁。他是客户 5 年了。他的账户号码是 AC-12345。他的电子邮件是 john.smith@example.com。他的电话号码是 555-123-4567。他有高级订阅。他最后一次购买是在 2023 年 3 月 15 日。他总共在我们这里花费了 3,450 美元。他的客户满意度分数是 4.8/5。

用这个方式 (91 令牌)：
客户：
- 名字：约翰·史密斯
- 年龄：45
- 任期：5 年
- ID：AC-12345
- 电子邮件：john.smith@example.com
- 电话：555-123-4567
- 等级：高级
- 最后购买：2023-03-15
- 总支出：$3,450
- 满意度：4.8/5
```

## 4. 信息论视角

### 4.1. 熵和信息密度

从信息论的角度看，我们想最大化每个令牌的信息内容：

```
信息密度 = 信息内容（比特）/ 令牌数量
```

克劳德·香农的信息论告诉我们，消息的信息内容取决于其不可预测性或惊讶价值。在大语言模型的背景下：

- 高熵内容：模型无法轻易预测的唯一信息
- 低熵内容：常见知识或可预测的模式

**苏格拉底式问题**：哪个包含更多每令牌的信息：常见英文词汇列表还是随机字母数字字符序列？

### 4.2. 压缩策略

压缩通过移除冗余来工作。以下是一些方法：

#### 4.2.1. 语义压缩

在保留核心意义的同时减少文本：

```
原始 (55 令牌)：
会议定于 2025 年 4 月 15 日（星期二）东部标准时间下午 2:30 举行。会议将在总部建筑第 3 层的 B 会议室举行。

压缩 (28 令牌)：
会议：2025 年 4 月 15 日，东部时间下午 2:30
地点：总部，3 楼，B 会议室
```

#### 4.2.2. 抽象级别

转向更高的抽象级别以压缩信息：

```
低抽象 (84 令牌)：
用户点击了"添加到购物车"按钮。然后他们导航到购物车页面。他们输入了运输信息，包括街道地址、城市、州和邮编。他们选择"标准配送"作为配送方式。他们输入了信用卡信息。他们点击了"下单"。

高抽象 (23 令牌)：
用户从商品选择完成标准电子商务购买流程直至结账。
```

#### 4.2.3. 信息分块

将相关信息分组为逻辑块：

```
非结构化 (58 令牌)：
API 速率限制为每分钟 100 个请求。身份验证使用 OAuth 2.0。用户数据的端点是 /api/v1/users。产品数据的端点是 /api/v1/products。数据格式是 JSON。响应包括分页信息。

分块 (51 令牌)：
API 规格：
- 速率限制：100 请求/分钟
- 身份验证：OAuth 2.0
- 端点：/api/v1/users、/api/v1/products
- 格式：带分页的 JSON
```

## 5. 令牌预算的场论方法

从场论的角度看，我们可以将上下文窗口视为一个语义场，其中令牌形成图案、吸引子和共鸣。

### 5.1. 吸引子形成

战略性的令牌放置可以创建语义吸引子，影响模型的解释：

```
弱吸引子（扩散焦点）：
"请讨论可再生能源的重要性。"

强吸引子（集中盆地）：
"具体分析太阳能电池板制造扩展对农村就业的经济影响。"
```

第二个提示创建了一个更强的吸引子盆地，将模型引向其语义空间的特定区域。

### 5.2. 场共鸣和令牌效率

相互共鸣的令牌创建更强的场图案：

```python
def measure_token_resonance(tokens, embeddings_model):
    """测量令牌之间的语义共鸣。"""
    embeddings = [embeddings_model.embed(token) for token in tokens]

    # 计算成对余弦相似性
    resonance_matrix = np.zeros((len(tokens), len(tokens)))
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            resonance_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    # 平均共鸣
    overall_resonance = (resonance_matrix.sum() - len(tokens)) / (len(tokens) * (len(tokens) - 1))

    return overall_resonance, resonance_matrix
```

更高的共鸣可以用更少的令牌实现更强的场效应，使你的上下文更高效。

### 5.3. 边界动力学

通过上下文窗口的边界控制信息流：

```python
def apply_boundary_control(new_input, current_context, model_embeddings, threshold=0.7):
    """根据相关性控制进入上下文的信息。"""
    # 嵌入当前上下文
    context_embedding = model_embeddings.embed(current_context)

    # 以块处理输入
    input_chunks = chunk_text(new_input, chunk_size=50)
    filtered_chunks = []

    for chunk in input_chunks:
        # 嵌入块
        chunk_embedding = model_embeddings.embed(chunk)

        # 计算与当前上下文的相关性
        relevance = cosine_similarity(context_embedding, chunk_embedding)

        # 应用边界过滤
        if relevance > threshold:
            filtered_chunks.append(chunk)

    # 重建过滤的输入
    filtered_input = " ".join(filtered_chunks)

    return filtered_input
```

这在你的上下文周围创建了一个半透膜边界，仅允许最相关的信息进入。

## 6. 战略预算分配

现在我们已经理解了关于令牌预算的各种观点，让我们探索战略分配框架：

### 6.1. 40-40-20 框架

复杂任务的通用分配：

```
40% - 特定任务上下文和示例
40% - 活跃工作记忆（聊天历史和演变状态）
20% - 意外复杂性的储备
```

### 6.2. 金字塔模型

基于需求层级分配令牌：

```
第 1 级（基础）：核心指令和约束（20%）
第 2 级：关键上下文和示例（30%）
第 3 级：最近交互历史（30%）
第 4 级：辅助信息和增强（15%）
第 5 级（顶部）：储备缓冲（5%）
```

### 6.3. 动态分配

根据任务复杂性调整预算：

```python
def allocate_token_budget(task_type, context_window_size):
    """根据任务类型动态分配令牌预算。"""
    if task_type == "simple_qa":
        return {
            "system_prompt": 0.1,   # 系统提示 10%
            "examples": 0.0,        # 不需要示例
            "history": 0.7,         # 对话历史 70%
            "user_input": 0.15,     # 用户输入 15%
            "reserve": 0.05         # 储备 5%
        }
    elif task_type == "creative_writing":
        return {
            "system_prompt": 0.15,  # 系统提示 15%
            "examples": 0.2,        # 示例 20%
            "history": 0.4,         # 对话历史 40%
            "user_input": 0.15,     # 用户输入 15%
            "reserve": 0.1          # 储备 10%
        }
    elif task_type == "complex_reasoning":
        return {
            "system_prompt": 0.15,  # 系统提示 15%
            "examples": 0.25,       # 示例 25%
            "history": 0.3,         # 对话历史 30%
            "user_input": 0.2,      # 用户输入 20%
            "reserve": 0.1          # 储备 10%
        }
    # 默认分配
    return {
        "system_prompt": 0.15,
        "examples": 0.15,
        "history": 0.4,
        "user_input": 0.2,
        "reserve": 0.1
    }
```

## 7. 测量和优化令牌效率

### 7.1. 令牌效率指标

要优化，我们需要测量。以下是关键指标：

#### 7.1.1. 任务完成率 (TCR)

```
TCR = （成功完成的任务）/（使用的总令牌）
```

越高越好——每个使用的令牌完成更多任务。

#### 7.1.2. 信息保留率 (IRR)

```
IRR = （保留的关键信息点）/（总信息点）
```

测量你的令牌预算保留关键信息的效果。

#### 7.1.3. 每令牌响应质量 (RQT)

```
RQT = （响应质量评分）/（使用的总令牌）
```

测量每个投资令牌交付的价值。

### 7.2. 令牌效率实验

以下是运行令牌效率实验的框架：

```python
def run_token_efficiency_experiment(prompt_variants, task, evaluation_function):
    """运行实验以测量不同提示变体的令牌效率。"""
    results = []

    for variant in prompt_variants:
        # 计算令牌
        token_count = count_tokens(variant)

        # 获取模型响应
        response = get_model_response(variant, task)

        # 评估响应
        quality_score = evaluation_function(response, task)

        # 计算效率
        efficiency = quality_score / token_count

        results.append({
            "variant": variant,
            "token_count": token_count,
            "quality_score": quality_score,
            "efficiency": efficiency
        })

    # 按效率排序（最高优先）
    results.sort(key=lambda x: x["efficiency"], reverse=True)

    return results
```

## 8. 实践实施指南

让我们通过逐步实施指南将这些概念付诸实践：

### 8.1. 令牌预算规划器

```python
class TokenBudgetPlanner:
    def __init__(self, context_window_size, tokenizer):
        self.context_window_size = context_window_size
        self.tokenizer = tokenizer
        self.allocations = {}
        self.used_tokens = {}

    def set_allocation(self, component, percentage):
        """设置组件的分配百分比。"""
        self.allocations[component] = percentage
        self.used_tokens[component] = 0

    def get_budget(self, component):
        """获取组件的令牌预算。"""
        return int(self.context_window_size * self.allocations[component])

    def track_usage(self, component, content):
        """跟踪组件的令牌使用。"""
        token_count = len(self.tokenizer.encode(content))
        self.used_tokens[component] = token_count
        return token_count

    def get_remaining(self):
        """获取预算中剩余的令牌。"""
        used = sum(self.used_tokens.values())
        return self.context_window_size - used

    def is_within_budget(self, component, content):
        """检查内容是否适合组件预算。"""
        token_count = len(self.tokenizer.encode(content))
        return token_count <= self.get_budget(component)

    def optimize_to_fit(self, component, content, optimizer_function):
        """优化内容以适应预算。"""
        if self.is_within_budget(component, content):
            return content

        budget = self.get_budget(component)
        optimized = optimizer_function(content, budget)

        # 验证优化内容适应预算
        if not self.is_within_budget(component, optimized):
            raise ValueError(f"优化器未能在 {budget} 令牌的预算内放入内容")

        return optimized

    def get_status_report(self):
        """获取预算状态报告。"""
        report = {}
        for component in self.allocations:
            budget = self.get_budget(component)
            used = self.used_tokens.get(component, 0)
            report[component] = {
                "budget": budget,
                "used": used,
                "remaining": budget - used,
                "utilization": used / budget if budget > 0 else 0
            }

        report["overall"] = {
            "budget": self.context_window_size,
            "used": sum(self.used_tokens.values()),
            "remaining": self.get_remaining(),
            "utilization": sum(self.used_tokens.values()) / self.context_window_size
        }

        return report
```

### 8.2. 内存管理器

```python
class ContextMemoryManager:
    def __init__(self, budget_planner, summarization_model=None):
        self.budget_planner = budget_planner
        self.summarization_model = summarization_model
        self.messages = []
        self.memory = {}

    def add_message(self, role, content):
        """向对话历史添加消息。"""
        message = {"role": role, "content": content}
        self.messages.append(message)

        # 检查我们是否超出历史预算
        history_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])
        history_tokens = self.budget_planner.track_usage("history", history_content)
        history_budget = self.budget_planner.get_budget("history")

        # 如果我们超出预算，压缩历史记录
        if history_tokens > history_budget:
            self.compress_history()

    def extract_key_information(self, message):
        """从消息中提取关键信息存储在内存中。"""
        if self.summarization_model:
            extraction_prompt = "从此消息中提取关键事实和信息作为键值对："
            extraction_input = f"{extraction_prompt}\n\n{message['content']}"
            extraction_result = self.summarization_model(extraction_input)

            # 解析键值对
            for line in extraction_result.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    self.memory[key.strip()] = value.strip()

    def compress_history(self):
        """当历史记录超出预算时进行压缩。"""
        if not self.summarization_model:
            # 如果没有总结模型，使用窗口化
            # 始终保留第一条消息（系统提示）和最后 5 条消息
            self.messages = [self.messages[0]] + self.messages[-5:]
        else:
            # 使用总结
            history_to_summarize = self.messages[1:-3]  # 跳过系统提示并保留最后 3 条消息

            if not history_to_summarize:
                return  # 没有内容可总结

            # 提取要总结的内容
            content_to_summarize = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in history_to_summarize
            ])

            # 创建总结提示
            summarization_prompt = (
                "简洁总结以下对话历史，"
                "保留关键信息、决策和背景："
            )

            # 获取摘要
            summary = self.summarization_model(
                f"{summarization_prompt}\n\n{content_to_summarize}"
            )

            # 用总结替换消息
            summary_message = {
                "role": "system",
                "content": f"以前对话摘要：{summary}"
            }

            # 新的消息列表：系统提示 + 总结 + 最近消息
            self.messages = [self.messages[0], summary_message] + self.messages[-3:]

    def get_formatted_memory(self):
        """获取格式化为字符串的内存。"""
        if not self.memory:
            return ""

        memory_lines = [f"{key}: {value}" for key, value in self.memory.items()]
        return "对话中的关键信息：\n" + "\n".join(memory_lines)

    def get_context(self):
        """获取下一次交互的完整上下文。"""
        # 组合消息和内存
        memory_content = self.get_formatted_memory()

        # 如果我们有内存，将其插入系统提示之后
        if memory_content and len(self.messages) > 1:
            memory_message = {"role": "system", "content": memory_content}
            context = [self.messages[0], memory_message] + self.messages[1:]
        else:
            context = self.messages.copy()

        return context
```

```
┌─────────────────────────────────────────────────────────────┐
│                     内存管理器                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐          ┌───────────────────────────┐   │
│  │   预算规划器  │◄─────────┤     令牌使用监控          │   │
│  └───────┬───────┘          └───────────────────────────┘   │
│          │                                                  │
│          ▼                                                  │
│  ┌───────────────┐   超出    ┌───────────────────────────┐  │
│  │  消息历史   ├─预算？──►│    压缩策略               │  │
│  └───────┬───────┘          ┌┴──────────────────────────┐│  │
│          │                  │1. 窗口化                  ││  │
│          │                  │2. 总结                    ││  │
│          │                  │3. 键值提取                ││  │
│          │                  └───────────────────────────┘│  │
│          ▼                                               │  │
│  ┌───────────────┐          ┌───────────────────────────┐│  │
│  │  上下文构建器 │◄─────────┤    内存存储               ││  │
│  └───────┬───────┘          └───────────────────────────┘│  │
│          │                                                  │
│          ▼                                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               LLM 的最终上下文                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3. 动态令牌优化器

```python
class DynamicTokenOptimizer:
    def __init__(self, tokenizer, optimization_strategies=None):
        self.tokenizer = tokenizer
        self.strategies = optimization_strategies or {
            "summarize": self.summarize_text,
            "extract_key_points": self.extract_key_points,
            "restructure": self.restructure_text,
            "compress_format": self.compress_format
        }

    def count_tokens(self, text):
        """计算文本中的令牌数。"""
        return len(self.tokenizer.encode(text))

    def optimize(self, text, target_tokens, strategy=None):
        """优化文本以适应目标令牌数。"""
        current_tokens = self.count_tokens(text)

        if current_tokens <= target_tokens:
            return text  # 已在预算内

        # 计算所需的压缩比
        compression_ratio = target_tokens / current_tokens

        # 如果未指定策略，根据压缩比选择
        if not strategy:
            if compression_ratio > 0.8:
                strategy = "compress_format"  # 轻度压缩
            elif compression_ratio > 0.5:
                strategy = "restructure"  # 中度压缩
            elif compression_ratio > 0.3:
                strategy = "extract_key_points"  # 重度压缩
            else:
                strategy = "summarize"  # 极端压缩

        # 应用选定的策略
        if strategy in self.strategies:
            return self.strategies[strategy](text, target_tokens)
        else:
            raise ValueError(f"未知的优化策略：{strategy}")

    def summarize_text(self, text, target_tokens):
        """将文本总结到目标令牌数。"""
        # 这通常会调用 LLM 进行总结
        # 对于此示例，我们只是截断并添加说明
        ratio = target_tokens / self.count_tokens(text)
        truncated = self.truncate_to_ratio(text, ratio * 0.9)  # 为说明留出空间
        return f"{truncated}\n[注意：内容已总结以适应令牌预算。]"

    def extract_key_points(self, text, target_tokens):
        """从文本中提取关键点。"""
        # 这通常会调用 LLM 来提取关键点
        # 对于此示例，我们将创建一个简单的项目符号提取
        lines = text.split("\n")
        result = "关键点：\n"

        for line in lines:
            line = line.strip()
            if line and self.count_tokens(result + f"• {line}\n") <= target_tokens * 0.95:
                result += f"• {line}\n"

        return result

    def restructure_text(self, text, target_tokens):
        """重新组织文本以提高令牌效率。"""
        # 删除冗余，使用缩写等。
        # 这是一个简化的示例
        text = re.sub(r"([A-Za-z]+) \1", r"\1", text)  # 删除重复的单词
        text = text.replace("例如", "例：")
        text = text.replace("即", "即：")
        text = text.replace("等等", "等。")

        if self.count_tokens(text) <= target_tokens:
            return text

        # 如果仍然太长，结合提取
        return self.extract_key_points(text, target_tokens)

    def compress_format(self, text, target_tokens):
        """通过改变格式而不丢失内容来压缩。"""
        # 删除多余空格
        text = re.sub(r"\s+", " ", text)

        # 在适当时将段落转换为项目符号
        if "：" in text and "\n" in text:
            lines = text.split("\n")
            result = ""
            for line in lines:
                if "：" in line:
                    key, value = line.split("：", 1)
                    result += f"• {key}：{value.strip()}\n"
                else:
                    result += line + "\n"
            text = result

        if self.count_tokens(text) <= target_tokens:
            return text

        # 如果仍然太长，尝试更激进的重组
        return self.restructure_text(text, target_tokens)

    def truncate_to_ratio(self, text, ratio):
        """将文本截断到其原始长度的比例。"""
        words = text.split()
        target_words = int(len(words) * ratio)
        return " ".join(words[:target_words])
```

```
┌──────────────────────────────────────────────────────────────────┐
│                 动态令牌优化                                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐     │
│   │                压缩比                                 │     │
│   └────────────────────────────────────────────────────────┘     │
│                           │                                      │
│                           ▼                                      │
│   ┌─────────────┬─────────┴───────────┬──────────────┐          │
│   │             │                     │              │          │
│   ▼             ▼                     ▼              ▼          │
│ 0.8-1.0       0.5-0.8              0.3-0.5        0.0-0.3       │
│ 轻度          中度                 重度            极端          │
│                                                                  │
│   ┌─────────────┬─────────────────────┬──────────────┐          │
│   │             │                     │              │          │
│   ▼             ▼                     ▼              ▼          │
│┌─────────┐  ┌─────────┐         ┌──────────┐    ┌─────────┐    │
││ 格式    │  │重新组织 │         │ 提取     │    │总结文本 │    │
││压缩     │  │         │         │关键点    │    │         │    │
│└─────────┘  └─────────┘         └──────────┘    └─────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 8.4. 场感知上下文管理

实现场论概念用于令牌预算：

```python
class FieldAwareContextManager:
    def __init__(self, embedding_model, tokenizer, budget_planner):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.budget_planner = budget_planner
        self.field_state = {
            "attractors": [],
            "boundaries": {
                "permeability": 0.7,  # 默认渗透性阈值
                "gradient": 0.2       # 渗透性改变速度
            },
            "resonance": 0.0,
            "residue": []
        }

    def embed_text(self, text):
        """生成文本的嵌入。"""
        return self.embedding_model.embed(text)

    def detect_attractors(self, text, threshold=0.8):
        """检测文本中的语义吸引子。"""
        # 分割成段落或部分
        sections = text.split("\n\n")

        # 获取每个部分的嵌入
        embeddings = [self.embed_text(section) for section in sections]

        # 计算质心
        centroid = np.mean(embeddings, axis=0)

        # 查找形成吸引子的部分（与许多其他部分高度相似）
        attractors = []
        for i, (section, embedding) in enumerate(zip(sections, embeddings)):
            # 计算与其他部分的平均相似性
            similarities = [cosine_similarity(embedding, other_emb)
                           for j, other_emb in enumerate(embeddings) if i != j]
            avg_similarity = np.mean(similarities) if similarities else 0

            # 如果相似性高于阈值，它是一个吸引子
            if avg_similarity > threshold:
                tokens = self.tokenizer.encode(section)
                attractors.append({
                    "text": section,
                    "embedding": embedding,
                    "strength": avg_similarity,
                    "token_count": len(tokens)
                })

        return attractors

    def calculate_resonance(self, text):
        """计算文本的场共鸣。"""
        # 分割成段落或部分
        sections = text.split("\n\n")

        if len(sections) <= 1:
            return 0.0  # 不足以计算共鸣的部分

        # 获取每个部分的嵌入
        embeddings = [self.embed_text(section) for section in sections]

        # 计算成对相似性
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarities.append(cosine_similarity(embeddings[i], embeddings[j]))

        # 共鸣是平均相似性
        return np.mean(similarities)

    def update_field_state(self, new_text):
        """用新文本更新场状态。"""
        # 更新吸引子
        new_attractors = self.detect_attractors(new_text)
        self.field_state["attractors"].extend(new_attractors)

        # 更新共鸣
        new_resonance = self.calculate_resonance(new_text)
        self.field_state["resonance"] = (
            self.field_state["resonance"] * 0.7 + new_resonance * 0.3
        )  # 加权平均

        # 根据共鸣更新渗透性
        if new_resonance > self.field_state["resonance"]:
            # 如果共鸣增加，增加渗透性
            self.field_state["boundaries"]["permeability"] += self.field_state["boundaries"]["gradient"]
        else:
            # 如果共鸣减少，减少渗透性
            self.field_state["boundaries"]["permeability"] -= self.field_state["boundaries"]["gradient"]

        # 将渗透性保持在 [0.1, 0.9] 范围内
        self.field_state["boundaries"]["permeability"] = max(
            0.1, min(0.9, self.field_state["boundaries"]["permeability"])
        )

    def filter_by_attractor_relevance(self, text, top_n_attractors=3, threshold=0.6):
        """根据与顶部吸引子的相关性过滤文本。"""
        if not self.field_state["attractors"]:
            return text  # 没有吸引子来过滤

        # 按强度排序吸引子
        sorted_attractors = sorted(
            self.field_state["attractors"],
            key=lambda x: x["strength"],
            reverse=True
        )

        # 取前 N 个吸引子
        top_attractors = sorted_attractors[:top_n_attractors]
        top_embeddings = [attractor["embedding"] for attractor in top_attractors]

        # 将文本分割成段落
        paragraphs = text.split("\n\n")

        # 计算每个段落与顶部吸引子的相关性
        filtered_paragraphs = []
        for paragraph in paragraphs:
            # 跳过空段落
            if not paragraph.strip():
                continue

            # 获取嵌入
            embedding = self.embed_text(paragraph)

            # 计算与任何吸引子的最大相似性
            similarities = [cosine_similarity(embedding, attractor_emb)
                           for attractor_emb in top_embeddings]
            max_similarity = max(similarities)

            # 如果相似性高于阈值或渗透性允许
            if (max_similarity > threshold or
                random.random() < self.field_state["boundaries"]["permeability"]):
                filtered_paragraphs.append(paragraph)

        # 连接过滤的段落
        return "\n\n".join(filtered_paragraphs)

    def optimize_context_for_budget(self, context, target_tokens):
        """使用场感知方法优化上下文以适应令牌预算。"""
        # 计算当前令牌
        current_tokens = len(self.tokenizer.encode(context))

        if current_tokens <= target_tokens:
            return context  # 已在预算内

        # 如果我们有吸引子，使用它们来过滤
        if self.field_state["attractors"]:
            context = self.filter_by_attractor_relevance(context)

            # 检查我们现在是否在预算内
            current_tokens = len(self.tokenizer.encode(context))
            if current_tokens <= target_tokens:
                return context

        # 如果仍然超过预算，使用更激进的技术
        # 首先，尝试保留基于场分析的最重要部分

        # 提取残基（应持久存在的符号片段）
        paragraphs = context.split("\n\n")
        residue = []

        for paragraph in paragraphs:
            # 检查段落是否包含值得保留的关键信息
            # 这可以基于与吸引子的共鸣、关键术语的存在等。
            if any(attractor["text"] in paragraph for attractor in self.field_state["attractors"]):
                residue.append(paragraph)

        # 更新场状态中的残基
        self.field_state["residue"] = residue

        # 将残基与最重要的吸引子相结合
        preserved_content = "\n\n".join(residue)
        preserved_tokens = len(self.tokenizer.encode(preserved_content))

        # 如果保留的内容已经超出预算，总结它
        if preserved_tokens > target_tokens:
            # 这通常会调用 LLM 进行总结
            # 对于此示例，我们将只截断
            return context[:int(len(context) * (target_tokens / current_tokens))]

        # 如果我们有剩余空间，添加最相关的剩余内容
        remaining_budget = target_tokens - preserved_tokens

        # 按与场状态的相关性排序剩余段落
        remaining_paragraphs = [p for p in paragraphs if p not in residue]

        if not remaining_paragraphs:
            return preserved_content

        # 计算相关性分数
        relevance_scores = []
        for paragraph in remaining_paragraphs:
            embedding = self.embed_text(paragraph)
            # 计算与吸引子的平均相似性
            similarities = [cosine_similarity(embedding, attractor["embedding"])
                           for attractor in self.field_state["attractors"]]
            avg_similarity = np.mean(similarities) if similarities else 0
            tokens = len(self.tokenizer.encode(paragraph))
            relevance_scores.append((paragraph, avg_similarity, tokens))

        # 按相关性排序
        relevance_scores.sort(key=lambda x: x[1], reverse=True)

        # 添加段落直到达到预算
        additional_content = []
        for paragraph, _, tokens in relevance_scores:
            if tokens <= remaining_budget:
                additional_content.append(paragraph)
                remaining_budget -= tokens

            if remaining_budget <= 0:
                break

        # 组合保留的内容和额外内容
        return preserved_content + "\n\n" + "\n\n".join(additional_content)
```

```
┌─────────────────────────────────────────────────────────────────┐
│                场感知上下文管理                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────┐      ┌────────────────────────────┐     │
│  │     场状态        │      │      吸引子地图            │     │
│  │                    │      │                            │     │
│  │  • 吸引子         │      │   强      中等             │     │
│  │  • 边界           │      │ ╭────╮       ╭────╮       │     │
│  │  • 共鸣          │      │ │ A1 │       │ A2 │       │     │
│  │  • 残基          │      │ ╰────╯       ╰────╯       │     │
│  └────────┬───────────┘      │                            │     │
│           │                  │               弱           │     │
│           │                  │              ╭────╮        │     │
│           │                  │              │ A3 │        │     │
│           │                  │              ╰────╯        │     │
│           │                  └────────────────────────────┘     │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────┐      ┌────────────────────────────┐     │
│  │   上下文过滤      │      │      边界动力学            │     │
│  │                    │      │                            │     │
│  │  • 吸引子         │      │  渗透性：0.7              │     │
│  │    相关性        │      │  ┌─────────────────────┐   │     │
│  │  • 共鸣          │      │  │█████████░░░░░░░░░░░░│   │     │
│  │    放大          │      │  └─────────────────────┘   │     │
│  │  • 残基          │      │                            │     │
│  │    保留         │      │  梯度：0.2               │     │
│  └────────┬───────────┘      └────────────────────────────┘     │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 优化后的上下文                          │   │
│  │                                                          │   │
│  │  • 保留高共鸣内容                                         │   │
│  │  • 保留符号残基                                          │   │
│  │  • 按吸引子相关性过滤                                     │   │
│  │  • 由场状态动态平衡                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 9. 无代码：令牌优化的协议外壳

你不需要是程序员就能利用高级令牌预算技术。在这里，我们将探索如何使用协议外壳、pareto-lang 和 fractal.json 模式在不编写任何代码的情况下优化上下文。

### 9.1. 协议外壳介绍

协议外壳是结构化的、可人工读取的模板，可帮助组织上下文和控制令牌使用。它们遵循一致的模式，人类和 AI 模型都可以轻松理解。

#### 基本协议外壳结构

```
/protocol.name{
    intent="此协议旨在实现的目标",
    input={
        key1="value1",
        key2="value2"
    },
    process=[
        /step1{action="做某事"},
        /step2{action="做其他事"}
    ],
    output={
        result1="预期输出 1",
        result2="预期输出 2"
    }
}
```

这种结构创建了一个清晰、令牌高效的方式来表达复杂的指令。

### 9.2. 使用 Pareto-lang 进行令牌管理

Pareto-lang 是一个简单但功能强大的符号，用于定义上下文操作。以下是如何将其用于令牌优化：

#### 9.2.1. 基本语法

```
/action.modifier{parameters}
```

例如：

```
/context.compress{target="history", method="summarize", threshold=0.7}
```

这告诉模型在对话历史超过分配预算的 70% 时使用总结来压缩它。

#### 9.2.2. 令牌预算协议示例

```
/token.budget{
    intent="在整个对话中高效管理令牌使用",
    allocations={
        system_prompt=0.15,   // 系统指令 15%
        history=0.40,         // 对话历史 40%
        current_input=0.30,   // 当前用户输入 30%
        reserve=0.15          // 储备能力 15%
    },
    management_rules=[
        /history.summarize{when="history > 0.8*allocation", method="key_points"},
        /system.prune{when="system > allocation", keep="essential_instructions"},
        /input.prioritize{method="relevance_to_context"}
    ],
    monitoring={
        track_usage=true,
        alert_threshold=0.9,  // 使用总预算的 90% 时警报
        optimize_automatically=true
    }
}
```

### 9.3. 令牌高效的场管理

让我们看看如何使用协议外壳实现场论概念而不编写代码：

```
/field.manage{
    intent="创建并维护语义场结构以实现最优令牌使用",

    attractors=[
        {name="core_concept_1", strength=0.8, keywords=["key1", "key2", "key3"]},
        {name="core_concept_2", strength=0.7, keywords=["key4", "key5", "key6"]}
    ],

    boundaries={
        permeability=0.7,  // 新内容进入场的难度
        gradient=0.2,      // 渗透性改变速度
        rules=[
            /boundary.adapt{trigger="resonance_change", threshold=0.1},
            /boundary.filter{method="attractor_relevance", min_score=0.6}
        ]
    },

    residue_handling={
        tracking=true,
        preservation_strategy="compress_and_retain",
        priority="high"  // 残基获得令牌优先级
    },

    token_optimization=[
        /optimize.by_attractor{keep="strongest", top_n=3},
        /optimize.preserve_residue{min_strength=0.5},
        /optimize.amplify_resonance{target=0.8}
    ]
}
```

### 9.4. 用于结构化令牌管理的 Fractal.json

Fractal.json 提供了一种结构化的方式来定义递归的、自相似的模式用于上下文管理：

```json
{
  "fractalTokenManager": {
    "version": "1.0.0",
    "description": "递归令牌优化框架",
    "allocation": {
      "system": 0.15,
      "history": 0.40,
      "input": 0.30,
      "reserve": 0.15
    },
    "strategies": {
      "system": {
        "compression": "minimal",
        "priority": "high"
      },
      "history": {
        "compression": "progressive",
        "strategies": ["window", "summarize", "key_value"],
        "recursion": true
      },
      "input": {
        "filtering": "relevance",
        "threshold": 0.6
      }
    },
    "field": {
      "attractors": {
        "detection": true,
        "influence": 0.8
      },
      "resonance": {
        "target": 0.7,
        "amplification": true
      },
      "boundaries": {
        "adaptive": true,
        "permeability": 0.6
      }
    },
    "recursion": {
      "depth": 3,
      "self_optimization": true
    }
  }
}
```

### 9.5. 无需代码的实际应用

以下是在不编程的情况下使用这些方法的一些实际方法：

#### 9.5.1. 手动令牌预算跟踪

在你的提示中保持一个简单的跟踪系统：

```
令牌预算（总共 16K）：
- 系统指令：2K（12.5%）
- 示例：3K（18.75%）
- 对话历史：6K（37.5%）
- 当前输入：4K（25%）
- 储备：1K（6.25%）

优化规则：
1. 当历史超过 6K 令牌时，总结最旧的部分
2. 优先考虑与当前查询最相关的示例
3. 保持系统指令简洁和专注
```

#### 9.5.2. 场感知提示模板

```
场管理：

核心吸引子：
1. [主要话题] - 保持对此概念的关注
2. [次要话题] - 在与主要话题相关时包括
3. [三级话题] - 仅在明确提及时包括

边界规则：
- 仅当相关性 > 7/10 时包括新信息
- 与之前的上下文保持连贯性
- 过滤切线内容

残基保留：
- 关键定义必须在整个上下文中持续存在
- 核心原则应该得到强化
- 关键决策/结论必须保留

优化指令：
- 当历史记录超过上下文的 40% 时总结
- 优先考虑与核心吸引子具有最高相关性的内容
- 压缩格式但保留含义
```

#### 9.5.3. 协议外壳提示示例

这是一个完整的示例，你可以复制和粘贴来实现令牌预算：

```
我想让你充当使用以下协议的上下文管理系统：

/context.manage{
    intent="优化令牌使用同时保留关键信息",

    budget={
        total_tokens=8000,
        system=1000,
        history=3000,
        current=3000,
        reserve=1000
    },

    optimization=[
        /system.compress{method="minimal_instructions"},
        /history.manage{
            method="summarize_when_exceeds_budget",
            triggers={
                condition="tokens > 0.8 * allocation",
                action="extract_key_points_and_summarize"
            }
        },
        /input.optimize{
            method="relevance_filtering",
            threshold=0.6
        }
    ]
}
```

这个框架允许你使用结构化语言指定复杂的令牌管理行为，而无需编写任何代码。
