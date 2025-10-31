# Context Engineering 中文翻译指南

## 📋 翻译规范

### 术语表 (Terminology)

本项目使用统一的术语翻译标准,确保文档的一致性和专业性。

#### 核心概念术语

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| Context Engineering | 上下文工程 | 核心概念,优于"语境工程" |
| Prompt Engineering | 提示工程 | 标准翻译 |
| Token Budget | 令牌预算 | Token保留或译为"令牌" |
| Few-Shot Learning | 少样本学习 | 标准机器学习术语 |
| Memory Systems | 内存系统 | 技术术语,不译为"记忆" |
| Retrieval Augmentation | 检索增强 | RAG相关术语 |
| Control Flow | 控制流 | 编程术语 |
| Context Pruning | 上下文修剪 | 优化技术 |
| Cognitive Tools | 认知工具 | 核心概念 |
| Neural Field Theory | 神经场论/神经场理论 | 前沿理论 |
| Attractor Dynamics | 吸引子动力学 | 动力系统术语 |
| Symbolic Mechanisms | 符号机制 | 符号处理 |
| Quantum Semantics | 量子语义学 | 前沿概念 |
| Protocol Shells | 协议外壳 | 系统架构 |
| Field Orchestration | 场编排 | 场论应用 |
| Emergence | 涌现 | 复杂系统术语 |
| Resonance | 共振 | 场论术语 |
| Persistence | 持久性 | 系统特性 |

#### 技术术语

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| LLM (Large Language Model) | 大型语言模型 | 缩写可保留 |
| Agent | 智能体/代理 | 根据上下文选择 |
| Embedding | 嵌入 | 向量表示 |
| Vector Database | 向量数据库 | 标准术语 |
| Inference | 推理 | AI术语 |
| Fine-tuning | 微调 | 模型训练 |
| Hallucination | 幻觉 | LLM问题 |
| Temperature | 温度 | 采样参数保留 |
| Top-k / Top-p | 保留英文 | 采样策略 |
| Chain-of-Thought (CoT) | 思维链 | 推理方法 |
| ReAct | ReAct | 框架名称保留 |
| RAG | RAG (检索增强生成) | 首次出现时注释 |

#### 生物隐喻术语

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| Atoms | 原子 | 基础层级 |
| Molecules | 分子 | 组合层级 |
| Cells | 细胞 | 内存层级 |
| Organs | 器官 | 应用层级 |
| Neural Systems | 神经系统 | 认知层级 |
| Neural Fields | 神经场 | 场论层级 |

### 翻译原则

#### 1. 格式保持原则
- ✅ **保留所有 Markdown 格式**:标题、列表、表格、引用
- ✅ **保留所有代码块**:不翻译代码内容,可翻译注释
- ✅ **保留所有链接**:URL、文件路径、锚点链接
- ✅ **保留 ASCII 图表**:保持原样,可选择性翻译标签
- ✅ **保留 Mermaid 图表**:保持语法,翻译节点文本
- ✅ **保留数学公式**:LaTeX公式不翻译

#### 2. 内容翻译原则
- 📝 **准确性优先**:技术准确性 > 语言流畅性
- 📝 **术语一致性**:使用统一的术语表
- 📝 **保留专业性**:避免口语化,保持学术风格
- 📝 **上下文理解**:根据技术语境选择合适翻译
- 📝 **保留英文术语**:括号内保留重要英文原文

#### 3. 特殊处理
- 🔧 **JSON/YAML**:保留键名,可翻译值和描述
- 🔧 **代码注释**:建议翻译Python/JS注释
- 🔧 **命令行**:保留命令,可翻译说明
- 🔧 **引用文献**:保留原文,可添加中文说明
- 🔧 **作者名字**:保留英文原名

### 文件结构规范

#### 目录结构
```
/app/Context-Engineering/
├── [原文目录]
│   ├── 00_foundations/
│   ├── 10_guides_zero_to_hero/
│   └── ...
│
└── cn/                          # 中文翻译目录
    ├── TRANSLATION_GUIDE.md    # 本文件
    ├── TRANSLATION_PROGRESS.md # 进度追踪
    ├── README.md               # 翻译版主页
    ├── 00_foundations/         # 保持原结构
    ├── 10_guides_zero_to_hero/
    └── ...
```

#### 文件命名
- **保持原文件名**:不翻译文件名,便于对照
- **保持目录结构**:与原项目完全对应
- **相对路径调整**:确保内部链接正确

### 翻译策略

#### 完整翻译 (Full Translation)
适用于:
- README 文件
- 核心理论文档
- 入门指南
- 短文档(< 500行)

特点:
- 翻译所有文本内容
- 保留所有格式和代码
- 详细的概念解释

#### 框架翻译 (Framework Translation)
适用于:
- 超长技术文档(> 1000行)
- 代码示例密集的文档
- 参考手册类文档

特点:
- 翻译所有标题和概述
- 翻译关键概念和原理
- 保留完整的代码示例
- 翻译重要的可视化说明

### 质量检查清单

翻译完成后检查:

- [ ] 所有标题已翻译
- [ ] 术语使用一致
- [ ] Markdown 格式正确
- [ ] 代码块保持不变
- [ ] 链接可以正常工作
- [ ] 图表显示正常
- [ ] 数学公式正确
- [ ] 中英文之间有适当空格
- [ ] 标点符号使用正确(中文标点)
- [ ] 无明显的机翻痕迹

### 协作规范

#### Git 提交规范
```bash
# 提交信息格式
git commit -m "翻译: [目录名]/文件名 - 简要说明"

# 示例
git commit -m "翻译: 00_foundations/01_atoms_prompting.md - 完整翻译"
git commit -m "翻译: 40_reference/ - 框架翻译13个文件"
```

#### Pull Request 规范
- 标题: `[翻译] 目录名称 - 文件数量`
- 描述: 列出翻译的文件清单和策略
- 标签: `translation`, `documentation`, `zh-CN`

### 工具建议

#### 辅助工具
- **术语管理**: 维护本地术语库
- **格式检查**: Markdown linter
- **拼写检查**: 中文拼写检查工具
- **对照阅读**: 使用分屏对比原文和译文

#### AI 辅助翻译
使用 AI 工具时:
- ✅ 作为初稿辅助
- ✅ 术语标准化
- ✅ 格式保持
- ⚠️ 必须人工审核
- ⚠️ 检查技术准确性
- ⚠️ 调整语言流畅度

## 常见问题 (FAQ)

### Q: 是否需要翻译代码注释?
A: 建议翻译,特别是教学性质的注释。保持代码本身不变。

### Q: 如何处理首字母缩写?
A: 首次出现时给出中文全称和英文缩写,后续可使用缩写。
示例: `RAG (Retrieval-Augmented Generation, 检索增强生成)`

### Q: 术语表中没有的术语如何翻译?
A: 参考学术文献和标准翻译,更新术语表,保持项目内一致。

### Q: 遇到技术难点如何处理?
A: 保留英文原文,添加译注说明,或在 Issue 中讨论。

### Q: 如何确保翻译质量?
A: 遵循质量检查清单,建议同行评审,优先保证技术准确性。

## 参考资源

- [中文技术文档写作规范](https://github.com/ruanyf/document-style-guide)
- [中文文案排版指北](https://github.com/sparanoid/chinese-copywriting-guidelines)
- [术语在线](https://www.termonline.cn/) - 全国科学技术名词审定委员会
- [机器学习术语表](https://developers.google.com/machine-learning/glossary)

## 维护说明

本文档由翻译团队维护,欢迎提出改进建议。

**最后更新**: 2025-10-30
**版本**: v1.0
