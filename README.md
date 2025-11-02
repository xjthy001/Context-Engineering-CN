# Context Engineering 中文版

> **上下文工程 - 超越提示工程的AI交互新范式**

[![翻译进度](https://img.shields.io/badge/翻译进度-100%25-brightgreen)](./TRANSLATION_PROGRESS.md)
[![文件数量](https://img.shields.io/badge/已翻译文件-123-blue)](./TRANSLATION_PROGRESS.md)
[![状态](https://img.shields.io/badge/状态-完成-success)](./PROJECT_COMPLETION_SUMMARY.md)
[![最后更新](https://img.shields.io/badge/最后更新-2025--11--02-orange)](./TRANSLATION_PROGRESS.md)

**[查看原项目](https://github.com/davidkimai/Context-Engineering)** | **[翻译指南](./TRANSLATION_GUIDE.md)** | **[翻译进度](./TRANSLATION_PROGRESS.md)**

---

## 💡 什么是上下文工程？

**上下文工程（Context Engineering）**是一门关于如何系统化地设计、构建和优化大型语言模型（LLM）输入上下文的科学与艺术。它不仅仅是编写提示词，而是：

🎯 **核心理念**
- 将上下文视为可以**数学形式化**的对象：`C = A(c₁, c₂, ..., cₙ)`
- 通过**检索、组装、处理、优化**等系统化方法，最大化模型的推理能力
- 从离散的token序列到连续的**语义场**，从静态提示到动态的**自适应系统**

📚 **你将学到什么？**

通过本项目，你将掌握：

1. **理论基础** (00_foundations + 00_COURSE)
   - 上下文的数学形式化 `C = A(c₁, c₂, ..., cₙ)`
   - 从原子级到场论的6层递进框架
   - 信息论、贝叶斯推理、优化理论在上下文中的应用

2. **系统技术** (12周完整课程)
   - **RAG（检索增强生成）**: 动态获取外部知识
   - **记忆系统**: 长期知识存储与检索
   - **多智能体协作**: 复杂任务分解与协调
   - **工具集成**: 让AI调用外部工具和API
   - **场论整合**: 将上下文理解为连续场的前沿视角

3. **实践技能** (NOCODE + 模板)
   - **零编程方案**: 无需代码也能应用高级技术
   - **即用模板**: 20+种提示模板和认知协议
   - **评估方法**: 如何衡量和优化系统表现

4. **前沿视野**
   - 从Software 1.0（手写规则）→ 2.0（机器学习）→ **3.0（上下文编程）**
   - 涌现智能、自我改进系统、AI-人类共生的未来

🎓 **适合谁？**

- **AI开发者**: 构建更强大的LLM应用系统
- **研究者**: 理解上下文工程的理论基础
- **产品经理**: 设计智能产品的上下文交互
- **学习者**: 从零系统学习现代AI工程实践

📖 **为什么需要系统学习？**

简单的提示工程（Prompt Engineering）已经不够了。现代AI系统需要：
- 处理**超长上下文**（100K+ tokens）
- 整合**多模态信息**（文本、图像、音频）
- 实现**持久记忆**和**工具使用**
- 支持**多智能体协作**和**自主决策**

这些都需要**系统化的上下文工程能力**。

---

## 🌐 在线阅读

**网站地址**: https://xjthy001.github.io/Context-Engineering-CN/

完整的交互式文档网站，包含搜索、导航和美化界面。

---

## 🎯 关于本项目

这是 [Context Engineering](https://github.com/davidkimai/Context-Engineering) 项目的完整中文翻译版本。

### ✨ 特色

- ✅ **100%完成**: 123个文件，750K+中文字符
- ✅ **系统课程**: 12周完整学习路径
- ✅ **理论基础**: 从原子到场论的6层递进
- ✅ **实践工具**: 模板、协议、代码示例
- ✅ **无代码方案**: 零编程也能掌握

### 📊 内容结构

- **00_COURSE/**: 12周系统课程
- **00_foundations/**: 理论基础（原子→场论）
- **NOCODE/**: 无代码实践方案
- **20_templates/**: 可复用模板库
- **cognitive-tools/**: 认知工具集
- **40_reference/**: 学术参考资源

---

## 🚀 快速开始

### 方式 1: 在线阅读（推荐）

访问网站: https://xjthy001.github.io/Context-Engineering-CN/

### 方式 2: 本地克隆

```bash
git clone https://github.com/xjthy001/Context-Engineering-CN.git
cd Context-Engineering-CN
```

### 方式 3: 运行本地网站

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 访问 http://localhost:5173
```

---

## 📚 学习路径

### 🌱 初学者路径
1. 从 [基础理论](./00_foundations/README.md) 开始
2. 学习 [NOCODE方案](./NOCODE/README.md)
3. 完成前4周课程

### 🔬 研究者路径
1. 深入 [数学基础](./00_COURSE/00_mathematical_foundations/README.md)
2. 研究 [神经场论](./00_COURSE/08_neural_field_theory/README.md)
3. 阅读 [学术参考](./40_reference/README.md)

### 🏗️ 工程师路径
1. 学习 [RAG系统](./00_COURSE/04_retrieval_augmented_generation/README.md)
2. 实践 [实验室项目](./00_COURSE/)
3. 使用 [模板和工具](./20_templates/README.md)

---

## 🎓 课程体系

完整的12周课程涵盖：

1. **Week 1-2**: 数学基础与上下文形式化
2. **Week 3-4**: 上下文检索与生成
3. **Week 5-6**: 上下文处理与管理
4. **Week 7-8**: RAG系统与记忆架构
5. **Week 9-10**: 工具集成与多智能体
6. **Week 11-12**: 神经场论与评估

详见 [课程大纲](./00_COURSE/COURSE_OUTLINE.md)

---

## 🛠️ 资源

### 文档
- [翻译指南](./TRANSLATION_GUIDE.md) - 翻译规范和术语表
- [翻译进度](./TRANSLATION_PROGRESS.md) - 详细进度追踪
- [完成总结](./PROJECT_COMPLETION_SUMMARY.md) - 项目总结
- [部署指南](./DEPLOYMENT.md) - 网站部署说明

### 配置文件
- [CLAUDE.md](./CLAUDE.md) - Claude 认知操作系统配置
- [GEMINI.md](./GEMINI.md) - Gemini 配置指南

### 代码和模板
- [20_templates/](./20_templates/) - 可复用模板库
- [cognitive-tools/](./cognitive-tools/) - 认知工具集
- [context-schemas/](./context-schemas/) - 上下文Schema定义

---

## 📈 翻译统计

- **总文件数**: 123个
- **总字符数**: 750,000+ 中文字符
- **代码文件**: 127个（Python、YAML、JSON等）
- **完成时间**: 3小时（并行AI代理策略）
- **完成度**: 100% ✅

---

## 🤝 贡献

欢迎贡献！

- 🐛 [报告问题](https://github.com/xjthy001/Context-Engineering-CN/issues)
- 💡 [提出建议](https://github.com/xjthy001/Context-Engineering-CN/discussions)
- 📝 提交改进

---

## 📜 许可证

本项目采用 [MIT 许可证](./LICENSE)

---

## 🙏 致谢

- **原作者**: Andrej Karpathy 及 Context Engineering 团队
- **研究机构**: IBM、普林斯顿、MIT、斯坦福等
- **学术支撑**: 1400+ 篇前沿研究论文

---

## 📞 联系

- **GitHub**: https://github.com/xjthy001/Context-Engineering-CN
- **Issues**: https://github.com/xjthy001/Context-Engineering-CN/issues
- **网站**: https://xjthy001.github.io/Context-Engineering-CN/

---

**开始你的上下文工程之旅！** 🚀
