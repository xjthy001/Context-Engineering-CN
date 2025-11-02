import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid(defineConfig({
  title: "Context Engineering 中文版",
  description: "上下文工程 - 超越提示工程的AI交互新范式",
  lang: 'zh-CN',

  // 基础路径
  base: '/Context-Engineering-CN/',

  // 忽略有问题的文件
  srcExclude: ['**/PODCASTS/**', '**/PROJECT_COMPLETION_SUMMARY.md'],

  // 忽略死链接
  ignoreDeadLinks: true,

  // 主题配置
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '课程', link: '/00_COURSE/README' },
      { text: '基础理论', link: '/00_foundations/README' },
      { text: '无代码方案', link: '/NOCODE/README' },
      {
        text: '更多',
        items: [
          { text: '指南', link: '/10_guides/README' },
          { text: '模板', link: '/20_templates/README' },
          { text: '示例', link: '/30_examples/README' },
          { text: '参考', link: '/40_reference/README' }
        ]
      }
    ],

    sidebar: {
      '/00_COURSE/': [
        {
          text: '📚 课程体系',
          items: [
            { text: '课程概览', link: '/00_COURSE/README' },
            { text: '课程大纲', link: '/00_COURSE/COURSE_OUTLINE' }
          ]
        },
        {
          text: '00. 数学基础',
          collapsed: false,
          items: [
            { text: '概览', link: '/00_COURSE/00_mathematical_foundations/README' },
            { text: '01. 上下文形式化', link: '/00_COURSE/00_mathematical_foundations/01_context_formalization' },
            { text: '02. 优化理论', link: '/00_COURSE/00_mathematical_foundations/02_optimization_theory' },
            { text: '03. 信息论', link: '/00_COURSE/00_mathematical_foundations/03_information_theory' },
            { text: '04. 贝叶斯推理', link: '/00_COURSE/00_mathematical_foundations/04_bayesian_inference' }
          ]
        },
        {
          text: '01. 上下文检索与生成',
          collapsed: false,
          items: [
            { text: '概览', link: '/00_COURSE/01_context_retrieval_generation/README' },
            { text: '00. 模块概览', link: '/00_COURSE/01_context_retrieval_generation/00_overview' },
            { text: '01. 提示工程', link: '/00_COURSE/01_context_retrieval_generation/01_prompt_engineering' },
            { text: '02. 外部知识', link: '/00_COURSE/01_context_retrieval_generation/02_external_knowledge' },
            { text: '03. 动态组装', link: '/00_COURSE/01_context_retrieval_generation/03_dynamic_assembly' }
          ]
        },
        {
          text: '02. 上下文处理',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/02_context_processing/README' },
            { text: '00. 模块概览', link: '/00_COURSE/02_context_processing/00_overview' },
            { text: '01. 长上下文处理', link: '/00_COURSE/02_context_processing/01_long_context_processing' },
            { text: '02. 自我精炼', link: '/00_COURSE/02_context_processing/02_self_refinement' },
            { text: '03. 多模态上下文', link: '/00_COURSE/02_context_processing/03_multimodal_context' },
            { text: '04. 结构化上下文', link: '/00_COURSE/02_context_processing/04_structured_context' }
          ]
        },
        {
          text: '03. 上下文管理',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/03_context_management/README' },
            { text: '00. 模块概览', link: '/00_COURSE/03_context_management/00_overview' },
            { text: '01. 基本约束', link: '/00_COURSE/03_context_management/01_fundamental_constraints' },
            { text: '02. 记忆层次', link: '/00_COURSE/03_context_management/02_memory_hierarchies' },
            { text: '03. 压缩技术', link: '/00_COURSE/03_context_management/03_compression_techniques' },
            { text: '04. 优化策略', link: '/00_COURSE/03_context_management/04_optimization_strategies' }
          ]
        },
        {
          text: '04. 检索增强生成 (RAG)',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/04_retrieval_augmented_generation/README' },
            { text: '00. RAG基础', link: '/00_COURSE/04_retrieval_augmented_generation/00_rag_fundamentals' },
            { text: '01. 模块化架构', link: '/00_COURSE/04_retrieval_augmented_generation/01_modular_architectures' },
            { text: '02. 代理式RAG', link: '/00_COURSE/04_retrieval_augmented_generation/02_agentic_rag' },
            { text: '03. 图增强RAG', link: '/00_COURSE/04_retrieval_augmented_generation/03_graph_enhanced_rag' },
            { text: '04. 高级应用', link: '/00_COURSE/04_retrieval_augmented_generation/04_advanced_applications' }
          ]
        },
        {
          text: '05. 记忆系统',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/05_memory_systems/README' },
            { text: '00. 记忆架构', link: '/00_COURSE/05_memory_systems/00_memory_architectures' },
            { text: '01. 持久化记忆', link: '/00_COURSE/05_memory_systems/01_persistent_memory' },
            { text: '02. 记忆增强智能体', link: '/00_COURSE/05_memory_systems/02_memory_enhanced_agents' },
            { text: '03. 评估挑战', link: '/00_COURSE/05_memory_systems/03_evaluation_challenges' },
            { text: '04. 重构式记忆', link: '/00_COURSE/05_memory_systems/04_reconstructive_memory' }
          ]
        },
        {
          text: '06. 工具集成推理',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/06_tool_integrated_reasoning/README' },
            { text: '00. 函数调用', link: '/00_COURSE/06_tool_integrated_reasoning/00_function_calling' },
            { text: '01. 工具集成', link: '/00_COURSE/06_tool_integrated_reasoning/01_tool_integration' },
            { text: '02. 智能体环境', link: '/00_COURSE/06_tool_integrated_reasoning/02_agent_environment' },
            { text: '03. 推理框架', link: '/00_COURSE/06_tool_integrated_reasoning/03_reasoning_frameworks' }
          ]
        },
        {
          text: '07. 多智能体系统',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/07_multi_agent_systems/README' },
            { text: '00. 通信协议', link: '/00_COURSE/07_multi_agent_systems/00_communication_protocols' },
            { text: '01. 编排机制', link: '/00_COURSE/07_multi_agent_systems/01_orchestration_mechanisms' },
            { text: '02. 协调策略', link: '/00_COURSE/07_multi_agent_systems/02_coordination_strategies' },
            { text: '03. 涌现行为', link: '/00_COURSE/07_multi_agent_systems/03_emergent_behaviors' }
          ]
        },
        {
          text: '08. 场论集成',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/08_field_theory_integration/README' },
            { text: '00. 神经场基础', link: '/00_COURSE/08_field_theory_integration/00_neural_field_foundations' },
            { text: '01. 吸引子动力学', link: '/00_COURSE/08_field_theory_integration/01_attractor_dynamics' },
            { text: '02. 场共振', link: '/00_COURSE/08_field_theory_integration/02_field_resonance' },
            { text: '03. 边界管理', link: '/00_COURSE/08_field_theory_integration/03_boundary_management' }
          ]
        },
        {
          text: '09. 评估方法论',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/09_evaluation_methodologies/README' },
            { text: '00. 评估框架', link: '/00_COURSE/09_evaluation_methodologies/00_evaluation_frameworks' },
            { text: '01. 组件评估', link: '/00_COURSE/09_evaluation_methodologies/01_component_assessment' },
            { text: '02. 系统集成', link: '/00_COURSE/09_evaluation_methodologies/02_system_integration' },
            { text: '03. 基准设计', link: '/00_COURSE/09_evaluation_methodologies/03_benchmark_design' }
          ]
        },
        {
          text: '10. 编排顶点项目',
          collapsed: true,
          items: [
            { text: '概览', link: '/00_COURSE/10_orchestration_capstone/README' },
            { text: '00. 项目概览', link: '/00_COURSE/10_orchestration_capstone/00_capstone_overview' }
          ]
        }
      ],

      '/00_foundations/': [
        {
          text: '🧬 基础理论',
          items: [
            { text: '理论概览', link: '/00_foundations/README' },
            { text: '01. 原子: 提示', link: '/00_foundations/01_atoms_prompting' },
            { text: '02. 分子: 少样本学习', link: '/00_foundations/02_molecules_few_shot' },
            { text: '03. 细胞: 记忆+智能体', link: '/00_foundations/03_cells_memory_agents' },
            { text: '04. 器官: 多智能体', link: '/00_foundations/04_organs_multi_agent' },
            { text: '05. 神经系统: 认知工具', link: '/00_foundations/05_nervous_system_cognitive_tools' },
            { text: '06. 场论: 神经场', link: '/00_foundations/06_unified_field_theory' },
            { text: '07. 神经场基础', link: '/00_foundations/07_neural_fields' },
            { text: '08. 吸引子动力学', link: '/00_foundations/08_attractor_dynamics' },
            { text: '09. 符号机制', link: '/00_foundations/09_symbolic_mechanisms' },
            { text: '10. 持久化系统', link: '/00_foundations/10_persistence' },
            { text: '11. 量子语义', link: '/00_foundations/11_quantum_semantics' },
            { text: '12. 元递归', link: '/00_foundations/12_meta_recursion' }
          ]
        }
      ],

      '/NOCODE/': [
        {
          text: '🎨 无代码方案',
          items: [
            { text: 'NOCODE概览', link: '/NOCODE/README' },
            { text: '00. 心智模型', link: '/NOCODE/00_mental_models/README' },
            { text: '01. 对话协议', link: '/NOCODE/01_conversation_protocols/README' },
            { text: '02. 文档协议', link: '/NOCODE/02_documentation_protocols/README' },
            { text: '03. 创意协议', link: '/NOCODE/03_creative_protocols/README' },
            { text: '04. 研究协议', link: '/NOCODE/04_research_protocols/README' },
            { text: '05. 问题解决', link: '/NOCODE/05_problem_solving/README' },
            { text: '06. 决策协议', link: '/NOCODE/06_decision_protocols/README' },
            { text: '07. 学习协议', link: '/NOCODE/07_learning_protocols/README' },
            { text: '08. 协作协议', link: '/NOCODE/08_collaboration_protocols/README' },
            { text: '09. 个人发展', link: '/NOCODE/09_personal_development/README' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/xjthy001/Context-Engineering-CN' }
    ],

    // 搜索配置
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换'
                }
              }
            }
          }
        }
      }
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/xjthy001/Context-Engineering-CN/edit/main/:path',
      text: '在 GitHub 上编辑此页'
    },

    // 页脚
    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2025 Context Engineering 中文版'
    },

    // 文档页脚
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    // 大纲标题
    outlineTitle: '页面导航',

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    }
  },

  // Markdown 配置
  markdown: {
    lineNumbers: true,
    math: true  // 启用数学公式支持
    // 注意: ascii 和 math 语言的警告是正常的，不影响构建
    // VitePress 会自动 fallback 到 text 显示
  },

  // Mermaid 配置
  mermaid: {
    // Mermaid 选项
  },

  // 头部配置
  head: [
    ['link', { rel: 'icon', href: '/Context-Engineering-CN/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }],
    ['meta', { name: 'keywords', content: 'Context Engineering, 上下文工程, AI, Prompt Engineering, RAG, 多智能体' }]
  ]
}))
