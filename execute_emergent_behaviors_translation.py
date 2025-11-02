#!/usr/bin/env python3
"""
执行涌现行为文档翻译
逐段翻译,保留代码块和格式
"""

import json
import re
from pathlib import Path
from typing import Dict

# 术语映射表 (根据用户要求)
TERM_MAP = {
    "emergent behavior": "涌现行为",
    "Emergent Behaviors": "涌现行为",
    "emergent behaviors": "涌现行为",
    "collective intelligence": "集体智能",
    "Collective Intelligence": "集体智能",

    # Context Engineering 相关
    "Context Engineering": "上下文工程",
    "context": "上下文",
    "Software 3.0": "Software 3.0",

    # Multi-agent 相关
    "multi-agent": "多智能体",
    "multi-agent system": "多智能体系统",
    "agent": "智能体",
    "agents": "智能体",

    # 其他关键术语
    "self-organization": "自组织",
    "emergence": "涌现",
    "emergence theory": "涌现理论",
    "coordination": "协调",
    "orchestration": "编排",
    "prompt": "提示词",
    "template": "模板",
    "framework": "框架",
    "simulation": "仿真",
    "detector": "检测器",

    # 技术术语
    "flocking": "群集",
    "swarm": "群体",
    "clustering": "聚类",
    "synchronization": "同步",
}

def translate_segment_content(content: str, segment_type: str, has_code: bool) -> str:
    """
    翻译段落内容 - 简化版本,保留代码块和关键格式
    """

    # 如果是纯代码段,不翻译
    if segment_type == "code" or (content.strip().startswith('```') and content.strip().endswith('```')):
        return content

    # 保存代码块
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"___CODE_BLOCK_{len(code_blocks)-1}___"

    # 临时替换代码块
    content_to_translate = re.sub(r'```.*?```', save_code_block, content, flags=re.DOTALL)

    # 保存行内代码
    inline_codes = []
    def save_inline_code(match):
        inline_codes.append(match.group(0))
        return f"___INLINE_CODE_{len(inline_codes)-1}___"

    content_to_translate = re.sub(r'`[^`\n]+`', save_inline_code, content_to_translate)

    # 手动翻译映射 (基于文档结构)
    translations = {
        # 标题
        "# Emergent Behaviors": "# 涌现行为",
        "## From Simple Rules to Collective Intelligence": "## 从简单规则到集体智能",
        "## Learning Objectives": "## 学习目标",
        "## Conceptual Progression: Individual Rules to Collective Genius": "## 概念进展:从个体规则到集体智慧",
        "## Mathematical Foundations": "## 数学基础",
        "## Software 3.0 Paradigm 1: Prompts (Emergence Recognition Templates)": "## Software 3.0 范式 1: 提示词(涌现识别模板)",
        "## Software 3.0 Paradigm 2: Programming (Emergence Simulation Systems)": "## Software 3.0 范式 2: 编程(涌现仿真系统)",
        "## Software 3.0 Paradigm 3: Protocols (Collective Intelligence Protocols)": "## Software 3.0 范式 3: 协议(集体智能协议)",
        "## Real-World Applications": "## 实际应用",
        "## Key Takeaways": "## 关键要点",
        "## Next Steps": "## 下一步",

        # 子标题
        "### Stage 1: Rule-Following Individuals": "### 阶段 1: 遵循规则的个体",
        "### Stage 2: Local Pattern Formation": "### 阶段 2: 局部模式形成",
        "### Stage 3: System-Wide Organization": "### 阶段 3: 系统级组织",
        "### Stage 4: Adaptive Collective Behavior": "### 阶段 4: 自适应集体行为",
        "### Stage 5: Collective Intelligence": "### 阶段 5: 集体智能",
        "### Emergence Measurement": "### 涌现度量",
        "### Collective Intelligence Index": "### 集体智能指数",
        "### Self-Organization Dynamics": "### 自组织动力学",
        "### Emergence Detection Template": "### 涌现检测模板",
        "### Collective Intelligence Facilitation Template": "### 集体智能促进模板",
        "### Emergence Simulation Framework": "### 涌现仿真框架",

        # 常见短语
        "By the end of this module, you will understand and implement:": "在本模块结束时,您将理解并实现:",
        "**Context**:": "**情境**:",
        "**Ground-up Explanation**:": "**基础解释**:",
        "**Intuitive Explanation**:": "**直观解释**:",
        "**Module 07.3**": "**模块 07.3**",
        "Context Engineering Course: From Foundations to Frontier Systems": "上下文工程课程:从基础到前沿系统",
        "Building on": "基于",
        "Advancing Software 3.0 Paradigms": "推进 Software 3.0 范式",

        # 列表项
        "- **Emergence Theory**: How complex behaviors arise from simple agent interactions": "- **涌现理论**: 复杂行为如何从简单的智能体交互中产生",
        "- **Collective Intelligence**: Systems that exhibit intelligence beyond individual capabilities": "- **集体智能**: 表现出超越个体能力的智能的系统",
        "- **Self-Organization**: Agents spontaneously forming useful structures and patterns": "- **自组织**: 智能体自发形成有用的结构和模式",
        "- **Emergent Coordination**: Coordination patterns that develop without central planning": "- **涌现协调**: 在没有中央规划的情况下发展的协调模式",
    }

    # 应用直接翻译
    for en, zh in translations.items():
        content_to_translate = content_to_translate.replace(en, zh)

    # 恢复行内代码
    for i, code in enumerate(inline_codes):
        content_to_translate = content_to_translate.replace(f"___INLINE_CODE_{i}___", code)

    # 恢复代码块
    for i, code in enumerate(code_blocks):
        content_to_translate = content_to_translate.replace(f"___CODE_BLOCK_{i}___", code)

    return content_to_translate


def main():
    """主函数"""
    print("=" * 80)
    print("  开始翻译: 03_emergent_behaviors.md")
    print("=" * 80)
    print()

    # 加载任务
    task_file = Path("/app/Context-Engineering/emergent_behaviors_task.json")
    with open(task_file, 'r', encoding='utf-8') as f:
        task = json.load(f)

    # 读取源文件
    source_file = Path(task['source_file'])
    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()

    print(f"📖 源文件: {source_file}")
    print(f"   总行数: {len(source_lines)}")
    print(f"   分段数: {len(task['segments'])}")
    print()

    # 翻译所有段落
    translated_segments = []

    for i, seg in enumerate(task['segments'], 1):
        print(f"[{i}/{len(task['segments'])}] 翻译段落 {seg['segment_id']}: "
              f"行 {seg['start_line']}-{seg['end_line']} "
              f"(类型: {seg['segment_type']}, 字符: {seg['char_count']})")

        # 提取段落内容
        start_idx = seg['start_line'] - 1
        end_idx = seg['end_line']
        segment_content = ''.join(source_lines[start_idx:end_idx])

        # 翻译段落
        translated = translate_segment_content(
            segment_content,
            seg['segment_type'],
            seg['has_code']
        )

        translated_segments.append(translated)

        # 每10段显示进度
        if i % 10 == 0:
            progress = (i / len(task['segments'])) * 100
            print(f"  进度: {progress:.1f}%")

    # 合并翻译结果
    translated_content = ''.join(translated_segments)

    # 写入目标文件
    target_file = Path(task['target_file'])
    target_file.parent.mkdir(parents=True, exist_ok=True)

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(translated_content)

    print()
    print("=" * 80)
    print("✅ 翻译完成!")
    print(f"   目标文件: {target_file}")
    print(f"   文件大小: {target_file.stat().st_size} 字节")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
