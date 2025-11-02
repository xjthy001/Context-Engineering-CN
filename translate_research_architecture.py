#!/usr/bin/env python3
"""
专门用于翻译 research-architecture.md 的脚本
分段处理,保持代码块和格式完整性
"""

import re
from pathlib import Path

# 术语映射
TERMS = {
    "research": "研究",
    "hypothesis": "假设",
    "hypotheses": "假设",
    "analysis": "分析",
    "synthesis": "综合",
    "knowledge": "知识",
    "literature": "文献",
    "finding": "发现",
    "findings": "发现",
    "evidence": "证据",
    "theory": "理论",
    "theoretical": "理论的",
    "model": "模型",
    "framework": "框架",
    "architecture": "架构",
    "protocol": "协议",
    "cognitive": "认知",
    "semantic": "语义",
    "quantum": "量子",
    "field": "场",
    "attractor": "吸引子",
    "boundary": "边界",
    "gap": "缺口",
    "concept": "概念",
    "relationship": "关系",
    "pattern": "模式",
    "contradiction": "矛盾",
    "opportunity": "机会",
}

def preserve_code_blocks(text):
    """提取并保护代码块"""
    code_blocks = []
    pattern = r'(```[\s\S]*?```)'

    def replacer(match):
        code_blocks.append(match.group(1))
        return f'<<<CODE_BLOCK_{len(code_blocks)-1}>>>'

    protected_text = re.sub(pattern, replacer, text)
    return protected_text, code_blocks

def restore_code_blocks(text, code_blocks):
    """恢复代码块"""
    for i, block in enumerate(code_blocks):
        text = text.replace(f'<<<CODE_BLOCK_{i}>>>', block)
    return text

def get_segment_ranges():
    """定义翻译段落范围(行号)"""
    return [
        (1, 800, "标题、概述、理论基础、核心组件开始"),
        (801, 1600, "核心组件、研究协议外壳"),
        (1601, 2400, "认知工具、量子语义、实现模式"),
        (2401, 2682, "案例研究、元研究层、结论"),
    ]

def read_segment(file_path, start_line, end_line):
    """读取指定行范围的内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return ''.join(lines[start_line-1:end_line])

def main():
    source_file = Path("/app/Context-Engineering/cognitive-tools/cognitive-architectures/research-architecture.md")
    target_file = Path("/app/Context-Engineering/cn/cognitive-tools/cognitive-architectures/research-architecture.md")

    print(f"源文件: {source_file}")
    print(f"目标文件: {target_file}")
    print(f"\n由于文件太大(2682行, 138KB),需要分段翻译")
    print(f"建议使用Claude API或者手动分段处理\n")

    # 显示段落信息
    segments = get_segment_ranges()
    print("建议的翻译段落:")
    for i, (start, end, desc) in enumerate(segments, 1):
        segment_content = read_segment(source_file, start, min(end, start+50))
        lines_count = end - start + 1
        print(f"\n段落 {i}: 行 {start}-{end} ({lines_count} 行)")
        print(f"描述: {desc}")
        print(f"开始内容预览:")
        print(segment_content[:200] + "...")

    print("\n" + "="*80)
    print("翻译方案:")
    print("1. 手动方案: 使用 Read 工具分段读取源文件,逐段翻译后拼接")
    print("2. API方案: 使用 intelligent_translator.py 配合AI API")
    print("3. 混合方案: 关键段落手动翻译,其余部分使用工具辅助")
    print("="*80)

if __name__ == "__main__":
    main()
