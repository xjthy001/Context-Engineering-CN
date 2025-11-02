#!/usr/bin/env python3
"""
自动生成的翻译子任务脚本
任务ID: TASK_021
源文件: /app/Context-Engineering/00_COURSE/03_context_management/00_overview.md
目标文件: /app/Context-Engineering/cn/00_COURSE/03_context_management/00_overview.md
章节: 03_context_management
段落数: 23
"""

import sys
from pathlib import Path

# 任务信息
TASK_ID = "TASK_021"
SOURCE_FILE = Path("/app/Context-Engineering/00_COURSE/03_context_management/00_overview.md")
TARGET_FILE = Path("/app/Context-Engineering/cn/00_COURSE/03_context_management/00_overview.md")
TOTAL_SEGMENTS = 23

def translate_segment(segment_id, content):
    """
    翻译单个段落
    这里需要调用实际的翻译服务或AI模型
    """
    # TODO: 实现实际的翻译逻辑
    # 这里是占位符,实际使用时需要调用翻译API
    print(f"  翻译段落 {segment_id}/{TOTAL_SEGMENTS}...")

    # 简单的标记处理(保持代码块不变)
    if '```' in content:
        # 代码块需要特殊处理
        return content  # 暂时保持原样

    # 实际翻译逻辑应该在这里
    return content

def main():
    print(f"开始翻译任务: {TASK_ID}")
    print(f"文件: {SOURCE_FILE.name}")

    # 读取源文件
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分段翻译
    segments = [{'segment_id': 1, 'start_line': 1, 'end_line': 5, 'char_count': 246, 'has_code': False, 'segment_type': 'header', 'content': '# Context Management: The Software 3.0 Revolution\n> "It is the mark of an educated mind to be able to entertain a thought without accepting it."\n>\n> — [Aristotle](https://www.goodreads.com/quotes/1629...'}, {'segment_id': 2, 'start_line': 6, 'end_line': 33, 'char_count': 1043, 'has_code': True, 'segment_type': 'header', 'content': '## The Shift: From Code to Context\n> [**Software Is Changing (Again) Talk @YC AI Startup School—Andrej Karpathy**](https://www.youtube.com/watch?v=LCEmiRjPEtQ)\n\nWe are witnessing the emergence of [**S...'}, {'segment_id': 3, 'start_line': 34, 'end_line': 35, 'char_count': 42, 'has_code': False, 'segment_type': 'header', 'content': "## The Three Pillars: A Beginner's Guide\n\n"}, {'segment_id': 4, 'start_line': 36, 'end_line': 42, 'char_count': 285, 'has_code': False, 'segment_type': 'header', 'content': '### What Are These Three Things?\n\n**Think of building a house:**\n- **PROMPTS** = Talking to the architect (communication)\n- **PROGRAMMING** = The construction tools and techniques (implementation)  \n-...'}, {'segment_id': 5, 'start_line': 43, 'end_line': 92, 'char_count': 1222, 'has_code': True, 'segment_type': 'header', 'content': '### Pillar 1: PROMPT TEMPLATES - The Communication Layer\n\n**What is a Prompt Template?**\nA prompt template is a reusable pattern for communicating with an AI system. Instead of writing unique prompts ...'}, {'segment_id': 6, 'start_line': 93, 'end_line': 181, 'char_count': 3082, 'has_code': True, 'segment_type': 'header', 'content': '### Pillar 2: PROGRAMMING - The Implementation Layer\n\nProgramming provides the computational infrastructure that supports context management.\n\n**Traditional Context Management Code:**\n```python\nclass ...'}, {'segment_id': 7, 'start_line': 182, 'end_line': 182, 'char_count': 1, 'has_code': False, 'segment_type': 'content', 'content': '\n'}, {'segment_id': 8, 'start_line': 183, 'end_line': 346, 'char_count': 5532, 'has_code': True, 'segment_type': 'header', 'content': '### Pillar 3: PROTOCOLS - The Orchestration Layer\n\n**What is a Protocol? (Simple Explanation)**\n\nA protocol is like a **recipe that thinks**. Just as a cooking recipe tells you:\n- What ingredients you...'}, {'segment_id': 9, 'start_line': 347, 'end_line': 347, 'char_count': 1, 'has_code': False, 'segment_type': 'content', 'content': '\n'}, {'segment_id': 10, 'start_line': 348, 'end_line': 349, 'char_count': 49, 'has_code': False, 'segment_type': 'header', 'content': '## The Integration: How All Three Work Together\n\n'}, {'segment_id': 11, 'start_line': 350, 'end_line': 487, 'char_count': 4385, 'has_code': True, 'segment_type': 'header', 'content': "### Real-World Example: Code Review System\n\nLet's build a comprehensive code review system that demonstrates all three pillars working together.\n\n**1. PROMPT TEMPLATES (Communication Layer):**\n\n```pyt..."}, {'segment_id': 12, 'start_line': 488, 'end_line': 657, 'char_count': 7395, 'has_code': True, 'segment_type': 'code', 'content': '\n**3. PROTOCOLS (Orchestration Layer):**\n\n```\n/code.review.comprehensive{\n    intent="Perform thorough, multi-dimensional code review with adaptive focus based on code characteristics",\n    \n    input...'}, {'segment_id': 13, 'start_line': 658, 'end_line': 718, 'char_count': 2058, 'has_code': True, 'segment_type': 'code', 'content': '\n**4. THE COMPLETE INTEGRATION:**\n\n```python\n# This is how all three pillars work together in practice:\n\nclass Software3CodeReviewer:\n    """Complete integration of prompts, programming, and protocols...'}, {'segment_id': 14, 'start_line': 719, 'end_line': 720, 'char_count': 33, 'has_code': False, 'segment_type': 'header', 'content': '## Why This Integration Matters\n\n'}, {'segment_id': 15, 'start_line': 721, 'end_line': 726, 'char_count': 201, 'has_code': False, 'segment_type': 'header', 'content': '### Traditional Approach Problems:\n- **Rigid**: Same analysis every time\n- **Inefficient**: Lots of redundant work\n- **Limited**: Single perspective\n- **Hard to Scale**: Manual customization required\n...'}, {'segment_id': 16, 'start_line': 727, 'end_line': 732, 'char_count': 289, 'has_code': False, 'segment_type': 'header', 'content': '### Software 3.0 Solution Benefits:\n- **Adaptive**: Changes based on context and requirements\n- **Efficient**: Reuses templates and context intelligently  \n- **Comprehensive**: Multiple perspectives i...'}, {'segment_id': 17, 'start_line': 733, 'end_line': 734, 'char_count': 33, 'has_code': False, 'segment_type': 'header', 'content': '## Key Principles for Beginners\n\n'}, {'segment_id': 18, 'start_line': 735, 'end_line': 751, 'char_count': 404, 'has_code': True, 'segment_type': 'header', 'content': '### 1. Start Simple, Build Complexity Gradually\n```\nLevel 1: Basic Prompt Templates\n├─ Fixed templates with placeholders\n└─ Simple substitution logic\n\nLevel 2: Programming Integration  \n├─ Dynamic tem...'}, {'segment_id': 19, 'start_line': 752, 'end_line': 756, 'char_count': 224, 'has_code': False, 'segment_type': 'header', 'content': '### 2. Think in Layers\n- **Communication Layer**: How you talk to the AI (prompts/templates)\n- **Logic Layer**: How you process information (programming)\n- **Orchestration Layer**: How you coordinate ...'}, {'segment_id': 20, 'start_line': 757, 'end_line': 761, 'char_count': 172, 'has_code': False, 'segment_type': 'header', 'content': '### 3. Focus on Reusability\n- Templates should work across similar scenarios\n- Code should be modular and composable\n- Protocols should be adaptable to different contexts\n\n'}, {'segment_id': 21, 'start_line': 762, 'end_line': 766, 'char_count': 190, 'has_code': False, 'segment_type': 'header', 'content': '### 4. Optimize for Context\n- Everything should be context-aware\n- Information should flow efficiently between layers\n- The system should adapt based on available resources and constraints\n\n'}, {'segment_id': 22, 'start_line': 767, 'end_line': 776, 'char_count': 538, 'has_code': False, 'segment_type': 'header', 'content': '## Next Steps in This Course\n\nThe following sections will dive deeper into:\n- **Fundamental Constraints**: How computational limits shape our approach\n- **Memory Hierarchies**: Multi-level storage and...'}, {'segment_id': 23, 'start_line': 777, 'end_line': 779, 'char_count': 287, 'has_code': False, 'segment_type': 'content', 'content': '---\n\n*This overview establishes the foundation for understanding how prompts, programming, and protocols work together to create sophisticated, adaptable, and efficient context management systems. The...'}]
    translated_segments = []

    for seg_info in segments:
        seg_id = seg_info['segment_id']
        start = seg_info['start_line']
        end = seg_info['end_line']

        # 提取段落内容
        lines = content.splitlines(keepends=True)
        segment_content = ''.join(lines[start-1:end])

        # 翻译
        translated = translate_segment(seg_id, segment_content)
        translated_segments.append(translated)

    # 合并翻译结果
    final_translation = ''.join(translated_segments)

    # 确保目标目录存在
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 写入目标文件
    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        f.write(final_translation)

    print(f"✅ 翻译完成: {TARGET_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
