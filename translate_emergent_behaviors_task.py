#!/usr/bin/env python3
"""
专门翻译 03_emergent_behaviors.md 的脚本
使用增量翻译方式,通过Claude API逐段翻译
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/app/Context-Engineering')

from pathlib import Path
from parallel_translate_manager import ParallelTranslationManager

def main():
    """主函数"""
    print("=" * 80)
    print("  涌现行为 (Emergent Behaviors) 文档翻译")
    print("=" * 80)
    print()

    # 源文件和目标文件
    source_file = Path("/app/Context-Engineering/00_COURSE/07_multi_agent_systems/03_emergent_behaviors.md")
    target_file = Path("/app/Context-Engineering/cn/00_COURSE/07_multi_agent_systems/03_emergent_behaviors.md")

    print(f"源文件: {source_file}")
    print(f"目标文件: {target_file}")
    print()

    if not source_file.exists():
        print(f"❌ 错误: 源文件不存在")
        return 1

    # 创建目标目录
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # 创建翻译管理器
    manager = ParallelTranslationManager()

    # 为此文件创建翻译任务
    task = manager._create_task_for_file(
        task_id=1,
        chapter="07_multi_agent_systems",
        source_file=source_file,
        target_file=target_file
    )

    print(f"📊 文档统计:")
    print(f"   - 总行数: {task.total_lines}")
    print(f"   - 总字符数: {task.total_chars}")
    print(f"   - 分段数: {len(task.segments)}")
    print(f"   - 优先级: {task.priority}")
    print()

    print(f"📝 翻译段落详情:")
    for i, seg in enumerate(task.segments, 1):
        print(f"   段落 {i}: 行 {seg['start_line']}-{seg['end_line']} "
              f"({seg['char_count']} 字符, 类型: {seg['segment_type']}, "
              f"包含代码: {'是' if seg['has_code'] else '否'})")
    print()

    # 保存任务信息
    import json
    task_file = Path("/app/Context-Engineering/emergent_behaviors_task.json")
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump({
            'task_id': task.task_id,
            'source_file': task.source_file,
            'target_file': task.target_file,
            'chapter': task.chapter,
            'file_name': task.file_name,
            'total_lines': task.total_lines,
            'total_chars': task.total_chars,
            'segments': task.segments,
            'status': task.status,
            'priority': task.priority
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 任务信息已保存到: {task_file}")
    print()
    print("下一步:")
    print("  1. 查看 emergent_behaviors_task.json 了解分段详情")
    print("  2. 使用 intelligent_translator.py 进行实际翻译")
    print("  3. 或者手动逐段翻译以确保质量")

    return 0

if __name__ == "__main__":
    sys.exit(main())
