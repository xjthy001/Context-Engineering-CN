#!/usr/bin/env python3
"""
专门翻译 02_self_refinement.md 文件的脚本
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from parallel_translate_manager import ParallelTranslationManager

def main():
    """翻译 02_self_refinement.md"""

    print("=" * 60)
    print("开始翻译 02_self_refinement.md")
    print("=" * 60)

    source_file = Path("/app/Context-Engineering/00_COURSE/02_context_processing/02_self_refinement.md")
    target_file = Path("/app/Context-Engineering/cn/00_COURSE/02_context_processing/02_self_refinement.md")

    if not source_file.exists():
        print(f"❌ 源文件不存在: {source_file}")
        return 1

    # 创建翻译管理器
    manager = ParallelTranslationManager()

    # 为这个文件创建任务
    print(f"\n📄 源文件: {source_file}")
    print(f"📄 目标文件: {target_file}")

    # 读取源文件
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"\n📊 文件统计:")
    print(f"  - 总行数: {len(content.splitlines())}")
    print(f"  - 总字符数: {len(content)}")
    print(f"  - 文件大小: {source_file.stat().st_size / 1024:.1f} KB")

    # 创建任务
    task = manager._create_task_for_file(
        task_id=1,
        chapter="02_context_processing",
        source_file=source_file,
        target_file=target_file
    )

    print(f"\n✅ 任务已创建:")
    print(f"  - 任务ID: {task.task_id}")
    print(f"  - 分段数: {len(task.segments)}")
    print(f"  - 优先级: {task.priority}")

    # 保存任务
    manager.tasks = [task]
    manager.save_tasks()

    print(f"\n💾 任务已保存到: {manager.task_file}")
    print("\n" + "=" * 60)
    print("✅ 任务创建完成！")
    print("=" * 60)

    # 显示分段信息
    print(f"\n📑 文件分段详情:")
    for i, segment in enumerate(task.segments, 1):
        print(f"\n  段落 {i}:")
        print(f"    - 行范围: {segment['start_line']}-{segment['end_line']}")
        print(f"    - 字符数: {segment['char_count']}")
        print(f"    - 类型: {segment['segment_type']}")
        print(f"    - 包含代码: {'是' if segment['has_code'] else '否'}")

    print(f"\n\n🚀 下一步:")
    print("  使用翻译工具逐段翻译每个片段")
    print("  或运行: python execute_parallel_translation.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())
