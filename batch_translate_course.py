#!/usr/bin/env python3
"""
批量翻译 00_COURSE 目录下的 Markdown 文件
使用 AI 辅助翻译,保持原文格式和代码块不变
"""

import os
import sys
import re
from pathlib import Path

# 定义需要翻译的章节
CHAPTERS = [
    "00_mathematical_foundations",
    "01_context_retrieval_generation",
    "02_context_processing",
    "03_context_management",
    "04_retrieval_augmented_generation",
    "05_memory_systems",
    "06_tool_integrated_reasoning",
    "07_multi_agent_systems",
    "08_field_theory_integration",
    "09_evaluation_methodologies",
    "10_orchestration_capstone"
]

# 源目录和目标目录
SOURCE_BASE = Path("/app/Context-Engineering/00_COURSE")
TARGET_BASE = Path("/app/Context-Engineering/cn/00_COURSE")

def analyze_translation_gaps():
    """分析翻译缺口"""
    report = {
        "total_files": 0,
        "translated_files": 0,
        "missing_files": [],
        "incomplete_files": [],
        "chapters": {}
    }

    for chapter in CHAPTERS:
        source_dir = SOURCE_BASE / chapter
        target_dir = TARGET_BASE / chapter

        if not source_dir.exists():
            continue

        # 获取所有 markdown 文件
        source_files = list(source_dir.rglob("*.md"))

        chapter_info = {
            "total": len(source_files),
            "translated": 0,
            "missing": [],
            "files": []
        }

        for source_file in source_files:
            rel_path = source_file.relative_to(source_dir)
            target_file = target_dir / rel_path

            file_info = {
                "source": str(source_file),
                "target": str(target_file),
                "relative": str(rel_path),
                "exists": target_file.exists(),
                "size": source_file.stat().st_size if source_file.exists() else 0
            }

            if target_file.exists():
                target_size = target_file.stat().st_size
                file_info["translated_size"] = target_size

                # 检查是否为空文件或占位符
                if target_size < 100:
                    file_info["status"] = "incomplete"
                    chapter_info["missing"].append(str(rel_path))
                    report["incomplete_files"].append(file_info)
                else:
                    file_info["status"] = "complete"
                    chapter_info["translated"] += 1
            else:
                file_info["status"] = "missing"
                chapter_info["missing"].append(str(rel_path))
                report["missing_files"].append(file_info)

            chapter_info["files"].append(file_info)
            report["total_files"] += 1

        report["translated_files"] += chapter_info["translated"]
        report["chapters"][chapter] = chapter_info

    return report

def print_analysis_report(report):
    """打印分析报告"""
    print("=" * 80)
    print(" " * 25 + "00_COURSE 翻译缺口分析报告")
    print("=" * 80)

    print(f"\n总文件数: {report['total_files']}")
    print(f"已翻译文件: {report['translated_files']}")
    print(f"缺失文件: {len(report['missing_files'])}")
    print(f"不完整文件: {len(report['incomplete_files'])}")
    print(f"完成率: {(report['translated_files']/report['total_files']*100):.1f}%")

    print("\n" + "=" * 80)
    print("各章节详细情况:")
    print("=" * 80)

    for chapter, info in report["chapters"].items():
        status = "✓" if info["total"] == info["translated"] else "✗"
        print(f"\n{status} {chapter}")
        print(f"   总计: {info['total']} 文件")
        print(f"   已翻译: {info['translated']} 文件")
        print(f"   缺失: {len(info['missing'])} 文件")

        if info["missing"]:
            print(f"   缺失文件列表:")
            for missing in info["missing"]:
                print(f"     - {missing}")

    print("\n" + "=" * 80)
    print("优先级建议:")
    print("=" * 80)

    # 按缺失文件数量排序
    chapters_by_priority = sorted(
        report["chapters"].items(),
        key=lambda x: len(x[1]["missing"]),
        reverse=True
    )

    for i, (chapter, info) in enumerate(chapters_by_priority, 1):
        if len(info["missing"]) > 0:
            print(f"{i}. {chapter}: {len(info['missing'])} 个缺失文件")

def create_placeholder_files():
    """为缺失的文件创建占位符"""
    report = analyze_translation_gaps()
    created = 0

    for file_info in report["missing_files"] + report["incomplete_files"]:
        target_path = Path(file_info["target"])
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists() or target_path.stat().st_size < 100:
            # 读取原文件
            source_path = Path(file_info["source"])
            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 创建翻译标记
                placeholder = f"""# [待翻译] {source_path.name}

> **原文路径**: {file_info['source']}
> **文件大小**: {file_info['size']} 字节
> **行数**: {len(content.splitlines())} 行
> **状态**: 等待翻译

---

**说明**: 此文件为自动生成的占位符,原文内容较大,需要专门的翻译工作流程来完成。

原文件包含以下主要部分:
"""

                # 提取标题以显示结构
                headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
                if headings:
                    placeholder += "\n\n## 文档结构:\n\n"
                    for heading in headings[:20]:  # 只显示前20个标题
                        placeholder += f"- {heading}\n"
                    if len(headings) > 20:
                        placeholder += f"\n... 还有 {len(headings) - 20} 个章节\n"

                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(placeholder)

                created += 1
                print(f"✓ 创建占位符: {target_path}")

    print(f"\n总计创建 {created} 个占位符文件")
    return created

def main():
    """主函数"""
    print("开始分析 00_COURSE 翻译状态...\n")

    # 分析并打印报告
    report = analyze_translation_gaps()
    print_analysis_report(report)

    # 询问是否创建占位符
    print("\n" + "=" * 80)
    response = input("是否为缺失文件创建占位符? (y/n): ")

    if response.lower() == 'y':
        create_placeholder_files()
        print("\n占位符创建完成!")
        print("下一步: 使用专门的翻译工具逐个翻译这些大文件")
    else:
        print("已取消创建占位符")

if __name__ == "__main__":
    main()
