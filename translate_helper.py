#!/usr/bin/env python3
"""
Helper script to facilitate translation of cognitive-tools documentation
This creates stub files with proper structure that can be completed
"""

import os
import glob

# Define the translation mappings for common terms
TERM_MAP = {
    "Cognitive Tools": "认知工具",
    "Understanding Templates": "理解模板",
    "Reasoning Templates": "推理模板",
    "Verification Templates": "验证模板",
    "Composition": "组合",
    "Basic Programs": "基础程序",
    "Advanced Programs": "高级程序",
    "User Schemas": "用户模式",
    "Domain Schemas": "领域模式",
    "Task Schemas": "任务模式",
    "Agentic Schemas": "代理模式",
    "Solver Architecture": "求解器架构",
    "Tutor Architecture": "导师架构",
    "Research Architecture": "研究架构"
}

# Get list of files to translate
source_dir = "/app/Context-Engineering/cognitive-tools"
target_dir = "/app/Context-Engineering/cn/cognitive-tools"

files_to_translate = glob.glob(f"{source_dir}/**/*.md", recursive=True)

print(f"Found {len(files_to_translate)} files to translate:")
for f in sorted(files_to_translate):
    rel_path = f.replace(source_dir, "").lstrip("/")
    target_path = os.path.join(target_dir, rel_path)
    target_parent = os.path.dirname(target_path)

    # Check if translation exists
    exists = os.path.exists(target_path)
    status = "✓" if exists else "✗"

    print(f"{status} {rel_path}")

print(f"\n Total: {len(files_to_translate)} files")
print(f"Target directory: {target_dir}")
