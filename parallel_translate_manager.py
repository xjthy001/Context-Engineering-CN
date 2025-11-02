#!/usr/bin/env python3
"""
并行翻译管理器 - 为每个文档创建子任务并智能分段翻译
支持大文件的逐段落翻译,避免超出模型长度限制
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

@dataclass
class TranslationTask:
    """翻译任务数据结构"""
    task_id: str
    source_file: str
    target_file: str
    chapter: str
    file_name: str
    total_lines: int
    total_chars: int
    segments: List[Dict]
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 1  # 1=高, 2=中, 3=低

@dataclass
class TranslationSegment:
    """翻译段落数据结构"""
    segment_id: int
    start_line: int
    end_line: int
    content: str
    char_count: int
    has_code: bool
    segment_type: str  # header, content, code, table, list

class ParallelTranslationManager:
    """并行翻译管理器"""

    # 配置参数
    MAX_SEGMENT_CHARS = 8000  # 每段最大字符数(考虑模型限制)
    SOURCE_BASE = Path("/app/Context-Engineering/00_COURSE")
    TARGET_BASE = Path("/app/Context-Engineering/cn/00_COURSE")

    def __init__(self):
        self.tasks: List[TranslationTask] = []
        self.task_file = Path("/app/Context-Engineering/translation_tasks.json")

    def scan_and_create_tasks(self) -> List[TranslationTask]:
        """扫描所有需要翻译的文件并创建任务"""
        print("🔍 扫描需要翻译的文件...")

        chapters = [
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

        task_id = 1
        for chapter in chapters:
            source_dir = self.SOURCE_BASE / chapter
            target_dir = self.TARGET_BASE / chapter

            if not source_dir.exists():
                continue

            # 获取所有markdown文件
            for source_file in sorted(source_dir.rglob("*.md")):
                rel_path = source_file.relative_to(source_dir)
                target_file = target_dir / rel_path

                # 检查是否需要翻译
                needs_translation = False
                if not target_file.exists():
                    needs_translation = True
                elif target_file.stat().st_size < 500:  # 小于500字节认为是占位符
                    needs_translation = True
                else:
                    # 检查是否包含"待翻译"标记
                    with open(target_file, 'r', encoding='utf-8') as f:
                        content = f.read(200)
                        if '[待翻译]' in content or '等待翻译' in content:
                            needs_translation = True

                if needs_translation:
                    task = self._create_task_for_file(
                        task_id, chapter, source_file, target_file
                    )
                    self.tasks.append(task)
                    task_id += 1

        print(f"✅ 创建了 {len(self.tasks)} 个翻译任务")
        return self.tasks

    def _create_task_for_file(self, task_id: int, chapter: str,
                             source_file: Path, target_file: Path) -> TranslationTask:
        """为单个文件创建翻译任务"""

        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分析文件
        total_lines = len(content.splitlines())
        total_chars = len(content)

        # 智能分段
        segments = self._segment_content(content)

        # 确定优先级
        priority = self._determine_priority(chapter, source_file.name)

        return TranslationTask(
            task_id=f"TASK_{task_id:03d}",
            source_file=str(source_file),
            target_file=str(target_file),
            chapter=chapter,
            file_name=source_file.name,
            total_lines=total_lines,
            total_chars=total_chars,
            segments=[self._segment_to_dict(seg) for seg in segments],
            status="pending",
            priority=priority
        )

    def _segment_content(self, content: str) -> List[TranslationSegment]:
        """智能分段 - 按照语义边界分割内容"""
        segments = []
        lines = content.splitlines(keepends=True)

        current_segment = []
        current_chars = 0
        segment_id = 1
        start_line = 1
        current_type = "content"
        in_code_block = False

        for i, line in enumerate(lines, 1):
            # 检测代码块
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    current_type = "code"
                else:
                    current_type = "content"

            # 检测标题(作为分段边界)
            is_heading = line.strip().startswith('#') and not in_code_block

            # 检测其他语义边界
            is_boundary = (
                is_heading or
                (line.strip() == '---' and not in_code_block) or  # 分隔线
                (line.strip() == '' and current_chars > 3000)  # 空行且内容足够
            )

            # 如果达到分段条件
            if is_boundary and current_segment and not in_code_block:
                # 创建段落
                segment_content = ''.join(current_segment)
                has_code = '```' in segment_content

                segments.append(TranslationSegment(
                    segment_id=segment_id,
                    start_line=start_line,
                    end_line=i-1,
                    content=segment_content,
                    char_count=len(segment_content),
                    has_code=has_code,
                    segment_type=self._detect_segment_type(segment_content)
                ))

                # 重置
                segment_id += 1
                start_line = i
                current_segment = []
                current_chars = 0

            # 添加当前行
            current_segment.append(line)
            current_chars += len(line)

            # 如果段落过大,强制分段(即使在代码块中)
            if current_chars >= self.MAX_SEGMENT_CHARS:
                segment_content = ''.join(current_segment)
                has_code = '```' in segment_content

                segments.append(TranslationSegment(
                    segment_id=segment_id,
                    start_line=start_line,
                    end_line=i,
                    content=segment_content,
                    char_count=len(segment_content),
                    has_code=has_code,
                    segment_type=self._detect_segment_type(segment_content)
                ))

                segment_id += 1
                start_line = i + 1
                current_segment = []
                current_chars = 0

        # 处理最后一段
        if current_segment:
            segment_content = ''.join(current_segment)
            has_code = '```' in segment_content

            segments.append(TranslationSegment(
                segment_id=segment_id,
                start_line=start_line,
                end_line=len(lines),
                content=segment_content,
                char_count=len(segment_content),
                has_code=has_code,
                segment_type=self._detect_segment_type(segment_content)
            ))

        return segments

    def _detect_segment_type(self, content: str) -> str:
        """检测段落类型"""
        if content.strip().startswith('#'):
            return "header"
        elif '```' in content:
            return "code"
        elif '|' in content and '---' in content:
            return "table"
        elif re.match(r'^\s*[-*+]\s', content.strip()):
            return "list"
        else:
            return "content"

    def _segment_to_dict(self, segment: TranslationSegment) -> Dict:
        """转换段落为字典"""
        return {
            "segment_id": segment.segment_id,
            "start_line": segment.start_line,
            "end_line": segment.end_line,
            "char_count": segment.char_count,
            "has_code": segment.has_code,
            "segment_type": segment.segment_type,
            "content": segment.content[:200] + "..." if len(segment.content) > 200 else segment.content
        }

    def _determine_priority(self, chapter: str, file_name: str) -> int:
        """确定任务优先级"""
        # 高优先级
        if chapter == "00_mathematical_foundations":
            return 1
        if file_name in ["README.md", "00_overview.md", "00_introduction.md"]:
            return 1

        # 中优先级
        if chapter.startswith("01_") or chapter.startswith("02_"):
            return 2

        # 低优先级
        return 3

    def save_tasks(self):
        """保存任务到JSON文件"""
        tasks_data = {
            "total_tasks": len(self.tasks),
            "pending": len([t for t in self.tasks if t.status == "pending"]),
            "in_progress": len([t for t in self.tasks if t.status == "in_progress"]),
            "completed": len([t for t in self.tasks if t.status == "completed"]),
            "tasks": [asdict(task) for task in self.tasks]
        }

        with open(self.task_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)

        print(f"💾 任务列表已保存到: {self.task_file}")

    def load_tasks(self) -> bool:
        """从JSON文件加载任务"""
        if not self.task_file.exists():
            return False

        with open(self.task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.tasks = [TranslationTask(**task_data) for task_data in data['tasks']]
        print(f"📂 已加载 {len(self.tasks)} 个任务")
        return True

    def generate_task_report(self):
        """生成任务报告"""
        print("\n" + "=" * 80)
        print(" " * 25 + "并行翻译任务报告")
        print("=" * 80)

        # 总体统计
        total = len(self.tasks)
        by_status = {
            "pending": len([t for t in self.tasks if t.status == "pending"]),
            "in_progress": len([t for t in self.tasks if t.status == "in_progress"]),
            "completed": len([t for t in self.tasks if t.status == "completed"]),
            "failed": len([t for t in self.tasks if t.status == "failed"])
        }

        print(f"\n📊 总体统计:")
        print(f"  总任务数: {total}")
        print(f"  待处理: {by_status['pending']}")
        print(f"  进行中: {by_status['in_progress']}")
        print(f"  已完成: {by_status['completed']}")
        print(f"  失败: {by_status['failed']}")

        # 按章节统计
        print(f"\n📚 按章节统计:")
        by_chapter = {}
        for task in self.tasks:
            if task.chapter not in by_chapter:
                by_chapter[task.chapter] = []
            by_chapter[task.chapter].append(task)

        for chapter, tasks in sorted(by_chapter.items()):
            pending = len([t for t in tasks if t.status == "pending"])
            completed = len([t for t in tasks if t.status == "completed"])
            total_chapter = len(tasks)
            progress = (completed / total_chapter * 100) if total_chapter > 0 else 0

            status_icon = "✅" if pending == 0 else "⏳"
            print(f"  {status_icon} {chapter}: {completed}/{total_chapter} ({progress:.0f}%)")

        # 按优先级统计
        print(f"\n🎯 按优先级统计:")
        by_priority = {1: [], 2: [], 3: []}
        for task in self.tasks:
            by_priority[task.priority].append(task)

        for priority, tasks in sorted(by_priority.items()):
            if tasks:
                pending = len([t for t in tasks if t.status == "pending"])
                print(f"  优先级 {priority}: {pending}/{len(tasks)} 待处理")

        # 文件大小统计
        print(f"\n📏 文件规模统计:")
        total_chars = sum(t.total_chars for t in self.tasks)
        total_lines = sum(t.total_lines for t in self.tasks)
        total_segments = sum(len(t.segments) for t in self.tasks)

        print(f"  总字符数: {total_chars:,}")
        print(f"  总行数: {total_lines:,}")
        print(f"  总段落数: {total_segments:,}")
        print(f"  平均每文件: {total_chars//total if total > 0 else 0:,} 字符")
        print(f"  平均每段落: {total_chars//total_segments if total_segments > 0 else 0:,} 字符")

        # 详细任务列表(前20个)
        print(f"\n📋 待处理任务列表 (前20个):")
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        pending_tasks.sort(key=lambda t: (t.priority, t.task_id))

        for i, task in enumerate(pending_tasks[:20], 1):
            priority_label = {1: "🔴高", 2: "🟡中", 3: "🟢低"}[task.priority]
            segments_info = f"{len(task.segments)}段"
            size_info = f"{task.total_chars//1000}K字符"

            print(f"  {i:2d}. [{task.task_id}] {priority_label} {task.file_name}")
            print(f"      章节: {task.chapter}")
            print(f"      规模: {task.total_lines}行, {size_info}, {segments_info}")

        if len(pending_tasks) > 20:
            print(f"\n  ... 还有 {len(pending_tasks) - 20} 个待处理任务")

        print("\n" + "=" * 80)

    def generate_subtask_scripts(self):
        """为每个任务生成独立的翻译子任务脚本"""
        print("\n🔧 生成子任务翻译脚本...")

        scripts_dir = Path("/app/Context-Engineering/translation_subtasks")
        scripts_dir.mkdir(exist_ok=True)

        # 为每个任务生成脚本
        for task in self.tasks:
            if task.status != "pending":
                continue

            script_content = self._generate_task_script(task)
            script_file = scripts_dir / f"{task.task_id}_{task.file_name.replace('.md', '')}.py"

            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)

            # 设置可执行权限
            os.chmod(script_file, 0o755)

        print(f"✅ 生成了 {len([t for t in self.tasks if t.status == 'pending'])} 个子任务脚本")
        print(f"📁 脚本目录: {scripts_dir}")

        # 生成批量执行脚本
        self._generate_batch_script(scripts_dir)

    def _generate_task_script(self, task: TranslationTask) -> str:
        """生成单个任务的翻译脚本"""
        return f'''#!/usr/bin/env python3
"""
自动生成的翻译子任务脚本
任务ID: {task.task_id}
源文件: {task.source_file}
目标文件: {task.target_file}
章节: {task.chapter}
段落数: {len(task.segments)}
"""

import sys
from pathlib import Path

# 任务信息
TASK_ID = "{task.task_id}"
SOURCE_FILE = Path("{task.source_file}")
TARGET_FILE = Path("{task.target_file}")
TOTAL_SEGMENTS = {len(task.segments)}

def translate_segment(segment_id, content):
    """
    翻译单个段落
    这里需要调用实际的翻译服务或AI模型
    """
    # TODO: 实现实际的翻译逻辑
    # 这里是占位符,实际使用时需要调用翻译API
    print(f"  翻译段落 {{segment_id}}/{{TOTAL_SEGMENTS}}...")

    # 简单的标记处理(保持代码块不变)
    if '```' in content:
        # 代码块需要特殊处理
        return content  # 暂时保持原样

    # 实际翻译逻辑应该在这里
    return content

def main():
    print(f"开始翻译任务: {{TASK_ID}}")
    print(f"文件: {{SOURCE_FILE.name}}")

    # 读取源文件
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分段翻译
    segments = {task.segments}
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

    print(f"✅ 翻译完成: {{TARGET_FILE}}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    def _generate_batch_script(self, scripts_dir: Path):
        """生成批量执行脚本"""
        batch_script = '''#!/bin/bash
# 批量执行所有翻译子任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL_TASKS=$(ls -1 "$SCRIPT_DIR"/TASK_*.py 2>/dev/null | wc -l)
COMPLETED=0
FAILED=0

echo "=================================================="
echo "         批量翻译任务执行器"
echo "=================================================="
echo "总任务数: $TOTAL_TASKS"
echo ""

for script in "$SCRIPT_DIR"/TASK_*.py; do
    if [ -f "$script" ]; then
        task_name=$(basename "$script" .py)
        log_file="$LOG_DIR/${task_name}.log"

        echo "执行: $task_name"

        if python3 "$script" > "$log_file" 2>&1; then
            echo "  ✅ 成功"
            ((COMPLETED++))
        else
            echo "  ❌ 失败 (详见日志: $log_file)"
            ((FAILED++))
        fi
    fi
done

echo ""
echo "=================================================="
echo "执行完成"
echo "成功: $COMPLETED"
echo "失败: $FAILED"
echo "总计: $TOTAL_TASKS"
echo "=================================================="
'''

        batch_file = scripts_dir / "run_all_tasks.sh"
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_script)

        os.chmod(batch_file, 0o755)
        print(f"✅ 生成批量执行脚本: {batch_file}")

def main():
    """主函数"""
    manager = ParallelTranslationManager()

    # 扫描并创建任务
    print("🚀 并行翻译任务管理系统启动\n")
    manager.scan_and_create_tasks()

    # 保存任务
    manager.save_tasks()

    # 生成报告
    manager.generate_task_report()

    # 生成子任务脚本
    manager.generate_subtask_scripts()

    print("\n" + "=" * 80)
    print("✅ 所有准备工作完成!")
    print("\n下一步:")
    print("  1. 查看任务报告了解详细信息")
    print("  2. 进入 /app/Context-Engineering/translation_subtasks/")
    print("  3. 执行单个任务脚本或使用 ./run_all_tasks.sh 批量执行")
    print("  4. 实现实际的翻译逻辑(translate_segment函数)")
    print("=" * 80)

if __name__ == "__main__":
    main()
