#!/usr/bin/env python3
"""
并行翻译执行器 - 使用智能翻译器执行所有翻译任务
支持并发执行、进度追踪、错误恢复
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
import concurrent.futures

# 导入智能翻译器
from intelligent_translator import IntelligentTranslator

@dataclass
class TranslationTask:
    """翻译任务"""
    task_id: str
    source_file: str
    target_file: str
    chapter: str
    file_name: str
    total_lines: int
    total_chars: int
    segments: List[dict]
    status: str
    priority: int

class ParallelTranslationExecutor:
    """并行翻译执行器"""

    def __init__(self, max_workers: int = 1):  # 先用单线程确保稳定性
        self.max_workers = max_workers
        self.translator = IntelligentTranslator()
        self.tasks_file = Path("/app/Context-Engineering/translation_tasks.json")
        self.tasks: List[TranslationTask] = []
        self.results = {
            "completed": 0,
            "failed": 0,
            "total": 0,
            "errors": []
        }

    def load_tasks(self) -> bool:
        """加载翻译任务"""
        if not self.tasks_file.exists():
            print("❌ 任务文件不存在,请先运行 parallel_translate_manager.py")
            return False

        print("📂 加载翻译任务...")
        with open(self.tasks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.tasks = [
            TranslationTask(**task_data)
            for task_data in data['tasks']
            if task_data['status'] == 'pending'  # 只加载待处理的任务
        ]

        self.results["total"] = len(self.tasks)
        print(f"✅ 加载了 {len(self.tasks)} 个待处理任务")
        return True

    def execute_all_tasks(self, priority_filter: int = None):
        """执行所有任务"""

        # 筛选任务
        tasks_to_execute = self.tasks
        if priority_filter:
            tasks_to_execute = [t for t in self.tasks if t.priority == priority_filter]
            print(f"🎯 筛选优先级 {priority_filter} 的任务: {len(tasks_to_execute)} 个")

        if not tasks_to_execute:
            print("⚠️ 没有符合条件的任务")
            return

        print(f"\n🚀 开始执行 {len(tasks_to_execute)} 个翻译任务")
        print(f"   并发数: {self.max_workers}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # 按优先级排序
        tasks_to_execute.sort(key=lambda t: (t.priority, t.task_id))

        # 执行任务
        if self.max_workers == 1:
            # 单线程执行
            for i, task in enumerate(tasks_to_execute, 1):
                print(f"\n[{i}/{len(tasks_to_execute)}] 执行任务: {task.task_id}")
                self._execute_single_task(task)
        else:
            # 多线程执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._execute_single_task, task): task
                    for task in tasks_to_execute
                }

                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"❌ 任务 {task.task_id} 失败: {e}")
                        self.results["errors"].append({
                            "task_id": task.task_id,
                            "error": str(e)
                        })

        # 统计结果
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        # 更新任务状态
        self._update_task_status()

    def _execute_single_task(self, task: TranslationTask) -> bool:
        """执行单个翻译任务"""
        try:
            source_path = Path(task.source_file)
            target_path = Path(task.target_file)

            print(f"  文件: {task.file_name}")
            print(f"  章节: {task.chapter}")
            print(f"  规模: {task.total_lines}行, {task.total_chars}字符, {len(task.segments)}段")

            # 执行翻译
            success = self.translator.translate_document(
                source_path,
                target_path,
                task.segments
            )

            if success:
                self.results["completed"] += 1
                task.status = "completed"
                print(f"  ✅ 成功")
                return True
            else:
                self.results["failed"] += 1
                task.status = "failed"
                print(f"  ❌ 失败")
                return False

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            self.results["failed"] += 1
            self.results["errors"].append({
                "task_id": task.task_id,
                "file": task.file_name,
                "error": str(e)
            })
            task.status = "failed"
            return False

    def _print_summary(self, elapsed_time: float):
        """打印执行摘要"""
        print(f"\n{'='*80}")
        print("📊 执行摘要")
        print(f"{'='*80}")
        print(f"  总任务数: {self.results['total']}")
        print(f"  ✅ 成功: {self.results['completed']}")
        print(f"  ❌ 失败: {self.results['failed']}")
        print(f"  ⏱️ 耗时: {elapsed_time:.2f} 秒")

        if self.results["completed"] > 0:
            avg_time = elapsed_time / self.results["completed"]
            print(f"  平均每任务: {avg_time:.2f} 秒")

        if self.results["errors"]:
            print(f"\n❌ 错误列表:")
            for error in self.results["errors"]:
                print(f"  - {error.get('task_id', 'N/A')}: {error['error']}")

        print(f"{'='*80}\n")

    def _update_task_status(self):
        """更新任务状态到文件"""
        print("💾 更新任务状态...")

        # 读取原始数据
        with open(self.tasks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 更新状态
        task_dict = {t.task_id: t for t in self.tasks}
        for task_data in data['tasks']:
            task_id = task_data['task_id']
            if task_id in task_dict:
                task_data['status'] = task_dict[task_id].status

        # 更新统计
        data['completed'] = len([t for t in data['tasks'] if t['status'] == 'completed'])
        data['pending'] = len([t for t in data['tasks'] if t['status'] == 'pending'])
        data['in_progress'] = len([t for t in data['tasks'] if t['status'] == 'in_progress'])

        # 保存
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 任务状态已更新")

    def execute_by_chapter(self, chapter: str):
        """按章节执行翻译"""
        chapter_tasks = [t for t in self.tasks if t.chapter == chapter]

        if not chapter_tasks:
            print(f"⚠️ 章节 {chapter} 没有待处理任务")
            return

        print(f"📚 执行章节: {chapter}")
        print(f"   任务数: {len(chapter_tasks)}")

        # 创建临时任务列表
        original_tasks = self.tasks
        self.tasks = chapter_tasks
        self.results["total"] = len(chapter_tasks)

        # 执行
        self.execute_all_tasks()

        # 恢复
        self.tasks = original_tasks

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="并行翻译执行器")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3],
                       help="只执行指定优先级的任务 (1=高, 2=中, 3=低)")
    parser.add_argument("--chapter", type=str,
                       help="只执行指定章节的任务")
    parser.add_argument("--workers", type=int, default=1,
                       help="并发工作线程数 (默认: 1)")
    parser.add_argument("--test", action="store_true",
                       help="测试模式: 只执行前3个任务")

    args = parser.parse_args()

    print("🚀 并行翻译执行器")
    print(f"{'='*80}\n")

    # 创建执行器
    executor = ParallelTranslationExecutor(max_workers=args.workers)

    # 加载任务
    if not executor.load_tasks():
        sys.exit(1)

    # 测试模式
    if args.test:
        print("🧪 测试模式: 只执行前3个任务")
        executor.tasks = executor.tasks[:3]
        executor.results["total"] = 3

    # 执行任务
    if args.chapter:
        executor.execute_by_chapter(args.chapter)
    elif args.priority:
        executor.execute_all_tasks(priority_filter=args.priority)
    else:
        # 执行所有任务
        executor.execute_all_tasks()

if __name__ == "__main__":
    main()
