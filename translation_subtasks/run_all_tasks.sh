#!/bin/bash
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
