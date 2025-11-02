#!/bin/bash

# 源目录和目标目录
SRC_DIR="/app/Context-Engineering"
DEST_DIR="/app/Context-Engineering/cn"

# 文件扩展名列表
extensions=("*.py" "*.yaml" "*.yml" "*.json" "*.sh" "*.js" "*.ts" "*.css" "*.html")

# 计数器
copied=0
skipped=0

# 查找并复制文件
for ext in "${extensions[@]}"; do
    while IFS= read -r -d '' file; do
        # 获取相对路径
        rel_path="${file#$SRC_DIR/}"
        
        # 跳过cn目录、.git目录和node_modules
        if [[ "$rel_path" == cn/* ]] || [[ "$rel_path" == .git/* ]] || [[ "$rel_path" == node_modules/* ]]; then
            continue
        fi
        
        # 目标文件路径
        dest_file="$DEST_DIR/$rel_path"
        dest_dir=$(dirname "$dest_file")
        
        # 创建目标目录
        mkdir -p "$dest_dir"
        
        # 复制文件
        if cp "$file" "$dest_file"; then
            echo "✓ 复制: $rel_path"
            ((copied++))
        else
            echo "✗ 失败: $rel_path"
            ((skipped++))
        fi
    done < <(find "$SRC_DIR" -type f -name "$ext" -print0)
done

echo ""
echo "====== 复制完成 ======"
echo "成功复制: $copied 个文件"
echo "失败: $skipped 个文件"
