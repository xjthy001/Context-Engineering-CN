# 待翻译文件清单

## 大型文件（需要专门处理）

### 1. 评估框架文档
**文件**: `00_COURSE/09_evaluation_methodologies/00_evaluation_frameworks.md`
- **状态**: 待翻译
- **大小**: 2548行 (~26,000 tokens)
- **优先级**: 高
- **建议方法**: 分段翻译，每次500行
- **注意事项**:
  - 包含大量代码示例和数学公式
  - 需要保持技术术语的一致性
  - 包含复杂的XML和Python代码块

## 翻译进度追踪

| 文件 | 总行数 | 已翻译 | 进度 | 最后更新 |
|------|--------|--------|------|----------|
| 00_evaluation_frameworks.md | 2548 | 0 | 0% | 2025-01-02 |

## 翻译指南

### 分段翻译步骤

对于大型文件（>1000行），建议采用以下步骤：

1. **分段读取**（每次500行）
   ```bash
   sed -n '1,500p' source.md > part1.md
   sed -n '501,1000p' source.md > part2.md
   # ... 依此类推
   ```

2. **翻译每个分段**
   - 使用AI辅助翻译
   - 保持代码块、公式、链接不变
   - 确保术语一致性

3. **合并翻译结果**
   ```bash
   cat part1_cn.md part2_cn.md part3_cn.md > complete_cn.md
   ```

4. **验证翻译质量**
   - 检查Markdown格式是否完整
   - 验证代码块是否正确
   - 确保链接有效
   - 检查术语一致性

### 术语对照表

| 英文 | 中文 | 说明 |
|------|------|------|
| Evaluation Framework | 评估框架 | |
| Emergent Intelligence | 涌现智能 | |
| Component Testing | 组件测试 | |
| Adaptive Assessment | 自适应评估 | |
| Holistic Integration | 整体集成 | |
| Performance Benchmarking | 性能基准测试 | |

## 已完成的翻译

（无）

## 注意事项

1. **优先级排序**: 按照课程顺序和用户访问频率确定翻译优先级
2. **质量控制**: 大文件需要多次审查，确保翻译质量
3. **版本控制**: 每次翻译后及时提交到Git
4. **工具使用**: 可以使用Claude API、DeepL等工具辅助翻译
