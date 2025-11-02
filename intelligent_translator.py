#!/usr/bin/env python3
"""
智能翻译执行器 - 使用AI模型逐段翻译大型文档
支持代码块保护、术语一致性、格式保留
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

class IntelligentTranslator:
    """智能翻译器 - 处理技术文档的专业翻译"""

    # 技术术语映射表
    TERM_MAP = {
        # 核心概念
        "Context Engineering": "上下文工程",
        "context assembly": "上下文组装",
        "context formalization": "上下文形式化",
        "optimization theory": "优化理论",
        "information theory": "信息论",
        "Bayesian inference": "贝叶斯推理",

        # Software 3.0
        "Prompts": "提示词",
        "Programming": "编程",
        "Protocols": "协议",

        # 数学术语
        "entropy": "熵",
        "mutual information": "互信息",
        "objective function": "目标函数",
        "gradient descent": "梯度下降",
        "Pareto optimization": "帕累托优化",

        # 技术术语
        "token budget": "token预算",
        "embedding": "嵌入",
        "LLM": "大语言模型",
        "RAG": "检索增强生成",
        "semantic similarity": "语义相似度",
        "relevance score": "相关性分数",

        # 架构术语
        "multi-agent": "多智能体",
        "agent": "智能体",
        "tool integration": "工具集成",
        "memory system": "记忆系统",
        "orchestration": "编排"
    }

    def __init__(self):
        self.translation_cache = {}  # 缓存已翻译的段落
        self.term_glossary = self.TERM_MAP.copy()

    def translate_document(self, source_file: Path, target_file: Path,
                          segments: List[Dict]) -> bool:
        """翻译整个文档"""
        print(f"\n{'='*80}")
        print(f"开始翻译: {source_file.name}")
        print(f"目标文件: {target_file}")
        print(f"总段落数: {len(segments)}")
        print(f"{'='*80}\n")

        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines(keepends=True)
        translated_segments = []

        # 逐段翻译
        for i, seg_info in enumerate(segments, 1):
            seg_id = seg_info['segment_id']
            start = seg_info['start_line']
            end = seg_info['end_line']
            seg_type = seg_info['segment_type']
            has_code = seg_info['has_code']

            print(f"[{i}/{len(segments)}] 翻译段落 {seg_id} (行 {start}-{end}, 类型: {seg_type})")

            # 提取段落内容
            segment_content = ''.join(lines[start-1:end])

            # 翻译段落
            translated = self.translate_segment(
                segment_content,
                seg_type=seg_type,
                has_code=has_code,
                segment_id=seg_id
            )

            translated_segments.append(translated)

            # 每10段显示进度
            if i % 10 == 0:
                progress = (i / len(segments)) * 100
                print(f"  进度: {progress:.1f}% ({i}/{len(segments)})")

        # 合并翻译结果
        final_translation = ''.join(translated_segments)

        # 后处理: 确保一致性
        final_translation = self.post_process(final_translation)

        # 确保目标目录存在
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # 写入目标文件
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(final_translation)

        print(f"\n✅ 翻译完成: {target_file}")
        print(f"   原文: {len(content)} 字符")
        print(f"   译文: {len(final_translation)} 字符")

        return True

    def translate_segment(self, content: str, seg_type: str = "content",
                         has_code: bool = False, segment_id: int = 0) -> str:
        """翻译单个段落"""

        # 如果是纯代码块,保持不变
        if seg_type == "code" and content.strip().startswith('```') and content.strip().endswith('```'):
            return content

        # 保护代码块和特殊标记
        protected_content, placeholders = self.protect_special_content(content)

        # 翻译核心内容
        if has_code or seg_type == "code":
            # 有代码的段落需要特殊处理
            translated = self.translate_mixed_content(protected_content)
        else:
            # 纯文本段落
            translated = self.translate_text(protected_content)

        # 恢复被保护的内容
        translated = self.restore_protected_content(translated, placeholders)

        return translated

    def protect_special_content(self, content: str) -> Tuple[str, Dict]:
        """保护代码块、数学公式、URL等特殊内容"""
        placeholders = {}
        protected = content

        # 保护代码块
        code_blocks = re.findall(r'```[\s\S]*?```', content, re.MULTILINE)
        for i, block in enumerate(code_blocks):
            placeholder = f"___CODE_BLOCK_{i}___"
            placeholders[placeholder] = block
            protected = protected.replace(block, placeholder, 1)

        # 保护行内代码
        inline_codes = re.findall(r'`[^`\n]+`', protected)
        for i, code in enumerate(inline_codes):
            placeholder = f"___INLINE_CODE_{i}___"
            placeholders[placeholder] = code
            protected = protected.replace(code, placeholder, 1)

        # 保护URL
        urls = re.findall(r'https?://[^\s)]+', protected)
        for i, url in enumerate(urls):
            placeholder = f"___URL_{i}___"
            placeholders[placeholder] = url
            protected = protected.replace(url, placeholder, 1)

        # 保护数学公式
        math_formulas = re.findall(r'\$[^\$]+\$', protected)
        for i, formula in enumerate(math_formulas):
            placeholder = f"___MATH_{i}___"
            placeholders[placeholder] = formula
            protected = protected.replace(formula, placeholder, 1)

        return protected, placeholders

    def restore_protected_content(self, text: str, placeholders: Dict) -> str:
        """恢复被保护的内容"""
        result = text
        for placeholder, original in placeholders.items():
            result = result.replace(placeholder, original)
        return result

    def translate_text(self, text: str) -> str:
        """翻译纯文本内容"""

        # 按段落分割
        paragraphs = text.split('\n\n')
        translated_paras = []

        for para in paragraphs:
            if not para.strip():
                translated_paras.append(para)
                continue

            # 检测特殊格式
            if para.strip().startswith('#'):  # 标题
                translated_paras.append(self.translate_heading(para))
            elif para.strip().startswith('>'):  # 引用
                translated_paras.append(self.translate_quote(para))
            elif para.strip().startswith(('- ', '* ', '+ ')):  # 列表
                translated_paras.append(self.translate_list(para))
            elif '|' in para and '---' in para:  # 表格
                translated_paras.append(self.translate_table(para))
            else:  # 普通段落
                translated_paras.append(self.translate_paragraph(para))

        return '\n\n'.join(translated_paras)

    def translate_heading(self, heading: str) -> str:
        """翻译标题"""
        # 提取标题级别和内容
        match = re.match(r'^(#+)\s*(.+)$', heading.strip())
        if not match:
            return heading

        level, content = match.groups()

        # 翻译内容 (简化版 - 实际应该调用翻译API)
        translated_content = self.simple_translate(content)

        return f"{level} {translated_content}\n"

    def translate_quote(self, quote: str) -> str:
        """翻译引用块"""
        lines = quote.split('\n')
        translated_lines = []

        for line in lines:
            if line.strip().startswith('>'):
                # 提取引用内容
                content = line.strip()[1:].strip()
                if content:
                    translated = self.simple_translate(content)
                    translated_lines.append(f"> {translated}")
                else:
                    translated_lines.append(line)
            else:
                translated_lines.append(line)

        return '\n'.join(translated_lines)

    def translate_list(self, list_text: str) -> str:
        """翻译列表"""
        lines = list_text.split('\n')
        translated_lines = []

        for line in lines:
            # 检测列表项
            match = re.match(r'^(\s*[-*+]\s+)(.+)$', line)
            if match:
                prefix, content = match.groups()
                translated = self.simple_translate(content)
                translated_lines.append(f"{prefix}{translated}")
            else:
                translated_lines.append(line)

        return '\n'.join(translated_lines)

    def translate_table(self, table: str) -> str:
        """翻译表格 - 保持结构,只翻译内容"""
        lines = table.split('\n')
        translated_lines = []

        for line in lines:
            if '|' not in line:
                translated_lines.append(line)
                continue

            # 分隔符行保持不变
            if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                translated_lines.append(line)
                continue

            # 翻译表格单元格
            cells = line.split('|')
            translated_cells = []

            for cell in cells:
                if cell.strip():
                    translated_cells.append(self.simple_translate(cell.strip()))
                else:
                    translated_cells.append(cell)

            translated_lines.append('| ' + ' | '.join(c for c in translated_cells if c) + ' |')

        return '\n'.join(translated_lines)

    def translate_paragraph(self, para: str) -> str:
        """翻译普通段落"""
        if not para.strip():
            return para

        # 简单翻译
        translated = self.simple_translate(para)
        return translated

    def translate_mixed_content(self, content: str) -> str:
        """翻译包含代码的混合内容"""
        # 分行处理
        lines = content.split('\n')
        translated_lines = []

        in_code_block = False

        for line in lines:
            # 检测代码块边界
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                translated_lines.append(line)
                continue

            # 代码块内的内容保持不变
            if in_code_block:
                translated_lines.append(line)
            else:
                # 非代码内容进行翻译
                if line.strip():
                    translated_lines.append(self.simple_translate(line))
                else:
                    translated_lines.append(line)

        return '\n'.join(translated_lines)

    def simple_translate(self, text: str) -> str:
        """
        简单翻译函数 - 这是占位符实现
        实际使用时应该调用真实的翻译API或模型
        """

        # 应用术语表
        result = text
        for en_term, zh_term in self.term_glossary.items():
            # 使用词边界确保精确匹配
            result = re.sub(r'\b' + re.escape(en_term) + r'\b', zh_term, result, flags=re.IGNORECASE)

        # TODO: 这里应该调用实际的翻译服务
        # 例如: result = call_translation_api(result)

        return result

    def post_process(self, translated: str) -> str:
        """后处理翻译结果"""

        # 修复常见问题
        result = translated

        # 确保中英文之间有空格(可选)
        # result = re.sub(r'([a-zA-Z0-9])([一-龥])', r'\1 \2', result)
        # result = re.sub(r'([一-龥])([a-zA-Z0-9])', r'\1 \2', result)

        # 修复标点符号
        result = result.replace('。。', '。')
        result = result.replace('，，', '，')

        return result

def main():
    """测试翻译器"""
    translator = IntelligentTranslator()

    # 示例: 翻译一个小段落
    sample_text = """
# Context Formalization: The Mathematical Heart of Context Engineering

> "Language shapes the way we think, and determines what we can think about."
>
> — Benjamin Lee Whorf

## The Bridge: From Metaphor to Mathematics

**Restaurant Experience Components**:
```
Ambiance + Menu + Chef Capabilities = Great Meal
```

This isn't just notation—it's a powerful framework.
"""

    print("测试翻译功能:")
    print("="*60)
    print("原文:")
    print(sample_text)
    print("\n" + "="*60)

    translated = translator.translate_segment(sample_text, has_code=True)

    print("译文:")
    print(translated)
    print("="*60)

if __name__ == "__main__":
    main()
