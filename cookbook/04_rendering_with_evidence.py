#!/usr/bin/env python3
"""
VERA Cookbook - 04: 文本渲染为图像（带evidence高亮）

本示例展示如何：
1. 准备文本内容和evidence
2. 渲染时高亮显示evidence部分
3. 对比带evidence和不带evidence的渲染结果
"""

import sys
from pathlib import Path
from typing import List

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import rendering


def main():
    print("=" * 60)
    print("VERA Cookbook - 04: 文本渲染为图像（带evidence高亮）")
    print("=" * 60)

    # ==================== 配置 ====================
    CONFIG_PATH = "config/config_en.json"

    # 示例文本内容
    SAMPLE_TEXT = """Visual Evidence Retrieval and Analysis (VERA)

VERA is a comprehensive framework for visual document understanding and evidence retrieval. It combines large vision-language models with attention-based retrieval mechanisms to extract relevant information from document images.

Key Features:
• Multi-modal document understanding
• Attention-based evidence retrieval
• Visual-textual retrieval methods
• Comprehensive analysis tools

The framework supports various document types including academic papers, technical reports, and business documents. It can handle complex layouts and extract precise evidence for question answering tasks.
"""

    # 要高亮显示的evidence文本列表
    EVIDENCE_TEXTS: List[str] = [
        "attention-based retrieval mechanisms",  # 高亮这部分
        "extract relevant information",          # 高亮这部分
        "question answering tasks"              # 高亮这部分
    ]

    # ==================== 1. 准备输出目录 ====================
    print("\n[Step 1] 准备输出目录...")

    output_dir = ROOT / "cookbook" / "output" / "04_rendering_with_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_no_evidence = output_dir / "without_evidence"
    output_dir_with_evidence = output_dir / "with_evidence"

    print(f"输出目录: {output_dir}")

    # ==================== 2. 加载配置 ====================
    print("\n[Step 2] 加载渲染配置...")
    print(f"配置文件: {CONFIG_PATH}")

    import json
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("✓ 配置加载完成")

    # ==================== 3. 渲染不带evidence的版本 ====================
    print("\n[Step 3] 渲染不带evidence的版本...")

    image_paths_no_evidence = rendering.text_to_image(
        text=SAMPLE_TEXT,
        output_dir=str(output_dir_no_evidence),
        config=config,
        evidence_text=None  # 不高亮
    )

    print(f"✓ 渲染完成，生成了 {len(image_paths_no_evidence)} 张图像")

    # ==================== 4. 渲染带evidence的版本 ====================
    print("\n[Step 4] 渲染带evidence的版本...")
    print(f"高亮 {len(EVIDENCE_TEXTS)} 个evidence片段:")
    for i, ev in enumerate(EVIDENCE_TEXTS, 1):
        print(f"  {i}. \"{ev}\"")

    image_paths_with_evidence = rendering.text_to_image(
        text=SAMPLE_TEXT,
        output_dir=str(output_dir_with_evidence),
        config=config,
        evidence_text=EVIDENCE_TEXTS  # 高亮这些文本
    )

    print(f"✓ 渲染完成，生成了 {len(image_paths_with_evidence)} 张图像")

    # ==================== 5. 对比结果 ====================
    print("\n[Step 5] 对比结果...")
    print("-" * 60)

    print("\n【不带evidence的图像】")
    for i, path in enumerate(image_paths_no_evidence, 1):
        rel_path = Path(path).relative_to(ROOT)
        print(f"  {i}. {rel_path}")

    print("\n【带evidence高亮的图像】")
    for i, path in enumerate(image_paths_with_evidence, 1):
        rel_path = Path(path).relative_to(ROOT)
        print(f"  {i}. {rel_path}")

    print("-" * 60)

    # ==================== 6. 保存evidence信息 ====================
    print("\n[Step 6] 保存evidence信息...")

    evidence_path = output_dir / "evidence_info.txt"
    with open(evidence_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Evidence高亮信息\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"原文长度: {len(SAMPLE_TEXT)} 字符\n\n")
        f.write("高亮的evidence片段:\n")
        for i, ev in enumerate(EVIDENCE_TEXTS, 1):
            f.write(f"{i}. {ev}\n")

    print(f"✓ Evidence信息已保存到: {evidence_path}")

    print("\n" + "=" * 60)
    print("渲染完成！")
    print("=" * 60)
    print(f"\n生成的图像保存在: {output_dir}")
    print("\n提示：对比两个文件夹中的图像，观察evidence高亮效果（红色标注）")


if __name__ == "__main__":
    main()
