#!/usr/bin/env python3
"""
VERA Cookbook - 03: 文本渲染为图像（基础版）

本示例展示如何：
1. 准备文本内容
2. 使用VERA渲染模块将文本渲染为图像
3. 查看渲染结果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import rendering


def main():
    print("=" * 60)
    print("VERA Cookbook - 03: 文本渲染为图像（基础版）")
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

    # ==================== 1. 准备输出目录 ====================
    print("\n[Step 1] 准备输出目录...")

    output_dir = ROOT / "cookbook" / "output" / "03_rendering_basic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # ==================== 2. 加载配置 ====================
    print("\n[Step 2] 加载渲染配置...")
    print(f"配置文件: {CONFIG_PATH}")

    import json
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("✓ 配置加载完成")
    print(f"  字体路径: {config.get('font-path', 'N/A')}")
    print(f"  字体大小: {config.get('font-size', 'N/A')}")
    print(f"  图片宽度: {config.get('img-width', 'N/A')}")

    # ==================== 3. 渲染文本为图像 ====================
    print("\n[Step 3] 渲染文本为图像...")
    print(f"文本长度: {len(SAMPLE_TEXT)} 字符")

    image_paths = rendering.text_to_image(
        text=SAMPLE_TEXT,
        output_dir=str(output_dir),
        config=config,
        evidence_text=None  # 基础版本不使用evidence
    )

    print(f"✓ 渲染完成，生成了 {len(image_paths)} 张图像")

    # ==================== 4. 查看结果 ====================
    print("\n[Step 4] 查看结果...")
    print("-" * 60)

    for i, path in enumerate(image_paths, 1):
        print(f"{i}. {path}")

        # 检查文件是否存在
        if Path(path).exists():
            file_size = Path(path).stat().st_size
            print(f"   文件大小: {file_size / 1024:.2f} KB")
        else:
            print(f"   ⚠ 文件不存在")

    print("-" * 60)

    # ==================== 5. 保存原始文本 ====================
    print("\n[Step 5] 保存原始文本...")

    text_path = output_dir / "original_text.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(SAMPLE_TEXT)

    print(f"✓ 原始文本已保存到: {text_path}")

    print("\n" + "=" * 60)
    print("渲染完成！")
    print("=" * 60)
    print(f"\n生成的图像保存在: {output_dir}")
    print("\n提示：可以使用图像查看器打开渲染的PNG文件")


if __name__ == "__main__":
    main()
