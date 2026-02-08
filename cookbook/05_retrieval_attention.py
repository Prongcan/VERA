#!/usr/bin/env python3
"""
VERA Cookbook - 05: 基于注意力的检索

本示例展示如何：
1. 准备注意力数据和word_mapping
2. 从注意力数据中提取Top-K patches
3. 从patches中提取证据文本

注意：此示例需要已有的注意力数据和word_mapping.json文件
通常在运行模型推理后会生成这些文件
"""

import sys
import json
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import retrieval


def create_mock_attention_data():
    """创建模拟的注意力数据用于演示"""
    # 模拟100个patch的注意力权重
    np.random.seed(42)
    attention_weights = np.random.rand(100).tolist()

    # 让某些patch有更高的注意力（模拟重要区域）
    attention_weights[20:25] = [0.9, 0.85, 0.88, 0.92, 0.87]  # Patch 20-24
    attention_weights[50:55] = [0.95, 0.89, 0.91, 0.86, 0.93]  # Patch 50-54
    attention_weights[75:80] = [0.88, 0.90, 0.87, 0.91, 0.85]  # Patch 75-79

    return attention_weights


def create_mock_word_mapping():
    """创建模拟的word_mapping.json用于演示"""
    return {
        "images": [
            {
                "index": 0,
                "width": 800,
                "height": 1200
            }
        ],
        "words": []
    }


def main():
    print("=" * 60)
    print("VERA Cookbook - 05: 基于注意力的检索")
    print("=" * 60)

    # ==================== 配置 ====================
    # 设置图像尺寸
    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 1200
    TOP_K = 10  # 提取Top-10 patches

    # ==================== 1. 准备数据 ====================
    print("\n[Step 1] 准备数据...")

    output_dir = ROOT / "cookbook" / "output" / "05_retrieval_attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建模拟数据
    print("创建模拟的注意力数据...")
    attention_data = create_mock_attention_data()
    print(f"  注意力数据: {len(attention_data)} patches")

    # 创建模拟的word_mapping
    print("创建模拟的word_mapping...")
    word_mapping = create_mock_word_mapping()

    # 添加一些模拟的单词（20行文本，每行分布在不同的高度）
    words_per_line = 10
    for line_num in range(20):
        y_start = 50 + line_num * 50
        for word_idx in range(words_per_line):
            x_start = 50 + word_idx * 70
            word_mapping["words"].append({
                "word": f"word_{line_num}_{word_idx}",
                "bbox": [x_start, y_start, x_start + 60, y_start + 30],
                "line": line_num
            })

    print(f"  添加了 {len(word_mapping['words'])} 个单词")

    # 保存word_mapping
    word_mapping_path = output_dir / "word_mapping.json"
    with open(word_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(word_mapping, f, indent=2, ensure_ascii=False)

    print(f"✓ word_mapping已保存到: {word_mapping_path}")

    # ==================== 2. 获取Top-K Patches ====================
    print(f"\n[Step 2] 获取Top-{TOP_K} Patches...")

    # 使用analysis模块的函数获取patch边界
    from vera import analysis
    from vera.analysis.heatmap import _calculate_patch_distribution

    # 计算patch网格分布
    grid_w, grid_h = _calculate_patch_distribution(IMAGE_WIDTH, IMAGE_HEIGHT, len(attention_data))
    print(f"  Patch网格: {grid_w} × {grid_h}")

    # 获取Top-K patches
    patch_bounds = analysis.get_top_k_patches(
        attention_data=attention_data,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        k=TOP_K
    )

    print(f"✓ 获取到 {len(patch_bounds)} 个patches")

    # 保存patch边界
    patches_path = output_dir / "top_k_patches.json"
    with open(patches_path, 'w') as f:
        json.dump(patch_bounds, f, indent=2)
    print(f"✓ Patch边界已保存到: {patches_path}")

    # ==================== 3. 从Patches提取证据 ====================
    print(f"\n[Step 3] 从Patches提取证据...")

    output_evidence_path = output_dir / "extracted_evidence.txt"

    extracted_text = retrieval.extract_evidence_from_patches(
        patch_bounds=patch_bounds,
        word_mapping_path=str(word_mapping_path),
        output_path=str(output_evidence_path)
    )

    print(f"✓ 提取了 {len(extracted_text)} 个字符")
    print(f"✓ 证据已保存到: {output_evidence_path}")

    # ==================== 4. 显示结果 ====================
    print("\n[Step 4] 显示结果...")
    print("-" * 60)

    print("\n提取的Top-K Patches坐标:")
    for i, (x1, y1, x2, y2) in enumerate(patch_bounds[:5], 1):  # 只显示前5个
        print(f"  {i}. Patch ({x1}, {y1}) -> ({x2}, {y2})")
    if len(patch_bounds) > 5:
        print(f"  ... 还有 {len(patch_bounds) - 5} 个patches")

    print("\n提取的文本内容（前200字符）:")
    print(extracted_text[:200])
    if len(extracted_text) > 200:
        print("...")

    print("-" * 60)

    # ==================== 5. 使用完整检索流程 ====================
    print("\n[Step 5] 使用完整的检索流程...")

    output_evidence_full_path = output_dir / "extracted_evidence_full.txt"

    extracted_text_full, patch_bounds_full = retrieval.retrieve_by_attention(
        attention_data=attention_data,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        word_mapping_path=str(word_mapping_path),
        top_k=TOP_K,
        output_path=str(output_evidence_full_path)
    )

    print(f"✓ 完整流程提取了 {len(extracted_text_full)} 个字符")
    print(f"✓ 结果已保存到: {output_evidence_full_path}")

    print("\n" + "=" * 60)
    print("检索示例完成！")
    print("=" * 60)
    print(f"\n所有结果保存在: {output_dir}")
    print("\n提示：这个示例使用了模拟数据")
    print("实际使用时，需要从模型推理中获取真实的注意力数据")


if __name__ == "__main__":
    main()
