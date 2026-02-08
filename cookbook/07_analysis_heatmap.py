#!/usr/bin/env python3
"""
VERA Cookbook - 07: 热力图生成

本示例展示如何：
1. 准备注意力和图像数据
2. 创建注意力热力图叠加
3. 创建Top-K patches高亮图
4. 保存可视化结果
"""

import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import analysis


def create_mock_attention_and_image():
    """创建模拟的注意力和图像数据"""

    # 创建模拟的注意力数据（100个patch）
    np.random.seed(42)
    attention_data = np.random.rand(100).tolist()

    # 让某些区域有更高的注意力
    attention_data[20:25] = [0.9, 0.85, 0.88, 0.92, 0.87]
    attention_data[50:55] = [0.95, 0.89, 0.91, 0.86, 0.93]
    attention_data[75:80] = [0.88, 0.90, 0.87, 0.91, 0.85]

    # 创建一个简单的图像（白底黑字）
    import cv2
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 添加一些文本区域作为"内容"
    cv2.rectangle(img, (50, 50), (750, 150), (0, 0, 0), 2)  # 标题区域
    cv2.rectangle(img, (50, 200), (750, 300), (0, 0, 0), 2)  # 内容区域1
    cv2.rectangle(img, (50, 350), (750, 450), (0, 0, 0), 2)  # 内容区域2
    cv2.rectangle(img, (50, 500), (750, 580), (0, 0, 0), 2)  # 内容区域3

    return attention_data, img


def main():
    print("=" * 60)
    print("VERA Cookbook - 07: 热力图生成")
    print("=" * 60)

    # ==================== 配置 ====================
    TOP_K = 10
    ALPHA = 0.5  # 叠加透明度

    # ==================== 1. 准备数据 ====================
    print("\n[Step 1] 准备数据...")

    output_dir = ROOT / "cookbook" / "output" / "07_analysis_heatmap"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建模拟数据
    attention_data, base_image = create_mock_attention_and_image()

    # 保存基础图像
    base_image_path = output_dir / "base_image.png"
    import cv2
    cv2.imwrite(str(base_image_path), base_image)
    print(f"✓ 创建了基础图像: {base_image_path}")
    print(f"  图像尺寸: {base_image.shape[1]}×{base_image.shape[0]}")
    print(f"  注意力数据: {len(attention_data)} patches")

    # ==================== 2. 生成热力图叠加 ====================
    print(f"\n[Step 2] 生成热力图叠加（alpha={ALPHA})...")

    heatmap_path = output_dir / "heatmap_overlay.png"

    result_path = analysis.create_heatmap(
        image_path=str(base_image_path),
        attention_data=attention_data,
        output_path=str(heatmap_path),
        mode="overlay",  # 热力图叠加模式
        alpha=ALPHA
    )

    print(f"✓ 热力图已保存到: {result_path}")

    # ==================== 3. 生成Top-K高亮图 ====================
    print(f"\n[Step 3] 生成Top-{TOP_K} Patches高亮图...")

    topk_path = output_dir / "top_k_highlight.png"

    result_path = analysis.create_heatmap(
        image_path=str(base_image_path),
        attention_data=attention_data,
        output_path=str(topk_path),
        mode="top_k",  # Top-K高亮模式
        alpha=0.5,
        top_k=TOP_K
    )

    print(f"✓ Top-K高亮图已保存到: {result_path}")

    # ==================== 4. 获取Top-K Patches坐标 ====================
    print(f"\n[Step 4] 获取Top-{TOP_K} Patches坐标...")

    patch_bounds = analysis.get_top_k_patches(
        attention_data=attention_data,
        image_height=base_image.shape[0],
        image_width=base_image.shape[1],
        k=TOP_K
    )

    print(f"✓ 获取到 {len(patch_bounds)} 个patches")

    # 保存patch坐标
    patches_json_path = output_dir / "top_k_patches.json"
    with open(patches_json_path, 'w') as f:
        json.dump(patch_bounds, f, indent=2)
    print(f"✓ Patch坐标已保存到: {patches_json_path}")

    # ==================== 5. 显示结果 ====================
    print("\n[Step 5] 显示结果...")
    print("-" * 60)

    print("\nTop-10 Patches坐标:")
    for i, (x1, y1, x2, y2) in enumerate(patch_bounds, 1):
        width = x2 - x1
        height = y2 - y1
        print(f"  {i:2d}. ({x1:4d}, {y1:4d}) -> ({x2:4d}, {y2:4d}) [size: {width}×{height}]")

    print("\n生成的可视化文件:")
    print(f"  1. base_image.png - 原始图像")
    print(f"  2. heatmap_overlay.png - 注意力热力图叠加")
    print(f"  3. top_k_highlight.png - Top-10 patches高亮")
    print(f"  4. top_k_patches.json - Patch坐标数据")

    print("-" * 60)

    # ==================== 6. 保存注意力数据 ====================
    print("\n[Step 6] 保存注意力数据...")

    attention_json_path = output_dir / "attention_data.json"
    with open(attention_json_path, 'w') as f:
        json.dump({
            "attention_weights": attention_data,
            "num_patches": len(attention_data),
            "description": "Mock attention data for demonstration"
        }, f, indent=2)

    print(f"✓ 注意力数据已保存到: {attention_json_path}")

    print("\n" + "=" * 60)
    print("热力图生成完成！")
    print("=" * 60)
    print(f"\n所有结果保存在: {output_dir}")
    print("\n提示：可以使用图像查看器打开生成的PNG文件")
    print("  - 热力图使用JET颜色映射（蓝色=低注意力，红色=高注意力）")
    print("  - Top-K图使用红色半透明高亮显示最关注的区域")


if __name__ == "__main__":
    main()
