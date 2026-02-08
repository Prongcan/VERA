#!/usr/bin/env python3
"""
VERA Cookbook - 08: 完整分析流程

本示例展示如何：
1. 运行完整的三阶段分析（scan + viz + global heatmap）
2. 理解各个阶段的输出
3. 查看分析结果

注意：此示例需要已包含数据的结果目录
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import analysis


def main():
    print("=" * 60)
    print("VERA Cookbook - 08: 完整分析流程")
    print("=" * 60)

    # ==================== 配置 ====================
    # 数据根目录（需要包含question文件夹，每个文件夹有attn_first_token.json和图像）
    # 这里我们创建一个示例目录结构来演示

    demo_data_dir = ROOT / "cookbook" / "output" / "08_full_pipeline_demo"
    demo_data_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 1. 创建示例数据结构 ====================
    print("\n[Step 1] 创建示例数据结构...")

    import json
    import numpy as np
    import cv2

    # 创建3个示例question文件夹
    for i in range(1, 4):
        question_dir = demo_data_dir / f"question_{i}"
        question_dir.mkdir(exist_ok=True)

        # 创建模拟的attn_first_token.json
        np.random.seed(42 + i)
        num_layers = 32
        num_heads = 32
        mock_attention = []

        for layer_idx in range(num_layers):
            layer_data = []
            for head_idx in range(num_heads):
                # 每个head返回一个注意力向量（100个patch）
                attn_vector = np.random.rand(100).tolist()
                layer_data.append([attn_vector])  # 注意包装在list中
            mock_attention.append(layer_data)

        attn_path = question_dir / "attn_first_token.json"
        with open(attn_path, 'w') as f:
            json.dump(mock_attention, f)

        # 创建模拟的merged.png和merged_evidence.png
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (750, 550), (0, 0, 0), 2)

        # 添加模拟的红色evidence区域（用于scan阶段）
        evidence_img = img.copy()
        cv2.rectangle(evidence_img, (100, 200), (700, 300), (0, 0, 255), -1)

        cv2.imwrite(str(question_dir / "merged.png"), img)
        cv2.imwrite(str(question_dir / "merged_evidence.png"), evidence_img)

        print(f"  ✓ 创建了 question_{i} 的数据")

    print(f"\n✓ 创建了3个示例question文件夹")
    print(f"  数据目录: {demo_data_dir}")

    # ==================== 2. 运行完整分析流程 ====================
    print("\n[Step 2] 运行完整分析流程...")
    print("  阶段1: Scan - 扫描所有文件夹，确定全局Top 5 heads")
    print("  阶段2: Visualization - 使用Top 5 heads生成可视化")
    print("  阶段3: Global Heatmap - 生成全局热力图和统计")

    try:
        stats = analysis.run_full_analysis(
            root_dir=str(demo_data_dir),
            output_folder_name="analysis_result",
            top_k_patches=10,
            num_workers=2,  # 使用2个worker
            mode="all",  # 运行全部三个阶段
            red_lower=(0, 0, 150),
            red_upper=(100, 100, 255),
            kernel_size=(12, 3),
            save_debug=True
        )

        # ==================== 3. 显示统计结果 ====================
        print("\n[Step 3] 显示统计结果...")
        print("-" * 60)

        print(f"\n分析统计:")
        print(f"  文件夹总数: {stats.get('total_folders', 0)}")
        print(f"  有效样本: {stats.get('valid_samples', 0)}")
        print(f"  全局Top 5 Heads: {stats.get('global_top_5_heads', [])}")

        if stats.get('top_5_scores'):
            print(f"\nTop 5 Heads的分数:")
            for i, ((layer, head), score) in enumerate(
                zip(stats.get('global_top_5_heads', []), stats.get('top_5_scores', [])), 1
            ):
                print(f"  {i}. Layer {layer:2d}, Head {head:2d}: {score:.4f}")

        print("-" * 60)

        # ==================== 4. 查看输出文件 ====================
        print("\n[Step 4] 查看输出文件...")

        result_dir = demo_data_dir / "analysis_result"

        if result_dir.exists():
            print(f"\n生成的文件:")

            # 全局文件
            global_heatmap = result_dir / "GLOBAL_attention_heatmap.png"
            global_matrix = result_dir / "GLOBAL_attention_matrix_normalized.json"

            if global_heatmap.exists():
                print(f"  ✓ {global_heatmap.relative_to(ROOT)}")
            if global_matrix.exists():
                print(f"  ✓ {global_matrix.relative_to(ROOT)}")

            # 各个question文件夹的结果
            print(f"\n各question的可视化结果:")
            for i in range(1, 4):
                question_result_dir = demo_data_dir / f"question_{i}" / "analysis_result"
                if question_result_dir.exists():
                    files = list(question_result_dir.glob("*.png"))
                    for f in files:
                        print(f"  ✓ {f.relative_to(ROOT)}")

    except Exception as e:
        print(f"\n⚠ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

        print("\n这可能是因为:")
        print("  1. 数据格式不正确")
        print("  2. 缺少必要的依赖")
        print("  3. 内存不足")

    print("\n" + "=" * 60)
    print("分析流程演示完成！")
    print("=" * 60)
    print(f"\n数据和结果保存在: {demo_data_dir}")
    print("\n说明:")
    print("  - Phase 1 (Scan): 分析所有attention heads，找出最有效的heads")
    print("  - Phase 2 (Viz): 使用Top heads生成可视化和提取evidence")
    print("  - Phase 3 (Global): 生成全局统计和热力图")
    print("\n使用方法:")
    print("  对于真实数据，修改root_dir指向你的数据目录")
    print("  数据目录应包含多个question文件夹，每个有attn_first_token.json和图像")


if __name__ == "__main__":
    main()
