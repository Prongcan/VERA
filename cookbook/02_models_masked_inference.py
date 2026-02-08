#!/usr/bin/env python3
"""
VERA Cookbook - 02: 掩盖注意力头的推理

本示例展示如何：
1. 初始化QwenEngineMasked（支持头掩码的版本）
2. 定义要掩盖的注意力头
3. 运行带掩码的推理
4. 对比有无掩码的结果
"""

import sys
from pathlib import Path
from typing import Set

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import models


def main():
    print("=" * 60)
    print("VERA Cookbook - 02: 掩盖注意力头的推理")
    print("=" * 60)

    # ==================== 配置 ====================
    MODEL_PATH = "/data3/guofang/peirongcan/vllm_log/Qwen3-VL-8B-Instruct"
    IMAGE_PATH = "cookbook/data/sample_image.png"

    # 定义要掩盖的注意力头 (layer_idx, head_idx)
    # 这里我们掩盖一些常见的"重要"头来观察效果
    MASKED_HEADS: Set[tuple] = {
        (24, 29),  # Layer 24, Head 29
        (21, 11),  # Layer 21, Head 11
        (24, 8),   # Layer 24, Head 8
    }

    # ==================== 1. 初始化模型（带掩码） ====================
    print("\n[Step 1] 初始化Qwen3-VL模型（Masked版本）...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"掩盖的heads: {MASKED_HEADS}")

    engine = models.initialize(
        model_path=MODEL_PATH,
        model_type="qwen-img-masked",  # 使用支持mask的版本
        max_new_tokens=2048
    )
    print("✓ 模型初始化完成")

    # ==================== 2. 准备输入 ====================
    print("\n[Step 2] 准备输入...")

    prompt_context = "Please answer the question based on the document image provided."
    question_text = "What is the main topic of this document?"
    image_paths = [IMAGE_PATH]

    print(f"问题: {question_text}")

    # ==================== 3. 运行带掩码的推理 ====================
    print("\n[Step 3] 运行带掩码的推理...")
    print(f"掩盖 {len(MASKED_HEADS)} 个注意力头")

    result_masked = engine.run(
        prompt_context=prompt_context,
        question_text=question_text,
        image_paths=image_paths,
        is_mask_heads=True,  # 启用mask
        heads_positions=MASKED_HEADS
    )

    # ==================== 4. 运行不带掩码的推理（对比） ====================
    print("\n[Step 4] 运行不带掩码的推理（对比）...")

    result_normal = engine.run(
        prompt_context=prompt_context,
        question_text=question_text,
        image_paths=image_paths,
        is_mask_heads=False,  # 不使用mask
        heads_positions=None
    )

    # ==================== 5. 对比结果 ====================
    print("\n[Step 5] 对比结果...")
    print("-" * 60)

    print("\n【正常推理的答案】")
    print(result_normal['answer'])

    print("\n【带掩码推理的答案】")
    print(result_masked['answer'])

    print("-" * 60)

    # 简单对比
    if result_normal['answer'] != result_masked['answer']:
        print("\n⚠ 掩码注意力头后，答案发生了变化")
    else:
        print("\n✓ 掩码注意力头后，答案保持一致")

    # ==================== 6. 保存结果 ====================
    print("\n[Step 6] 保存结果...")

    output_dir = ROOT / "cookbook" / "output"
    output_dir.mkdir(exist_ok=True)

    # 保存对比结果
    comparison_path = output_dir / "02_masked_inference_comparison.txt"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("掩盖注意力头推理对比\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"掩盖的heads: {MASKED_HEADS}\n\n")

        f.write("【正常推理的答案】\n")
        f.write(result_normal['answer'])
        f.write("\n\n")

        f.write("【带掩码推理的答案】\n")
        f.write(result_masked['answer'])
        f.write("\n\n")

        f.write(f"输入长度: {result_normal['input_len']} tokens\n")

    print(f"✓ 对比结果已保存到: {comparison_path}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("\n提示：可以尝试修改MASKED_HEADS来观察不同head的影响")


if __name__ == "__main__":
    main()
