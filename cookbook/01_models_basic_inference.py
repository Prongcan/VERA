#!/usr/bin/env python3
"""
VERA Cookbook - 01: 基础模型推理

本示例展示如何：
1. 初始化Qwen3-VL模型
2. 加载图像
3. 运行推理
4. 获取结果和注意力数据
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import models


def main():
    print("=" * 60)
    print("VERA Cookbook - 01: 基础模型推理")
    print("=" * 60)

    # ==================== 配置 ====================
    # 请根据实际情况修改模型路径
    MODEL_PATH = "/data3/guofang/peirongcan/vllm_log/Qwen3-VL-8B-Instruct"

    # 示例图像路径（可以是本地路径或URL）
    IMAGE_PATH = "cookbook/data/sample.png"

    # ==================== 1. 初始化模型 ====================
    print("\n[Step 1] 初始化Qwen3-VL模型...")
    print(f"模型路径: {MODEL_PATH}")

    engine = models.initialize(
        model_path=MODEL_PATH,
        model_type="qwen-img",  # 使用标准版本
        max_new_tokens=2048
    )
    print("✓ 模型初始化完成")

    # ==================== 2. 准备输入 ====================
    print("\n[Step 2] 准备输入...")

    # 提示上下文
    prompt_context = "Please answer the question based on the document image provided."

    # 问题文本
    question_text = "How many robots are in the image?"

    # 图像路径列表
    image_paths = [IMAGE_PATH]

    print(f"提示上下文: {prompt_context}")
    print(f"问题: {question_text}")
    print(f"图像数量: {len(image_paths)}")

    # ==================== 3. 运行推理 ====================
    print("\n[Step 3] 运行推理...")

    result = engine.run(
        prompt_context=prompt_context,
        question_text=question_text,
        image_paths=image_paths,
        is_mask_heads=False,  # 不使用mask
        heads_positions=None
    )

    # ==================== 4. 输出结果 ====================
    print("\n[Step 4] 输出结果...")
    print("-" * 60)

    # 打印答案
    print(f"模型答案:\n{result['answer']}")

    print("-" * 60)

    # 打印输入信息
    print(f"\n输入长度: {result['input_len']} tokens")
    print(f"输入token数量: {len(result['input_tokens'])}")

    # 检查是否有注意力数据
    if result.get('attn_data'):
        num_layers = len(result['attn_data'])
        num_heads = len(result['attn_data'][0]) if num_layers > 0 else 0
        print(f"注意力数据: {num_layers} 层 × {num_heads} 头")
    else:
        print("注意力数据: 未捕获")

    # 检查是否有错误
    if result.get('attn_error'):
        print(f"错误信息: {result['attn_error']}")

    # ==================== 5. 保存结果 ====================
    print("\n[Step 5] 保存结果...")

    output_dir = ROOT / "cookbook" / "output"
    output_dir.mkdir(exist_ok=True)

    # 保存答案
    answer_path = output_dir / "01_basic_inference_answer.txt"
    with open(answer_path, 'w', encoding='utf-8') as f:
        f.write(result['answer'])
    print(f"✓ 答案已保存到: {answer_path}")

    # 保存输入tokens（前50个）
    tokens_path = output_dir / "01_basic_inference_tokens.txt"
    with open(tokens_path, 'w', encoding='utf-8') as f:
        for i, token in enumerate(result['input_tokens'][:50]):
            f.write(f"{i}: {token}\n")
        if len(result['input_tokens']) > 50:
            f.write(f"\n... (共 {len(result['input_tokens'])} tokens)")
    print(f"✓ Tokens已保存到: {tokens_path}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
