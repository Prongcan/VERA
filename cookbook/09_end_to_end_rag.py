#!/usr/bin/env python3
"""
VERA Cookbook - 09: 端到端RAG示例

本示例展示完整的RAG流程：
1. 文本渲染为图像
2. 第一次推理：捕获注意力
3. 基于注意力提取证据
4. 第二次推理：使用提取的证据生成最终答案

这是VERA框架的典型使用场景
"""

import sys
import json
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import models, rendering, retrieval


def main():
    print("=" * 60)
    print("VERA Cookbook - 09: 端到端RAG示例")
    print("=" * 60)

    # ==================== 配置 ====================
    MODEL_PATH = "/data3/guofang/peirongcan/vllm_log/Qwen3-VL-8B-Instruct"
    CONFIG_PATH = "config/config_en.json"

    # 示例文档和问题
    SAMPLE_TEXT = """Attention Mechanisms in Deep Learning

Attention mechanisms have become a fundamental component in modern deep learning architectures. They allow models to focus on specific parts of the input when producing an output, similar to how human attention works.

Self-Attention
Self-attention is a mechanism that relates different positions of a single sequence in order to compute a representation of the sequence. It has been particularly successful in natural language processing tasks.

Key advantages of self-attention include:
• Ability to capture long-range dependencies
• Parallel computation unlike recurrent networks
• Interpretability through attention weights

Applications
Attention mechanisms are widely used in:
1. Machine Translation
2. Image Classification
3. Document Understanding
4. Question Answering Systems

The attention weight between two positions is computed as a function of their representations, typically using softmax of a compatibility function."""

    GOLD_EVIDENCE = [
        "Self-attention is a mechanism that relates different positions",
        "Ability to capture long-range dependencies",
        "Interpretability through attention weights"
    ]

    QUESTION = "What are the key advantages of self-attention?"

    # 固定的Top-20 heads（用于聚合注意力）
    TARGET_HEADS = [
        (24, 29), (21, 11), (24, 8), (26, 26), (24, 13),
        (26, 15), (28, 16), (27, 18), (23, 30), (24, 31),
        (28, 3), (21, 8), (23, 28), (28, 0), (26, 31),
        (26, 20), (23, 10), (21, 10), (23, 13), (20, 15)
    ]

    # ==================== 1. 准备输出目录 ====================
    print("\n[Step 1] 准备输出目录...")

    output_dir = ROOT / "cookbook" / "output" / "09_end_to_end_rag"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # ==================== 2. 文本渲染为图像 ====================
    print("\n[Step 2] 文本渲染为图像...")
    print(f"文本长度: {len(SAMPLE_TEXT)} 字符")

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        render_config = json.load(f)

    image_paths = rendering.text_to_image(
        text=SAMPLE_TEXT,
        output_dir=str(output_dir / "rendered_images"),
        config=render_config,
        evidence_text=GOLD_EVIDENCE  # 高亮gold evidence
    )

    print(f"✓ 渲染了 {len(image_paths)} 张图像")
    for i, path in enumerate(image_paths, 1):
        print(f"  {i}. {Path(path).name}")

    # ==================== 3. 初始化模型 ====================
    print("\n[Step 3] 初始化模型...")
    print(f"模型路径: {MODEL_PATH}")

    try:
        engine = models.initialize(
            model_path=MODEL_PATH,
            model_type="qwen-img",
            max_new_tokens=512
        )
        print("✓ 模型初始化完成")
    except Exception as e:
        print(f"\n⚠ 模型初始化失败: {e}")
        print("\n提示：请确保模型路径正确，且已安装必要的依赖")
        return

    # ==================== 4. 第一次推理：捕获注意力 ====================
    print("\n[Step 4] 第一次推理（捕获注意力）...")
    prompt_context = "Please answer the question based on the document image provided."

    print(f"问题: {QUESTION}")

    try:
        result1 = engine.run(
            prompt_context=prompt_context,
            question_text=QUESTION,
            image_paths=image_paths,
            is_mask_heads=False,
            heads_positions=None
        )

        print(f"✓ 推理完成")
        print(f"\n第一次推理的答案:")
        print("-" * 60)
        print(result1['answer'])
        print("-" * 60)

        # 保存第一次结果
        with open(output_dir / "first_answer.txt", 'w', encoding='utf-8') as f:
            f.write(result1['answer'])

        # 检查是否有注意力数据
        if not result1.get('attn_data'):
            print("\n⚠ 未捕获到注意力数据，无法进行RAG")
            print("这可能是因为模型不支持注意力捕获")
            return

        # 保存input_tokens
        with open(output_dir / "input_tokens.json", 'w', encoding='utf-8') as f:
            json.dump(result1['input_tokens'], f, ensure_ascii=False)

        # 获取visual token索引
        visual_start, visual_end = None, None
        for i, token in enumerate(result1['input_tokens']):
            if token == '<|vision_start|>':
                visual_start = i
            elif token == '<|vision_end|>':
                visual_end = i

        if visual_start is None or visual_end is None:
            print("\n⚠ 未找到visual token标记")
            return

        visual_token_count = visual_end - visual_start
        print(f"\nVisual tokens: {visual_token_count} (索引 {visual_start} 到 {visual_end})")

        # ==================== 5. 聚合注意力数据 ====================
        print("\n[Step 5] 聚合Top-20 heads的注意力...")

        attn_data = result1['attn_data']
        avg_attn_vector = np.zeros(visual_token_count)
        valid_heads_count = 0

        for layer_idx, head_idx in TARGET_HEADS:
            if layer_idx < len(attn_data) and head_idx < len(attn_data[layer_idx]):
                full_target_attn = np.array(attn_data[layer_idx][head_idx][0])
                visual_attn_part = full_target_attn[visual_start:visual_end]

                if len(visual_attn_part) < visual_token_count:
                    visual_attn_part = np.pad(visual_attn_part, (0, visual_token_count - len(visual_attn_part)))
                elif len(visual_attn_part) > visual_token_count:
                    visual_attn_part = visual_attn_part[:visual_token_count]

                avg_attn_vector += visual_attn_part
                valid_heads_count += 1

        if valid_heads_count > 0:
            avg_attn_vector /= valid_heads_count

        print(f"✓ 聚合了 {valid_heads_count} 个heads的注意力")

        # ==================== 6. 计算Top-10 Patches ====================
        print("\n[Step 6] 获取Top-10 Patches...")

        TOP_K = 10
        top_k_indices = np.argsort(avg_attn_vector)[-TOP_K:]

        # 计算patch边界（需要图像尺寸和网格）
        # 这里简化处理，假设从word_mapping获取尺寸
        word_mapping_path = Path(image_paths[0]).parent / "word_mapping.json"
        if not word_mapping_path.exists():
            print("\n⚠ word_mapping.json不存在，使用简化边界")
            # 使用简单的边界
            patch_bounds = [(i*80, i*60, (i+1)*80, (i+1)*60) for i in range(TOP_K)]
        else:
            # 实际应该从word_mapping读取尺寸并计算
            with open(word_mapping_path, 'r') as f:
                wm = json.load(f)
            img_w = wm["images"][0]["width"]
            img_h = wm["images"][0]["height"]

            # 简化的patch计算
            grid_w = 10
            grid_h = 10
            patch_w = img_w / grid_w
            patch_h = img_h / grid_h

            patch_bounds = []
            for idx in top_k_indices:
                row = idx // grid_w
                col = idx % grid_w
                x1 = int(col * patch_w)
                y1 = int(row * patch_h)
                x2 = int((col + 1) * patch_w)
                y2 = int((row + 1) * patch_h)
                patch_bounds.append((x1, y1, x2, y2))

        print(f"✓ 获取了 {len(patch_bounds)} 个patches")

        # ==================== 7. 提取证据文本 ====================
        print("\n[Step 7] 从Patches提取证据...")

        extracted_text = retrieval.extract_evidence_from_patches(
            patch_bounds=patch_bounds,
            word_mapping_path=str(word_mapping_path),
            output_path=str(output_dir / "extracted_evidence.txt")
        )

        print(f"✓ 提取了 {len(extracted_text)} 个字符")
        print(f"\n提取的证据（前200字符）:")
        print("-" * 60)
        print(extracted_text[:200])
        if len(extracted_text) > 200:
            print("...")
        print("-" * 60)

        # ==================== 8. 第二次推理：使用提取的证据 ====================
        print("\n[Step 8] 第二次推理（使用提取的证据）...")

        enhanced_context = f"""{prompt_context}

I've extracted some relevant evidence from the document:
{extracted_text}

Please use this evidence to provide a more accurate answer."""

        result2 = engine.run(
            prompt_context=enhanced_context,
            question_text=QUESTION,
            image_paths=image_paths,
            is_mask_heads=False,
            heads_positions=None
        )

        print(f"✓ 第二次推理完成")
        print(f"\n第二次推理的答案（使用RAG）:")
        print("=" * 60)
        print(result2['answer'])
        print("=" * 60)

        # 保存第二次结果
        with open(output_dir / "final_answer.txt", 'w', encoding='utf-8') as f:
            f.write(result2['answer'])

        # ==================== 9. 对比结果 ====================
        print("\n[Step 9] 对比两次推理的结果...")

        comparison_path = output_dir / "comparison.txt"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAG流程对比\n")
            f.write("=" * 60 + "\n\n")

            f.write("【问题】\n")
            f.write(f"{QUESTION}\n\n")

            f.write("【第一次推理（无RAG）】\n")
            f.write(result1['answer'])
            f.write("\n\n")

            f.write("【提取的证据】\n")
            f.write(extracted_text)
            f.write("\n\n")

            f.write("【第二次推理（使用RAG）】\n")
            f.write(result2['answer'])
            f.write("\n\n")

        print(f"✓ 对比结果已保存到: {comparison_path}")

    except Exception as e:
        print(f"\n⚠ 推理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("端到端RAG示例完成！")
    print("=" * 60)
    print(f"\n所有结果保存在: {output_dir}")
    print("\nRAG流程总结:")
    print("  1. 文本 → 图像（渲染）")
    print("  2. 图像 + 问题 → 答案 + 注意力（第一次推理）")
    print("  3. 注意力 → Top-K Patches → 证据文本（检索）")
    print("  4. 图像 + 问题 + 证据 → 更好的答案（第二次推理）")


if __name__ == "__main__":
    main()
