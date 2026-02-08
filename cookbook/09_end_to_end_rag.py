#!/usr/bin/env python3
"""
VERA Cookbook - 09: 端到端RAG示例（标准版）

本示例展示完整的VERA RAG流程：
1. 文本渲染为图像
2. 第一次推理：捕获注意力
3. 基于注意力提取证据（使用 vera API）
4. 第二次推理：使用提取的证据生成最终答案

这是VERA框架的标准使用方式，推荐在生产环境中使用
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 使用 VERA 标准API
from vera import models, rendering, retrieval, utils, analysis


def main():
    print("=" * 60)
    print("VERA Cookbook - 09: 端到端RAG示例（标准版）")
    print("=" * 60)

    # ==================== 配置 ====================
    # 从配置文件加载模型路径
    MODEL_CONFIG_PATH = ROOT / "config" / "model_config.json"
    if MODEL_CONFIG_PATH.exists():
        with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        MODEL_PATH = model_config.get("model_path", {}).get("qwen")
        if not MODEL_PATH:
            print("\n⚠ 错误：未在 config/model_config.json 中找到 model_path.qwen")
            print("请确保配置文件正确")
            return
    else:
        # 如果没有配置文件，使用默认路径
        MODEL_PATH = "/data3/guofang/peirongcan/vllm_log/Qwen3-VL-8B-Instruct"
        print(f"\n提示：未找到配置文件，使用默认模型路径")

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

    # ==================== 1. 准备输出目录 ====================
    print("\n[Step 1] 准备输出目录...")

    output_dir = ROOT / "cookbook" / "output" / "09_end_to_end_rag"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # ==================== 2. 加载渲染配置 ====================
    print("\n[Step 2] 加载渲染配置...")

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            render_config = json.load(f)
        print("✓ 渲染配置加载成功")
    except Exception as e:
        print(f"\n⚠ 加载渲染配置失败: {e}")
        return

    # ==================== 3. 文本渲染为图像 ====================
    print("\n[Step 3] 文本渲染为图像...")
    print(f"文本长度: {len(SAMPLE_TEXT)} 字符")

    image_paths = rendering.text_to_image(
        text=SAMPLE_TEXT,
        output_dir=str(output_dir / "rendered_images"),
        config=render_config,
        evidence_text=GOLD_EVIDENCE  # 高亮gold evidence
    )

    print(f"✓ 渲染了 {len(image_paths)} 张图像")
    for i, path in enumerate(image_paths, 1):
        print(f"  {i}. {Path(path).name}")

    # ==================== 4. 初始化模型 ====================
    print("\n[Step 4] 初始化模型...")
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

    # ==================== 5. 第一次推理：捕获注意力 ====================
    print("\n[Step 5] 第一次推理（捕获注意力）...")
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

        print("✓ 推理完成")
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
        input_tokens_path = output_dir / "input_tokens.json"
        with open(input_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(result1['input_tokens'], f, ensure_ascii=False)

        print(f"✓ Input tokens 已保存")

    except Exception as e:
        print(f"\n⚠ 推理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 6. 获取 Visual Token 索引（使用 VERA API）====================
    print("\n[Step 6] 获取 Visual Token 索引...")

    visual_indices = utils.get_visual_token_indices(str(input_tokens_path))
    if visual_indices is None:
        print("\n⚠ 无法找到 visual token 标记")
        return

    visual_start, visual_end = visual_indices
    visual_token_count = visual_end - visual_start
    print(f"✓ Visual tokens: {visual_token_count} (索引 {visual_start} 到 {visual_end})")

    # ==================== 7. 计算 Patch 分布（使用 VERA API）====================
    print("\n[Step 7] 计算 Patch 分布...")

    # 从 word_mapping.json 获取图像尺寸
    word_mapping_path = Path(image_paths[0]).parent / "word_mapping.json"
    if not word_mapping_path.exists():
        print(f"\n⚠ word_mapping.json 不存在: {word_mapping_path}")
        return

    try:
        with open(word_mapping_path, 'r', encoding='utf-8') as f:
            word_data = json.load(f)

        img_info = word_data["images"][0]

        # 从实际图像文件获取尺寸（而不是从word_mapping）
        from PIL import Image
        actual_image_path = Path(image_paths[0])
        if actual_image_path.exists():
            with Image.open(actual_image_path) as img:
                img_width, img_height = img.size
            print(f"✓ 使用实际图像尺寸: {img_width}x{img_height}")
        else:
            # 回退到word_mapping中的尺寸
            img_width = img_info["width"]
            img_height = img_info["height"]
            print(f"⚠ 使用word_mapping中的尺寸: {img_width}x{img_height}")

        # 使用 VERA API 计算 patch 分布
        grid_w, grid_h = analysis.calculate_patch_distribution(
            img_width=img_width,
            img_height=img_height,
            total_visual_tokens=visual_token_count
        )

        print(f"✓ Patch 网格: {grid_w}x{grid_h} (总共 {grid_w * grid_h} 个 patches)")

    except Exception as e:
        print(f"\n⚠ 计算 patch 分布失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 8. 提取 Top-K Patches 对应的文本（使用 VERA API）====================
    print("\n[Step 8] 提取 Top-10 Patches 对应的文本...")

    TOP_K = 10

    try:
        # 使用 VERA 高级 API 一站式提取证据
        extracted_text, top_patch_coords = retrieval.extract_top_patches_with_attention_retrieve(
            attn_data=result1['attn_data'],
            visual_indices=visual_indices,
            word_mapping_path=str(word_mapping_path),
            grid_w=grid_w,
            grid_h=grid_h,
            img_width=img_width,
            img_height=img_height,
            top_k=TOP_K,
            debug_image_path=str(output_dir / "debug_top10_patches.png")
        )

        print(f"✓ 提取了 {len(extracted_text)} 个字符")
        print(f"\n提取的证据（前200字符）:")
        print("-" * 60)
        print(extracted_text[:200])
        if len(extracted_text) > 200:
            print("...")
        print("-" * 60)

        # 保存提取的证据
        evidence_path = output_dir / "extracted_evidence.txt"
        with open(evidence_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        # 保存 patch 坐标
        coords_path = output_dir / "top_patch_coords.json"
        with open(coords_path, 'w') as f:
            json.dump(top_patch_coords, f)

        print(f"✓ 证据已保存到: {evidence_path}")
        print(f"✓ 坐标已保存到: {coords_path}")

    except Exception as e:
        print(f"\n⚠ 提取证据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 9. 第二次推理：使用提取的证据 ====================
    print("\n[Step 9] 第二次推理（使用提取的证据）...")

    enhanced_context = f"""{prompt_context}

I've extracted some relevant evidence from the document:
{extracted_text}

Please use this evidence to provide a more accurate answer."""

    try:
        result2 = engine.run(
            prompt_context=enhanced_context,
            question_text=QUESTION,
            image_paths=image_paths,
            is_mask_heads=False,
            heads_positions=None
        )

        print("✓ 第二次推理完成")
        print(f"\n第二次推理的答案（使用RAG）:")
        print("=" * 60)
        print(result2['answer'])
        print("=" * 60)

        # 保存第二次结果
        with open(output_dir / "final_answer.txt", 'w', encoding='utf-8') as f:
            f.write(result2['answer'])

    except Exception as e:
        print(f"\n⚠ 第二次推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 10. 对比结果 ====================
    print("\n[Step 10] 对比两次推理的结果...")

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

        f.write("【VERA API 使用说明】\n")
        f.write("-" * 60 + "\n")
        f.write("本示例使用了以下 VERA 标准API：\n\n")
        f.write("1. utils.get_visual_token_indices()\n")
        f.write("   - 自动查找 visual token 的起始和结束索引\n\n")
        f.write("2. analysis.calculate_patch_distribution()\n")
        f.write("   - 根据图像尺寸和 visual token 数量计算 patch 网格\n\n")
        f.write("3. retrieval.extract_top_patches_with_attention_retrieve()\n")
        f.write("   - 一站式提取证据\n")
        f.write("   - 自动处理注意力聚合\n")
        f.write("   - 自动选择 top-k patches\n")
        f.write("   - 自动提取对应文本\n\n")
        f.write("这些 API 封装了复杂逻辑，推荐在生产环境中使用\n")
        f.write("-" * 60 + "\n")

    print(f"✓ 对比结果已保存到: {comparison_path}")

if __name__ == "__main__":
    main()
