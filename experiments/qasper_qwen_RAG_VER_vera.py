"""
VERA-based RAG with Visual Evidence Retrieval
使用 VERA 软件包进行检索增强生成
"""

import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import gc
from tqdm import tqdm
import cv2

# --- 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 设置 CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 使用 VERA API ---
from vera import models, rendering, retrieval
from data.dataset_loader import DatasetUtils


def get_visual_token_indices(input_tokens_path):
    """
    从 input_tokens.json 获取视觉 token 的索引范围

    Args:
        input_tokens_path: input_tokens.json 文件路径

    Returns:
        (start_idx, end_idx) 或 None
    """
    if not os.path.exists(input_tokens_path):
        return None

    try:
        with open(input_tokens_path, 'r') as f:
            tokens = json.load(f)

        vision_start_idx = None
        vision_end_idx = None

        for i, token in enumerate(tokens):
            if token == '<|vision_start|>':
                vision_start_idx = i
            elif token == '<|vision_end|>':
                vision_end_idx = i

        if vision_start_idx is not None and vision_end_idx is not None:
            return vision_start_idx + 1, vision_end_idx
    except Exception:
        pass

    return None


def aggregate_attention_with_target_heads(attn_data, visual_start, visual_end, visual_token_count, target_heads):
    """
    使用预定义的 target heads 聚合 attention 数据

    Args:
        attn_data: 多层 attention 数据
        visual_start: 视觉 token 起始索引
        visual_end: 视觉 token 结束索引
        visual_token_count: 视觉 token 总数
        target_heads: 目标 heads 列表 [(layer, head), ...]

    Returns:
        聚合后的 attention 向量
    """
    avg_attn_vector = np.zeros(visual_token_count)
    valid_heads_count = 0

    for layer_idx, head_idx in target_heads:
        if layer_idx < len(attn_data) and head_idx < len(attn_data[layer_idx]):
            full_target_attn = np.array(attn_data[layer_idx][head_idx][0])
            visual_attn_part = full_target_attn[visual_start:visual_end]

            # 对齐长度
            if len(visual_attn_part) < visual_token_count:
                visual_attn_part = np.pad(visual_attn_part, (0, visual_token_count - len(visual_attn_part)))
            elif len(visual_attn_part) > visual_token_count:
                visual_attn_part = visual_attn_part[:visual_token_count]

            avg_attn_vector += visual_attn_part
            valid_heads_count += 1

    if valid_heads_count > 0:
        avg_attn_vector /= valid_heads_count

    return avg_attn_vector


def calculate_patch_distribution(img_path, total_visual_tokens):
    """
    根据图片比例和 token 数量计算最合适的 patch 网格分布

    Args:
        img_path: 图像路径
        total_visual_tokens: 视觉 token 总数

    Returns:
        (grid_w, grid_h): patch 网格尺寸
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h_img, w_img = img.shape[:2]
    aspect_ratio = h_img / w_img
    num_patches = total_visual_tokens

    # 找所有因子对
    factors = []
    for i in range(1, int(np.sqrt(num_patches)) + 1):
        if num_patches % i == 0:
            factors.append((i, num_patches // i))
            if i != num_patches // i:
                factors.append((num_patches // i, i))

    # 找最接近图片比例的因子对
    best_fit = None
    best_ratio_diff = float('inf')

    for w, h in factors:
        if w > 0 and h > 0:
            current_ratio = h / w
            ratio_diff = abs(current_ratio - aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_fit = (w, h)

    if best_fit is None:
        grid_size = int(np.sqrt(num_patches))
        best_fit = (grid_size, grid_size)

    return best_fit


def extract_top_patches_with_retrieval_module(
    attn_data,
    visual_indices,
    word_mapping_path,
    grid_w,
    grid_h,
    img_width,
    img_height,
    top_k=10,
    target_heads=None
):
    """
    使用 VERA retrieval 模块提取 top-k patches 对应的文本

    Args:
        attn_data: attention 数据
        visual_indices: (start, end) 视觉 token 索引
        word_mapping_path: word_mapping.json 路径
        grid_w, grid_h: patch 网格尺寸
        img_width, img_height: 图像尺寸
        top_k: top k patches
        target_heads: 目标 heads 列表（如果为 None，使用默认的 Top-20）

    Returns:
        (extracted_text, top_patch_coords)
    """
    # 默认使用 Top-20 heads
    if target_heads is None:
        target_heads = [
            (24, 29), (21, 11), (24, 8), (26, 26), (24, 13),
            (26, 15), (28, 16), (27, 18), (23, 30), (24, 31),
            (28, 3), (21, 8), (23, 28), (28, 0), (26, 31),
            (26, 20), (23, 10), (21, 10), (23, 13), (20, 15)
        ]

    visual_start, visual_end = visual_indices
    visual_token_count = visual_end - visual_start

    # 1. 聚合 attention 数据
    avg_attn_vector = aggregate_attention_with_target_heads(
        attn_data, visual_start, visual_end, visual_token_count, target_heads
    )

    # 2. 获取 top-k patches 的索引
    if top_k > len(avg_attn_vector):
        top_k = len(avg_attn_vector)

    top_k_indices = np.argsort(avg_attn_vector)[-top_k:]

    # 3. 计算 patch 像素坐标
    patch_height = img_height / grid_h
    patch_width = img_width / grid_w

    top_patch_coords = []
    for patch_idx in top_k_indices:
        patch_row = patch_idx // grid_w
        patch_col = patch_idx % grid_w

        y1 = int(patch_row * patch_height)
        x1 = int(patch_col * patch_width)
        y2 = int((patch_row + 1) * patch_height)
        x2 = int((patch_col + 1) * patch_width)

        top_patch_coords.append((x1, y1, x2, y2))

    # 4. 使用 VERA retrieval 模块提取文本
    extracted_text = retrieval.extract_evidence_from_patches(
        patch_bounds=top_patch_coords,
        word_mapping_path=word_mapping_path,
        output_path=None  # 不保存到文件，直接返回
    )

    # 5. 生成 debug 图像
    try:
        merged_evidence_path = os.path.join(os.path.dirname(word_mapping_path), "merged_evidence.png")
        if os.path.exists(merged_evidence_path):
            debug_img = cv2.imread(merged_evidence_path)

            # 绘制 top-k patches
            for i, (x1, y1, x2, y2) in enumerate(top_patch_coords):
                color = (0, 255 - i * 25, i * 25)  # 渐变色
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_img, f"{i+1}", (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            debug_output_path = os.path.join(os.path.dirname(word_mapping_path), "debug_top10_patches.png")
            cv2.imwrite(debug_output_path, debug_img)
            print(f"  Debug image saved to: {debug_output_path}")
    except Exception as e:
        print(f"  Warning: Failed to generate debug image: {e}")

    return extracted_text, top_patch_coords


def is_question_processed(save_path, qid, processed_qids):
    """检查问题是否已经处理过"""
    if qid in processed_qids:
        return True

    if not os.path.exists(save_path):
        return False

    required_files = ["answer.txt", "context.txt", "gold_evidence.json"]
    return all(os.path.exists(os.path.join(save_path, f)) for f in required_files)


def main():
    parser = argparse.ArgumentParser(
        description="VERA-based RAG with Visual Evidence Retrieval"
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data3/guofang/peirongcan/vllm_log/Qwen3-VL-8B-Instruct",
        help="Path to the Qwen model"
    )

    # 数据集参数
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data3/guofang/peirongcan/deepseekOCR/Loong/Glyph/qasper/qasper-test-v0.3.json",
        help="Path to the dataset"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data3/guofang/peirongcan/deepseekOCR/Loong/Glyph/tem/ver_pipeline_qasper_results_20_vera",
        help="Directory to save results"
    )

    # 渲染配置
    parser.add_argument(
        "--config_path",
        type=str,
        default="/data3/guofang/peirongcan/deepseekOCR/Loong/Glyph/config/config_en.json",
        help="Path to the render configuration JSON file"
    )

    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Override font path in config if needed"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of papers to process"
    )

    # 新增：max_context_chars 参数
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=60000,
        help="Maximum context characters for build_context (default: 60000)"
    )

    args = parser.parse_args()

    # 1. 加载渲染配置
    print(f"Loading render config from: {args.config_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    with open(args.config_path, 'r', encoding='utf-8') as f:
        render_config = json.load(f)

    if args.font_path:
        print(f"Overriding font path with: {args.font_path}")
        render_config["font_path"] = args.font_path

    # 2. 使用 VERA API 初始化模型
    print(f"\nInitializing VERA model from {args.model_path}...")
    engine = models.initialize(
        model_path=args.model_path,
        model_type="qwen-img"
    )

    # 3. 加载数据集
    print(f"\nLoading dataset from {args.data_path}...")
    papers = DatasetUtils.load_qasper(args.data_path)
    print(f"Loaded {len(papers)} papers.")

    # 4. 初始化 prediction.jsonl
    os.makedirs(args.save_dir, exist_ok=True)
    prediction_file_path = os.path.join(args.save_dir, "prediction.jsonl")

    processed_qids = set()
    if os.path.exists(prediction_file_path):
        with open(prediction_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        pred_entry = json.loads(line.strip())
                        processed_qids.add(pred_entry.get("question_id", ""))
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_qids)} already processed questions.")

    print(f"Prediction file: {prediction_file_path}")

    # 5. 开始处理
    processed_count = 0
    skipped_count = 0

    for paper in tqdm(papers[:args.limit], desc="Processing papers"):
        # 修复：使用 args.max_context_chars 而不是 config.max_context_chars
        pid, full_text_context = DatasetUtils.build_context(paper, args.max_context_chars)

        for qa in paper.get("qas", []):
            qid = qa.get("question_id", "nan")
            question_text = qa.get("question", "")

            # 提取 Golden Evidence
            gold_evidence = DatasetUtils.extract_evidence(qa)

            # 定义保存路径
            save_path = os.path.join(
                args.save_dir,
                DatasetUtils.safe_filename(pid),
                DatasetUtils.safe_filename(qid)
            )

            # 检查是否已处理
            if is_question_processed(save_path, qid, processed_qids):
                skipped_count += 1
                continue

            os.makedirs(save_path, exist_ok=True)
            processed_count += 1

            print(f"\n{'='*60}")
            print(f"Processing {pid}-{qid}")
            print(f"{'='*60}")

            # 图片保存目录
            img_output_dir = os.path.join(save_path, "rendered_images")
            os.makedirs(img_output_dir, exist_ok=True)

            # Phase 1: 文本转图像（使用 VERA rendering API）
            print(f"Phase 1: Rendering images...")
            image_paths = rendering.text_to_image(
                text=full_text_context,
                output_dir=img_output_dir,
                config=render_config,
                evidence_text=gold_evidence
            )

            if not image_paths:
                print(f"  Warning: Image generation failed for {pid}-{qid}. Skipping.")
                continue

            print(f"  Generated {len(image_paths)} images")

            # Phase 2: 捕获 attention（使用 VERA models API）
            print(f"Phase 2: Capturing attention...")
            prompt_context = "Please answer the question based on the document images provided."

            try:
                # 使用 VERA engine.run() 获取 attention 数据
                result = engine.run(
                    prompt_context=prompt_context,
                    question_text=question_text,
                    image_paths=image_paths,
                    is_mask_heads=False,
                    heads_positions=None
                )

                if result.get("attn_error") or result.get("attn_data") is None:
                    print(f"  Warning: Failed to capture attention for {pid}-{qid}. Skipping.")
                    continue

                attn_data = result["attn_data"]
                tokens = result["input_tokens"]

                # 保存 input tokens
                input_tokens_path = os.path.join(save_path, "input_tokens.json")
                with open(input_tokens_path, "w", encoding='utf-8') as f:
                    json.dump(tokens, f, ensure_ascii=False)

                # 获取 visual token 索引
                visual_indices = get_visual_token_indices(input_tokens_path)
                if visual_indices is None:
                    print(f"  Warning: Could not find visual token indices for {pid}-{qid}. Skipping.")
                    continue

                print(f"  Visual token indices: {visual_indices}")

                # 计算 patch 分布
                merged_path = image_paths[0]
                visual_token_count = visual_indices[1] - visual_indices[0]
                grid_w, grid_h = calculate_patch_distribution(merged_path, visual_token_count)

                print(f"  Patch grid: {grid_w}x{grid_h}")

                # 获取 word_mapping 路径
                word_mapping_path = os.path.join(os.path.dirname(image_paths[0]), "word_mapping.json")
                if not os.path.exists(word_mapping_path):
                    print(f"  Warning: word_mapping.json not found at {word_mapping_path}. Skipping.")
                    continue

                # 获取图像尺寸
                with open(word_mapping_path, 'r', encoding='utf-8') as f:
                    word_data = json.load(f)
                img_info = word_data["images"][0]
                img_width = img_info["width"]
                img_height = img_info["height"]

                # Phase 3: 提取 top-10 patches 对应的文本（使用 VERA retrieval API）
                print(f"Phase 3: Extracting top-10 patches text...")
                extracted_text, top_patch_coords = extract_top_patches_with_retrieval_module(
                    attn_data=attn_data,
                    visual_indices=visual_indices,
                    word_mapping_path=word_mapping_path,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    img_width=img_width,
                    img_height=img_height,
                    top_k=10
                )

                print(f"  Extracted {len(extracted_text)} characters")

                # 保存提取的文本和 patch 坐标
                with open(os.path.join(save_path, "extracted_evidence.txt"), "w", encoding='utf-8') as f:
                    f.write(extracted_text)

                with open(os.path.join(save_path, "top_patch_coords.json"), "w") as f:
                    json.dump(top_patch_coords, f)

                # Phase 4: 基于提取的文本重新生成答案
                print(f"Phase 4: Generating final answer...")
                enhanced_context = f"{prompt_context}\n\nExtracted evidence (maybe useful):\n{extracted_text}"

                final_result = engine.run(
                    prompt_context=enhanced_context,
                    question_text=question_text,
                    image_paths=image_paths,
                    is_mask_heads=False,
                    heads_positions=None
                )

                # 保存结果
                with open(os.path.join(save_path, "context.txt"), "w") as f:
                    f.write(full_text_context)

                with open(os.path.join(save_path, "answer.txt"), "w") as f:
                    f.write(final_result["answer"])

                with open(os.path.join(save_path, "gold_evidence.json"), "w", encoding='utf-8') as f:
                    json.dump(gold_evidence, f, ensure_ascii=False, indent=2)

                # 追加到 prediction.jsonl
                pred_entry = {
                    "question_id": qid,
                    "predicted_answer": final_result["answer"],
                    "predicted_evidence": extracted_text
                }

                with open(prediction_file_path, "a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")

                print(f"  ✓ Completed {pid}-{qid}")

            except Exception as e:
                print(f"  ✗ Error during inference for {pid}-{qid}: {e}")
                import traceback
                traceback.print_exc()

            # 垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

    # 打印统计信息
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Total questions: {processed_count + skipped_count}")


if __name__ == "__main__":
    main()
