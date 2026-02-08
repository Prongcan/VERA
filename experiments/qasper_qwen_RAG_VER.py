"""
VERA-based RAG with Visual Evidence Retrieval (使用新的 VERA 工具函数 API)
使用 VERA 软件包进行检索增强生成

这个版本使用了 vera 包中的工具函数，而不是自定义函数
"""
#
import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import gc
from tqdm import tqdm

# --- 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 设置 CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 加载模型配置 ---
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/model_config.json")
if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    DEFAULT_MODEL_PATH = model_config.get("model_path", {}).get("qwen")
    if not DEFAULT_MODEL_PATH:
        raise ValueError("model_path.qwen not found in config/model_config.json")
else:
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_PATH}")

# --- 使用 VERA API ---
from vera import models, rendering, retrieval, utils, analysis
from data.dataset_loader import DatasetUtils


def main():
    parser = argparse.ArgumentParser(
        description="VERA-based RAG with Visual Evidence Retrieval (Refactored)"
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the Qwen model"
    )

    # 数据集参数
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/qasper/qasper-test-sample.json",
        help="Path to the dataset"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="tem/qasper_vera_rag",
        help="Directory to save results"
    )

    # 渲染配置
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config_en.json",
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

    if os.path.exists(prediction_file_path):
        os.remove(prediction_file_path)
    print(f"Prediction file will be saved to: {prediction_file_path}")

    # 5. 开始处理
    for paper in tqdm(papers[:args.limit], desc="Processing papers"):
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
            os.makedirs(save_path, exist_ok=True)

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

                # ========== 使用新的 VERA 工具函数 ==========

                # 获取 visual token 索引（使用 vera.utils）
                visual_indices = utils.get_visual_token_indices(input_tokens_path)
                if visual_indices is None:
                    print(f"  Warning: Could not find visual token indices for {pid}-{qid}. Skipping.")
                    continue

                print(f"  Visual token indices: {visual_indices}")

                # 计算 patch 分布（使用 vera.analysis）
                merged_path = image_paths[0]
                visual_token_count = visual_indices[1] - visual_indices[0]

                # 获取图像尺寸
                word_mapping_path = os.path.join(os.path.dirname(image_paths[0]), "word_mapping.json")
                if not os.path.exists(word_mapping_path):
                    print(f"  Warning: word_mapping.json not found. Skipping.")
                    continue

                with open(word_mapping_path, 'r', encoding='utf-8') as f:
                    word_data = json.load(f)
                img_info = word_data["images"][0]
                img_width = img_info["width"]
                img_height = img_info["height"]

                grid_w, grid_h = analysis.calculate_patch_distribution(
                    img_width=img_width,
                    img_height=img_height,
                    total_visual_tokens=visual_token_count
                )

                print(f"  Patch grid: {grid_w}x{grid_h}")

                # Phase 3: 提取 top-10 patches 对应的文本（使用 vera.retrieval 高级 API）
                print(f"Phase 3: Extracting top-10 patches text...")
                extracted_text, top_patch_coords = retrieval.extract_top_patches_with_attention_retrieve(
                    attn_data=attn_data,
                    visual_indices=visual_indices,
                    word_mapping_path=word_mapping_path,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    img_width=img_width,
                    img_height=img_height,
                    top_k=10,
                    debug_image_path=os.path.join(os.path.dirname(word_mapping_path), "debug_top10_patches.png")
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


if __name__ == "__main__":
    main()
