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
import math

# --- 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 添加PyTorch内存优化环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 假设 VLM_and_LLM 在 ROOT 下
if str(ROOT / "VLM_and_LLM") not in sys.path:
    sys.path.insert(0, str(ROOT / "VLM_and_LLM"))

# --- 模块导入 ---
from models.wrapper import QwenEngine_img_no_eager, InferenceConfig
from data.dataset_loader import DatasetUtils
# 导入文本转图片函数
from VLM_and_LLM.Text2img import process_single_text_evidence

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

def calculate_patch_distribution(img_path, total_visual_tokens):
    """根据图片比例和token数量计算最合适的patch网格分布"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h_img, w_img = img.shape[:2]
    aspect_ratio = h_img / w_img
    num_patches = total_visual_tokens

    factors = []
    for i in range(1, int(math.sqrt(num_patches)) + 1):
        if num_patches % i == 0:
            factors.append((i, num_patches // i))
            if i != num_patches // i:
                factors.append((num_patches // i, i))

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
        grid_size = int(math.sqrt(num_patches))
        best_fit = (grid_size, grid_size)

    grid_w, grid_h = best_fit
    return grid_w, grid_h

def get_visual_token_indices(question_folder):
    """获取视觉tokens在完整序列中的索引范围"""
    input_tokens_path = os.path.join(question_folder, "input_tokens.json")
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

def is_question_processed(save_path, qid, processed_qids):
    """
    检查问题是否已经完全处理过
    Args:
        save_path: 保存路径
        qid: 问题ID
        processed_qids: 已处理的question_id集合
    Returns:
        bool: 是否已经处理过
    """
    # 检查是否在prediction.jsonl中
    if qid in processed_qids:
        return True

    # 检查保存目录是否存在且包含必要的文件
    if not os.path.exists(save_path):
        return False

    # 检查核心输出文件是否存在
    required_files = ["answer.txt", "context.txt", "gold_evidence.json"]
    return all(os.path.exists(os.path.join(save_path, f)) for f in required_files)

def extract_top_patches_text(attn_data, visual_indices, word_mapping_path, grid_w, grid_h, top_k=10):
    """
    从注意力数据中提取top-k patches对应的文本

    Args:
        attn_data: 注意力数据
        visual_indices: (start_idx, end_idx) 视觉token的索引范围
        word_mapping_path: word_mapping.json文件路径
        grid_w, grid_h: patch网格尺寸
        top_k: 提取前k个patch

    Returns:
        extracted_text: 提取的文本
        top_patch_coords: top-k patches的坐标列表
    """
    # 目标头定义
    TARGET_HEADS = [
        (24, 29), # Rank 1
        (21, 11), # Rank 2
        (24, 8),  # Rank 3
        (26, 26), # Rank 4
        (24, 13),  # Rank 5
        (26, 15),  # Rank 6
        (28, 16),  # Rank 7
        (27, 18),  # Rank 8
        (23, 30),  # Rank 9
        (24, 31),  # Rank 10
        (28, 3),  # Rank 11
        (21, 8),  # Rank 12
        (23, 28),  # Rank 13
        (28, 0),  # Rank 14
        (26, 31),  # Rank 15
        (26, 20),
        (23, 10),
        (21, 10),
        (23, 13),
        (20, 15)  
    ]

    visual_start, visual_end = visual_indices
    visual_token_count = visual_end - visual_start

    # 计算平均注意力向量
    avg_attn_vector = np.zeros(visual_token_count)
    valid_heads_count = 0

    for layer_idx, head_idx in TARGET_HEADS:
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

    # 获取top-k patches的索引
    if top_k > len(avg_attn_vector):
        top_k = len(avg_attn_vector)

    top_k_indices = np.argsort(avg_attn_vector)[-top_k:]

    # 加载word_mapping
    with open(word_mapping_path, 'r', encoding='utf-8') as f:
        word_data = json.load(f)

    # 获取图片尺寸
    img_info = word_data["images"][0]
    img_width = img_info["width"]
    img_height = img_info["height"]

    # 计算每个patch的像素坐标
    patch_height = img_height / grid_h
    patch_width = img_width / grid_w

    # 收集涉及的行
    involved_lines = set()

    for patch_idx in top_k_indices:
        # 计算patch在网格中的位置
        patch_row = patch_idx // grid_w
        patch_col = patch_idx % grid_w

        # 计算patch的像素坐标
        y1 = int(patch_row * patch_height)
        x1 = int(patch_col * patch_width)
        y2 = int((patch_row + 1) * patch_height)
        x2 = int((patch_col + 1) * patch_width)

        # 找到与这个patch区域相交的所有单词行
        for word_info in word_data["words"]:
            word_bbox = word_info["bbox"]  # [x1, y1, x2, y2]

            # 检查patch与单词bbox是否有重叠（相交）
            # 使用AABB（轴对齐包围盒）相交检测
            if not (x2 < word_bbox[0] or word_bbox[2] < x1 or
                    y2 < word_bbox[1] or word_bbox[3] < y1):
                involved_lines.add(word_info["line"])

    # 提取涉及行的文本
    line_texts = {}
    for word_info in word_data["words"]:
        line_num = word_info["line"]
        if line_num in involved_lines:
            if line_num not in line_texts:
                line_texts[line_num] = []
            line_texts[line_num].append(word_info["word"])

    # 合并每行的文本
    extracted_lines = []
    for line_num in sorted(line_texts.keys()):
        line_text = " ".join(line_texts[line_num])
        extracted_lines.append(line_text)

    extracted_text = "\n".join(extracted_lines)

    # 返回patch坐标用于调试
    top_patch_coords = []
    for patch_idx in top_k_indices:
        patch_row = patch_idx // grid_w
        patch_col = patch_idx % grid_w
        y1 = int(patch_row * patch_height)
        x1 = int(patch_col * patch_width)
        y2 = int((patch_row + 1) * patch_height)
        x2 = int((patch_col + 1) * patch_width)
        top_patch_coords.append((x1, y1, x2, y2))

    # 生成debug图像
    try:
        merged_evidence_path = os.path.join(os.path.dirname(word_mapping_path), "merged_evidence.png")
        if os.path.exists(merged_evidence_path):
            debug_img = cv2.imread(merged_evidence_path)

            # 在debug图像上绘制top10 patches
            for i, (x1, y1, x2, y2) in enumerate(top_patch_coords):
                # 使用不同的颜色区分不同的patch
                color = (0, 255 - i * 25, i * 25)  # 从蓝到红的渐变色
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

                # 添加patch编号
                cv2.putText(debug_img, f"{i+1}", (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 保存debug图像
            debug_output_path = os.path.join(os.path.dirname(word_mapping_path), "debug_top10_patches.png")
            cv2.imwrite(debug_output_path, debug_img)
            print(f"Debug image saved to: {debug_output_path}")
        else:
            print(f"Warning: merged_evidence.png not found at {merged_evidence_path}")
    except Exception as e:
        print(f"Warning: Failed to generate debug image: {e}")

    return extracted_text, top_patch_coords

def main():
    parser = argparse.ArgumentParser()
    # 模型路径
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    # 数据集路径
    parser.add_argument("--data_path", type=str, default="data/qasper/qasper-test-sample.json")
    # 结果保存路径
    parser.add_argument("--save_dir", type=str, default="tem/ver_pipeline_qasper_results_20")

    # --- 渲染配置文件路径 ---
    parser.add_argument("--config_path", type=str,
                        default="config/config_en.json",
                        help="Path to the render configuration JSON file")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of papers to process")
    parser.add_argument("--font_path", type=str, default=None, help="Override font path in config if needed")
    
    args = parser.parse_args()

    # --- 1. 加载渲染配置 ---
    print(f"Loading render config from: {args.config_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
        
    with open(args.config_path, 'r', encoding='utf-8') as f:
        render_config = json.load(f)

    # 如果命令行指定了字体，覆盖配置文件中的设置
    if args.font_path:
        print(f"Overriding font path with: {args.font_path}")
        render_config["font_path"] = args.font_path

    # --- 2. 初始化引擎 ---
    config = InferenceConfig(
        model_path=args.model_path, 
        dataset_path=args.data_path, 
        save_base_dir=args.save_dir
    )
    
    print(f"Loading model from {args.model_path}...")
    engine = QwenEngine_img_no_eager(config)
    
    print("Loading dataset...")
    papers = DatasetUtils.load_qasper(args.data_path)
    print(f"Loaded {len(papers)} papers.")

    # --- 新增部分：初始化 prediction.jsonl 文件 ---
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    prediction_file_path = os.path.join(args.save_dir, "prediction.jsonl")

    # 加载已有的预测结果，避免重复处理
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
    else:
        print("No existing prediction file found. Starting fresh.")

    print(f"Prediction file will be saved to: {prediction_file_path}")
    # --------------------------------------------

    # --- 3. 开始循环处理 ---
    processed_count = 0
    skipped_count = 0

    for paper in tqdm(papers[:args.limit]):
        pid, full_text_context = DatasetUtils.build_context(paper, config.max_context_chars)

        # 遍历该论文下的所有问题
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

            # 检查是否已经处理过
            if is_question_processed(save_path, qid, processed_qids):
                skipped_count += 1
                continue

            os.makedirs(save_path, exist_ok=True)
            processed_count += 1
            
            # 图片保存目录
            img_output_dir = os.path.join(save_path, "rendered_images")
            os.makedirs(img_output_dir, exist_ok=True)

            # --- 4. 文本转图片 ---
            # print(f"Rendering images for Question {qid}...")
            
            image_paths = process_single_text_evidence(
                txt=full_text_context,
                output_root=img_output_dir,
                config=render_config, 
                evidence_text=gold_evidence 
            )

            if not image_paths:
                print(f"Warning: Image generation failed for {pid}-{qid}. Skipping.")
                continue

            # --- 5. 第一阶段：捕获第一个token的注意力 ---
            print(f"Phase 1: Capturing first token attention for {qid}...")
            prompt_context = "Please answer the question based on the document images provided."

            try:
                # 捕获注意力并获取tokens
                attn_result = engine.get_first_attention(prompt_context, question_text, image_paths)

                if attn_result is None or attn_result["attn_data"] is None:
                    print(f"Warning: Failed to capture attention for {pid}-{qid}. Skipping.")
                    continue

                attn_data = attn_result["attn_data"]
                tokens = attn_result["input_tokens"]

                # 保存input tokens
                with open(os.path.join(save_path, "input_tokens.json"), "w", encoding='utf-8') as f:
                    json.dump(tokens, f, ensure_ascii=False)

                # 获取视觉token索引
                visual_indices = get_visual_token_indices(save_path)
                if visual_indices is None:
                    print(f"Warning: Could not find visual token indices for {pid}-{qid}. Skipping.")
                    continue

                # 计算patch分布 - 使用实际返回的图片路径
                merged_path = image_paths[0]  # process_single_text_evidence 返回的第一个路径就是 merged.png
                visual_token_count = visual_indices[1] - visual_indices[0]
                grid_w, grid_h = calculate_patch_distribution(merged_path, visual_token_count)

                # 提取word_mapping路径 - word_mapping.json 和图片在同一目录
                word_mapping_path = os.path.join(os.path.dirname(image_paths[0]), "word_mapping.json")

                if not os.path.exists(word_mapping_path):
                    print(f"Warning: word_mapping.json not found at {word_mapping_path} for {pid}-{qid}. Skipping.")
                    continue

                # --- 6. 提取top-10 patches对应的文本 ---
                print(f"Phase 2: Extracting top-10 patches text for {qid}...")
                extracted_text, top_patch_coords = extract_top_patches_text(
                    attn_data, visual_indices, word_mapping_path, grid_w, grid_h, top_k=10
                )

                # 保存提取的文本和patch坐标用于调试
                with open(os.path.join(save_path, "extracted_evidence.txt"), "w", encoding='utf-8') as f:
                    f.write(extracted_text)

                with open(os.path.join(save_path, "top_patch_coords.json"), "w") as f:
                    json.dump(top_patch_coords, f)

                # --- 7. 第二阶段：基于提取的文本重新生成答案 ---
                print(f"Phase 3: Generating final answer for {qid}...")

                # 构建新的prompt，包含提取的证据
                enhanced_context = f"{prompt_context}\n\nExtracted evidence (maybe useful):\n{extracted_text}"
                print("新的Prompt：",enhanced_context)

                res = engine.run_no_attention(enhanced_context, question_text, image_paths)
                
                # --- 6. 保存结果 ---
                
                # 原始文本
                with open(os.path.join(save_path, "context.txt"), "w") as f: 
                    f.write(full_text_context)
                
                # 答案
                with open(os.path.join(save_path, "answer.txt"), "w") as f: 
                    f.write(res["answer"])
                
                # Golden Evidence
                with open(os.path.join(save_path, "gold_evidence.json"), "w", encoding='utf-8') as f:
                    json.dump(gold_evidence, f, ensure_ascii=False, indent=2)

                # Top patch坐标
                with open(os.path.join(save_path, "top_patch_coords.json"), "w") as f:
                    json.dump(top_patch_coords, f)

                # --- 新增部分：追加写入 prediction.jsonl ---
                pred_entry = {
                    "question_id": qid,
                    "predicted_answer": res["answer"],
                    "predicted_evidence": extracted_text  # 使用提取的证据文本
                }
                
                with open(prediction_file_path, "a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")
                # -----------------------------------------------
            
            except Exception as e:
                print(f"Error during inference for {pid}-{qid}: {e}")
                import traceback
                traceback.print_exc()

            # 垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

    # 打印统计信息
    print("Processing completed!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Total questions: {processed_count + skipped_count}")

if __name__ == "__main__":
    main()