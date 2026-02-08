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

# 添加PyTorch内存优化环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# --- 模块导入 (使用新的 VERA API) ---
from vera import models, rendering
from data.dataset_loader import MusiqueLoader

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

MASK_CONFIG = {(29, 24), (11, 21),(26, 26),(23, 10),(24, 8)}

def main():
    parser = argparse.ArgumentParser()
    # 模型路径
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    # 数据集路径
    parser.add_argument("--data_path", type=str, default="data/musique/musique_dev_sample.jsonl")
    # 结果保存路径
    parser.add_argument("--save_dir", type=str, default="tem/musique_qwen_img_mask")

    # --- 渲染配置文件路径 ---
    parser.add_argument("--config_path", type=str,
                        default="config/config_en.json",
                        help="Path to the render configuration JSON file")

    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions to process")
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

    # --- 2. 初始化引擎 (使用新的 VERA API) ---
    print(f"Initializing VERA model from {args.model_path}...")
    engine = models.initialize(
        model_path=args.model_path,
        model_type="qwen-img-masked"
    )

    print("Loading dataset...")
    loader = MusiqueLoader(args.data_path)
    questions = loader.get_samples(args.limit)
    print(f"Loaded {len(questions)} questions.")

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

    for question in tqdm(questions):
        qid = question.get("question_id", "nan")
        question_text = question.get("question", "")

        # 提取 Golden Evidence（现在是列表）
        gold_evidence = loader.extract_evidence(question)

        # 构建上下文
        pid, full_text_context = loader.build_context(question, 100000000)

        # 定义保存路径
        save_path = os.path.join(
            args.save_dir,
            loader.safe_filename(pid),
            loader.safe_filename(qid)
        )

        # 检查是否已经处理过
        if is_question_processed(save_path, qid, processed_qids):
            skipped_count += 1
            continue

        os.makedirs(save_path, exist_ok=True)

        # 图片保存目录
        img_output_dir = os.path.join(save_path, "rendered_images")
        os.makedirs(img_output_dir, exist_ok=True)

        processed_count += 1

        # --- 4. 文本转图片 ---
        # print(f"Rendering images for Question {qid}...")

        image_paths = rendering.text_to_image(
            text=full_text_context,
            output_dir=img_output_dir,
            config=render_config,
            evidence_text=gold_evidence  # 现在传递列表
        )

        if not image_paths:
            print(f"Warning: Image generation failed for {pid}-{qid}. Skipping.")
            continue

        # --- 5. 运行推理 ---
        prompt_context = "Please answer the question based on the document images provided."

        try:
            res = engine.run(prompt_context, question_text, image_paths, mask_heads=MASK_CONFIG)

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

            # --- 新增部分：追加写入 prediction.jsonl ---
            # 注意：目前的 prompt 并没有要求模型显式输出 evidence，所以 predicted_evidence 暂时留空。
            # 如果你的模型输出了 evidence，请将 "" 替换为 res["evidence"] 或相应的变量。
            pred_entry = {
                "question_id": qid,
                "predicted_answer": res["answer"],
                "predicted_evidence": ""  # Musique 标准通常是列表，但这里按你要求设为字符串
            }

            with open(prediction_file_path, "a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")
            # -----------------------------------------------

            # Input Tokens
            if res.get("input_tokens"):
                with open(os.path.join(save_path, "input_tokens.json"), "w", encoding='utf-8') as f:
                    json.dump(res["input_tokens"], f, ensure_ascii=False)

            # Attention Data
            if res.get("attn_data"):
                cleaned = []
                for layer_d in res["attn_data"]:
                    arr = np.array(layer_d)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr.squeeze(0)
                    elif arr.ndim == 4:
                        arr = arr.squeeze(0)
                    cleaned.append(arr.tolist())

                with open(os.path.join(save_path, "attn_first_token.json"), "w") as f:
                    json.dump(cleaned, f)

        except Exception as e:
            print(f"Error during inference for {pid}-{qid}: {e}")
            import traceback
            traceback.print_exc()

        # 垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

    # --- 最终统计 ---
    print(f"\n{'='*50}")
    print("处理完成统计:")
    print(f"  总问题数: {len(questions)}")
    print(f"  已跳过: {skipped_count}")
    print(f"  新处理: {processed_count}")
    print(f"  预测文件: {prediction_file_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
