"""
Qwen3-VL Inference Script (using VERA API)
Updated to use the new vera.models and vera.rendering APIs
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

# --- 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# 添加PyTorch内存优化环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 模块导入 (使用新的 VERA API) ---
from vera import models, rendering
from data.dataset_loader import DatasetUtils

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

def main():
    parser = argparse.ArgumentParser()
    # 模型路径
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    # 数据集路径
    parser.add_argument("--data_path", type=str, default="data/qasper/qasper-test-sample.json")
    # 结果保存路径
    parser.add_argument("--save_dir", type=str, default="tem/qasper_qwen_img")

    # 渲染配置文件路径
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

    # --- 2. 初始化引擎 (使用新的 VERA API) ---
    print(f"Initializing VERA model from {args.model_path}...")
    engine = models.initialize(
        model_path=args.model_path,
        model_type="qwen-img"  # 标准版本，不使用 masking
    )

    print("Loading dataset...")
    papers = DatasetUtils.load_qasper(args.data_path)
    print(f"Loaded {len(papers)} papers.")

    # 初始化 prediction.jsonl 文件
    os.makedirs(args.save_dir, exist_ok=True)
    prediction_file_path = os.path.join(args.save_dir, "prediction.jsonl")

    if os.path.exists(prediction_file_path):
        os.remove(prediction_file_path)
    print(f"Prediction file will be saved to: {prediction_file_path}")

    # --- 3. 开始循环处理 ---
    for paper in tqdm(papers[:args.limit]):
        pid, full_text_context = DatasetUtils.build_context(paper, 100000000)  # max_context_chars

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
            os.makedirs(save_path, exist_ok=True)

            # 图片保存目录
            img_output_dir = os.path.join(save_path, "rendered_images")
            os.makedirs(img_output_dir, exist_ok=True)

            # --- 4. 文本转图片 (使用新的 VERA API) ---
            image_paths = rendering.text_to_image(
                text=full_text_context,
                output_dir=img_output_dir,
                config=render_config,
                evidence_text=gold_evidence  # 可选：高亮显示证据
            )

            if not image_paths:
                print(f"Warning: Image generation failed for {pid}-{qid}. Skipping.")
                continue

            # --- 5. 运行推理 (使用新的 VERA API) ---
            prompt_context = "Please answer the question based on the document images provided."

            try:
                res = engine.run(
                    prompt_context=prompt_context,
                    question_text=question_text,
                    image_paths=image_paths,
                    is_mask_heads=False,  # 不使用 masking
                    heads_positions=None
                )

                # --- 6. 保存结果 ---

                # 原始文本
                with open(os.path.join(save_path, "context.txt"), "w", encoding="utf-8") as f:
                    f.write(full_text_context)

                # 答案
                with open(os.path.join(save_path, "answer.txt"), "w", encoding="utf-8") as f:
                    f.write(res["answer"])

                # Golden Evidence
                with open(os.path.join(save_path, "gold_evidence.json"), "w", encoding='utf-8') as f:
                    json.dump(gold_evidence, f, ensure_ascii=False, indent=2)

                # 追加写入 prediction.jsonl
                pred_entry = {
                    "question_id": qid,
                    "predicted_answer": res["answer"],
                    "predicted_evidence": ""  # 暂时留空
                }

                with open(prediction_file_path, "a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")

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

                    with open(os.path.join(save_path, "attn_first_token.json"), "w", encoding="utf-8") as f:
                        json.dump(cleaned, f, ensure_ascii=False)

            except Exception as e:
                print(f"Error during inference for {pid}-{qid}: {e}")
                import traceback
                traceback.print_exc()

            # 垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
