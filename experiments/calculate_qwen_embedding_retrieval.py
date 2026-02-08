"""
Qwen3-VL Embedding 检索脚本 (使用 VERA API)
使用 Qwen3-VL-8B-Instruct 的 embedding 模型进行检索
检索出与 question 相关度排名前 K 的句子，提取对应文字

注意：qasper 数据集没有 word_mapping.json，使用 context.txt 进行文本检索

Note: This script now uses the new retrieval.qwen_embedding_retrieve() API
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

# ================= 配置区域 =================
# 路径设置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 使用新的 VERA API
from vera import retrieval

ROOT_DIR = "tem/qasper_qwen_img"

# 加载模型配置
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/model_config.json")
if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    MODEL_PATH = model_config.get("model_path", {}).get("qwen")
    if not MODEL_PATH:
        raise ValueError("model_path.qwen not found in config/model_config.json")
else:
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_PATH}")

# Top K 句子数量
TOP_K_SENTENCES = 10
# ===========================================


def extract_question_from_tokens(input_tokens_path: str, tokenizer) -> Optional[str]:
    """Extract question text from input_tokens.json"""
    if not os.path.exists(input_tokens_path) or tokenizer is None:
        return None

    try:
        with open(input_tokens_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)

        vision_end_idx = None
        for i, token in enumerate(tokens):
            if token == '<|vision_end|>':
                vision_end_idx = i
                break

        if vision_end_idx is None:
            return None

        question_tokens = []
        for i in range(vision_end_idx + 1, len(tokens)):
            token = tokens[i]
            if token == '<|im_end|>':
                break
            question_tokens.append(token)

        if tokenizer is not None:
            token_ids = tokenizer.convert_tokens_to_ids(question_tokens)
            question_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return question_text.strip()

        return None

    except Exception:
        return None


def main():
    import argparse
    from tqdm import tqdm
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description="Qwen Embedding Retrieval using VERA API")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to Qwen model")
    parser.add_argument("--data_path", type=str, default=ROOT_DIR,
                        help="Path to data directory containing question folders")
    parser.add_argument("--save_dir", type=str, default=ROOT_DIR,
                        help="Directory to save retrieval results")
    parser.add_argument("--top_k", type=int, default=TOP_K_SENTENCES,
                        help="Number of top sentences to retrieve")

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen Embedding Retrieval (using new VERA API)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Top K: {args.top_k}")
    print("=" * 60)

    # Load Qwen processor for question extraction
    qwen_processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    qwen_tokenizer = qwen_processor.tokenizer

    # Find all question folders
    folders = []
    for root, dirs, files in os.walk(args.data_path):
        if "result" in root:
            continue
        if "input_tokens.json" in files and "context.txt" in files:
            folders.append(root)

    print(f"Found {len(folders)} question folders")

    # Process each folder using the new API
    stats = {"total": len(folders), "success": 0, "failed": 0, "errors": []}

    for folder_path in tqdm(folders, desc="Processing questions"):
        try:
            folder_name = os.path.basename(folder_path)

            # Get file paths
            input_tokens_path = os.path.join(folder_path, "input_tokens.json")
            context_path = os.path.join(folder_path, "context.txt")
            output_path = os.path.join(folder_path, "extracted_evidence_qwen_embedding.txt")

            # Extract question
            question = extract_question_from_tokens(input_tokens_path, qwen_tokenizer)
            if question is None:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "Failed to extract question"))
                continue

            # Clean question text
            prefix_marker = "Please answer the question based on the document images provided."
            if question.startswith(prefix_marker):
                question = question[len(prefix_marker):].strip()
            cutoff_marker = "Please output your answer **directly**"
            if cutoff_marker in question:
                question = question.split(cutoff_marker)[0].strip()

            # Read context
            with open(context_path, 'r', encoding='utf-8') as f:
                context = f.read().strip()

            if not context:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "Empty context.txt"))
                continue

            # Use the new API: retrieval.qwen_embedding_retrieve()
            extracted_text = retrieval.qwen_embedding_retrieve(
                model_path=args.model_path,
                context_text=context,
                query=question,
                top_k=args.top_k
            )

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            stats["success"] += 1

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append((folder_name, str(e)))

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Retrieval Complete")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for folder, error in stats['errors'][:5]:  # 只显示前 5 个错误
            print(f"  - {folder}: {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    print("=" * 60)


if __name__ == "__main__":
    main()
