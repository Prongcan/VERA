"""
ColPali Embedding 检索脚本 (使用 VERA API)
使用 ColPali 模型进行 embedding 检索
检索出与 question 相关度排名前 K 的 patch，提取对应文字

Note: This script now uses the new retrieval.colpali_retrieve() API
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
WORD_MAPPING_ROOT_DIR = "tem/qasper_qwen_img"

# 加载模型配置
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/model_config.json")
if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    MODEL_NAME = model_config.get("model_path", {}).get("colpali")
    QWEN_MODEL_PATH = model_config.get("model_path", {}).get("qwen")
    if not MODEL_NAME:
        raise ValueError("model_path.colpali not found in config/model_config.json")
    if not QWEN_MODEL_PATH:
        raise ValueError("model_path.qwen not found in config/model_config.json")
else:
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_PATH}")

OUTPUT_FOLDER_NAME = "result"

# Top K patches 数量
TOP_K_PATCHES = 20

# 切分图片时的重叠像素数
OVERLAP = 0
# ===========================================


def find_image_in_folder(folder_path: str) -> Optional[str]:
    """在文件夹中查找图像文件"""
    # First try merged.png in current folder
    merged_path = os.path.join(folder_path, "merged.png")
    if os.path.exists(merged_path):
        return merged_path

    # Try to find in rendered_images subdirectory
    for root, dirs, files in os.walk(folder_path):
        if "merged_evidence.png" in files:
            return os.path.join(root, "merged_evidence.png")
        elif "merged.png" in files:
            return os.path.join(root, "merged.png")
        # Fallback to any PNG
        elif files:
            png_files = [f for f in files if f.endswith('.png')]
            if png_files:
                return os.path.join(root, png_files[0])

    return None


def find_word_mapping_in_folder(folder_path: str) -> Optional[str]:
    """在文件夹中查找 word_mapping.json"""
    for root, dirs, files in os.walk(folder_path):
        if "word_mapping.json" in files:
            return os.path.join(root, "word_mapping.json")
    return None


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

        token_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        question_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return question_text.strip()

    except Exception:
        return None


def main():
    import argparse
    from tqdm import tqdm
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description="ColPali Retrieval using VERA API")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="ColPali model name (e.g., 'vidore/colpali-v1.2')")
    parser.add_argument("--data_path", type=str, default=ROOT_DIR,
                        help="Path to data directory")
    parser.add_argument("--save_dir", type=str, default=ROOT_DIR,
                        help="Directory to save retrieval results")
    parser.add_argument("--top_k", type=int, default=TOP_K_PATCHES,
                        help="Number of top patches to retrieve")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help="Overlap in pixels when splitting images")
    parser.add_argument("--qwen_model_path", type=str, default=QWEN_MODEL_PATH,
                        help="Path to Qwen model (for question extraction)")

    args = parser.parse_args()

    print("=" * 60)
    print("ColPali Retrieval (using new VERA API)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Top K: {args.top_k}")
    print("=" * 60)

    # Load Qwen tokenizer for question extraction
    qwen_processor = AutoProcessor.from_pretrained(
        args.qwen_model_path,
        trust_remote_code=True
    )
    qwen_tokenizer = qwen_processor.tokenizer

    # Find all question folders (folders with input_tokens.json)
    folders = []
    for root, dirs, files in os.walk(args.data_path):
        if "result" in root:
            continue
        if "input_tokens.json" in files:
            folders.append(root)

    print(f"Found {len(folders)} question folders")

    # Process each folder using the new API
    stats = {"total": len(folders), "success": 0, "failed": 0, "errors": []}

    for folder_path in tqdm(folders, desc="Processing with ColPali"):
        try:
            folder_name = os.path.basename(folder_path)

            # Get file paths
            input_tokens_path = os.path.join(folder_path, "input_tokens.json")

            # Find word_mapping.json in subdirectories
            word_mapping_path = find_word_mapping_in_folder(folder_path)
            if word_mapping_path is None:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "No word_mapping.json found"))
                continue

            # Find image
            image_path = find_image_in_folder(folder_path)
            if image_path is None:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "No image found"))
                continue

            output_path = os.path.join(folder_path, "extracted_evidence_colpali.txt")

            # Extract question
            question = extract_question_from_tokens(input_tokens_path, qwen_tokenizer)
            if question is None:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "Failed to extract question"))
                continue

            # Clean question
            prefix_marker = "Please answer the question based on the document images provided."
            if question.startswith(prefix_marker):
                question = question[len(prefix_marker):].strip()
            cutoff_marker = "Please output your answer **directly**"
            if cutoff_marker in question:
                question = question.split(cutoff_marker)[0].strip()

            # Use the new API: retrieval.colpali_retrieve()
            extracted_text = retrieval.colpali_retrieve(
                model_name=args.model_name,
                image_path=image_path,
                word_mapping_path=word_mapping_path,
                query=question,
                top_k=args.top_k,
                overlap=args.overlap
            )

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            stats["success"] += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
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
