"""
Qwen3-VL Embedding 检索脚本 (使用 VERA API)
使用 Qwen3-VL-8B-Instruct 的 embedding 模型进行检索
检索出与 question 相关度排名前 K 的句子，提取对应文字

注意：qasper 数据集没有 word_mapping.json，使用 context.txt 进行文本检索
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


def main():
    import argparse

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
    print("Qwen Embedding Retrieval (using VERA API)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Top K: {args.top_k}")
    print("=" * 60)

    # 调用 vera.retrieval.qwen_embedding 函数
    stats = retrieval.qwen_embedding(
        model_path=args.model_path,
        data_path=args.data_path,
        save_dir=args.save_dir,
        top_k=args.top_k
    )

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
