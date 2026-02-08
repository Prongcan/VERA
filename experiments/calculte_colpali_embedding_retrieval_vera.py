"""
ColPali Embedding 检索脚本 (使用 VERA API)
使用 ColPali 模型进行 embedding 检索
检索出与 question 相关度排名前 K 的 patch，提取对应文字
"""

import os
import sys
import json
from pathlib import Path

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


def main():
    import argparse

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
    print("ColPali Retrieval (using VERA API)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Top K: {args.top_k}")
    print("=" * 60)

    # 调用 vera.retrieval.colpali 函数
    stats = retrieval.colpali(
        model_name=args.model_name,
        data_path=args.data_path,
        save_dir=args.save_dir,
        top_k=args.top_k,
        overlap=args.overlap,
        qwen_model_path=args.qwen_model_path
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
