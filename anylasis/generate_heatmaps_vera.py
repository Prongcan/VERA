"""
Heatmap Generation Script (using VERA API)
使用 attention 数据生成热力图可视化
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# ================= 配置区域 =================
ROOT_DIR = "tem/hotpot_qwen_img"
OUTPUT_FOLDER_NAME = "result"
TOP_K_PATCHES = 10
# ===========================================

# 使用新的 VERA API
from vera import analysis

OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_FOLDER_NAME)


def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def find_leaf_folders(root_dir):
    """查找所有包含 attn_first_token.json 的文件夹"""
    leaf_folders = []
    for root, dirs, files in os.walk(root_dir):
        if OUTPUT_FOLDER_NAME in root:
            continue
        # 查找包含 attention 数据的文件夹
        if "attn_first_token.json" in files:
            leaf_folders.append(root)
    return leaf_folders


def process_folder(folder_path: str):
    """
    处理单个文件夹，生成热力图

    Args:
        folder_path: 文件夹路径

    Returns:
        dict with processing status
    """
    folder_name = os.path.basename(folder_path)

    try:
        # 查找图像文件（递归搜索子文件夹）
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.png') and 'merged' in f.lower():
                    # 使用相对路径构建完整路径
                    full_path = os.path.join(root, f)
                    image_files.append(full_path)

        # 如果没有找到 merged.png，尝试找任何 PNG 文件
        if not image_files:
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    if f.endswith('.png'):
                        full_path = os.path.join(root, f)
                        image_files.append(full_path)
                        break
                if image_files:
                    break

        if not image_files:
            return {"folder": folder_name, "status": "skipped", "reason": "no image"}

        # 使用第一个找到的图像
        image_path = image_files[0]

        # 读取 attention 数据
        attn_path = os.path.join(folder_path, "attn_first_token.json")
        if not os.path.exists(attn_path):
            return {"folder": folder_name, "status": "skipped", "reason": "no attention data"}

        with open(attn_path, 'r') as f:
            attention_data = json.load(f)

        # 生成 overlay 热力图
        heatmap_overlay_path = os.path.join(OUTPUT_DIR, f"{folder_name}_heatmap_overlay.png")
        analysis.create_heatmap(
            image_path=image_path,
            attention_data=attention_data,
            output_path=heatmap_overlay_path,
            mode="overlay",
            alpha=0.5
        )

        # 生成 Top K 热力图
        heatmap_top_k_path = os.path.join(OUTPUT_DIR, f"{folder_name}_heatmap_top_k.png")
        analysis.create_heatmap(
            image_path=image_path,
            attention_data=attention_data,
            output_path=heatmap_top_k_path,
            mode="top_k",
            top_k=TOP_K_PATCHES
        )

        # 获取 Top K patches 信息
        try:
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()

            top_k_patches = analysis.get_top_k_patches(
                attention_data=attention_data,
                image_height=img_height,
                image_width=img_width,
                k=TOP_K_PATCHES
            )

            # 保存 Top K patches 信息
            patches_info_path = os.path.join(OUTPUT_DIR, f"{folder_name}_top_k_patches.json")
            with open(patches_info_path, 'w') as f:
                json.dump({
                    "top_k": TOP_K_PATCHES,
                    "patches": top_k_patches
                }, f, indent=2)
        except Exception as e:
            print(f"  Warning: Failed to extract Top K patches: {e}")

        return {"folder": folder_name, "status": "success"}

    except Exception as e:
        import traceback
        return {"folder": folder_name, "status": "error", "error": str(e)}


def main():
    import argparse

    # 声明使用全局变量
    global ROOT_DIR, OUTPUT_DIR, TOP_K_PATCHES

    parser = argparse.ArgumentParser(description="Generate Heatmaps using VERA API")
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                        help="Root directory containing question folders")
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER_NAME,
                        help="Output folder name")
    parser.add_argument("--top_k", type=int, default=TOP_K_PATCHES,
                        help="Number of top patches to highlight")

    args = parser.parse_args()

    # 更新全局变量
    ROOT_DIR = args.root_dir
    OUTPUT_DIR = os.path.join(ROOT_DIR, args.output_folder)
    TOP_K_PATCHES = args.top_k

    ensure_output_dir()

    print("=" * 60)
    print("Heatmap Generation (using VERA API)")
    print("=" * 60)
    print(f"Root: {ROOT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Top K: {TOP_K_PATCHES}")
    print("=" * 60)

    # 查找所有需要处理的文件夹
    folders = find_leaf_folders(ROOT_DIR)
    print(f"Found {len(folders)} folders to process")

    if not folders:
        print("No folders found. Please check the root directory path.")
        return

    # 处理每个文件夹
    results = []
    for folder_path in tqdm(folders, desc="Processing folders"):
        result = process_folder(folder_path)
        results.append(result)

    # 统计结果
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['folder']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
