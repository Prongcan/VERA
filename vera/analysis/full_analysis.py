"""
Full Analysis Module for VERA
提供完整的注意力分析流程：Phase 1 扫描、Phase 2 可视化、Phase 3 全局热力图
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import List, Tuple, Dict, Optional, Tuple as TupleType
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

# 从 retrieval 模块导入文本检索函数
from vera.retrieval import find_word_mapping_path, extract_evidence_from_patches


def calculate_patch_distribution(img_path, total_visual_tokens):
    """计算 patch 分布（从图像宽高比推断）"""
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
        current_ratio = h / w
        ratio_diff = abs(current_ratio - aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_fit = (w, h)

    if best_fit is None:
        grid_size = int(math.sqrt(num_patches))
        best_fit = (grid_size, grid_size)
    return best_fit


def get_visual_token_indices(question_folder):
    """从 input_tokens.json 中提取 visual token 的索引范围"""
    input_tokens_path = os.path.join(question_folder, "input_tokens.json")
    if not os.path.exists(input_tokens_path):
        return None
    try:
        with open(input_tokens_path, 'r') as f:
            tokens = json.load(f)
        vision_start_idx, vision_end_idx = None, None
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


def load_attention_data(question_folder):
    """加载 attention 数据"""
    attn_path = os.path.join(question_folder, "attn_first_token.json")
    if not os.path.exists(attn_path):
        return None
    try:
        with open(attn_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def calculate_entropy_confidence(prob_dist):
    """计算基于归一化熵的置信度"""
    probs = prob_dist + 1e-12
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log2(probs))
    n = len(probs)
    if n <= 1:
        return 1.0
    max_entropy = np.log2(n)
    confidence = 1.0 - (entropy / max_entropy)
    return max(0.0, min(1.0, confidence))


def calculate_iou(attention_vector, mask_vector):
    """计算Attention向量A与Mask向量M的IoU"""
    min_len = min(len(attention_vector), len(mask_vector))
    A = attention_vector[:min_len]
    M = mask_vector[:min_len]

    intersection = np.sum(A * M)
    union = np.sum(A) + np.sum(M) - intersection

    if union <= 1e-12:
        return 0.0

    iou = intersection / union
    return max(0.0, min(1.0, iou))


def calculate_mse(attention_vector, mask_vector):
    """计算Attention向量A与Mask向量M的MSE"""
    min_len = min(len(attention_vector), len(mask_vector))
    A = attention_vector[:min_len]
    M = mask_vector[:min_len]
    mse = np.mean((A - M) ** 2)
    return mse


def create_attention_overlay(base_img, attn_array, grid_h, grid_w, alpha=0.5):
    """创建 attention 叠加热力图"""
    h_img, w_img = base_img.shape[:2]
    num_patches = grid_h * grid_w
    if len(attn_array) < num_patches:
        attn_array = np.pad(attn_array, (0, num_patches - len(attn_array)))
    attn_array = attn_array[:num_patches]

    try:
        attn_grid = attn_array.reshape((grid_h, grid_w))
    except ValueError:
        return base_img

    attn_norm = cv2.normalize(attn_grid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    attn_resized = cv2.resize(attn_norm, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(base_img, 1.0 - alpha, heatmap, alpha, 0)
    return overlay_img


def create_top_k_overlay(base_img, attn_array, grid_h, grid_w, k=10, color=(0, 0, 255)):
    """创建 Top K patches 叠加图"""
    h_img, w_img = base_img.shape[:2]
    num_patches = grid_h * grid_w
    if len(attn_array) < num_patches:
        attn_array = np.pad(attn_array, (0, num_patches - len(attn_array)))
    attn_array = attn_array[:num_patches]

    top_k_indices = np.argsort(attn_array)[-k:]
    mask_grid = np.zeros((grid_h * grid_w), dtype=np.uint8)
    mask_grid[top_k_indices] = 1

    try:
        mask_grid_2d = mask_grid.reshape((grid_h, grid_w))
    except ValueError:
        return base_img

    mask_resized = cv2.resize(mask_grid_2d, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    color_layer = np.zeros_like(base_img)
    color_layer[:] = color
    output_img = base_img.copy()

    alpha = 0.5
    roi = output_img[mask_resized == 1]
    overlay_roi = color_layer[mask_resized == 1]
    if len(roi) > 0:
        blended = cv2.addWeighted(roi, 1.0 - alpha, overlay_roi, alpha, 0)
        output_img[mask_resized == 1] = blended

    return output_img


def get_top_k_patch_pixel_bounds(attn_array, grid_h, grid_w, img_h, img_w, k=10):
    """计算 Top K patch 的像素边界"""
    num_patches = grid_h * grid_w
    if len(attn_array) < num_patches:
        attn_array = np.pad(attn_array, (0, num_patches - len(attn_array)))
    attn_array = attn_array[:num_patches]

    top_k_indices = np.argsort(attn_array)[-k:]

    patch_width = img_w / grid_w
    patch_height = img_h / grid_h

    bounds_list = []
    for patch_idx in top_k_indices:
        row = patch_idx // grid_w
        col = patch_idx % grid_w

        x1 = int(col * patch_width)
        y1 = int(row * patch_height)
        x2 = int((col + 1) * patch_width)
        y2 = int((row + 1) * patch_height)

        bounds_list.append((x1, y1, x2, y2))

    return bounds_list



def process_single_folder(
    folder_path: str,
    candidate_heads: Optional[List[TupleType[int, int]]] = None,
    mode: str = 'scan',
    red_lower: Tuple[int, int, int] = (0, 0, 150),
    red_upper: Tuple[int, int, int] = (100, 100, 255),
    kernel_size: Tuple[int, int] = (12, 3),
    dilation_iterations: int = 1,
    debug_box_color: Tuple[int, int, int] = (255, 0, 0),
    debug_box_thickness: int = 2,
    debug_heatmap_alpha: float = 0.6,
    top_k_patches: int = 10,
    top_k_color: Tuple[int, int, int] = (0, 0, 255),
    save_debug: bool = False,
    output_dir: Optional[str] = None
):
    """
    处理单个文件夹（核心逻辑）

    Args:
        folder_path: 文件夹路径
        candidate_heads: 候选的 (layer, head) 列表
        mode: 'scan' 或 'viz'
        red_lower, red_upper: 红色范围
        kernel_size: 膨胀核大小
        dilation_iterations: 膨胀次数
        debug_box_color, debug_box_thickness: debug 边框设置
        debug_heatmap_alpha: 热力图透明度
        top_k_patches: Top K 数量
        top_k_color: Top K 颜色
        save_debug: 是否保存 debug 图
        output_dir: 输出目录

    Returns:
        scan 模式: (folder_name, score_matrix, confidence_matrix)
        viz 模式: folder_name 或 None
    """
    try:
        folder_name = os.path.basename(folder_path)
        question_folder = os.path.dirname(os.path.dirname(folder_path))
        evidence_path = os.path.join(folder_path, "merged_evidence.png")
        merged_path = os.path.join(folder_path, "merged.png")

        attn_data = load_attention_data(question_folder)
        if attn_data is None:
            return None if mode == 'scan' else None
        visual_indices = get_visual_token_indices(question_folder)
        if visual_indices is None:
            return None if mode == 'scan' else None

        visual_start, visual_end = visual_indices
        visual_token_count = visual_end - visual_start
        grid_w, grid_h = calculate_patch_distribution(merged_path, visual_token_count)

        # 从 evidence 图像提取 mask
        img = cv2.imread(evidence_path)
        if img is None:
            return None if mode == 'scan' else None

        h_img, w_img = img.shape[:2]
        RED_LOWER = np.array(red_lower)
        RED_UPPER = np.array(red_upper)
        DILATION_KERNEL_SIZE = kernel_size

        mask_color = cv2.inRange(img, RED_LOWER, RED_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATION_KERNEL_SIZE)
        mask_dilated = cv2.dilate(mask_color, kernel, iterations=dilation_iterations)
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_boxes = np.zeros((h_img, w_img), dtype=np.uint8)
        debug_base_img = img.copy() if save_debug else None

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 4 and h > 4:
                cv2.rectangle(mask_boxes, (x, y), (x + w, y + h), 255, thickness=-1)
                if save_debug:
                    cv2.rectangle(debug_base_img, (x, y), (x + w, y + h), debug_box_color, thickness=debug_box_thickness)

        if save_debug and debug_base_img is not None:
            debug_path = os.path.join(output_dir, f"{folder_name}_debug_boxes.png")
            if not os.path.exists(debug_path):
                cv2.imwrite(debug_path, debug_base_img)

        # 创建 patch 级别的 mask
        patch_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
        patch_h = h_img // grid_h
        patch_w = w_img // grid_w

        for i in range(grid_h):
            for j in range(grid_w):
                y1 = i * patch_h
                y2 = min((i + 1) * patch_h, h_img)
                x1 = j * patch_w
                x2 = min((j + 1) * patch_w, w_img)

                patch_region = mask_boxes[y1:y2, x1:x2]
                if np.any(patch_region > 0):
                    patch_mask[i, j] = 1

        patch_weights = patch_mask.flatten().astype(np.float32)

        num_layers = len(attn_data)
        num_heads = len(attn_data[0])
        score_matrix = np.zeros((num_layers, num_heads))
        confidence_matrix = np.zeros((num_layers, num_heads))

        # 计算得分矩阵
        for layer_idx in range(num_layers):
            heads_data = attn_data[layer_idx]
            for head_idx, attn_list in enumerate(heads_data):
                full_attn_array = np.array(attn_list[0])
                visual_attn = full_attn_array[visual_start:visual_end]

                if len(visual_attn) != len(patch_weights):
                    min_len = min(len(visual_attn), len(patch_weights))
                    attn_array = visual_attn[:min_len]
                    current_weights = patch_weights[:min_len]
                else:
                    attn_array = visual_attn
                    current_weights = patch_weights

                weighted_score = np.sum(attn_array * current_weights)
                total_attn = np.sum(attn_array) + 1e-9
                score_matrix[layer_idx, head_idx] = weighted_score / total_attn
                confidence_matrix[layer_idx, head_idx] = calculate_entropy_confidence(attn_array)

        # Scan Mode: 直接返回
        if mode == 'scan':
            return (folder_name, score_matrix, confidence_matrix)

        # Viz Mode: 生成可视化
        if mode == 'viz' and debug_base_img is not None and candidate_heads is not None:
            best_5_heads = [(l, h) for (l, h) in candidate_heads if l < num_layers and h < num_heads]

            # 聚合 Top 5 头的 attention
            avg_attn_vector = np.zeros(visual_token_count)
            valid_heads_count = 0

            for layer_idx, head_idx in best_5_heads:
                full_target_attn = np.array(attn_data[layer_idx][head_idx][0])
                visual_attn_part = full_target_attn[visual_start:visual_end]

                if len(visual_attn_part) < visual_token_count:
                    visual_attn_part = np.pad(visual_attn_part, (0, visual_token_count - len(visual_attn_part)))
                elif len(visual_attn_part) > visual_token_count:
                    visual_attn_part = visual_attn_part[:visual_token_count]

                avg_attn_vector += visual_attn_part
                valid_heads_count += 1

            if valid_heads_count > 0:
                avg_attn_vector /= valid_heads_count

                # 生成热力图
                final_debug_img = create_attention_overlay(
                    debug_base_img, avg_attn_vector, grid_h, grid_w, alpha=debug_heatmap_alpha
                )
                debug_path = os.path.join(output_dir, f"{folder_name}_GlobalTop5_heatmap.png")
                cv2.imwrite(debug_path, final_debug_img)

                # 生成 Top K patches 图
                top10_debug_img = create_top_k_overlay(
                    debug_base_img, avg_attn_vector, grid_h, grid_w, k=top_k_patches, color=top_k_color
                )
                top10_path = os.path.join(output_dir, f"{folder_name}_GlobalTop5_Top{top_k_patches}Patch.png")
                cv2.imwrite(top10_path, top10_debug_img)

                # 提取证据文本
                img_for_bounds = cv2.imread(merged_path)
                if img_for_bounds is not None:
                    h_img, w_img = img_for_bounds.shape[:2]

                    patch_bounds = get_top_k_patch_pixel_bounds(
                        avg_attn_vector, grid_h, grid_w, h_img, w_img, k=top_k_patches
                    )

                    word_mapping_path = find_word_mapping_path(question_folder, question_folder)

                    if word_mapping_path:
                        output_evidence_path = os.path.join(question_folder, "extracted_evidence.txt")
                        extract_evidence_from_patches(
                            patch_bounds, word_mapping_path, output_evidence_path
                        )
                    else:
                        print(f"  Warning: Could not find word_mapping.json for {question_folder}")

        return (folder_name, score_matrix, confidence_matrix)

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        import traceback
        traceback.print_exc()
        return None if mode == 'scan' else None


def run_full_analysis(
    root_dir: str,
    output_folder_name: str = "result",
    top_k_patches: int = 10,
    num_workers: int = 100,
    mode: str = "all",  # "all", "scan", "viz"
    red_lower: Tuple[int, int, int] = (0, 0, 150),
    red_upper: Tuple[int, int, int] = (100, 100, 255),
    kernel_size: Tuple[int, int] = (12, 3),
    dilation_iterations: int = 1,
    debug_box_color: Tuple[int, int, int] = (255, 0, 0),
    debug_box_thickness: int = 2,
    debug_heatmap_alpha: float = 0.6,
    top_k_color: Tuple[int, int, int] = (0, 0, 255),
    save_debug: bool = True,
    save_plots: bool = True,
    sample_count: int = 10
):
    """
    运行完整的分析流程：Phase 1 扫描、Phase 2 可视化、Phase 3 全局热力图

    Args:
        root_dir: 数据根目录
        output_folder_name: 输出文件夹名称
        top_k_patches: Top K patches 数量
        num_workers: 并行处理 worker 数量
        mode: 运行模式
            - "all": 运行全部三个阶段（默认）
            - "scan": 仅运行 Phase 1 扫描，确定全局 Top 5 heads
            - "viz": 仅运行 Phase 2 可视化（需要 Phase 1 已完成）
        red_lower, red_upper: 红色范围
        kernel_size: 膨胀核大小
        dilation_iterations: 膨胀次数
        debug_box_color: debug 边框颜色
        debug_box_thickness: debug 边框粗细
        debug_heatmap_alpha: 热力图透明度
        top_k_color: Top K 颜色
        save_debug: 是否保存 debug 图
        save_plots: 是否保存全局热力图
        sample_count: IoU 计算的样本数量

    Returns:
        dict: 分析结果统计
            - num_folders: 文件夹总数
            - success_count: 成功处理数量
            - skipped_count: 跳过数量
            - error_count: 错误数量
            - errors: 错误详情列表

    Examples:
        >>> from vera import analysis
        >>> stats = analysis.run_full_analysis(
        ...     root_dir="tem/qasper_qwen_img",
        ...     output_folder_name="result",
        ...     top_k_patches=10
        ... )

        仅运行扫描阶段：
        >>> stats = analysis.run_full_analysis(
        ...     root_dir="tem/qasper_qwen_img",
        ...     mode="scan"
        ... )
    """
    output_dir = os.path.join(root_dir, output_folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 查找所有文件夹
    folders = []
    for root, dirs, files in os.walk(root_dir):
        if output_folder_name in root:
            continue
        if "merged_evidence.png" in files:
            folders.append(root)

    print(f"Found {len(folders)} leaf folders.")

    # ================= Phase 1: Scan =================
    print(f"\n=== Phase 1: Scanning to determine Global Top 5 Heads by Average Score (Workers: {num_workers}) ===")

    sum_matrix = None
    count_valid_samples = 0

    scan_func = partial(process_single_folder, mode='scan',
                          red_lower=red_lower, red_upper=red_upper,
                          kernel_size=kernel_size, dilation_iterations=dilation_iterations,
                          debug_box_color=debug_box_color, debug_box_thickness=debug_box_thickness,
                          debug_heatmap_alpha=debug_heatmap_alpha, top_k_patches=top_k_patches,
                          top_k_color=top_k_color, save_debug=save_debug,
                          output_dir=output_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(scan_func, folders), total=len(folders), desc="Scanning"))

    for res in results:
        if res is None:
            continue
        folder_name, matrix, conf_matrix = res

        if sum_matrix is None:
            sum_matrix = matrix
        else:
            if sum_matrix.shape == matrix.shape:
                sum_matrix += matrix
            else:
                continue
        count_valid_samples += 1

    # 计算全局 Top 5
    global_top_5_heads = []
    top_5_scores = []

    if count_valid_samples > 0 and sum_matrix is not None:
        avg_matrix_phase1 = sum_matrix / count_valid_samples
        flattened_avg_scores = avg_matrix_phase1.flatten()
        sorted_indices = np.argsort(flattened_avg_scores)[::-1]
        top_5_indices = sorted_indices[:5]
        top_5_coords = [np.unravel_index(idx, avg_matrix_phase1.shape) for idx in top_5_indices]
        top_5_scores = [flattened_avg_scores[idx] for idx in top_5_indices]
        global_top_5_heads = top_5_coords

    # ================= Phase 2: Visualization =================
    if mode == "all" or mode == "viz":
        print(f"\n=== Phase 2: Generating Visualizations & Extracting Evidence ===")

        viz_func = partial(process_single_folder, candidate_heads=global_top_5_heads, mode='viz',
                          red_lower=red_lower, red_upper=red_upper,
                          kernel_size=kernel_size, dilation_iterations=dilation_iterations,
                          debug_box_color=debug_box_color, debug_box_thickness=debug_box_thickness,
                          debug_heatmap_alpha=debug_heatmap_alpha, top_k_patches=top_k_patches,
                          top_k_color=top_k_color, save_debug=save_debug,
                          output_dir=output_dir)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(viz_func, folders), total=len(folders), desc="Visualizing & Extracting"))
    else:
        print(f"\n=== Skipping Phase 2 (mode='{mode}') ===")
        results = []

    # ================= Phase 3: Global Heatmap =================
    if mode == "all" and save_plots and count_valid_samples > 0 and sum_matrix is not None:
        avg_matrix = sum_matrix / count_valid_samples
        print(f"\nGenerating Global Average Heatmap...")

        # 找到全局 Top 20
        flattened_scores = avg_matrix.flatten()
        sorted_indices = np.argsort(flattened_scores)[::-1]
        top_20_indices = sorted_indices[:20]
        top_20_coords = [np.unravel_index(idx, avg_matrix.shape) for idx in top_20_indices]
        top_20_scores = [flattened_scores[idx] for idx in top_20_indices]

        overall_avg_score = np.mean(avg_matrix)
        print("\n" + "="*80)
        print("GLOBAL AVERAGE ATTENTION SCORE (All Heads Average)")
        print("="*80)
        print(f"Overall Average Score: {overall_avg_score:.6f}")
        print("="*80)

        print("\n" + "="*80)
        print("TOP 20 HEADS BY GLOBAL AVERAGE ATTENTION SCORE")
        print("="*80)
        print(f"{'Rank':<5} | {'Layer':<10} | {'Head':<10} | {'Avg Attention Score':<20}")
        print("-" * 80)
        for rank, ((layer, head), score) in enumerate(zip(top_20_coords, top_20_scores), 1):
            print(f"{rank:<5} | {layer:<10} | {head:<10} | {score:<20.6f}")
        print("="*80)

        # 生成热图
        plt.figure(figsize=(14, 10))
        sns.heatmap(avg_matrix, cmap="inferno", vmin=0, cbar_kws={'label': 'Avg Attention Score'})
        plt.title(f"GLOBAL Attention Distribution\n(Marked: Top 20 Heads by Global Average Score)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")

        for (layer, head) in top_20_coords:
            plt.gca().add_patch(plt.Rectangle((head, layer), 1, 1, fill=False, edgecolor='cyan', lw=3))

        avg_save_path = os.path.join(root_dir, "GLOBAL_attention_heatmap.png")
        plt.savefig(avg_save_path, dpi=150)
        plt.close()
        print(f"\nSaved Global Heatmap to {avg_save_path}")

        # 保存归一化矩阵
        min_val = np.min(avg_matrix)
        max_val = np.max(avg_matrix)
        if max_val > min_val:
            normalized_matrix = (avg_matrix - min_val) / (max_val - min_val)
        else:
            normalized_matrix = avg_matrix

        json_output_path = os.path.join(root_dir, "GLOBAL_attention_matrix_normalized.json")
        matrix_list = normalized_matrix.tolist()

        with open(json_output_path, 'w') as f:
            json.dump(matrix_list, f)

        print(f"Saved normalized matrix to {json_output_path}")
        print("="*80)

    return {
        "total_folders": len(folders),
        "valid_samples": count_valid_samples,
        "global_top_5_heads": global_top_5_heads,
        "top_5_scores": top_5_scores
    }
