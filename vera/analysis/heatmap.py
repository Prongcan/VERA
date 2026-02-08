"""
Heatmap generation for VERA
提供注意力热力图生成功能
"""

import os
import json
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np


def create_heatmap(
    image_path: str,
    attention_data: list,
    output_path: str,
    mode: str = "overlay",
    alpha: float = 0.5,
    top_k: int = 10
) -> str:
    """
    生成 attention 热力图

    Args:
        image_path: 基础图像路径
        attention_data: attention 数据 (从 attn_first_token.json 读取的列表)
        output_path: 输出路径
        mode: "overlay" (热力图叠加) 或 "top_k" (Top K 高亮)
        alpha: 叠加透明度 (0-1)
        top_k: Top K 数量 (仅在 mode="top_k" 时使用)

    Returns:
        str: 生成的热力图路径

    Examples:
        >>> import json
        >>> with open('attn_first_token.json') as f:
        ...     attn_data = json.load(f)
        >>> path = analysis.create_heatmap(
        ...     image_path="merged.png",
        ...     attention_data=attn_data,
        ...     output_path="heatmap.png",
        ...     mode="overlay"
        ... )
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h_img, w_img = img.shape[:2]

    # Convert attention data to numpy array
    attn_array = np.array(attention_data).flatten()

    # Calculate grid dimensions from total visual tokens
    num_patches = len(attn_array)
    grid_w, grid_h = _calculate_patch_distribution(w_img, h_img, num_patches)

    if mode == "overlay":
        result = _create_attention_overlay(img, attn_array, grid_h, grid_w, alpha)
    elif mode == "top_k":
        result = _create_top_k_overlay(img, attn_array, grid_h, grid_w, top_k)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'overlay' or 'top_k'")

    # Save result
    cv2.imwrite(output_path, result)
    return output_path


def get_top_k_patches(
    attention_data: list,
    image_height: int,
    image_width: int,
    k: int = 10
) -> List[Tuple[int, int, int, int]]:
    """
    获取 Top K patch 的像素边界

    Args:
        attention_data: attention 数据列表
        image_height: 图像高度
        image_width: 图像宽度
        k: Top K 数量

    Returns:
        List of (x1, y1, x2, y2) 像素边界列表

    Examples:
        >>> patches = analysis.get_top_k_patches(
        ...     attention_data=attn_data,
        ...     image_height=1000,
        ...     image_width=800,
        ...     k=10
        ... )
        >>> print(patches)  # [(x1, y1, x2, y2), ...]
    """
    attn_array = np.array(attention_data).flatten()
    num_patches = len(attn_array)

    grid_w, grid_h = _calculate_patch_distribution(image_width, image_height, num_patches)

    top_k_indices = np.argsort(attn_array)[-k:]

    patch_width = image_width / grid_w
    patch_height = image_height / grid_h

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


def _calculate_patch_distribution(img_w: int, img_h: int, total_visual_tokens: int) -> Tuple[int, int]:
    """Calculate patch grid dimensions from total visual tokens and image aspect ratio"""
    aspect_ratio = img_h / img_w
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


def _create_attention_overlay(base_img, attn_array, grid_h, grid_w, alpha=0.5):
    """Create attention heatmap overlay"""
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


def _create_top_k_overlay(base_img, attn_array, grid_h, grid_w, k=10):
    """Create top-k highlight overlay"""
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
    color_layer[:] = (0, 0, 255)  # Red color
    output_img = base_img.copy()

    overlay_alpha = 0.5
    roi = output_img[mask_resized == 1]
    overlay_roi = color_layer[mask_resized == 1]
    if len(roi) > 0:
        blended = cv2.addWeighted(roi, 1.0 - overlay_alpha, overlay_roi, overlay_alpha, 0)
        output_img[mask_resized == 1] = blended

    return output_img
