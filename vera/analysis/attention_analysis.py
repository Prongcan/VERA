"""
Attention analysis utilities for VERA
提供 attention 数据分析的工具函数
"""

import numpy as np
from typing import List, Tuple


def aggregate_attention_with_target_heads(
    attn_data,
    visual_start: int,
    visual_end: int,
    visual_token_count: int,
    target_heads: List[Tuple[int, int]]
) -> np.ndarray:
    """
    使用预定义的 target heads 聚合 attention 数据

    Args:
        attn_data: 多层 attention 数据
        visual_start: 视觉 token 起始索引
        visual_end: 视觉 token 结束索引
        visual_token_count: 视觉 token 总数
        target_heads: 目标 heads 列表 [(layer, head), ...]

    Returns:
        聚合后的 attention 向量 (numpy array)

    Examples:
        >>> from vera.analysis import aggregate_attention_with_target_heads
        >>>
        >>> target_heads = [(24, 29), (21, 11), (24, 8)]
        >>> avg_attn = aggregate_attention_with_target_heads(
        ...     attn_data=attn_data,
        ...     visual_start=0,
        ...     visual_end=100,
        ...     visual_token_count=100,
        ...     target_heads=target_heads
        ... )
    """
    avg_attn_vector = np.zeros(visual_token_count)
    valid_heads_count = 0

    for layer_idx, head_idx in target_heads:
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

    return avg_attn_vector


def calculate_patch_distribution(
    img_width: int,
    img_height: int,
    total_visual_tokens: int
) -> Tuple[int, int]:
    """
    根据图片比例和 token 数量计算最合适的 patch 网格分布

    Args:
        img_width: 图像宽度（像素）
        img_height: 图像高度（像素）
        total_visual_tokens: 视觉 token 总数

    Returns:
        (grid_w, grid_h): patch 网格尺寸

    Examples:
        >>> from vera.analysis import calculate_patch_distribution
        >>>
        >>> grid_w, grid_h = calculate_patch_distribution(
        ...     img_width=1920,
        ...     img_height=1080,
        ...     total_visual_tokens=256
        ... )
        >>> print(f"Patch grid: {grid_w}x{grid_h}")
    """
    aspect_ratio = img_height / img_width
    num_patches = total_visual_tokens

    # 找所有因子对
    factors = []
    for i in range(1, int(np.sqrt(num_patches)) + 1):
        if num_patches % i == 0:
            factors.append((i, num_patches // i))
            if i != num_patches // i:
                factors.append((num_patches // i, i))

    # 找最接近图片比例的因子对
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
        grid_size = int(np.sqrt(num_patches))
        best_fit = (grid_size, grid_size)

    return best_fit


def get_top_patches_from_attn(
    attn_vector: np.ndarray,
    grid_w: int,
    grid_h: int,
    img_width: int,
    img_height: int,
    top_k: int = 10
) -> List[Tuple[int, int, int, int]]:
    """
    从聚合的 attention 向量获取 top-k patches 的像素坐标

    Args:
        attn_vector: 聚合后的 attention 向量
        grid_w, grid_h: patch 网格尺寸
        img_width, img_height: 图像尺寸（像素）
        top_k: 返回 top k 个 patches

    Returns:
        List of patch coordinates [(x1, y1, x2, y2), ...]

    Examples:
        >>> from vera.analysis import get_top_patches_from_attn
        >>>
        >>> patches = get_top_patches_from_attn(
        ...     attn_vector=avg_attn,
        ...     grid_w=16,
        ...     grid_h=16,
        ...     img_width=1920,
        ...     img_height=1080,
        ...     top_k=10
        ... )
    """
    # 获取 top-k patches 的索引
    if top_k > len(attn_vector):
        top_k = len(attn_vector)

    top_k_indices = np.argsort(attn_vector)[-top_k:]

    # 计算 patch 像素坐标
    patch_height = img_height / grid_h
    patch_width = img_width / grid_w

    top_patch_coords = []
    for patch_idx in top_k_indices:
        patch_row = patch_idx // grid_w
        patch_col = patch_idx % grid_w

        y1 = int(patch_row * patch_height)
        x1 = int(patch_col * patch_width)
        y2 = int((patch_row + 1) * patch_height)
        x2 = int((patch_col + 1) * patch_width)

        top_patch_coords.append((x1, y1, x2, y2))

    return top_patch_coords
