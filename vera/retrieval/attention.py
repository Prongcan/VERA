"""
Attention-based Retrieval Methods for VERA
基于注意力的检索方法
"""

import os
import json
from typing import List, Tuple, Optional


def attention_retrieve(
    attention_data,
    image_width: int,
    image_height: int,
    word_mapping_path: str,
    top_k: int = 10,
    output_path: Optional[str] = None
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    使用注意力数据进行检索的完整流程

    这是一个高层 API，结合了 attention 分析和文本提取

    Args:
        attention_data: Attention 数据 (numpy array 或 list)
        image_width: 图像宽度（像素）
        image_height: 图像高度（像素）
        word_mapping_path: word_mapping.json 文件路径
        top_k: 提取 Top K 个 patches
        output_path: 可选的输出文件路径

    Returns:
        (提取的文本, patch 边界列表)

    Examples:
        >>> from vera import retrieval
        >>> import numpy as np
        >>>
        >>> # 假设已有 attention 数据
        >>> attn_data = np.array([...])
        >>>
        >>> # 检索相关文本
        >>> text, patches = retrieval.attention_retrieve(
        ...     attention_data=attn_data,
        ...     image_width=1920,
        ...     image_height=1080,
        ...     word_mapping_path="word_mapping.json",
        ...     top_k=10
        ... )
    """
    import numpy as np
    from vera.analysis import get_top_k_patches

    # 1. 获取 Top-K patches
    patch_bounds = get_top_k_patches(
        attention_data=attention_data,
        image_height=image_height,
        image_width=image_width,
        k=top_k
    )

    # 2. 提取文本
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(word_mapping_path),
            "retrieved_evidence.txt"
        )

    extracted_text = extract_evidence_from_patches(
        patch_bounds=patch_bounds,
        word_mapping_path=word_mapping_path,
        output_path=output_path
    )

    return extracted_text, patch_bounds


def find_word_mapping_path(folder_path: str, root_dir: str) -> Optional[str]:
    """
    查找 word_mapping.json 文件路径

    Args:
        folder_path: 文件夹路径
        root_dir: 根目录

    Returns:
        word_mapping.json 的完整路径，如果找不到则返回 None
    """
    # 首先尝试本地目录
    local_path = os.path.join(folder_path, "word_mapping.json")
    if os.path.exists(local_path):
        return local_path

    # 从 folder_path 提取相对路径信息
    try:
        rendered_images_dir = os.path.dirname(folder_path.rstrip('/'))
        question_folder = os.path.dirname(rendered_images_dir)
        paper_folder = os.path.dirname(question_folder)

        question_hash = os.path.basename(question_folder)
        paper_id = os.path.basename(paper_folder)

        target_rendered_images = os.path.join(root_dir, paper_id, question_hash, "rendered_images")

        if os.path.exists(target_rendered_images):
            for subdir in os.listdir(target_rendered_images):
                word_mapping_path = os.path.join(target_rendered_images, subdir, "word_mapping.json")
                if os.path.exists(word_mapping_path):
                    return word_mapping_path
    except Exception:
        pass

    return None


def extract_evidence_from_patches(
    patch_bounds: List[Tuple[int, int, int, int]],
    word_mapping_path: str,
    output_path: str
) -> str:
    """
    根据注意力 patch 像素位置提取证据文本

    此函数实现了基于注意力的文本检索：
    1. 给定 Top-K attention patches 的像素边界
    2. 在 word_mapping.json 中查找与这些 patches 重叠的文本
    3. 提取相关行的文本内容

    Args:
        patch_bounds: Patch 边界列表 [(x_min, y_min, x_max, y_max), ...]
        word_mapping_path: word_mapping.json 文件路径
        output_path: 输出文本文件路径

    Returns:
        提取的文本内容

    Example:
        >>> from vera import retrieval
        >>>
        >>> # 获取 top-k patches
        >>> patches = analysis.get_top_k_patches(
        ...     attention_data=attn_array,
        ...     image_height=1000,
        ...     image_width=800,
        ...     k=10
        ... )
        >>>
        >>> # 提取证据文本
        >>> evidence = retrieval.extract_evidence_from_patches(
        ...     patch_bounds=patches,
        ...     word_mapping_path="word_mapping.json",
        ...     output_path="evidence.txt"
        ... )
    """
    if not os.path.exists(word_mapping_path):
        return ""

    try:
        with open(word_mapping_path, 'r', encoding='utf-8') as f:
            word_data = json.load(f)
    except Exception as e:
        return ""

    involved_lines = set()

    # 找出与 patches 重叠的行
    for (px1, py1, px2, py2) in patch_bounds:
        for word_info in word_data.get("words", []):
            word_bbox = word_info.get("bbox", [])
            if len(word_bbox) < 4:
                continue

            wx1, wy1, wx2, wy2 = word_bbox

            # 检查是否重叠
            if not (px2 < wx1 or wx2 < px1 or py2 < wy1 or wy2 < py1):
                involved_lines.add(word_info.get("line", -1))

    # 提取行的文本
    line_texts = {}
    for word_info in word_data.get("words", []):
        line_num = word_info.get("line", -1)
        if line_num in involved_lines:
            if line_num not in line_texts:
                line_texts[line_num] = word_info.get("word", "")

    # 按行号排序
    sorted_lines = sorted(line_texts.keys())
    extracted_lines = [line_texts[ln] for ln in sorted_lines]
    extracted_text = "\n".join(extracted_lines)

    # 保存到文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
    except Exception:
        pass

    return extracted_text


def retrieve_by_attention(
    attention_data,
    image_height: int,
    image_width: int,
    word_mapping_path: str,
    top_k: int = 10,
    output_path: Optional[str] = None
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    [DEPRECATED] 使用注意力数据进行检索的完整流程

    .. deprecated::
        Use `attention_retrieve()` instead. This function is kept for backward compatibility.
        The parameter order has been changed to match the convention (width before height).

    Args:
        attention_data: Attention 数据 (numpy array 或 list)
        image_height: 图像高度
        image_width: 图像宽度
        word_mapping_path: word_mapping.json 文件路径
        top_k: 提取 Top K 个 patches
        output_path: 可选的输出文件路径

    Returns:
        (提取的文本, patch 边界列表)

    Examples:
        >>> # Old API (deprecated)
        >>> text, patches = retrieval.retrieve_by_attention(
        ...     attention_data=attn_data,
        ...     image_height=1000,
        ...     image_width=800,
        ...     word_mapping_path="word_mapping.json",
        ...     top_k=10
        ... )
        >>>
        >>> # New API (recommended)
        >>> text, patches = retrieval.attention_retrieve(
        ...     attention_data=attn_data,
        ...     image_width=800,
        ...     image_height=1000,
        ...     word_mapping_path="word_mapping.json",
        ...     top_k=10
        ... )
    """
    import warnings
    warnings.warn(
        "retrieval.retrieve_by_attention() is deprecated. Use retrieval.attention_retrieve() instead. "
        "Note that the new API uses (image_width, image_height) parameter order.",
        DeprecationWarning,
        stacklevel=2
    )

    # Call the new API (note: swap height/width to match new convention)
    return attention_retrieve(
        attention_data=attention_data,
        image_width=image_width,
        image_height=image_height,
        word_mapping_path=word_mapping_path,
        top_k=top_k,
        output_path=output_path
    )


def extract_top_patches_with_attention_retrieve(
    attn_data,
    visual_indices: Tuple[int, int],
    word_mapping_path: str,
    grid_w: int,
    grid_h: int,
    img_width: int,
    img_height: int,
    top_k: int = 10,
    target_heads: Optional[List[Tuple[int, int]]] = None,
    debug_image_path: Optional[str] = None
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    使用 VERA retrieval 模块提取 top-k patches 对应的文本（完整流程）

    这是一个高级 API，整合了 attention 聚合、patch 计算和文本提取的完整流程。

    Args:
        attn_data: attention 数据（多层多头）
        visual_indices: (start, end) 视觉 token 索引
        word_mapping_path: word_mapping.json 路径
        grid_w, grid_h: patch 网格尺寸
        img_width, img_height: 图像尺寸（像素）
        top_k: top k patches
        target_heads: 目标 heads 列表（如果为 None，使用默认的 Top-20）
        debug_image_path: 可选的 debug 图像保存路径

    Returns:
        (extracted_text, top_patch_coords)

    Examples:
        >>> from vera.retrieval import extract_top_patches_with_attention_retrieve
        >>>
        >>> text, patches = extract_top_patches_with_attention_retrieve(
        ...     attn_data=attn_data,
        ...     visual_indices=(0, 256),
        ...     word_mapping_path="word_mapping.json",
        ...     grid_w=16,
        ...     grid_h=16,
        ...     img_width=1920,
        ...     img_height=1080,
        ...     top_k=10
        ... )
    """
    import numpy as np
    from vera.analysis import aggregate_attention_with_target_heads, get_top_patches_from_attn

    # 默认使用 Top-20 heads
    if target_heads is None:
        target_heads = [
            (24, 29), (21, 11), (24, 8), (26, 26), (24, 13),
            (26, 15), (28, 16), (27, 18), (23, 30), (24, 31),
            (28, 3), (21, 8), (23, 28), (28, 0), (26, 31),
            (26, 20), (23, 10), (21, 10), (23, 13), (20, 15)
        ]

    visual_start, visual_end = visual_indices
    visual_token_count = visual_end - visual_start

    # 1. 聚合 attention 数据
    avg_attn_vector = aggregate_attention_with_target_heads(
        attn_data, visual_start, visual_end, visual_token_count, target_heads
    )

    # 2. 获取 top-k patches 的坐标
    top_patch_coords = get_top_patches_from_attn(
        attn_vector=avg_attn_vector,
        grid_w=grid_w,
        grid_h=grid_h,
        img_width=img_width,
        img_height=img_height,
        top_k=top_k
    )

    # 3. 使用 VERA retrieval 模块提取文本
    extracted_text = extract_evidence_from_patches(
        patch_bounds=top_patch_coords,
        word_mapping_path=word_mapping_path,
        output_path=None  # 不保存到文件，直接返回
    )

    # 4. 生成 debug 图像（如果需要）
    if debug_image_path:
        try:
            import cv2
            merged_evidence_path = os.path.join(os.path.dirname(word_mapping_path), "merged_evidence.png")
            if os.path.exists(merged_evidence_path):
                debug_img = cv2.imread(merged_evidence_path)

                # 绘制 top-k patches
                for i, (x1, y1, x2, y2) in enumerate(top_patch_coords):
                    color = (0, 255 - i * 25, i * 25)  # 渐变色
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_img, f"{i+1}", (x1 + 5, y1 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imwrite(debug_image_path, debug_img)
        except Exception as e:
            print(f"  Warning: Failed to generate debug image: {e}")

    return extracted_text, top_patch_coords
