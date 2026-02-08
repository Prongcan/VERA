"""
Attention-based Retrieval Methods for VERA
基于注意力的检索方法
"""

import os
import json
from typing import List, Tuple, Optional


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
    使用注意力数据进行检索的完整流程

    这是一个高层 API，结合了 attention 分析和文本提取

    Args:
        attention_data: Attention 数据 (numpy array 或 list)
        image_height: 图像高度
        image_width: 图像宽度
        word_mapping_path: word_mapping.json 文件路径
        top_k: 提取 Top K 个 patches
        output_path: 可选的输出文件路径

    Returns:
        (提取的文本, patch 边界列表)

    Example:
        >>> from vera import retrieval
        >>> import numpy as np
        >>>
        >>> # 假设已有 attention 数据
        >>> attn_data = np.array([...])
        >>>
        >>> # 检索相关文本
        >>> text, patches = retrieval.retrieve_by_attention(
        ...     attention_data=attn_data,
        ...     image_height=1000,
        ...     image_width=800,
        ...     word_mapping_path="word_mapping.json",
        ...     top_k=10,
        ...     output_path="retrieved_evidence.txt"
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
