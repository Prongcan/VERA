"""
ColPali retrieval for VERA
使用 ColPali 模型进行视觉 patch 级别的检索
"""

import os
import sys
import json
import math
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


def colpali_retrieve(
    model_name: str,
    image_path: str,
    word_mapping_path: str,
    query: str,
    top_k: int = 20,
    overlap: int = 0
) -> str:
    """
    使用 ColPali 进行单样本检索

    Args:
        model_name: HuggingFace 模型 ID (如 "vidore/colpali-v1.2")
        image_path: 文档图像路径
        word_mapping_path: word_mapping.json 文件路径
        query: 查询文本
        top_k: 返回 Top K 个 patches
        overlap: 切分图片时的重叠像素数

    Returns:
        str: 提取的文本上下文

    Examples:
        >>> from vera import retrieval
        >>> context = retrieval.colpali_retrieve(
        ...     model_name="vidore/colpali-v1.2",
        ...     image_path="/path/to/document.png",
        ...     word_mapping_path="/path/to/word_mapping.json",
        ...     query="What is the main contribution?",
        ...     top_k=20
        ... )
        >>> print(context)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try to import colpali
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
        from colpali_engine.utils.torch_utils import get_torch_device
        from colpali_engine.interpretability import get_similarity_maps_from_embeddings
    except ImportError:
        raise ImportError(
            "colpali-engine is not installed. "
            "Install it with: pip install colpali-engine"
        )

    # Load model and processor (cached in module-level variable for efficiency)
    if not hasattr(colpali_retrieve, '_model') or colpali_retrieve._model_name != model_name:
        print(f"Loading ColPali model: {model_name}")
        colpali_retrieve._model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        colpali_retrieve._processor = ColPaliProcessor.from_pretrained(model_name)
        colpali_retrieve._model_name = model_name
        colpali_retrieve._device = device

    model = colpali_retrieve._model
    processor = colpali_retrieve._processor
    device = colpali_retrieve._device

    # Load image
    image = Image.open(image_path)

    # Split image into crops
    crops = _split_image_into_squares(image, overlap)

    # Process each crop
    all_patches = []
    for crop_img, start_x, start_y, crop_height in crops:
        # Get patch embeddings
        batch_images = processor.process_images([crop_img]).to(device)
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)

        n_patches = processor.get_n_patches(image_size=crop_img.size, patch_size=model.patch_size)

        # Get query embedding
        batch_queries = processor.process_queries([query]).to(device)
        with torch.no_grad():
            query_embeddings = model.forward(**batch_queries)

        # Compute similarity
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=None,
        )

        similarity_maps = batched_similarity_maps[0]
        n_patches_x, n_patches_y = n_patches
        num_tokens = similarity_maps.shape[0]

        # Calculate patch pixel dimensions
        patch_pixel_width = crop_img.size[0] / n_patches_x
        patch_pixel_height = crop_img.size[1] / n_patches_y

        # Collect patches
        for patch_y in range(n_patches_y):
            for patch_x in range(n_patches_x):
                local_x_start = int(patch_x * patch_pixel_width)
                local_x_end = int((patch_x + 1) * patch_pixel_width)
                local_y_start = int(patch_y * patch_pixel_height)
                local_y_end = int((patch_y + 1) * patch_pixel_height)

                global_x_start = start_x + local_x_start
                global_x_end = start_x + local_x_end
                global_y_start = start_y + local_y_start
                global_y_end = start_y + local_y_end

                token_scores = {}
                for token_idx in range(num_tokens):
                    score = similarity_maps[token_idx, patch_y, patch_x].item()
                    token_scores[token_idx] = score

                avg_score = sum(token_scores.values()) / len(token_scores)
                max_score = max(token_scores.values())

                all_patches.append({
                    'coords': (global_x_start, global_y_start, global_x_end, global_y_end),
                    'max_score': max_score,
                    'avg_score': avg_score
                })

    # Sort and get top K
    sorted_patches = sorted(all_patches, key=lambda p: p['max_score'], reverse=True)
    top_k_patches = sorted_patches[:top_k]

    # Extract evidence from word mapping
    if os.path.exists(word_mapping_path):
        with open(word_mapping_path, 'r', encoding='utf-8') as f:
            word_mapping = json.load(f)

        evidence_lines = []
        for patch in top_k_patches:
            x1, y1, x2, y2 = patch['coords']
            # Find words that intersect with this patch
            for word_entry in word_mapping.get('words', []):
                bbox = word_entry.get('bbox', [0, 0, 0, 0])
                # Check intersection
                if not (x2 < bbox[0] or x1 > bbox[2] or y2 < bbox[1] or y1 > bbox[3]):
                    word_text = word_entry.get('word', '')
                    if word_text and word_text not in evidence_lines:
                        evidence_lines.append(word_text)

        extracted_text = " ".join(evidence_lines)
    else:
        extracted_text = ""

    return extracted_text


def colpali(
    model_name: str,
    data_path: str,
    save_dir: str,
    top_k: int = 20,
    overlap: int = 0,
    qwen_model_path: Optional[str] = None
) -> dict:
    """
    [DEPRECATED] 使用 ColPali 进行检索

    .. deprecated::
        Use `colpali_retrieve()` instead. This function is kept for backward compatibility.
        Users should handle file traversal in their own code.

    Args:
        model_name: HuggingFace 模型 ID (如 "vidore/colpali-v1.2")
        data_path: 数据集路径（包含 merged.png 和 word_mapping.json 的文件夹）
        save_dir: 保存目录
        top_k: 返回 Top K 个 patches
        overlap: 切分图片时的重叠像素数
        qwen_model_path: Qwen 模型路径（用于提取问题）

    Returns:
        dict: 检索结果统计信息

    Examples:
        >>> # Old API (deprecated)
        >>> stats = retrieval.colpali(
        ...     model_name="vidore/colpali-v1.2",
        ...     data_path="tem/qasper_qwen_img",
        ...     save_dir="tem/qasper_qwen_img",
        ...     top_k=20
        ... )
        >>>
        >>> # New API (recommended)
        >>> context = retrieval.colpali_retrieve(
        ...     model_name="vidore/colpali-v1.2",
        ...     image_path="/path/to/image.png",
        ...     word_mapping_path="/path/to/word_mapping.json",
        ...     query="What is X?",
        ...     top_k=20
        ... )
    """
    warnings.warn(
        "retrieval.colpali() is deprecated. Use retrieval.colpali_retrieve() instead. "
        "File traversal should be handled in user code.",
        DeprecationWarning,
        stacklevel=2
    )

    # Load Qwen tokenizer for question extraction (if provided)
    qwen_tokenizer = None
    if qwen_model_path:
        try:
            from transformers import AutoProcessor
            qwen_processor = AutoProcessor.from_pretrained(
                qwen_model_path,
                trust_remote_code=True
            )
            qwen_tokenizer = qwen_processor.tokenizer
        except Exception as e:
            print(f"Warning: Failed to load Qwen tokenizer: {e}")

    # Find all question folders (folders with input_tokens.json)
    folders = []
    for root, dirs, files in os.walk(data_path):
        if "result" in root:
            continue
        if "input_tokens.json" in files:
            folders.append(root)

    print(f"Found {len(folders)} question folders")

    # Process each folder
    stats = {"total": len(folders), "success": 0, "failed": 0, "errors": []}

    for folder_path in tqdm(folders, desc="Processing with ColPali"):
        try:
            folder_name = os.path.basename(folder_path)

            # Get file paths
            input_tokens_path = os.path.join(folder_path, "input_tokens.json")

            # Find word_mapping.json in subdirectories
            word_mapping_path = None
            for root, dirs, files in os.walk(folder_path):
                if "word_mapping.json" in files:
                    word_mapping_path = os.path.join(root, "word_mapping.json")
                    break

            # Find image (merged.png or any PNG in rendered_images)
            image_path = None
            # First try merged.png in current folder
            merged_path = os.path.join(folder_path, "merged.png")
            if os.path.exists(merged_path):
                image_path = merged_path
            else:
                # Try to find in rendered_images subdirectory
                for root, dirs, files in os.walk(folder_path):
                    if "merged_evidence.png" in files:
                        image_path = os.path.join(root, "merged_evidence.png")
                        break
                    elif "merged.png" in files:
                        image_path = os.path.join(root, "merged.png")
                        break
                    # Fallback to any PNG
                    elif not image_path and files:
                        png_files = [f for f in files if f.endswith('.png')]
                        if png_files:
                            image_path = os.path.join(root, png_files[0])
                            break

            output_path = os.path.join(folder_path, "extracted_evidence_colpali.txt")

            # Check if image exists
            if image_path is None or not os.path.exists(image_path):
                stats["failed"] += 1
                stats["errors"].append((folder_name, "No image found"))
                continue

            # Extract question
            question = _extract_question_from_tokens(input_tokens_path, qwen_tokenizer)
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

            # Use the new API
            extracted_text = colpali_retrieve(
                model_name=model_name,
                image_path=image_path,
                word_mapping_path=word_mapping_path,
                query=question,
                top_k=top_k,
                overlap=overlap
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

    return stats


def _split_image_into_squares(image: Image.Image, overlap: int = 0) -> list:
    """Split image into square crops"""
    width, height = image.size
    square_size = width

    crops = []

    if height <= width:
        if height < width:
            new_image = Image.new(image.mode, (width, width), (255, 255, 255))
            new_image.paste(image, (0, 0))
            crops.append((new_image, 0, 0, height))
        else:
            crops.append((image.copy(), 0, 0, height))
    else:
        step = square_size - overlap
        num_crops = math.ceil(height / step)

        for i in range(num_crops):
            start_y = i * step
            end_y = min(start_y + square_size, height)
            actual_height = end_y - start_y

            crop = image.crop((0, start_y, width, end_y))

            if actual_height < square_size:
                padded_crop = Image.new(image.mode, (square_size, square_size), (255, 255, 255))
                padded_crop.paste(crop, (0, 0))
                crops.append((padded_crop, 0, start_y, actual_height))
            else:
                crops.append((crop, 0, start_y, actual_height))

    return crops


def _extract_question_from_tokens(input_tokens_path: str, tokenizer) -> Optional[str]:
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
