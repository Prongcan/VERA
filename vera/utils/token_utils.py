"""
Token utilities for VERA
提供处理 token 的工具函数
"""

import os
import json
from typing import Optional, Tuple


def get_visual_token_indices(input_tokens_path: str) -> Optional[Tuple[int, int]]:
    """
    从 input_tokens.json 获取视觉 token 的索引范围

    Args:
        input_tokens_path: input_tokens.json 文件路径

    Returns:
        (start_idx, end_idx) 或 None

    Examples:
        >>> from vera.utils import get_visual_token_indices
        >>> indices = get_visual_token_indices("input_tokens.json")
        >>> if indices:
        ...     start, end = indices
        ...     print(f"Visual tokens: {start} to {end}")
    """
    if not os.path.exists(input_tokens_path):
        return None

    try:
        with open(input_tokens_path, 'r') as f:
            tokens = json.load(f)

        vision_start_idx = None
        vision_end_idx = None

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


def extract_question_from_tokens(
    input_tokens_path: str,
    tokenizer,
    clean: bool = True
) -> Optional[str]:
    """
    从 input_tokens.json 提取问题文本

    Args:
        input_tokens_path: input_tokens.json 文件路径
        tokenizer: 分词器（用于解码 tokens）
        clean: 是否清理问题文本（移除标记）

    Returns:
        提取的问题文本，失败返回 None

    Examples:
        >>> from transformers import AutoProcessor
        >>> from vera.utils import extract_question_from_tokens
        >>>
        >>> processor = AutoProcessor.from_pretrained("/path/to/model")
        >>> question = extract_question_from_tokens(
        ...     "input_tokens.json",
        ...     processor.tokenizer
        ... )
    """
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

        if clean:
            # 清理问题文本
            prefix_marker = "Please answer the question based on the document images provided."
            if question_text.startswith(prefix_marker):
                question_text = question_text[len(prefix_marker):].strip()
            cutoff_marker = "Please output your answer **directly**"
            if cutoff_marker in question_text:
                question_text = question_text.split(cutoff_marker)[0].strip()

        return question_text.strip()

    except Exception:
        return None
