"""
Models module for VERA
提供模型推理功能
"""

from typing import Optional
from vera.models.qwen import QwenEngine, QwenEngineMasked


def initialize(model_path: str, model_type: str = "qwen-img", max_new_tokens: int = 2048):
    """
    Initialize a VERA model engine

    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model to initialize
            - "qwen-img": Standard Qwen3-VL image model
            - "qwen-img-masked": Qwen3-VL with attention head masking support
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Engine instance with run() method

    Examples:
        >>> engine = vera.models.initialize(
        ...     model_path="/path/to/Qwen3-VL-8B-Instruct",
        ...     model_type="qwen-img"
        ... )
        >>> result = engine.run(
        ...     prompt_context="Please answer based on the document images",
        ...     question_text="What is the main contribution?",
        ...     image_paths=["doc1.png", "doc2.png"],
        ...     is_mask_heads=False,
        ...     heads_positions=None
        ... )

        >>> # Use masked version
        >>> engine = vera.models.initialize(
        ...     model_path="/path/to/Qwen3-VL-8B-Instruct",
        ...     model_type="qwen-img-masked"
        ... )
        >>> mask_config = {(29, 24), (11, 21), (8, 24)}
        >>> result = engine.run(
        ...     prompt_context="...",
        ...     question_text="...",
        ...     image_paths=["..."],
        ...     is_mask_heads=True,
        ...     heads_positions=mask_config
        ... )
    """
    if model_type == "qwen-img":
        return QwenEngine(model_path, max_new_tokens)
    elif model_type == "qwen-img-masked":
        return QwenEngineMasked(model_path, max_new_tokens)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Supported types: 'qwen-img', 'qwen-img-masked'")


__all__ = ["initialize"]
