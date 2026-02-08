"""
Retrieval module for VERA
提供各种检索方法
"""

from vera.retrieval.qwen_embedding import qwen_embedding, qwen_embedding_retrieve
from vera.retrieval.colpali import colpali, colpali_retrieve
from vera.retrieval.attention import (
    find_word_mapping_path,
    extract_evidence_from_patches,
    retrieve_by_attention,
    attention_retrieve,
    extract_top_patches_with_attention_retrieve
)

__all__ = [
    # New API (recommended)
    "colpali_retrieve",
    "qwen_embedding_retrieve",
    "attention_retrieve",
    "extract_top_patches_with_attention_retrieve",

    # Old API (deprecated, kept for backward compatibility)
    "colpali",
    "qwen_embedding",
    "retrieve_by_attention",

    # Utility functions (still useful)
    "find_word_mapping_path",
    "extract_evidence_from_patches",
]
