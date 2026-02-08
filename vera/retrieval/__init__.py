"""
Retrieval module for VERA
提供各种检索方法
"""

from vera.retrieval.qwen_embedding import qwen_embedding
from vera.retrieval.colpali import colpali
from vera.retrieval.attention import (
    find_word_mapping_path,
    extract_evidence_from_patches,
    retrieve_by_attention
)

__all__ = [
    "qwen_embedding",
    "colpali",
    "find_word_mapping_path",
    "extract_evidence_from_patches",
    "retrieve_by_attention"
]
