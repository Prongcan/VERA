"""
Analysis module for VERA
提供分析和可视化功能
"""

from vera.analysis.heatmap import create_heatmap, get_top_k_patches
from vera.analysis.full_analysis import run_full_analysis
from vera.analysis.attention_analysis import (
    aggregate_attention_with_target_heads,
    calculate_patch_distribution,
    get_top_patches_from_attn
)

__all__ = [
    # Heatmap and visualization
    "create_heatmap",
    "get_top_k_patches",

    # Full analysis
    "run_full_analysis",

    # Attention analysis
    "aggregate_attention_with_target_heads",
    "calculate_patch_distribution",
    "get_top_patches_from_attn",
]
