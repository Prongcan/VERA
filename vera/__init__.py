"""
VERA: Visual Evidence Retrieval Augmentation
视觉证据检索增强库
"""
#
__version__ = "0.1.0"

# 暴露主要 API
from vera import models, rendering, retrieval, analysis, utils

__all__ = ["models", "rendering", "retrieval", "analysis", "utils"]
