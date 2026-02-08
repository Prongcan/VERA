"""
VERA: Visual Evidence Retrieval and Analysis
视觉证据检索与分析库
"""

__version__ = "0.1.0"

# 暴露主要 API
from vera import models, rendering, retrieval, analysis

__all__ = ["models", "rendering", "retrieval", "analysis"]
