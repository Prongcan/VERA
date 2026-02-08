"""
Text to image rendering module for VERA
提供文本渲染为图像的功能
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.word2png_function import text_to_images, text_to_images_evidence, load_config


def text_to_image(
    text: str,
    output_dir: str,
    config: Union[str, dict],
    evidence_text: Optional[List[str]] = None
) -> List[str]:
    """
    将文本渲染为图像

    Args:
        text: 文本内容
        output_dir: 输出目录
        config: 配置字典或配置文件路径
        evidence_text: 证据文本列表（可选，用于高亮显示）

    Returns:
        List[str]: 生成的图像路径列表

    Examples:
        >>> # Basic usage
        >>> paths = rendering.text_to_image(
        ...     text="Hello, world!",
        ...     output_dir="./output",
        ...     config="config/config_en.json"
        ... )

        >>> # With evidence highlighting
        >>> paths = rendering.text_to_image(
        ...     text="This is a document with important information.",
        ...     output_dir="./output",
        ...     config="config/config_en.json",
        ...     evidence_text=["important information"]
        ... )

        >>> # Using config dict
        >>> paths = rendering.text_to_image(
        ...     text="Hello, world!",
        ...     output_dir="./output",
        ...     config={"font-path": "/path/to/font.ttf", "font-size": 12}
        ... )
    """
    # Normalize config to dict
    if isinstance(config, str):
        cfg = load_config(config)
    elif isinstance(config, dict):
        cfg = config
    else:
        raise TypeError(f"config must be str or dict, got {type(config)}")

    # Choose function based on whether evidence is provided
    if evidence_text:
        image_paths = text_to_images_evidence(
            text=text,
            output_dir=output_dir,
            config_dict=cfg,
            evidence_text=evidence_text
        )
    else:
        image_paths = text_to_images(
            text=text,
            output_dir=output_dir,
            config_dict=cfg
        )

    return image_paths
