#!/usr/bin/env python3
"""
Batch process all text files in doc directory and convert them to images using multiprocessing.
"""
import os
import sys
import uuid
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

# Import the text_to_images function from word2png_function
try:
    from scripts.word2png_function import text_to_images, load_config, text_to_images_evidence
except ModuleNotFoundError:
    # Ensure project root (parent of this file's directory) is on sys.path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.word2png_function import text_to_images, load_config

def process_single_text(txt, output_root, config):
    """
    Render a single text string to images.
    - config can be a dict (preferred) or a path to json.
    """
    try:
        # Normalize config to dict
        cfg = None
        if isinstance(config, (str, os.PathLike)):
            cfg = load_config(str(config))
        elif isinstance(config, dict):
            cfg = config
        elif config is None:
            raise ValueError("config is None; provide a config dict or a path to config json")
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        image_paths = text_to_images(
            text=txt,
            output_dir=output_root,
            config_dict=cfg,
        )
        print(image_paths)
        return image_paths

    except Exception as e:
        print(f"[process_single_text] failed: {e}")
        return None


def process_single_text_evidence(txt, output_root, config, evidence_text):
    """
    Render a single text string to images.
    - config can be a dict (preferred) or a path to json.
    """
    try:
        # Normalize config to dict
        cfg = None
        if isinstance(config, (str, os.PathLike)):
            cfg = load_config(str(config))
        elif isinstance(config, dict):
            cfg = config
        elif config is None:
            raise ValueError("config is None; provide a config dict or a path to config json")
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        image_paths = text_to_images_evidence(
            text=txt,
            output_dir=output_root,
            config_dict=cfg,
            evidence_text=evidence_text,
        )
        print(image_paths)
        return image_paths

    except Exception as e:
        print(f"[process_single_text] failed: {e}")
        return None

if __name__ == "__main__":
    process_single_text(
        "Hello, world!",
        "/data3/guofang/peirongcan/deepseekOCR/Loong/Glyph/VLM_and_LLM/img_tem",
        "/data3/guofang/peirongcan/deepseekOCR/Loong/Glyph/config/config_en.json",
    )